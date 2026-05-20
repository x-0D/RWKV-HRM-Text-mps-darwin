"""
RADLADS-style model surgery + optional DotResize-style width reduction.

Usage:
    # Full 1B -> RWKV-7 hybrid (requires 6GB+ VRAM)
    uv run python scripts/convert_hrm_pretrained.py \
        --hf-path sapientinc/HRM-Text-1B \
        --output checkpoints/hrm_rwkv_1b_init.pth

    # Compressed 768 -> RWKV-7 hybrid (fits 6GB VRAM)
    uv run python scripts/convert_hrm_pretrained.py \
        --hf-path sapientinc/HRM-Text-1B \
        --target-hidden 768 \
        --output checkpoints/hrm_rwkv_768m_init.pth
"""
import os
import sys
import argparse

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rwkv7_layers import Rwkv7TimeMix, HEAD_SIZE
from models.layers import find_multiple


def global_dim_importance(hf_state: dict[str, torch.Tensor],
                          hidden_size: int) -> torch.Tensor:
    """Score each hidden dimension by total Frobenius weight connected to it."""
    imp = torch.zeros(hidden_size, dtype=torch.float32)
    count = 0
    for k, v in hf_state.items():
        if v.ndim != 2:
            continue
        col_norm = v.float().norm(dim=0) ** 2
        if col_norm.shape[0] == hidden_size:
            imp += col_norm
            count += 1
        row_norm = v.float().norm(dim=1) ** 2
        if row_norm.shape[0] == hidden_size:
            imp += row_norm
            count += 1
    imp /= max(count, 1)
    return imp


def reduce_with_perm(W: torch.Tensor, d_out: int, d_in: int,
                     perm_out: torch.Tensor | None,
                     perm_in: torch.Tensor | None) -> torch.Tensor:
    """
    Reduce a weight matrix from (D_out, D_in) to (d_out, d_in).
    If a perm is None, truncate/pad without reordering (first d rows/cols).
    """
    D_out, D_in = W.shape
    # Reduce output dim
    if D_out > d_out:
        if perm_out is not None:
            W = W[perm_out[:d_out].to(W.device)]
        else:
            W = W[:d_out]
    elif D_out < d_out:
        pad = torch.zeros(d_out - D_out, W.shape[1], dtype=W.dtype, device=W.device)
        W = torch.cat([W, pad], dim=0)
    # Reduce input dim
    if D_in > d_in:
        if perm_in is not None:
            W = W[:, perm_in[:d_in].to(W.device)]
        else:
            W = W[:, :d_in]
    elif D_in < d_in:
        pad = torch.zeros(W.shape[0], d_in - D_in, dtype=W.dtype, device=W.device)
        W = torch.cat([W, pad], dim=1)
    return W


def intermediate_importance_for_layer(hf_state: dict[str, torch.Tensor],
                                      stack: str, lid: int) -> torch.Tensor:
    """Score each intermediate neuron by gate+up row norms + down col norms."""
    imp = torch.zeros(4096, dtype=torch.float32)
    count = 0
    for suffix in ['gate_proj', 'up_proj']:
        k = f"model.{stack}.layers.{lid}.mlp.{suffix}.weight"
        if k in hf_state:
            imp += hf_state[k].float().norm(dim=1) ** 2
            count += 1
    k = f"model.{stack}.layers.{lid}.mlp.down_proj.weight"
    if k in hf_state:
        imp += hf_state[k].float().norm(dim=0) ** 2
        count += 1
    imp /= max(count, 1)
    return imp


def convert(hf_state: dict[str, torch.Tensor], hf_config: dict,
            dtype: torch.dtype, target_hidden: int | None = None
            ) -> dict[str, torch.Tensor]:
    import math
    D_orig = hf_config["hidden_size"]
    D = target_hidden or D_orig
    n_per_stack = hf_config["num_layers_per_stack"]
    I_orig = hf_config["intermediate_size"]
    I = find_multiple(round(I_orig * D / D_orig), 256) if target_hidden else I_orig

    new_sd: dict[str, torch.Tensor] = {}

    # --- Compute global dimension importance for width reduction ---
    perm_hidden: torch.Tensor | None = None
    if target_hidden and D < D_orig:
        print(f"  Computing global dimension importance ({D_orig}->{D})...")
        imp = global_dim_importance(hf_state, D_orig)
        _, perm_hidden = torch.sort(imp, descending=True)

    def _reduce_hidden(W, d_out=None, d_in=None):
        """Reduce where BOTH dims are hidden dim (e.g. q/k/v/o projections)."""
        if perm_hidden is None:
            return W
        return reduce_with_perm(W, d_out or D, d_in or D, perm_hidden, perm_hidden)

    # --- Shared weights ---
    def _copy(key_src, key_dst):
        if key_src in hf_state:
            w = hf_state[key_src].to(dtype)
            if perm_hidden is not None and w.ndim == 1 and w.shape[0] == D_orig:
                # 1D bias like zL_init
                w = w[perm_hidden[:D].to(w.device)]
            new_sd[key_dst] = w
    _copy("model.z_L_init", "zL_init")
    # Embedding: [vocab, hidden] -> reduce hidden dim only
    emb = hf_state.get("model.embed_tokens.weight")
    if emb is not None:
        w = emb.to(dtype)
        if perm_hidden is not None and w.shape[1] == D_orig:
            w = reduce_with_perm(w, w.shape[0], D, None, perm_hidden)
        new_sd["embed_tokens.embedding_weight"] = w
    # LM head: [vocab, hidden] -> reduce hidden dim only
    lm = hf_state.get("lm_head.weight")
    if lm is not None:
        w = lm.to(dtype)
        if perm_hidden is not None:
            w = reduce_with_perm(w, w.shape[0], D, None, perm_hidden)
        new_sd["lm_head.weight"] = w

    # --- Per-stack per-layer: copy + optionally reduce ---
    hf_stack_map = {"L_module": "L_level", "H_module": "H_level"}
    for hf_stack, rwkv_stack in hf_stack_map.items():
        for lid in range(n_per_stack):
            # Attention: q->receptance, k->key, v->value, o->output
            for src_name, dst_name in [
                ("self_attn.q_proj.weight", "attn.receptance.weight"),
                ("self_attn.k_proj.weight", "attn.key.weight"),
                ("self_attn.v_proj.weight", "attn.value.weight"),
                ("self_attn.o_proj.weight", "attn.output.weight"),
            ]:
                src = f"model.{hf_stack}.layers.{lid}.{src_name}"
                if src in hf_state:
                    new_sd[f"{rwkv_stack}.layers.{lid}.{dst_name}"] = _reduce_hidden(hf_state[src].to(dtype))

            # MLP: compute per-layer intermediate importance, fuse gate+up
            gate = hf_state.get(f"model.{hf_stack}.layers.{lid}.mlp.gate_proj.weight")
            up = hf_state.get(f"model.{hf_stack}.layers.{lid}.mlp.up_proj.weight")
            down = hf_state.get(f"model.{hf_stack}.layers.{lid}.mlp.down_proj.weight")
            if gate is not None and up is not None:
                # Compute per-layer intermediate permutation
                if perm_hidden is not None and I < I_orig:
                    imp_inter = intermediate_importance_for_layer(hf_state, hf_stack, lid)
                    _, perm_inter = torch.sort(imp_inter, descending=True)
                else:
                    perm_inter = None
                gate_r = reduce_with_perm(gate.to(dtype), I, D, perm_inter, perm_hidden)
                up_r = reduce_with_perm(up.to(dtype), I, D, perm_inter, perm_hidden)
                new_sd[f"{rwkv_stack}.layers.{lid}.mlp.gate_up_proj.weight"] = torch.cat([gate_r, up_r], dim=0)
            if down is not None:
                if perm_hidden is not None and I < I_orig:
                    imp_inter = intermediate_importance_for_layer(hf_state, hf_stack, lid)
                    _, perm_inter = torch.sort(imp_inter, descending=True)
                else:
                    perm_inter = None
                new_sd[f"{rwkv_stack}.layers.{lid}.mlp.down_proj.weight"] = reduce_with_perm(down.to(dtype), D, I, perm_hidden, perm_inter)

    # --- RWKV-7 specific params (init from scratch at target size) ---
    H = D // HEAD_SIZE
    N = HEAD_SIZE
    print(f"  Initializing RWKV-7 params: hidden={D}, heads={H}, head_dim={N}")
    for stack in ["L_level", "H_level"]:
        for lid in range(n_per_stack):
            tmix = Rwkv7TimeMix(hidden_size=D, layer_id=lid,
                                n_layer=n_per_stack, head_size=N)
            for name, param in tmix.named_parameters():
                full_name = f"{stack}.layers.{lid}.attn.{name}"
                if full_name not in new_sd:
                    new_sd[full_name] = param.to(dtype)

    return new_sd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-path", default="sapientinc/HRM-Text-1B")
    parser.add_argument("--target-hidden", type=int, default=None,
                        help="Reduced hidden_size (e.g. 768 for 6GB VRAM)")
    parser.add_argument("--output", default="checkpoints/hrm_rwkv_init.pth")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoConfig

    config = AutoConfig.from_pretrained(args.hf_path, trust_remote_code=True)
    dtype = getattr(torch, args.dtype)

    print(f"Downloading {args.hf_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_path, trust_remote_code=True,
        torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    hf_sd = model.state_dict()
    del model

    n_layers = config.num_layers_per_stack
    D = config.hidden_size
    D_target = args.target_hidden or D
    total_layers = config.H_cycles * (config.L_cycles + 1) * n_layers
    print(f"Orig: hidden={D}, per_stack={n_layers}, total_slots={total_layers}")
    print(f"Target: hidden={D_target}")

    new_sd = convert(hf_sd, config.to_dict(), dtype, D_target)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(new_sd, args.output)
    total = sum(v.numel() for v in new_sd.values())
    print(f"Saved {len(new_sd)} tensors ({total/1e6:.1f}M params) -> {args.output}")
    print(f"  Fits 6GB VRAM: {'YES' if total < 500e6 else 'MAYBE'} (model size) + optimizer ~3x -> ~{3*total/1e9:.2f}GB")


if __name__ == "__main__":
    main()
