"""
Neuron stitching via SVD-based width reduction (DotResize-inspired).

Reduces hidden_size D → d while preserving maximal pretrained knowledge.
Strategy: compute a global importance ordering across all weight matrices,
reorder dimensions consistently, then truncate.

Usage:
    uv run python scripts/compress_hrm_width.py \
        --checkpoint checkpoints/hrm_rwkv_1b_init.pth \
        --target-hidden 768 \
        --output checkpoints/hrm_rwkv_768m_init.pth
"""
import os
import sys
import argparse
import math

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rwkv7_layers import Rwkv7TimeMix, HEAD_SIZE
from models.layers import find_multiple


def global_dim_importance(checkpoint: dict[str, torch.Tensor],
                          hidden_size: int) -> torch.Tensor:
    """Score each hidden dimension by total Frobenius weight connected to it."""
    imp = torch.zeros(hidden_size, dtype=torch.float32)
    count = 0
    for k, v in checkpoint.items():
        if v.ndim != 2:  # only linear weights
            continue
        # Column importance: sum of squared weights per input dimension
        col_norm = v.float().norm(dim=0) ** 2  # [D_in]
        if col_norm.shape[0] == hidden_size:
            imp += col_norm[:hidden_size]
            count += 1
        # Row importance: sum of squared weights per output dimension
        row_norm = v.float().norm(dim=1) ** 2  # [D_out]
        if row_norm.shape[0] == hidden_size:
            imp += row_norm[:hidden_size]
            count += 1
    imp /= max(count, 1)
    return imp


def reduce_linear(W: torch.Tensor, d_out: int, d_in: int,
                  perm_out: torch.Tensor, perm_in: torch.Tensor) -> torch.Tensor:
    """Reduce a weight matrix using dimension permutations."""
    D_out, D_in = W.shape
    # Permute and truncate output dims
    if D_out >= d_out:
        W_perm = W[perm_out[:d_out].to(W.device)]
    else:
        # Expand (shouldn't happen for our use case)
        pad = torch.zeros(d_out - D_out, D_in, dtype=W.dtype, device=W.device)
        W_perm = torch.cat([W, pad], dim=0)
    # Permute and truncate input dims
    if D_in >= d_in:
        W_out = W_perm[:, perm_in[:d_in].to(W.device)]
    else:
        pad = torch.zeros(d_out, d_in - D_in, dtype=W.dtype, device=W.device)
        W_out = torch.cat([W_perm, pad], dim=1)
    return W_out


def compress_checkpoint(ckpt: dict[str, torch.Tensor],
                        D: int, d: int,
                        I: int, i: int) -> dict[str, torch.Tensor]:
    """Reduce all weight matrices from size D→d and I→i."""
    print(f"Computing global dimension importance (D={D}→d={d}, I={I}→i={i})...")
    imp = global_dim_importance(ckpt, D)
    _, perm = torch.sort(imp, descending=True)  # most important first
    perm_out = perm  # same ordering for out and in dims
    perm_in = perm

    compressed: dict[str, torch.Tensor] = {}
    for k, v in ckpt.items():
        if v.ndim == 2:
            D_out, D_in = v.shape
            # Determine if this is an MLP weight (intermediate_size dims)
            if D_out == I:
                # MLP gate_up_proj.weight: [2*I, D] → [2*i, d]
                W_new = reduce_linear(v, 2 * i, d, perm_out.repeat(2), perm_in)
            elif D_in == I:
                # MLP down_proj.weight: [D, I] → [d, i]
                W_new = reduce_linear(v, d, i, perm_out, perm_in)
            elif D_out == D_in == D:
                W_new = reduce_linear(v, d, d, perm_out, perm_in)
            elif D_out == 2 * D and D_in == D:
                # gate_up_proj after RWKV conversion: [2*d, d]
                W_new = reduce_linear(v, 2 * d, d, perm_out.repeat(2), perm_in)
            else:
                # Skip if can't determine shape (e.g., lm_head)
                W_new = v
            compressed[k] = W_new.to(v.dtype)
        elif v.ndim == 1:
            if v.shape[0] == D:
                compressed[k] = v[perm[:d].to(v.device)]
            elif v.shape[0] == 2 * D:
                compressed[k] = v[perm[:d].to(v.device).repeat(2)]
            else:
                compressed[k] = v
        else:
            compressed[k] = v
    return compressed


def rebuild_rwkv_params(ckpt: dict[str, torch.Tensor],
                         d: int, per_stack_layers: int) -> dict[str, torch.Tensor]:
    """Rebuild RWKV-7 specific params for the smaller hidden_size."""
    H = d // HEAD_SIZE
    N = HEAD_SIZE
    new_params: dict[str, torch.Tensor] = {}
    for stack in ["L_level", "H_level"]:
        for lid in range(per_stack_layers):
            tmix = Rwkv7TimeMix(hidden_size=d, layer_id=lid,
                                n_layer=per_stack_layers, head_size=N)
            for name, param in tmix.named_parameters():
                full_name = f"{stack}.layers.{lid}.attn.{name}"
                if full_name not in ckpt:
                    new_params[full_name] = param.detach().clone()
    return new_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/hrm_rwkv_1b_init.pth")
    parser.add_argument("--target-hidden", type=int, default=768)
    parser.add_argument("--target-intermediate", type=int, default=None)
    parser.add_argument("--output", default="checkpoints/hrm_rwkv_768m_init.pth")
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    D = 1536
    d = args.target_hidden
    I_orig = 4096
    i = args.target_intermediate or find_multiple(round(d * 4 * 2 / 3), 256)

    print(f"Compressing: {D}→{d}, {I_orig}→{i}")
    compressed = compress_checkpoint(ckpt, D, d, I_orig, i)

    n_layers = 16
    print("Rebuilding RWKV-7 specific params for new size...")
    rwkv_params = rebuild_rwkv_params(compressed, d, n_layers)
    compressed.update(rwkv_params)

    total = sum(v.numel() for v in compressed.values())
    print(f"Compressed checkpoint: {len(compressed)} tensors, {total/1e6:.1f}M params")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(compressed, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
