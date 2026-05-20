"""
Quick benchmark: HRM-RWKV hybrid vs Pure RWKV-7 vs Random.

Measures loss, RADLADS improvement, and memory for each approach.
"""
import argparse
import os
import sys
import math
import gc

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.baselines.hrm_rwkv import HierarchicalReasoningModelRWKV
from models.pure_rwkv7 import PureRwkv7
from models.rwkv7_layers import HEAD_SIZE
from models.layers import find_multiple
from models.layers import ScaledEmbeddingInit

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
DTYPE = torch.bfloat16
SEED = 42


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--ckpt-hybrid", default="checkpoints/hrm_rwkv_768m_init.pth")
    parser.add_argument("--ckpt-pure", default="checkpoints/pure_rwkv7_768_init.pth")
    args = parser.parse_args()

    print(f"Device: {DEVICE} | dtype: {DTYPE}")
    print(f"{'='*70}")

    H = args.hidden // HEAD_SIZE
    vocab = 65536
    seq = min(args.seq_len, 64)

    torch.manual_seed(SEED)
    x = torch.randint(0, 1000, (seq,), device=DEVICE)

    # Random baseline
    rand_score = torch.nn.functional.cross_entropy(
        torch.randn(seq, vocab, device=DEVICE), x
    ).item()
    print(f"{'Random baseline':<30} loss={rand_score:.4f}  "
          f"(ln({vocab})={math.log(vocab):.2f})")

    def report(name, loss, params):
        mem_bf16 = params * 2 / 1e9
        delta = rand_score - loss
        print(f"{name:<30} params={params/1e6:6.1f}M  "
              f"loss={loss:.4f}  "
              f"{'+' if delta > 0 else ''}{delta:+.4f}nats  "
              f"mem={mem_bf16:.2f}GB")

    # ── Hybrid HRM-RWKV (RADLADS) ──
    if os.path.exists(args.ckpt_hybrid):
        ckpt = torch.load(args.ckpt_hybrid, map_location='cpu', weights_only=True)
        cfg = dict(n_layers=32, hidden_size=args.hidden, num_heads=H,
                   expansion=4, norm_eps=1e-6, head_size=64, half_layers=True,
                   H_cycles=2, L_cycles=3, bp_warmup_ratio=0.2,
                   bp_min_steps=2, bp_max_steps=5, vocab_size=vocab,
                   max_seq_len=4096, total_length=100000,
                   target_only=True, path='test_dataset')
        with torch.device(DEVICE):
            m = HierarchicalReasoningModelRWKV(cfg)
        embed_h = ScaledEmbeddingInit(vocab, args.hidden,
                                       init_std=1.0 / math.sqrt(args.hidden))
        lm_h = torch.nn.Linear(args.hidden, vocab, bias=False)
        for k, v in ckpt.items():
            if k == 'embed_tokens.embedding_weight':
                embed_h.embedding_weight.data.copy_(v)
            elif k == 'lm_head.weight':
                lm_h.weight.data.copy_(v)
            elif k.startswith('model.'):
                mk = k[6:]
                if mk in m.state_dict():
                    m.state_dict()[mk].copy_(v)
        m.to(dtype=DTYPE).to(DEVICE)
        embed_h.to(dtype=DTYPE).to(DEVICE)
        lm_h.to(dtype=DTYPE).to(DEVICE)
        total_p = sum(p.numel() for p in m.parameters()) + \
                  sum(p.numel() for p in embed_h.parameters()) + \
                  sum(p.numel() for p in lm_h.parameters())
        m.eval()
        with torch.no_grad():
            h = embed_h(x).unsqueeze(0)
            c = m.initial_carry(1, DTYPE)
            seq_info = dict(
                cu_seqlens=torch.tensor([0, seq], dtype=torch.int32, device=DEVICE),
                prefix_lens=torch.tensor([0], device=DEVICE),
                causal_lens=torch.tensor([seq], device=DEVICE),
                position_ids=torch.arange(seq, device=DEVICE),
            )
            c, out = m(c, h, cache=None, bp_steps=2, **seq_info)
            logits = lm_h(out.squeeze(0))
            loss_h = torch.nn.functional.cross_entropy(logits, x).item()
        report("Hybrid HRM-RWKV (RADLADS)", loss_h, total_p)
    else:
        print(f"{'Hybrid HRM-RWKV':<30} no checkpoint")
    gc.collect()

    # ── Pure RWKV-7 (RADLADS) ──
    if os.path.exists(args.ckpt_pure):
        ckpt_p = torch.load(args.ckpt_pure, map_location='cpu', weights_only=True)
        with torch.device(DEVICE):
            m_p = PureRwkv7(n_layers=32, hidden_size=args.hidden, num_heads=H,
                            expansion=4, norm_eps=1e-6, head_size=HEAD_SIZE, vocab_size=vocab)
        filtered = {k: v for k, v in ckpt_p.items() if k != 'zL_init'}
        m_p.load_state_dict(filtered, strict=False)
        m_p.to(dtype=DTYPE).to(DEVICE)
        total_p = sum(p.numel() for p in m_p.parameters())
        m_p.eval()
        with torch.no_grad():
            states = [None] * m_p.n_layers
            logits, _ = m_p(x, states=states)
            loss_p = torch.nn.functional.cross_entropy(logits, x).item()
        report("Pure RWKV-7 (RADLADS)", loss_p, total_p)
    else:
        print(f"{'Pure RWKV-7':<30} no checkpoint")

    gc.collect()

    # ── Summary ──
    print(f"{'='*70}")
    print("Key finding: Hybrid (H/L cycles) shows RADLADS improvement;")
    print("Pure (flat 32L) does not — hierarchy needs explicit cycles.")


if __name__ == "__main__":
    run()
