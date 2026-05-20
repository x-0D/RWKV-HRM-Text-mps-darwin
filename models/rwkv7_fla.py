"""
Unified RWKV-7 kernel wrapper: FLA Triton (CUDA), Metal (MPS), or pure-PyTorch (CPU).

On CUDA: delegates to `fla.ops.rwkv7` (Triton, ~100x faster than loop).
On MPS: delegates to custom Metal compute kernel (ported from WebRWKV WGSL).
On CPU: falls back to pure-PyTorch sequential WKV.
"""
import torch
from torch import Tensor

# Type alias for the state tensor
WkvState = Tensor  # shape [B, H, N, N]

# ── Try importing FLA Triton kernel ──────────────────────
_fla_chunk = None
_fla_recurrent = None
try:
    import triton  # noqa: F401 — ensures triton is loadable
    from fla.ops.rwkv7.chunk import chunk_rwkv7 as _fla_chunk
    from fla.ops.rwkv7.fused_recurrent import (
        fused_mul_recurrent_rwkv7 as _fla_recurrent,
    )
except (ImportError, ModuleNotFoundError, RuntimeError):
    pass


def wkv7(r: Tensor, w: Tensor, k: Tensor, v: Tensor,
         a: Tensor, b: Tensor, state: WkvState | None = None
         ) -> tuple[Tensor, WkvState]:
    """
    Unified RWKV-7 kernel.

    Args:
        r: shape [B, T, C] — receptance
        w: shape [B, T, C] — log decay
        k: shape [B, T, C] — key
        v: shape [B, T, C] — value
        a: shape [B, T, C] — delta rule A (=-kk)
        b: shape [B, T, C] — delta rule B (=kk * sigmoid(a))
        state: shape [B, H, N, N] or None

    Returns:
        output: shape [B, T, C]
        new_state: shape [B, H, N, N]
    """
    device = r.device
    use_fla = _fla_chunk is not None and device.type == 'cuda'

    if use_fla:
        return _wkv7_fla(r, w, k, v, a, b, state)
    
    if device.type == 'mps':
        try:
            from models.rwkv7_metal import wkv7 as _wkv7_metal
            return _wkv7_metal(r, w, k, v, a, b, state)
        except (ImportError, RuntimeError):
            pass
    
    return _wkv7_pytorch(r, w, k, v, a, b, state)


def is_fla_available() -> bool:
    """Check if Triton RWKV-7 kernel is available (CUDA only)."""
    return _fla_chunk is not None


# ── FLA Triton kernel (CUDA) ─────────────────────────────
def _wkv7_fla(r: Tensor, w: Tensor, k: Tensor, v: Tensor,
              a: Tensor, b: Tensor, state: WkvState | None = None
              ) -> tuple[Tensor, WkvState]:
    B, T, C = r.shape
    H = C // 64  # HEAD_SIZE = 64
    N = 64

    # Reshape to [B, T, H, N]
    r_h = r.view(B, T, H, N)
    w_h = w.view(B, T, H, N)
    k_h = k.view(B, T, H, N)
    v_h = v.view(B, T, H, N)
    a_h = a.view(B, T, H, N)
    b_h = b.view(B, T, H, N)

    # FLA state shape: [B, H, N, N] (same as ours)
    if state is not None and state.ndim == 4:
        init_state = state
    else:
        init_state = None

    if _fla_recurrent is not None:
        # Use fused recurrent kernel (supports stateful)
        out, new_state = _fla_recurrent(
            r=r_h, w=w_h, k=k_h, v=v_h, a=a_h, b=b_h,
            scale=1.0,
            initial_state=init_state,
            output_final_state=True,
        )
    else:
        # Use chunk kernel (doesn't pass state through easily)
        out, new_state = _fla_chunk(
            r=r_h, w=w_h, k=k_h, v=v_h, a=a_h, b=b_h,
            scale=1.0,
            initial_state=init_state,
            output_final_state=True,
        )

    out = out.reshape(B, T, C)
    return out, new_state


# ── Pure-PyTorch fallback (MPS/CPU) ──────────────────────
def _wkv7_pytorch(r: Tensor, w: Tensor, k: Tensor, v: Tensor,
                  a: Tensor, b: Tensor, state: WkvState | None = None
                  ) -> tuple[Tensor, WkvState]:
    B, T, C = r.shape
    H = C // 64
    N = 64
    orig_dtype = r.dtype

    r = r.view(B, T, H, N).float()
    k = k.view(B, T, H, N).float()
    v = v.view(B, T, H, N).float()
    a = a.view(B, T, H, N).float()
    b = b.view(B, T, H, N).float()
    w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))

    out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)
    if state is None:
        state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)
    else:
        state = state.float()

    for t in range(T):
        kk = k[:, t].view(B, H, 1, N)
        rr = r[:, t].view(B, H, N, 1)
        vv = v[:, t].view(B, H, N, 1)
        aa = a[:, t].view(B, H, N, 1)
        bb = b[:, t].view(B, H, 1, N)

        state = state * w[:, t, :, None, :] + state @ aa @ bb + vv @ kk
        out[:, t] = (state @ rr).view(B, H, N)

    return out.view(B, T, C).to(dtype=orig_dtype), state.to(dtype=orig_dtype)
