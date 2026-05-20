"""
Custom Metal RWKV-7 WKV kernel for Apple MPS.

Ported from WebRWKV WGSL shader (time_mix_v7.wgsl).
Threadgroup of N threads per (batch, head), each handling one state row.
"""

import torch
from torch import Tensor

_forward_kernel = None

_FORWARD_KERNEL_SRC = """
#include <metal_stdlib>
using namespace metal;

kernel void wkv7_forward(
    device const float* r          [[buffer(0)]],
    device const float* w          [[buffer(1)]],
    device const float* k          [[buffer(2)]],
    device const float* v          [[buffer(3)]],
    device const float* a          [[buffer(4)]],
    device const float* b          [[buffer(5)]],
    device float* state            [[buffer(6)]],
    device float* output           [[buffer(7)]],
    device float* all_states       [[buffer(8)]],
    constant int& T                [[buffer(9)]],
    constant int& H                [[buffer(10)]],
    constant int& N                [[buffer(11)]],
    uint group_id                  [[threadgroup_position_in_grid]],
    uint thread_id                 [[thread_position_in_threadgroup]]
) {
    if (thread_id >= N) return;

    int batch = group_id / H;
    int head = group_id % H;
    int row = thread_id;

    int state_start = (batch * H + head) * N * N;
    int data_start_h = head * N;
    int data_stride = H * N;
    int all_states_slot = (T + 1) * N * N;
    int all_states_start = (batch * H + head) * all_states_slot;

    threadgroup float aa_arr[64];
    threadgroup float bb_arr[64];
    threadgroup float kk_arr[64];
    threadgroup float rr_arr[64];
    threadgroup float decay_arr[64];

    float state_row[64];
    int base_off = state_start + row * N;
    for (int j = 0; j < N; j++) {
        state_row[j] = state[base_off + j];
        all_states[all_states_start + 0 * N * N + row * N + j] = state_row[j];
    }

    for (int t = 0; t < T; t++) {
        int t_base = (batch * T + t) * data_stride + data_start_h;

        aa_arr[row] = a[t_base + row];
        bb_arr[row] = b[t_base + row];
        kk_arr[row] = k[t_base + row];
        rr_arr[row] = r[t_base + row];
        decay_arr[row] = exp(-exp(w[t_base + row]));

        threadgroup_barrier(mem_flags::mem_threadgroup);

        float vv_val = v[t_base + row];

        float dot = 0.0;
        for (int j = 0; j < N; j++) {
            dot += state_row[j] * aa_arr[j];
        }

        float new_row[64];
        for (int j = 0; j < N; j++) {
            new_row[j] = state_row[j] * decay_arr[j] + dot * bb_arr[j] + vv_val * kk_arr[j];
        }

        float out_val = 0.0;
        for (int j = 0; j < N; j++) {
            out_val += new_row[j] * rr_arr[j];
        }
        output[(batch * T + t) * data_stride + data_start_h + row] = out_val;

        int st_off = all_states_start + (t + 1) * N * N + row * N;
        for (int j = 0; j < N; j++) {
            state_row[j] = new_row[j];
            all_states[st_off + j] = new_row[j];
        }
    }

    for (int j = 0; j < N; j++) {
        state[base_off + j] = state_row[j];
    }
}
"""

class Wkv7MetalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w, k, v, a, b, state):
        B, T, C = r.shape
        H = C // 64
        N = 64
        orig_dtype = r.dtype

        r_f = r.contiguous().float()
        w_f = w.contiguous().float()
        k_f = k.contiguous().float()
        v_f = v.contiguous().float()
        a_f = a.contiguous().float()
        b_f = b.contiguous().float()
        if state is None:
            state_f = torch.zeros(B, H, N, N, device=r.device, dtype=torch.float32)
        else:
            state_f = state.contiguous().float()

        output = torch.zeros(B, T, C, device=r.device, dtype=torch.float32)
        all_states = torch.zeros(B, H, T + 1, N, N, device=r.device, dtype=torch.float32)

        kernel = _get_forward_kernel()
        num_groups = B * H

        kernel(
            r_f, w_f, k_f, v_f, a_f, b_f,
            state_f, output, all_states,
            T, H, N,
            threads=(num_groups * N,), group_size=(N,),
        )

        ctx.save_for_backward(r_f, w_f, k_f, v_f, a_f, b_f)
        ctx.all_states = all_states
        ctx.H = H
        ctx.N = N
        ctx.T = T
        ctx.orig_dtype = orig_dtype
        ctx.state_batch = state_f.shape[0]

        return output.to(orig_dtype), state_f.to(orig_dtype)

    @staticmethod
    def backward(ctx, grad_output, grad_state_final):
        r, w, k, v, a, b = ctx.saved_tensors
        all_states = ctx.all_states
        B, T, C = r.shape
        H = ctx.H
        N = ctx.N
        device = r.device

        # View parameters as [B, T, H, N]
        r_h = r.view(B, T, H, N)
        w_h = w.view(B, T, H, N)
        k_h = k.view(B, T, H, N)
        v_h = v.view(B, T, H, N)
        a_h = a.view(B, T, H, N)
        b_h = b.view(B, T, H, N)

        w_decay = torch.exp(-torch.exp(w_h))

        go = grad_output.contiguous().float().view(B, T, H, N)
        gs_final = grad_state_final.contiguous().float() if grad_state_final is not None else None

        grad_r = torch.zeros(B, T, H, N, device=device, dtype=torch.float32)
        grad_w = torch.zeros(B, T, H, N, device=device, dtype=torch.float32)
        grad_k = torch.zeros(B, T, H, N, device=device, dtype=torch.float32)
        grad_v = torch.zeros(B, T, H, N, device=device, dtype=torch.float32)
        grad_a = torch.zeros(B, T, H, N, device=device, dtype=torch.float32)
        grad_b = torch.zeros(B, T, H, N, device=device, dtype=torch.float32)

        gs = torch.zeros(B, H, N, N, device=device, dtype=torch.float32)
        if gs_final is not None:
            gs = gs_final[:B].clone()
        state_batch = ctx.state_batch

        for t in range(T - 1, -1, -1):
            s_t = all_states[:, :, t]
            s_t1 = all_states[:, :, t + 1]
            go_t = go[:, t]

            r_t = r_h[:, t]
            w_decay_t = w_decay[:, t]
            a_t = a_h[:, t]
            b_t = b_h[:, t]
            v_t = v_h[:, t]
            k_t = k_h[:, t]

            # gs_input = gs + go_t @ r_t^T
            gs_input = gs + (go_t.unsqueeze(-1) @ r_t.unsqueeze(-2))

            # grad_r = s_{t+1}^T @ go_t
            grad_r[:, t] = (s_t1.transpose(-2, -1) @ go_t.unsqueeze(-1)).squeeze(-1)

            # grad_w: chain rule through exp(-exp(w))
            decay_grad = (gs_input * s_t).sum(dim=-2)
            grad_w[:, t] = decay_grad * (-w_decay_t * torch.exp(w_h[:, t]))

            # grad_a
            grad_a[:, t] = (s_t.transpose(-2, -1) @ (gs_input @ b_t.unsqueeze(-1))).squeeze(-1)

            # grad_b
            s_aa = s_t @ a_t.unsqueeze(-1)
            grad_b[:, t] = (s_aa.transpose(-2, -1) @ gs_input).squeeze(-2)

            # grad_v
            grad_v[:, t] = (gs_input @ k_t.unsqueeze(-1)).squeeze(-1)

            # grad_k
            grad_k[:, t] = (v_t.unsqueeze(-1).transpose(-2, -1) @ gs_input).squeeze(-2)

            # Propagate gs backward
            gs = gs_input * w_decay_t.unsqueeze(-2) + (gs_input @ b_t.unsqueeze(-1)) @ a_t.unsqueeze(-2)

        grad_state = torch.zeros(state_batch, H, N, N, device=device, dtype=torch.float32)
        grad_state[:B] = gs
        grad_r = grad_r.view(B, T, C)
        grad_w = grad_w.view(B, T, C)
        grad_k = grad_k.view(B, T, C)
        grad_v = grad_v.view(B, T, C)
        grad_a = grad_a.view(B, T, C)
        grad_b = grad_b.view(B, T, C)

        return grad_r, grad_w, grad_k, grad_v, grad_a, grad_b, grad_state


def _get_forward_kernel():
    global _forward_kernel
    if _forward_kernel is None:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS not available")
        lib = torch.mps.compile_shader(_FORWARD_KERNEL_SRC)
        _forward_kernel = lib.wkv7_forward
    return _forward_kernel


def is_metal_available() -> bool:
    return torch.backends.mps.is_available()


def wkv7(r: Tensor, w: Tensor, k: Tensor, v: Tensor,
         a: Tensor, b: Tensor, state: Tensor | None = None
         ) -> tuple[Tensor, Tensor]:
    """
    Metal-accelerated RWKV-7 WKV kernel (MPS only).
    Falls back to PyTorch if MPS not available.

    Args:
        r: [B, T, C] receptance
        w: [B, T, C] log decay
        k: [B, T, C] key
        v: [B, T, C] value
        a: [B, T, C] in-context learning rate
        b: [B, T, C] delta rule term
        state: [B, H, N, N] or None
    Returns:
        output: [B, T, C]
        new_state: [B, H, N, N]
    """
    if not torch.backends.mps.is_available():
        from models.rwkv7_fla import _wkv7_pytorch
        return _wkv7_pytorch(r, w, k, v, a, b, state)

    return Wkv7MetalFunction.apply(r, w, k, v, a, b, state)
