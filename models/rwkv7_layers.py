import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.common import trunc_normal_init_


HEAD_SIZE = 64


def get_head_size(hidden_size: int) -> int:
    return HEAD_SIZE


def wkv7_pytorch(r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor,
                 state: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
    B, T, C = r.shape
    H = C // HEAD_SIZE
    N = HEAD_SIZE
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


def _lora_dim(dim: int, multiplier: float) -> int:
    d = int(round(multiplier * (dim ** 0.5)))
    d = max(32, d)
    d = (d // 32) * 32
    return d


class Rwkv7TimeMix(nn.Module):
    def __init__(self, hidden_size: int, layer_id: int, n_layer: int, head_size: int = HEAD_SIZE):
        super().__init__()
        self.layer_id = layer_id
        self.head_size = head_size
        self.n_head = hidden_size // head_size
        C = hidden_size
        H = self.n_head
        N = head_size

        ddd = torch.ones(1, 1, C)
        for i in range(C):
            ddd[0, 0, i] = i / C

        ratio_0_to_1 = layer_id / max(1, n_layer - 1)
        ratio_1_to_almost0 = 1.0 - (layer_id / max(1, n_layer))

        self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
        self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

        def ortho_init(x, scale):
            shape = x.shape
            if len(shape) == 2:
                gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                nn.init.orthogonal_(x, gain=gain * scale)
            elif len(shape) == 3:
                gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                for i in range(shape[0]):
                    nn.init.orthogonal_(x[i], gain=gain * scale)
            return x

        www = torch.zeros(C)
        zigzag = torch.zeros(C)
        linear = torch.zeros(C)
        for n in range(C):
            linear[n] = n / (C - 1) - 0.5
            zigzag[n] = ((n % N) - ((N - 1) / 2)) / ((N - 1) / 2)
            zigzag[n] = zigzag[n] * abs(zigzag[n])
            www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)

        D_DECAY_LORA = _lora_dim(C, 2.5)
        self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
        self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
        self.w0 = nn.Parameter(www.reshape(1, 1, C) + 0.5 + zigzag * 2.5)

        D_AAA_LORA = _lora_dim(C, 2.5)
        self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
        self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
        self.a0 = nn.Parameter(torch.zeros(1, 1, C) - 0.19 + zigzag * 0.3 + linear * 0.4)

        D_MV_LORA = _lora_dim(C, 1.7)
        self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
        self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
        self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 0.73 - linear * 0.4)

        D_GATE_LORA = _lora_dim(C, 5.0)
        self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
        self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

        self.k_k = nn.Parameter(torch.zeros(1, 1, C) + 0.71 - linear * 0.1)
        self.k_a = nn.Parameter(torch.zeros(1, 1, C) + 1.02)
        self.r_k = nn.Parameter(torch.zeros(H, N) - 0.04)

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

        self.receptance.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
        self.key.weight.data.uniform_(-0.05 / (C ** 0.5), 0.05 / (C ** 0.5))
        self.value.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
        self.output.weight.data.zero_()

    def forward(self, x: Tensor, v_first: Optional[Tensor] = None,
                state: Optional[Tensor] = None) -> tuple[Tensor, Tensor, Tensor]:
        unsqueeze = x.ndim == 2
        if unsqueeze:
            x = x.unsqueeze(0)
        B, T, C = x.shape
        H = self.n_head

        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)

        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)

        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        from models.rwkv7_fla import wkv7 as _wkv7
        x, state = _wkv7(r, w, k, v, -kk, kk * a, state=state)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        x = self.output(x * g)

        if unsqueeze:
            x = x.squeeze(0)
        return x, v_first, state
