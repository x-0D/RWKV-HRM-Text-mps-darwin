import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.common import trunc_normal_init_
from models.layers import find_multiple
from models.rwkv7_layers import Rwkv7TimeMix, get_head_size


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int,
                 init_std_in: float = 0.02, init_std_out: float = 0.02):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        nn.init.trunc_normal_(self.gate_up_proj.weight, std=init_std_in)
        nn.init.trunc_normal_(self.down_proj.weight, std=init_std_out)

    def forward(self, x: Tensor) -> Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Rwkv7Block(nn.Module):
    """RWKV-7 block: TimeMix + SwiGLU MLP."""

    def __init__(self, hidden_size: int, layer_id: int, n_layer: int,
                 intermediate_size: int, norm_eps: float = 1e-6,
                 head_size: int = 64):
        super().__init__()
        self.attn = Rwkv7TimeMix(
            hidden_size=hidden_size,
            layer_id=layer_id,
            n_layer=n_layer,
            head_size=head_size,
        )
        self.mlp = SwiGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            init_std_in=1.0 / math.sqrt(hidden_size),
            init_std_out=1.0 / math.sqrt(intermediate_size),
        )
        self.norm = lambda x: F.rms_norm(x, (x.shape[-1],), eps=norm_eps)

    def forward(self, x: Tensor, v_first: Optional[Tensor] = None,
                state: Optional[Tensor] = None) -> tuple[Tensor, Tensor, Tensor]:
        attn_out, v_first, state = self.attn(self.norm(x), v_first, state=state)
        x = x + attn_out
        x = x + self.mlp(self.norm(x))
        return x, v_first, state


class PureRwkv7(nn.Module):
    """
    Pure RWKV-7 model: flat stack of Rwkv7Blocks.
    
    No attention, no KV cache, no H/L stacks, no explicit carry.
    Memory and hierarchy run through RWKV-7's recurrent state transitions.
    
    Features:
    - Stateful forward: pass `states` for persistent WKV across calls
    - Zero attention: O(T) memory regardless of sequence length
    - Constant memory KV: the state is fixed-size (B, H, N, N) per layer
    - HRM hierarchy via RADLADS: heads initialized from HRM H/L module weights
    
    Head architecture (hidden=768, head_dim=64, heads=12):
    - heads 0-5: initialized from HRM's H_module (slow decay, high-level)
    - heads 6-11: initialized from HRM's L_module (fast decay, low-level)
    - Hierarchy emerges from differing state dynamics across heads.
    """

    def __init__(self, n_layers: int, hidden_size: int, num_heads: int,
                 expansion: float = 4, norm_eps: float = 1e-6,
                 head_size: int = 64, vocab_size: int = 65536):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.n_layers = n_layers

        intermediate_size = find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layers = nn.ModuleList([
            Rwkv7Block(
                hidden_size=hidden_size,
                layer_id=i,
                n_layer=n_layers,
                intermediate_size=intermediate_size,
                norm_eps=norm_eps,
                head_size=head_size,
            ) for i in range(n_layers)
        ])
        self.norm_f = lambda x: F.rms_norm(x, (x.shape[-1],), eps=norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        self.head_hint = {
            "in": {"dim": hidden_size, "init_std": 1.0 / math.sqrt(hidden_size)},
            "out": {"dim": hidden_size, "init_std": 1.0 / math.sqrt(hidden_size)},
        }

    def forward(self, x: Tensor, states: Optional[list[Optional[Tensor]]] = None
                ) -> tuple[Tensor, list[Tensor]]:
        unsqueeze = x.ndim == 1
        if unsqueeze:
            x = x.unsqueeze(0)

        x = self.embed_tokens(x)
        v_first = None
        new_states: list[Tensor] = []

        for i, layer in enumerate(self.layers):
            s = states[i] if states is not None else None
            x, v_first, s = layer(x, v_first=v_first, state=s)
            new_states.append(s)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        if unsqueeze:
            logits = logits.squeeze(0)
        return logits, new_states

    def initial_state(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> list[None]:
        return [None for _ in range(self.n_layers)]

    def create_cache(self, **kwargs) -> None:
        return None
