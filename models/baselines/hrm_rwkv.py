import math
from typing import Any, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.common import trunc_normal_init_
from models.layers import find_multiple, SwiGLU
from models.rwkv7_layers import Rwkv7TimeMix, get_head_size


class RwkvTransformerBlockConfig:
    def __init__(self, hidden_size: int, intermediate_size: int, n_layer_in_stack: int,
                 norm_eps: float = 1e-6, head_size: int = 64):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.n_layer_in_stack = n_layer_in_stack
        self.norm_eps = norm_eps
        self.head_size = head_size


class RwkvTransformerBlock(nn.Module):
    def __init__(self, config: RwkvTransformerBlockConfig, layer_id: int):
        super().__init__()
        self.attn = Rwkv7TimeMix(
            hidden_size=config.hidden_size,
            layer_id=layer_id,
            n_layer=config.n_layer_in_stack,
            head_size=config.head_size,
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            init_std_in=1.0 / math.sqrt(config.hidden_size),
            init_std_out=1.0 / math.sqrt(config.intermediate_size),
        )
        self.norm = lambda x: F.rms_norm(x, (x.shape[-1],), eps=config.norm_eps)

    def forward(self, x: Tensor, v_first: Optional[Tensor] = None,
                state: Optional[Tensor] = None, **seq_info) -> tuple[Tensor, Tensor, Tensor]:
        attn_out, v_first, state = self.attn(self.norm(x), v_first, state=state)
        x = x + attn_out
        x = x + self.mlp(self.norm(x))
        return x, v_first, state


class RwkvStack(nn.Module):
    def __init__(self, config: RwkvTransformerBlockConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            RwkvTransformerBlock(config, i) for i in range(config.n_layer_in_stack)
        ])
        self.norm_f = lambda x: F.rms_norm(x, (x.shape[-1],), eps=config.norm_eps)

    def forward(self, x: Tensor, wkv_states: Optional[list[Tensor]] = None,
                **seq_info) -> tuple[Tensor, Tensor, list[Tensor]]:
        v_first = None
        next_states: list[Tensor] = []
        for i, layer in enumerate(self.layers):
            s = wkv_states[i] if wkv_states is not None else None
            x, v_first, s = layer(x, v_first=v_first, state=s, **seq_info)
            next_states.append(s)
        return self.norm_f(x), v_first, next_states


class HierarchicalReasoningModelRWKVConfig:
    def __init__(self, **kwargs):
        self.n_layers: int = kwargs.get("n_layers", 12)
        self.hidden_size: int = kwargs.get("hidden_size", 768)
        self.num_heads: int = kwargs.get("num_heads", 12)
        self.expansion: float = kwargs.get("expansion", 4)
        self.norm_eps: float = kwargs.get("norm_eps", 1e-6)
        self.head_size: int = kwargs.get("head_size", 64)
        self.half_layers: bool = kwargs.get("half_layers", False)
        self.H_cycles: int = kwargs.get("H_cycles", 2)
        self.L_cycles: int = kwargs.get("L_cycles", 3)
        self.bp_warmup_ratio: float = kwargs.get("bp_warmup_ratio", 0.0)
        self.bp_min_steps: int = kwargs.get("bp_min_steps", 2)
        self.bp_max_steps: int = kwargs.get("bp_max_steps", 5)
        self.vocab_size: int = kwargs.get("vocab_size", 65536)

    @property
    def intermediate_size(self):
        return find_multiple(round(self.expansion * self.hidden_size * 2 / 3), 256)

    @property
    def layers_per_stack(self):
        if self.half_layers:
            return self.n_layers // 2
        return self.n_layers


class HierarchicalReasoningModelRWKV(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        config = HierarchicalReasoningModelRWKVConfig(**config_dict)

        stack_config = RwkvTransformerBlockConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            n_layer_in_stack=config.layers_per_stack,
            norm_eps=config.norm_eps,
            head_size=config.head_size,
        )

        self.H_level = RwkvStack(stack_config)
        self.L_level = RwkvStack(stack_config)

        self.H_cycles = config.H_cycles
        self.L_cycles = config.L_cycles
        self.bp_warmup_ratio = config.bp_warmup_ratio
        self.bp_min_steps = config.bp_min_steps
        self.bp_max_steps = config.bp_max_steps

        self.hidden_size = config.hidden_size
        self.head_size = config.head_size
        self.layers_per_stack = config.layers_per_stack
        self.head_hint = {
            "in": {"dim": config.hidden_size, "init_std": 1.0 / math.sqrt(config.hidden_size)},
            "out": {"dim": config.hidden_size, "init_std": 1.0 / math.sqrt(config.hidden_size)},
        }

        self.zL_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=torch.bfloat16), std=1.0),
            persistent=True,
        )

    def _make_empty_states(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> tuple[list[Tensor], list[Tensor]]:
        H = self.hidden_size // self.head_size
        N = self.head_size
        n = self.layers_per_stack
        def empty():
            return [torch.zeros(batch_size, H, N, N, device=device, dtype=dtype) for _ in range(n)]
        return empty(), empty()

    def forward(self, carry: None, x: Tensor, cache=None, bp_steps: int = 2,
                **seq_info) -> tuple[tuple[list[Tensor], list[Tensor]], Tensor]:
        z_H, z_L = x, self.zL_init

        # Unpack or create persistent WKV states
        if carry is None:
            H_states, L_states = self._make_empty_states(1, x.dtype, x.device)
        else:
            H_states, L_states = carry

        H_bp_steps = min(self.H_cycles, bp_steps - 1)
        L_bp_steps = bp_steps - H_bp_steps

        for i in range(self.H_cycles):
            for k in range(self.L_cycles):
                with torch.set_grad_enabled(
                    torch.is_grad_enabled() and (k >= self.L_cycles - L_bp_steps)
                ):
                    z_L_out, _, L_states = self.L_level(z_L + z_H, wkv_states=L_states, **seq_info)
                    z_L = z_L_out

            with torch.set_grad_enabled(
                torch.is_grad_enabled() and (i >= self.H_cycles - H_bp_steps)
            ):
                z_H_out, _, H_states = self.H_level(z_H + z_L.to(z_H.device), wkv_states=H_states, **seq_info)
                z_H = z_H_out

        return ([s.detach() for s in H_states], [s.detach() for s in L_states]), z_H

    def compute_train_extra_args(self, train_state: Any) -> dict[str, Any]:
        ratio = min(1, train_state.step / max(1, train_state.total_steps * self.bp_warmup_ratio))
        return dict(
            bp_steps=self.bp_min_steps + int(ratio * (self.bp_max_steps - self.bp_min_steps))
        )

    def initial_carry(self, batch_size: int, dtype: torch.dtype) -> tuple[list[Tensor], list[Tensor]]:
        return self._make_empty_states(batch_size, dtype, next(self.parameters()).device)

    def create_cache(self, **kwargs) -> None:
        return None
