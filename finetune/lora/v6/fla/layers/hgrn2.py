# -*- coding: utf-8 -*-

# "HGRN2: Gated Linear RNNs with State Expansion"[https://arxiv.org/abs/2404.07904]

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.cache_utils import Cache

from fla.modules import RMSNorm, ShortConvolution
from fla.modules.activations import swish
from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla


class HGRN2Attention(nn.Module):

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        num_heads: Optional[int] = None,
        expand_ratio: Optional[int] = 128,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        share_conv_kernel: bool = True,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        layer_idx: int = None
    ) -> HGRN2Attention:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size

        if expand_ratio is None and num_heads is not None:
            expand_ratio = hidden_size // num_heads
        elif expand_ratio is not None and num_heads is None:
            num_heads = hidden_size // expand_ratio
        else:
            raise RuntimeError("One of `expand_ratio` or `num_heads` should be provided.")
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.share_conv_kernel = share_conv_kernel

        self.forget_dim = int(self.num_heads * self.expand_ratio)
        self.input_dim = hidden_size
        self.layer_idx = layer_idx

        assert mode in ['chunk', 'fused_recurrent', 'fused_chunk'], f"Not suppoerted mode `{mode}`."
        assert self.forget_dim % num_heads == 0, f"forget dim must be divisible by num_heads of {num_heads}"
        assert self.input_dim % num_heads == 0, f"input dim must be divisible by num_heads of {num_heads}"

        self.head_f_dim = self.expand_ratio
        self.head_i_dim = self.hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, self.forget_dim, bias=False)
        self.f_proj = nn.Linear(hidden_size, self.forget_dim, bias=False)
        self.i_proj = nn.Linear(hidden_size, self.input_dim, bias=False)

        if use_short_conv:
            self.conv_size = conv_size
            if share_conv_kernel:
                self.h_conv1d = ShortConvolution(hidden_size, conv_size, activation='silu')
            else:
                self.q_conv1d = ShortConvolution(self.forget_dim, conv_size, activation='silu')
                self.f_conv1d = ShortConvolution(self.forget_dim, conv_size, activation='silu')
                self.i_conv1d = ShortConvolution(self.input_dim, conv_size, activation='silu')

        self.g_norm = RMSNorm(self.hidden_size, elementwise_affine, norm_eps)
        self.o_proj = nn.Linear(self.input_dim, hidden_size, bias=False)

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        lower_bound: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        # launching the triton kernel for just one token will actually be slower
        mode = 'fused_recurrent' if hidden_states.shape[1] == 1 else self.mode

        last_state = past_key_values[self.layer_idx] if use_cache else None
        if self.use_short_conv:
            conv_state = last_state[0] if use_cache else None
            if self.share_conv_kernel:
                # conv state is updated inplace
                hidden_states = self.h_conv1d(hidden_states, attention_mask, conv_state)
                q = self.q_proj(hidden_states)
                f = self.f_proj(hidden_states)
                i = self.i_proj(hidden_states)
            else:
                conv_state_q = last_state[0] if use_cache else None
                conv_state_f = last_state[1] if use_cache else None
                conv_state_i = last_state[2] if use_cache else None
                q = self.q_proj(hidden_states)
                f = self.f_proj(hidden_states)
                i = self.i_proj(hidden_states)
                q = self.q_conv1d(q, attention_mask, conv_state_q)
                f = self.f_conv1d(f, attention_mask, conv_state_f)
                i = self.i_conv1d(i, attention_mask, conv_state_i)
        else:
            q = self.q_proj(hidden_states)
            f = self.f_proj(hidden_states)
            i = self.i_proj(hidden_states)

        # dealing with left-padding
        if attention_mask is not None:
            i = i.mul_(attention_mask.unsqueeze(-1))

        q = swish(q)
        # the lower bound for the first layer is zero
        if lower_bound is None or self.layer_idx == 0:
            k, g = 1 - f.sigmoid(), F.logsigmoid(f)
        else:
            g = lower_bound + (1 - lower_bound) * f.sigmoid()
            k, g = 1 - g, g.log()
        q, k, i, g = map(lambda x: rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads), (q, k, i, g))

        recurrent_state = last_state[-1] if use_cache else None
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gla(q, k, i, g, initial_state=recurrent_state, output_final_state=use_cache)
        elif mode == 'fused_chunk':
            o, recurrent_state = fused_chunk_gla(q, k, i, g, initial_state=recurrent_state, output_final_state=use_cache)
        elif mode == 'chunk':
            o, recurrent_state = chunk_gla(q, k, i, g, initial_state=recurrent_state, output_final_state=use_cache)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            if self.use_short_conv:
                if self.share_conv_kernel:
                    last_state = (conv_state, recurrent_state)
                else:
                    last_state = (conv_state_q, conv_state_f, conv_state_i, recurrent_state)
            else:
                last_state = (recurrent_state,)
            past_key_values.update(last_state, self.layer_idx, q.shape[2])

        o = self.g_norm(rearrange(o, 'b h l d -> b l (h d)'))
        o = self.o_proj(o)

        return o, None, past_key_values

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = tuple()
        if self.use_short_conv:
            if self.share_conv_kernel:
                state += (param.new_zeros(batch_size, self.hidden_size, self.conv_size),)
            else:
                state += (param.new_zeros(batch_size, self.forget_dim, self.conv_size),
                          param.new_zeros(batch_size, self.forget_dim, self.conv_size),
                          param.new_zeros(batch_size, self.input_dim, self.conv_size))
        state += (param.new_zeros(batch_size, self.num_heads, self.head_f_dim, self.head_i_dim),)
        return state

    def state_size(self, **kwargs) -> int:
        state_size = self.forget_dim * self.head_i_dim
        for module in self.children():
            if isinstance(module, ShortConvolution):
                state_size += module.state_size
        return state_size
