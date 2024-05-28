# -*- coding: utf-8 -*-

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers.cache_utils import Cache

from fla.modules import (FusedRMSNormSwishGateLinear, RMSNormLinear,
                         RotaryEmbedding, ShortConvolution)
from fla.modules.activations import ACT2FN, swiglu_linear, swish
from fla.ops.abc.chunk_gate import chunk_gated_abc


class GatedABCAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 1024,
        expand_k: float = 1.,
        expand_v: float = 1.,
        num_heads: int = 4,
        num_kv_heads: Optional[int] = None,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        share_conv_kernel: bool = True,
        num_slots: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        gate_low_rank_dim: Optional[int] = None,
        gate_logit_normalizer: int = 16,
        feature_map: str = 'swish',
        use_rope: bool = False,
        use_output_gate: bool = False,
        use_norm: bool = True,
        layer_idx: Optional[int] = None,
        **kwargs
    ) -> GatedABCAttention:
        super().__init__()

        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups
        self.head_k_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.share_conv_kernel = share_conv_kernel

        if gate_low_rank_dim is None:
            gate_low_rank_dim = self.hidden_size // 16
        self.gate_low_rank_dim = gate_low_rank_dim
        self.gate_logit_normalizer = gate_logit_normalizer

        self.feature_map = feature_map
        self.use_rope = use_rope
        self.use_output_gate = use_output_gate
        self.use_norm = use_norm

        if num_slots is None:
            num_slots = self.head_k_dim
        self.num_slots = num_slots

        self.layer_idx = layer_idx

        if layer_idx is None:
            warnings.warn(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim_per_group, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim_per_group, bias=False)
        self.f_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.num_slots, bias=False)

        if use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            self.conv_size = conv_size
            if share_conv_kernel:
                self.h_conv1d = ShortConvolution(hidden_size, conv_size, activation='silu')
            else:
                self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
                self.k_conv1d = ShortConvolution(self.key_dim_per_group, conv_size, activation='silu')
                self.v_conv1d = ShortConvolution(self.value_dim_per_group, conv_size, activation='silu')

        if self.use_norm:
            if self.use_output_gate:
                self.g_norm = FusedRMSNormSwishGateLinear(self.hidden_size, elementwise_affine, norm_eps)
            else:
                self.g_norm = RMSNormLinear(self.hidden_size, elementwise_affine, norm_eps)
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        if self.use_rope:
            self.rotary = RotaryEmbedding(self.head_k_dim)

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

        last_state = past_key_values[self.layer_idx] if use_cache else None
        if self.use_short_conv:
            conv_state = last_state[0] if use_cache else None
            if self.share_conv_kernel:
                # conv state is updated inplace
                hidden_states = self.h_conv1d(hidden_states, attention_mask, conv_state)
                q = self.q_proj(hidden_states)
                k = self.k_proj(hidden_states)
                v = self.v_proj(hidden_states)
            else:
                conv_state_q = last_state[0] if use_cache else None
                conv_state_k = last_state[1] if use_cache else None
                conv_state_v = last_state[2] if use_cache else None
                q = self.q_proj(hidden_states)
                k = self.k_proj(hidden_states)
                v = self.v_proj(hidden_states)
                q = self.q_conv1d(q, attention_mask, conv_state_q)
                k = self.k_conv1d(k, attention_mask, conv_state_k)
                v = self.v_conv1d(v, attention_mask, conv_state_v)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        f = self.f_proj(hidden_states)

        if self.use_rope:
            q = rearrange(q, '... (h d) -> ... h d', h=self.num_heads)
            k = rearrange(k, '... (h d) -> ... h d', h=self.num_kv_heads)
            seqlen_offset = 0
            if past_key_values is not None:
                seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            q, k = self.rotary(q, k, seqlen_offset)
            q = rearrange(q, 'b n h d -> b h n d', h=self.num_heads)
            k = rearrange(k, 'b n h d -> b h n d', h=self.num_kv_heads)
        else:
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
            if self.num_kv_groups > 1:
                k = repeat(k, 'b n (h d) -> b (h g) n d', h=self.num_kv_heads, g=self.num_kv_groups)
            else:
                k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_kv_heads)
        if self.num_kv_groups > 1:
            v = repeat(v, 'b n (h d) -> b (h g) n d', h=self.num_kv_heads, g=self.num_kv_groups)
            f = repeat(f, 'b n (h m) -> b (h g) n m', h=self.num_kv_heads, g=self.num_kv_groups)
        else:
            v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_kv_heads)
            f = rearrange(f, 'b n (h m) -> b h n m', h=self.num_kv_heads)

        if self.feature_map is not None:
            q, k, v = map(lambda x: ACT2FN[self.feature_map](x), (q, k, v))
        f = F.logsigmoid(f) / self.gate_logit_normalizer
        s = (1 - f.exp()).to(f.dtype)
        # dealing with left-padding
        if attention_mask is not None:
            s = s.mul_(attention_mask.view(attention_mask.shape[0], 1, -1, 1))
            v = v.mul_(attention_mask.view(attention_mask.shape[0], 1, -1, 1))

        recurrent_state = last_state[-2:] if use_cache else None
        o, recurrent_state = chunk_gated_abc(q, k, v, s, f,
                                             initial_state=recurrent_state,
                                             output_final_state=use_cache)
        if past_key_values is not None:
            if self.use_short_conv:
                if self.share_conv_kernel:
                    last_state = (conv_state,) + recurrent_state
                else:
                    last_state = (conv_state_q, conv_state_k, conv_state_v) + recurrent_state
            else:
                last_state = recurrent_state
            past_key_values.update(last_state, self.layer_idx, q.shape[2])

        o = rearrange(o, 'b h t d -> b t (h d)')
        if self.use_norm and not self.use_output_gate:
            o = swish(o)
            o = self.g_norm(o, self.o_proj.weight, self.o_proj.bias)
        elif self.use_output_gate and not self.use_norm:
            o = swiglu_linear(self.g_proj(hidden_states), o, self.o_proj.weight, self.o_proj.bias)
        elif self.use_output_gate and self.use_norm:
            o = self.g_norm(o, self.g_proj(hidden_states), self.o_proj.weight, self.o_proj.bias)
        else:
            o = self.o_proj(o)
        return o, None, past_key_values

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = tuple()
        if self.use_short_conv:
            if self.share_conv_kernel:
                state += (param.new_zeros(batch_size, self.hidden_size, self.conv_size),)
            else:
                state += (param.new_zeros(batch_size, self.key_dim, self.conv_size),
                          param.new_zeros(batch_size, self.key_dim, self.conv_size),
                          param.new_zeros(batch_size, self.value_dim, self.conv_size))
        state += (param.new_zeros(batch_size, self.num_heads, self.head_k_dim, self.num_slots),
                  param.new_zeros(batch_size, self.num_heads, self.num_slots, self.head_v_dim))
        return state

    def state_size(self, sequence_length: int = 2048):
        return self.num_heads * self.key_dim * self.head_v_dim
