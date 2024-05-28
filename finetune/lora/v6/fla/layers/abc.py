# -*- coding: utf-8 -*-

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from transformers.cache_utils import Cache

from fla.modules import (FusedRMSNormSwishGate, RMSNorm, RotaryEmbedding,
                         ShortConvolution)
from fla.modules.activations import swiglu, swish
from fla.modules.convolution import proj_then_conv1d
from fla.ops.abc.chunk import chunk_abc


class ABCAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 1024,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        num_heads: int = 4,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        share_conv_kernel: bool = True,
        num_slots: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        gate_low_rank_dim: int = 16,
        gate_logit_normalizer: int = 16,
        use_input_gate: bool = False,
        use_output_gate: bool = True,
        use_norm: bool = True,
        clamp_min: Optional[float] = -32,
        clamp_max: Optional[float] = 32,
        layer_idx: Optional[int] = None,
        **kwargs
    ) -> ABCAttention:
        super().__init__()

        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.key_dim = int(self.hidden_size * self.expand_k)
        self.value_dim = int(self.hidden_size * self.expand_v)
        self.head_k_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.share_conv_kernel = share_conv_kernel

        self.gate_low_rank_dim = gate_low_rank_dim
        self.gate_logit_normalizer = gate_logit_normalizer

        self.use_input_gate = use_input_gate
        self.use_output_gate = use_output_gate
        self.use_norm = use_norm

        if num_slots is None:
            num_slots = self.head_k_dim
        self.num_slots = num_slots

        self.norm_eps = norm_eps

        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.layer_idx = layer_idx

        if layer_idx is None:
            warnings.warn(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)

        if use_output_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.s_proj = nn.Linear(self.hidden_size, self.num_heads * self.num_slots, bias=False)
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        if use_short_conv:
            self.conv_size = conv_size
            if share_conv_kernel:
                self.h_conv1d = ShortConvolution(hidden_size, conv_size, activation='silu')
            else:
                self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
                self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
                self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')

        if self.use_norm:
            if self.use_output_gate:
                self.g_norm = FusedRMSNormSwishGate(self.head_v_dim, elementwise_affine, norm_eps)
            else:
                self.g_norm = RMSNorm(self.head_v_dim, elementwise_affine, norm_eps)

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
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:

        if self.use_short_conv:
            if self.share_conv_kernel:
                hidden_states = self.h_conv1d(hidden_states)
                q = self.q_proj(hidden_states)
                k = self.k_proj(hidden_states)
                v = self.v_proj(hidden_states)
            else:
                q = proj_then_conv1d(hidden_states, self.q_proj.weight, self.q_conv1d.weight, self.q_conv1d.bias)
                k = proj_then_conv1d(hidden_states, self.k_proj.weight, self.k_conv1d.weight, self.k_conv1d.bias)
                v = proj_then_conv1d(hidden_states, self.v_proj.weight, self.v_conv1d.weight, self.v_conv1d.bias)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        if self.use_input_gate:
            q, k, v = map(lambda x: swish(x), (q, k, v))

        if self.use_rope:
            q = rearrange(q, '... (h d) -> ... h d', h=self.num_heads)
            k = rearrange(k, '... (h d) -> ... h d', h=self.num_heads)
            seqlen_offset = 0
            if past_key_values is not None:
                seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            q, k = self.rotary(q, k, seqlen_offset)
            q = rearrange(q, 'b n h d -> b h n d', h=self.num_heads)
            k = rearrange(k, 'b n h d -> b h n d', h=self.num_heads)
        else:
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
            k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        # [batch_size, n_heads, seq_len, num_slots]
        s = rearrange(self.s_proj(hidden_states), 'b t (h m) -> b h t m', h=self.num_heads)
        s = s.clamp_(self.clamp_min, self.clamp_max)

        last_state = past_key_values[self.layer_idx] if use_cache else None
        o, last_state = chunk_abc(q, k, v, s, initial_state=last_state, output_final_state=use_cache)
        if past_key_values is not None and last_state is not None:
            past_key_values.update(last_state, self.layer_idx, q.shape[2])

        o = rearrange(o, 'b h t d -> b t h d')
        if self.use_norm and not self.use_output_gate:
            o = self.g_norm(o)
        elif self.use_output_gate:
            g = rearrange(self.g_proj(hidden_states), 'b t (h d) -> b t h d', h=self.num_heads)
            o = self.g_norm(o, g) if self.use_norm else swiglu(g, o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        return o, None, past_key_values

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = tuple()
        if self.use_short_conv:
            state += (param.new_zeros(batch_size, self.hidden_size, self.conv_size),)
        state += (param.new_zeros(batch_size, self.num_heads, self.head_k_dim, self.num_slots),
                  param.new_zeros(batch_size, self.num_heads, self.num_slots, self.head_v_dim))
        return state

    def state_size(self, sequence_length: int = 2048):
        return self.num_heads * self.key_dim * self.head_v_dim
