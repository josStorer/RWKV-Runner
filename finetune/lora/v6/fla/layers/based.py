# -*- coding: utf-8 -*-

"""
Linear attention in Based.
https://github.com/HazyResearch/zoology/blob/main/zoology/mixers/based.py
"""

import torch
import torch.nn as nn
from einops import rearrange

from fla.modules.feature_map import TaylorFeatureMap
from fla.ops.based import parallel_based
from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn


class BasedLinearAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        l_max: int = 2048,
        feature_dim: int = 16,
        num_key_value_heads: int = 12,
        num_heads: int = 12,
        feature_name: str = "taylor_exp",
        eps: float = 1e-12,
        causal: bool = True,
        mode: str = "parallel",
    ):
        super().__init__()
        self.hidden_size
        self.l_max = l_max
        self.mode = mode
        assert self.mode in ["fused_chunk", "parallel", 'chunk']

        # linear attention
        self.feature_name = feature_name
        self.feature_dim = feature_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_key_value_heads
        self.causal = causal

        self.q_proj = nn.Linear(self.hidden_size, self.feature_dim * self.num_heads, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.feature_dim * self.num_heads, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.dropout = nn.Identity()
        self.feature_map = TaylorFeatureMap(feature_dim)
        self.eps = eps

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        mode = self.mode
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        q, k, v = map(lambda x: rearrange(x, "b l (h d) -> b h l d", h=self.num_heads), [q, k, v])
        if mode == "fused_chunk":
            q, k = self.feature_map(q), self.feature_map(k)
            o = fused_chunk_linear_attn(q, k, v, normalize=True, scale=1)
        elif mode == 'chunk':
            q, k = self.feature_map(q), self.feature_map(k)
            o = chunk_linear_attn(q, k, v, normalize=True, scale=1)
        elif mode == 'parallel':
            assert q.shape[-1] <= 128
            o = parallel_based(q, k, v, True, True)
        o = rearrange(o, "b h l d -> b l (h d)")
        o = self.o_proj(o)
        o = self.dropout(o)
        return o

    # https://github.com/HazyResearch/zoology/blob/main/zoology/mixers/based.py#L119

    def forward_reference(self, hidden_states: torch.Tensor, filters: torch.Tensor = None, *args, **kwargs):
        """
        x (torch.Tensor): tensor of shape (b, d, l)
        y (torch.Tensor): tensor of shape (b, d, l)
        """
        # hidden_states = hidden_states.transpose(1, 2)
        b, l, _ = hidden_states.size()
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)

        q = q.view(b, l, self.num_heads, self.feature_dim).transpose(1, 2)
        k = k.view(b, l, self.num_key_value_heads, self.feature_dim).transpose(1, 2)
        v = v.view(b, l, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Linear attention
        q, k = self.feature_map(q), self.feature_map(k)
        q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)

        # Compute attention
        if self.causal:
            y = ((q * (k * v).cumsum(2)).sum(-1) / ((q * k.cumsum(2)).sum(-1) + self.eps))
        else:
            y = ((q * (k * v).sum(2, True)).sum(-1) / ((q * k.sum(2, True)).sum(-1) + self.eps))
        y = rearrange(y, 'b h l d -> b l (h d)')
        y = self.o_proj(y.to(hidden_states.dtype))
        y = self.dropout(y)
        return y.to(hidden_states.dtype)


if __name__ == '__main__':
    batch = 4
    seq_len = 1024
    hidden_size = 1024
    dtype = torch.float32
    x = torch.randn(batch, seq_len, hidden_size).to(dtype).cuda().requires_grad_(True)
    dy = torch.randn(batch, seq_len, hidden_size).to(dtype).cuda()
    model = BasedLinearAttention(hidden_size, mode='chunk').to(dtype).cuda()
    y = model(x)
    y.backward(dy, retain_graph=True)
    x_grad, x.grad = x.grad, None
    y2 = model.forward_reference(x)
    y2.backward(dy)
    assert y.allclose(y2, 0, 1e-4), breakpoint()
    assert x_grad.allclose(x.grad, 0, 1e-4), breakpoint()
    print("Pass")
