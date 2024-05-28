# -*- coding: utf-8 -*-

from typing import Optional

import torch


def naive_recurrent_abc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[int] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: Optional[bool] = False
) -> torch.Tensor:
    dtype = q.dtype

    # [batch_size, n_heads, seq_len, n_slots]
    if g is None:
        z = s.float().logcumsumexp(2)
        g = torch.cat((z[:, :, :1], z[:, :, :-1]), 2) - z
        s = torch.exp(s - z)
    q, k, v, s, g = map(lambda x: x.float(), (q, k, v, s, g))
    B, H, T, K, V, M = *q.shape, v.shape[-1], s.shape[-1]

    hk = torch.zeros(B, H, K, M, dtype=torch.float, device=q.device)
    ok = torch.zeros_like(s)

    if scale is None:
        scale = q.shape[-1] ** -0.5

    final_state = None
    if initial_state is not None:
        hk += initial_state[0]

    for i in range(T):
        q_i = q[:, :, i] * scale
        k_i = k[:, :, i]
        v_i = s[:, :, i]
        g_i = g[:, :, i].exp()
        hk = hk * g_i[..., None, :] + k_i[..., None] * v_i[..., None, :]
        ok[:, :, i] = (q_i[..., None] * hk).sum(-2)

    qv = ok.softmax(-1)
    hv = torch.zeros(B, H, M, V, dtype=torch.float, device=q.device)
    ov = torch.zeros_like(v)
    if initial_state is not None:
        hv += initial_state[1]

    for i in range(T):
        q_i = qv[:, :, i]
        k_i = s[:, :, i]
        v_i = v[:, :, i]
        g_i = g[:, :, i].exp()
        hv = hv * g_i[..., :, None] + k_i[..., None] * v_i[..., None, :]
        ov[:, :, i] = (q_i[..., None] * hv).sum(-2)

    if output_final_state:
        final_state = (hk, hv)
    return ov.to(dtype), final_state


def naive_cumsum_abc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor
) -> torch.Tensor:
    """
    A simple implementation of vanilla ABC that is more aligned with the descriptions in the paper.
    This is just for demonstration purposes, with no numerical stabilities guaranteed.
    """

    dtype = q.dtype
    q, k, v, s = map(lambda x: x.float(), (q, k, v, s))

    scale = q.shape[-1] ** -0.5
    # [batch_size, n_heads, seq_len, n_slots]
    s = (s - s.max(2, True)[0]).exp()
    z = s.cumsum(2)
    # [batch_size, n_heads, seq_len, n_slots, d_head]
    K = (s.unsqueeze(-1) * k.unsqueeze(-2)).cumsum(2) / z.unsqueeze(-1)
    V = (s.unsqueeze(-1) * v.unsqueeze(-2)).cumsum(2) / z.unsqueeze(-1)
    # [batch_size, n_heads, seq_len, n_slots]
    p = torch.einsum('...d,...md->...m', q * scale, K).softmax(-1)
    # [batch_size, n_heads, seq_len, d_head]
    o = torch.einsum('...m,...md->...d', p, V)
    return o.to(dtype), None
