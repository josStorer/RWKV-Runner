# -*- coding: utf-8 -*-

from typing import Optional

import torch


def naive_recurrent_rwkv6(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: Optional[bool] = False
):
    orig_dtype = q.dtype
    B, H, T, K, V = *q.shape, v.shape[-1]
    q, k, v, w, u = map(lambda x: x.float(), (q, k, v, w, u))
    h = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    o = torch.zeros_like(v)

    if scale is None:
        scale = K ** -0.5

    if initial_state is not None:
        h += initial_state

    for i in range(T):
        q_i = q[:, :, i, :] * scale
        k_i = k[:, :, i]
        v_i = v[:, :, i, :]
        w_i = w[:, :, i].exp()
        kv_i = k_i[..., None] * v_i[..., None, :]
        o_i = (h + u[None, ..., None] * kv_i) * q_i[..., None]
        o[:, :, i] = o_i.sum(-2)
        h = h * w_i[..., None] + kv_i
    ht = h if output_final_state else None
    return o.to(orig_dtype), ht


def naive_recurrent_rwkv6_bwd(
    q,
    k,
    v,
    w,
    u,
    o,
    do,
    initial_state=None,
    output_final_state=False
):
    q, k, v, w, u, o, do = map(lambda x: x.float(), (q, k, v, w, u, o, do))
    B, H, T, K, V = *q.shape, v.shape[-1]
    h = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    dq = torch.zeros_like(q)
    dq_aux = torch.zeros_like(q)

    if initial_state is not None:
        h += initial_state

    for i in range(T):
        k_i = k[:, :, i]
        v_i = v[:, :, i]
        w_i = w[:, :, i].exp()
        kv_i = k_i[..., None] * v_i[..., None, :]
        h_i = (h + u[None, ..., None] * kv_i)
        dq_i = (do[:, :, i, None, :] * h_i).sum(-1)
        dq_aux_i = (do[:, :, i, None, :] * h).sum(-1)
        dq[:, :, i] = dq_i
        dq_aux[:, :, i] = dq_aux_i
        h = h * w_i[..., None] + kv_i

    du = torch.zeros_like(u)
    dh = torch.zeros_like(h)
    dk = torch.zeros_like(k)
    dk_aux = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    for i in range(T - 1, -1, -1):
        d_kv_i = do[:, :, i, None, :] * q[:, :, i, :, None]
        k_i = k[:, :, i]
        v_i = v[:, :, i]
        du_i = (d_kv_i * k_i[..., None] * v_i[..., None, :]).sum(-1)
        du += du_i
        dk_i = (dh * v_i[..., None, :]).sum(-1)
        dk_aux[:, :, i] = dk_i
        dk_i += (d_kv_i * u[None, ..., None] * v_i[..., None, :]).sum(-1)
        dv_i = (d_kv_i * u[None, ..., None] * k_i[..., None]).sum(-2)
        dv_i += (dh * k_i[..., None]).sum(-2)

        dk[:, :, i] = dk_i
        dv[:, :, i] = dv_i
        dh = dh * w[:, :, i, :, None].exp() + d_kv_i

    # dw = q * dq_aux - k * dk_aux
    dw = torch.zeros_like(w)
    for i in range(T - 2, -1, -1):
        dw[:, :, i] = dw[:, :, i+1] + dq_aux[:, :, i+1] * q[:, :, i+1] - dk_aux[:, :, i] * k[:, :, i]

    return dq, dk, dv, dw, du
