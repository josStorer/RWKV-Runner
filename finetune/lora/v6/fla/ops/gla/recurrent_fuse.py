# -*- coding: utf-8 -*-

# Copyright (c) 2023, Songlin Yang

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd

from fla.utils import contiguous

# on-the-fly computation without materializing hidden statets into HBMs


@triton.jit
def fused_recurrent_gla_fwd_kernel(
    # B: batch_size, H: n_heads, T: seq_len, D: d_head
    q,  # query [B, H, L, D_head_K]
    k,  # key [B, H, L, D_head_K]
    v,  # value [B, H, L, D_head_V]
    gk,  # log gate [B, H, L, D_head_K]
    gv,  # log gate [B, H, L, D_head_V]
    o,  # output [B, H, L, D_head_V]
    # initial hidden state initialization [B, H, D_head_K, D_head_V]
    initial_state,
    final_state,  # final hidden state [B, H, D_head_K, D_head_V]

    s_qk_h,  # stride size: L * D_head_K
    s_qk_t,  # stride size: D_head_K
    s_qk_d,  # stride size: 1

    s_vo_h,  # stride size: L * D_head_V
    s_vo_t,  # stride size: D_head_V
    s_vo_d,  # stride size: 1

    B,  # batch size
    H,  # n_heads
    T,  # seq_len
    scale,  # D_head_K ** -0.5
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    DK: tl.constexpr,  # D_head_K
    DV: tl.constexpr,  # D_head_V
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    STORE_FINAL_STATE: tl.constexpr,  # whether to store final state
    REVERSE: tl.constexpr,  # whether to do autoregressive modeling in the reverse direction
    USE_GK: tl.constexpr,  # whether to use gk
    USE_GV: tl.constexpr,  # whether to use gv
):
    # indices
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    p_q = q + i_bh * s_qk_h + i_k * BK + \
        tl.arange(0, BK) + ((T-1) * DK if REVERSE else 0)
    p_k = k + i_bh * s_qk_h + i_k * BK + \
        tl.arange(0, BK) + ((T-1) * DK if REVERSE else 0)
    p_v = v + i_bh * s_vo_h + i_v * BV + \
        tl.arange(0, BV) + ((T-1) * DV if REVERSE else 0)
    p_o = o + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + \
        tl.arange(0, BV) + ((T-1) * DV if REVERSE else 0)

    if USE_GK:
        p_gk = gk + i_bh * s_qk_h + i_k * BK + \
            tl.arange(0, BK) + ((T-1) * DK if REVERSE else 0)
    if USE_GV:
        p_gv = gv + i_bh * s_vo_h + i_v * BV + \
            tl.arange(0, BV) + ((T-1) * DV if REVERSE else 0)

    mask_bk = (i_k * BK + tl.arange(0, BK)) < DK
    mask_bv = (i_v * BV + tl.arange(0, BV)) < DV

    h = tl.zeros([BV, BK], dtype=tl.float32)

    mask_kv = mask_bk[None, :] & mask_bv[:, None]

    if USE_INITIAL_STATE:
        p_init_s = initial_state + i_bh * DK * DV + \
            (i_k * BK + tl.arange(0, BK)[None, :]) * \
            DV + (i_v * BV + tl.arange(0, BV)[:, None])
        h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)

    for _ in range(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        if USE_GK:
            _gk = tl.load(p_gk, mask=mask_bk, other=0).to(tl.float32)
            h = h * _gk[None, :]
        if USE_GV:
            _gv = tl.load(p_gv, mask=mask_bv, other=0).to(tl.float32)
            h = h * _gv[:, None]
        h += _k[None, :] * _v[:, None]
        _o = h * _q[None, :]
        _o = tl.sum(_o, axis=1)
        tl.store(p_o, _o.to(p_o.dtype.element_ty), mask=mask_bv)
        p_q += -DK if REVERSE else DK
        p_k += -DK if REVERSE else DK
        p_o += -DV if REVERSE else DV
        p_v += -DV if REVERSE else DV
        if USE_GK:
            p_gk += -DK if REVERSE else DK
        if USE_GV:
            p_gv += -DV if REVERSE else DV

    if STORE_FINAL_STATE:
        p_final_s = final_state + i_bh * DK * DV + \
            (i_k * BK + tl.arange(0, BK)[None, :]) * \
            DV + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_final_s, h.to(p_final_s.dtype.element_ty), mask=mask_kv)


# Similar to Algorithm1 of https://arxiv.org/abs/2006.16236
@triton.jit
def fused_recurrent_gla_bwd_kernel(
    # B: batch_size, H: n_heads, T: seq_len, D: d_head
    # NV: number of split in the V dimension. NK: number of split in the K dimension
    q,  # query [B, H, L, D_head_K]
    k,  # key [B, H, L, D_head_V]
    v,  # value [B, H, L, D_head_V]
    gk,  # log gate [B, H, L, D_head_K] \alpha
    gv,  # log gate [B, H, L, D_head_V] \bete

    do,  # gradient of output [B, H, L, D_head_V]
    dq,  # gradient of query [NV, B, H, L, D_head_K]
    dk,  # gradient of key [NV, B, H, L, D_head_K]
    dv,  # gradient of value [NK, B, H, L, D_head_V]

    # initial hidden state initialization [B, H, D_head_K, D_head_V]
    initial_state,

    s_qk_h,  # stride size: L * D_head_K
    s_qk_t,  # stride size: D_head_K
    s_qk_d,  # stride size: 1

    s_vo_h,  # stride size: L * D_head_V
    s_vo_t,  # stride size: D_head_V
    s_vo_d,  # stride size: 1

    B,  # batch_size
    H,  # n_heads
    T,  # seq_len
    scale,  # D_head_K ** -0.5
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    DK: tl.constexpr,  # D_head_K
    DV: tl.constexpr,  # D_head_V
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    REVERSE: tl.constexpr,  # whether to do autoregressive modeling in the reverse direction
    USE_GK: tl.constexpr,  # whether to use gk
    USE_GV: tl.constexpr,  # whether to use gv
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    p_q = q + i_bh * s_qk_h + i_k * BK + \
        tl.arange(0, BK) + ((T-1) * DK if REVERSE else 0)
    p_k = k + i_bh * s_qk_h + i_k * BK + \
        tl.arange(0, BK) + ((T-1) * DK if REVERSE else 0)
    p_v = v + i_bh * s_vo_h + i_v * BV + \
        tl.arange(0, BV) + ((T-1) * DV if REVERSE else 0)
    p_do = do + i_bh * s_vo_h + i_v * BV + \
        tl.arange(0, BV) + ((T-1) * DV if REVERSE else 0)
    p_dq = dq + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + \
        tl.arange(0, BK) + ((T-1) * DK if REVERSE else 0)
    if USE_GK:
        p_gk = gk + i_bh * s_qk_h + i_k * BK + \
            tl.arange(0, BK) + ((T-1) * DK if REVERSE else 0)
    if USE_GV:
        p_gv = gv + i_bh * s_vo_h + i_v * BV + \
            tl.arange(0, BV) + ((T-1) * DV if REVERSE else 0)
    mask_bk = i_k * BK + tl.arange(0, BK) < DK
    mask_bv = i_v * BV + tl.arange(0, BV) < DV
    mask_kv = mask_bk[:, None] & mask_bv[None, :]
    h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_init_s = initial_state + i_bh * DK * DV + \
            (i_k * BK + tl.arange(0, BK)[:, None]) * \
            DV + (i_v * BV + tl.arange(0, BV)[None, :])
        h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)

    for i in range(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        if USE_GK:
            _gk = tl.load(p_gk, mask=mask_bk, other=0).to(tl.float32)
            h = h * _gk[:, None]
        if USE_GV:
            _gv = tl.load(p_gv, mask=mask_bv, other=0).to(tl.float32)
            h = h * _gv[None, :]
        h += _k[:, None] * _v[None, :]
        _d_q = h * _do[None, :]
        d_q = tl.sum(_d_q, axis=1) * scale
        tl.store(p_dq, d_q.to(p_dq.dtype.element_ty), mask=mask_bk)

        p_k += -DK if REVERSE else DK
        p_v += -DV if REVERSE else DV
        p_q += -DK if REVERSE else DK
        p_do += -DV if REVERSE else DV
        p_dq += -DK if REVERSE else DK
        if USE_GK:
            p_gk += -DK if REVERSE else DK
        if USE_GV:
            p_gv += -DV if REVERSE else DV

    # sync threads
    tl.debug_barrier()

    p_q = q + i_bh * s_qk_h + i_k * BK + \
        tl.arange(0, BK) + ((T - 1) * DK if not REVERSE else 0)
    p_k = k + i_bh * s_qk_h + i_k * BK + \
        tl.arange(0, BK) + ((T - 1) * DK if not REVERSE else 0)
    p_do = do + i_bh * s_vo_h + i_v * BV + \
        tl.arange(0, BV) + ((T - 1) * DV if not REVERSE else 0)
    p_v = v + i_bh * s_vo_h + i_v * BV + \
        tl.arange(0, BV) + ((T - 1) * DV if not REVERSE else 0)
    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * \
        BK + tl.arange(0, BK) + ((T - 1) * DK if not REVERSE else 0)
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * \
        BV + tl.arange(0, BV) + ((T - 1) * DV if not REVERSE else 0)
    if USE_GK:
        p_gk = gk + i_bh * s_qk_h + i_k * BK + \
            tl.arange(0, BK) + ((T - 1) * DK if not REVERSE else 0)
    if USE_GV:
        p_gv = gv + i_bh * s_vo_h + i_v * BV + \
            tl.arange(0, BV) + ((T - 1) * DV if not REVERSE else 0)

    d_h = tl.zeros([BK, BV], dtype=tl.float32)

    for _ in range(T):
        _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        d_h += _q[:, None] * _do[None, :]
        d_k = tl.sum(d_h * _v[None, :], axis=1)
        d_v = tl.sum(d_h * _k[:, None], axis=0)
        if USE_GK:
            _gk = tl.load(p_gk, mask=mask_bk, other=0).to(tl.float32)
            d_h *= _gk[:, None]
        if USE_GV:
            _gv = tl.load(p_gv, mask=mask_bv, other=0).to(tl.float32)
            d_h *= _gv[None, :]
        tl.store(p_dk, d_k.to(p_dk.dtype.element_ty), mask=mask_bk)
        tl.store(p_dv, d_v.to(p_dv.dtype.element_ty), mask=mask_bv)

        p_do += DV if REVERSE else -DV
        p_q += DK if REVERSE else -DK
        p_k += DK if REVERSE else -DK
        p_v += DV if REVERSE else -DV
        p_dk += DK if REVERSE else -DK
        p_dv += DV if REVERSE else -DV
        if USE_GK:
            p_gk += DK if REVERSE else -DK
        if USE_GV:
            p_gv += DV if REVERSE else -DV


class FusedRecurrentGLAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @custom_fwd
    def forward(ctx, q, k, v, gk, gv, scale=None, initial_state=None, output_final_state=False, reverse=False):
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]
        # default scale
        if scale is None:
            scale = d_head_qk ** -0.5
        if gk is not None:
            gk = gk.float().exp()
        if gv is not None:
            gv = gv.float().exp()

        BK, BV = min(d_head_qk, 32), min(d_head_v, 32)
        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        num_stages = 1
        num_warps = 1

        o = q.new_empty(NK, batch_size, n_heads, seq_len,
                        d_head_v, dtype=torch.float32)

        if output_final_state:
            final_state = q.new_empty(batch_size, n_heads, d_head_qk, d_head_v)
        else:
            final_state = None

        grid = (NV, NK, batch_size * n_heads)
        fused_recurrent_gla_fwd_kernel[grid](
            q, k, v, gk, gv, o, initial_state, final_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            batch_size, n_heads, seq_len, scale,
            DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=final_state is not None,
            USE_GK=gk is not None,
            USE_GV=gv is not None,
            REVERSE=reverse,
            num_warps=num_warps,
            num_stages=num_stages
        )

        o = o.sum(0)
        ctx.save_for_backward(q, k, v, gk, gv, initial_state, o)
        ctx.scale = scale
        ctx.reverse = reverse
        # we do not need the gradient of the final state from the next chunk
        # similiar to Trunctated BPTT
        if final_state is not None:
            final_state = final_state.detach()
        return o.to(q.dtype), final_state

    @staticmethod
    @contiguous
    @custom_bwd
    def backward(ctx, do, d_final_state=None):
        q, k, v, gk, gv, initial_state, o = ctx.saved_tensors
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]
        scale = ctx.scale

        BK, BV = min(d_head_qk, 32), min(d_head_v, 32)
        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        num_stages = 1
        num_warps = 1

        dq = q.new_empty(NV, batch_size, n_heads,  seq_len,
                         d_head_qk, dtype=torch.float32)
        dk = q.new_empty(NV, batch_size, n_heads,  seq_len,
                         d_head_qk, dtype=torch.float32)
        dv = q.new_empty(NK, batch_size, n_heads, seq_len,
                         d_head_v, dtype=torch.float32)
        grid = (NV, NK, batch_size * n_heads)

        fused_recurrent_gla_bwd_kernel[grid](
            q, k, v, gk, gv, do, dq, dk, dv, initial_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            batch_size, n_heads, seq_len, scale,
            DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
            num_warps=num_warps,
            num_stages=num_stages,
            USE_INITIAL_STATE=initial_state is not None,
            REVERSE=ctx.reverse,
            USE_GK=gk is not None,
            USE_GV=gv is not None
        )
        dq = dq.sum(0)
        dk = dk.sum(0)
        dv = dv.sum(0)
        if gk is not None:
            _dgk = dq * q.float() - dk * k.float()
            if ctx.reverse:
                dgk = _dgk.cumsum(-2)
            else:
                _dgk_cumsum = _dgk.cumsum(-2)
                dgk = _dgk + _dgk_cumsum[:, :, -1, None] - _dgk_cumsum
        else:
            dgk = None

        if gv is not None:
            _dgv = do.float() * o.float() - dv * v.float()
            if ctx.reverse:
                dgv = _dgv.cumsum(-2)
            else:
                _dgv_cumsum = _dgv.cumsum(-2)
                dgv = _dgv + _dgv_cumsum[:, :, -1, None] - _dgv_cumsum
        else:
            dgv = None

        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dgk, dgv, None, None, None, None


# if scale is None, use d_head_qk ** -0.5 by default. Otherwise specify the scale yourself. e.g. scale = 1.0
def fused_recurrent_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor = None,
    gv: torch.Tensor = None,
    scale: int = -1,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    causal: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale == -1:
        scale = q.shape[-1] ** -0.5
    if initial_state is not None:
        initial_state = initial_state.detach()
    if causal:
        o, final_state = FusedRecurrentGLAFunction.apply(q, k, v, gk, gv, scale, initial_state, output_final_state)
        return o, final_state
    else:
        # do not support initial_state yet. looks very strange for bidirectional modeling
        assert initial_state is None
        assert output_final_state is False
        o, final_state = FusedRecurrentGLAFunction.apply(
            q, k, v, gk, gv, scale, initial_state, output_final_state, False)
        o_reversed, final_state = FusedRecurrentGLAFunction.apply(
            q, k, v, gk, gv, scale, initial_state, output_final_state, True)
        return [o, o_reversed]
