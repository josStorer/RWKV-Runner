# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024, Yu Zhang, Songlin Yang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils import (chunk_reversed_cumsum_fwd, softmax_bwd_kernel,
                           softmax_fwd_kernel)
from fla.utils import contiguous


@triton.autotune(
    configs=[
        triton.Config({'BS': 16}, num_warps=2),
        triton.Config({'BS': 16}, num_warps=4),
        triton.Config({'BS': 16}, num_warps=8),
        triton.Config({'BS': 32}, num_warps=2),
        triton.Config({'BS': 32}, num_warps=4),
        triton.Config({'BS': 32}, num_warps=8),
        triton.Config({'BS': 64}, num_warps=2),
        triton.Config({'BS': 64}, num_warps=4),
        triton.Config({'BS': 64}, num_warps=8),
    ],
    key=['S']
)
@triton.jit
def chunk_gated_abc_fwd_kernel_cum(
    s,
    o,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.).to(tl.float32)

    p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_gated_abc_fwd_kernel_h(
    k,
    v,
    g,
    h,
    h0,
    ht,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    s_h_h,
    s_h_t,
    s_h_d,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    GATEK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)
    for i_t in range(NT):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        if GATEK:
            p_g = tl.make_block_ptr(g + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_gn = tl.make_block_ptr(g + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + BT - 1) * K + i_k * BK,), (BK,), (0,))
            # [BK,]
            b_gn = tl.load(p_gn, boundary_check=(0,))
            # [BK, BV]
            b_h *= tl.exp(b_gn)[:, None]
            # [BK, BT]
            b_g = tl.load(p_g, boundary_check=(0, 1))
            b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(b_k.dtype)
        else:
            p_g = tl.make_block_ptr(g + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_gn = tl.make_block_ptr(g + i_bh * s_v_h, (T * V,), (s_v_d,), ((i_t * BT + BT - 1) * V + i_v * BV,), (BV,), (0,))
            # [BV,]
            b_gn = tl.load(p_gn, boundary_check=(0,))
            # [BK, BV]
            b_h *= tl.exp(b_gn)[None, :]
            # [BT, BV]
            b_g = tl.load(p_g, boundary_check=(0, 1))
            b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(b_v.dtype)
        # [BK, BV]
        b_h += tl.dot(b_k, b_v, allow_tf32=False)

    if STORE_FINAL_STATE:
        p_h = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_gated_abc_fwd_kernel_intra_K(
    v,
    g,
    o,
    A,
    s_v_h,
    s_v_t,
    s_v_d,
    T: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_c // NC, i_c % NC

    p_g = tl.make_block_ptr(g + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_gn = tl.make_block_ptr(g + i_bh * s_v_h, (T * V,), (s_v_d,), ((i_t * BT + i_i * BC) * V + i_v * BV,), (BV,), (0,))
    # [BV,]
    b_gn = tl.load(p_gn, boundary_check=(0,))
    # [BC, BV]
    b_o = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(0, i_i):
        p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        p_gv = tl.make_block_ptr(g + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        # [BC, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_gv = tl.load(p_gv, boundary_check=(0, 1))
        b_vg = (b_v * tl.exp(b_gn[None, :] - b_gv)).to(b_v.dtype)
        # [BC, BC]
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_o += tl.dot(b_A, b_vg, allow_tf32=False)
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_o *= tl.exp(b_g - b_gn[None, :])

    o_i = tl.arange(0, BC)
    o_A = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_i * BC
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    for j in range(0, BC):
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T * V,), (1,), ((i_t * BT + i_i * BC + j) * V + i_v * BV,), (BV,), (0,))
        p_gv = tl.make_block_ptr(g + i_bh * s_v_h, (T * V,), (1,), ((i_t * BT + i_i * BC + j) * V + i_v * BV,), (BV,), (0,))
        # [BC,]
        b_A = tl.load(A + o_A + j, mask=m_A, other=0)
        # [BV,]
        b_v = tl.load(p_v, boundary_check=(0,)).to(tl.float32)
        b_gv = tl.load(p_gv, boundary_check=(0,)).to(tl.float32)
        # [BC, BV]
        b_vg = b_v[None, :] * tl.exp(b_g - b_gv[None, :])
        # avoid 0 * inf = inf
        m_i = o_i[:, None] >= j
        b_o += tl.where(m_i, b_A[:, None] * b_vg, 0.)
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))

    b_o += tl.load(p_o, boundary_check=(0, 1))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_gated_abc_fwd_kernel_K(
    q,
    k,
    h,
    g,
    o,
    A,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    s_h_h,
    s_h_t,
    s_h_d,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BV]
        b_o += tl.dot(b_q, b_h, allow_tf32=False)
        # [BT, BT]
        b_A += tl.dot(b_q, b_k, allow_tf32=False)
    p_g = tl.make_block_ptr(g + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    # [BT, BV]
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_o = b_o * tl.exp(b_g)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    # [BT, BT]
    b_A = tl.where(m_s, b_A, 0.)
    if i_v == 0:
        tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_gated_abc_fwd_kernel_intra_V(
    q,
    k,
    g,
    A,
    s_k_h,
    s_k_t,
    s_k_d,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i, i_j = i_c // (NC * NC), (i_c % (NC * NC)) // NC, (i_c % (NC * NC)) % NC
    n_bh = tl.num_programs(2)

    if i_i > i_j:
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gn = tl.make_block_ptr(g + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_i * BC) * K + i_k * BK,), (BK,), (0,))
        p_A = tl.make_block_ptr(A + (i_k*n_bh+i_bh)*T*BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        # [BK,]
        b_gn = tl.load(p_gn, boundary_check=(0,))
        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = (b_q * tl.exp(b_g - b_gn[None, :]) * scale).to(b_q.dtype)
        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = (b_k * tl.exp(b_gn[:, None] - b_gk)).to(b_k.dtype)
        # [BC, BC]
        b_A = tl.dot(b_qg, b_kg, allow_tf32=False)
        tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))
    elif i_i == i_j:
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_j * BC) * K + i_k * BK,), (BK,), (0,))
        p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_j * BC) * K + i_k * BK,), (BK,), (0,))
        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))

        o_i = tl.arange(0, BC)
        o_A = (i_bh + i_k * n_bh) * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_j * BC
        m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
        for j in range(0, BC):
            # [BK,]
            b_k = tl.load(p_k, boundary_check=(0,)).to(tl.float32)
            b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
            # [BC,]
            b_A = tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]) * scale, 1)
            b_A = tl.where(o_i >= j, b_A, 0.)
            tl.store(A + o_A + j, b_A.to(b_q.dtype), mask=m_A)

            p_k = tl.advance(p_k, (K,))
            p_gk = tl.advance(p_gk, (K,))


@triton.jit
def chunk_gated_abc_fwd_kernel_V(
    q,
    v,
    g,
    h,
    o,
    A,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    s_h_h,
    s_h_t,
    s_h_d,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BK]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        # [BT, BK]
        b_qg = (b_q * tl.exp(b_g)).to(b_q.dtype)
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # works but dkw, owing to divine benevolence
        # [BT, BV]
        if i_k >= 0:
            b_o += tl.dot(b_qg, b_h, allow_tf32=False)
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    # [BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_o += tl.dot(b_A, b_v, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_gated_abc_bwd_kernel_dh(
    q,
    g,
    do,
    dh,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    s_h_h,
    s_h_t,
    s_h_d,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    GATEK: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t in range(NT - 1, -1, -1):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K*V, (K, V), (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        # [BK, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))

        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        if GATEK:
            p_g = tl.make_block_ptr(g + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_gn = tl.make_block_ptr(g + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + BT - 1) * K + i_k * BK,), (BK,), (0,))
            # [BK,]
            b_gn = tl.load(p_gn, boundary_check=(0,))
            # [BK, BV]
            b_dh *= tl.exp(b_gn)[:, None]
            # [BK, BT]
            b_g = tl.load(p_g, boundary_check=(0, 1))
            b_q = (b_q * tl.exp(b_g)).to(b_q.dtype)
        else:
            p_g = tl.make_block_ptr(g + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_gn = tl.make_block_ptr(g + i_bh * s_v_h, (T * V,), (s_v_d,), ((i_t * BT + BT - 1) * V + i_v * BV,), (BV,), (0,))
            # [BV,]
            b_gn = tl.load(p_gn, boundary_check=(0,))
            # [BK, BV]
            b_dh *= tl.exp(b_gn)[None, :]
            # [BT, BV]
            b_g = tl.load(p_g, boundary_check=(0, 1))
            b_do = (b_do * tl.exp(b_g)).to(b_do.dtype)
        # [BK, BV]
        b_dh += tl.dot(b_q, b_do, allow_tf32=False)


@triton.jit
def chunk_gated_abc_bwd_kernel_V(
    k,
    v,
    h,
    g,
    A,
    do,
    dh,
    dq,
    dk,
    dv,
    dA,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    s_h_h,
    s_h_t,
    s_h_d,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)

    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_gn = tl.make_block_ptr(g + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + BT - 1) * K + i_k * BK,), (BK,), (0,))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (BT, T), (1, BT), (0, i_t * BT), (BT, BT), (0, 1))

    # [BK,]
    # [BT, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_gn = tl.exp(tl.load(p_gn, boundary_check=(0,))[None, :] - b_gk)
    b_k = (b_k * b_gn).to(b_k.dtype)
    # [BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1))

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * V * K, (V, K), (s_h_d, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K*V, (K, V), (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k*n_bh+i_bh) * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BK, BV]
        b_dh = tl.load(p_dh, boundary_check=(0, 1))

        # [BT, BV]
        b_dv = tl.dot(b_k, b_dh, allow_tf32=False)
        if i_k == 0:
            b_dv += tl.dot(b_A, b_do, allow_tf32=False)
        b_do = (b_do * scale).to(b_do.dtype)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
        # [BT, BT]
        b_dA += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
        # [BT, BK]
        b_dq += tl.dot(b_do, b_h, allow_tf32=False)
        # [BT, BK]
        b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
    b_dq = b_dq * tl.exp(b_gk)
    b_dk = b_dk * b_gn

    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT, ), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    # [BT, BT]
    b_dA = tl.where(m_s, b_dA, 0.).to(b_k.dtype)
    if i_k == 0:
        tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_gated_abc_bwd_kernel_intra_V(
    q,
    k,
    g,
    dA,
    dq,
    dk,
    dg,
    s_k_h,
    s_k_t,
    s_k_d,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    OVERWRITE: tl.constexpr
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_c // NC, i_c % NC

    p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_gn = tl.make_block_ptr(g + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_i * BC) * K + i_k * BK,), (BK,), (0,))
    # [BK,]
    b_gn = tl.load(p_gn, boundary_check=(0,))
    # [BC, BK]
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    for i_j in range(0, i_i):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
        p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
        p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        # [BC, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = (b_k * tl.exp(b_gn[None, :] - b_gk)).to(b_k.dtype)
        # [BC, BC]
        b_dA = tl.load(p_dA, boundary_check=(0, 1))
        # [BC, BK]
        b_dq += tl.dot(b_dA, b_kg, allow_tf32=False)
    b_dq *= tl.exp(b_g - b_gn[None, :])

    o_i = tl.arange(0, BC)
    o_dA = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_i * BC
    m_dA = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    for j in range(0, BC):
        p_kj = tl.make_block_ptr(k + i_bh * s_k_h, (T * K,), (1,), ((i_t * BT + i_i*BC+j) * K + i_k * BK,), (BK,), (0,))
        p_gkj = tl.make_block_ptr(g + i_bh * s_k_h, (T * K,), (1,), ((i_t * BT + i_i*BC+j) * K + i_k * BK,), (BK,), (0,))
        # [BC,]
        b_dA = tl.load(dA + o_dA + j, mask=m_dA, other=0)
        # [BK,]
        b_kj = tl.load(p_kj, boundary_check=(0,)).to(tl.float32)
        b_gkj = tl.load(p_gkj, boundary_check=(0,)).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] >= j
        # [BC, BK]
        b_dq += tl.where(m_i, b_dA[:, None] * b_kj[None, :] * tl.exp(b_g - b_gkj[None, :]), 0.)
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))

    b_dq = b_dq + tl.load(p_dq, boundary_check=(0, 1))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_gn = tl.make_block_ptr(g + i_bh * s_k_h, (T*K,), (s_k_d,), ((i_t * BT + i_i * BC + BC - 1) * K + i_k * BK,), (BK,), (0,))
    # [BK,]
    b_gn = tl.load(p_gn, boundary_check=(0,))
    # [BC, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
        p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_j * BC, i_i * BC), (BC, BC), (1, 0))
        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = (b_q * tl.exp(b_g - b_gn[None, :])).to(b_q.dtype)
        # [BC, BC]
        b_dA = tl.load(p_dA, boundary_check=(0, 1))
        # [BC, BK]
        b_dk += tl.dot(tl.trans(b_dA), b_qg, allow_tf32=False)
    b_dk *= tl.exp(b_gn[None, :] - b_gk)

    o_dA = i_bh * T * BT + (i_t * BT + i_i * BC) * BT + i_i * BC + tl.arange(0, BC)
    for j in range(0, BC):
        p_qj = tl.make_block_ptr(q + i_bh * s_k_h, (T * K,), (1,), ((i_t * BT + i_i * BC + j) * K + i_k * BK,), (BK,), (0,))
        p_gqj = tl.make_block_ptr(g + i_bh * s_k_h, (T * K,), (1,), ((i_t * BT + i_i * BC + j) * K + i_k * BK,), (BK,), (0,))
        # [BC,]
        b_dA = tl.load(dA + o_dA + j * BT, mask=(i_t * BT + i_i * BC + j < T), other=0)
        # [BK,]
        b_qj = tl.load(p_qj, boundary_check=(0,)).to(tl.float32)
        b_gqj = tl.load(p_gqj, boundary_check=(0,)).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] <= j
        b_dk += tl.where(m_i, b_dA[:, None] * b_qj[None, :] * tl.exp(b_gqj[None, :] - b_gk), 0.)
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
    b_dk = b_dk + tl.load(p_dk, boundary_check=(0, 1)).to(tl.float32)
    b_dg = b_q * b_dq - b_k * b_dk
    if not OVERWRITE:
        b_dg = b_dg + tl.load(p_dg, boundary_check=(0, 1)).to(tl.float32)

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_gated_abc_bwd_kernel_intra_K(
    v,
    g,
    do,
    dA,
    s_v_h,
    s_v_t,
    s_v_d,
    scale,
    T: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i, i_j = i_c // (NC * NC), (i_c % (NC * NC)) // NC, (i_c % (NC * NC)) % NC
    n_bh = tl.num_programs(2)

    if i_i > i_j:
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (V, T), (s_v_d, s_v_t), (i_v * BV, i_t * BT + i_j * BC), (BV, BC), (0, 1))
        p_gv = tl.make_block_ptr(g + i_bh * s_v_h, (V, T), (s_v_d, s_v_t), (i_v * BV, i_t * BT + i_j * BC), (BV, BC), (0, 1))
        p_gn = tl.make_block_ptr(g + i_bh * s_v_h, (T * V,), (s_v_d,), ((i_t * BT + i_i * BC) * V + i_v * BV,), (BV,), (0,))
        p_g = tl.make_block_ptr(g + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_dA = tl.make_block_ptr(dA+(i_bh+i_v*n_bh)*T*BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        # [BV,]
        b_gn = tl.load(p_gn, boundary_check=(0,))
        # [BC, BV]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = (b_do * tl.exp(b_g - b_gn[None, :]) * scale).to(b_do.dtype)
        # [BV, BC]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_gv = tl.load(p_gv, boundary_check=(0, 1))
        b_vg = (b_v * tl.exp(b_gn[:, None] - b_gv)).to(b_v.dtype)
        # [BC, BC]
        b_dA = tl.dot(b_do, b_vg, allow_tf32=False)
        tl.store(p_dA, b_dA.to(dA.dtype.element_ty), boundary_check=(0, 1))
    elif i_i == i_j:
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T * V,), (s_v_d,), ((i_t * BT + i_j * BC) * V + i_v * BV,), (BV,), (0,))
        p_gv = tl.make_block_ptr(g + i_bh * s_v_h, (T * V,), (s_v_d,), ((i_t * BT + i_j * BC) * V + i_v * BV,), (BV,), (0,))
        p_g = tl.make_block_ptr(g + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        # [BC, BV]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1)) * scale

        o_i = tl.arange(0, BC)
        o_A = (i_bh + i_v * n_bh) * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_j * BC
        m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
        for j in range(0, BC):
            # [BV,]
            b_v = tl.load(p_v, boundary_check=(0,)).to(tl.float32)
            b_gv = tl.load(p_gv, boundary_check=(0,)).to(tl.float32)
            # [BC,]
            b_dA = tl.sum(b_do * b_v[None, :] * tl.exp(b_g - b_gv[None, :]), 1)
            b_dA = tl.where(o_i >= j, b_dA, 0)
            tl.store(dA + o_A + j, b_dA.to(b_do.dtype), mask=m_A)

            p_v = tl.advance(p_v, (V,))
            p_gv = tl.advance(p_gv, (V,))


@triton.jit
def chunk_gated_abc_bwd_kernel_K(
    q,
    k,
    v,
    h,
    g,
    A,
    do,
    dh,
    dq,
    dk,
    dv,
    dA,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    s_h_h,
    s_h_t,
    s_h_d,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_A = tl.make_block_ptr(A + (i_k*n_bh+i_bh) * T * BT, (T, BT, ), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))

    # [BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    # [BT, BT]
    b_A = tl.dot((b_q * scale).to(b_q.dtype), tl.trans(b_k), allow_tf32=False)
    b_A = tl.where(m_s, b_A, 0.)
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K*V, (V, K), (s_h_d, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_g = tl.make_block_ptr(g + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_gn = tl.make_block_ptr(g + i_bh * s_v_h, (T * V,), (s_v_d,), ((i_t * BT + BT - 1) * V + i_v * BV,), (BV,), (0,))

        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K*V, (K, V), (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k*n_bh+i_bh) * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        # [BV,]
        b_gn = tl.load(p_gn, boundary_check=(0,))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_v = b_v * tl.exp(b_gn[None, :] - b_g)
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = (b_do * tl.exp(b_g) * scale).to(b_do.dtype)
        # [BK, BV]
        b_dh = tl.load(p_dh, boundary_check=(0, 1))

        # [BT, BK]
        b_dq += tl.dot(b_do, b_h, allow_tf32=False)
        b_dk += tl.dot(b_v.to(b_dh.dtype), tl.trans(b_dh), allow_tf32=False)
        # [BT, BV]
        b_dv = tl.exp(b_gn[None, :] - b_g) * tl.dot(b_k, b_dh, allow_tf32=False)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT, ), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    # [BT, BT]
    b_dA = tl.load(p_dA, boundary_check=(0, 1))
    # [BT, BK]
    b_dq += tl.dot(b_dA, b_k, allow_tf32=False)
    b_dk += tl.dot(tl.trans(b_dA).to(b_k.dtype), b_q, allow_tf32=False)

    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_gated_abc_bwd_kernel_intra_KV(
    v,
    g,
    o,
    A,
    do,
    dv,
    dg,
    s_v_h,
    s_v_t,
    s_v_d,
    T: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
    OVERWRITE: tl.constexpr
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_c // NC, i_c % NC

    p_gv = tl.make_block_ptr(g + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_gn = tl.make_block_ptr(g + i_bh * s_v_h, (T*V,), (s_v_d,), ((i_t * BT + i_i * BC + BC - 1) * V + i_v * BV,), (BV,), (0,))
    # [BV,]
    b_gn = tl.load(p_gn, boundary_check=(0,))
    # [BC, BV]
    b_gv = tl.load(p_gv, boundary_check=(0, 1))
    b_dv = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        p_g = tl.make_block_ptr(g + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_j * BC, i_v * BV),  (BC, BV), (1, 0))
        p_A = tl.make_block_ptr(A + i_bh * T * BT, (BT, T), (1, BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        # [BC, BV]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = (b_do * tl.exp(b_g - b_gn[None, :])).to(b_do.dtype)
        # [BC, BC]
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_dv += tl.dot(b_A, b_do, allow_tf32=False)
    b_dv *= tl.exp(b_gn[None, :] - b_gv)

    o_i = tl.arange(0, BC)
    for j in range(0, BC):
        p_g = tl.make_block_ptr(g + i_bh * s_v_h, (T * V,), (1,), ((i_t * BT + i_i * BC + j) * V + i_v * BV,), (BV,), (0,))
        p_A = tl.make_block_ptr(A + i_bh * T * BT, (T * BT,), (1,), ((i_t * BT + i_i * BC + j) * BT + i_i * BC,), (BC,), (0,))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T * V,), (1,), ((i_t * BT + i_i * BC + j) * V + i_v * BV,), (BV,), (0,))
        # [BC,]
        b_A = tl.load(p_A, boundary_check=(0,))
        # [BV,]
        b_g = tl.load(p_g, boundary_check=(0,))
        b_do = tl.load(p_do, boundary_check=(0,))
        # [BC, BV]
        m_i = o_i[:, None] <= j
        b_dv += tl.where(m_i, tl.exp(b_g[None, :] - b_gv) * b_A[:, None] * b_do[None, :], 0.)
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_dv = tl.make_block_ptr(dv + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_dg = tl.make_block_ptr(dg + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))

    b_o = tl.load(p_o, boundary_check=(0, 1)).to(tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
    b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
    b_dv = b_dv + tl.load(p_dv, boundary_check=(0, 1)).to(tl.float32)
    b_dg = b_o * b_do - b_v * b_dv
    if not OVERWRITE:
        b_dg = b_dg + tl.load(p_dg, boundary_check=(0, 1)).to(tl.float32)
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


def fwd_pre(g, B, H, T, S, BT):
    NT = triton.cdiv(T, BT)
    g_org, g = g, torch.empty_like(g, dtype=torch.float)
    def grid(meta): return (triton.cdiv(meta['S'], meta['BS']), NT, B * H)
    # keep cummulative normalizer in fp32
    # this kernel is equivalent to
    # g = g.view(B, H, NT, BT, -1).cumsum(-2).view(B, H, T, -1)
    chunk_gated_abc_fwd_kernel_cum[grid](
        g_org, g,
        g.stride(1), g.stride(2), g.stride(3),
        T=T, S=S, BT=BT
    )
    return g


def fwd_inner(q, k, v, g, B, H, T, K, V, BT, BK, BV, gatek=False, h0=None, ht=None):
    NT = triton.cdiv(T, BT)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    num_warps = 4 if BK == 64 else 2
    num_stages = 1

    h = q.new_empty(B, H, NT * K, V)
    grid = (NV, NK, B * H)
    chunk_gated_abc_fwd_kernel_h[grid](
        k, v, g, h, h0, ht,
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        h.stride(1), h.stride(2), h.stride(3),
        T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
        GATEK=gatek,
        USE_INITIAL_STATE=h0 is not None,
        STORE_FINAL_STATE=ht is not None,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return h


def fwd_v(q, k, v, g, B, H, T, K, V, BT, BK, BV, BC, h0=None, ht=None, scale=1.):
    NT = triton.cdiv(T, BT)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    NC = triton.cdiv(BT, BC)
    num_warps = 4 if BK == 64 else 2
    num_stages = 1

    h = fwd_inner(
        q=q, k=k, v=v, g=g,
        B=B, H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV,
        gatek=True,
        h0=h0,
        ht=ht
    )
    A = q.new_zeros(NK, B, H, T, BT)
    grid = (NK, NT * NC * NC, B * H)
    chunk_gated_abc_fwd_kernel_intra_V[grid](
        q, k, g, A,
        k.stride(1), k.stride(2), k.stride(3),
        scale,
        T=T, K=K, BT=BT, BC=BC, BK=BK, NC=NC,
        num_warps=2,
        num_stages=num_stages
    )
    A = A.sum(0, dtype=A.dtype)
    o = torch.empty_like(v)
    grid = (NV, NT, B * H)
    chunk_gated_abc_fwd_kernel_V[grid](
        q, v, g, h, o, A,
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        h.stride(1), h.stride(2), h.stride(3),
        scale,
        T=T, K=K, V=V, BT=BT, BK=BK, BV=BV,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return o, h, A


def fwd_k(q, k, v, g, B, H, T, K, V, BT, BK, BV, BC, h0=None, ht=None, scale=1.):
    NT = triton.cdiv(T, BT)
    NV = triton.cdiv(V, BV)
    NC = triton.cdiv(BT, BC)
    num_warps = 4 if BK == 64 else 2
    num_stages = 1

    h = fwd_inner(
        q=q, k=k, v=v, g=g,
        B=B, H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV,
        gatek=False,
        h0=h0,
        ht=ht
    )
    o = torch.empty_like(v)
    A = q.new_empty(B, H, T, BT)
    grid = (NV, NT, B * H)
    chunk_gated_abc_fwd_kernel_K[grid](
        q, k, h, g, o, A,
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        h.stride(1), h.stride(2), h.stride(3),
        scale,
        T=T, K=K, V=V, BT=BT, BK=BK, BV=BV,
        num_warps=num_warps,
        num_stages=num_stages
    )
    grid = (NV, NT * NC, B * H)
    chunk_gated_abc_fwd_kernel_intra_K[grid](
        v, g, o, A,
        v.stride(1), v.stride(2), v.stride(3),
        T=T, V=V, BT=BT, BC=BC, BV=BV, NC=NC,
        num_warps=2,
        num_stages=num_stages
    )
    return o, h, A


def bwd_inner(q, g, do, B, H, T, K, V, BT, BK, BV, scale, gatek=False):
    NT = triton.cdiv(T, BT)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    num_warps = 4 if BK == 64 else 2
    num_stages = 1

    dh = q.new_empty(B, H, NT * K, V)
    grid = (NK, NV, B * H)
    chunk_gated_abc_bwd_kernel_dh[grid](
        q, g, do, dh,
        q.stride(1), q.stride(2), q.stride(3),
        do.stride(1), do.stride(2), do.stride(3),
        dh.stride(1), dh.stride(2), dh.stride(3),
        scale,
        T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
        GATEK=gatek,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return dh


def bwd_v(q, k, v, g, h, A, do, dg, B, H, T, K, V, BT, BK, BV, BC, scale=1.):
    NT = triton.cdiv(T, BT)
    NK = triton.cdiv(K, BK)
    NC = triton.cdiv(BT, BC)
    num_warps = 4 if BK == 64 else 2
    num_stages = 1

    overwrite_dg = dg is None
    dh = bwd_inner(
        q, g, do,
        B=B, H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV,
        scale=scale,
        gatek=True
    )
    dq = torch.empty_like(q, dtype=torch.float)
    dk = torch.empty_like(k, dtype=torch.float)
    dv = v.new_empty(NK, *v.shape)
    dg = torch.empty_like(g, dtype=torch.float) if dg is None else dg
    dA = v.new_zeros(B, H, T, BT)

    grid = (NK, NT, B * H)
    chunk_gated_abc_bwd_kernel_V[grid](
        k, v, h, g, A, do, dh, dq, dk, dv, dA,
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        h.stride(1), h.stride(2), h.stride(3),
        scale,
        T=T, K=K, V=V, BT=BT, BK=BK, BV=BV,
        num_warps=num_warps,
        num_stages=num_stages
    )
    dv = dv.sum(0, dtype=dv.dtype)
    grid = (NK, NT * NC, B * H)
    chunk_gated_abc_bwd_kernel_intra_V[grid](
        q, k, g, dA, dq, dk, dg,
        k.stride(1), k.stride(2), k.stride(3),
        T=T, K=K, BT=BT, BC=BC, BK=BK, NC=NC,
        OVERWRITE=overwrite_dg,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return dq, dk, dv, dg


def bwd_k(q, k, v, g, h, o, do, dg, B, H, T, K, V, BT, BK, BV, BC, scale=1.):
    NT = triton.cdiv(T, BT)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    NC = triton.cdiv(BT, BC)
    num_warps = 4 if BK == 64 else 2
    num_stages = 1

    overwrite_dg = dg is None
    dh = bwd_inner(
        q, g, do,
        B=B, H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV,
        scale=scale,
        gatek=False
    )
    dA = q.new_zeros(NV, B, H, T, BT)
    grid = (NV, NT * NC * NC, B * H)
    chunk_gated_abc_bwd_kernel_intra_K[grid](
        v, g, do, dA,
        v.stride(1), v.stride(2), v.stride(3),
        scale,
        T=T, V=V, BT=BT, BC=BC, BV=BV, NC=NC,
        num_warps=num_warps,
        num_stages=num_stages
    )
    dA = dA.sum(0, dtype=dA.dtype)

    A = do.new_zeros(NK, B, H, T, BT)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = v.new_empty(NK, *v.shape)
    dg = torch.empty_like(g, dtype=torch.float) if dg is None else dg
    grid = (NK, NT, B * H)
    chunk_gated_abc_bwd_kernel_K[grid](
        q, k, v, h, g, A, do, dh, dq, dk, dv, dA,
        q.stride(1), q.stride(2), q.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        h.stride(1), h.stride(2), h.stride(3),
        scale,
        T=T, K=K, V=V, BT=BT, BK=BK, BV=BV,
        num_warps=num_warps,
        num_stages=num_stages
    )
    A = A.sum(0, dtype=A.dtype)
    dv = dv.sum(0, dtype=dv.dtype)
    grid = (NV, NT * NC, B * H)
    chunk_gated_abc_bwd_kernel_intra_KV[grid](
        v, g, o, A, do, dv, dg,
        v.stride(1), v.stride(2), v.stride(3),
        T=T, V=V, BT=BT, BC=BC, BV=BV, NC=NC,
        OVERWRITE=overwrite_dg,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return dq, dk, dv, dg


class ChunkGatedABCFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, s, g, scale, initial_state, output_final_state, checkpoint_level):
        B, H, T, K, V, M = *q.shape, v.shape[-1], s.shape[-1]
        BT, BC = 64, 16
        BK = min(64, triton.next_power_of_2(K))
        BV = min(64, triton.next_power_of_2(V))
        BM = min(64, triton.next_power_of_2(M))

        final_state = None
        if output_final_state:
            final_state = (q.new_empty(B, H, K, M, dtype=torch.float),
                           q.new_empty(B, H, M, V, dtype=torch.float))

        g_org, g = g, fwd_pre(g, B, H, T, M, BT)
        ok, hk, _ = fwd_k(
            q=q, k=k, v=s, g=g,
            B=B, H=H, T=T, K=K, V=M, BT=BT, BK=BK, BV=BM, BC=BC,
            h0=initial_state[0] if initial_state is not None else None,
            ht=final_state[0] if final_state is not None else None,
            scale=scale
        )

        # equivalent to:
        # p = ok.softmax(-1, torch.float)
        # p is kept in fp32 for safe softmax backward
        p = torch.empty_like(ok, dtype=torch.float)
        def grid(meta): return (triton.cdiv(meta['T'], meta['BT']), B * H)
        softmax_fwd_kernel[grid](
            ok, p,
            s.stride(1), s.stride(2), s.stride(3),
            T=T, S=M, BT=BT
        )

        ov, hv, Av = fwd_v(
            q=p.to(q.dtype), k=s, v=v, g=g,
            B=B, H=H, T=T, K=M, V=V, BT=BT, BK=BM, BV=BV, BC=BC,
            h0=initial_state[1] if initial_state is not None else None,
            ht=final_state[1] if final_state is not None else None,
            scale=1.
        )

        if checkpoint_level >= 1:
            del g
            g = g_org
        if checkpoint_level > 1:
            del hk
            del hv
            hk, hv = None, None
            initial_state = tuple() if initial_state is None else initial_state
        else:
            initial_state = tuple()

        ctx.save_for_backward(q, k, v, s, g, ok, p, hk, hv, Av, *initial_state)
        ctx.checkpoint_level = checkpoint_level
        ctx.scale = scale
        ctx.BT = BT
        return ov, final_state

    @staticmethod
    @contiguous
    def backward(ctx, dov, dht=None):
        q, k, v, s, g, ok, p, hk, hv, Av, *initial_state = ctx.saved_tensors
        qv = p.to(q.dtype)
        B, H, T, K, V, M = *q.shape, v.shape[-1], s.shape[-1]
        BT, BC = ctx.BT, 16
        BK = min(64, triton.next_power_of_2(K))
        BV = min(64, triton.next_power_of_2(V))
        BM = min(64, triton.next_power_of_2(M))

        if ctx.checkpoint_level >= 1:
            g = fwd_pre(g, B, H, T, M, BT)

        # rerun the forward pass to get h if checkpoint_level >= 1
        if ctx.checkpoint_level > 1:
            hk = fwd_inner(
                q=q, k=k, v=s, g=g,
                B=B, H=H, T=T, K=K, V=M, BT=BT, BK=BK, BV=BM,
                gatek=False,
                h0=initial_state[0] if len(initial_state) > 0 else None,
                ht=None
            )
            hv = fwd_inner(
                q=qv, k=s, v=v, g=g,
                B=B, H=H, T=T, K=M, V=V, BT=BT, BK=BM, BV=BV,
                gatek=True,
                h0=initial_state[1] if len(initial_state) > 0 else None,
                ht=None
            )

        dqv, dsv, dv, dg = bwd_v(
            q=qv, k=s, v=v, g=g, h=hv, A=Av, do=dov, dg=None,
            B=B, H=H, T=T, K=M, V=V, BT=BT, BK=BM, BV=BV, BC=BC,
            scale=1.
        )

        # softmax gradient, equivalent to:
        # dok = qv * (dqv - (qv * dqv).sum(-1, True))
        dok = torch.empty_like(ok)
        def grid(meta): return (triton.cdiv(meta['T'], meta['BT']), B * H)
        softmax_bwd_kernel[grid](
            p, dqv, dok,
            s.stride(1), s.stride(2), s.stride(3),
            T=T, S=M, BT=BT
        )

        dq, dk, dsk, dg = bwd_k(
            q=q, k=k, v=s, g=g, h=hk, o=ok, do=dok, dg=dg,
            B=B, H=H, T=T, K=K, V=M, BT=BT, BK=BK, BV=BM, BC=BC,
            scale=ctx.scale
        )

        ds = dsv.add_(dsk)
        # reversed cumsum, equivalent to:
        #
        # def reversed_cumsum(x, dim=-1):
        #     c = x.cumsum(dim)
        #     return x + c.index_select(dim, x.new_tensor([c.shape[dim]-1], dtype=torch.long)) - c
        dg = chunk_reversed_cumsum_fwd(dg).to(s.dtype)
        return dq, dk, dv, ds, dg, None, None, None, None


def chunk_gated_abc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[int] = None,
    initial_state: Optional[Tuple[torch.Tensor]] = None,
    output_final_state: Optional[bool] = False,
    checkpoint_level: Optional[int] = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `(B, H, T, K)`
        k (torch.Tensor):
            keys of shape `(B, H, T, K)`
        v (torch.Tensor):
            values of shape `(B, H, T, V)`
        g (torch.Tensor):
            Forget gates of shape `(B, H, T, M)` applied to keys.
            If not provided, this function is equivalent to vanilla ABC.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[Tuple[torch.Tensor]]):
            Initial state tuple having tensors of shape `(B, H, K, V)`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state tuple, having tensors of shape `(B, H, K, V)`. Default: `False`.
        checkpoint_level (Optional[int]):
            Checkpointing level; higher values will save more memories and do more recomputations during backward.
            Default: `2`:
            - Level `0`: no memory saved, no recomputation.
            - Level `1`: recompute the fp32 cumulative values during backward.
            - Level `2`: recompute the fp32 cumulative values and forward hidden states during backward.
    """
    assert checkpoint_level in [0, 1, 2]
    if initial_state is not None:
        initial_state = tuple(i.detach() for i in initial_state)
    if g is None:
        # TODO: this 3 steps took huge amount of time, ought to be optimized
        z = s.float().logcumsumexp(2)
        g = torch.cat((z[:, :, :1], z[:, :, :-1]), 2) - z
        s = torch.exp(s - z).to(k.dtype)
    if scale is None:
        scale = q.shape[-1] ** -0.5
    ov, final_state = ChunkGatedABCFunction.apply(q, k, v, s, g, scale, initial_state, output_final_state, checkpoint_level)
    return ov, final_state
