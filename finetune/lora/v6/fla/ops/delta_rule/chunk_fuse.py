# -*- coding: utf-8 -*-

from typing import Tuple

import torch
import triton
import triton.language as tl
from packaging import version
from torch.cuda.amp import custom_bwd, custom_fwd

from fla.ops.delta_rule.utils import bwd_prepare_wy_repr, fwd_prepare_wy_repr
from fla.utils import contiguous


# on-the-fly computation without materializing hidden statets into HBMs
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8)
    ],
    key=["BT", "BK"],
)
@triton.jit
def fused_chunk_delta_rule_fwd_kernel(
    # B: batch_size, H: n_heads, T: seq_len, D: d_head
    q,  # query [B, H, L, D_head_K]
    k,  # key [B, H, L, D_head_K]
    v,  # value [B, H, L, D_head_V]
    v_new,
    d,  # decay [B, H, L, D_head_K]
    o,  # output [B, H, L, D_head_V]
    initial_state,  # initial state of the chunk [B, H, D_head_K, D_head_V]
    final_state,  # final state of the chunk [B, H, D_head_K, D_head_V]
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
    BT: tl.constexpr,  # BLOCK SIZE along the sequence dimension, a.k.a. chunk size
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    DK: tl.constexpr,  # D_head_K
    DV: tl.constexpr,  # D_head_V
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    CHECK: tl.constexpr
):
    # indices
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    o_i = tl.arange(0, BT)

    # [BT, BT]
    m_s = o_i[:, None] >= o_i[None, :]
    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    # make block pointers
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BT), (0, 1))
    p_d = tl.make_block_ptr(d + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh+i_k*B*H) * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_v_new = tl.make_block_ptr(v_new + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))

    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)

    for i in range(0, tl.cdiv(T, BT)):
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_d = tl.load(p_d, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_k.dtype)

        # [BT, BT]
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        # [BT, BV]
        b_v_prime = tl.dot(b_d, b_h.to(b_q.dtype), allow_tf32=False)
        b_v = b_v - b_v_prime
        tl.store(p_v_new, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        b_o = tl.dot(b_s.to(b_q.dtype), b_v.to(b_q.dtype), allow_tf32=False)
        if CHECK and i == 0:
            b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False)
            b_h = b_h + tl.dot(b_k, b_v.to(b_k.dtype), allow_tf32=False)
        else:
            b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False)
            b_h = b_h + tl.dot(b_k, b_v.to(b_k.dtype), allow_tf32=False)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_v_new = tl.advance(p_v_new, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))
        p_d = tl.advance(p_d, (BT, 0))

    if STORE_FINAL_STATE:
        p_final = tl.make_block_ptr(final_state + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_final, b_h.to(p_final.dtype.element_ty), boundary_check=(0, 1))


# Similar to Algorithm1 of https://arxiv.org/abs/2006.16236
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def fused_chunk_delta_rule_bwd_kernel(
    # B: batch_size, H: n_heads, T: seq_len, D: d_head
    # NV: number of split in the V dimension. NK: number of split in the K dimension
    q,  # query [B, H, L, D_head_K]
    k,  # key [B, H, L, D_head_V]
    v,  # value [B, H, L, D_head_V]
    d,  # decay [B, H, L, D_head_K]
    do,  # gradient of output [B, H, L, D_head_V]
    dq,  # gradient of query [NV, B, H, L, D_head_K]
    dk,  # gradient of key [NV, B, H, L, D_head_K]
    dv,  # gradient of value [NK, B, H, L, D_head_V]
    dd,  # gradient of decay [NV, B, H, L, D_head_K]
    initial_state,  # initial state of the chunk [B, H, D_head_K, D_head_V]
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
    BT: tl.constexpr,  # BLOCK SIZE along the sequence dimension, a.k.a. chunk size
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    DK: tl.constexpr,  # D_head_K
    DV: tl.constexpr,  # D_head_V
    USE_INITIAL_STATE: tl.constexpr,
    CHECK: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)

    # first reverse
    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    m_s = o_i[:, None] <= o_i[None, :]
    for i in range(1, tl.cdiv(T, BT) + 1):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, T - i * BT), (BK, BT), (0, 1))
        p_d = tl.make_block_ptr(d + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, T - i * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (T - i * BT, i_k * BK), (BT, BK), (1, 0))

        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_bh+i_v*B*H) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (T - i*BT, i_k*BK), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_bh+i_k*B*H) * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (T - i*BT, i_v*BV), (BT, BV), (1, 0))
        # [DK, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, DK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, DV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))

        # [BT, BT]
        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False)
        b_ds = tl.where(m_s, b_ds, 0).to(b_q.dtype)
        # [BT, BT]
        b_s = tl.dot(b_k, b_q, allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0).to(b_q.dtype)
        # [BT, DK]
        b_dk = tl.dot(b_ds, tl.trans(b_q), allow_tf32=False)
        # [BT, DV]
        b_dv = tl.dot(b_s, b_do, allow_tf32=False)
        b_d = tl.load(p_d, boundary_check=(0, 1))
        if CHECK and i == 1:
            b_dk += tl.dot(b_v, tl.trans(b_dh).to(b_v.dtype),  allow_tf32=False)
            b_dv += tl.dot(b_k, b_dh.to(b_k.dtype), allow_tf32=False)
            b_dh += tl.dot(b_q, b_do, allow_tf32=False)
            b_dh -= tl.dot(b_d, b_dv.to(b_d.dtype), allow_tf32=False)
        else:
            b_dk += tl.dot(b_v, tl.trans(b_dh).to(b_v.dtype),  allow_tf32=False)
            b_dv += tl.dot(b_k, b_dh.to(b_k.dtype), allow_tf32=False)
            b_dh += tl.dot(b_q, b_do, allow_tf32=False)
            b_dh -= tl.dot(b_d, b_dv.to(b_d.dtype), allow_tf32=False)

        tl.store(p_dk, (b_dk).to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    # sync threads
    b_h = None
    tl.debug_barrier()
    m_s = o_i[:, None] >= o_i[None, :]
    # [BV, BK]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * DK * DV, (DV, DK), (1, DV), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)
    NT = tl.cdiv(T, BT)
    for i in range(0, NT):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t), (i_v * BV, i * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_bh + i_v*B*H) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i*BT, i_k*BK), (BT, BK), (1, 0))

        # [BT, DK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [DV, BT]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, DV]
        b_do = tl.load(p_do, boundary_check=(0, 1))

        # [BT, BT]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        b_ds = tl.where(m_s, b_ds, 0)
        # [BT, DK]
        b_dq = tl.dot(b_ds.to(b_k.dtype), b_k, allow_tf32=False)
        # [DV, DK]
        if CHECK and i == 0:
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype), allow_tf32=False)
            b_h = b_h + tl.dot(b_v, b_k, allow_tf32=False)
        else:
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype), allow_tf32=False)
            b_h = b_h + tl.dot(b_v, b_k, allow_tf32=False)
        b_dq *= scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

        if i < (NT - 1):
            p_dv = tl.make_block_ptr(dv + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), ((i + 1) * BT, i_v * BV), (BT, BV), (1, 0))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_dd = tl.dot(b_dv.to(b_k.dtype), b_h.to(b_k.dtype), allow_tf32=False)
            p_dd = tl.make_block_ptr(dd + (i_bh + i_v*B*H) * s_qk_h, (T, DK), (s_qk_t, s_qk_d),
                                     ((i+1) * BT, i_k * BK), (BT, BK), (1, 0))
            tl.store(p_dd, -b_dd.to(p_dd.dtype.element_ty), boundary_check=(0, 1))


def fused_chunk_delta_rule_fwd(q, k, v, d, BT, initial_state, output_final_state):
    batch_size, n_heads, seq_len, d_head_qk = q.shape
    d_head_v = v.shape[-1]
    scale = d_head_qk ** -0.5
    BT = BT
    # ctx.BT = BT
    BK, BV = triton.next_power_of_2(d_head_qk), min(triton.next_power_of_2(d_head_v), 32)
    NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
    assert NK == 1, 'NK should be 1'
    o = q.new_empty(batch_size, n_heads, seq_len, d_head_v)
    if output_final_state:
        final_state = q.new_empty(batch_size, n_heads, d_head_qk, d_head_v, dtype=torch.float32, requires_grad=False)
    else:
        final_state = None
    CHECK = True
    # if version.parse(triton.__version__) < version.parse('2.2.0'):
    #     import warnings
    #     warnings.warn(
    #         "Triton<2.2.0 detected for running this kernel, "
    #         "which is known to have some weird compiler issues (refer to https://github.com/openai/triton/issues/2852) "
    #         "that lead to significant precision loss. "
    #         "We've add some initial condition checks to resolve this, sadly at the sacrifice of the speed. "
    #         "For optimal performance, it is recommended to install Triton>=2.2.0 (if possible)."
    #     )
    #     CHECK = True
    grid = (NV, NK, batch_size * n_heads)
    v_new = torch.empty_like(v)
    fused_chunk_delta_rule_fwd_kernel[grid](
        q, k, v, v_new, d, o, initial_state, final_state,
        q.stride(1), q.stride(2), q.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        batch_size, n_heads, seq_len, scale,
        BT=BT, DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=output_final_state,
        CHECK=CHECK,
    )
    return o, v_new, CHECK, final_state


def fused_chunk_delta_rule_bwd(q, k, v, d, do, BT, CHECK, initial_state):
    batch_size, n_heads,  seq_len, d_head_qk = q.shape
    d_head_v = v.shape[-1]
    scale = d_head_qk ** -0.5
    BK, BV = triton.next_power_of_2(d_head_qk), min(triton.next_power_of_2(d_head_v), 32)
    NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
    assert NK == 1
    dq = q.new_empty(NV, batch_size, n_heads,  seq_len, d_head_qk)
    dk = q.new_empty(NV, batch_size, n_heads,  seq_len, d_head_qk)
    dd = q.new_empty(NV, batch_size, n_heads,  seq_len, d_head_qk)
    dv = q.new_empty(NK, batch_size, n_heads, seq_len, d_head_v)
    grid = (NV, NK, batch_size * n_heads)
    fused_chunk_delta_rule_bwd_kernel[grid](
        q, k, v, d, do, dq, dk, dv, dd, initial_state,
        q.stride(1), q.stride(2), q.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        batch_size, n_heads, seq_len, scale,
        BT=BT, DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
        USE_INITIAL_STATE=initial_state is not None,
        CHECK=CHECK,
        # num_warps=num_warps,
        # num_stages=num_stages
    )
    dq = dq.sum(0)
    dk = dk.sum(0)
    dv = dv.sum(0)
    dd = dd.sum(0)
    dd[:, :, 0:BT] = 0
    return dq, dk, dv, dd

class FusedChunkDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    @custom_fwd
    def forward(ctx, q, k, v, beta, BT, initial_state, output_final_state, checkpoint_level=0):
        # lvl=1 will recompute ``fwd_prepare_wy_repr`` for saving memory.
        assert checkpoint_level in [0, 1]
        k_origin = k
        # k = _l2_norm_fwd(k_origin)
        k = k
        d, v_new = fwd_prepare_wy_repr(k, v, beta, BT)
        o, v_new2, CHECK, final_state = fused_chunk_delta_rule_fwd(q, k, v_new, d, BT, initial_state, output_final_state)
        if checkpoint_level == 1:
            d, v_new = None, None
        ctx.save_for_backward(q, k_origin, v, v_new, v_new2, d, beta, initial_state)
        ctx.CHECK = CHECK
        ctx.chunk_size = BT
        return o.to(q.dtype), final_state

    @staticmethod
    @custom_bwd
    @contiguous
    def backward(ctx, do, d_final_state=None):
        q, k_origin, v, v_new, v_new2, d, beta, initial_state = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        k = k_origin
        # k = _l2_norm_fwd(k_origin)
        if d is None:
            d, v_new = fwd_prepare_wy_repr(k, v, beta, chunk_size)
        dq, dk, dv, dd = fused_chunk_delta_rule_bwd(q, k, v_new2, d, do, chunk_size, ctx.CHECK, initial_state)
        dk2, dv, dbeta = bwd_prepare_wy_repr(k, v, beta, d, v_new, dd, dv, chunk_size)
        dk.add_(dk2)
        # dk = _l2_norm_bwd(k_origin, dk)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dbeta.to(d.dtype), None, None, None


def fused_chunk_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    BT: int,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if initial_state is not None:
        initial_state = initial_state.detach()
    o, final_state = FusedChunkDeltaRuleFunction.apply(q, k, v, beta, BT, initial_state, output_final_state)
    return o, final_state


def delta_rule_recurrence(q, k, v, beta):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    o = torch.zeros_like(v)
    S = torch.zeros(b, h, d_k, d_v).to(v)
    q = q * (d_k ** -0.5)
    k = torch.nn.functional.normalize(k, p=2, dim=-1)
    for i in range(l):
        _k = k[:, :, i]
        _q = q[:, :, i]
        _v = v[:, :, i].clone()
        beta_i = beta[:, :, i]
        _v = _v - (S.clone() * _k[..., None]).sum(-2)
        _v = _v * beta_i[..., None]
        S = S.clone() + _k.unsqueeze(-1) * _v.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q, S)
    return o


if __name__ == "__main__":
    import torch.nn.functional as F
    seq_len = 128
    b = 2
    h = 4
    q = F.normalize(torch.randn(b, h, seq_len, 64), 2, -1)
    k = F.normalize(torch.randn(b, h, seq_len, 64), 2, -1)
    v = F.normalize(torch.randn(b, h, seq_len, 128), 2, -1)
    beta = torch.rand(b, h, seq_len).sigmoid()
    q, k, v, beta = map(lambda x: x.cuda().to(torch.float32).requires_grad_(True), (q, k, v, beta))
    do = torch.rand_like(v)
    o2 = delta_rule_recurrence(q, k, v.clone(), beta)
    o2.backward(do, retain_graph=True)
    q_grad2, k_grad2, v_grad2, beta_grad2 = q.grad, k.grad, v.grad, beta.grad
    q.grad = k.grad = v.grad = beta.grad = None
    o, _ = fused_chunk_delta_rule(q, k, v, beta, 32)
    o.backward(do, retain_graph=True)
    q_grad, k_grad, v_grad, beta_grad = q.grad, k.grad, v.grad, beta.grad
    q.grad = k.grad = v.grad = beta.grad = None
    print((o - o2).abs().max())
    print((q_grad - q_grad2).abs().max())
    print((k_grad - k_grad2).abs().max())
    print((v_grad - v_grad2).abs().max())
    print((beta_grad - beta_grad2).abs().max())
