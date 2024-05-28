# -*- coding: utf-8 -*-

import torch
import triton
import triton.language as tl
from einops import rearrange
from torch.cuda.amp import custom_bwd, custom_fwd

from fla.utils import contiguous

# Inspired by "THE WY REPRESENTATION FOR PRODUCTS OF HOUSEHOLDER MATRICES" https://epubs.siam.org/doi/pdf/10.1137/0908009
# o: cumprod
# o2: cumprodsum
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
def fwd_prepare_wy_repr_kernel(
    k,
    v,
    beta,
    w,  
    u,
    A, 
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    T,
    K,
    V,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None]).to(b_k.dtype)
        b_A += tl.dot(b_kb, tl.trans(b_k), allow_tf32=False)

    b_A = -tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_A, 0)

    for i in range(1, BT):
        mask = tl.arange(0, BT) == i
        b_a = tl.sum(tl.where(mask[:, None], b_A, 0), 0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0) * (tl.arange(0, BT) < i)
        b_A = tl.where(mask[:, None], b_a, b_A)

    b_A += tl.arange(0, BT)[:, None] == tl.arange(0, BT)[None, :]

    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_A, (b_A).to(p_A.dtype.element_ty), boundary_check=(0, 1))
    b_A = b_A.to(k.dtype.element_ty)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        p_u = tl.make_block_ptr(u + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_u, (b_u).to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb, allow_tf32=False)
        p_w = tl.make_block_ptr(w + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))



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
def fwd_recompute_w_u_kernel(
    k,
    v,
    beta,
    w,  
    u,
    A, 
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    T,
    K,
    V,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    
    p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))

    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(k.dtype.element_ty)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        p_u = tl.make_block_ptr(u + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_u, (b_u).to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb, allow_tf32=False)
        p_w = tl.make_block_ptr(w + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))





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
def bwd_prepare_wy_repr_kernel(
    k, v, beta, A,  
    dw, du,
    dk, dv, dbeta,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    T,
    K,
    V,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(k.dtype.element_ty)

    b_dbeta = tl.zeros([BT], dtype=tl.float32)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))

    for i_v in range(tl.cdiv(V, BV)):
        p_v =  tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_du = tl.make_block_ptr(du + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_v_beta = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_du = tl.load(p_du, boundary_check=(0, 1))
        b_dA += tl.dot(b_du, tl.trans(b_v_beta), allow_tf32=False)
        b_dv_beta = tl.dot(tl.trans(b_A), b_du, allow_tf32=False)
        b_dv = b_dv_beta * b_beta[:, None]
        b_dbeta += tl.sum(b_dv_beta * b_v, 1)
        # store
        p_dv = tl.make_block_ptr(dv + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()    
    b_A2 = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dw = tl.make_block_ptr(dw + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))        
        b_k_beta = (b_k * b_beta[:, None]).to(b_k.dtype)
        b_dw = tl.load(p_dw, boundary_check=(0, 1))
        b_dA += tl.dot(b_dw, tl.trans(b_k_beta), allow_tf32=False)       
        b_A2 += tl.dot(b_k_beta, tl.trans(b_k), allow_tf32=False)
        b_dk_beta = tl.dot(tl.trans(b_A), b_dw, allow_tf32=False)
        b_dk = b_dk_beta * b_beta[:, None]
        b_dbeta += tl.sum(b_dk_beta * b_k, 1)
        # store        
        p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    b_A -= (tl.arange(0, BT)[:, None] == tl.arange(0, BT)[None, :])
    b_A2 = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], -b_A2, 0)
    b_dA = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_dA, 0)
    tl.debug_barrier()

    for i in range(BT-1, 0, -1):
        mask = tl.arange(0, BT) == i
        b_da = tl.sum(tl.where(mask[:, None], b_dA, 0), 0) 
        b_a =  tl.sum(tl.where(mask[:, None], b_A2, 0), 0) 
        b_da2 = b_da + tl.sum(b_da[None, :] * b_A, 1)     
        b_dA = tl.where(mask[:, None], b_da2, b_dA)
        b_dA += b_da[None, :] * b_a[:, None]

    b_dA = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], -b_dA, 0).to(k.dtype.element_ty)
    tl.debug_barrier()

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))        
        b_dk = tl.load(p_dk, boundary_check=(0, 1))
        b_k_beta = (b_k * b_beta[:, None]).to(b_k.dtype)

        b_dk_beta = tl.dot(b_dA, b_k, allow_tf32=False)
        b_dbeta += tl.sum(b_dk_beta * b_k, 1)
        b_dk += tl.dot(tl.trans(b_dA), b_k_beta, allow_tf32=False) 
        b_dk += b_dk_beta * b_beta[:, None]        
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    
    p_dbeta = tl.make_block_ptr(dbeta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dbeta, b_dbeta.to(p_dbeta.dtype.element_ty),boundary_check=(0,))


def fwd_prepare_wy_repr(k, v, beta, BT):
    B, H, T, K, V = *k.shape, v.shape[-1]
    u = torch.empty_like(v)
    w = torch.empty_like(k)
    NT = triton.cdiv(T, BT)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    A = torch.empty(B, H, T, BT, device=k.device, dtype=k.dtype)
    fwd_prepare_wy_repr_kernel[(NT, B*H)](
        k, v, beta, w, u, A,
        k.stride(1), k.stride(2), k.stride(3), 
        v.stride(1), v.stride(2), v.stride(3),
        T, K, V, BT, BK, BV
    )
    return w, u, A



def fwd_recompute_w_u(k, v, beta, A, BT):
    B, H, T, K, V = *k.shape, v.shape[-1]
    u = torch.empty_like(v)
    w = torch.empty_like(k)
    NT = triton.cdiv(T, BT)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    fwd_recompute_w_u_kernel[(NT, B*H)](
        k, v, beta, w, u, A,
        k.stride(1), k.stride(2), k.stride(3), 
        v.stride(1), v.stride(2), v.stride(3),
        T, K, V, BT, BK, BV
    )
    return w, u





def bwd_prepare_wy_repr(k, v, beta, A, dw, du, BT):
    B, H, T, K, V = *k.shape, v.shape[-1]

    NT = triton.cdiv(T, BT)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NT = triton.cdiv(T, BT)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v).contiguous()
    dbeta = torch.zeros_like(beta)

    bwd_prepare_wy_repr_kernel[(NT, B*H)](
        k, v, beta, A,
        dw, du,  
        dk, dv, dbeta,
        k.stride(1), k.stride(2), k.stride(3), 
        v.stride(1), v.stride(2), v.stride(3),
        T, K, V, BT, BK, BV
    )
    return dk, dv, dbeta


class WYRepresentationPrepration(torch.autograd.Function):
    @staticmethod
    @contiguous
    @custom_fwd
    def forward(ctx, k, v, beta, chunk_size):
        ctx.BT = chunk_size
        w, u, A = fwd_prepare_wy_repr(k, v, beta,  ctx.BT)
        ctx.save_for_backward(k, v, beta, A)
        return w, u

    @staticmethod
    @contiguous
    @custom_bwd
    def backward(ctx, dw, du):
        k, v, beta, A = ctx.saved_tensors
        BT = ctx.BT
        dk, dv, dbeta = bwd_prepare_wy_repr(k, v, beta, A, dw, du, BT)
        return dk, dv, dbeta, None




prepare_wy_repr = WYRepresentationPrepration.apply

def naive(k, v, beta, chunk_size):
    l_org = k.shape[2]
    l_new = triton.next_power_of_2(l_org)
    # pad k, v, beta
    k = torch.cat([k, torch.zeros_like(k)[:, :, :l_new-l_org, :]], dim=2)
    v = torch.cat([v, torch.zeros_like(v)[:, :, :l_new-l_org, :]], dim=2)
    beta = torch.cat([beta, torch.zeros_like(beta)[:, :, :l_new-l_org]], dim=2)

    k, v = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), (k, v))
    # k = torch.nn.functional.normalize(k, dim=-1, p=2)
    beta = rearrange(beta, 'b h (n c) -> b h n c', c=chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=k.device), diagonal=0)
    k_beta = k * beta[..., None]
    v = v * beta[..., None]
    attn = (k @ k.transpose(-1, -2)).masked_fill_(mask, 0)
    attn = attn * beta[..., None]
    x = attn @ v

    o = torch.zeros_like(k)
    o2 = torch.zeros_like(v)

    o[..., 0, :] = k_beta[..., 0, :].clone()
    o2[..., 0, :] = x[..., 0, :].clone()
    for i in range(1, chunk_size):
        o_i = (o[..., :i, :]).clone()
        o[..., i, :] = -(attn[..., i, :i, None] * o_i).sum(3) + k_beta[..., i, :]
        o2_i = (o2[..., :i, :]).clone()
        o2[..., i, :] = -(attn[..., i, :i, None] * o2_i).sum(3) + x[..., i, :]
    return map(lambda x: rearrange(x, 'b h n c d -> b h (n c) d')[:, :, :l_org], (o, v-o2))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    seq_len = 1024
    b = 4
    h = 4
    k = torch.nn.functional.normalize(torch.randn(b, h, seq_len, 128), dim=-1, p=2)
    v = torch.randn(b, h, seq_len, 128) 
    beta = torch.rand(b, h, seq_len).sigmoid()
    # beta = torch.ones(b, h, seq_len)
    require_grad = True

    k, v, beta = map(lambda x: x.cuda().requires_grad_(require_grad), (k, v, beta))
    do = torch.rand_like(k)
    do2 = torch.rand_like(v)

    o1, o2 = naive(k.clone(), v.clone(), beta.clone(), 64)
    if require_grad:
        o1.backward(do, retain_graph=True)
        o2.backward(do2, retain_graph=True)

        k_grad2, v_grad2, beta_grad2 = k.grad, v.grad, beta.grad
        k.grad = v.grad = beta.grad = None

    o3, o4 = prepare_wy_repr(k.clone(), v.clone(), beta.clone())
    print((o1-o3).abs().max())
    print((o2-o4).abs().max())

    if require_grad:
        o3.backward(do, retain_graph=True)
        o4.backward(do2, retain_graph=True)
        k_grad, v_grad, beta_grad = k.grad, v.grad, beta.grad
        print((k_grad2-k_grad).abs().max())
        print((v_grad2-v_grad).abs().max())
        print((beta_grad2-beta_grad).abs().max())
    breakpoint()

