# -*- coding: utf-8 -*-

import torch
from einops import rearrange

from fla.ops.based.chunk_fuse import fused_chunk_based
from fla.ops.based.parallel import parallel_based


def naive_parallel_based(q, k, v, use_scale=True, use_norm=True):
    if use_scale:
        q = q * (q.shape[-1] ** -0.5)
    attn = q @ k.transpose(-2, -1)
    attn = 1 + attn + 1/2 * (attn ** 2)
    attn.masked_fill_(~torch.tril(torch.ones(
        q.shape[-2], q.shape[-2], dtype=torch.bool, device=q.device)), 0)
    o = attn @ v
    if use_norm:
        z = attn.sum(-1)
        return o / (z[..., None] + 1e-6)
    else:
        return o


def naive_chunk_based(q, k, v, chunk_size=256):
    q = q * (q.shape[-1] ** -0.5)

    # compute normalizer.
    k_cumsum = torch.cumsum(k, dim=-2)
    kk_cumsum = torch.cumsum(k.unsqueeze(-1) * k.unsqueeze(-2), dim=-3)
    # first
    z = (q * k_cumsum).sum(-1)
    # second order
    z += (q.unsqueeze(-1) * q.unsqueeze(-2) * kk_cumsum).sum((-1, -2)) * 0.5
    # zero-th order
    z += (torch.arange(0, q.shape[-2]).to(z.device) * 1.0 + 1.0)[None, None, :]

    # compute o
    # constant term
    _o = v.cumsum(-2)

    q = rearrange(q, 'b h (n c) d -> b h n c d', c=chunk_size)

    k = rearrange(k, 'b h (n c) d -> b h n c d', c=chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size)

    intra_chunk_attn = q @ k.transpose(-2, -1)
    intra_chunk_attn = intra_chunk_attn + 1/2 * (intra_chunk_attn ** 2)
    intra_chunk_attn.masked_fill_(
        ~torch.tril(
            torch.ones(chunk_size, chunk_size,
                       dtype=torch.bool, device=q.device),
        ), 0)
    o = intra_chunk_attn @ v

    # quadractic term
    kv = torch.einsum(
        'b h n c x, b h n c y, b h n c z -> b h n x y z', k, k, v)
    kv = kv.cumsum(2)
    kv = torch.cat([torch.zeros_like(kv[:, :, :1]), kv[:, :, :-1]], dim=2)

    o += 0.5 * torch.einsum('b h n x y z, b h n c x, b h n c y -> b h n c z', kv, q, q)

    # linear term
    kv = torch.einsum('b h n c x, b h n c y -> b h n x y', k, v)
    kv = kv.cumsum(2)
    kv = torch.cat([torch.zeros_like(kv[:, :, :1]), kv[:, :, :-1]], dim=2)
    o += torch.einsum('b h n x y, b h n c x -> b h n c y', kv, q)

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    o = o + _o
    return o / (z[..., None] + 1e-6)


if __name__ == "__main__":
    B = 4
    H = 4
    L = 128
    # D = 15
    dtype = torch.float32
    q = (torch.randn(B, H, L, 16).cuda().to(dtype)).requires_grad_(True)
    k = (torch.randn(B, H, L, 16).cuda().to(dtype)).requires_grad_(True)
    v = torch.randn(B, H, L, 128).cuda().to(dtype).requires_grad_(True)

    do = torch.randn_like(v).cuda()
    ref = naive_parallel_based(q, k, v, True, True)
    ref.backward(do, retain_graph=True)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    # tri = naive_chunk_based(q, k, v)
    # tri.backward(do, retain_graph=True)
    # tri_dq, q.grad = q.grad.clone(), None
    # tri_dk, k.grad = k.grad.clone(), None
    # tri_dv, v.grad = v.grad.clone(), None

    # assert ref.allclose(tri, 0, 1e-4), breakpoint()
    # assert ref_dq.allclose(tri_dq, 0, 1e-4), breakpoint()
    # assert ref_dk.allclose(tri_dk, 0, 1e-4), breakpoint()
    # assert ref_dv.allclose(tri_dv, 0, 1e-4), breakpoint()

    tri = fused_chunk_based(q, k, v, True, True)
    tri.backward(do, retain_graph=True)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    print((ref-tri).abs().max())
    print((ref_dq-tri_dq).abs().max())
    print((ref_dk-tri_dk).abs().max())
    print((ref_dv-tri_dv).abs().max())

    # assert ref.allclose(tri, 0, 1e-4), breakpoint()
    # assert ref_dq.allclose(tri_dq, 0, 1e-4), breakpoint()
    # assert ref_dk.allclose(tri_dk, 0, 1e-4), breakpoint()
    # assert ref_dv.allclose(tri_dv, 0, 1e-4), breakpoint()

    tri = parallel_based(q, k, v, True, True)
    tri.backward(do, retain_graph=True)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    print((ref-tri).abs().max())
    print((ref_dq-tri_dq).abs().max())
    print((ref_dk-tri_dk).abs().max())
    print((ref_dv-tri_dv).abs().max())

    # assert ref.allclose(tri, 0, 1e-4), breakpoint()
    # assert ref_dq.allclose(tri_dq, 0, 1e-4), breakpoint()
    # assert ref_dk.allclose(tri_dk, 0, 1e-4), breakpoint()
    # assert ref_dv.allclose(tri_dv, 0, 1e-4), breakpoint()
