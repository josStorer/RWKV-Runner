# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

from fla.ops.gla.recurrent_fuse import fused_recurrent_gla


def ceildiv(a, b):
    return -(a // -b)


def naive_recurrent_gla(
    q,
    k,
    v,
    gk,
    initial_state=None,
    output_final_state=False,
    causal=True
):
    orig_dtype = q.dtype
    q, k, v, gk = map(lambda x: x.float(), (q, k, v, gk))
    batch_size, n_heads, seq_len, d_head_k = q.shape
    _, _, _, d_head_v = v.shape
    h = torch.zeros(batch_size, n_heads, d_head_k, d_head_v, dtype=torch.float32, device=q.device)
    o = torch.zeros_like(v)
    scale = d_head_k ** -0.5

    if initial_state is not None:
        h += initial_state

    for i in range(seq_len):
        q_i = q[:, :, i, :] * scale
        k_i = k[:, :, i]
        v_i = v[:, :, i, :]
        gk_i = gk[:, :, i].exp()
        kv_i = k_i[..., None] * v_i[..., None, :]
        h = h * gk_i[..., None] + kv_i
        o_i = (q_i[..., None] * h).sum(-2)
        o[:, :, i] = o_i

    if causal:
        return o.to(orig_dtype), h
    else:
        o_reverse = torch.zeros_like(v)
        h = torch.zeros(batch_size, n_heads, d_head_k, d_head_v, dtype=torch.float32, device=q.device)
        for i in range(seq_len-1, -1, -1):
            q_i = q[:, :, i, :] * scale
            k_i = k[:, :, i]
            v_i = v[:, :, i, :]
            gk_i = gk[:, :, i].exp()
            kv_i = k_i[..., None] * v_i[..., None, :]
            h = h * gk_i[..., None] + kv_i
            o_i = (q_i[..., None] * h).sum(-2)
            o_reverse[:, :, i] = o_i

        return o, o_reverse


if __name__ == "__main__":
    B = 4
    H = 4
    L = 512
    D = 128
    dtype = torch.float32
    q = (torch.randn(B, H, L, D).cuda().to(dtype)).requires_grad_(True)
    k = (torch.randn(B, H, L, D).cuda().to(dtype)).requires_grad_(True)
    v = torch.randn(B, H, L, D).cuda().to(dtype).requires_grad_(True)
    g = F.logsigmoid(torch.rand(B, H, L, D)).cuda(
    ).clamp_min(-1).to(torch.float32).requires_grad_(True)

    do = torch.rand_like(v).cuda()
    do2 = torch.rand_like(v).cuda()
    intial_state = torch.rand(B, H, D, D).cuda()

    ref, ref_rev = naive_recurrent_gla(q, k, v, g, causal=False)

    ref.backward(do, retain_graph=True)
    ref_rev.backward(do2, retain_graph=True)

    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None

    tri, tri_rev = fused_recurrent_gla(
        q, k, v, g, initial_state=None, scale=D**-0.5, output_final_state=False, causal=False)
    tri.backward(do, retain_graph=True)
    tri_rev.backward(do2, retain_graph=True)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None

    assert ref.allclose(tri, 0, 1e-5), breakpoint()
    assert ref_rev.allclose(tri_rev, 0, 1e-5), breakpoint()
    assert ref_dq.allclose(tri_dq, 0, 1e-5), breakpoint()
    assert ref_dk.allclose(tri_dk, 0, 1e-5), breakpoint()
    assert ref_dv.allclose(tri_dv, 0, 1e-5), breakpoint()
    assert ref_dg.allclose(tri_dg, 0, 1e-4), breakpoint()

    # tri = fused_chunk_gla(q, k, v, g)
    # tri.backward(do, retain_graph=True)
    # tri_dq, q.grad = q.grad.clone(), None
    # tri_dk, k.grad = k.grad.clone(), None
    # tri_dv, v.grad = v.grad.clone(), None
    # tri_dg, g.grad = g.grad.clone(), None

    # assert ref.allclose(tri, 0, 1e-5), breakpoint()
    # assert ref_dq.allclose(tri_dq, 0, 1e-5), breakpoint()
    # assert ref_dk.allclose(tri_dk, 0, 1e-5), breakpoint()
    # assert ref_dv.allclose(tri_dv, 0, 1e-5), breakpoint()
    # assert ref_dg.allclose(tri_dg, 0, 1e-4), breakpoint()
    # breakpoint()
    print("Pass")
