# -*- coding: utf-8 -*-

import torch
from einops import rearrange

from fla.ops.rwkv6.chunk import chunk_rwkv6
from fla.ops.rwkv6.recurrent_fuse import fused_recurrent_rwkv6


def naive_chunk_rwkv6(
    q,
    k,
    v,
    w,
    u,
    chunk_size=32,
    initial_state=None,
    output_final_state=True,
):
    assert q.shape[-2] % chunk_size == 0
    orig_dtype = q.dtype
    num_chunk = q.shape[-2] // chunk_size
    u = u.unsqueeze(0)

    q, k, v, w = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size).float(), (q, k, v, w))

    w_cumsum = w.cumsum(-2)

    kw = k * (w_cumsum[..., -1, None, :] - w_cumsum).exp()
    wkv = kw.transpose(-1, -2) @ v

    wkv_new = torch.zeros_like(wkv)

    for i in range(num_chunk - 1):
        wkv_new[:, :, i+1] = (wkv_new[:, :, i] * w_cumsum[:, :, i, -1, :, None].exp()) + wkv[:, :, i]

    o_inter = torch.einsum('b h n d p, b h n c d -> b h n c p', wkv_new, (q * (w_cumsum - w).exp()))

    o_intra = torch.zeros_like(o_inter)
    for i in range(chunk_size):
        attn = (q[:, :, :, i, None] * k * (w_cumsum[:, :, :, i, None] - w[:, :, :, i, None] - w_cumsum).exp()).sum(-1)
        mask = (torch.arange(0, chunk_size) < i).to(attn.device)
        attn.masked_fill_(~mask, 0)
        intra_inter_o = (attn.unsqueeze(-1) * v).sum(-2)
        intra_intra_o = (q[:, :, :, i] * u.unsqueeze(2) * k[:, :, :, i]).sum(-1).unsqueeze(-1) * v[:, :, :, i]
        o_intra[:, :, :, i] = intra_inter_o + intra_intra_o
    o = o_inter + o_intra
    return rearrange(o, 'b h n c d -> b h (n c) d').to(orig_dtype)


if __name__ == "__main__":
    B = 4
    H = 4
    L = 1024
    D = 100
    dtype = torch.bfloat16
    require_grad = True
    q = (torch.randn(B, H, L, D).cuda().to(dtype)).requires_grad_(require_grad)
    k = (torch.randn(B, H, L, D).cuda().to(dtype)).requires_grad_(require_grad)
    v = torch.randn(B, H, L, 2*D).cuda().to(dtype).requires_grad_(require_grad)
    w = torch.nn.functional.logsigmoid(torch.randn(B, H, L, D)).cuda().to(dtype).requires_grad_(require_grad)
    u = (torch.randn(H, D).cuda().to(dtype)).requires_grad_(require_grad)
    do = torch.rand_like(v).cuda()
    o2, _ = chunk_rwkv6(q, k, v, w.clone(), u)
    o, _ = fused_recurrent_rwkv6(q, k, v, w, u, scale=1.0)
    o.backward(do)
    dq, q.grad = q.grad.clone(), None
    dk, k.grad = k.grad.clone(), None
    dv, v.grad = v.grad.clone(), None
    dw, w.grad = w.grad.clone(), None
    du, u.grad = u.grad.clone(), None
    print((o - o2).abs().max())
    o2.backward(do)
    print((o-o2).abs().max())
    print((q.grad - dq).abs().max())
    print((k.grad - dk).abs().max())
    print((v.grad - dv).abs().max())
    print((w.grad - dw).abs().max())
    print((u.grad - du).abs().max())
