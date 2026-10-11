"""Attention over short sequences, GQA with 2 KV heads: non-causal (LocDiT: 11 tokens per stream, LocEnc: 5) and causal
varlen (prefill steps of short sequences without a cached prefix, e.g. a burst's head-start step).

FA2 tiles queries in blocks of 128 rows, so an 11-token sequence uses <10% of each tile, and one launch covers
batch x heads tiny problems. Here one program owns one sequence: for each KV group it loads the group's query
heads for all LP (= next power of 2 >= L) positions as one [LP * G, D] tile, the [LP, D] keys / values, and does
softmax(Q K^T) V with the padded key columns masked. Because a program owns whole tokens, the epilogue can also
quantize the output per token for the o_proj int8 GEMM (o * inv_smooth, max over all heads -> int8 + scale),
which removes the bf16 o round trip and the separate quantize pass.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _attend_group_short(q_ptr, k_ptr, v_ptr, b, L, g, q_stride, kv_stride, sm_scale,
                        H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, LP: tl.constexpr):
    G: tl.constexpr = H // HK
    r = tl.arange(0, LP * G)
    l = r // G
    h = g * G + r % G
    d = tl.arange(0, D)
    rows = b.to(tl.int64) * L + l
    q = tl.load(q_ptr + rows[:, None] * q_stride + h[:, None] * D + d[None, :], mask=(l < L)[:, None], other=0.0)
    lk = tl.arange(0, LP)
    krows = b.to(tl.int64) * L + lk
    kv_off = krows[:, None] * kv_stride + g * D + d[None, :]
    k = tl.load(k_ptr + kv_off, mask=(lk < L)[:, None], other=0.0)
    v = tl.load(v_ptr + kv_off, mask=(lk < L)[:, None], other=0.0)
    s = tl.dot(q, tl.trans(k)) * sm_scale
    s = tl.where((lk < L)[None, :], s, -1.0e30)
    p = tl.exp(s - tl.max(s, axis=1)[:, None])
    p = p / tl.sum(p, axis=1)[:, None]
    return tl.dot(p.to(v.dtype), v)  # [LP * G, D], rows (token, head of the group)


@triton.jit
def _short_attn(
    q_ptr, k_ptr, v_ptr, o_ptr, scale_ptr, inv_smooth_ptr,
    L, q_stride, kv_stride, sm_scale,
    H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, LP: tl.constexpr,
    QUANT: tl.constexpr, HAS_SMOOTH: tl.constexpr,
):
    b = tl.program_id(0)
    G: tl.constexpr = H // HK
    W: tl.constexpr = G * D
    o0 = tl.reshape(_attend_group_short(q_ptr, k_ptr, v_ptr, b, L, 0, q_stride, kv_stride, sm_scale, H, HK, D, LP), (LP, W))
    o1 = tl.reshape(_attend_group_short(q_ptr, k_ptr, v_ptr, b, L, 1, q_stride, kv_stride, sm_scale, H, HK, D, LP), (LP, W))
    lt = tl.arange(0, LP)
    ok = lt < L
    col = tl.arange(0, W)
    rows = b.to(tl.int64) * L + lt
    base = o_ptr + rows[:, None] * (H * D) + col[None, :]
    if QUANT:
        if HAS_SMOOTH:
            o0 = o0 * tl.load(inv_smooth_ptr + col)[None, :]
            o1 = o1 * tl.load(inv_smooth_ptr + W + col)[None, :]
        amax = tl.maximum(tl.maximum(tl.max(tl.abs(o0), axis=1), tl.max(tl.abs(o1), axis=1)), 1e-6)
        scale = amax / 127.0
        tl.store(base, tl.extra.cuda.libdevice.rint(o0 / scale[:, None]).to(tl.int8), mask=ok[:, None])
        tl.store(base + W, tl.extra.cuda.libdevice.rint(o1 / scale[:, None]).to(tl.int8), mask=ok[:, None])
        tl.store(scale_ptr + rows, scale, mask=ok)
    else:
        tl.store(base, o0.to(o_ptr.dtype.element_ty), mask=ok[:, None])
        tl.store(base + W, o1.to(o_ptr.dtype.element_ty), mask=ok[:, None])


def short_attention(q, k, v, sm_scale: float, quant: bool = False, inv_smooth=None, num_warps: int = 4):
    """q [B, L, H, D], k / v [B, L, HK, D] (token rows may be strided views, e.g. of the qkv output; head and dim
    contiguous). Returns o [B * L, H * D] in q's dtype, or (int8 o, fp32 per-token scale) with ``quant``."""
    B, L, H, D = q.shape
    HK = k.shape[2]
    assert HK == 2 and q.stride(3) == 1 and q.stride(2) == D and k.stride(2) == D and k.stride() == v.stride()
    qs, ks = (q.stride(1), k.stride(1)) if L > 1 else (q.stride(0), k.stride(0))  # token row strides
    assert q.stride(0) == L * qs and k.stride(0) == L * ks
    LP = max(triton.next_power_of_2(L), 16)  # tl.dot needs K >= 16 for P @ V
    if quant:
        o = torch.empty(B * L, H * D, device=q.device, dtype=torch.int8)
        scale = torch.empty(B * L, device=q.device, dtype=torch.float32)
    else:
        o = torch.empty(B * L, H * D, device=q.device, dtype=q.dtype)
        scale = o
    _short_attn[(B,)](
        q, k, v, o, scale, inv_smooth if inv_smooth is not None else scale,
        L, qs, ks, sm_scale,
        H=H, HK=HK, D=D, LP=LP, QUANT=quant, HAS_SMOOTH=inv_smooth is not None, num_warps=num_warps,
    )
    return (o, scale) if quant else o



@triton.jit
def _attend_group_causal(q_ptr, k_ptr, v_ptr, start, L, g, q_stride, kv_stride, sm_scale,
                         H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, LP: tl.constexpr):
    """_attend_group_short for one sequence of a varlen batch (rows start .. start + L), causal."""
    G: tl.constexpr = H // HK
    r = tl.arange(0, LP * G)
    l = r // G
    h = g * G + r % G
    d = tl.arange(0, D)
    q = tl.load(q_ptr + (start + l)[:, None] * q_stride + h[:, None] * D + d[None, :], mask=(l < L)[:, None], other=0.0)
    lk = tl.arange(0, LP)
    kv_off = (start + lk)[:, None] * kv_stride + g * D + d[None, :]
    k = tl.load(k_ptr + kv_off, mask=(lk < L)[:, None], other=0.0)
    v = tl.load(v_ptr + kv_off, mask=(lk < L)[:, None], other=0.0)
    s = tl.dot(q, tl.trans(k)) * sm_scale
    s = tl.where((lk[None, :] <= l[:, None]) & (lk < L)[None, :], s, -1.0e30)
    p = tl.exp(s - tl.max(s, axis=1)[:, None])
    p = p / tl.sum(p, axis=1)[:, None]
    return tl.dot(p.to(v.dtype), v)


@triton.jit
def _short_causal_attn(
    q_ptr, k_ptr, v_ptr, cu_ptr, o_ptr, scale_ptr, inv_smooth_ptr,
    q_stride, kv_stride, sm_scale,
    H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, LP: tl.constexpr,
    QUANT: tl.constexpr, HAS_SMOOTH: tl.constexpr,
):
    b = tl.program_id(0)
    start = tl.load(cu_ptr + b).to(tl.int64)
    L = tl.load(cu_ptr + b + 1) - tl.load(cu_ptr + b)
    if L <= 0:
        return
    G: tl.constexpr = H // HK
    W: tl.constexpr = G * D
    o0 = tl.reshape(_attend_group_causal(q_ptr, k_ptr, v_ptr, start, L, 0, q_stride, kv_stride, sm_scale, H, HK, D, LP),
                    (LP, W))
    o1 = tl.reshape(_attend_group_causal(q_ptr, k_ptr, v_ptr, start, L, 1, q_stride, kv_stride, sm_scale, H, HK, D, LP),
                    (LP, W))
    lt = tl.arange(0, LP)
    ok = lt < L
    col = tl.arange(0, W)
    rows = start + lt
    base = o_ptr + rows[:, None] * (H * D) + col[None, :]
    if QUANT:
        if HAS_SMOOTH:
            o0 = o0 * tl.load(inv_smooth_ptr + col)[None, :]
            o1 = o1 * tl.load(inv_smooth_ptr + W + col)[None, :]
        amax = tl.maximum(tl.maximum(tl.max(tl.abs(o0), axis=1), tl.max(tl.abs(o1), axis=1)), 1e-6)
        scale = amax / 127.0
        tl.store(base, tl.extra.cuda.libdevice.rint(o0 / scale[:, None]).to(tl.int8), mask=ok[:, None])
        tl.store(base + W, tl.extra.cuda.libdevice.rint(o1 / scale[:, None]).to(tl.int8), mask=ok[:, None])
        tl.store(scale_ptr + rows, scale, mask=ok)
    else:
        tl.store(base, o0.to(o_ptr.dtype.element_ty), mask=ok[:, None])
        tl.store(base + W, o1.to(o_ptr.dtype.element_ty), mask=ok[:, None])


# the longest sequence short_causal_attention takes: LP * G query rows per KV group in one tile (32 * 8 = 256)
SHORT_CAUSAL_MAX_LEN = 32


def short_causal_attention(q, k, v, cu_seqlens, max_seqlen: int, sm_scale: float, quant: bool = False,
                           inv_smooth=None):
    """Causal attention of a prefill step's short sequences over their own tokens (no cached prefix): q [T, H, D],
    k / v [T, HK, D] (token rows may be strided views of the qkv output), cu_seqlens [B + 1]. FA2's varlen kernel
    gives every sequence a 128-row query tile; a burst's head-start step (200 sequences of ~7 tokens) spent 113 us
    per layer there. Returns o [T, H * D] in q's dtype, or (int8 o, fp32 per-token scale) with ``quant``."""
    T, H, D = q.shape
    HK = k.shape[1]
    assert HK == 2 and q.stride(2) == 1 and q.stride(1) == D and k.stride(1) == D and k.stride() == v.stride()
    assert 0 < max_seqlen <= SHORT_CAUSAL_MAX_LEN
    LP = max(triton.next_power_of_2(max_seqlen), 16)  # tl.dot needs K >= 16 for P @ V
    B = cu_seqlens.numel() - 1
    if quant:
        o = torch.empty(T, H * D, device=q.device, dtype=torch.int8)
        scale = torch.empty(T, device=q.device, dtype=torch.float32)
    else:
        o = torch.empty(T, H * D, device=q.device, dtype=q.dtype)
        scale = o
    _short_causal_attn[(B,)](
        q, k, v, cu_seqlens, o, scale, inv_smooth if inv_smooth is not None else scale,
        q.stride(0), k.stride(0), sm_scale,
        H=H, HK=HK, D=D, LP=LP, QUANT=quant, HAS_SMOOTH=inv_smooth is not None, num_warps=4 if LP == 16 else 8,
    )
    return (o, scale) if quant else o
