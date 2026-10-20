"""Paged causal prefill attention for GQA with 2 KV heads (VoxCPM2: 16 query heads, head_dim 128), optionally
with int8 output.

One program handles BLOCK_Q query tokens of one sequence and all of their heads, one KV group after the other
(query tile rows: (token, head of the group)). Keys are read from the paged KV cache (block table, PAGE slots per
page) and cover the cached prefix plus this step's tokens (already stored), causally: query token t of a sequence
with nq new tokens and nk keys sits at position nk - nq + t.

Because a program owns whole tokens, the epilogue can quantize the output for the o_proj int8 GEMM directly:
per-token max |o * inv_smooth| over all heads -> int8 o + fp32 scale, without the bf16 o round trip and the
separate quantize pass. Voice-cloning prefill (44 sequences x ~46 new tokens on a ~50-token cached prefix), GPU
time per layer: FA2 paged 70-76 us, + quantize 103 us; this kernel 66-74 us including the quantization."""

import torch
import triton
import triton.language as tl


@triton.jit
def _attend_group(q_ptr, k_ptr, v_ptr, bt_ptr, b, q0, nq, nk, qb, g, q_stride, bt_stride, sm_scale,
                  H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
                  BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr):
    """Flash attention of the BLOCK_Q tokens' query heads of KV group g ([BLOCK_Q * H/HK, D] rows (token, head))."""
    G: tl.constexpr = H // HK
    R: tl.constexpr = BLOCK_Q * G
    r = tl.arange(0, R)
    t = qb * BLOCK_Q + r // G
    h = g * G + r % G
    d = tl.arange(0, D)
    q = tl.load(q_ptr + (q0 + t).to(tl.int64)[:, None] * q_stride + h[:, None] * D + d[None, :],
                mask=(t < nq)[:, None], other=0.0)
    pos = nk - nq + t
    c = tl.arange(0, BLOCK_K)
    m_i = tl.full((R,), -1.0e30, dtype=tl.float32)
    l_i = tl.zeros((R,), dtype=tl.float32)
    acc = tl.zeros((R, D), dtype=tl.float32)
    kv_end = tl.minimum(nk, nk - nq + (qb + 1) * BLOCK_Q)
    for kb in range(0, tl.cdiv(kv_end, BLOCK_K)):
        start = kb * BLOCK_K
        page = tl.load(bt_ptr + b * bt_stride + start // PAGE).to(tl.int64)
        j = start + c
        slot = page * PAGE + start % PAGE + c
        kv_off = (slot * HK + g)[:, None] * D + d[None, :]
        k = tl.load(k_ptr + kv_off, mask=(j < nk)[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * sm_scale
        valid = (j[None, :] <= pos[:, None]) & (j[None, :] < nk)
        s = tl.where(valid, s, -1.0e30)
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.where(valid, tl.exp(s - m_new[:, None]), 0.0)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptr + kv_off, mask=(j < nk)[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    return acc / tl.maximum(l_i, 1e-30)[:, None]


@triton.jit(do_not_specialize=["bt_stride"])
def _paged_prefill_attn(
    q_ptr, k_ptr, v_ptr, bt_ptr, cu_q_ptr, cu_k_ptr, o_ptr, scale_ptr, inv_smooth_ptr,
    q_stride, bt_stride, sm_scale,
    H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, QUANT: tl.constexpr, HAS_SMOOTH: tl.constexpr,
):
    """One program: BLOCK_Q query tokens of sequence b, both KV groups attended one after the other; the per-token
    scale is taken over both groups' outputs."""
    b = tl.program_id(0)
    qb = tl.program_id(1)
    q0 = tl.load(cu_q_ptr + b)
    nq = tl.load(cu_q_ptr + b + 1) - q0
    nk = tl.load(cu_k_ptr + b + 1) - tl.load(cu_k_ptr + b)
    if qb * BLOCK_Q >= nq:
        return
    G: tl.constexpr = H // HK
    W: tl.constexpr = G * D  # one group's columns of a token
    o0 = tl.reshape(_attend_group(q_ptr, k_ptr, v_ptr, bt_ptr, b, q0, nq, nk, qb, 0, q_stride, bt_stride, sm_scale,
                                  H, HK, D, PAGE, BLOCK_Q, BLOCK_K), (BLOCK_Q, W))
    o1 = tl.reshape(_attend_group(q_ptr, k_ptr, v_ptr, bt_ptr, b, q0, nq, nk, qb, 1, q_stride, bt_stride, sm_scale,
                                  H, HK, D, PAGE, BLOCK_Q, BLOCK_K), (BLOCK_Q, W))
    tq = qb * BLOCK_Q + tl.arange(0, BLOCK_Q)
    ok = tq < nq
    col = tl.arange(0, W)
    base = o_ptr + (q0 + tq).to(tl.int64)[:, None] * (H * D) + col[None, :]
    if QUANT:
        if HAS_SMOOTH:
            o0 = o0 * tl.load(inv_smooth_ptr + col)[None, :]
            o1 = o1 * tl.load(inv_smooth_ptr + W + col)[None, :]
        amax = tl.maximum(tl.maximum(tl.max(tl.abs(o0), axis=1), tl.max(tl.abs(o1), axis=1)), 1e-6)
        scale = amax / 127.0
        tl.store(base, tl.extra.cuda.libdevice.rint(o0 / scale[:, None]).to(tl.int8), mask=ok[:, None])
        tl.store(base + W, tl.extra.cuda.libdevice.rint(o1 / scale[:, None]).to(tl.int8), mask=ok[:, None])
        tl.store(scale_ptr + q0 + tq, scale, mask=ok)
    else:
        tl.store(base, o0.to(o_ptr.dtype.element_ty), mask=ok[:, None])
        tl.store(base + W, o1.to(o_ptr.dtype.element_ty), mask=ok[:, None])


def paged_prefill_attention(q, k_cache, v_cache, block_table, cu_seqlens_q, cu_seqlens_k, max_seqlen_q: int,
                            sm_scale: float, quant: bool = False, inv_smooth=None, block_q: int = 8, block_k: int = 32,
                            num_warps: int = 4):
    """q: [T, H, D] (row stride may exceed H * D, e.g. a view of the qkv output); k/v_cache: [pages, PAGE, HK, D].

    Returns o [T, H * D] in q's dtype, or (int8 o [T, H * D], fp32 scale [T]) with ``quant`` (o * inv_smooth
    quantized per token)."""
    T, H, D = q.shape
    pages, page, HK, _ = k_cache.shape
    assert q.stride(2) == 1 and q.stride(1) == D and k_cache.is_contiguous() and v_cache.is_contiguous()
    assert page % block_k == 0 and HK == 2 and H % HK == 0
    B = cu_seqlens_q.numel() - 1
    if quant:
        o = torch.empty(T, H * D, device=q.device, dtype=torch.int8)
        scale = torch.empty(T, device=q.device, dtype=torch.float32)
    else:
        o = torch.empty(T, H * D, device=q.device, dtype=q.dtype)
        scale = o
    grid = (B, triton.cdiv(max_seqlen_q, block_q))
    _paged_prefill_attn[grid](
        q, k_cache, v_cache, block_table, cu_seqlens_q, cu_seqlens_k, o, scale,
        inv_smooth if inv_smooth is not None else scale,
        q.stride(0), block_table.stride(0), sm_scale,
        H=H, HK=HK, D=D, PAGE=page, BLOCK_Q=block_q, BLOCK_K=block_k, QUANT=quant, HAS_SMOOTH=inv_smooth is not None,
        num_warps=num_warps,
    )
    return (o, scale) if quant else o
