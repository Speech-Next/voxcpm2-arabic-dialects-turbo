"""Fused MiniCPM4 decoder layer for nanovllm-voxcpm (VoxCPM2 base_lm / residual_lm / LocEnc / LocDiT).

Reference layer (per call ~20 kernels; RoPE alone is ~6 elementwise passes):
    h = rmsnorm(x); q, k, v = split(qkv(h)); q, k = rope(q, k)
    x2 = x + o_proj(attn(q, k, v)); out = x2 + down(silu(gate) * up)  with gate|up = gate_up(rmsnorm(x2))

Here, for W8A8 calls (rows >= min_rows):
    rmsnorm+quant -> int8 qkv GEMM -> in-place RoPE on q|k -> attention -> quant -> int8 o GEMM (+x in the
    epilogue) -> rmsnorm+quant -> int8 gate_up GEMM -> silu*mul+quant -> int8 down GEMM (+x2 in the epilogue)
and for bf16 calls the same graph with cuBLAS GEMMs (RoPE stays fused).
"""

import atexit
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .int8_linear import W8A8Linear, quantize_rows


# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _rope_qk_inplace(qkv_ptr, pos_ptr, cos_ptr, sin_ptr, row_stride, n_rope_heads,
                     HALF: tl.constexpr, BH: tl.constexpr):
    """Rotate heads [0, n_rope_heads) of each row in place: (x1, x2) -> (x1 c - x2 s, x2 c + x1 s)."""
    row = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    j = tl.arange(0, HALF)
    hm = (h < n_rope_heads)[:, None]
    pos = tl.load(pos_ptr + row)
    c = tl.load(cos_ptr + pos * (2 * HALF) + j)[None, :]
    s = tl.load(sin_ptr + pos * (2 * HALF) + j)[None, :]
    base = qkv_ptr + row * row_stride + h[:, None] * (2 * HALF) + j[None, :]
    x1 = tl.load(base, mask=hm, other=0.0).to(tl.float32)
    x2 = tl.load(base + HALF, mask=hm, other=0.0).to(tl.float32)
    tl.store(base, (x1 * c - x2 * s).to(qkv_ptr.dtype.element_ty), mask=hm)
    tl.store(base + HALF, (x2 * c + x1 * s).to(qkv_ptr.dtype.element_ty), mask=hm)


def rope_qk_inplace(qkv: torch.Tensor, positions: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                    n_rope_heads: int, head_dim: int) -> None:
    """qkv [rows, width] (q heads then k heads first); cos/sin [max_pos, head_dim] fp32 with equal halves."""
    rows = qkv.shape[0]
    bh = 4
    _rope_qk_inplace[(rows, triton.cdiv(n_rope_heads, bh))](
        qkv, positions, cos, sin, qkv.stride(0), n_rope_heads, HALF=head_dim // 2, BH=bh, num_warps=2,
    )


# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _rmsnorm_quant(x_ptr, w_ptr, q_ptr, scale_ptr, K, stride_x, eps, BLOCK_K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, BLOCK_K)
    km = k < K
    x = tl.load(x_ptr + row * stride_x + k, mask=km, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / K + eps)
    y = x * rstd * tl.load(w_ptr + k, mask=km, other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-6) / 127.0
    tl.store(q_ptr + row * K + k, tl.extra.cuda.libdevice.rint(y / scale).to(tl.int8), mask=km)
    tl.store(scale_ptr + row, scale)


def rmsnorm_quant(x: torch.Tensor, weight: torch.Tensor, eps: float):
    M, K = x.shape
    q = torch.empty(M, K, device=x.device, dtype=torch.int8)
    scale = torch.empty(M, device=x.device, dtype=torch.float32)
    block = triton.next_power_of_2(K)
    _rmsnorm_quant[(M,)](x, weight, q, scale, K, x.stride(0), eps, BLOCK_K=block, num_warps=8 if block >= 4096 else 4)
    return q, scale


@triton.jit
def _silu_mul_quant(x_ptr, inv_smooth_ptr, q_ptr, scale_ptr, I, stride_x, HAS_SMOOTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, BLOCK)
    km = k < I
    g = tl.load(x_ptr + row * stride_x + k, mask=km, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + row * stride_x + I + k, mask=km, other=0.0).to(tl.float32)
    a = g / (1.0 + tl.exp(-g)) * u
    if HAS_SMOOTH:
        a = a * tl.load(inv_smooth_ptr + k, mask=km, other=1.0)
    scale = tl.maximum(tl.max(tl.abs(a), axis=0), 1e-6) / 127.0
    tl.store(q_ptr + row * I + k, tl.extra.cuda.libdevice.rint(a / scale).to(tl.int8), mask=km)
    tl.store(scale_ptr + row, scale)


def silu_mul_quant(x: torch.Tensor, inv_smooth: torch.Tensor | None = None):
    """x [M, 2I] = gate | up -> int8 silu(gate) * up / smooth [M, I] + per-row scale."""
    M, two_i = x.shape
    i = two_i // 2
    q = torch.empty(M, i, device=x.device, dtype=torch.int8)
    scale = torch.empty(M, device=x.device, dtype=torch.float32)
    block = triton.next_power_of_2(i)
    _silu_mul_quant[(M,)](
        x, inv_smooth if inv_smooth is not None else scale, q, scale, i, x.stride(0),
        HAS_SMOOTH=inv_smooth is not None, BLOCK=block, num_warps=8 if block >= 4096 else 4,
    )
    return q, scale


# ---------------------------------------------------------------------------------------------------------------
@triton.jit
def _res_epilogue(acc, rm, rn, a_scale, b_scale_ptr, res_ptr, c_ptr, M, N):
    out = acc.to(tl.float32) * a_scale[:, None] * tl.load(b_scale_ptr + rn, mask=rn < N, other=0.0)[None, :]
    cm = (rm[:, None] < M) & (rn[None, :] < N)
    offs = rm.to(tl.int64)[:, None] * N + rn[None, :]
    out += tl.load(res_ptr + offs, mask=cm, other=0.0).to(tl.float32)
    tl.store(c_ptr + offs, out.to(c_ptr.dtype.element_ty), mask=cm)


@triton.jit
def _w8a8_gemm_res(
    a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, res_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
    SPLIT_EPILOGUE: tl.constexpr,
):
    """C = dequant(A @ B^T) + RES  (int8 A [M, K], int8 B [N, K], RES/C [M, N]).

    SPLIT_EPILOGUE: dequantize, add RES and store the tile in two N-halves. The fp32 tile plus the RES tile no
    longer have to be live at once, so the kernel fits 128 registers (2 CTAs/SM for 128x128 tiles, instead of 154
    registers and 1 CTA/SM)."""
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group = GROUP_M * num_n
    first_m = (pid // group) * GROUP_M
    gm = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % gm
    pid_n = (pid % group) // gm
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a = a_ptr + rm.to(tl.int64)[:, None] * K + rk[None, :]
    b = b_ptr + rn.to(tl.int64)[None, :] * K + rk[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kk = k * BLOCK_K + rk
        acc += tl.dot(tl.load(a, mask=(rm[:, None] < M) & (kk[None, :] < K), other=0),
                      tl.load(b, mask=(rn[None, :] < N) & (kk[:, None] < K), other=0))
        a += BLOCK_K
        b += BLOCK_K
    a_scale = tl.load(a_scale_ptr + rm, mask=rm < M, other=0.0)
    if SPLIT_EPILOGUE:
        HN: tl.constexpr = BLOCK_N // 2
        acc0, acc1 = tl.split(tl.permute(tl.reshape(acc, (BLOCK_M, 2, HN)), (0, 2, 1)))
        rn0 = pid_n * BLOCK_N + tl.arange(0, HN)
        _res_epilogue(acc0, rm, rn0, a_scale, b_scale_ptr, res_ptr, c_ptr, M, N)
        _res_epilogue(acc1, rm, rn0 + HN, a_scale, b_scale_ptr, res_ptr, c_ptr, M, N)
    else:
        _res_epilogue(acc, rm, rn, a_scale, b_scale_ptr, res_ptr, c_ptr, M, N)


# VOXCPM_KERNELS_RES_SPLIT=0: single-pass epilogue in the residual GEMMs (A/B, debugging)
_RES_SPLIT = os.environ.get("VOXCPM_KERNELS_RES_SPLIT", "1") != "0"


def _gemm(q, s, lin: W8A8Linear, out_dtype, residual=None):
    from .int8_linear import _tiles, w8a8_gemm

    if residual is None:
        return w8a8_gemm(q, s, lin.w_q, lin.w_scale, lin.bias, out_dtype)
    assert lin.bias is None
    M, K = q.shape
    N = lin.out_features
    c = torch.empty(M, N, device=q.device, dtype=out_dtype)
    bm, bn, bk, warps, stages = _tiles(M, N, "res")
    _w8a8_gemm_res[(triton.cdiv(M, bm) * triton.cdiv(N, bn),)](
        q, lin.w_q, c, s, lin.w_scale, residual.contiguous(), M, N, K,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, SPLIT_EPILOGUE=_RES_SPLIT and bm * bn >= 128 * 128,
        num_warps=warps, num_stages=stages,
    )
    return c


# ---------------------------------------------------------------------------------------------------------------
# SwiGLU in the gate_up epilogue. The int8 gate_up weight is stored with gate and up rows interleaved (row 2j =
# gate_j, row 2j+1 = up_j), so each N tile holds matching gate/up columns and the epilogue writes
# h = silu(g) * u [M, I] directly (fp32 math, one bf16 rounding), plus the per-(row, tile) max |h|. A second, 2-D
# tiled pass reduces those partials to the per-row scale and quantizes h for the down GEMM. Versus gate_up ->
# bf16 [M, 2I] -> silu_mul_quant, the [M, 2I] round trip becomes [M, I], and the quant pass needs no row reduction.
@triton.jit
def _w8a8_gemm_swiglu(
    a_ptr, b_ptr, h_ptr, amax_ptr, a_scale_ptr, b_scale_ptr,
    M, N, K, NT,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group = GROUP_M * num_n
    first_m = (pid // group) * GROUP_M
    gm = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % gm
    pid_n = (pid % group) // gm
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a = a_ptr + rm.to(tl.int64)[:, None] * K + rk[None, :]
    b = b_ptr + rn.to(tl.int64)[None, :] * K + rk[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kk = k * BLOCK_K + rk
        acc += tl.dot(tl.load(a, mask=(rm[:, None] < M) & (kk[None, :] < K), other=0),
                      tl.load(b, mask=(rn[None, :] < N) & (kk[:, None] < K), other=0))
        a += BLOCK_K
        b += BLOCK_K
    out = acc.to(tl.float32) * tl.load(a_scale_ptr + rm, mask=rm < M, other=0.0)[:, None]
    out = out * tl.load(b_scale_ptr + rn, mask=rn < N, other=0.0)[None, :]
    g, u = tl.split(tl.reshape(out, (BLOCK_M, BLOCK_N // 2, 2)))
    h = (g / (1.0 + tl.exp(-g)) * u).to(h_ptr.dtype.element_ty)
    HN: tl.constexpr = BLOCK_N // 2
    rh = pid_n * HN + tl.arange(0, HN)
    half = N // 2
    rows = rm.to(tl.int64)
    tl.store(h_ptr + rows[:, None] * half + rh[None, :], h, mask=(rm[:, None] < M) & (rh[None, :] < half))
    # max |h| of the bf16-rounded values the quant pass reads, so |h / scale| <= 127 exactly
    tl.store(amax_ptr + rows * NT + pid_n, tl.max(tl.abs(h.to(tl.float32)), axis=1), mask=rm < M)


@triton.jit
def _quant_rows_partials(h_ptr, amax_ptr, q_ptr, scale_ptr, M, I, NT,
                         NT_P2: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    t = tl.arange(0, NT_P2)
    rmask = r < M
    rows = r.to(tl.int64)
    parts = tl.load(amax_ptr + rows[:, None] * NT + t[None, :], mask=rmask[:, None] & (t[None, :] < NT), other=0.0)
    scale = tl.maximum(tl.max(parts, axis=1), 1e-6) / 127.0
    m = rmask[:, None] & (c[None, :] < I)
    x = tl.load(h_ptr + rows[:, None] * I + c[None, :], mask=m, other=0.0).to(tl.float32)
    tl.store(q_ptr + rows[:, None] * I + c[None, :], tl.extra.cuda.libdevice.rint(x / scale[:, None]).to(tl.int8), mask=m)
    tl.store(scale_ptr + r, scale, mask=rmask & (tl.program_id(1) == 0))


def swiglu_gemm_quant(q, s, lin: W8A8Linear, out_dtype):
    """int8 x [M, K] (+ row scales) -> int8 silu(gate) * up [M, I] + row scales, for a gate_up layer prepared by
    ``swiglu_weights`` (interleaved rows, down's SmoothQuant 1/s folded into the up rows' scales)."""
    from .int8_linear import _tiles

    assert lin.bias is None
    M, K = q.shape
    N = lin.out_features
    half = N // 2
    bm, bn, bk, warps, stages = _tiles(M, N, "swiglu")
    nt = triton.cdiv(N, bn)
    h = torch.empty(M, half, device=q.device, dtype=out_dtype)
    amax = torch.empty(M, nt, device=q.device, dtype=torch.float32)
    _w8a8_gemm_swiglu[(triton.cdiv(M, bm) * nt,)](
        q, lin.w_q, h, amax, s, lin.w_scale, M, N, K, nt,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, num_warps=warps, num_stages=stages,
    )
    hq = torch.empty(M, half, device=q.device, dtype=torch.int8)
    scale = torch.empty(M, device=q.device, dtype=torch.float32)
    br, bc = 8, 1024
    _quant_rows_partials[(triton.cdiv(M, br), triton.cdiv(half, bc))](
        h, amax, hq, scale, M, half, nt, NT_P2=triton.next_power_of_2(nt), BLOCK_R=br, BLOCK_C=bc, num_warps=4,
    )
    return hq, scale


# ---------------------------------------------------------------------------------------------------------------
# RoPE in the qkv epilogue. With BLOCK_N == head_dim every N tile of the qkv output is exactly one head. The int8
# qkv weight rows are permuted within each head (``rope_weights``) so that the rotate-half pair (j, j + head_dim/2)
# comes out of the GEMM as adjacent columns (2j, 2j+1); the epilogue splits them, rotates q/k heads in fp32 (one
# bf16 rounding instead of two) and stores the halves back at j and j + head_dim/2, so the qkv layout in memory is
# unchanged. The separate in-place RoPE pass over q|k goes away.
@triton.jit
def _w8a8_gemm_qkv_rope(
    a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, bias_ptr, pos_ptr, cos_ptr, sin_ptr, add_ptr,
    M, N, K, n_rope_heads,
    HAS_BIAS: tl.constexpr, HAS_ADD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group = GROUP_M * num_n
    first_m = (pid // group) * GROUP_M
    gm = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % gm
    pid_n = (pid % group) // gm
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a = a_ptr + rm.to(tl.int64)[:, None] * K + rk[None, :]
    b = b_ptr + rn.to(tl.int64)[None, :] * K + rk[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kk = k * BLOCK_K + rk
        acc += tl.dot(tl.load(a, mask=(rm[:, None] < M) & (kk[None, :] < K), other=0),
                      tl.load(b, mask=(rn[None, :] < N) & (kk[:, None] < K), other=0))
        a += BLOCK_K
        b += BLOCK_K
    out = acc.to(tl.float32) * tl.load(a_scale_ptr + rm, mask=rm < M, other=0.0)[:, None]
    out = out * tl.load(b_scale_ptr + rn)[None, :]
    if HAS_BIAS:
        out += tl.load(bias_ptr + rn).to(tl.float32)[None, :]
    HALF: tl.constexpr = BLOCK_N // 2
    x1, x2 = tl.split(tl.reshape(out, (BLOCK_M, HALF, 2)))  # columns (2j, 2j+1) = head dims (j, j + HALF)
    j = tl.arange(0, HALF)
    rmask = rm < M
    offs = rm.to(tl.int64)[:, None] * N + pid_n * BLOCK_N + j[None, :]
    if HAS_ADD:  # e.g. the LoRA term: un-permuted qkv layout, added before RoPE like the base GEMM's output
        x1 += tl.load(add_ptr + offs, mask=rmask[:, None], other=0.0).to(tl.float32)
        x2 += tl.load(add_ptr + offs + HALF, mask=rmask[:, None], other=0.0).to(tl.float32)
    if pid_n < n_rope_heads:
        pos = tl.load(pos_ptr + rm, mask=rmask, other=0).to(tl.int64)
        cs = pos[:, None] * BLOCK_N + j[None, :]  # cos/sin rows: [head_dim] fp32 with equal halves
        c = tl.load(cos_ptr + cs)
        sn = tl.load(sin_ptr + cs)
        x1, x2 = x1 * c - x2 * sn, x2 * c + x1 * sn
    tl.store(c_ptr + offs, x1.to(c_ptr.dtype.element_ty), mask=rmask[:, None])
    tl.store(c_ptr + offs + HALF, x2.to(c_ptr.dtype.element_ty), mask=rmask[:, None])


# VOXCPM_KERNELS_SHORT_ATTN=0: FA2 + a separate quantize pass for LocDiT / LocEnc attention and short causal prefill
# sequences (A/B, debugging)
_SHORT_ATTN = os.environ.get("VOXCPM_KERNELS_SHORT_ATTN", "1") != "0"
# VOXCPM_KERNELS_PREFILL_ATTN=0: FA2 + a separate quantize pass for paged prefill attention (A/B, debugging)
_PREFILL_ATTN = os.environ.get("VOXCPM_KERNELS_PREFILL_ATTN", "1") != "0"
# VOXCPM_KERNELS_LORA_MERGE=0: LoRA terms through the LoRA kernels (lora_delta_) instead of per-adapter merged int8
# weights run by the segmented GEMMs (seg_gemm.py)
_LORA_MERGE = os.environ.get("VOXCPM_KERNELS_LORA_MERGE", "1") != "0"
# VOXCPM_KERNELS_ROPE_EPILOGUE=0: qkv GEMM, then the separate in-place RoPE pass (A/B, debugging)
_ROPE_EPILOGUE = os.environ.get("VOXCPM_KERNELS_ROPE_EPILOGUE", "1") != "0"


def rope_weights(qkv: W8A8Linear, head_dim: int) -> W8A8Linear:
    """A qkv W8A8Linear for ``qkv_rope_gemm``: within every head, rows (j, j + head_dim/2) become adjacent."""
    half = head_dim // 2
    n = qkv.out_features
    order = (torch.arange(n, device=qkv.w_q.device).view(-1, 2, half).transpose(1, 2).reshape(-1))
    out = W8A8Linear.__new__(W8A8Linear)
    out.w_q, out.w_scale = qkv.w_q[order].contiguous(), qkv.w_scale[order].contiguous()
    out.bias = None if qkv.bias is None else qkv.bias[order].contiguous()
    out.inv_smooth = qkv.inv_smooth
    out.out_features, out.in_features = qkv.out_features, qkv.in_features
    return out


def qkv_rope_gemm(q, s, lin: W8A8Linear, out_dtype, positions, cos, sin, n_rope_heads: int, addend=None):
    """int8 qkv GEMM (weights from ``rope_weights``) with RoPE on the first ``n_rope_heads`` heads in the epilogue;
    ``addend`` [M, N] (qkv layout, e.g. the LoRA term) is added before the rotation."""
    from .int8_linear import _tiles

    M, K = q.shape
    N = lin.out_features
    head_dim = cos.shape[1]
    bm, bn, bk, warps, stages = _tiles(max(M, 65), N, "rope")  # every config for M > 64 has BLOCK_N = 128
    assert bn == head_dim and N % bn == 0
    c = torch.empty(M, N, device=q.device, dtype=out_dtype)
    _w8a8_gemm_qkv_rope[(triton.cdiv(M, bm) * (N // bn),)](
        q, lin.w_q, c, s, lin.w_scale, lin.bias if lin.bias is not None else lin.w_scale, positions, cos, sin,
        addend if addend is not None else c, M, N, K, n_rope_heads, HAS_BIAS=lin.bias is not None,
        HAS_ADD=addend is not None,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, num_warps=warps, num_stages=stages,
        # 128x128x64 tiles: 150 registers -> 1 CTA/SM; capped at 128 (2 spilled) the kernel runs at 2 CTAs/SM and is
        # as fast as the plain qkv GEMM at the A10 power cap (M=2024: 234 vs 231 us; GEMM + RoPE pass: 272 us)
        **({"maxnreg": 128} if (bm, bn, bk, warps) == (128, 128, 64, 8) else {}),
    )
    return c


def swiglu_weights(gate_up: W8A8Linear, down: W8A8Linear) -> W8A8Linear:
    """A gate_up W8A8Linear for ``swiglu_gemm_quant``: rows interleaved (gate_j, up_j) and down's SmoothQuant
    1/s folded into the up rows' scales (silu(g_j) * u_j / s_j, exact)."""
    half = gate_up.out_features // 2
    scale = gate_up.w_scale.clone()
    if down.inv_smooth is not None:
        scale[half:] *= down.inv_smooth
    order = torch.arange(2 * half, device=gate_up.w_q.device).view(2, half).t().reshape(-1)
    out = W8A8Linear.__new__(W8A8Linear)
    out.w_q, out.w_scale = gate_up.w_q[order].contiguous(), scale[order].contiguous()
    out.bias, out.inv_smooth = None, None
    out.out_features, out.in_features = gate_up.out_features, gate_up.in_features
    return out


# ---------------------------------------------------------------------------------------------------------------
# --- SmoothQuant -------------------------------------------------------------------------------------------------
# VOXCPM_KERNELS_CALIBRATE=<file>: bf16 everywhere, record per-channel |activation| maxima of the four linear inputs
#   of every fused layer (inside CUDA graphs too) and save them at process exit.
# VOXCPM_KERNELS_SMOOTH=<file> (+ VOXCPM_KERNELS_SMOOTH_ALPHA, default 0.7): per-input-channel factors
#   s = max|X|^a / max|W|^(1-a); 1/s is folded into the RMSNorm weight (qkv, gate_up) or applied by the
#   quantizer (o, down), and s into the int8 weights. Default: smooth_stats.pt next to this file (VoxCPM2's
#   calibration; layers without stats in it are not smoothed); "0" = no smoothing.
_CALIBRATE = os.environ.get("VOXCPM_KERNELS_CALIBRATE")
_SMOOTH = os.environ.get("VOXCPM_KERNELS_SMOOTH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "smooth_stats.pt"))
if _SMOOTH in ("", "0") or not os.path.exists(_SMOOTH):
    _SMOOTH = None
_SMOOTH_ALPHA = float(os.environ.get("VOXCPM_KERNELS_SMOOTH_ALPHA", "0.7"))
_CALIB_STATS: dict[str, torch.Tensor] = {}
# VOXCPM_KERNELS_SWIGLU=0: gate_up -> bf16 -> silu_mul_quant instead of the SwiGLU epilogue (A/B, debugging)
_SWIGLU = os.environ.get("VOXCPM_KERNELS_SWIGLU", "1") != "0"
_SMOOTH_STATS: dict | None = None


def _save_calibration() -> None:
    torch.save({k: v.cpu() for k, v in _CALIB_STATS.items()}, _CALIBRATE)


if _CALIBRATE:
    atexit.register(_save_calibration)


def _smooth_factor(act_max: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    w_max = weight.detach().float().abs().amax(dim=0).clamp(min=1e-5)
    a_max = act_max.to(weight.device).float().clamp(min=1e-5)
    return (a_max.pow(_SMOOTH_ALPHA) / w_max.pow(1 - _SMOOTH_ALPHA)).clamp(min=1e-3, max=1e3)


class FusedCpm4Layer:
    """Replacement forward for one nanovllm Cpm4DecoderLayer."""

    def __init__(self, layer, quantize: bool, min_rows: int, name: str = ""):
        attn = layer.self_attn
        if attn.q_norm is not None or attn.k_norm is not None:
            raise NotImplementedError("qk-norm layers are not supported")
        self.layer = layer
        self.attn = attn
        self.min_rows = min_rows
        self.eps_in = layer.input_layernorm.eps
        self.eps_post = layer.post_attention_layernorm.eps
        self.rope = attn.rotary_emb
        if self.rope is not None:
            cos, sin = self.rope.cos_cached.float(), self.rope.sin_cached.float()
            half = attn.head_dim // 2
            assert torch.equal(cos[:, :half], cos[:, half:]) and torch.equal(sin[:, :half], sin[:, half:])
            self.cos, self.sin = cos.contiguous(), sin.contiguous()
        mlp = layer.mlp
        if any(getattr(m, "supports_lora", False) for m in (mlp.gate_up_proj, mlp.down_proj)):
            raise NotImplementedError("LoRA on the MLP projections is not supported by the fused layers")
        self.name = name
        self.calib = None
        if _CALIBRATE:
            dev = attn.qkv_proj.weight.device
            self.calib = {
                key: _CALIB_STATS.setdefault(f"{name}.{key}", torch.zeros(dim, device=dev, dtype=torch.float32))
                for key, dim in (("qkv", attn.qkv_proj.weight.shape[1]), ("o", attn.o_proj.weight.shape[1]),
                                 ("gate_up", mlp.gate_up_proj.weight.shape[1]), ("down", mlp.down_proj.weight.shape[1]))
            }
            quantize = False
        self.q8 = None
        self.ln_in_q = layer.input_layernorm.weight
        self.ln_post_q = layer.post_attention_layernorm.weight
        if quantize:
            smooth = {}
            global _SMOOTH_STATS
            if _SMOOTH:
                if _SMOOTH_STATS is None:
                    _SMOOTH_STATS = torch.load(_SMOOTH, map_location="cpu", weights_only=True)
                weights = {"qkv": attn.qkv_proj.weight, "o": attn.o_proj.weight,
                           "gate_up": mlp.gate_up_proj.weight, "down": mlp.down_proj.weight}
                smooth = {k: _smooth_factor(_SMOOTH_STATS[f"{name}.{k}"], w) for k, w in weights.items()
                          if f"{name}.{k}" in _SMOOTH_STATS}
            self.q8 = {
                "qkv": W8A8Linear(attn.qkv_proj.weight, attn.qkv_proj.bias, smooth.get("qkv")),
                "o": W8A8Linear(attn.o_proj.weight, attn.o_proj.bias, smooth.get("o")),
                "gate_up": W8A8Linear(mlp.gate_up_proj.weight, None, smooth.get("gate_up")),
                "down": W8A8Linear(mlp.down_proj.weight, None, smooth.get("down")),
            }
            # the norm-fed linears take 1/s through the RMSNorm weight instead of the quantizer
            for key, ln_attr in (("qkv", "ln_in_q"), ("gate_up", "ln_post_q")):
                lin = self.q8[key]
                if lin.inv_smooth is not None:
                    setattr(self, ln_attr, (getattr(self, ln_attr).float() * lin.inv_smooth).contiguous())
                    lin.inv_smooth = None
            if _SWIGLU:
                self.q8["gate_up"] = swiglu_weights(self.q8["gate_up"], self.q8["down"])
            if _ROPE_EPILOGUE and self.rope is not None and attn.head_dim == 128:
                self.q8["qkv"] = rope_weights(self.q8["qkv"], attn.head_dim)
        self.swiglu = self.q8 is not None and _SWIGLU
        self.qkv_rope = self.q8 is not None and _ROPE_EPILOGUE and self.rope is not None and attn.head_dim == 128
        # LoRA adapters merged into int8 q/k/v/o weights, one per slot ([0] = the base model): a batch mixing
        # adapters runs segmented GEMMs (rows grouped by adapter) and no LoRA kernels. refresh_lora(slot) rebuilds a
        # slot when the runner loads an adapter into it.
        self.stacks = None
        if self.q8 is not None and _LORA_MERGE and getattr(attn.qkv_proj, "supports_lora", False) \
                and getattr(attn.o_proj, "supports_lora", False):
            n = attn.qkv_proj.max_loras + 1
            self._smooth = smooth
            self.stacks = {}
            for key in ("qkv", "o"):
                lin = self.q8[key]
                w = torch.empty((n,) + tuple(lin.w_q.shape), dtype=torch.int8, device=lin.w_q.device)
                sc = torch.empty(n, lin.out_features, dtype=torch.float32, device=lin.w_q.device)
                w[:] = lin.w_q
                sc[:] = lin.w_scale
                self.stacks[key] = (w, sc)
            for slot in range(n - 1):  # adapters loaded before fusing
                if attn.qkv_proj._effective_lora_rank_values[slot] or attn.o_proj._effective_lora_rank_values[slot]:
                    self.refresh_lora(slot)

    @torch.no_grad()
    def refresh_lora(self, slot: int) -> None:
        """Rebuild slot ``slot``'s merged weights from the LoRA modules' current A / B (B holds the scaling)."""
        if self.stacks is None:
            return
        attn = self.attn
        qkv, o = attn.qkv_proj, attn.o_proj
        rows = []
        for t, size in (("q", qkv.q_size), ("k", qkv.kv_size), ("v", qkv.kv_size)):
            if t in qkv.target_to_index:
                a = qkv.lora_A[qkv.target_to_index[t], slot].float()
                rows.append(getattr(qkv, f"lora_B_{t}")[slot].float() @ a)
            else:
                rows.append(torch.zeros(size, qkv.weight.shape[1], device=qkv.weight.device))
        lin = W8A8Linear(qkv.weight.float() + torch.cat(rows), qkv.bias, self._smooth.get("qkv"))
        if self.qkv_rope:
            lin = rope_weights(lin, attn.head_dim)
        w, sc = self.stacks["qkv"]
        w[slot + 1].copy_(lin.w_q)
        sc[slot + 1].copy_(lin.w_scale)
        lin = W8A8Linear(o.weight.float() + o.lora_B[slot].float() @ o.lora_A[slot].float(), o.bias,
                         self._smooth.get("o"))
        w, sc = self.stacks["o"]
        w[slot + 1].copy_(lin.w_q)
        sc[slot + 1].copy_(lin.w_scale)

    def _seg(self, rows: int):
        """Segmented-GEMM tiles for this call when the batch has LoRA slots and merged weights exist, else None."""
        if self.stacks is None:
            return None
        from nanovllm_voxcpm.utils.context import get_lora_context

        from .seg_gemm import tiles_for

        domain = self.attn.qkv_proj.lora_domain
        ctx = get_lora_context(domain)
        capturing = torch.cuda.is_current_stream_capturing()
        if ctx.no_lora_flag or (ctx.token_to_slot is None and ctx.host_token_to_slot is None):
            return None
        n_w = self.stacks["qkv"][0].shape[0]
        if capturing:
            return tiles_for(None, rows, domain, n_w)
        cache = ctx.__dict__.setdefault("_seg_cache", {})
        slots = ctx.host_token_to_slot
        if slots is None:
            slots = cache.get("slots")
            if slots is None:
                slots = cache["slots"] = ctx.token_to_slot.tolist()
        return tiles_for(slots, rows, domain, n_w, cache)

    @staticmethod
    def _lora_on(mod, rows: torch.Tensor) -> bool:
        """True when an adapter of the current batch touches the LoRA module ``mod`` (rows: its input rows). The
        int8 path never builds the module's bf16 input/output, so it adds the LoRA term itself (``lora_delta_``);
        the bf16 path calls the module, which adds it."""
        return getattr(mod, "supports_lora", False) and mod._resolve_token_slots(rows) is not None

    @staticmethod
    def _lora_context_on(mod) -> bool:
        """True when the batch has LoRA token slots in ``mod``'s domain (even if no adapter touches ``mod``): the
        module then requires one slot per input row, so it cannot run on a subset of the rows."""
        if not getattr(mod, "supports_lora", False):
            return False
        from nanovllm_voxcpm.utils.context import get_lora_context

        ctx = get_lora_context(mod.lora_domain)
        return ctx.token_to_slot is not None and not ctx.no_lora_flag and ctx.token_to_slot.numel() > 0

    def _qkv_lora(self, x: torch.Tensor) -> torch.Tensor | None:
        """The LoRA term of qkv(rmsnorm(x)) [rows, q+2kv] (bf16 norm with the unsmoothed weight), None = no adapter."""
        mod = self.attn.qkv_proj
        if not self._lora_on(mod, x):
            return None
        h = F.rms_norm(x, (x.shape[-1],), self.layer.input_layernorm.weight, self.eps_in)
        d = torch.zeros(x.shape[0], mod.weight.shape[0], device=x.device, dtype=x.dtype)
        mod.lora_delta_(h, d)
        return d

    def _qkv(self, x: torch.Tensor, pos_rows, quant: bool, dtype, seg=None) -> torch.Tensor:
        """qkv(rmsnorm(x)) with RoPE applied to the q and k heads (pos_rows: per-row positions, None = no RoPE).
        seg: segmented-GEMM tiles (merged adapter weights per row segment)."""
        attn = self.attn
        n_rope = attn.num_heads + attn.num_kv_heads
        if seg is not None:
            from .seg_gemm import seg_gemm, seg_qkv_rope_gemm

            xq = rmsnorm_quant(x, self.ln_in_q, self.eps_in)
            w, sc = self.stacks["qkv"]
            if self.qkv_rope:
                return seg_qkv_rope_gemm(*xq, w, sc, dtype, pos_rows, self.cos, self.sin, n_rope, *seg)
            qkv = seg_gemm(*xq, w, sc, self.q8["qkv"].bias, dtype, *seg)
            if pos_rows is not None:
                rope_qk_inplace(qkv, pos_rows, self.cos, self.sin, n_rope, attn.head_dim)
            return qkv
        if quant:
            xq = rmsnorm_quant(x, self.ln_in_q, self.eps_in)
            lora = self._qkv_lora(x)
            if self.qkv_rope:
                return qkv_rope_gemm(*xq, self.q8["qkv"], dtype, pos_rows, self.cos, self.sin, n_rope, addend=lora)
            qkv = _gemm(*xq, self.q8["qkv"], dtype)
            if lora is not None:
                qkv += lora
        else:
            h = F.rms_norm(x, (x.shape[-1],), self.layer.input_layernorm.weight, self.eps_in)
            if self.calib is not None:
                torch.maximum(self.calib["qkv"], h.abs().amax(0).float(), out=self.calib["qkv"])
            qkv = attn.qkv_proj(h)
        if pos_rows is not None:
            rope_qk_inplace(qkv, pos_rows, self.cos, self.sin, n_rope, attn.head_dim)
        return qkv

    def _paged_prefill_q(self, q, k, v, quant: bool = True):
        """Causal prefill reading K/V through block tables (eager steps): store this step's K/V, then the Triton
        paged attention writes o already quantized for o_proj (int8 + per-token scale), or bf16 o without ``quant``
        (an adapter needs o_proj's input). None = not applicable."""
        attn = self.attn
        if not _PREFILL_ATTN or not attn.is_causal or torch.cuda.is_current_stream_capturing():
            return None
        from nanovllm_voxcpm.utils.context import get_context

        ctx = get_context()
        inner = attn.attn
        if not ctx.is_prefill or ctx.block_tables is None or not inner.k_cache.numel() or attn.num_kv_heads != 2 \
                or inner.k_cache.shape[1] % 32:
            return None
        from nanovllm_voxcpm.layers.attention import store_kvcache

        from .prefill_attn import paged_prefill_attention

        store_kvcache(k.view(-1, attn.num_kv_heads, attn.head_dim), v.view(-1, attn.num_kv_heads, attn.head_dim),
                      inner.k_cache, inner.v_cache, ctx.slot_mapping)
        return paged_prefill_attention(q.view(-1, attn.num_heads, attn.head_dim), inner.k_cache, inner.v_cache,
                                       ctx.block_tables, ctx.cu_seqlens_q, ctx.cu_seqlens_k, ctx.max_seqlen_q,
                                       inner.scale, quant=quant, inv_smooth=self.q8["o"].inv_smooth if quant else None)

    def _short_prefill_q(self, q, k, v, quant: bool = True):
        """Causal prefill of short sequences over their own tokens (no cached prefix), e.g. a burst's head-start step
        (200 sequences of ~7 tokens): this step's K/V stored, then short_causal_attention (one program per sequence,
        FA2 gives each a 128-row query tile) with o already quantized for o_proj, or bf16 o without ``quant``.
        None = not applicable."""
        attn = self.attn
        if not _SHORT_ATTN or not attn.is_causal or attn.num_kv_heads != 2 or attn.head_dim != 128:
            return None
        from nanovllm_voxcpm.utils.context import get_context

        from .short_attn import SHORT_CAUSAL_MAX_LEN, short_causal_attention

        ctx = get_context()
        if not ctx.is_prefill or ctx.block_tables is not None or ctx.cu_seqlens_q is None \
                or not 0 < ctx.max_seqlen_q <= SHORT_CAUSAL_MAX_LEN:
            return None
        inner = attn.attn
        k3, v3 = k.view(-1, attn.num_kv_heads, attn.head_dim), v.view(-1, attn.num_kv_heads, attn.head_dim)
        if inner.k_cache.numel() and inner.v_cache.numel():
            from nanovllm_voxcpm.layers.attention import store_kvcache

            store_kvcache(k3, v3, inner.k_cache, inner.v_cache, ctx.slot_mapping)
        return short_causal_attention(q.view(-1, attn.num_heads, attn.head_dim), k3, v3, ctx.cu_seqlens_q,
                                      ctx.max_seqlen_q, inner.scale, quant=quant,
                                      inv_smooth=self.q8["o"].inv_smooth if quant else None)

    def _mlp_q(self, x2: torch.Tensor, dtype) -> torch.Tensor:
        """x2 + down(silu(gate) * up), gate|up = gate_up(rmsnorm(x2)), all int8 GEMMs."""
        xq = rmsnorm_quant(x2, self.ln_post_q, self.eps_post)
        if self.swiglu:
            hq = swiglu_gemm_quant(*xq, self.q8["gate_up"], dtype)
        else:
            hq = silu_mul_quant(_gemm(*xq, self.q8["gate_up"], dtype), self.q8["down"].inv_smooth)
        return _gemm(*hq, self.q8["down"], dtype, residual=x2)

    def __call__(self, positions: torch.Tensor, hidden: torch.Tensor, residual=None):
        layer, attn = self.layer, self.attn
        shape = hidden.shape
        x = hidden.reshape(-1, shape[-1])
        rows = x.shape[0]
        seg = self._seg(rows)  # adapters in the batch: merged int8 weights per row segment, at any batch size
        use_q = self.q8 is not None and (rows >= self.min_rows or seg is not None)
        dtype = hidden.dtype

        calib = self.calib
        pos_rows = None
        if self.rope is not None:
            pos_rows = positions if attn.is_causal else positions.repeat(shape[0])
        qkv = self._qkv(x, pos_rows, use_q, dtype, seg)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        lora_o = use_q and seg is None and self._lora_on(attn.o_proj, x)  # o_proj's LoRA term needs bf16 o
        oq = self._paged_prefill_q(q, k, v, quant=not lora_o) if use_q else None
        if oq is None and use_q:
            oq = self._short_prefill_q(q, k, v, quant=not lora_o)
        if oq is None and use_q and not attn.is_causal and _SHORT_ATTN and shape[1] <= 16 and attn.num_kv_heads == 2:
            from .short_attn import short_attention

            b = shape[0]
            oq = short_attention(q.view(b, -1, attn.num_heads, attn.head_dim), k.view(b, -1, attn.num_kv_heads, attn.head_dim),
                                 v.view(b, -1, attn.num_kv_heads, attn.head_dim), attn.attn.scale, quant=not lora_o,
                                 inv_smooth=None if lora_o else self.q8["o"].inv_smooth)
        if lora_o and oq is not None:  # the fused attention kernels returned bf16 o
            o, oq = oq, None
        elif oq is not None:
            o = None
        elif attn.is_causal:
            o = attn.attn(q.view(-1, attn.num_heads, attn.head_dim), k.view(-1, attn.num_kv_heads, attn.head_dim),
                          v.view(-1, attn.num_kv_heads, attn.head_dim))
        else:
            b = shape[0]
            o = attn.attn(q.view(b, -1, attn.num_heads, attn.head_dim), k.view(b, -1, attn.num_kv_heads, attn.head_dim),
                          v.view(b, -1, attn.num_kv_heads, attn.head_dim))
        if o is not None:
            o = o.reshape(rows, -1)

        if use_q:
            if oq is None:
                oq = quantize_rows(o, self.q8["o"].inv_smooth)
            if seg is not None:
                from .seg_gemm import seg_gemm

                x2 = seg_gemm(*oq, *self.stacks["o"], None, dtype, *seg, residual=x)
            else:
                x2 = _gemm(*oq, self.q8["o"], dtype, residual=x)
            if lora_o:
                attn.o_proj.lora_delta_(o, x2)
            out = self._mlp_q(x2, dtype)
        elif calib is not None:
            torch.maximum(calib["o"], o.abs().amax(0).float(), out=calib["o"])
            x2 = x + attn.o_proj(o)
            h = F.rms_norm(x2, (x2.shape[-1],), layer.post_attention_layernorm.weight, self.eps_post)
            torch.maximum(calib["gate_up"], h.abs().amax(0).float(), out=calib["gate_up"])
            g, u = layer.mlp.gate_up_proj(h).chunk(2, -1)
            act = F.silu(g) * u
            torch.maximum(calib["down"], act.abs().amax(0).float(), out=calib["down"])
            out = x2 + layer.mlp.down_proj(act)
        else:
            x2 = x + attn.o_proj(o)
            h = F.rms_norm(x2, (x2.shape[-1],), layer.post_attention_layernorm.weight, self.eps_post)
            out = x2 + layer.mlp(h)
        return out.view(shape), x2.view(shape)

    def last_rows(self, positions: torch.Tensor, hidden: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """Like ``__call__``, for a layer whose output is needed only at ``rows`` (the last token of every sequence in
        a prefill step, e.g. the residual LM's last layer): K/V are computed and stored for every row, attention runs
        with one query per sequence, and o_proj, the post-attention norm and the MLP run on ``rows`` only."""
        from nanovllm_voxcpm.utils.context import get_context

        layer, attn = self.layer, self.attn
        x = hidden.reshape(-1, hidden.shape[-1])
        n = rows.numel()
        # n >= rows: e.g. a small prefill graph bucket, padded to more sequences than it has tokens; LoRA slots in
        # the batch cover every row, so o_proj runs on every row too
        if n >= x.shape[0] or self._lora_context_on(attn.o_proj):
            return self(positions, hidden)[0].reshape(x.shape)[rows]
        dtype = hidden.dtype
        qkv = self._qkv(x, positions if self.rope is not None else None,
                        self.q8 is not None and x.shape[0] >= self.min_rows, dtype)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        ctx = get_context()
        saved = ctx.cu_seqlens_q, ctx.max_seqlen_q
        ctx.cu_seqlens_q, ctx.max_seqlen_q = torch.arange(n + 1, dtype=torch.int32, device=x.device), 1
        try:  # the KV store inside uses every row (slot_mapping is unchanged); queries: one per sequence
            o = attn.attn(q[rows].view(-1, attn.num_heads, attn.head_dim), k.view(-1, attn.num_kv_heads, attn.head_dim),
                          v.view(-1, attn.num_kv_heads, attn.head_dim))
        finally:
            ctx.cu_seqlens_q, ctx.max_seqlen_q = saved
        o = o.reshape(n, -1)
        xs = x[rows]
        if self.q8 is not None and n >= self.min_rows:
            x2 = _gemm(*quantize_rows(o, self.q8["o"].inv_smooth), self.q8["o"], dtype, residual=xs)
            return self._mlp_q(x2, dtype)
        x2 = xs + attn.o_proj(o)
        h = F.rms_norm(x2, (x2.shape[-1],), layer.post_attention_layernorm.weight, self.eps_post)
        return x2 + layer.mlp(h)


def fuse_cpm4_layers(model: torch.nn.Module, quant_prefixes, fuse_prefixes, min_rows: int = 512) -> tuple[int, int]:
    """Swap the forward of every Cpm4DecoderLayer under ``fuse_prefixes``; quantize those under ``quant_prefixes``."""
    fused = quantized = 0
    for name, mod in model.named_modules():
        if type(mod).__name__ != "Cpm4DecoderLayer" or not name.startswith(tuple(fuse_prefixes)):
            continue
        quant = name.startswith(tuple(quant_prefixes))
        mod.forward = FusedCpm4Layer(mod, quant, min_rows, name)
        fused += 1
        quantized += int(quant)
    return fused, quantized
