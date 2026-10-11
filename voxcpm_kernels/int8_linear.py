"""W8A8 linear layers: per-token dynamic activation quantization + per-channel int8 weights.

    y = (q(x / s) @ q(W * s)^T) * a_scale[m] * w_scale[n] + bias

``s`` is an optional per-input-channel SmoothQuant factor (1 when no
calibration data is given). The activation quantizer is one Triton pass; the
GEMM is a Triton int8 tensor-core kernel with the dequantization, bias and
bf16 cast fused into its epilogue (no int32 [M, N] round trip).
Tile sizes are picked by a fixed heuristic (no autotuning), so the kernels can
be captured in CUDA graphs right after one warm-up call.
"""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _quant_rows(x_ptr, inv_smooth_ptr, q_ptr, scale_ptr, K, stride_x, HAS_SMOOTH: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, BLOCK_K)
    km = k < K
    x = tl.load(x_ptr + row * stride_x + k, mask=km, other=0.0).to(tl.float32)
    if HAS_SMOOTH:
        x = x * tl.load(inv_smooth_ptr + k, mask=km, other=1.0)
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-6)
    scale = amax / 127.0
    q = tl.extra.cuda.libdevice.rint(x / scale)
    tl.store(q_ptr + row * K + k, q.to(tl.int8), mask=km)
    tl.store(scale_ptr + row, scale)


def quantize_rows(x: torch.Tensor, inv_smooth: torch.Tensor | None = None):
    """x [M, K] (any float dtype, last dim contiguous) -> (int8 [M, K], fp32 scale [M])."""
    M, K = x.shape
    q = torch.empty(M, K, device=x.device, dtype=torch.int8)
    scale = torch.empty(M, device=x.device, dtype=torch.float32)
    block = triton.next_power_of_2(K)
    _quant_rows[(M,)](
        x, inv_smooth if inv_smooth is not None else scale, q, scale, K, x.stride(0),
        HAS_SMOOTH=inv_smooth is not None, BLOCK_K=block, num_warps=8 if block >= 4096 else 4,
    )
    return q, scale


@triton.jit
def _w8a8_gemm(
    a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, bias_ptr,
    M, N, K,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    """C[M, N] = A[M, K] (int8, row-major) @ B[N, K]^T (int8, row-major), dequantized to C's dtype."""
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
        av = tl.load(a, mask=(rm[:, None] < M) & (kk[None, :] < K), other=0)
        bv = tl.load(b, mask=(rn[None, :] < N) & (kk[:, None] < K), other=0)
        acc += tl.dot(av, bv)
        a += BLOCK_K
        b += BLOCK_K
    out = acc.to(tl.float32) * tl.load(a_scale_ptr + rm, mask=rm < M, other=0.0)[:, None]
    out = out * tl.load(b_scale_ptr + rn, mask=rn < N, other=0.0)[None, :]
    if HAS_BIAS:
        out += tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
    cm = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr + rm.to(tl.int64)[:, None] * N + rn[None, :], out.to(c_ptr.dtype.element_ty), mask=cm)


# VOXCPM_KERNELS_LARGE_TILES=0: 128x128x64 tiles for every kernel above 256 rows (A/B, debugging)
_LARGE_TILES = os.environ.get("VOXCPM_KERNELS_LARGE_TILES", "1") != "0"

# Above 256 rows, per kernel, measured at the A10's 150 W power cap (each GEMM in a CUDA graph). Under the
# cap the clock follows energy per op, and bigger tiles move less data through shared memory / L2 per MAC: at
# 1 CTA/SM they run at a higher clock than 128x128x64 tiles at 2 CTAs/SM. M=2030 (LM prefill): o / down 178 / 538 ->
# 161 / 435 us, gate_up (SwiGLU) 910 -> 854 us, qkv (RoPE) 218 -> 213 us. LocDiT (M=2816) / LocEnc (M=1280) decode
# shapes gain 2-12% per GEMM. The residual GEMM takes whichever of two tiles fills the last wave better.
_LARGE = {"plain": [(128, 128, 128, 8, 3)], "rope": [(128, 128, 128, 8, 3)],
          "swiglu": [(256, 128, 64, 8, 3), (128, 128, 128, 8, 3)], "res": [(128, 256, 64, 8, 3), (128, 128, 128, 8, 3)]}
_NUM_SMS = None


def _wave_fill(M: int, N: int, bm: int, bn: int) -> float:
    """Fraction of the SM slots (1 CTA/SM) that the tiles fill over all waves."""
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    tiles = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    return tiles / (triton.cdiv(tiles, _NUM_SMS) * _NUM_SMS)


def _tiles(M: int, N: int, kind: str = "plain"):
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) for an M x N output of kernel ``kind``."""
    if M <= 16:
        return 16, 64, 128, 4, 4
    if M <= 64:
        return 64, 64, 128, 4, 4
    if M <= 256:
        # measured at the 150 W cap: 64x128x128 with 3 stages fits 2 CTAs/SM (48 KB smem;
        # 4 stages need 72 KB -> 1 CTA/SM); the wide gate_up (N >= 8192) has enough tiles for 128x128x64. LM decode
        # gate_up with the SwiGLU epilogue: M=128 / 192 / 256 90 / 141 / 153 -> 83 / 116 / 131 us
        if _LARGE_TILES and kind == "swiglu" and N >= 8192:
            return (128, 128, 128, 8, 3) if M > 192 else (64, 128, 128, 4, 3)
        return (128, 128, 64, 8, 4) if N >= 8192 else (64, 128, 128, 4, 3)
    if _LARGE_TILES and kind in _LARGE:
        cands = [c for c in _LARGE[kind] if N % c[1] == 0]
        if cands:
            return max(cands, key=lambda c: _wave_fill(M, N, c[0], c[1]))  # first one on ties
    return 128, 128, 64, 8, 4


def w8a8_gemm(q: torch.Tensor, a_scale: torch.Tensor, w_q: torch.Tensor, w_scale: torch.Tensor,
              bias: torch.Tensor | None, out_dtype: torch.dtype) -> torch.Tensor:
    M, K = q.shape
    N = w_q.shape[0]
    c = torch.empty(M, N, device=q.device, dtype=out_dtype)
    bm, bn, bk, warps, stages = _tiles(M, N)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _w8a8_gemm[grid](
        q, w_q, c, a_scale, w_scale, bias if bias is not None else w_scale,
        M, N, K, HAS_BIAS=bias is not None,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8, num_warps=warps, num_stages=stages,
    )
    return c


class W8A8Linear:
    """Holds an int8 copy of a linear layer; call with x [..., K] -> [..., N] in x's dtype."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None, smooth: torch.Tensor | None = None):
        w = weight.detach().float()
        if smooth is not None:
            smooth = smooth.detach().float().to(w.device).clamp(min=1e-5)
            w = w * smooth[None, :]
            self.inv_smooth = smooth.reciprocal().contiguous()
        else:
            self.inv_smooth = None
        w_scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        self.w_q = torch.round(w / w_scale[:, None]).clamp(-127, 127).to(torch.int8).contiguous()
        self.w_scale = w_scale.contiguous()
        self.bias = None if bias is None else bias.detach().float().contiguous()
        self.out_features, self.in_features = self.w_q.shape

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if x2.stride(-1) != 1:
            x2 = x2.contiguous()
        q, a_scale = quantize_rows(x2, self.inv_smooth)
        y = w8a8_gemm(q, a_scale, self.w_q, self.w_scale, self.bias, x.dtype)
        return y.view(*shape[:-1], self.out_features)


LINEAR_SUFFIXES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")


def quantize_linears(model: torch.nn.Module, prefixes, min_rows: int = 512, suffixes=LINEAR_SUFFIXES) -> int:
    """Route big-batch calls of matching linear layers through W8A8; small batches keep bf16.

    A module matches when its name starts with one of ``prefixes`` and ends with
    one of ``suffixes``; its ``forward`` must be equivalent to F.linear(x, weight, bias).
    Returns the number of patched modules.
    """
    count = 0
    for name, mod in model.named_modules():
        if not name.startswith(tuple(prefixes)) or not name.endswith(tuple(suffixes)):
            continue
        weight = getattr(mod, "weight", None)
        if weight is None or weight.dim() != 2:
            continue
        q = W8A8Linear(weight, getattr(mod, "bias", None))
        original = mod.forward

        def forward(x, _q=q, _original=original, _min=min_rows):
            return _q(x) if x.numel() // x.shape[-1] >= _min else _original(x)

        mod.forward = forward
        count += 1
    return count
