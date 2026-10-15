"""2:4 sparse int8 GEMMs for the LM's FFN (gate_up with the SwiGLU epilogue, down with the residual add).

A W8A8 FFN weight that is 2:4 sparse along K (at most 2 non-zeros in every 4 consecutive input channels of a row, as
after 2:4 pruning) runs on Ampere's sparse tensor cores (mma.sp, cuda/sparse24_gemm.cu). The output is bit-identical to
the dense kernels of fused_layer.py on the same weight: the int32 accumulators are equal, and the epilogues repeat
the dense ones' float operations.

The packed weight (``Sparse24Linear``):
- ``wf`` [N/16, K/64, 32, 16] uint8: for every 16-row tile and 64-wide k block, the compressed values (2 per 4) in
  mma.sp's A-fragment order, 16 bytes per lane (lane = 4 g + t): a0 = row g, compressed columns 4t..4t+3; a1 = row
  g+8, same columns; a2 / a3 = the same rows, columns 16+4t..16+4t+3.
- ``mf`` [N/16, K/64, 32] uint32: lane (g, t)'s metadata, row g + 8 (t & 1), groups 8 (t >> 1) + n of the block in
  nibble n (index of the first kept value in bits 0-1, of the second in bits 2-3, first < second). Layouts measured
  on the A10 (tests/gpu/test_sparse24.py checks them against a CPU reference).
- gate_up: the rows of every 16-row tile are gate channels 8T..8T+7, then the up channels of the same 8 (the SwiGLU
  epilogue finds both in one thread).

VOXCPM_KERNELS_SPARSE24=0: the dense kernels, even for 2:4 weights (A/B, debugging).
"""

import os

import torch

from . import cuda_ext

ENABLED = os.environ.get("VOXCPM_KERNELS_SPARSE24", "1") != "0"
_EXT = "sparse24_gemm"


def available() -> bool:
    return ENABLED and torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0) \
        and cuda_ext.available(_EXT)


def is_24(w: torch.Tensor) -> bool:
    """True when the [N, K] weight has at most 2 non-zeros in every group of 4 consecutive K entries."""
    n, k = w.shape
    return k % 4 == 0 and bool(((w.reshape(n, k // 4, 4) != 0).sum(-1) <= 2).all())


def prune_24(w: torch.Tensor) -> torch.Tensor:
    """The 2 largest |w| of every group of 4 along K kept, the rest zeroed (magnitude pruning; for tests)."""
    n, k = w.shape
    g = w.reshape(n, k // 4, 4)
    keep = g.abs().float().topk(2, dim=-1).indices
    mask = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, keep, True)
    return (g * mask).reshape(n, k)


def compress_24(w_q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """int8 [N, K] (2:4 along K; N % 16 == 0, K % 128 == 0) -> (wf, mf) in the fragment order (module docstring)."""
    n, k = w_q.shape
    assert n % 16 == 0 and k % 128 == 0, (n, k)
    groups = w_q.reshape(n, k // 4, 4)
    nz = groups != 0
    if bool((nz.sum(-1) > 2).any()):
        raise ValueError("weight is not 2:4 sparse along K")
    # the kept pair: the non-zeros, completed with the first zero positions; ascending
    pos = torch.arange(4, device=w_q.device)
    idx = ((~nz).to(torch.int64) * 4 + pos).argsort(-1)[..., :2].sort(-1).values  # [N, G, 2]
    vals = groups.gather(-1, idx).reshape(n, k // 2)  # compressed row: column 2 G + j
    # values: [R, h(g / g+8), g, Kb, q(cols 0-15 / 16-31), t, byte] -> [R, Kb, g, t, q, h, byte]
    wf = vals.view(torch.uint8).reshape(n // 16, 2, 8, k // 64, 2, 4, 4).permute(0, 3, 2, 5, 4, 1, 6)
    wf = wf.contiguous().reshape(n // 16, k // 64, 32, 16)
    # metadata: nibble = i0 | i1 << 2 per (row, group); [R, h, g, Kb, hi, n] -> [R, Kb, g, hi, h, n], t = 2 hi + h
    nib = (idx[..., 0] | (idx[..., 1] << 2)).reshape(n // 16, 2, 8, k // 64, 2, 8).permute(0, 3, 2, 4, 1, 5)
    words = (nib.reshape(n // 16, k // 64, 32, 8) << (4 * torch.arange(8, device=w_q.device))).sum(-1)
    mf = torch.where(words >= 1 << 31, words - (1 << 32), words).to(torch.int32).view(torch.uint8)
    return wf, mf.reshape(n // 16, k // 64, 32 * 4).contiguous()


def decompress_24(wf: torch.Tensor, mf: torch.Tensor) -> torch.Tensor:
    """The dense int8 [N, K] weight of a packed one (inverse of ``compress_24``; for tests)."""
    r, kb = wf.shape[:2]
    n, k = r * 16, kb * 64
    vals = wf.reshape(r, kb, 8, 4, 2, 2, 4).permute(0, 5, 2, 1, 4, 3, 6).reshape(n, k // 2).view(torch.int8)
    words = mf.view(torch.int32).to(torch.int64).bitwise_and(0xFFFFFFFF).reshape(r, kb, 32)
    nib = (words[..., None] >> (4 * torch.arange(8, device=wf.device))) & 0xF  # [R, Kb, 32, 8]
    nib = nib.reshape(r, kb, 8, 2, 2, 8).permute(0, 4, 2, 1, 3, 5).reshape(n, k // 4)
    idx = torch.stack([nib & 3, nib >> 2], -1)  # [N, G, 2]
    out = torch.zeros(n, k // 4, 4, dtype=torch.int8, device=wf.device)
    out.scatter_(-1, idx, vals.reshape(n, k // 4, 2))
    return out.reshape(n, k)


class Sparse24Linear:
    """A W8A8Linear's 2:4 weight packed for mma.sp (``rows``: the dense rows in packed order, e.g. the SwiGLU tiles)."""

    def __init__(self, lin, rows: torch.Tensor | None = None):
        assert lin.bias is None  # (inv_smooth belongs to the quantizer of the input, not to the GEMM)
        w_q, scale = (lin.w_q, lin.w_scale) if rows is None else (lin.w_q[rows], lin.w_scale[rows])
        self.wf, self.mf = compress_24(w_q.contiguous())
        self.w_scale = scale.float().contiguous()
        self.out_features, self.in_features = w_q.shape


def swiglu_rows(n: int, device=None) -> torch.Tensor:
    """Packed row order of an interleaved gate_up weight (row 2j = gate_j, 2j+1 = up_j; fused_layer.swiglu_weights):
    each 16-row tile holds gate_8T..gate_8T+7, then up_8T..up_8T+7."""
    j = torch.arange(n // 2, device=device).view(-1, 8)
    return torch.cat([2 * j, 2 * j + 1], 1).reshape(-1)


# tile configurations of sparse24_gemm.cu: (BO, BT) per id
CONFIGS = {0: (128, 128), 1: (128, 64), 2: (128, 32), 3: (64, 16), 4: (64, 8), 5: (64, 128), 6: (128, 128), 7: (64, 32),
           8: (128, 64), 9: (128, 64), 10: (64, 128), 11: (128, 128), 12: (256, 128), 13: (128, 256)}
SWIGLU_ONLY = {12, 13}  # their residual epilogue's staging does not fit in shared memory


def _cfg(m: int, n: int, kind: str) -> int:
    """Tile configuration by row count, measured on the A10 at base_lm's FFN shapes (tests/gpu/bench_sparse24.py):
    gate_up (SwiGLU, N = 12288, K = 2048) and down (residual, N = 2048, K = 6144)."""
    if kind == "swiglu":
        for rows, cfg in ((8, 4), (16, 3), (32, 7), (64, 2), (128, 1)):
            if m <= rows:
                return cfg
        return 0
    for rows, cfg in ((16, 4), (64, 3), (128, 2), (200, 5), (256, 1)):
        if m <= rows:
            return cfg
    return 0


def res_fma(m: int, n: int) -> bool:
    """Whether the dense residual GEMM (fused_layer._w8a8_gemm_res) adds the residual in one fma at this shape: its
    single-pass epilogue compiles to fma(acc * as, bs, res), the split one to two roundings."""
    from . import fused_layer
    from .int8_linear import _tiles

    bm, bn = _tiles(m, n, "res")[:2]
    return not (fused_layer._RES_SPLIT and bm * bn >= 128 * 128)


def gemm_res(q: torch.Tensor, s: torch.Tensor, lin: Sparse24Linear, out_dtype, residual: torch.Tensor,
             cfg: int | None = None) -> torch.Tensor:
    """out = dequant(q @ W^T) + residual, bit-identical to fused_layer._gemm(..., residual=residual)."""
    assert out_dtype == torch.bfloat16
    m = q.shape[0]
    n = lin.out_features
    out = torch.empty(m, n, device=q.device, dtype=out_dtype)
    cfg = _cfg(m, n, "res") if cfg is None else cfg
    cuda_ext.load(_EXT).sp24_gemm_res(lin.wf, lin.mf, q, s, lin.w_scale, residual.contiguous(), out, cfg,
                                      int(res_fma(m, n)))
    return out


def gemm_swiglu(q: torch.Tensor, s: torch.Tensor, lin: Sparse24Linear, out_dtype, cfg: int | None = None):
    """(h [M, N/2] bf16, amax [M, N tiles] fp32) of the SwiGLU GEMM (rows in ``swiglu_rows`` order)."""
    assert out_dtype == torch.bfloat16
    m = q.shape[0]
    n = lin.out_features
    cfg = _cfg(m, n, "swiglu") if cfg is None else cfg
    h = torch.empty(m, n // 2, device=q.device, dtype=out_dtype)
    amax = torch.empty(m, n // CONFIGS[cfg][0], device=q.device, dtype=torch.float32)
    cuda_ext.load(_EXT).sp24_gemm_swiglu(lin.wf, lin.mf, q, s, lin.w_scale, h, amax, cfg)
    return h, amax


def swiglu_gemm_quant(q: torch.Tensor, s: torch.Tensor, lin: Sparse24Linear, out_dtype, cfg: int | None = None):
    """Like fused_layer.swiglu_gemm_quant: int8 silu(gate) * up [M, I] + row scales, bit-identical."""
    import triton

    from .fused_layer import _quant_rows_partials

    h, amax = gemm_swiglu(q, s, lin, out_dtype, cfg)
    m, half = h.shape
    nt = amax.shape[1]
    hq = torch.empty(m, half, device=q.device, dtype=torch.int8)
    scale = torch.empty(m, device=q.device, dtype=torch.float32)
    br, bc = 8, 1024
    _quant_rows_partials[(triton.cdiv(m, br), triton.cdiv(half, bc))](
        h, amax, hq, scale, m, half, nt, NT_P2=triton.next_power_of_2(nt), BLOCK_R=br, BLOCK_C=bc, num_warps=4,
    )
    return hq, scale
