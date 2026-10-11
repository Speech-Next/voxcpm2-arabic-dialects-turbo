"""Segmented W8A8 GEMMs: one weight per row segment, for batches that mix LoRA adapters.

Every adapter's q/k/v/o projections are merged into int8 weights (``W + B A``, quantized like the base), stacked as
``[1 + slots, N, K]`` (index 0 = the base model). A batch whose rows are grouped by adapter (the scheduler orders
sequences by adapter) is described by a tile table: one ``(row_start, row_end, weight_index)`` entry per BLOCK_M
rows of a segment; program ``pid`` computes tile ``pid // NUM_N`` x N-tile ``pid % NUM_N`` with that tile's weight.
No LoRA kernels run: the adapter costs nothing per row beyond the base GEMM.

Under CUDA graphs the tables live in static buffers (:class:`SegTables`) that the runner fills before each replay;
the grid covers the most tiles a batch of the captured size can need (``max_tiles``), extra tiles exit at once.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

_EMPTY = np.zeros((0, 3), dtype=np.int32)


def build_tiles(slots, block_m: int) -> np.ndarray:
    """[T, 3] int32 tile table for per-row slots (-1 = base model -> weight 0, slot s -> weight s + 1)."""
    s = np.asarray(slots, dtype=np.int32)
    if s.size == 0:
        return _EMPTY
    change = np.flatnonzero(s[1:] != s[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [s.size]])
    rows = []
    for a, b in zip(starts.tolist(), ends.tolist()):
        w = int(s[a]) + 1
        for r in range(a, b, block_m):
            rows.append((r, min(r + block_m, b), w))
    return np.asarray(rows, dtype=np.int32)


def max_tiles(rows: int, block_m: int, n_weights: int) -> int:
    """Tiles a table of ``rows`` rows can need: every segment boundary adds at most one partial tile."""
    return triton.cdiv(rows, block_m) + n_weights


class SegTables:
    """Static tile tables per LoRA domain and BLOCK_M for CUDA graphs: captured kernels read them, the runner
    rewrites them (from the step's per-row slots) before every replay."""

    def __init__(self) -> None:
        self.bufs: dict[tuple[str, int], torch.Tensor] = {}
        self.staging: dict[tuple[str, int], torch.Tensor] = {}
        # row segments a step of the domain may have (default: one per weight); CFG lays the LocDiT's rows out as
        # [conditional batch][unconditional batch], each grouped by adapter: twice as many segments
        self.segments: dict[str, int] = {}

    def max_segments(self, domain: str, n_weights: int) -> int:
        return max(n_weights, self.segments.get(domain, 0))

    def get(self, domain: str, block_m: int, rows: int, n_weights: int) -> torch.Tensor:
        key = (domain, block_m)
        need = max_tiles(rows, block_m, self.max_segments(domain, n_weights))
        buf = self.bufs.get(key)
        if buf is None or buf.shape[0] < need:
            assert not torch.cuda.is_current_stream_capturing(), "SegTables.reserve() before capturing"
            buf = torch.zeros(need, 3, dtype=torch.int32, device="cuda")
            self.bufs[key] = buf
            self.staging[key] = torch.zeros(need, 3, dtype=torch.int32, device="cpu", pin_memory=True)
        return buf

    def reserve(self, domain: str, rows: int, n_weights: int, block_ms=(64, 128), segments: int | None = None) -> None:
        if segments:
            self.segments[domain] = segments
        for bm in block_ms:
            self.get(domain, bm, rows, n_weights)

    def update(self, domain: str, slots, n_weights: int) -> None:
        """Fill every table of ``domain`` for this step's per-row slots (stale tiles become empty). The captured
        grids cover ``max_tiles`` (a segment per weight): rows not grouped by adapter would leave rows uncomputed,
        so that is refused."""
        s = np.asarray(slots, dtype=np.int32)
        if s.size and int((s[1:] != s[:-1]).sum()) + 1 > self.max_segments(domain, n_weights):
            raise ValueError("LoRA rows are not grouped by adapter (the scheduler orders a step's sequences)")
        for (dom, bm), buf in self.bufs.items():
            if dom != domain:
                continue
            t = build_tiles(s, bm)
            if t.shape[0] > buf.shape[0]:
                raise ValueError(f"{t.shape[0]} tiles exceed the captured table ({buf.shape[0]})")
            st = self.staging[(dom, bm)]
            st.zero_()
            if t.shape[0]:
                st[: t.shape[0]].copy_(torch.from_numpy(t))
            buf.copy_(st, non_blocking=True)


SEG_TABLES = SegTables()


@triton.jit
def _tile_order(pid, NUM_N, NUM_T, GROUP_T: tl.constexpr):
    """(tile, N-tile) of program ``pid``: groups of GROUP_T tiles walk the N-tiles together (the base kernels'
    GROUP_M swizzle), so the A rows and the weight tiles a group shares stay in L2."""
    group = GROUP_T * NUM_N
    first = (pid // group) * GROUP_T
    gt = tl.minimum(NUM_T - first, GROUP_T)
    return first + (pid % group) % gt, (pid % group) // gt


@triton.jit
def _seg_w8a8_gemm(
    a_ptr, w_ptr, c_ptr, a_scale_ptr, w_scale_ptr, bias_ptr, tiles_ptr,
    M, N, K, NUM_N, NUM_T,
    HAS_BIAS: tl.constexpr, GROUP_T: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    t, pid_n = _tile_order(pid, NUM_N, NUM_T, GROUP_T)
    r0 = tl.load(tiles_ptr + 3 * t)
    r1 = tl.minimum(tl.load(tiles_ptr + 3 * t + 1), M)  # a graph's table may cover more rows than its batch
    if r0 >= r1:
        return
    wi = tl.load(tiles_ptr + 3 * t + 2).to(tl.int64)
    rm = r0 + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a = a_ptr + rm.to(tl.int64)[:, None] * K + rk[None, :]
    b = w_ptr + wi * N * K + rn.to(tl.int64)[None, :] * K + rk[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kk = k * BLOCK_K + rk
        acc += tl.dot(tl.load(a, mask=(rm[:, None] < r1) & (kk[None, :] < K), other=0),
                      tl.load(b, mask=(rn[None, :] < N) & (kk[:, None] < K), other=0))
        a += BLOCK_K
        b += BLOCK_K
    out = acc.to(tl.float32) * tl.load(a_scale_ptr + rm, mask=rm < r1, other=0.0)[:, None]
    out = out * tl.load(w_scale_ptr + wi * N + rn, mask=rn < N, other=0.0)[None, :]
    if HAS_BIAS:
        out += tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
    cm = (rm[:, None] < r1) & (rn[None, :] < N)
    tl.store(c_ptr + rm.to(tl.int64)[:, None] * N + rn[None, :], out.to(c_ptr.dtype.element_ty), mask=cm)


@triton.jit
def _seg_w8a8_gemm_res(
    a_ptr, w_ptr, c_ptr, a_scale_ptr, w_scale_ptr, res_ptr, tiles_ptr,
    M, N, K, NUM_N, NUM_T,
    GROUP_T: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """C = dequant(A @ W_seg^T) + RES."""
    pid = tl.program_id(0)
    t, pid_n = _tile_order(pid, NUM_N, NUM_T, GROUP_T)
    r0 = tl.load(tiles_ptr + 3 * t)
    r1 = tl.minimum(tl.load(tiles_ptr + 3 * t + 1), M)  # a graph's table may cover more rows than its batch
    if r0 >= r1:
        return
    wi = tl.load(tiles_ptr + 3 * t + 2).to(tl.int64)
    rm = r0 + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a = a_ptr + rm.to(tl.int64)[:, None] * K + rk[None, :]
    b = w_ptr + wi * N * K + rn.to(tl.int64)[None, :] * K + rk[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kk = k * BLOCK_K + rk
        acc += tl.dot(tl.load(a, mask=(rm[:, None] < r1) & (kk[None, :] < K), other=0),
                      tl.load(b, mask=(rn[None, :] < N) & (kk[:, None] < K), other=0))
        a += BLOCK_K
        b += BLOCK_K
    out = acc.to(tl.float32) * tl.load(a_scale_ptr + rm, mask=rm < r1, other=0.0)[:, None]
    out = out * tl.load(w_scale_ptr + wi * N + rn, mask=rn < N, other=0.0)[None, :]
    cm = (rm[:, None] < r1) & (rn[None, :] < N)
    offs = rm.to(tl.int64)[:, None] * N + rn[None, :]
    out += tl.load(res_ptr + offs, mask=cm, other=0.0).to(tl.float32)
    tl.store(c_ptr + offs, out.to(c_ptr.dtype.element_ty), mask=cm)


@triton.jit
def _seg_w8a8_gemm_qkv_rope(
    a_ptr, w_ptr, c_ptr, a_scale_ptr, w_scale_ptr, pos_ptr, cos_ptr, sin_ptr, tiles_ptr,
    M, N, K, NUM_N, NUM_T, n_rope_heads,
    GROUP_T: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """qkv GEMM with RoPE in the epilogue (weights from ``rope_weights``: within a head, rows j and j + D/2
    adjacent), one weight per segment; BLOCK_N = head_dim."""
    pid = tl.program_id(0)
    t, pid_n = _tile_order(pid, NUM_N, NUM_T, GROUP_T)
    r0 = tl.load(tiles_ptr + 3 * t)
    r1 = tl.minimum(tl.load(tiles_ptr + 3 * t + 1), M)  # a graph's table may cover more rows than its batch
    if r0 >= r1:
        return
    wi = tl.load(tiles_ptr + 3 * t + 2).to(tl.int64)
    rm = r0 + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a = a_ptr + rm.to(tl.int64)[:, None] * K + rk[None, :]
    b = w_ptr + wi * N * K + rn.to(tl.int64)[None, :] * K + rk[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kk = k * BLOCK_K + rk
        acc += tl.dot(tl.load(a, mask=(rm[:, None] < r1) & (kk[None, :] < K), other=0),
                      tl.load(b, mask=(rn[None, :] < N) & (kk[:, None] < K), other=0))
        a += BLOCK_K
        b += BLOCK_K
    rmask = rm < r1
    out = acc.to(tl.float32) * tl.load(a_scale_ptr + rm, mask=rmask, other=0.0)[:, None]
    out = out * tl.load(w_scale_ptr + wi * N + rn)[None, :]
    HALF: tl.constexpr = BLOCK_N // 2
    x1, x2 = tl.split(tl.reshape(out, (BLOCK_M, HALF, 2)))  # columns (2j, 2j+1) = head dims (j, j + HALF)
    j = tl.arange(0, HALF)
    if pid_n < n_rope_heads:
        pos = tl.load(pos_ptr + rm, mask=rmask, other=0).to(tl.int64)
        cs = pos[:, None] * BLOCK_N + j[None, :]
        c = tl.load(cos_ptr + cs)
        sn = tl.load(sin_ptr + cs)
        x1, x2 = x1 * c - x2 * sn, x2 * c + x1 * sn
    offs = rm.to(tl.int64)[:, None] * N + pid_n * BLOCK_N + j[None, :]
    tl.store(c_ptr + offs, x1.to(c_ptr.dtype.element_ty), mask=rmask[:, None])
    tl.store(c_ptr + offs + HALF, x2.to(c_ptr.dtype.element_ty), mask=rmask[:, None])


def _block_m(M: int) -> int:
    return 128 if M > 256 else 64


def tiles_for(slots_host, M: int, domain: str, n_weights: int, cache: dict | None = None):
    """(tile table on the GPU, grid tiles, BLOCK_M) for a call of M rows. Under graph capture: the domain's static
    table (filled by the runner per replay) and its worst-case tile count; eager: built from the step's slots."""
    bm = _block_m(M)
    if torch.cuda.is_current_stream_capturing():
        buf = SEG_TABLES.get(domain, bm, M, n_weights)
        return buf, max_tiles(M, bm, SEG_TABLES.max_segments(domain, n_weights)), bm
    key = (bm, M)
    if cache is not None and key in cache:
        return cache[key]
    t = build_tiles(slots_host[:M] if len(slots_host) >= M else list(slots_host) + [-1] * (M - len(slots_host)), bm)
    dev = torch.from_numpy(t).pin_memory().to("cuda", non_blocking=True) if t.shape[0] else torch.zeros(
        1, 3, dtype=torch.int32, device="cuda")
    out = (dev, max(t.shape[0], 1), bm)
    if cache is not None:
        cache[key] = out
    return out


# (BLOCK_K, num_warps, num_stages) per kind and BLOCK_M; BLOCK_N = 128 (head_dim for the RoPE kernel). Large tiles for
# prefill-size calls as int8_linear._LARGE (measured at the A10's 150 W cap), 64-row tiles for decode-size calls.
_CFG = {("plain", 128): (128, 8, 3), ("res", 128): (128, 8, 3), ("rope", 128): (128, 8, 3),
        ("plain", 64): (128, 4, 3), ("res", 64): (128, 4, 3), ("rope", 64): (128, 4, 3)}


def seg_gemm(q, s, w_stack, ws_stack, bias, out_dtype, tiles, n_tiles, bm, residual=None):
    """int8 [M, K] x per-segment int8 [N, K] -> [M, N] (+ residual)."""
    M, K = q.shape
    N = w_stack.shape[1]
    bn = 128
    num_n = triton.cdiv(N, bn)
    c = torch.empty(M, N, device=q.device, dtype=out_dtype)
    grid = (n_tiles * num_n,)
    if residual is not None:
        bk, warps, stages = _CFG[("res", bm)]
        _seg_w8a8_gemm_res[grid](q, w_stack, c, s, ws_stack, residual.contiguous(), tiles, M, N, K, num_n, n_tiles,
                                 GROUP_T=8, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=warps, num_stages=stages)
    else:
        bk, warps, stages = _CFG[("plain", bm)]
        _seg_w8a8_gemm[grid](q, w_stack, c, s, ws_stack, bias if bias is not None else ws_stack, tiles, M, N, K, num_n,
                             n_tiles, HAS_BIAS=bias is not None, GROUP_T=8, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                             num_warps=warps, num_stages=stages)
    return c


def seg_qkv_rope_gemm(q, s, w_stack, ws_stack, out_dtype, positions, cos, sin, n_rope_heads, tiles, n_tiles, bm):
    M, K = q.shape
    N = w_stack.shape[1]
    bn = cos.shape[1]
    assert N % bn == 0
    num_n = N // bn
    c = torch.empty(M, N, device=q.device, dtype=out_dtype)
    bk, warps, stages = _CFG[("rope", bm)]
    _seg_w8a8_gemm_qkv_rope[(n_tiles * num_n,)](
        q, w_stack, c, s, ws_stack, positions, cos, sin, tiles, M, N, K, num_n, n_tiles, n_rope_heads,
        GROUP_T=8, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=warps, num_stages=stages,
        **({"maxnreg": 128} if bm == 128 else {}),  # as qkv_rope_gemm: 2 CTAs/SM at 128x128x128
    )
    return c
