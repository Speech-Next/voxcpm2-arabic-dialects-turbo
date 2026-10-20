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

from .int8_linear import _int8_dot

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
        # the pinned staging buffer's last H2D copy: it is rewritten only once that copy has run
        self.copied: dict[tuple[str, int], torch.cuda.Event] = {}
        # row segments a step of the domain may have (default: one per weight); CFG lays the LocDiT's rows out as
        # [conditional batch][unconditional batch], each grouped by adapter: twice as many segments
        self.segments: dict[str, int] = {}
        self.base: set[str] = set()  # domains whose tables hold the all-base layout (update_base)

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
            self.base.discard(domain)
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
        self.base.discard(domain)
        for (dom, bm), buf in self.bufs.items():
            if dom != domain:
                continue
            t = build_tiles(s, bm)
            if t.shape[0] > buf.shape[0]:
                raise ValueError(f"{t.shape[0]} tiles exceed the captured table ({buf.shape[0]})")
            st = self.staging[(dom, bm)]
            ev = self.copied.get((dom, bm))
            if ev is not None:
                ev.synchronize()  # the previous copy may still read this staging buffer
            st.zero_()
            if t.shape[0]:
                st[: t.shape[0]].copy_(torch.from_numpy(t))
            buf.copy_(st, non_blocking=True)
            if not torch.cuda.is_current_stream_capturing():
                if ev is None:
                    ev = self.copied[(dom, bm)] = torch.cuda.Event()
                ev.record()

    def update_base(self, domain: str, rows: int, n_weights: int) -> None:
        """All ``rows`` on the base weights (a step without adapters in ``domain``): a LoRA graph replayed for the
        other domains' adapters must not read this domain's last adapter layout. No-op while the tables hold it."""
        if domain in self.base:
            return
        self.update(domain, np.full(rows, -1, dtype=np.int32), n_weights)
        self.base.add(domain)


SEG_TABLES = SegTables()


@triton.jit
def _tile_order(pid, NUM_N, NUM_T, GROUP_T: tl.constexpr):
    """(tile, N-tile) of program ``pid``: groups of GROUP_T tiles walk the N-tiles together (the base kernels'
    GROUP_M swizzle), so the A rows and the weight tiles a group shares stay in L2."""
    group = GROUP_T * NUM_N
    first = (pid // group) * GROUP_T
    gt = tl.minimum(NUM_T - first, GROUP_T)
    return first + (pid % group) % gt, (pid % group) // gt


@triton.jit(do_not_specialize=["M", "NUM_T"])
def _seg_w8a8_gemm(
    a_ptr, w_ptr, c_ptr, a_scale_ptr, w_scale_ptr, bias_ptr, tiles_ptr, remap_ptr,
    M, N, K, NUM_N, NUM_T,
    HAS_BIAS: tl.constexpr, HAS_REMAP: tl.constexpr, GROUP_T: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    t, pid_n = _tile_order(pid, NUM_N, NUM_T, GROUP_T)
    r0 = tl.load(tiles_ptr + 3 * t)
    r1 = tl.minimum(tl.load(tiles_ptr + 3 * t + 1), M)  # a graph's table may cover more rows than its batch
    if r0 >= r1:
        return
    wi = tl.load(tiles_ptr + 3 * t + 2)
    if HAS_REMAP:  # a stack holding only some adapters' weights: the tile's weight index (slot + 1) -> its entry
        wi = tl.load(remap_ptr + wi)
    wi = wi.to(tl.int64)
    rm = r0 + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = _int8_dot(a_ptr, w_ptr, wi, rm, rn, r1, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
    out = acc.to(tl.float32) * tl.load(a_scale_ptr + rm, mask=rm < r1, other=0.0)[:, None]
    out = out * tl.load(w_scale_ptr + wi * N + rn, mask=rn < N, other=0.0)[None, :]
    if HAS_BIAS:
        out += tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
    cm = (rm[:, None] < r1) & (rn[None, :] < N)
    tl.store(c_ptr + rm.to(tl.int64)[:, None] * N + rn[None, :], out.to(c_ptr.dtype.element_ty), mask=cm)


@triton.jit(do_not_specialize=["M", "NUM_T"])
def _seg_w8a8_gemm_res(
    a_ptr, w_ptr, c_ptr, a_scale_ptr, w_scale_ptr, res_ptr, tiles_ptr, remap_ptr,
    M, N, K, NUM_N, NUM_T,
    HAS_REMAP: tl.constexpr, GROUP_T: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """C = dequant(A @ W_seg^T) + RES."""
    pid = tl.program_id(0)
    t, pid_n = _tile_order(pid, NUM_N, NUM_T, GROUP_T)
    r0 = tl.load(tiles_ptr + 3 * t)
    r1 = tl.minimum(tl.load(tiles_ptr + 3 * t + 1), M)  # a graph's table may cover more rows than its batch
    if r0 >= r1:
        return
    wi = tl.load(tiles_ptr + 3 * t + 2)
    if HAS_REMAP:
        wi = tl.load(remap_ptr + wi)
    wi = wi.to(tl.int64)
    rm = r0 + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = _int8_dot(a_ptr, w_ptr, wi, rm, rn, r1, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
    out = acc.to(tl.float32) * tl.load(a_scale_ptr + rm, mask=rm < r1, other=0.0)[:, None]
    out = out * tl.load(w_scale_ptr + wi * N + rn, mask=rn < N, other=0.0)[None, :]
    cm = (rm[:, None] < r1) & (rn[None, :] < N)
    offs = rm.to(tl.int64)[:, None] * N + rn[None, :]
    out += tl.load(res_ptr + offs, mask=cm, other=0.0).to(tl.float32)
    tl.store(c_ptr + offs, out.to(c_ptr.dtype.element_ty), mask=cm)


@triton.jit(do_not_specialize=["M", "NUM_T"])
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
    acc = _int8_dot(a_ptr, w_ptr, wi, rm, rn, r1, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
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


@triton.jit(do_not_specialize=["M", "NUM_T"])
def _seg_w8a8_gemm_swiglu(
    a_ptr, w_ptr, h_ptr, amax_ptr, a_scale_ptr, w_scale_ptr, tiles_ptr, remap_ptr,
    M, N, K, NUM_N, NUM_T,
    HAS_REMAP: tl.constexpr, GROUP_T: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """fused_layer._w8a8_gemm_swiglu with one weight per row segment: interleaved gate/up rows, h = silu(g) * u
    [M, N/2] and the per-(row, N-tile) max |h| for the quant pass."""
    pid = tl.program_id(0)
    t, pid_n = _tile_order(pid, NUM_N, NUM_T, GROUP_T)
    r0 = tl.load(tiles_ptr + 3 * t)
    r1 = tl.minimum(tl.load(tiles_ptr + 3 * t + 1), M)
    if r0 >= r1:
        return
    wi = tl.load(tiles_ptr + 3 * t + 2)
    if HAS_REMAP:
        wi = tl.load(remap_ptr + wi)
    wi = wi.to(tl.int64)
    rm = r0 + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = _int8_dot(a_ptr, w_ptr, wi, rm, rn, r1, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
    rmask = rm < r1
    out = acc.to(tl.float32) * tl.load(a_scale_ptr + rm, mask=rmask, other=0.0)[:, None]
    out = out * tl.load(w_scale_ptr + wi * N + rn, mask=rn < N, other=0.0)[None, :]
    g, u = tl.split(tl.reshape(out, (BLOCK_M, BLOCK_N // 2, 2)))
    h = (g / (1.0 + tl.exp(-g)) * u).to(h_ptr.dtype.element_ty)
    HN: tl.constexpr = BLOCK_N // 2
    rh = pid_n * HN + tl.arange(0, HN)
    half = N // 2
    rows = rm.to(tl.int64)
    tl.store(h_ptr + rows[:, None] * half + rh[None, :], h, mask=rmask[:, None] & (rh[None, :] < half))
    tl.store(amax_ptr + rows * NUM_N + pid_n, tl.max(tl.abs(h.to(tl.float32)), axis=1), mask=rmask)


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
        ("plain", 64): (128, 4, 3), ("res", 64): (128, 4, 3), ("rope", 64): (128, 4, 3),
        ("swiglu", 128): (128, 8, 3), ("swiglu", 64): (128, 4, 3)}


def seg_gemm(q, s, w_stack, ws_stack, bias, out_dtype, tiles, n_tiles, bm, residual=None, remap=None):
    """int8 [M, K] x per-segment int8 [N, K] -> [M, N] (+ residual). remap: [1 + slots] int32, a tile's weight index
    (slot + 1) -> its entry of ``w_stack`` (a stack of only some adapters' weights, e.g. the MLP's)."""
    M, K = q.shape
    N = w_stack.shape[1]
    bn = 128
    num_n = triton.cdiv(N, bn)
    c = torch.empty(M, N, device=q.device, dtype=out_dtype)
    grid = (n_tiles * num_n,)
    rp = remap if remap is not None else tiles
    if residual is not None:
        bk, warps, stages = _CFG[("res", bm)]
        _seg_w8a8_gemm_res[grid](q, w_stack, c, s, ws_stack, residual.contiguous(), tiles, rp, M, N, K, num_n, n_tiles,
                                 HAS_REMAP=remap is not None, GROUP_T=8, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                                 num_warps=warps, num_stages=stages)
    else:
        bk, warps, stages = _CFG[("plain", bm)]
        _seg_w8a8_gemm[grid](q, w_stack, c, s, ws_stack, bias if bias is not None else ws_stack, tiles, rp, M, N, K,
                             num_n, n_tiles, HAS_BIAS=bias is not None, HAS_REMAP=remap is not None, GROUP_T=8,
                             BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=warps, num_stages=stages)
    return c


def seg_swiglu_gemm_quant(q, s, w_stack, ws_stack, out_dtype, tiles, n_tiles, bm, remap=None):
    """fused_layer.swiglu_gemm_quant with one weight per row segment (stacked ``swiglu_weights`` layers): int8
    silu(gate) * up [M, N/2] + per-row scales for the down GEMM."""
    from .fused_layer import _quant_rows_partials

    M, K = q.shape
    N = w_stack.shape[1]
    half = N // 2
    bn = 128
    num_n = triton.cdiv(N, bn)
    h = torch.empty(M, half, device=q.device, dtype=out_dtype)
    amax = torch.empty(M, num_n, device=q.device, dtype=torch.float32)
    bk, warps, stages = _CFG[("swiglu", bm)]
    _seg_w8a8_gemm_swiglu[(n_tiles * num_n,)](
        q, w_stack, h, amax, s, ws_stack, tiles, remap if remap is not None else tiles, M, N, K, num_n, n_tiles,
        HAS_REMAP=remap is not None, GROUP_T=8, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=warps,
        num_stages=stages,
    )
    hq = torch.empty(M, half, device=q.device, dtype=torch.int8)
    scale = torch.empty(M, device=q.device, dtype=torch.float32)
    br, bc = 8, 1024
    _quant_rows_partials[(triton.cdiv(M, br), triton.cdiv(half, bc))](
        h, amax, hq, scale, M, half, num_n, NT_P2=triton.next_power_of_2(num_n), BLOCK_R=br, BLOCK_C=bc, num_warps=4,
    )
    return hq, scale


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
