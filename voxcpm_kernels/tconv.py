"""Fused snake + streaming causal ConvTranspose1d (kernel 2s, stride s) for the AudioVAE decoder blocks.

Reference: y = snake(affine(x)); full = [state | y]; out = conv_transpose1d(full)[..., s : -trim]
(state = last input frame of the previous chunk). With kernel 2s the output phase r of frame j is

    out[:, j*s + r] = W[:, :, r]^T y[j] + W[:, :, r + s]^T y[j - 1] + b      (y[-1] = state)

so every phase is one GEMM [W_r^T | W_{r+s}^T] @ [y ; y_prev]. The snake runs once over the (small) input;
the GEMM reads y and the state directly, which removes the state concat, cuDNN's layout transposes and the slice.
(Applying the snake inside the GEMM's operand load recomputes it per phase and output block: 3x slower.)
"""

import torch
import triton
import triton.language as tl

from .snake import Snake


@triton.jit
def _tconv_gemm(
    y_ptr, state_ptr, idx_ptr, w_ptr, b_ptr, out_ptr,
    C_in, C_out, T, NB, S, state_slot_stride,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """out[b, :, j*S + r] = [W_r^T | W_{r+S}^T] @ [y[b, :, j] ; y[b, :, j-1]] + bias  (y[b, :, -1] = state).

    GEMM columns are the flattened (batch, frame) pairs n = b * T + j, so a weight tile is loaded once per
    column block and reused across requests (weights of the first block are ~67 MB).
    """
    pid0 = tl.program_id(0)
    r = pid0 % S  # output phase
    co = (pid0 // S) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    nm = n < NB * T
    bi = n // T
    j = n % T
    slot = tl.load(idx_ptr + bi, mask=nm, other=0).to(tl.int64)
    ycol = bi.to(tl.int64) * C_in * T + j  # + ci * T
    first = j == 0
    wr = w_ptr + r.to(tl.int64) * C_out * 2 * C_in  # packed [S, C_out, 2 * C_in]
    com = (co < C_out)[:, None]
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, C_in, BK):
        ci = k0 + tl.arange(0, BK)
        cm = ci < C_in
        m = cm[:, None] & nm[None, :]
        y = tl.load(y_ptr + ycol[None, :] + ci[:, None] * T, mask=m, other=0.0)
        yp = tl.load(y_ptr + ycol[None, :] - 1 + ci[:, None] * T, mask=m & (~first)[None, :], other=0.0)
        sv = tl.load(state_ptr + slot[None, :] * state_slot_stride + ci[:, None], mask=m & first[None, :], other=0.0)
        w0 = tl.load(wr + co[:, None] * (2 * C_in) + ci[None, :], mask=com & cm[None, :], other=0.0)
        w1 = tl.load(wr + co[:, None] * (2 * C_in) + C_in + ci[None, :], mask=com & cm[None, :], other=0.0)
        acc += tl.dot(w0, y)
        acc += tl.dot(w1, yp + sv)  # exactly one of yp / sv is unmasked per column
    acc += tl.load(b_ptr + co, mask=co < C_out, other=0.0).to(tl.float32)[:, None]
    optr = out_ptr + (bi.to(tl.int64) * C_out * T * S + j * S + r)[None, :] + co[:, None] * (T * S)
    tl.store(optr, acc.to(out_ptr.dtype.element_ty), mask=com & nm[None, :])


@triton.jit
def _tconv_gemm_allph(
    y_ptr, state_ptr, idx_ptr, w_ptr, b_ptr, out_ptr,
    C_in, C_out, T, NB, state_slot_stride,
    S: tl.constexpr, SP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """All S output phases of a [BM channels, BN frames] tile in one program. The phases' weights are stacked along
    M (SP = next power of 2 of S rows groups, groups >= S are zero), so each K step is one dot over the shared input
    tiles; the [SP, BM, BN] result is permuted to [BM, BN, SP] and stored contiguously along time (t = j*S + r)."""
    co0 = tl.program_id(0) * BM
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    nm = n < NB * T
    bi = n // T
    j = n % T
    slot = tl.load(idx_ptr + bi, mask=nm, other=0).to(tl.int64)
    ycol = bi.to(tl.int64) * C_in * T + j
    first = j == 0
    rows = tl.arange(0, SP * BM)
    ph = rows // BM
    co_r = co0 + rows % BM
    rmask = (ph < S) & (co_r < C_out)
    wrow = w_ptr + (ph.to(tl.int64) * C_out + co_r) * (2 * C_in)  # packed [S, C_out, 2 * C_in]
    acc = tl.zeros([SP * BM, BN], dtype=tl.float32)
    for k0 in range(0, C_in, BK):
        ci = k0 + tl.arange(0, BK)
        cm = ci < C_in
        m = cm[:, None] & nm[None, :]
        y = tl.load(y_ptr + ycol[None, :] + ci[:, None] * T, mask=m, other=0.0)
        yp = tl.load(y_ptr + ycol[None, :] - 1 + ci[:, None] * T, mask=m & (~first)[None, :], other=0.0)
        yp = yp + tl.load(state_ptr + slot[None, :] * state_slot_stride + ci[:, None], mask=m & first[None, :], other=0.0)
        wm = rmask[:, None] & cm[None, :]
        acc += tl.dot(tl.load(wrow[:, None] + ci[None, :], mask=wm, other=0.0), y)
        acc += tl.dot(tl.load(wrow[:, None] + C_in + ci[None, :], mask=wm, other=0.0), yp)
    acc += tl.load(b_ptr + co_r, mask=co_r < C_out, other=0.0).to(tl.float32)[:, None]
    out = tl.reshape(tl.permute(tl.reshape(acc, [SP, BM, BN]), (1, 2, 0)), [BM, BN * SP])
    q = tl.arange(0, BN * SP)
    r = q % SP
    n2 = tl.program_id(1) * BN + q // SP
    b2 = n2 // T
    j2 = n2 % T
    co = co0 + tl.arange(0, BM)
    optr = out_ptr + (b2.to(tl.int64) * C_out * T * S + j2 * S + r)[None, :] + co[:, None] * (T * S)
    tl.store(optr, out.to(out_ptr.dtype.element_ty), mask=(co < C_out)[:, None] & ((r < S) & (n2 < NB * T))[None, :])


@triton.jit
def _tconv_gemm_flat(
    y_ptr, state_ptr, idx_ptr, w_ptr, b_ptr, out_ptr,
    C_in, C_out, T, NB, state_slot_stride,
    S: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """All phases with the M axis flattened channel-major (row = co * S + r, weights packed [C_out * S, 2 * C_in]):
    M = S * C_out exactly, where the all-phase kernel pads S to a power of 2 (S = 6 / 5: 25 / 37.5 % zero rows)."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    rm = rows < C_out * S
    co = rows // S
    r = rows % S
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    nm = n < NB * T
    bi = n // T
    j = n % T
    slot = tl.load(idx_ptr + bi, mask=nm, other=0).to(tl.int64)
    ycol = bi.to(tl.int64) * C_in * T + j
    first = j == 0
    wrow = w_ptr + rows.to(tl.int64) * (2 * C_in)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, C_in, BK):
        ci = k0 + tl.arange(0, BK)
        cm = ci < C_in
        m = cm[:, None] & nm[None, :]
        y = tl.load(y_ptr + ycol[None, :] + ci[:, None] * T, mask=m, other=0.0)
        yp = tl.load(y_ptr + ycol[None, :] - 1 + ci[:, None] * T, mask=m & (~first)[None, :], other=0.0)
        yp = yp + tl.load(state_ptr + slot[None, :] * state_slot_stride + ci[:, None], mask=m & first[None, :], other=0.0)
        wm = rm[:, None] & cm[None, :]
        acc += tl.dot(tl.load(wrow[:, None] + ci[None, :], mask=wm, other=0.0), y)
        acc += tl.dot(tl.load(wrow[:, None] + C_in + ci[None, :], mask=wm, other=0.0), yp)
    acc += tl.load(b_ptr + co, mask=rm, other=0.0).to(tl.float32)[:, None]
    optr = out_ptr + (bi.to(tl.int64) * C_out * T * S + j * S)[None, :] + (co.to(tl.int64) * (T * S) + r)[:, None]
    tl.store(optr, acc.to(out_ptr.dtype.element_ty), mask=rm[:, None] & nm[None, :])


# (BM, BN, BK, warps) by input channels: wide early blocks have tiny T, narrow late blocks have long T
# Two weight and two input tiles per K step: keep stages * (2*BM*BK + 2*BK*BN) * 2 bytes under ~99 KB.
TILES = {2048: (128, 64, 64, 4), 1024: (128, 64, 64, 4), 512: (128, 64, 64, 4), 256: (128, 64, 32, 4),
         128: (64, 128, 32, 4), 64: (32, 128, 32, 4)}
STAGES = {}
# All-phase kernel (S > 2) per input channels, (BM, BN, BK, warps, stages), tuned at
# B=256 on an A10 at the power cap. Per decode step 2.64 / 4.34 / 6.11 -> 1.59 / 2.85 /
# 3.95 ms for the stride 8 / 6 / 5 blocks, despite the zero phase rows that pad S to a power of 2.
ALLPH_TILES = {2048: (32, 128, 32, 8, 2), 1024: (32, 64, 32, 4, 2), 512: (32, 64, 32, 4, 2)}
# Flattened all-phase kernel per input channels, (BM, BN, BK, warps, stages); takes precedence over the above.
# Tuned at B=256 on an A10 at the power cap: stride 5 3.97 -> 3.54 ms, stride 6
# 2.79 -> 2.70 ms; slower for stride 8 (no padding to remove) and stride 2 (strided stores, see _tconv_gemm_s2)
FLAT_TILES = {1024: (256, 128, 32, 8, 2), 512: (256, 64, 32, 4, 2)}


class FusedSnakeTransposeConv:
    """snake(affine(x)) -> streaming CausalTransposeConv1d with a [slots, C_in, 1] state."""

    def __init__(self, snake_mod, tconv_mod, slots: int, affine=None):
        s = tconv_mod.stride[0]
        k = tconv_mod.kernel_size[0]
        if k != 2 * s or tconv_mod.groups != 1 or tconv_mod.dilation[0] != 1:
            raise NotImplementedError("only kernel = 2 * stride, groups = 1 transposed convs are fused")
        trim = tconv_mod._CausalTransposeConv1d__padding * 2 - tconv_mod._CausalTransposeConv1d__output_padding
        if trim != s:
            raise NotImplementedError("output trimming other than one stride is not supported")
        w = tconv_mod.weight.detach()  # [C_in, C_out, 2s]
        self.C_in, self.C_out, self.S = w.shape[0], w.shape[1], s
        # pack per phase r: [W[:, :, r]^T | W[:, :, r + s]^T] -> [S, C_out, 2 * C_in]
        self.w = torch.cat([w[:, :, :s].permute(2, 1, 0), w[:, :, s:].permute(2, 1, 0)], dim=2).contiguous()
        self.b = tconv_mod.bias.detach().float().contiguous()
        self.snake = Snake(snake_mod.alpha, *(affine or (None, None)), fast_sin=True)
        self.state = torch.zeros(slots, self.C_in, 1, device=w.device, dtype=w.dtype)
        self.tiles = TILES.get(self.C_in, (64, 64, 64, 4))
        self.stages = STAGES.get(self.C_in, 2)
        self.pre_snaked = False
        self.allph = ALLPH_TILES.get(self.C_in)  # (BM, BN, BK, warps, stages) or None
        self.flat = FLAT_TILES.get(self.C_in)  # (BM, BN, BK, warps, stages) of _tconv_gemm_flat, or None
        self.w_flat = self.w.permute(1, 0, 2).reshape(self.C_out * s, 2 * self.C_in).contiguous()  # row = co * S + r

    def __call__(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        B, C_in, T = x.shape
        # the snake runs once over the input (the GEMM then needs no sin), or already ran in the epilogue of the op
        # that produced x (pre_snaked, see vae_stream._fuse_post_snakes)
        y = x if self.pre_snaked else self.snake(x)
        out = torch.empty(B, self.C_out, T * self.S, device=x.device, dtype=x.dtype)
        bm, bn, bk, warps = self.tiles
        if self.flat is not None:
            bm, bn, bk, warps, stages = self.flat
            _tconv_gemm_flat[(triton.cdiv(self.C_out * self.S, bm), triton.cdiv(B * T, bn))](
                y, self.state, idx, self.w_flat, self.b, out,
                C_in, self.C_out, T, B, self.state.stride(0),
                S=self.S, BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=stages,
            )
        elif self.S == 2 and self.allph is None:
            _tconv_gemm_s2[(triton.cdiv(self.C_out, bm), triton.cdiv(B * T, bn))](
                y, self.state, idx, self.w, self.b, out,
                C_in, self.C_out, T, B, self.state.stride(0),
                BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=self.stages,
            )
        elif self.allph is not None:
            bm, bn, bk, warps, stages = self.allph
            _tconv_gemm_allph[(triton.cdiv(self.C_out, bm), triton.cdiv(B * T, bn))](
                y, self.state, idx, self.w, self.b, out,
                C_in, self.C_out, T, B, self.state.stride(0),
                S=self.S, SP=triton.next_power_of_2(self.S), BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=stages,
            )
        else:
            _tconv_gemm[(triton.cdiv(self.C_out, bm) * self.S, triton.cdiv(B * T, bn))](
                y, self.state, idx, self.w, self.b, out,
                C_in, self.C_out, T, B, self.S, self.state.stride(0),
                BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=self.stages,
            )
        self.state.index_copy_(0, idx, y[:, :, -1:])
        return out


@triton.jit
def _tconv_gemm_s2(
    y_ptr, state_ptr, idx_ptr, w_ptr, b_ptr, out_ptr,
    C_in, C_out, T, NB, state_slot_stride,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """Stride-2 variant: both phases in one program, interleaved with tl.join and stored contiguously."""
    co = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    nm = n < NB * T
    bi = n // T
    j = n % T
    slot = tl.load(idx_ptr + bi, mask=nm, other=0).to(tl.int64)
    ycol = bi.to(tl.int64) * C_in * T + j
    first = j == 0
    w0p = w_ptr  # phase 0: [C_out, 2 * C_in]
    w1p = w_ptr + C_out * 2 * C_in  # phase 1
    com = (co < C_out)[:, None]
    acc0 = tl.zeros([BM, BN], dtype=tl.float32)
    acc1 = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, C_in, BK):
        ci = k0 + tl.arange(0, BK)
        cm = ci < C_in
        m = cm[:, None] & nm[None, :]
        y = tl.load(y_ptr + ycol[None, :] + ci[:, None] * T, mask=m, other=0.0)
        yp = tl.load(y_ptr + ycol[None, :] - 1 + ci[:, None] * T, mask=m & (~first)[None, :], other=0.0)
        yp = yp + tl.load(state_ptr + slot[None, :] * state_slot_stride + ci[:, None], mask=m & first[None, :], other=0.0)
        wm = com & cm[None, :]
        acc0 += tl.dot(tl.load(w0p + co[:, None] * (2 * C_in) + ci[None, :], mask=wm, other=0.0), y)
        acc0 += tl.dot(tl.load(w0p + co[:, None] * (2 * C_in) + C_in + ci[None, :], mask=wm, other=0.0), yp)
        acc1 += tl.dot(tl.load(w1p + co[:, None] * (2 * C_in) + ci[None, :], mask=wm, other=0.0), y)
        acc1 += tl.dot(tl.load(w1p + co[:, None] * (2 * C_in) + C_in + ci[None, :], mask=wm, other=0.0), yp)
    bias = tl.load(b_ptr + co, mask=co < C_out, other=0.0).to(tl.float32)[:, None]
    out = tl.reshape(tl.join(acc0 + bias, acc1 + bias), [BM, 2 * BN])  # [..., j, phase] -> time 2j + phase
    t2 = tl.arange(0, 2 * BN)
    n2 = tl.program_id(1) * BN + t2 // 2
    b2 = n2 // T
    j2 = n2 % T
    optr = out_ptr + (b2.to(tl.int64) * C_out * T * 2 + j2 * 2 + t2 % 2)[None, :] + co[:, None] * (T * 2)
    tl.store(optr, out.to(out_ptr.dtype.element_ty), mask=com & (n2 < NB * T)[None, :])
