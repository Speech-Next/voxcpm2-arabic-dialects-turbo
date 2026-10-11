"""Fused streaming CausalResidualUnit for the AudioVAE decoder.

Reference (per stream, causal state = last 6*d inputs of the dilated conv):
    h1 = snake1(x); h2 = dwconv7_dilated(cat(state, h1)); h3 = snake2(h2)
    out = x + conv1x1(h3);  state <- last 6*d samples of cat(state, h1)

Unfused this reads/writes the [B, C, T] activation ~14 times per unit. Here:
  1. h1p = [state | snake1(x)] in one elementwise pass (the conv taps reuse h1
     7 times, so recomputing snake per tap would make the fused kernel sin-bound;
     the state prefix removes the per-tap state/x branch),
  2. one kernel: dilated depthwise conv over [state | h1] -> snake2 -> 1x1 conv
     (tl.dot) -> + bias + x, written once,
  3. the causal state is advanced from the padded [state | h1] buffer.
"""

import os

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice



@triton.jit
def _sin(x, FAST: tl.constexpr):
    if FAST:
        return libdevice.fast_sinf(x)
    else:
        return tl.sin(x)


@triton.jit
def _snake_pad(
    x_ptr, state_ptr, idx_ptr, out_ptr, a1_ptr, i1_ptr,
    C, T, P, state_slot_stride,
    FAST_SIN: tl.constexpr, BLOCK: tl.constexpr,
):
    """out[b, c, :] = [state[slot_b, c, :P] | snake1(x[b, c, :T])]  (row length P + T)."""
    b = tl.program_id(0)
    c = tl.program_id(1)
    j = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    L = P + T
    jm = j < L
    slot = tl.load(idx_ptr + b).to(tl.int64)
    in_x = j >= P
    xv = tl.load(x_ptr + (b.to(tl.int64) * C + c) * T + (j - P), mask=jm & in_x, other=0.0).to(tl.float32)
    a1 = tl.load(a1_ptr + c)
    sx = _sin(a1 * xv, FAST_SIN)
    hx = xv + sx * sx * tl.load(i1_ptr + c)
    sv = tl.load(state_ptr + slot * state_slot_stride + c * P + j, mask=jm & (~in_x), other=0.0).to(tl.float32)
    tl.store(out_ptr + (b.to(tl.int64) * C + c) * L + j, tl.where(in_x, hx, sv).to(out_ptr.dtype.element_ty), mask=jm)


@triton.jit
def _resunit_fwd(
    x_ptr, h1_ptr, out_ptr,
    w7_ptr, b7_ptr, a2_ptr, i2_ptr, w1_ptr, b1_ptr,
    C, T, P,
    DIL: tl.constexpr, KSIZE: tl.constexpr, FAST_SIN: tl.constexpr,
    BC: tl.constexpr, BCO: tl.constexpr, BT: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1) * BT + tl.arange(0, BT)
    co = tl.program_id(2) * BCO + tl.arange(0, BCO)
    row = b.to(tl.int64) * C * T
    prow = b.to(tl.int64) * C * (P + T)
    tmask = t < T
    acc = tl.zeros([BCO, BT], dtype=tl.float32)
    for c0 in range(0, C, BC):
        c = c0 + tl.arange(0, BC)
        cm = c < C
        h2 = tl.zeros([BC, BT], dtype=tl.float32) + tl.load(b7_ptr + c, mask=cm, other=0.0)[:, None]
        hrow = h1_ptr + prow + c[:, None] * (P + T)
        for k in tl.static_range(KSIZE):
            hv = tl.load(hrow + (t + k * DIL)[None, :], mask=cm[:, None] & tmask[None, :], other=0.0)
            w = tl.load(w7_ptr + c * KSIZE + k, mask=cm, other=0.0)[:, None]
            h2 += w * hv.to(tl.float32)
        s2 = _sin(tl.load(a2_ptr + c, mask=cm, other=1.0)[:, None] * h2, FAST_SIN)
        h3 = tl.where(cm[:, None] & tmask[None, :], h2 + s2 * s2 * tl.load(i2_ptr + c, mask=cm, other=1.0)[:, None], 0.0)
        w1 = tl.load(w1_ptr + co[:, None] * C + c[None, :], mask=(co < C)[:, None] & cm[None, :], other=0.0)
        acc += tl.dot(w1, h3.to(w1.dtype))
    om = (co < C)[:, None] & tmask[None, :]
    res = tl.load(x_ptr + row + co[:, None] * T + t[None, :], mask=om, other=0.0).to(tl.float32)
    y = acc + tl.load(b1_ptr + co, mask=co < C, other=0.0)[:, None] + res
    tl.store(out_ptr + row + co[:, None] * T + t[None, :], y.to(out_ptr.dtype.element_ty), mask=om)


# (BC, BCO, BT, num_warps) per channel count, tuned on an A10 at B=256 (bench/tune_resunit.py)
TILES = {32: (32, 32, 64, 4), 64: (64, 64, 32, 4), 128: (16, 128, 32, 4), 256: (16, 128, 64, 8)}


class FusedResidualUnit:
    """Streaming CausalResidualUnit; state tensor [slots, C, 6 * dilation] holds snake1 outputs."""

    max_channels = 256  # wider units have short T per step; the unfused path is fine there

    def __init__(self, mod, slots: int, fast_sin: bool = False):
        snake1, conv7, snake2, conv1 = list(mod.block)
        self.C = conv7.in_channels
        self.dil = conv7.dilation[0]
        self.ksize = conv7.kernel_size[0]
        assert conv7.groups == self.C and conv1.kernel_size[0] == 1 and conv1.groups == 1
        self.P = conv7._CausalConv1d__padding * 2 - getattr(conv7, "_CausalConv1d__output_padding", 0)
        assert self.P == (self.ksize - 1) * self.dil
        dev, dt = conv7.weight.device, conv7.weight.dtype
        f32 = dict(device=dev, dtype=torch.float32)
        self.a1 = snake1.alpha.detach().reshape(-1).to(**f32).contiguous()
        self.i1 = (self.a1 + 1e-9).reciprocal()
        self.a2 = snake2.alpha.detach().reshape(-1).to(**f32).contiguous()
        self.i2 = (self.a2 + 1e-9).reciprocal()
        self.w7 = conv7.weight.detach().reshape(self.C, self.ksize).to(**f32).contiguous()
        self.b7 = conv7.bias.detach().to(**f32).contiguous()
        self.w1 = conv1.weight.detach().reshape(self.C, self.C).to(dt).contiguous()
        self.b1 = conv1.bias.detach().to(**f32).contiguous()
        self.state = torch.zeros(slots, self.C, self.P, device=dev, dtype=dt)
        self.fast_sin = fast_sin
        self.post_snake = None  # (alpha, 1/alpha, scale, bias) of a following (affine +) snake applied in the epilogue
        self.bc, self.bco, self.bt, self.warps = TILES.get(self.C, (32, 64, 32, 8))

    def __call__(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        B, C, T = x.shape
        L = self.P + T
        h1p = torch.empty(B, C, L, device=x.device, dtype=x.dtype)
        _snake_pad[(B, C, triton.cdiv(L, 1024))](
            x, self.state, idx, h1p, self.a1, self.i1,
            C, T, self.P, self.state.stride(0),
            FAST_SIN=self.fast_sin, BLOCK=1024,
        )
        self.state.index_copy_(0, idx, h1p[:, :, T:])
        out = torch.empty_like(x)
        _resunit_fwd[(B, triton.cdiv(T, self.bt), triton.cdiv(C, self.bco))](
            x, h1p, out,
            self.w7, self.b7, self.a2, self.i2, self.w1, self.b1,
            C, T, self.P,
            DIL=self.dil, KSIZE=self.ksize, FAST_SIN=self.fast_sin,
            BC=self.bc, BCO=self.bco, BT=self.bt,
            num_warps=self.warps,
        )
        return out


# ---------------------------------------------------------------------------------------------------------------
# Split variant for wide units: Triton depthwise conv + snake2, then the 1x1 conv as one cuBLAS batched GEMM.
# The fused kernel recomputes the depthwise conv once per output-channel block, which dominates when C >= 256.
@triton.jit
def _dw_snake2(
    h1_ptr, x_ptr, h3_ptr, o_ptr,
    w7_ptr, b7_ptr, a2_ptr, i2_ptr, b1_ptr,
    C, T, P,
    DIL: tl.constexpr, KSIZE: tl.constexpr, FAST_SIN: tl.constexpr, BC: tl.constexpr, BT: tl.constexpr,
):
    """h3 = snake2(dwconv7(h1p)); o = x + b1  (h1p rows are [state | snake1(x)], length P + T)."""
    b = tl.program_id(0)
    t = tl.program_id(1) * BT + tl.arange(0, BT)
    c = tl.program_id(2) * BC + tl.arange(0, BC)
    cm = c < C
    m = cm[:, None] & (t < T)[None, :]
    hrow = h1_ptr + b.to(tl.int64) * C * (P + T) + c[:, None] * (P + T)
    h2 = tl.zeros([BC, BT], dtype=tl.float32) + tl.load(b7_ptr + c, mask=cm, other=0.0)[:, None]
    for k in tl.static_range(KSIZE):
        hv = tl.load(hrow + (t + k * DIL)[None, :], mask=m, other=0.0)
        h2 += tl.load(w7_ptr + c * KSIZE + k, mask=cm, other=0.0)[:, None] * hv.to(tl.float32)
    s2 = _sin(tl.load(a2_ptr + c, mask=cm, other=1.0)[:, None] * h2, FAST_SIN)
    h3 = h2 + s2 * s2 * tl.load(i2_ptr + c, mask=cm, other=1.0)[:, None]
    offs = b.to(tl.int64) * C * T + c[:, None] * T + t[None, :]
    tl.store(h3_ptr + offs, h3.to(h3_ptr.dtype.element_ty), mask=m)
    xv = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(o_ptr + offs, (xv + tl.load(b1_ptr + c, mask=cm, other=0.0)[:, None]).to(o_ptr.dtype.element_ty), mask=m)


class SplitResidualUnit(FusedResidualUnit):
    """Same math and state layout as FusedResidualUnit; the 1x1 conv runs as cuBLAS bmm."""

    def __call__(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        B, C, T = x.shape
        L = self.P + T
        h1p = torch.empty(B, C, L, device=x.device, dtype=x.dtype)
        _snake_pad[(B, C, triton.cdiv(L, 1024))](
            x, self.state, idx, h1p, self.a1, self.i1,
            C, T, self.P, self.state.stride(0),
            FAST_SIN=self.fast_sin, BLOCK=1024,
        )
        self.state.index_copy_(0, idx, h1p[:, :, T:])
        h3 = torch.empty_like(x)
        out = torch.empty_like(x)
        bt = min(64, triton.next_power_of_2(T))
        _dw_snake2[(B, triton.cdiv(T, bt), triton.cdiv(C, 32))](
            h1p, x, h3, out, self.w7, self.b7, self.a2, self.i2, self.b1,
            C, T, self.P, DIL=self.dil, KSIZE=self.ksize, FAST_SIN=self.fast_sin, BC=32, BT=bt, num_warps=4,
        )
        return out.baddbmm_(self.w1.unsqueeze(0).expand(B, C, C), h3)


# ---------------------------------------------------------------------------------------------------------------
# No-padded-buffer variant: snake1 recomputed per tap with the hardware approximate sin (sin.approx: one MUFU op),
# reading [state | x] directly, so the [B, C, P + T] snake buffer is never written or read.
@triton.jit
def _dw_snake12(
    x_ptr, state_ptr, idx_ptr, h3_ptr, o_ptr,
    a1_ptr, i1_ptr, w7_ptr, b7_ptr, a2_ptr, i2_ptr, b1_ptr,
    C, T, P, state_slot_stride,
    DIL: tl.constexpr, KSIZE: tl.constexpr, BC: tl.constexpr, BT: tl.constexpr, FAST_SIN: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1) * BT + tl.arange(0, BT)
    c = tl.program_id(2) * BC + tl.arange(0, BC)
    cm = c < C
    tm = t < T
    m = cm[:, None] & tm[None, :]
    slot = tl.load(idx_ptr + b).to(tl.int64)
    xrow = x_ptr + b.to(tl.int64) * C * T + c[:, None] * T
    srow = state_ptr + slot * state_slot_stride + c[:, None] * P
    a1 = tl.load(a1_ptr + c, mask=cm, other=1.0)[:, None]
    i1 = tl.load(i1_ptr + c, mask=cm, other=1.0)[:, None]
    h2 = tl.zeros([BC, BT], dtype=tl.float32) + tl.load(b7_ptr + c, mask=cm, other=0.0)[:, None]
    for k in tl.static_range(KSIZE):
        src = t + k * DIL - P
        in_x = src >= 0
        xv = tl.load(xrow + src[None, :], mask=cm[:, None] & (in_x & tm)[None, :], other=0.0).to(tl.float32)
        sx = _sin(a1 * xv, FAST_SIN)
        sv = tl.load(srow + (src + P)[None, :], mask=cm[:, None] & ((~in_x) & tm)[None, :], other=0.0).to(tl.float32)
        h1 = tl.where(in_x[None, :], xv + sx * sx * i1, sv)
        h2 += tl.load(w7_ptr + c * KSIZE + k, mask=cm, other=0.0)[:, None] * h1
    s2 = _sin(tl.load(a2_ptr + c, mask=cm, other=1.0)[:, None] * h2, FAST_SIN)
    h3 = h2 + s2 * s2 * tl.load(i2_ptr + c, mask=cm, other=1.0)[:, None]
    offs = b.to(tl.int64) * C * T + c[:, None] * T + t[None, :]
    tl.store(h3_ptr + offs, h3.to(h3_ptr.dtype.element_ty), mask=m)
    xv = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(o_ptr + offs, (xv + tl.load(b1_ptr + c, mask=cm, other=0.0)[:, None]).to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _snake_state(x_ptr, state_ptr, idx_ptr, a1_ptr, i1_ptr, C, T, P, state_slot_stride, BC: tl.constexpr, BP: tl.constexpr, FAST_SIN: tl.constexpr):
    """state[slot, c, j] <- [state | snake1(x)][T + j] (one program owns whole rows; loads precede stores)."""
    b = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    j = tl.arange(0, BP)
    slot = tl.load(idx_ptr + b).to(tl.int64)
    srow = state_ptr + slot * state_slot_stride + c[:, None] * P
    cm = c < C
    jm = j < P
    src = T + j - P
    in_x = src >= 0
    xv = tl.load(x_ptr + b.to(tl.int64) * C * T + c[:, None] * T + src[None, :], mask=cm[:, None] & (in_x & jm)[None, :], other=0.0).to(tl.float32)
    sx = _sin(tl.load(a1_ptr + c, mask=cm, other=1.0)[:, None] * xv, FAST_SIN)
    hx = xv + sx * sx * tl.load(i1_ptr + c, mask=cm, other=1.0)[:, None]
    sv = tl.load(srow + (T + j)[None, :], mask=cm[:, None] & ((~in_x) & jm)[None, :], other=0.0).to(tl.float32)
    new = tl.where(in_x[None, :], hx, sv)
    # When T < P a row shifts within itself (reads T + j, writes j): every thread must finish its loads
    # before any thread stores, or a store can clobber a value another thread has not read yet.
    tl.debug_barrier()
    tl.store(srow + j[None, :], new.to(state_ptr.dtype.element_ty), mask=cm[:, None] & jm[None, :])


DIRECT_FAST_SIN = os.environ.get("VOXCPM_KERNELS_FAST_SIN", "1") == "1"
HEAD_CUDA = os.environ.get("VOXCPM_KERNELS_HEAD_CUDA", "1") == "1"


# ---------------------------------------------------------------------------------------------------------------
# The decoder's output head: snake -> causal conv (C -> 1 channel, kernel K) -> tanh, in one pass. The unfused ops
# concatenate the conv state onto the whole activation (a full copy) and run cuDNN with layout conversions for a
# single output channel. Here a program owns BT samples of one stream: per tap it loads [C, BT] inputs (the conv
# state for the first K-1 positions), applies the snake (fast sin, recomputed per tap as in _dw_snake12), weights
# and sums over channels, then tanh -> output. The state (last K-1 snake outputs) advances in _snake_state after.
@triton.jit
def _snake_conv1_tanh(
    x_ptr, state_ptr, idx_ptr, o_ptr, a_ptr, ia_ptr, w_ptr, bias,
    T, P, state_slot_stride,
    C: tl.constexpr, KSIZE: tl.constexpr, BT: tl.constexpr, FAST_SIN: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1) * BT + tl.arange(0, BT)
    c = tl.arange(0, C)
    tm = t < T
    slot = tl.load(idx_ptr + b).to(tl.int64)
    xrow = x_ptr + b.to(tl.int64) * C * T + c[:, None] * T
    srow = state_ptr + slot * state_slot_stride + c[:, None] * P
    a = tl.load(a_ptr + c)[:, None]
    ia = tl.load(ia_ptr + c)[:, None]
    acc = tl.zeros([C, BT], dtype=tl.float32)
    for k in tl.static_range(KSIZE):
        src = t + k - P
        in_x = src >= 0
        xv = tl.load(xrow + src[None, :], mask=(in_x & tm)[None, :], other=0.0).to(tl.float32)
        sx = _sin(a * xv, FAST_SIN)
        sv = tl.load(srow + (src + P)[None, :], mask=((~in_x) & tm)[None, :], other=0.0).to(tl.float32)
        acc += tl.load(w_ptr + c * KSIZE + k)[:, None] * tl.where(in_x[None, :], xv + sx * sx * ia, sv)
    y = tl.sum(acc, axis=0) + bias
    e = tl.exp(-2.0 * tl.abs(y))
    th = (1.0 - e) / (1.0 + e)
    tl.store(o_ptr + b.to(tl.int64) * T + t, tl.where(y < 0, -th, th).to(o_ptr.dtype.element_ty), mask=tm)


class SnakeConvTanhHead:
    """Streaming snake -> CausalConv1d(C, 1, K) -> tanh; state [slots, C, K-1] holds the last snake outputs."""

    # BT 32 / 4 warps: 112 registers; wider tiles spill (BT 256 / 4 warps: 79 ms instead of 0.97 ms at B=256)
    def __init__(self, snake_mod, conv_mod, slots: int, bt: int = 32, warps: int = 4):
        assert conv_mod.out_channels == 1 and conv_mod.groups == 1 and conv_mod.dilation[0] == 1
        self.C, self.K = conv_mod.in_channels, conv_mod.kernel_size[0]
        self.P = self.K - 1
        dev, dt = conv_mod.weight.device, conv_mod.weight.dtype
        self.a = snake_mod.alpha.detach().reshape(-1).float().contiguous()
        self.ia = (self.a + 1e-9).reciprocal()
        self.w = conv_mod.weight.detach().reshape(self.C, self.K).float().contiguous()
        self.bias = float(conv_mod.bias.detach().float().item()) if conv_mod.bias is not None else 0.0
        self.state = torch.zeros(slots, self.C, self.P, device=dev, dtype=dt)
        self.bt, self.warps = bt, warps
        self.ext = None  # CUDA head (cuda/head_fused.cu): snake staged in shared memory once instead of per tap
        if HEAD_CUDA and self.C == 32 and self.K == 7 and dt == torch.float16:
            from .cuda_ext import available, load

            self.ext = load("head_fused") if available("head_fused") else None

    def __call__(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        B, C, T = x.shape
        out = torch.empty(B, 1, T, device=x.device, dtype=x.dtype)
        if self.ext is not None:
            self.ext.head_fused(x, self.state, idx, out, self.a, self.ia, self.w, self.bias)
        else:
            bt = min(self.bt, max(16, triton.next_power_of_2(T)))
            _snake_conv1_tanh[(B, triton.cdiv(T, bt))](
                x, self.state, idx, out, self.a, self.ia, self.w, self.bias, T, self.P, self.state.stride(0),
                C=C, KSIZE=self.K, BT=bt, FAST_SIN=DIRECT_FAST_SIN, num_warps=self.warps,
            )
        _snake_state[(B, triton.cdiv(C, 32))](
            x, self.state, idx, self.a, self.ia, C, T, self.P, self.state.stride(0),
            BC=32, BP=triton.next_power_of_2(self.P), FAST_SIN=DIRECT_FAST_SIN,
        )
        return out


# ---------------------------------------------------------------------------------------------------------------
# One-pass unit: a program owns BT time steps of one stream and *all* C channels. It computes the depthwise conv
# with per-tap fast-sin snake1 (as _dw_snake12) and snake2 once, keeps h3 [C, BT] on chip, and runs the 1x1 conv
# as tl.dot over output-channel blocks, adding bias and the residual before the single store. Versus
# _dw_snake12 + cuBLAS baddbmm, h3 and the bias-initialised output are never written to or re-read from DRAM.
@triton.jit
def _resunit_onepass(
    x_ptr, state_ptr, idx_ptr, o_ptr,
    a1_ptr, i1_ptr, w7_ptr, b7_ptr, a2_ptr, i2_ptr, w1_ptr, b1_ptr,
    T, P, state_slot_stride, pa_ptr, pia_ptr, ps_ptr, pb_ptr,
    C: tl.constexpr, DIL: tl.constexpr, KSIZE: tl.constexpr, BT: tl.constexpr, BCO: tl.constexpr,
    FAST_SIN: tl.constexpr, POST_SNAKE: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1) * BT + tl.arange(0, BT)
    c = tl.arange(0, C)
    tm = t < T
    slot = tl.load(idx_ptr + b).to(tl.int64)
    base = b.to(tl.int64) * C * T
    xrow = x_ptr + base + c[:, None] * T
    srow = state_ptr + slot * state_slot_stride + c[:, None] * P
    a1 = tl.load(a1_ptr + c)[:, None]
    i1 = tl.load(i1_ptr + c)[:, None]
    h2 = tl.zeros([C, BT], dtype=tl.float32) + tl.load(b7_ptr + c)[:, None]
    for k in tl.static_range(KSIZE):
        src = t + k * DIL - P
        in_x = src >= 0
        xv = tl.load(xrow + src[None, :], mask=(in_x & tm)[None, :], other=0.0).to(tl.float32)
        sx = _sin(a1 * xv, FAST_SIN)
        sv = tl.load(srow + (src + P)[None, :], mask=((~in_x) & tm)[None, :], other=0.0).to(tl.float32)
        h2 += tl.load(w7_ptr + c * KSIZE + k)[:, None] * tl.where(in_x[None, :], xv + sx * sx * i1, sv)
    s2 = _sin(tl.load(a2_ptr + c)[:, None] * h2, FAST_SIN)
    h3 = (h2 + s2 * s2 * tl.load(i2_ptr + c)[:, None]).to(w1_ptr.dtype.element_ty)
    for co0 in tl.static_range(0, C, BCO):
        co = co0 + tl.arange(0, BCO)
        acc = tl.dot(tl.load(w1_ptr + co[:, None] * C + c[None, :]), h3)
        offs = base + co[:, None] * T + t[None, :]
        res = tl.load(x_ptr + offs, mask=tm[None, :], other=0.0).to(tl.float32)
        y = acc + tl.load(b1_ptr + co)[:, None] + res
        if POST_SNAKE:  # the next op's (affine +) snake, its only consumer; see _fuse_post_snakes in vae_stream.py
            y = y * tl.load(ps_ptr + co)[:, None] + tl.load(pb_ptr + co)[:, None]
            sy = _sin(tl.load(pa_ptr + co)[:, None] * y, True)
            y = y + sy * sy * tl.load(pia_ptr + co)[:, None]
        tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), mask=tm[None, :])


# ---------------------------------------------------------------------------------------------------------------
# Two-pass unit for wider channels: _dw_snake12's depthwise conv + snakes writing only h3, then a Triton batched
# 1x1 conv whose epilogue adds bias and the residual and writes the output once (cuBLAS baddbmm needs the output
# pre-initialised to x + b1 by the first pass, i.e. one more write and read of the activation).
@triton.jit
def _dw_snake12_h3(
    x_ptr, state_ptr, idx_ptr, h3_ptr,
    a1_ptr, i1_ptr, w7_ptr, b7_ptr, a2_ptr, i2_ptr,
    C, T, P, state_slot_stride,
    DIL: tl.constexpr, KSIZE: tl.constexpr, BC: tl.constexpr, BT: tl.constexpr, FAST_SIN: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1) * BT + tl.arange(0, BT)
    c = tl.program_id(2) * BC + tl.arange(0, BC)
    cm = c < C
    tm = t < T
    slot = tl.load(idx_ptr + b).to(tl.int64)
    xrow = x_ptr + b.to(tl.int64) * C * T + c[:, None] * T
    srow = state_ptr + slot * state_slot_stride + c[:, None] * P
    a1 = tl.load(a1_ptr + c, mask=cm, other=1.0)[:, None]
    i1 = tl.load(i1_ptr + c, mask=cm, other=1.0)[:, None]
    h2 = tl.zeros([BC, BT], dtype=tl.float32) + tl.load(b7_ptr + c, mask=cm, other=0.0)[:, None]
    PC: tl.constexpr = (KSIZE - 1) * DIL  # == P; as a constant the tap offsets fold into the addresses
    if tl.program_id(1) * BT >= PC:  # interior tile: every tap reads x, no state loads or selects
        for k in tl.static_range(KSIZE):
            xv = tl.load(xrow + (t + (k * DIL - PC))[None, :], mask=cm[:, None] & tm[None, :], other=0.0).to(tl.float32)
            sx = _sin(a1 * xv, FAST_SIN)
            h2 += tl.load(w7_ptr + c * KSIZE + k, mask=cm, other=0.0)[:, None] * (xv + sx * sx * i1)
    else:
        for k in tl.static_range(KSIZE):
            src = t + (k * DIL - PC)
            in_x = src >= 0
            xv = tl.load(xrow + src[None, :], mask=cm[:, None] & (in_x & tm)[None, :], other=0.0).to(tl.float32)
            sx = _sin(a1 * xv, FAST_SIN)
            sv = tl.load(srow + (src + PC)[None, :], mask=cm[:, None] & ((~in_x) & tm)[None, :], other=0.0).to(tl.float32)
            h2 += tl.load(w7_ptr + c * KSIZE + k, mask=cm, other=0.0)[:, None] * tl.where(in_x[None, :], xv + sx * sx * i1, sv)
    s2 = _sin(tl.load(a2_ptr + c, mask=cm, other=1.0)[:, None] * h2, FAST_SIN)
    h3 = h2 + s2 * s2 * tl.load(i2_ptr + c, mask=cm, other=1.0)[:, None]
    tl.store(h3_ptr + b.to(tl.int64) * C * T + c[:, None] * T + t[None, :], h3.to(h3_ptr.dtype.element_ty),
             mask=cm[:, None] & tm[None, :])


@triton.jit
def _conv1x1_res(h3_ptr, x_ptr, w1_ptr, b1_ptr, o_ptr, C, T, pa_ptr, pia_ptr, ps_ptr, pb_ptr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr, POST_SNAKE: tl.constexpr):
    """o[b] = W1 @ h3[b] + b1 + x[b]  (per stream: [C, C] @ [C, T]); W1 tiles are reused across GROUP streams."""
    pid_n = tl.program_id(0)
    co = tl.program_id(1) * BM + tl.arange(0, BM)
    b = tl.program_id(2)
    t = pid_n * BN + tl.arange(0, BN)
    com = co < C
    tm = t < T
    base = b.to(tl.int64) * C * T
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, C, BK):
        k = k0 + tl.arange(0, BK)
        km = k < C
        w = tl.load(w1_ptr + co[:, None] * C + k[None, :], mask=com[:, None] & km[None, :], other=0.0)
        h = tl.load(h3_ptr + base + k[:, None] * T + t[None, :], mask=km[:, None] & tm[None, :], other=0.0)
        acc += tl.dot(w, h)
    m = com[:, None] & tm[None, :]
    offs = base + co[:, None] * T + t[None, :]
    y = acc + tl.load(b1_ptr + co, mask=com, other=0.0)[:, None] + tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    if POST_SNAKE:  # the next op's (affine +) snake, its only consumer; see _fuse_post_snakes in vae_stream.py
        y = y * tl.load(ps_ptr + co, mask=com, other=1.0)[:, None] + tl.load(pb_ptr + co, mask=com, other=0.0)[:, None]
        sy = _sin(tl.load(pa_ptr + co, mask=com, other=1.0)[:, None] * y, True)
        y = y + sy * sy * tl.load(pia_ptr + co, mask=com, other=1.0)[:, None]
    tl.store(o_ptr + offs, y.to(o_ptr.dtype.element_ty), mask=m)


# (BM, BN, BK, num_warps, num_stages) of the 1x1 conv per channel count, tuned at
# B=256 on an A10 at the power cap; vs DirectResidualUnit per unit: C=128 / 256 / 512 / 1024
# 2.28 / 2.65 / 1.35 / 0.92 -> 2.10 / 2.33 / 1.20 / 0.68 ms.
# (BC, BT, num_warps) of the depthwise pass per channel count (same tuning):
# C=128 2.06 -> 1.88 ms per unit; within 3% elsewhere
DW_TILES = {128: (16, 128, 8), 256: (16, 256, 8)}
TWOPASS_TILES = {128: (128, 128, 64, 8, 2), 256: (128, 128, 32, 4, 3), 512: (128, 128, 32, 4, 3), 1024: (128, 128, 64, 4, 2)}


class TwoPassResidualUnit(FusedResidualUnit):
    """Same math and state layout as DirectResidualUnit; the 1x1 conv adds bias and residual in its epilogue."""

    def __init__(self, mod, slots: int, fast_sin: bool = False, tiles=None, dw_tiles=None):
        super().__init__(mod, slots, fast_sin)
        self.twopass_tiles = tiles or TWOPASS_TILES.get(self.C, (128, 64, 64, 4, 3))
        self.dw_tiles = dw_tiles or DW_TILES.get(self.C, (32, 64, 4))

    def __call__(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        B, C, T = x.shape
        h3 = torch.empty_like(x)
        bc, bt, dw_warps = self.dw_tiles
        bt = min(bt, max(16, triton.next_power_of_2(T)))
        _dw_snake12_h3[(B, triton.cdiv(T, bt), triton.cdiv(C, bc))](
            x, self.state, idx, h3, self.a1, self.i1, self.w7, self.b7, self.a2, self.i2,
            C, T, self.P, self.state.stride(0), DIL=self.dil, KSIZE=self.ksize, BC=bc, BT=bt,
            FAST_SIN=DIRECT_FAST_SIN, num_warps=dw_warps,
        )
        _snake_state[(B, triton.cdiv(C, 32))](
            x, self.state, idx, self.a1, self.i1, C, T, self.P, self.state.stride(0),
            BC=32, BP=triton.next_power_of_2(self.P), FAST_SIN=DIRECT_FAST_SIN,
        )
        out = torch.empty_like(x)
        bm, bn, bk, warps, stages = self.twopass_tiles
        bn = min(bn, max(16, triton.next_power_of_2(T)))
        _conv1x1_res[(triton.cdiv(T, bn), triton.cdiv(C, bm), B)](
            h3, x, self.w1, self.b1, out, C, T, *(self.post_snake or (self.a1, self.i1, self.a1, self.i1)),
            BM=bm, BN=bn, BK=bk, GROUP=1, POST_SNAKE=self.post_snake is not None, num_warps=warps, num_stages=stages,
        )
        return out


# (BT, BCO, num_warps) per channel count, tuned at B=256 on an A10 at the power cap.
# C=32 / 64: 2.16 / 2.31 -> 1.33 / 1.50 ms per unit. From C=128 the 1x1 conv is real
# GEMM work and the split unit (cuBLAS) stays faster (C=128: 2.36 vs 2.43 ms, C=256: 2.75 vs 4.52 ms).
ONEPASS_TILES = {32: (128, 32, 8), 64: (64, 64, 8), 128: (32, 64, 8), 256: (32, 128, 8)}


class OnePassResidualUnit(FusedResidualUnit):
    """Same math and state layout as DirectResidualUnit; depthwise conv, snakes, 1x1 conv and residual in one pass."""

    def __init__(self, mod, slots: int, fast_sin: bool = False, tiles=None):
        super().__init__(mod, slots, fast_sin)
        self.onepass_tiles = tiles or ONEPASS_TILES[self.C]
        assert self.C % self.onepass_tiles[1] == 0, "BCO must divide C"

    def __call__(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        B, C, T = x.shape
        out = torch.empty_like(x)
        bt, bco, warps = self.onepass_tiles
        bt = min(bt, max(16, triton.next_power_of_2(T)))
        _resunit_onepass[(B, triton.cdiv(T, bt))](
            x, self.state, idx, out, self.a1, self.i1, self.w7, self.b7, self.a2, self.i2, self.w1, self.b1,
            T, self.P, self.state.stride(0), *(self.post_snake or (self.a1, self.i1, self.a1, self.i1)), C=C, DIL=self.dil, KSIZE=self.ksize,
            BT=bt, BCO=bco, FAST_SIN=DIRECT_FAST_SIN, POST_SNAKE=self.post_snake is not None, num_warps=warps,
        )
        # the causal state advances after the conv has read the old one (programs run in any order)
        _snake_state[(B, triton.cdiv(C, 32))](
            x, self.state, idx, self.a1, self.i1, C, T, self.P, self.state.stride(0),
            BC=32, BP=triton.next_power_of_2(self.P), FAST_SIN=DIRECT_FAST_SIN,
        )
        return out


class CudaResidualUnit(FusedResidualUnit):
    """Same math and state layout as DirectResidualUnit; one CUDA kernel (cuda/resunit_fused.cu) with the snaked
    tile staged in shared memory and the 1x1 conv on tensor cores, then _snake_state advances the causal state."""

    def __init__(self, mod, slots: int, fast_sin: bool = False, variant: int = 0, sweep: bool = False):
        super().__init__(mod, slots, fast_sin)
        from .cuda_ext import load

        self.ext = load("resunit_fused", sweep)
        self.variant = variant  # tile shape (cuda/resunit_fused.cu, dispatch_c)
        self.params = [self.a1, self.i1, self.w7, self.b7, self.a2, self.i2, self.w1, self.b1]

    def __call__(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        B, C, T = x.shape
        out = torch.empty_like(x)
        post = self.post_snake is not None
        self.ext.resunit_fused(x, self.state, idx, out, self.dil, self.params + list(self.post_snake or self.params[:4]), post,
                               self.variant)
        _snake_state[(B, triton.cdiv(C, 32))](
            x, self.state, idx, self.a1, self.i1, C, T, self.P, self.state.stride(0),
            BC=32, BP=triton.next_power_of_2(self.P), FAST_SIN=DIRECT_FAST_SIN,
        )
        return out


class DirectResidualUnit(FusedResidualUnit):
    """Split unit without the padded snake buffer (fast-sin recompute per tap) + cuBLAS bmm for the 1x1 conv."""

    def __call__(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        B, C, T = x.shape
        h3 = torch.empty_like(x)
        out = torch.empty_like(x)
        bt = min(64, triton.next_power_of_2(T))
        _dw_snake12[(B, triton.cdiv(T, bt), triton.cdiv(C, 32))](
            x, self.state, idx, h3, out, self.a1, self.i1, self.w7, self.b7, self.a2, self.i2, self.b1,
            C, T, self.P, self.state.stride(0), DIL=self.dil, KSIZE=self.ksize, BC=32, BT=bt,
            FAST_SIN=DIRECT_FAST_SIN, num_warps=4,
        )
        _snake_state[(B, triton.cdiv(C, 32))](
            x, self.state, idx, self.a1, self.i1, C, T, self.P, self.state.stride(0),
            BC=32, BP=triton.next_power_of_2(self.P), FAST_SIN=DIRECT_FAST_SIN,
        )
        return out.baddbmm_(self.w1.unsqueeze(0).expand(B, C, C), h3)
