// One-pass streaming CausalResidualUnit of the AudioVAE decoder (CUDA C++, sm_80+):
//   out = x + b1 + W1 @ snake2(dwconv7_dil([state | snake1(x)]) + b7)   [ -> optional (affine +) snake of the next op ]
//
// A block owns BN time steps of one stream and all C output channels, and walks the input channels in chunks of KC:
//   1. snake1 of the chunk's x tile (16-byte loads) and of its causal halo (x, or the state for t < 0) -> smem, fp32
//   2. the dilated depthwise conv reads its 7 taps from smem (snake1 once per element, not once per tap), adds the
//      bias, applies snake2 and writes h3 [KC, BN] to smem as fp16
//   3. tensor cores (WMMA 16x16x16, fp32 accumulate): acc[C, BN] += W1[:, chunk] @ h3
// The epilogue adds bias and residual (and the post-snake) and writes the output once. Versus the Triton two-pass
// unit, h3 never goes to DRAM (x read twice, out written once, instead of 3 reads + 2 writes) and the sin count of
// the depthwise conv drops 7x. The causal state advances in a separate kernel (_snake_state) after this one.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

using namespace nvcuda;

namespace {

// x + sin^2(a x) / a with sin.approx, as the Triton kernels (libdevice fast_sinf): accurate to ~2^-20 for
// |a x| < 100 pi, and the decoder's activations stay below 10 (measured on the benchmark voices)
__device__ __forceinline__ float snake(float x, float a, float ia) {
  float s;
  asm("sin.approx.ftz.f32 %0, %1;" : "=f"(s) : "f"(a * x));
  return fmaf(s * s, ia, x);
}

template <int C, int DIL, int BN, int KC, int WM, int WN>
struct Cfg {
  static constexpr int K = 7;
  static constexpr int P = (K - 1) * DIL;
  static constexpr int PH = (P + 7) / 8 * 8;  // halo columns, rounded so the main tile starts 32-byte aligned
  static constexpr int TPC = 256 / KC;  // threads per channel in the depthwise pass
  static constexpr int LXS = (PH + BN + 31) / 32 * 32 + TPC;  // xs row stride (floats): conflict-free tap reads
  static constexpr int HS = BN + 16;  // h3 row stride (halves): multiple of 8 for WMMA, conflict-free stores
  static constexpr int WS = KC + 8;  // W1 chunk row stride (halves)
  static constexpr int FM = C / 16 / WM, FN = BN / 16 / WN;  // accumulator fragments per warp
  static constexpr int XS_BYTES = (KC * LXS * 4 > 8 * 256 * 4 ? KC * LXS * 4 : 8 * 256 * 4);  // >= epilogue scratch
  static constexpr int HS_BYTES = KC * HS * 2;
  static constexpr int SMEM = XS_BYTES + HS_BYTES + C * WS * 2;
  static_assert(WM * WN == 8, "8 warps");
  static_assert(C % (16 * WM) == 0 && BN % (16 * WN) == 0 && C % KC == 0 && KC % 16 == 0, "tiling");
  static_assert(256 % KC == 0 && BN % TPC == 0, "depthwise mapping");
  static_assert(XS_BYTES % 32 == 0 && HS_BYTES % 32 == 0, "alignment");
};

template <int C, int DIL, int BN, int KC, int WM, int WN, int MINB, bool POST>
__global__ void __launch_bounds__(256, MINB) resunit_fused_kernel(
    const half* __restrict__ x, const half* __restrict__ state, const int64_t* __restrict__ idx, half* __restrict__ out,
    const float* __restrict__ a1, const float* __restrict__ i1, const float* __restrict__ w7,
    const float* __restrict__ b7, const float* __restrict__ a2, const float* __restrict__ i2,
    const half* __restrict__ w1, const float* __restrict__ b1, const float* __restrict__ pa,
    const float* __restrict__ pia, const float* __restrict__ ps, const float* __restrict__ pb, int T,
    int64_t state_slot_stride) {
  using G = Cfg<C, DIL, BN, KC, WM, WN>;
  extern __shared__ __align__(128) unsigned char smem[];
  float* xs = reinterpret_cast<float*>(smem);
  half* hs = reinterpret_cast<half*>(smem + G::XS_BYTES);
  half* ws = reinterpret_cast<half*>(smem + G::XS_BYTES + G::HS_BYTES);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int b = blockIdx.y, t0 = blockIdx.x * BN;
  const half* xb = x + (int64_t)b * C * T;
  const half* sb = state + idx[b] * state_slot_stride;
  const int wm = warp / WN, wn = warp % WN;
  // T a multiple of 8 (every chunk of whole latent frames): time groups of 8 are 16-byte aligned and lie wholly
  // before 0 (state), inside [0, T) or past T, so every tile stages and stores by groups, the first and last too
  const bool t8 = (T % 8 == 0);

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[G::FM][G::FN];
#pragma unroll
  for (int i = 0; i < G::FM; i++)
#pragma unroll
    for (int j = 0; j < G::FN; j++) wmma::fill_fragment(acc[i][j], 0.0f);

  for (int kc = 0; kc < C; kc += KC) {
    __syncthreads();  // the previous chunk's MMAs have read hs / ws
    for (int e = tid; e < C * KC / 8; e += 256) {  // W1[:, kc:kc+KC] -> ws
      const int co = e / (KC / 8), q = e % (KC / 8);
      *reinterpret_cast<int4*>(ws + co * G::WS + q * 8) = *reinterpret_cast<const int4*>(w1 + (int64_t)co * C + kc + q * 8);
    }
    if (t8) {  // halo and main tile, times t0 - PH .. t0 + BN - 1, 8 per thread: x by 16-byte loads (zeros past T),
               // the causal state for t < 0 (columns before t0 - P are never read)
      constexpr int Q = (G::PH + BN) / 8;
      for (int e = tid; e < KC * Q; e += 256) {
        const int c = e / Q, q = e % Q, cg = kc + c, ts = t0 - G::PH + q * 8;
        float v[8];
        if (ts >= 0 && ts < T) {
          const int4 raw = *reinterpret_cast<const int4*>(xb + (int64_t)cg * T + ts);
          const half2* h = reinterpret_cast<const half2*>(&raw);
          const float a = a1[cg], ia = i1[cg];
#pragma unroll
          for (int u = 0; u < 4; u++) {
            const float2 f = __half22float2(h[u]);
            v[2 * u] = snake(f.x, a, ia);
            v[2 * u + 1] = snake(f.y, a, ia);
          }
        } else if (ts < 0) {
#pragma unroll
          for (int u = 0; u < 8; u++) v[u] = ts + u >= -G::P ? __half2float(sb[cg * G::P + ts + u + G::P]) : 0.f;
        } else {
#pragma unroll
          for (int u = 0; u < 8; u++) v[u] = 0.f;
        }
        float4* dst = reinterpret_cast<float4*>(xs + c * G::LXS + q * 8);
        dst[0] = make_float4(v[0], v[1], v[2], v[3]);
        dst[1] = make_float4(v[4], v[5], v[6], v[7]);
      }
    } else {  // T not a multiple of 8: one time step per thread
      for (int e = tid; e < KC * (G::P + BN); e += 256) {
        const int c = e / (G::P + BN), j = e % (G::P + BN), cg = kc + c, t = t0 - G::P + j;
        float v;
        if (t >= 0)
          v = t < T ? snake(__half2float(xb[(int64_t)cg * T + t]), a1[cg], i1[cg]) : 0.f;
        else
          v = __half2float(sb[cg * G::P + t + G::P]);
        xs[c * G::LXS + G::PH - G::P + j] = v;
      }
    }
    __syncthreads();
    {  // depthwise conv + snake2 -> hs (fp16); thread = (channel, every TPC-th time step)
      const int c = tid / G::TPC, sub = tid % G::TPC, cg = kc + c;
      float w[G::K];
#pragma unroll
      for (int k = 0; k < G::K; k++) w[k] = w7[cg * G::K + k];
      const float bias = b7[cg], a = a2[cg], ia = i2[cg];
      const float* row = xs + c * G::LXS + G::PH - G::P;
      for (int t = sub; t < BN; t += G::TPC) {
        float h = bias;
#pragma unroll
        for (int k = 0; k < G::K; k++) h = fmaf(w[k], row[t + k * DIL], h);
        h = snake(h, a, ia);
        hs[c * G::HS + t] = __float2half_rn(t0 + t < T ? h : 0.f);
      }
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < KC; kk += 16) {  // B fragments once, A fragments one at a time: fewer live registers
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[G::FN];
#pragma unroll
      for (int j = 0; j < G::FN; j++) wmma::load_matrix_sync(bf[j], hs + kk * G::HS + (wn * G::FN + j) * 16, G::HS);
#pragma unroll
      for (int i = 0; i < G::FM; i++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
        wmma::load_matrix_sync(af, ws + (wm * G::FM + i) * 16 * G::WS + kk, G::WS);
#pragma unroll
        for (int j = 0; j < G::FN; j++) wmma::mma_sync(acc[i][j], af, bf[j], acc[i][j]);
      }
    }
  }
  __syncthreads();  // xs is free: per-warp 16x16 scratch for the epilogue
  float* scratch = xs + warp * 256;
  half* ob = out + (int64_t)b * C * T;
#pragma unroll
  for (int i = 0; i < G::FM; i++) {
#pragma unroll
    for (int j = 0; j < G::FN; j++) {
      wmma::store_matrix_sync(scratch, acc[i][j], 16, wmma::mem_row_major);
      __syncwarp();
      // lane -> one row, 8 consecutive time steps: per-channel parameters once per 8 outputs, 16-byte residual
      // loads and output stores (two lanes cover a 32-byte sector)
      const int co = (wm * G::FM + i) * 16 + (lane >> 1), c8 = (lane & 1) * 8;
      const int t = t0 + (wn * G::FN + j) * 16 + c8;
      const float4 s0 = *reinterpret_cast<const float4*>(scratch + (lane >> 1) * 16 + c8);
      const float4 s1 = *reinterpret_cast<const float4*>(scratch + (lane >> 1) * 16 + c8 + 4);
      float y[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
      const float bias = b1[co];
      const int64_t off = (int64_t)co * T + t;
      const bool vec = t8 && t < T;  // 8 time steps inside [0, T); with t8, a group past T is skipped whole
      if (vec) {
        const int4 raw = *reinterpret_cast<const int4*>(xb + off);
        const half2* h = reinterpret_cast<const half2*>(&raw);
#pragma unroll
        for (int u = 0; u < 4; u++) {
          const float2 f = __half22float2(h[u]);
          y[2 * u] += bias + f.x;
          y[2 * u + 1] += bias + f.y;
        }
      } else {
#pragma unroll
        for (int u = 0; u < 8; u++) y[u] += bias + (t + u < T ? __half2float(xb[off + u]) : 0.f);
      }
      if (POST) {
        const float sc = ps[co], sh = pb[co], a = pa[co], ia = pia[co];
#pragma unroll
        for (int u = 0; u < 8; u++) y[u] = snake(fmaf(y[u], sc, sh), a, ia);
      }
      if (vec) {
        int4 packed;
        half2* o = reinterpret_cast<half2*>(&packed);
#pragma unroll
        for (int u = 0; u < 4; u++) o[u] = __floats2half2_rn(y[2 * u], y[2 * u + 1]);
        *reinterpret_cast<int4*>(ob + off) = packed;
      } else if (!t8) {
#pragma unroll
        for (int u = 0; u < 8; u++)
          if (t + u < T) ob[off + u] = __float2half_rn(y[u]);
      }
      __syncwarp();
    }
  }
}

template <int C, int DIL, int BN, int KC, int WM, int WN, int MINB, bool POST>
void launch(const torch::Tensor& x, const torch::Tensor& state, const torch::Tensor& idx, torch::Tensor& out,
            const std::vector<torch::Tensor>& p, cudaStream_t stream) {
  using G = Cfg<C, DIL, BN, KC, WM, WN>;
  auto kern = resunit_fused_kernel<C, DIL, BN, KC, WM, WN, MINB, POST>;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    attr = true;
  }
  const int B = x.size(0), T = x.size(2);
  dim3 grid((T + BN - 1) / BN, B);
  kern<<<grid, 256, G::SMEM, stream>>>(
      (const half*)x.data_ptr(), (const half*)state.data_ptr(), idx.data_ptr<int64_t>(), (half*)out.data_ptr(),
      p[0].data_ptr<float>(), p[1].data_ptr<float>(), p[2].data_ptr<float>(), p[3].data_ptr<float>(),
      p[4].data_ptr<float>(), p[5].data_ptr<float>(), (const half*)p[6].data_ptr(), p[7].data_ptr<float>(),
      p[8].data_ptr<float>(), p[9].data_ptr<float>(), p[10].data_ptr<float>(), p[11].data_ptr<float>(), T,
      state.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Tile shapes per channel count: <BN, KC, WM, WN, MINB> (time tile, channel chunk, warp grid, min blocks per SM).
// Variant 0 is the default (sweep_resunit_cuda.py at B=256 on an A10 at the power cap: C=32 / 64 per unit
// 1.23-1.26 / 1.38-1.47 ms with <256, 16, 1, 8, 1> -> 0.98-1.00 / 1.17-1.21 ms; C=128 1.61-1.71 ms with
// <128, 32, 2, 4, 1> -> 1.38-1.46 ms, the Triton two-pass unit 1.57-1.70 ms); the others are compiled for the sweep only.
#define L(...) launch<C, DIL, __VA_ARGS__, POST>(x, state, idx, out, p, s); return
template <int C, int DIL, bool POST>
void dispatch_c(int variant, const torch::Tensor& x, const torch::Tensor& state, const torch::Tensor& idx,
                torch::Tensor& out, const std::vector<torch::Tensor>& p, cudaStream_t s) {
  if constexpr (C == 32) {
    switch (variant) {
      case 0: L(128, 32, 2, 4, 2);
#ifdef RESUNIT_SWEEP
      case 1: L(256, 16, 1, 8, 2);
      case 2: L(128, 32, 1, 8, 2);
      case 3: L(512, 16, 1, 8, 1);
      case 4: L(256, 16, 1, 8, 1);
      case 5: L(256, 32, 1, 8, 1);
      case 6: L(128, 32, 2, 4, 3);
      case 7: L(128, 32, 1, 8, 3);
      case 8: L(64, 32, 2, 4, 3);
      case 9: L(128, 16, 2, 4, 3);
#endif
    }
  } else if constexpr (C == 64) {
    switch (variant) {
      case 0: L(128, 32, 2, 4, 2);
#ifdef RESUNIT_SWEEP
      case 1: L(256, 16, 1, 8, 2);
      case 2: L(256, 16, 1, 8, 1);
      case 3: L(128, 64, 2, 4, 1);
      case 4: L(128, 16, 2, 4, 2);
      case 5: L(256, 32, 1, 8, 1);
      case 6: L(128, 32, 2, 4, 3);
      case 7: L(64, 32, 4, 2, 2);
      case 8: L(64, 32, 4, 2, 3);
      case 9: L(128, 16, 4, 2, 2);
#endif
    }
  } else if constexpr (C == 128) {
    switch (variant) {
      case 0: L(128, 32, 2, 4, 2);
#ifdef RESUNIT_SWEEP
      case 1: L(128, 32, 2, 4, 1);
      case 2: L(64, 32, 2, 4, 2);
      case 3: L(64, 32, 4, 2, 2);
      case 4: L(128, 32, 4, 2, 2);
      case 5: L(64, 16, 4, 2, 3);
      case 6: L(64, 32, 4, 2, 3);
#endif
    }
  } else {
    switch (variant) {
      case 0: L(64, 32, 4, 2, 2);
#ifdef RESUNIT_SWEEP
      case 1: L(64, 32, 4, 2, 1);
      case 2: L(32, 32, 4, 2, 2);
      case 3: L(64, 32, 8, 1, 2);
      case 4: L(32, 32, 8, 1, 2);
      case 5: L(32, 16, 8, 1, 3);
      case 6: L(64, 16, 4, 2, 2);
#endif
    }
  }
  TORCH_CHECK(false, "resunit_fused: variant ", variant, " not compiled for C=", C);
}
#undef L

template <int C, bool POST>
void dispatch_dil(int variant, int dil, const torch::Tensor& x, const torch::Tensor& state, const torch::Tensor& idx,
                  torch::Tensor& out, const std::vector<torch::Tensor>& p, cudaStream_t s) {
  switch (dil) {
    case 1: dispatch_c<C, 1, POST>(variant, x, state, idx, out, p, s); break;
    case 3: dispatch_c<C, 3, POST>(variant, x, state, idx, out, p, s); break;
    case 9: dispatch_c<C, 9, POST>(variant, x, state, idx, out, p, s); break;
    default: TORCH_CHECK(false, "dilation ", dil, " not compiled");
  }
}

template <bool POST>
void dispatch(int variant, int dil, const torch::Tensor& x, const torch::Tensor& state, const torch::Tensor& idx, torch::Tensor& out,
              const std::vector<torch::Tensor>& p, cudaStream_t s) {
  switch (x.size(1)) {
    case 32: dispatch_dil<32, POST>(variant, dil, x, state, idx, out, p, s); break;
    case 64: dispatch_dil<64, POST>(variant, dil, x, state, idx, out, p, s); break;
    case 128: dispatch_dil<128, POST>(variant, dil, x, state, idx, out, p, s); break;
    case 256: dispatch_dil<256, POST>(variant, dil, x, state, idx, out, p, s); break;
    default: TORCH_CHECK(false, "channels ", x.size(1), " not compiled");
  }
}

}  // namespace

// params: a1, i1, w7 [C, 7], b7, a2, i2 (fp32), w1 [C, C] (fp16), b1, then pa, pia, ps, pb (fp32, read only if post)
void resunit_fused(torch::Tensor x, torch::Tensor state, torch::Tensor idx, torch::Tensor out, int64_t dil,
                   std::vector<torch::Tensor> params, bool post, int64_t variant) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kHalf && x.is_contiguous() && x.dim() == 3, "x: [B, C, T] fp16");
  TORCH_CHECK(state.scalar_type() == torch::kHalf && state.stride(2) == 1 && state.stride(1) == 6 * dil, "state");
  TORCH_CHECK(idx.scalar_type() == torch::kLong && params.size() == 12, "idx / params");
  const c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t s = c10::cuda::getCurrentCUDAStream();
  if (post) dispatch<true>(variant, dil, x, state, idx, out, params, s);
  else dispatch<false>(variant, dil, x, state, idx, out, params, s);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("resunit_fused", &resunit_fused, "fused streaming residual unit"); }
