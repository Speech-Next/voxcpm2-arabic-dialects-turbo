// Output head of the AudioVAE decoder (CUDA C++): tanh(b + conv1d_causal(snake(x), W[1, C, K])), streaming.
//
// A block owns BN samples of one stream: snake of the [C, BN] tile (16-byte loads) and of its K-1 sample causal halo
// (x, or the state = snake of the previous step's tail for t < 0) go to shared memory once; each thread then sums
// C x K taps from shared memory for its output sample. The Triton head recomputed the snake per tap from scalar,
// misaligned global loads. The state advances in a separate kernel (_snake_state) afterwards.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

__device__ __forceinline__ float snake(float x, float a, float ia) {
  float s;
  asm("sin.approx.ftz.f32 %0, %1;" : "=f"(s) : "f"(a * x));
  return fmaf(s * s, ia, x);
}

// 256 threads = 4 channel groups x 64 time groups; a thread sums CG = C / 4 channels for 4 consecutive outputs from a
// 10-sample window (three aligned 16-byte shared loads per channel, weights in registers), and the 4 partial sums
// per output are added through shared memory.
template <int C, int K, int BN>
__global__ void __launch_bounds__(256) head_kernel(const half* __restrict__ x, const half* __restrict__ state,
                                                   const int64_t* __restrict__ idx, half* __restrict__ out,
                                                   const float* __restrict__ a, const float* __restrict__ ia,
                                                   const float* __restrict__ w, float bias, int T,
                                                   int64_t state_slot_stride) {
  static_assert(K == 7 && BN == 256 && C % 4 == 0, "4 outputs x 64 time groups per block, 7 taps");
  constexpr int P = K - 1, PH = 8, LX = PH + BN + 8, CG = C / 4;  // column j <-> time t0 - PH + j
  __shared__ __align__(16) float xs[C * LX];
  __shared__ float red[4 * BN];
  const int tid = threadIdx.x, b = blockIdx.y, t0 = blockIdx.x * BN;
  const half* xb = x + (int64_t)b * C * T;
  const half* sb = state + idx[b] * state_slot_stride;
  if (T % 8 == 0 && t0 >= PH && t0 + BN <= T) {  // interior tile: halo and tile, 8 samples per load
    constexpr int Q = (PH + BN) / 8;
    for (int e = tid; e < C * Q; e += 256) {
      const int c = e / Q, q = e % Q;
      const int4 raw = *reinterpret_cast<const int4*>(xb + (int64_t)c * T + t0 - PH + q * 8);
      const half2* h = reinterpret_cast<const half2*>(&raw);
      const float ac = a[c], iac = ia[c];
      float v[8];
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const float2 f = __half22float2(h[u]);
        v[2 * u] = snake(f.x, ac, iac);
        v[2 * u + 1] = snake(f.y, ac, iac);
      }
      float4* dst = reinterpret_cast<float4*>(xs + c * LX + q * 8);
      dst[0] = make_float4(v[0], v[1], v[2], v[3]);
      dst[1] = make_float4(v[4], v[5], v[6], v[7]);
    }
  } else {
    for (int e = tid; e < C * (P + BN); e += 256) {
      const int c = e / (P + BN), j = e % (P + BN), t = t0 - P + j;
      float v;
      if (t >= 0)
        v = t < T ? snake(__half2float(xb[(int64_t)c * T + t]), a[c], ia[c]) : 0.f;
      else
        v = __half2float(sb[c * P + t + P]);
      xs[c * LX + PH - P + j] = v;
    }
  }
  __syncthreads();
  const int cg = tid / 64, tg = tid % 64;  // a warp: 32 consecutive time groups of one channel group
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 2
  for (int cc = 0; cc < CG; cc++) {
    const int c = cg * CG + cc;
    float wk[K];
#pragma unroll
    for (int k = 0; k < K; k++) wk[k] = w[c * K + k];
    // outputs t0 + 4 tg + o read columns 4 tg + o + PH - P + k = 4 tg + 2 + o + k  (0 <= o + k <= 9)
    const float4* src = reinterpret_cast<const float4*>(xs + c * LX + 4 * tg);
    const float4 v0 = src[0], v1 = src[1], v2 = src[2];
    const float win[12] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w, v2.x, v2.y, v2.z, v2.w};
#pragma unroll
    for (int o = 0; o < 4; o++)
#pragma unroll
      for (int k = 0; k < K; k++) acc[o] = fmaf(wk[k], win[PH - P + o + k], acc[o]);
  }
#pragma unroll
  for (int o = 0; o < 4; o++) red[cg * BN + 4 * tg + o] = acc[o];
  __syncthreads();
  const int t = t0 + tid;
  if (t >= T) return;
  const float y = bias + red[tid] + red[BN + tid] + red[2 * BN + tid] + red[3 * BN + tid];
  const float e = __expf(-2.f * fabsf(y)), th = (1.f - e) / (1.f + e);  // overflow-safe tanh
  out[(int64_t)b * T + t] = __float2half_rn(copysignf(th, y));
}

}  // namespace

void head_fused(torch::Tensor x, torch::Tensor state, torch::Tensor idx, torch::Tensor out, torch::Tensor a,
                torch::Tensor ia, torch::Tensor w, double bias) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kHalf && x.is_contiguous() && x.dim() == 3, "x: [B, C, T] fp16");
  TORCH_CHECK(x.size(1) == 32 && w.numel() == 32 * 7 && state.stride(1) == 6 && state.stride(2) == 1,
              "compiled for C=32, K=7");
  const c10::cuda::CUDAGuard guard(x.device());
  const int B = x.size(0), T = x.size(2);
  constexpr int BN = 256;
  dim3 grid((T + BN - 1) / BN, B);
  head_kernel<32, 7, BN><<<grid, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      (const half*)x.data_ptr(), (const half*)state.data_ptr(), idx.data_ptr<int64_t>(), (half*)out.data_ptr(),
      a.data_ptr<float>(), ia.data_ptr<float>(), w.data_ptr<float>(), (float)bias, T, state.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("head_fused", &head_fused, "fused streaming output head"); }
