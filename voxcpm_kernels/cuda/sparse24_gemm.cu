// 2:4 sparse int8 GEMMs for the LM's FFN (CUDA C++): mma.sp.m16n8k64 (s8) on Ampere.
//
// C^T = W (2:4 sparse along K, compressed) x X^T: the sparse weight is the MMA's A operand (16 output channels per
// tile), the tokens are its N (8 per tile), so a few tokens waste little. The weight is packed offline into the
// per-lane fragment order (sparse24.py: 16 B of values + 4 B of metadata per lane, tile and 64-wide k block), the
// activations go through shared memory (rows of 128 k bytes, 16-byte chunks XOR-swizzled by row) and ldmatrix.x4.
// int8 products are summed in int32 exactly, in any order, so the accumulators equal the dense kernel's on the
// pruned weight; the epilogues repeat the dense Triton kernels' float operations one for one (fused_layer.py):
//   EPI_RES     out = bf16((f32(acc) * a_scale) * b_scale + res), as one fma or two roundings (fma_res: like the
//               dense kernel's tile configuration compiles it)
//   EPI_SWIGLU  h = bf16(silu(g) * u), g / u from the gate / up rows (rows 0-7 / 8-15 of every 16-row tile), plus
//               the per-(row, tile) max |h| for _quant_rows_partials
// Both epilogues go through shared memory and store 16-byte rows.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int EPI_RES = 0, EPI_SWIGLU = 1;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr)
               : "memory");
}

__device__ __forceinline__ void mma_sp(int (&d)[4], const int4& a, const uint32_t (&b)[4], uint32_t e) {
  asm("mma.sp.sync.aligned.m16n8k64.row.col.satfinite.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(e));
}

// The dense SwiGLU epilogue's operations (Triton 3.4 PTX of _w8a8_gemm_swiglu): g = (acc * as) * bs,
// -g = fma(-(acc * as), bs, 0), exp(-g) = ex2.approx(-g * log2 e), silu(g) = div.full(g, 1 + exp(-g)); h = silu(g) * u.
__device__ __forceinline__ float swiglu(int ag, int au, float a_s, float bg, float bu) {
  const float xg = __fmul_rn(__int2float_rn(ag), a_s);
  const float g = __fmul_rn(xg, bg);
  const float t = __fmul_rn(__fmaf_rn(-xg, bg, 0.f), __int_as_float(0x3FB8AA3B));
  float e, q;
  asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(t));
  asm("div.full.f32 %0, %1, %2;" : "=f"(q) : "f"(g), "f"(__fadd_rn(e, 1.f)));
  const float u = __fmul_rn(__fmul_rn(__int2float_rn(au), a_s), bu);
  return __fmul_rn(q, u);
}

template <int BO_, int BT_, int WO_, int WT_, int STAGES_>
struct Cfg {
  static constexpr int BO = BO_, BT = BT_, WO = WO_, WT = WT_, STAGES = STAGES_;
  static constexpr int WARPS_O = BO / WO, WARPS_T = BT / WT, WARPS = WARPS_O * WARPS_T, THREADS = 32 * WARPS;
  static constexpr int FO = WO / 16, FT = WT / 8;  // A (16 channels) / B (8 tokens) fragments per warp
  static constexpr int A_BYTES = BO / 16 * 2 * 512;  // per stage: 128 logical k = 2 k blocks of 64
  static constexpr int M_BYTES = BO / 16 * 2 * 128;
  static constexpr int B_BYTES = BT * 128;
  static constexpr int STAGE = A_BYTES + M_BYTES + B_BYTES;
  static constexpr int RES_PITCH = BO + 4;      // int32 words per token row of the staged accumulators
  static constexpr int H_PITCH = BO / 2 + 8;    // bf16 per token row of the staged h
  static constexpr int EPI_RES_BYTES = BT * RES_PITCH * 4;
  static constexpr int EPI_H_BYTES = BT * H_PITCH * 2;
  static_assert(BO % WO == 0 && BT % WT == 0 && WO % 16 == 0 && WT % 8 == 0, "tile shapes");
  static_assert(STAGES * STAGE <= 99 * 1024, "A10: 99 KB of shared memory per block");
};

// shared memory of a kernel: the pipeline, reused by the epilogue's staging; 0 = does not fit (that epilogue is not
// available in this tile configuration)
template <class C, int EPI>
constexpr int smem_bytes() {
  constexpr int pipe = C::STAGES * C::STAGE, epi = EPI == EPI_RES ? C::EPI_RES_BYTES : C::EPI_H_BYTES;
  return (pipe > epi ? pipe : epi) <= 99 * 1024 ? (pipe > epi ? pipe : epi) : 0;
}

template <class C, int EPI>
__global__ void __launch_bounds__(C::THREADS)
    sp24_gemm_kernel(const uint8_t* __restrict__ wf, const uint8_t* __restrict__ mf, const int8_t* __restrict__ x,
                     const float* __restrict__ a_scale, const float* __restrict__ b_scale,
                     const __nv_bfloat16* __restrict__ res, __nv_bfloat16* __restrict__ out, float* __restrict__ amax,
                     int M, int N, int K, int NT, int fma_res) {
  extern __shared__ __align__(16) uint8_t smem[];
  const int TT = (M + C::BT - 1) / C::BT;
  const int tt = blockIdx.x % TT, ot = blockIdx.x / TT;  // token tiles first: one weight tile serves them from L2
  const int m_base = tt * C::BT, o_base = ot * C::BO;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wo = warp / C::WARPS_T, wt = warp % C::WARPS_T;
  const int KB = K / 64, KT = K / 128, R0 = o_base / 16;

  auto load_stage = [&](int stage, int kt) {
    uint8_t* s = smem + stage * C::STAGE;
    for (int c = tid; c < C::BO / 16 * 64; c += C::THREADS) {  // weights: 1 KB per 16-channel tile
      const int r = c >> 6, q = c & 63;
      cp_async16(smem_u32(s + r * 1024 + q * 16), wf + ((int64_t)(R0 + r) * KB + 2 * kt) * 512 + q * 16, 16);
    }
    uint8_t* sm = s + C::A_BYTES;
    for (int c = tid; c < C::BO / 16 * 16; c += C::THREADS) {  // metadata: 256 B per tile
      const int r = c >> 4, q = c & 15;
      cp_async16(smem_u32(sm + r * 256 + q * 16), mf + ((int64_t)(R0 + r) * KB + 2 * kt) * 128 + q * 16, 16);
    }
    uint8_t* sb = sm + C::M_BYTES;
    for (int c = tid; c < C::BT * 8; c += C::THREADS) {  // activations: 128 B per token, zeros past M
      const int r = c >> 3, q = c & 7, m = m_base + r;
      cp_async16(smem_u32(sb + r * 128 + ((q ^ (r & 7)) << 4)), x + (int64_t)min(m, M - 1) * K + kt * 128 + q * 16,
                 m < M ? 16 : 0);
    }
  };

  int acc[C::FO][C::FT][4];
#pragma unroll
  for (int i = 0; i < C::FO; i++)
#pragma unroll
    for (int j = 0; j < C::FT; j++)
#pragma unroll
      for (int v = 0; v < 4; v++) acc[i][j][v] = 0;

#pragma unroll
  for (int s = 0; s < C::STAGES - 1; s++) {
    if (s < KT) load_stage(s, s);
    cp_commit();
  }
  for (int kt = 0; kt < KT; kt++) {
    cp_wait<C::STAGES - 2>();
    __syncthreads();
    {
      const int nk = kt + C::STAGES - 1;
      if (nk < KT) load_stage(nk % C::STAGES, nk);
      cp_commit();
    }
    const uint8_t* s = smem + (kt % C::STAGES) * C::STAGE;
    const int4* sa = reinterpret_cast<const int4*>(s);
    const uint32_t* sm = reinterpret_cast<const uint32_t*>(s + C::A_BYTES);
    const uint32_t sb = smem_u32(s + C::A_BYTES + C::M_BYTES);
#pragma unroll
    for (int kb = 0; kb < 2; kb++) {
      int4 a[C::FO];
      uint32_t e[C::FO], b[C::FT][4];
#pragma unroll
      for (int i = 0; i < C::FO; i++) {
        const int r = wo * C::FO + i;
        a[i] = sa[(r * 2 + kb) * 32 + lane];
        e[i] = sm[(r * 2 + kb) * 32 + lane];
      }
#pragma unroll
      for (int j = 0; j < C::FT; j++) {
        const int row = wt * C::WT + j * 8 + (lane & 7), chunk = kb * 4 + (lane >> 3);
        ldsm_x4(sb + row * 128 + ((chunk ^ (row & 7)) << 4), b[j]);
      }
#pragma unroll
      for (int i = 0; i < C::FO; i++)
#pragma unroll
        for (int j = 0; j < C::FT; j++) mma_sp(acc[i][j], a[i], b[j], e[i]);
    }
  }
  cp_wait<0>();
  __syncthreads();

  const int g = lane >> 2, t = lane & 3;
  if constexpr (EPI == EPI_RES) {
    int* st = reinterpret_cast<int*>(smem);
#pragma unroll
    for (int i = 0; i < C::FO; i++)
#pragma unroll
      for (int j = 0; j < C::FT; j++) {
        const int o = wo * C::WO + i * 16 + g, m = wt * C::WT + j * 8 + 2 * t;
        st[m * C::RES_PITCH + o] = acc[i][j][0];
        st[(m + 1) * C::RES_PITCH + o] = acc[i][j][1];
        st[m * C::RES_PITCH + o + 8] = acc[i][j][2];
        st[(m + 1) * C::RES_PITCH + o + 8] = acc[i][j][3];
      }
    __syncthreads();
    constexpr int CH = C::BO / 8;
    for (int c = tid; c < C::BT * CH; c += C::THREADS) {
      const int r = c / CH, q = c % CH, m = m_base + r;
      if (m >= M) continue;
      const int o = o_base + q * 8;
      const float a_s = a_scale[m];
      const int4 a0 = *reinterpret_cast<const int4*>(st + r * C::RES_PITCH + q * 8);
      const int4 a1 = *reinterpret_cast<const int4*>(st + r * C::RES_PITCH + q * 8 + 4);
      const int av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
      const float4 b0 = *reinterpret_cast<const float4*>(b_scale + o);
      const float4 b1 = *reinterpret_cast<const float4*>(b_scale + o + 4);
      const float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
      const int4 rr = *reinterpret_cast<const int4*>(res + (int64_t)m * N + o);
      const __nv_bfloat16* rv = reinterpret_cast<const __nv_bfloat16*>(&rr);
      int4 ov;
      __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(&ov);
#pragma unroll
      for (int u = 0; u < 8; u++) {
        const float v = __fmul_rn(__int2float_rn(av[u]), a_s), r32 = __bfloat162float(rv[u]);
        op[u] = __float2bfloat16_rn(fma_res ? __fmaf_rn(v, bv[u], r32) : __fadd_rn(__fmul_rn(v, bv[u]), r32));
      }
      *reinterpret_cast<int4*>(out + (int64_t)m * N + o) = ov;
    }
  } else {
    __nv_bfloat16* sh = reinterpret_cast<__nv_bfloat16*>(smem);
#pragma unroll
    for (int i = 0; i < C::FO; i++) {
      const int orow = o_base + wo * C::WO + i * 16 + g;  // gate row; its up row is orow + 8
      const float bg = b_scale[orow], bu = b_scale[orow + 8];
      const int ch = (wo * C::WO + i * 16) / 2 + g;
#pragma unroll
      for (int j = 0; j < C::FT; j++) {
        const int m = wt * C::WT + j * 8 + 2 * t;
        const float s0 = m_base + m < M ? a_scale[m_base + m] : 0.f;
        const float s1 = m_base + m + 1 < M ? a_scale[m_base + m + 1] : 0.f;
        sh[m * C::H_PITCH + ch] = __float2bfloat16_rn(swiglu(acc[i][j][0], acc[i][j][2], s0, bg, bu));
        sh[(m + 1) * C::H_PITCH + ch] = __float2bfloat16_rn(swiglu(acc[i][j][1], acc[i][j][3], s1, bg, bu));
      }
    }
    __syncthreads();
    constexpr int CH = C::BO / 16;  // 16-byte chunks (8 channels) per token row of h
    static_assert(C::THREADS % CH == 0 && (C::BT * CH) % 32 == 0 && CH <= 32, "row groups in one warp");
    const int half = N / 2;
    for (int c = tid; c < C::BT * CH; c += C::THREADS) {  // warp-uniform: BT * CH is a multiple of 32
      const int r = c / CH, q = c % CH, m = m_base + r;
      const int4 hv = *reinterpret_cast<const int4*>(sh + r * C::H_PITCH + q * 8);
      const __nv_bfloat16* hp = reinterpret_cast<const __nv_bfloat16*>(&hv);
      float mx = 0.f;
#pragma unroll
      for (int u = 0; u < 8; u++) mx = fmaxf(mx, fabsf(__bfloat162float(hp[u])));
#pragma unroll
      for (int w = 1; w < CH; w <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, w));
      if (m < M) {
        *reinterpret_cast<int4*>(out + (int64_t)m * half + o_base / 2 + q * 8) = hv;
        if (q == 0) amax[(int64_t)m * NT + ot] = mx;
      }
    }
  }
}

template <class C, int EPI>
void launch(const torch::Tensor& wf, const torch::Tensor& mf, const torch::Tensor& x, const torch::Tensor& as,
            const torch::Tensor& bs, const torch::Tensor* res, torch::Tensor& out, torch::Tensor* amax, int N,
            int fma_res) {
  constexpr int SMEM = smem_bytes<C, EPI>();
  if constexpr (SMEM == 0) {
    TORCH_CHECK(false, "sp24: this tile configuration has no ", EPI == EPI_RES ? "residual" : "SwiGLU", " epilogue");
  } else {
  const int M = x.size(0), K = x.size(1);
  TORCH_CHECK(N % C::BO == 0 && K % 128 == 0, "sp24: N must be a multiple of ", C::BO, ", K of 128");
  auto kern = sp24_gemm_kernel<C, EPI>;
  static bool attr = false;  // per instantiation
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    // the largest shared-memory carveout: configurations of <= 49 KB then fit 2 CTAs per SM
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    attr = true;
  }
  const int grid = (M + C::BT - 1) / C::BT * (N / C::BO);
  auto stream = at::cuda::getCurrentCUDAStream();
  kern<<<grid, C::THREADS, SMEM, stream>>>(
      wf.data_ptr<uint8_t>(), mf.data_ptr<uint8_t>(), x.data_ptr<int8_t>(), as.data_ptr<float>(),
      bs.data_ptr<float>(), res ? reinterpret_cast<const __nv_bfloat16*>(res->data_ptr()) : nullptr,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), amax ? amax->data_ptr<float>() : nullptr, M, N, K,
      amax ? (int)amax->size(1) : 0, fma_res);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

// tile configurations (BO, BT, WO, WT, STAGES); sparse24.py picks one by row count
using C0 = Cfg<128, 128, 64, 32, 3>;  // 8 warps
using C1 = Cfg<128, 64, 64, 32, 4>;   // 4 warps
using C2 = Cfg<128, 32, 32, 32, 4>;   // 4 warps
using C3 = Cfg<64, 16, 16, 16, 4>;    // 4 warps
using C4 = Cfg<64, 8, 16, 8, 4>;      // 4 warps
using C5 = Cfg<64, 128, 32, 32, 4>;   // 8 warps
using C6 = Cfg<128, 128, 64, 64, 3>;  // 4 warps
using C7 = Cfg<64, 32, 16, 32, 4>;    // 4 warps
using C8 = Cfg<128, 64, 32, 32, 2>;   // 8 warps, 2 CTAs / SM
using C9 = Cfg<128, 64, 64, 32, 2>;   // 4 warps, 2 CTAs / SM
using C10 = Cfg<64, 128, 32, 32, 2>;  // 8 warps, 2 CTAs / SM
using C11 = Cfg<128, 128, 64, 32, 2>;  // 8 warps
using C12 = Cfg<256, 128, 64, 64, 2>;  // 8 warps of 64 x 64 (SwiGLU only)
using C13 = Cfg<128, 256, 64, 64, 2>;  // 8 warps of 64 x 64 (SwiGLU only)

template <int EPI>
void dispatch(int cfg, const torch::Tensor& wf, const torch::Tensor& mf, const torch::Tensor& x,
              const torch::Tensor& as, const torch::Tensor& bs, const torch::Tensor* res, torch::Tensor& out,
              torch::Tensor* amax, int N, int fma_res) {
  switch (cfg) {
    case 0: return launch<C0, EPI>(wf, mf, x, as, bs, res, out, amax, N, fma_res);
    case 1: return launch<C1, EPI>(wf, mf, x, as, bs, res, out, amax, N, fma_res);
    case 2: return launch<C2, EPI>(wf, mf, x, as, bs, res, out, amax, N, fma_res);
    case 3: return launch<C3, EPI>(wf, mf, x, as, bs, res, out, amax, N, fma_res);
    case 4: return launch<C4, EPI>(wf, mf, x, as, bs, res, out, amax, N, fma_res);
    case 5: return launch<C5, EPI>(wf, mf, x, as, bs, res, out, amax, N, fma_res);
    case 6: return launch<C6, EPI>(wf, mf, x, as, bs, res, out, amax, N, fma_res);
    case 7: return launch<C7, EPI>(wf, mf, x, as, bs, res, out, amax, N, fma_res);
    case 8: return launch<C8, EPI>(wf, mf, x, as, bs, res, out, amax, N, fma_res);
    case 9: return launch<C9, EPI>(wf, mf, x, as, bs, res, out, amax, N, fma_res);
    case 10: return launch<C10, EPI>(wf, mf, x, as, bs, res, out, amax, N, fma_res);
    case 11: return launch<C11, EPI>(wf, mf, x, as, bs, res, out, amax, N, fma_res);
    case 12: return launch<C12, EPI>(wf, mf, x, as, bs, res, out, amax, N, fma_res);
    case 13: return launch<C13, EPI>(wf, mf, x, as, bs, res, out, amax, N, fma_res);
  }
  TORCH_CHECK(false, "sp24: unknown tile configuration ", cfg);
}

void check(const torch::Tensor& wf, const torch::Tensor& mf, const torch::Tensor& x, const torch::Tensor& as,
           const torch::Tensor& bs) {
  TORCH_CHECK(wf.is_cuda() && wf.scalar_type() == at::kByte && wf.is_contiguous(), "sp24: wf uint8 contiguous");
  TORCH_CHECK(mf.scalar_type() == at::kByte && mf.is_contiguous(), "sp24: mf uint8 contiguous");
  TORCH_CHECK(x.scalar_type() == at::kChar && x.is_contiguous() && x.dim() == 2, "sp24: x int8 [M, K] contiguous");
  TORCH_CHECK(as.scalar_type() == at::kFloat && bs.scalar_type() == at::kFloat && bs.is_contiguous(), "sp24: scales");
}

}  // namespace

// out [M, N] bf16 = (x @ W^T) * as * bs + res
void sp24_gemm_res(torch::Tensor wf, torch::Tensor mf, torch::Tensor x, torch::Tensor as, torch::Tensor bs,
                   torch::Tensor res, torch::Tensor out, int64_t cfg, int64_t fma_res) {
  check(wf, mf, x, as, bs);
  TORCH_CHECK(res.scalar_type() == at::kBFloat16 && res.is_contiguous() && out.is_contiguous(), "sp24: res / out");
  if (x.size(0) == 0) return;
  const at::cuda::OptionalCUDAGuard guard(x.device());
  dispatch<EPI_RES>(cfg, wf, mf, x, as, bs, &res, out, nullptr, out.size(1), fma_res);
}

// h [M, N / 2] bf16 = silu(gate) * up, amax [M, NT] = per-(row, N tile) max |h|
void sp24_gemm_swiglu(torch::Tensor wf, torch::Tensor mf, torch::Tensor x, torch::Tensor as, torch::Tensor bs,
                      torch::Tensor h, torch::Tensor amax, int64_t cfg) {
  check(wf, mf, x, as, bs);
  TORCH_CHECK(h.scalar_type() == at::kBFloat16 && h.is_contiguous() && amax.is_contiguous(), "sp24: h / amax");
  if (x.size(0) == 0) return;
  const at::cuda::OptionalCUDAGuard guard(x.device());
  dispatch<EPI_SWIGLU>(cfg, wf, mf, x, as, bs, nullptr, h, &amax, 2 * h.size(1), 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sp24_gemm_res", &sp24_gemm_res);
  m.def("sp24_gemm_swiglu", &sp24_gemm_swiglu);
}
