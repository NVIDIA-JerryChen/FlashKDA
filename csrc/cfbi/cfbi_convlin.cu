// Launchers of the CF-BI conv / linear kernels (no host synchronisation anywhere: CUDA-graph safe).
#include <climits>
#include <cstdio>
#include <stdexcept>

#include "cfbi/cfbi_conv.cuh"
#include "cfbi/cfbi_convlin.h"
#include "cfbi/cfbi_linear.cuh"

namespace cfbi {

// ---- conv --------------------------------------------------------------------------------------------------
namespace {

template <int W>
void launch_conv_w(const conv::Args& a, int vec, cudaStream_t stream) {
    const int ch = conv::THREADS * vec;
    const int64_t dtiles = (a.D + ch - 1) / ch;
    const int64_t blocks = dtiles * a.tiles * a.N;
    if (blocks <= 0) return;
    if (blocks > INT_MAX) throw std::runtime_error("cfbi conv: too many blocks, pass a smaller max_seqlen / split the call");
    if (vec == 2) conv::conv_kernel<W, 2><<<(unsigned)blocks, conv::THREADS, 0, stream>>>(a);
    else conv::conv_kernel<W, 1><<<(unsigned)blocks, conv::THREADS, 0, stream>>>(a);
}

}  // namespace

void launch_conv(const ConvLaunch& l, cudaStream_t stream) {
    static_assert(CONV_TILE_TOKENS == conv::TILE_T, "tile size out of sync");
    conv::Args a{};
    a.x = static_cast<const __nv_bfloat16*>(l.x);
    a.y = static_cast<__nv_bfloat16*>(l.y);
    a.y_f32 = l.y_f32;
    a.weight = l.weight;
    a.bias = l.bias;
    a.state = static_cast<__nv_bfloat16*>(l.state);
    a.cu_seqlens = l.cu_seqlens;
    a.x_stride = l.x_stride;
    a.y_stride = l.y_stride;
    a.T = l.T;
    a.D = l.D;
    a.N = l.N;
    a.tiles = l.tiles;
    a.silu = l.silu;
    a.w_bf16 = l.w_bf16;
    a.bias_bf16 = l.bias_bf16;
    switch (l.W) {
        case 2: launch_conv_w<2>(a, l.vec, stream); break;
        case 3: launch_conv_w<3>(a, l.vec, stream); break;
        case 4: launch_conv_w<4>(a, l.vec, stream); break;
        case 5: launch_conv_w<5>(a, l.vec, stream); break;
        case 6: launch_conv_w<6>(a, l.vec, stream); break;
        case 7: launch_conv_w<7>(a, l.vec, stream); break;
        case 8: launch_conv_w<8>(a, l.vec, stream); break;
        default: throw std::runtime_error("cfbi conv: kernel width must be in 2..8");
    }
}

// ---- linear ------------------------------------------------------------------------------------------------
namespace {

template <class C>
void run_linear(const lin::Args& a, cudaStream_t stream) {
    static bool attr_set[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev >= 0 && dev < 64 && !attr_set[dev]) {
        cudaFuncSetAttribute(lin::linear_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
        attr_set[dev] = true;
    }
    const dim3 grid((a.N + C::BN - 1) / C::BN, (a.M + C::BM - 1) / C::BM);
    lin::linear_kernel<C><<<grid, C::THREADS, C::SMEM, stream>>>(a);
}

}  // namespace

// Chosen from benchmarks/micro/linear_configs.py (B300, K = N = 4096, weights streaming from HBM): the 16-row tile is the best up
// to M = 64 (the weight tile is re-read from L2 by every 16-row CTA, but the CTAs fill the machine); the choice only affects
// speed - every configuration computes the same bits (tests/cfbi/test_linear.py::test_all_tile_configurations_agree).
int linear_pick_cfg(int M, int N, int num_sms) {
    if (M <= 64) return 0;
    if (M <= 128) return 1;
    auto tiles = [&](int bm, int bn) { return (int64_t)((M + bm - 1) / bm) * ((N + bn - 1) / bn); };
    if (tiles(128, 128) >= num_sms) return 4;
    if (tiles(128, 64) >= num_sms * 3 / 4) return 3;
    return 2;
}

void launch_linear(const LinearLaunch& l, int cfg, cudaStream_t stream) {
    if (l.M <= 0 || l.N <= 0) return;
    lin::Args a{};
    a.x = static_cast<const __nv_bfloat16*>(l.x);
    a.w = static_cast<const __nv_bfloat16*>(l.w);
    a.bias = l.bias;
    a.out = static_cast<__nv_bfloat16*>(l.out);
    a.out_f32 = l.out_f32;
    a.M = l.M;
    a.N = l.N;
    a.K = l.K;
    a.lda = l.lda;
    a.ldb = l.ldb;
    a.ldc = l.ldc;
    a.bias_bf16 = l.bias_bf16;
    a.vec_out = l.vec_out;
    if (cfg < 0) {
        static int sms[64] = {};
        int dev = 0;
        cudaGetDevice(&dev);
        int n = (dev >= 0 && dev < 64) ? sms[dev] : 0;
        if (n == 0) {
            cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
            if (dev >= 0 && dev < 64) sms[dev] = n;
        }
        cfg = linear_pick_cfg(l.M, l.N, n > 0 ? n : 148);
    }
    switch (cfg) {
        case 0: run_linear<lin::CfgS>(a, stream); break;
        case 1: run_linear<lin::CfgM1>(a, stream); break;
        case 2: run_linear<lin::CfgM2>(a, stream); break;
        case 3: run_linear<lin::CfgL1>(a, stream); break;
        case 4: run_linear<lin::CfgL2>(a, stream); break;
        default: throw std::runtime_error("cfbi linear: cfg must be -1 (auto) or 0..4");
    }
}

// ---- probes ------------------------------------------------------------------------------------------------
void launch_probe_mma_bf16(const void* A, const void* B, float* D, int ntiles, int K, int reverse_k, cudaStream_t stream) {
    lin::probe_mma_bf16_kernel<<<ntiles, 32, 0, stream>>>(static_cast<const __nv_bfloat16*>(A),
                                                           static_cast<const __nv_bfloat16*>(B), D, K, reverse_k);
}

namespace {
__global__ void probe_unary_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n, int which) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = which == 0 ? ex2_ftz(x[i]) : rcp_ftz(x[i]);
}
}  // namespace

void launch_probe_unary(const float* x, float* y, int64_t n, int which, cudaStream_t stream) {
    probe_unary_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(x, y, n, which);
}

}  // namespace cfbi
