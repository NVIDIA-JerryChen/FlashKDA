// Host-side declarations of the conv / linear launchers (plain C++; definitions in cfbi_convlin.cu).
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace cfbi {

// ---- short causal depthwise conv ---------------------------------------------------------------------------
struct ConvLaunch {
    const void* x;            // bf16, token (n, t) at x + (bos + t) * x_stride (+ channel)
    void* y;                  // bf16, same indexing with y_stride
    float* y_f32;             // optional fp32 value before the bf16 rounding (tests), or null
    const void* weight;       // [D][W] fp32 or bf16
    const void* bias;         // [D] fp32 or bf16, or null
    void* state;              // bf16 [N][D][W-1] (updated in place) or null
    const int64_t* cu_seqlens;  // [N+1] or null
    int64_t x_stride, y_stride;
    int64_t T;                // tokens per sequence when cu_seqlens == null
    int D, W, N;
    int tiles;                // upper bound of ceil(len / tile) over the sequences
    int silu, w_bf16, bias_bf16;
    int vec;                  // 1 or 2 channels per thread (2 needs even D, 4-byte aligned rows)
};
constexpr int CONV_TILE_TOKENS = 64;
void launch_conv(const ConvLaunch& a, cudaStream_t stream);

// ---- batch-invariant bf16 linear ---------------------------------------------------------------------------
struct LinearLaunch {
    const void* x;            // bf16 [M][K]
    const void* w;            // bf16 [N][K]
    const void* bias;         // [N] fp32 or bf16, or null
    void* out;                // bf16 [M][N]
    float* out_f32;           // optional fp32 [M][N] (contiguous) value before the bf16 rounding (tests), or null
    int M, N, K;
    int64_t lda, ldb, ldc;
    int bias_bf16, vec_out;
};
constexpr int LINEAR_NUM_CFG = 5;
// cfg < 0: choose by shape; otherwise force one of the tile configurations (tests: all must give the same bits)
void launch_linear(const LinearLaunch& a, int cfg, cudaStream_t stream);
int linear_pick_cfg(int M, int N, int num_sms);

// D[tile] (16x8) = chain of m16n8k16 bf16 MMAs over K (ascending, or descending if reverse_k), one warp per tile.
void launch_probe_mma_bf16(const void* A, const void* B, float* D, int ntiles, int K, int reverse_k, cudaStream_t stream);

// y[i] = approx(x[i]); which: 0 = ex2.approx.ftz, 1 = rcp.approx.ftz (the two instructions of the SiLU).
void launch_probe_unary(const float* x, float* y, int64_t n, int which, cudaStream_t stream);

}  // namespace cfbi
