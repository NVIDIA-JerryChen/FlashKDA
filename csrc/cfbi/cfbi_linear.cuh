// CF-BI linear: a batch-invariant bf16 GEMM, out[m][n] = bf16_rn(acc[m][n] (+ bias[n])).
//
// Program of one output element (the contract, see docs/20261103-cfbi-conv-linear.md):
//     acc = +0.0f
//     for s = 0 .. K/16 - 1 (ascending):  acc = mma.sync.m16n8k16.bf16(x[m][16s .. 16s+15], w[n][16s .. 16s+15], acc)
//     out = bf16_rn(bias ? fadd_rn(acc, float(bias[n])) : acc)
// One accumulator chain, one MMA instruction shape, K ascending, no split-K, no second accumulator.
// The tiling (block / warp tile, pipeline depth, which kernel runs) depends on M and N only to pick speed: it only
// decides WHICH lane/register holds an output element, never what is added in which order. That the tensor-core result of
// an element does not depend on its slot in the tile or on the other rows/columns of the tile is the hardware premise,
// probed by tests/cfbi/test_linear_probe.py with probe_mma_bf16_kernel below (same instruction, same K loop).
//
// K must be a multiple of 16; M, N arbitrary (rows / columns beyond M / N are zero-filled, never stored).
#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cfbi/cfbi_primitives.cuh"

namespace cfbi {
namespace lin {

// ---- PTX wrappers (own namespace: no dependency on other kernels' helpers) -----------------------------------
__device__ __forceinline__ uint32_t smem_addr(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
    const int bytes = valid ? 16 : 0;   // src-size 0: nothing is read, the 16 bytes are zero-filled
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}
__device__ __forceinline__ void ldsm_x2(uint32_t (&r)[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

// ---- the tensor-core primitive: m16n8k16, bf16 x bf16 + fp32 -> fp32 ----------------------------------------------
// Lane = 4*g + t (g = lane>>2, t = lane&3).
//   A (16x16, row):  a0 = [g][2t,2t+1]  a1 = [g+8][2t,2t+1]  a2 = [g][2t+8,2t+9]  a3 = [g+8][2t+8,2t+9]   (2 x bf16 per register)
//   B (16x8, col):   b0 = [k=2t,2t+1][n=g]  b1 = [k=2t+8,2t+9][n=g]
//   C/D (16x8):      c0 = [g][2t]  c1 = [g][2t+1]  c2 = [g+8][2t]  c3 = [g+8][2t+1]
__device__ __forceinline__ void mma_bf16_m16n8k16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---- probe: one 16x8 output tile per warp, A [16][K] row-major, B [K][8] row-major (B[k][n]), fragments loaded with
// plain scalar loads (no smem, no ldmatrix): the same instruction and the same ascending K loop as the GEMM.
__global__ void probe_mma_bf16_kernel(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
                                      float* __restrict__ D, int K, int reverse_k) {
    const int tile = blockIdx.x;
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const uint16_t* a = reinterpret_cast<const uint16_t*>(A) + (int64_t)tile * 16 * K;
    const uint16_t* b = reinterpret_cast<const uint16_t*>(B) + (int64_t)tile * K * 8;
    const int nks = K / 16;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    auto pack = [](uint16_t lo, uint16_t hi) { return (uint32_t)lo | ((uint32_t)hi << 16); };
    for (int s = 0; s < nks; ++s) {
        const int k0 = 16 * (reverse_k ? (nks - 1 - s) : s);
        uint32_t af[4], bf[2];
        af[0] = pack(a[g * K + k0 + 2 * t], a[g * K + k0 + 2 * t + 1]);
        af[1] = pack(a[(g + 8) * K + k0 + 2 * t], a[(g + 8) * K + k0 + 2 * t + 1]);
        af[2] = pack(a[g * K + k0 + 2 * t + 8], a[g * K + k0 + 2 * t + 9]);
        af[3] = pack(a[(g + 8) * K + k0 + 2 * t + 8], a[(g + 8) * K + k0 + 2 * t + 9]);
        bf[0] = pack(b[(k0 + 2 * t) * 8 + g], b[(k0 + 2 * t + 1) * 8 + g]);
        bf[1] = pack(b[(k0 + 2 * t + 8) * 8 + g], b[(k0 + 2 * t + 9) * 8 + g]);
        mma_bf16_m16n8k16(acc, af, bf);
    }
    float* d = D + (int64_t)tile * 16 * 8;
    d[g * 8 + 2 * t] = acc[0];
    d[g * 8 + 2 * t + 1] = acc[1];
    d[(g + 8) * 8 + 2 * t] = acc[2];
    d[(g + 8) * 8 + 2 * t + 1] = acc[3];
}

// ---- the GEMM ------------------------------------------------------------------------------------------------
struct Args {
    const __nv_bfloat16* x;   // [M][K], row stride lda (elements, multiple of 8)
    const __nv_bfloat16* w;   // [N][K], row stride ldb
    const void* bias;         // [N] fp32 or bf16, or null
    __nv_bfloat16* out;       // [M][N], row stride ldc
    float* out_f32;           // optional [M][N] contiguous: the fp32 value before the bf16 rounding (tests), or null
    int M, N, K;
    int64_t lda, ldb, ldc;
    int bias_bf16;
    int vec_out;              // out is 4-byte aligned and ldc even: column pairs are stored as one 32-bit word
};

// block tile BM x BN, k-tile BK (multiple of 64: 128-byte smem rows, 16-byte chunks XOR-swizzled with row & 7),
// WARPS_M x WARPS_N warps, cp.async pipeline of STAGES stages. REVERSE: test-only negative control (K tiles and
// k16 steps walked in descending order), requires K % BK == 0.
template <int BM_, int BN_, int BK_, int WARPS_M_, int WARPS_N_, int STAGES_, bool REVERSE_ = false>
struct Cfg {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, WARPS_M = WARPS_M_, WARPS_N = WARPS_N_, STAGES = STAGES_;
    static constexpr bool REVERSE = REVERSE_;
    static constexpr int THREADS = 32 * WARPS_M * WARPS_N;
    static constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
    static constexpr int MT = WM / 16, NT = WN / 8;
    static constexpr int CPR = BK / 8;   // 16-byte chunks per smem row
    static constexpr int ROW_BYTES = BK * 2;
    static constexpr int SMEM_A = BM * ROW_BYTES, SMEM_B = BN * ROW_BYTES;
    static constexpr int STAGE_BYTES = SMEM_A + SMEM_B;
    static constexpr int SMEM = STAGES * STAGE_BYTES;
    static constexpr int A_ITERS = (BM * CPR + THREADS - 1) / THREADS, B_ITERS = (BN * CPR + THREADS - 1) / THREADS;
    static_assert(BK % 64 == 0, "128-byte rows needed for the swizzle");
    static_assert(WM % 16 == 0 && WN % 8 == 0 && BM % WARPS_M == 0 && BN % WARPS_N == 0, "warp tile");
    static_assert(NT == 1 || NT % 2 == 0, "B fragments are loaded 1 or 2 n8-tiles at a time");
    static_assert(STAGES >= 2, "pipeline");
};

template <class C>
__global__ void __launch_bounds__(C::THREADS) linear_kernel(const Args a) {
    extern __shared__ __align__(1024) uint8_t smem[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int warp_m = warp / C::WARPS_N, warp_n = warp % C::WARPS_N;
    const int bm = blockIdx.y * C::BM, bn = blockIdx.x * C::BN;
    const int KT = (a.K + C::BK - 1) / C::BK;
    const uint32_t smem0 = smem_addr(smem);

    auto load_stage = [&](int stage, int kt) {
        const int k0 = kt * C::BK;
        const uint32_t sA = smem0 + stage * C::STAGE_BYTES;
        const uint32_t sB = sA + C::SMEM_A;
#pragma unroll
        for (int it2 = 0; it2 < C::A_ITERS; ++it2) {
            const int i = tid + it2 * C::THREADS;
            if (C::BM * C::CPR % C::THREADS == 0 || i < C::BM * C::CPR) {
                const int row = i / C::CPR, ch = i % C::CPR;
                const bool valid = (bm + row < a.M) && (k0 + ch * 8 < a.K);
                const __nv_bfloat16* src = valid ? a.x + (int64_t)(bm + row) * a.lda + k0 + ch * 8 : a.x;
                cp_async16(sA + row * C::ROW_BYTES + ((ch ^ (row & 7)) << 4), src, valid);
            }
        }
#pragma unroll
        for (int it2 = 0; it2 < C::B_ITERS; ++it2) {
            const int i = tid + it2 * C::THREADS;
            if (C::BN * C::CPR % C::THREADS == 0 || i < C::BN * C::CPR) {
                const int row = i / C::CPR, ch = i % C::CPR;
                const bool valid = (bn + row < a.N) && (k0 + ch * 8 < a.K);
                const __nv_bfloat16* src = valid ? a.w + (int64_t)(bn + row) * a.ldb + k0 + ch * 8 : a.w;
                cp_async16(sB + row * C::ROW_BYTES + ((ch ^ (row & 7)) << 4), src, valid);
            }
        }
    };

    float acc[C::MT][C::NT][4];
#pragma unroll
    for (int mi = 0; mi < C::MT; ++mi)
#pragma unroll
        for (int ni = 0; ni < C::NT; ++ni)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[mi][ni][i] = 0.0f;   // +0.0f

    // prologue: stages 0 .. STAGES-2
#pragma unroll
    for (int s = 0; s < C::STAGES - 1; ++s) {
        const int kt = C::REVERSE ? KT - 1 - s : s;
        if (s < KT) load_stage(s, kt);
        cp_commit();
    }

    for (int it = 0; it < KT; ++it) {
        cp_wait<C::STAGES - 2>();
        __syncthreads();   // stage it is landed for every thread; every warp is done with stage it-1 (refilled below)
        {
            const int nit = it + C::STAGES - 1;
            if (nit < KT) load_stage(nit % C::STAGES, C::REVERSE ? KT - 1 - nit : nit);
            cp_commit();
        }
        const int stage = it % C::STAGES;
        const int kt = C::REVERSE ? KT - 1 - it : it;
        const int k0 = kt * C::BK;
        const uint32_t sA = smem0 + stage * C::STAGE_BYTES;
        const uint32_t sB = sA + C::SMEM_A;
#pragma unroll
        for (int kki = 0; kki < C::BK / 16; ++kki) {
            const int kk = C::REVERSE ? C::BK / 16 - 1 - kki : kki;
            if (k0 + kk * 16 < a.K) {   // uniform: skip k16 steps beyond K (never executed as zero products)
                uint32_t af[C::MT][4], bfr[C::NT][2];
#pragma unroll
                for (int mi = 0; mi < C::MT; ++mi) {
                    const int row = warp_m * C::WM + mi * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    const int ch = kk * 2 + (lane >> 4);
                    ldsm_x4(af[mi], sA + row * C::ROW_BYTES + ((ch ^ (row & 7)) << 4));
                }
                if constexpr (C::NT == 1) {
                    const int row = warp_n * C::WN + (lane & 7);
                    const int ch = kk * 2 + ((lane >> 3) & 1);
                    ldsm_x2(bfr[0], sB + row * C::ROW_BYTES + ((ch ^ (row & 7)) << 4));
                } else {
#pragma unroll
                    for (int p = 0; p < C::NT / 2; ++p) {
                        const int row = warp_n * C::WN + (2 * p + ((lane >> 4) & 1)) * 8 + (lane & 7);
                        const int ch = kk * 2 + ((lane >> 3) & 1);
                        uint32_t r[4];
                        ldsm_x4(r, sB + row * C::ROW_BYTES + ((ch ^ (row & 7)) << 4));
                        bfr[2 * p][0] = r[0];
                        bfr[2 * p][1] = r[1];
                        bfr[2 * p + 1][0] = r[2];
                        bfr[2 * p + 1][1] = r[3];
                    }
                }
#pragma unroll
                for (int mi = 0; mi < C::MT; ++mi)
#pragma unroll
                    for (int ni = 0; ni < C::NT; ++ni) mma_bf16_m16n8k16(acc[mi][ni], af[mi], bfr[ni]);
            }
        }
    }

    // epilogue: (+ bias) -> bf16 round-to-nearest-even -> global
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int mi = 0; mi < C::MT; ++mi) {
#pragma unroll
        for (int ni = 0; ni < C::NT; ++ni) {
            const int col = bn + warp_n * C::WN + ni * 8 + 2 * t;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = bm + warp_m * C::WM + mi * 16 + g + 8 * h;
                if (row >= a.M) continue;
                float v0 = acc[mi][ni][2 * h], v1 = acc[mi][ni][2 * h + 1];
                if (a.bias != nullptr) {
                    if (col < a.N)
                        v0 = __fadd_rn(v0, a.bias_bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(a.bias)[col])
                                                       : static_cast<const float*>(a.bias)[col]);
                    if (col + 1 < a.N)
                        v1 = __fadd_rn(v1, a.bias_bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(a.bias)[col + 1])
                                                       : static_cast<const float*>(a.bias)[col + 1]);
                }
                if (a.out_f32 != nullptr) {
                    if (col < a.N) a.out_f32[(int64_t)row * a.N + col] = v0;
                    if (col + 1 < a.N) a.out_f32[(int64_t)row * a.N + col + 1] = v1;
                }
                __nv_bfloat16* o = a.out + (int64_t)row * a.ldc + col;
                if (col + 1 < a.N && a.vec_out) {
                    __nv_bfloat162 pr;
                    pr.x = __float2bfloat16_rn(v0);
                    pr.y = __float2bfloat16_rn(v1);
                    *reinterpret_cast<__nv_bfloat162*>(o) = pr;
                } else {
                    if (col < a.N) o[0] = __float2bfloat16_rn(v0);
                    if (col + 1 < a.N) o[1] = __float2bfloat16_rn(v1);
                }
            }
        }
    }
}

// ---- configurations: all compute the same program, they differ in tiling only --------------------------------
#ifdef CFBI_LIN_NEG_CONTROL
constexpr bool kNegControl = true;
#else
constexpr bool kNegControl = false;
#endif
using CfgS = Cfg<16, 32, 128, 1, 4, 4, kNegControl>;   // decode: M <= 16 (only this config walks K backwards in the control build)
using CfgM1 = Cfg<64, 32, 128, 4, 1, 4>;               // M <= 64
using CfgM2 = Cfg<128, 32, 64, 4, 2, 4>;               // M <= 128
using CfgL1 = Cfg<128, 64, 64, 4, 2, 4>;               // medium M
using CfgL2 = Cfg<128, 128, 64, 2, 4, 3>;              // large M
constexpr int NUM_CFG = 5;

}  // namespace lin
}  // namespace cfbi
