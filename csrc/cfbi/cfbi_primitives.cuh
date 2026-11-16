// CF-BI KDA: pinned floating-point primitives.
//
// Everything that decides bits lives here, so that prefill (K1/K2), decode (row engine) and the
// training forward are the *same* arithmetic by construction:
//   * elementwise ops only through explicit IEEE intrinsics (__fmul_rn / __fadd_rn / __fmaf_rn);
//     there is no `a*b+c` in numerics-critical code (and the extension is built with -fmad=false);
//   * three approximations, each through one PTX instruction: ex2.approx.ftz, rcp.approx.ftz,
//     rsqrt.approx.ftz (tanh.approx is only kept for the probe: on B300 its max relative error is 1.1e-5);
//   * ONE dot3<> primitive (three-term TF32 MMA, fixed K order) for every tensor-core product.
//
// See docs/20261102-cfbi-numerical-contract.md for the contract these implement.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace cfbi {

constexpr int HEAD_K = 128;   // key dim (K)
constexpr int HEAD_V = 128;   // value dim (V)
constexpr int BT = 64;        // chunk length, anchored at the sequence start
constexpr int SBR = 16;       // rows per sub-block
constexpr int NSB = BT / SBR; // sub-blocks per chunk
constexpr int NTHREADS = 256; // all CF-BI kernels use 8 warps

// Row-cache planes (fp32 [64][128] each), see CfbiState in flash_kda/cfbi/state.py.
constexpr int PLANE_KHAT = 0;
constexpr int PLANE_G2 = 1;
constexpr int PLANE_UT = 2;
constexpr int PLANE_W = 3;
constexpr int PLANE_U = 4;
constexpr int NPLANES = 5;

// log2(e) rounded to fp32 (torch: `A_log * LOG2E` is an fp32 multiply with the fp32-rounded scalar).
constexpr float LOG2E_F = 1.44269502162933349609375f;

// ------------------------------------------------------------------------------------------------
// Elementwise primitives
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ float fmul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float fadd(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float fsub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float ffma(float a, float b, float c) { return __fmaf_rn(a, b, c); }

__device__ __forceinline__ float ex2_ftz(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float tanh_approx(float x) {
    float y;
    asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float rsqrt_ftz(float x) {
    float y;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float rcp_ftz(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 f32_to_bf16_rn(float x) { return __float2bfloat16_rn(x); }

__device__ __forceinline__ float neg_zero() { return __int_as_float(0x80000000); }

// ------------------------------------------------------------------------------------------------
// Tensor-core primitive: mma.sync m16n8k8 TF32 (fp32 accumulate)
//
// Lane = 4*g + t (g = lane>>2 in 0..7, t = lane&3).
//   A (16x8, row):  a0=(g,   t)  a1=(g+8, t)  a2=(g,   t+4)  a3=(g+8, t+4)      [row, k]
//   B (8x8, col):   b0=(k=t, n=g)             b1=(k=t+4, n=g)
//   C/D (16x8):     c0=(g, 2t)  c1=(g, 2t+1)  c2=(g+8, 2t)  c3=(g+8, 2t+1)       [row, n]
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ void mma_tf32_m16n8k8(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// x_hi = x with the low 13 mantissa bits cleared (truncation to TF32); x_lo = TF32 truncation of the
// (exact) remainder x - x_hi. Both are exactly representable as TF32, so the hardware conversion is
// the identity on them.
__device__ __forceinline__ void split_tf32(float x, uint32_t& hi, uint32_t& lo) {
    hi = __float_as_uint(x) & 0xFFFFE000u;
    lo = __float_as_uint(__fsub_rn(x, __uint_as_float(hi))) & 0xFFFFE000u;
}

// Three-term TF32 product of one 16x8 output tile over `ksteps` ascending K-steps of 8:
//     BIG = SMALL = -0
//     per K-step:  SMALL = mma(A_lo, B_hi, SMALL); SMALL = mma(A_hi, B_lo, SMALL); BIG = mma(A_hi, B_hi, BIG)
//     out = fadd_rn(BIG, SMALL)
// load_a(ks, float (&a)[4]) / load_b(ks, float (&b)[2]) return the fp32 fragment values of K-step `ks`
// for this lane (layout above). Every call site uses the same tile geometry: consumer row t of a
// sub-block sits in A-row slot t%16, an N index n in B-column slot n%8, invalid slots are zero.
template <class LoadA, class LoadB>
__device__ __forceinline__ void dot3(int ksteps, LoadA&& load_a, LoadB&& load_b, float (&out)[4]) {
    float big[4], small[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) { big[i] = neg_zero(); small[i] = neg_zero(); }
#pragma unroll 1
    for (int ks = 0; ks < ksteps; ++ks) {
        float a[4], b[2];
        load_a(ks, a);
        load_b(ks, b);
        uint32_t ah[4], al[4], bh[2], bl[2];
#pragma unroll
        for (int i = 0; i < 4; ++i) split_tf32(a[i], ah[i], al[i]);
#pragma unroll
        for (int i = 0; i < 2; ++i) split_tf32(b[i], bh[i], bl[i]);
        mma_tf32_m16n8k8(small, al, bh);
        mma_tf32_m16n8k8(small, ah, bl);
        mma_tf32_m16n8k8(big, ah, bh);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) out[i] = __fadd_rn(big[i], small[i]);
}

// Fragment coordinates of this lane.
struct LaneIdx {
    int lane, g, t;
    __device__ __forceinline__ LaneIdx() {
        lane = threadIdx.x & 31;
        g = lane >> 2;
        t = lane & 3;
    }
};

// C-fragment element i (0..3) -> (row slot, column slot) inside the 16x8 tile.
__device__ __forceinline__ int c_row(const LaneIdx& L, int i) { return L.g + 8 * (i >> 1); }
__device__ __forceinline__ int c_col(const LaneIdx& L, int i) { return 2 * L.t + (i & 1); }

}  // namespace cfbi
