// Chunk-64 KDA (bf16 program): shared constants, pinned floating-point primitives, tensor-core / shared-memory helpers.
//
// Everything that decides bits lives here and in c64_k1.cuh / c64_k2.cuh, so that the prefill kernels (K1 / K2) and the
// open-chunk replay kernel run the SAME device functions:
//   * elementwise fp32 math only through explicit IEEE intrinsics (__fmul_rn / __fadd_rn / __fmaf_rn): no `a*b+c`
//     anywhere in numerics-critical code (the extension is also built with -fmad=false);
//   * three approximations, each one PTX instruction: ex2.approx.ftz, tanh.approx, rsqrt.approx.ftz;
//   * every tensor-core product is mma.sync m16n8k16 bf16 with fp32 accumulate, K ascending in steps of 16.
// See docs/20261125-replay-c64.md for the numerical contract.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace c64 {

constexpr int D = 128;          // head dim (K = V = 128)
constexpr int CHUNK = 64;       // rows per chunk, anchored at the sequence start
constexpr int SB = 16;          // rows per sub-block
constexpr int NSB = CHUNK / SB; // sub-blocks per chunk

// shared-memory row strides (elements)
constexpr int RS = D + 8;       // bf16 [rows][128]: 272 B rows -> conflict-free ldmatrix
constexpr int MS = CHUNK + 8;   // bf16 [rows][64]: 144 B rows
constexpr int GS = D + 8;       // fp32 [rows][128]: 544 B rows -> conflict-free float2 fragment reads
constexpr int LS = CHUNK + 4;   // fp32 [rows][64]

// ------------------------------------------------------------------------------------------------------------------
// scalar primitives
// ------------------------------------------------------------------------------------------------------------------
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
// sigmoid(x) = tanh(x/2) * 1/2 + 1/2 (flash_kda.fwd's gate / beta sigmoid)
__device__ __forceinline__ float sigmoid_f(float x) {
    return fadd(fmul(tanh_approx(fmul(x, 0.5f)), 0.5f), 0.5f);
}

// bf16 <-> fp32 (bits as uint16_t)
__device__ __forceinline__ uint16_t f2bf(float x) {
    uint16_t r;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ float bf2f(uint16_t b) { return __uint_as_float(static_cast<uint32_t>(b) << 16); }
__device__ __forceinline__ float rbf(float x) { return bf2f(f2bf(x)); }   // round to bf16, keep as fp32

// two fp32 -> one bf16x2 register, `lo` in the low half (the lower column / element index)
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
__device__ __forceinline__ float bf_lo(uint32_t p) { return __uint_as_float(p << 16); }
__device__ __forceinline__ float bf_hi(uint32_t p) { return __uint_as_float(p & 0xFFFF0000u); }


// packed bf16x2 arithmetic, one rounding per element (add / sub / mul of two bf16 values are exact in fp32 up to the
// product's 16 bits, so mul.rn.bf16x2 equals bf16(fmul(a, b)); sub / add differ from bf16(fp32 op) only in the rare
// double-rounding cases)
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
__device__ __forceinline__ uint32_t hsub2(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("sub.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
__device__ __forceinline__ uint32_t hadd2(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

// ------------------------------------------------------------------------------------------------------------------
// tensor core: mma.sync m16n8k16 bf16 x bf16 -> fp32
//   lane = 4*g + t (g = lane >> 2, t = lane & 3)
//   A (16x16, row):   a0 = (g, 2t..2t+1)  a1 = (g+8, 2t..2t+1)  a2 = (g, 2t+8..2t+9)  a3 = (g+8, 2t+8..2t+9)   [row, k]
//   B (16x8, "col"):  b0 = (k = 2t..2t+1, n = g)  b1 = (k = 2t+8..2t+9, n = g)
//   C/D (16x8):       c0 = (g, 2t)  c1 = (g, 2t+1)  c2 = (g+8, 2t)  c3 = (g+8, 2t+1)                          [row, n]
// ------------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* smem) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(smem)));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* smem) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(smem)));
}
__device__ __forceinline__ void stsm_x4_t(void* smem, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                 :: "r"(smem_u32(smem)), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}
// transpose an 8x8 bf16 matrix held in fragment layout (thread (g,t) holds row g, columns 2t..2t+1)
__device__ __forceinline__ uint32_t movm_t(uint32_t x) {
    uint32_t y;
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(y) : "r"(x));
    return y;
}

// ldmatrix lane addressing (row r, column c of the 16x16 bf16 tile at `base`, row stride `ld` elements):
//   A fragment (non-trans): matrices (rows 0-7,c 0-7) (rows 8-15,c 0-7) (rows 0-7,c 8-15) (rows 8-15,c 8-15)
__device__ __forceinline__ const void* a_addr(const uint16_t* base, int ld, int lane) {
    return base + ((lane & 7) + ((lane >> 3) & 1) * 8) * ld + (lane >> 4) * 8;
}
//   B fragment from an [n][k] row-major tile (k contiguous), two n8 tiles: regs = (b0 u0, b1 u0, b0 u1, b1 u1)
__device__ __forceinline__ const void* b_addr(const uint16_t* base, int ld, int lane) {
    return base + ((lane & 7) + (lane >> 4) * 8) * ld + ((lane >> 3) & 1) * 8;
}
//   B fragment from a [k][n] row-major tile (n contiguous), .trans: regs = (b0 u0, b1 u0, b0 u1, b1 u1)
__device__ __forceinline__ const void* bt_addr(const uint16_t* base, int ld, int lane) {
    return base + ((lane & 7) + ((lane >> 3) & 1) * 8) * ld + (lane >> 4) * 8;
}
//   A fragment from a [k][m] row-major tile (m contiguous), .trans: matrices (k 0-7,m 0-7) (k 0-7,m 8-15) (k 8-15,m 0-7)
//   (k 8-15,m 8-15) = (a0, a1, a2, a3)
__device__ __forceinline__ const void* at_addr(const uint16_t* base, int ld, int lane) {
    return base + ((lane & 7) + (lane >> 4) * 8) * ld + ((lane >> 3) & 1) * 8;
}

// ------------------------------------------------------------------------------------------------------------------
// asynchronous copies
// ------------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_u32(smem)), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }

// ------------------------------------------------------------------------------------------------------------------
// small helpers
// ------------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint4 ld_smem_u4(const void* p) { return *reinterpret_cast<const uint4*>(p); }
__device__ __forceinline__ void st_smem_u4(void* p, uint4 v) { *reinterpret_cast<uint4*>(p) = v; }
__device__ __forceinline__ uint32_t ld_u32(const void* p) { return *reinterpret_cast<const uint32_t*>(p); }
__device__ __forceinline__ void st_u32(void* p, uint32_t v) { *reinterpret_cast<uint32_t*>(p) = v; }

__device__ __forceinline__ void bar_sync(int id, int nthreads) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(nthreads) : "memory");
}

}  // namespace c64
