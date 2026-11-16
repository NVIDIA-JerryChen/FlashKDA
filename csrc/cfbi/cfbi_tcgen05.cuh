// CF-BI KDA: thin PTX wrappers for the Blackwell tensor core path (tcgen05 / UMMA + tensor memory).
//
// Validated by benchmarks/micro/umma_probe.cu (B300, sm_103a): tcgen05.mma.cta_group::1.kind::tf32 with K = 8 per
// instruction, issued in the dot3 order (SMALL += lo*hi; SMALL += hi*lo; BIG += hi*hi, accumulate = 0 on the first
// K-step), is bit-identical to the mma.sync dot3 for every M in {64,128}, N in {8..128}, K <= 128.
//
// The tcgen05 instructions only exist in the arch-specific targets (sm_100a / sm_103a); in any other compile pass the
// kernels that use them are compiled out (CFBI_HAVE_TCGEN05 undefined).
#pragma once

#include <cstdint>
#include <cstdio>

#include "cfbi/cfbi_async.cuh"

#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM101_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL) || \
    defined(__CUDA_ARCH_FEAT_SM110_ALL)
#define CFBI_HAVE_TCGEN05 1
#endif

namespace cfbi {

__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }

// Warp-wide: allocates `ncols` (power of two, >= 32) tensor-memory columns; the base address is written to *slot (smem).
__device__ __forceinline__ void tmem_alloc(uint32_t* slot, int ncols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(slot)), "r"(ncols) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory");
}

// D(tmem) (+)= A(smem descriptor) x B(smem descriptor); accumulate == 0: D = A x B. ONE thread issues it.
__device__ __forceinline__ void umma_tf32(uint32_t d_tmem, uint64_t adesc, uint64_t bdesc, uint32_t idesc, uint32_t accumulate) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t}\n"
        :
        : "r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accumulate), "r"(0), "r"(0), "r"(0), "r"(0)
        : "memory");
}
// All earlier tcgen05 operations of the issuing thread complete => one arrival on the mbarrier.
__device__ __forceinline__ void umma_commit(uint64_t* bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(bar)) : "memory");
}

// UMMA shared-memory matrix descriptor, no swizzle, K-major: start>>4 [0,14), LBO>>4 [16,30), SBO>>4 [32,46), version 1.
// Canonical layout: a core matrix is 8 rows x 16 B (4 tf32 of one row) stored contiguously (128 B); LBO = byte stride
// between the two core matrices of one K = 8 slice, SBO = byte stride between 8-row groups.
__device__ __forceinline__ uint64_t make_smem_desc(uint32_t saddr, uint32_t lbo_bytes, uint32_t sbo_bytes) {
    uint64_t d = (uint64_t)((saddr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo_bytes >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    return d;
}
// Instruction descriptor, kind::tf32: fp32 accumulate, tf32 A/B, both K-major, N>>3 @17, M>>4 @24.
__host__ __device__ constexpr uint32_t make_idesc(int M, int N) {
    return (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}

// tcgen05.ld.32x32b.x16: warp w reads tensor-memory lanes 32*(w%4)..+31; thread = lane, 16 consecutive 32-bit columns.
__device__ __forceinline__ void tmem_ld16(uint32_t taddr, uint32_t (&r)[16]) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]),
          "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
        : "r"(taddr));
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

// mbarrier helpers beyond cfbi_async.cuh --------------------------------------------------------------------
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
// Wait with a bound: a protocol bug becomes a trap with a message instead of a hung GPU.
__device__ __forceinline__ void mbar_wait_bounded(uint64_t* bar, uint32_t parity, int tag) {
    uint32_t spins = 0;
    while (true) {
        uint32_t ok;
        asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}\n"
                     : "=r"(ok) : "r"(smem_u32(bar)), "r"(parity) : "memory");
        if (ok) return;
        if (++spins > (1u << 22)) {
            printf("[cfbi umma] mbarrier timeout: tag %d block (%d,%d,%d) thread %d parity %u\n", tag, blockIdx.x, blockIdx.y, blockIdx.z,
                   threadIdx.x, parity);
            __trap();
        }
    }
}

}  // namespace cfbi
