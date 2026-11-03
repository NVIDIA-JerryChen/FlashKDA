// CF-BI KDA: thin PTX wrappers for mbarriers and bulk (non-tensor) async copies, global -> shared.
//
// Used by the pipelined kernels. The copies move bytes only; nothing here touches the numerics.
//   cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes  dst(smem), src(gmem), bytes, [mbar]
// dst/src must be 16-byte aligned and bytes a multiple of 16. A consumer waits on the mbarrier phase.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace cfbi {

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t arrive_count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(arrive_count) : "memory");
}

// Make the initialised barriers visible to the async proxy (call once, then __syncthreads()).
__device__ __forceinline__ void mbar_init_fence() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }

// One thread: arrive once and announce `bytes` of async-copy traffic for the current phase.
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}

// Wait for the phase with the given parity to complete (parity of the first phase is 0).
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1 bra DONE;\n"
        "bra LAB_WAIT;\n"
        "DONE:\n"
        "}\n" ::"r"(smem_u32(bar)),
        "r"(parity)
        : "memory");
}

// Order this thread's earlier generic-proxy shared accesses before later async-proxy writes (WAR across proxies).
__device__ __forceinline__ void fence_proxy_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

// global -> shared bulk copy, completion signalled (tx bytes) on `bar`.
__device__ __forceinline__ void bulk_g2s(void* smem_dst, const void* gmem_src, uint32_t bytes, uint64_t* bar) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smem_u32(smem_dst)),
        "l"(gmem_src), "r"(bytes), "r"(smem_u32(bar))
        : "memory");
}

}  // namespace cfbi
