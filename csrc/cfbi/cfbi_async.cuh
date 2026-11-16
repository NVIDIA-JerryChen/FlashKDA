// CF-BI KDA: thin PTX wrappers for mbarriers and bulk (non-tensor) async copies, global -> shared.
//
// Used by the pipelined kernels. The copies move bytes only; nothing here touches the numerics.
//   cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes  dst(smem), src(gmem), bytes, [mbar]
// dst/src must be 16-byte aligned and bytes a multiple of 16. A consumer waits on the mbarrier phase.
//
// Architecture guard: the bulk copy, the mbarrier transaction counts (expect_tx / try_wait) and the async-proxy
// fence exist on sm_90 and newer only. nvcc compiles every .cu once per target architecture, so for sm_80 / 86 / 89
// the wrappers below compile to a trap; the launcher never selects a kernel that calls them on such a device
// (resolve_impl in cfbi_launch.cu picks the reference kernels, which use none of this). The Ampere-style cp.async
// helpers at the end are available from sm_80 on and are not guarded.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace cfbi {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
#define CFBI_ASYNC_SM90 0
#else
#define CFBI_ASYNC_SM90 1   // sm_90+ device pass, or the host pass (device bodies are not generated there)
#endif

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t arrive_count) {
#if CFBI_ASYNC_SM90
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(arrive_count) : "memory");
#else
    __trap();
#endif
}

// Make the initialised barriers visible to the async proxy (call once, then __syncthreads()).
__device__ __forceinline__ void mbar_init_fence() {
#if CFBI_ASYNC_SM90
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#else
    __trap();
#endif
}

// One thread: arrive once and announce `bytes` of async-copy traffic for the current phase.
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
#if CFBI_ASYNC_SM90
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
#else
    __trap();
#endif
}

// Wait for the phase with the given parity to complete (parity of the first phase is 0).
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
#if CFBI_ASYNC_SM90
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
#else
    __trap();
#endif
}

// Order this thread's earlier generic-proxy shared accesses before later async-proxy writes (WAR across proxies).
__device__ __forceinline__ void fence_proxy_async_smem() {
#if CFBI_ASYNC_SM90
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#else
    __trap();
#endif
}

// global -> shared bulk copy, completion signalled (tx bytes) on `bar`.
__device__ __forceinline__ void bulk_g2s(void* smem_dst, const void* gmem_src, uint32_t bytes, uint64_t* bar) {
#if CFBI_ASYNC_SM90
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smem_u32(smem_dst)),
        "l"(gmem_src), "r"(bytes), "r"(smem_u32(bar))
        : "memory");
#else
    __trap();
#endif
}

// Ampere-style 16-byte async copy global -> shared (bypasses L1), tracked with commit groups (sm_80+).
__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(smem_dst)), "l"(gmem_src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }

}  // namespace cfbi
