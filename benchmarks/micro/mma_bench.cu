// Micro-benchmarks that ground the CF-BI kernel design decisions (run on the target GPU):
//   1. mma.sync m16n8k8 TF32: dependent-chain latency and aggregate throughput per SM
//   2. the dot3 k-step (3 MMAs: SMALL chain of 2 + BIG) with the TF32 hi/lo split in ALU: achievable k-steps/clk
//   3. cp.async.bulk (global -> shared, mbarrier completion): per-SM bandwidth for L2-resident data
//   4. ex2.approx.ftz throughput
// build: nvcc -O3 -arch=sm_103a -o mma_bench mma_bench.cu && ./mma_bench
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)

__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// 1a. latency: one warp, one dependent chain
__global__ void lat_kernel(float* out, long long* cyc, int iters) {
    uint32_t a[4] = {0x3f800000u + threadIdx.x, 0x3f800000u, 0x3f800000u, 0x3f800000u}, b[2] = {0x3f800000u, 0x3f800000u};
    float d[4] = {0, 0, 0, 0};
    long long t0 = clock64();
#pragma unroll 1
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < 16; ++j) mma(d, a, b);
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) cyc[0] = t1 - t0;
    out[threadIdx.x] = d[0] + d[1] + d[2] + d[3];
}

// 1b. throughput: W warps per block, NACC independent accumulators per warp
template <int NACC>
__global__ void thr_kernel(float* out, long long* cyc, int iters) {
    uint32_t a[4] = {0x3f800000u + threadIdx.x, 0x3f800000u, 0x3f800000u, 0x3f800000u}, b[2] = {0x3f800000u, 0x3f800000u};
    float d[NACC][4];
#pragma unroll
    for (int k = 0; k < NACC; ++k) { d[k][0] = d[k][1] = d[k][2] = d[k][3] = 0.f; }
    __syncthreads();
    long long t0 = clock64();
#pragma unroll 1
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int k = 0; k < NACC; ++k) mma(d[k], a, b);
    }
    __syncthreads();
    long long t1 = clock64();
    if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
    float s = 0;
#pragma unroll
    for (int k = 0; k < NACC; ++k) s += d[k][0] + d[k][1] + d[k][2] + d[k][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// 2. dot3 k-steps incl. the hi/lo split: NT independent tiles per warp share the A fragment (like K2's two V tiles)
__device__ __forceinline__ void split(float x, uint32_t& hi, uint32_t& lo) {
    hi = __float_as_uint(x) & 0xFFFFE000u;
    lo = __float_as_uint(__fsub_rn(x, __uint_as_float(hi))) & 0xFFFFE000u;
}
template <int NT>
__global__ void dot3_kernel(const float* __restrict__ src, float* out, long long* cyc, int ksteps, int reps) {
    __shared__ float4 sm[2048];
    for (int i = threadIdx.x; i < 2048; i += blockDim.x) sm[i] = make_float4(src[i & 255] + i, 1.f, 2.f, 3.f);
    __syncthreads();
    float acc = 0.f;
    long long t0 = clock64();
#pragma unroll 1
    for (int r = 0; r < reps; ++r) {
        float big[NT][4], small[NT][4];
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int i = 0; i < 4; ++i) { big[t][i] = -0.f; small[t][i] = -0.f; }
#pragma unroll 4
        for (int ks = 0; ks < ksteps; ++ks) {
            const float4 av = sm[(ks * 32 + (threadIdx.x & 31)) & 2047];
            uint32_t ah[4], al[4];
            split(av.x, ah[0], al[0]); split(av.y, ah[1], al[1]); split(av.z, ah[2], al[2]); split(av.w, ah[3], al[3]);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const float4 bv = sm[(1024 + (ks * NT + t) * 32 + (threadIdx.x & 31)) & 2047];   // hi/lo already split in smem: pretend
                uint32_t bh[2] = {__float_as_uint(bv.x), __float_as_uint(bv.y)}, bl[2] = {__float_as_uint(bv.z), __float_as_uint(bv.w)};
                mma(small[t], al, bh);
                mma(small[t], ah, bl);
                mma(big[t], ah, bh);
            }
        }
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc += __fadd_rn(big[t][i], small[t][i]);
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

// 3. cp.async.bulk bandwidth: each CTA repeatedly copies `bytes` from global (small footprint => L2 hits) to smem
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__global__ void bulk_kernel(const char* __restrict__ src, long long* cyc, int bytes, int reps, size_t stride, int* sink) {
    extern __shared__ __align__(128) char smem[];
    __shared__ __align__(8) uint64_t bar;
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&bar)));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    long long t0 = clock64();
    uint32_t phase = 0;
    for (int r = 0; r < reps; ++r) {
        if (threadIdx.x == 0) {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(&bar)), "r"(bytes) : "memory");
            const char* s = src + (size_t)((blockIdx.x + r) & 3) * stride;
            // split into 32 KiB pieces (copy size limit is large, but keep it simple)
            for (int off = 0; off < bytes; off += 16384)
                asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                             ::"r"(smem_u32(smem + off)), "l"(s + off), "r"(16384), "r"(smem_u32(&bar)) : "memory");
        }
        // wait
        asm volatile("{\n .reg .pred p;\n WAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT;\n}\n" ::"r"(smem_u32(&bar)), "r"(phase) : "memory");
        phase ^= 1;
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { cyc[blockIdx.x] = t1 - t0; sink[blockIdx.x] = *(int*)smem; }
}

// 4. ex2 throughput
__global__ void ex2_kernel(float* out, long long* cyc, int iters) {
    float x[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) x[i] = -0.001f * (threadIdx.x + i);
    long long t0 = clock64();
#pragma unroll 1
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(x[j]));
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
    float s = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) s += x[i];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

int main() {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    int clockRate = 0;
    CK(cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, 0));
    printf("%s  SMs=%d  cc=%d.%d  smem/block optin=%zu KiB  L2=%.1f MB  clock=%d kHz\n", p.name, p.multiProcessorCount, p.major, p.minor,
           p.sharedMemPerBlockOptin / 1024, p.l2CacheSize / 1e6, clockRate);
    const int nsm = p.multiProcessorCount;
    float* out; long long* cyc; int* sink;
    CK(cudaMalloc(&out, 64 << 20)); CK(cudaMalloc(&cyc, 1 << 20)); CK(cudaMalloc(&sink, 1 << 20));
    long long h[4096];

    // 1a latency
    {
        const int iters = 2000;
        lat_kernel<<<1, 32>>>(out, cyc, iters);
        lat_kernel<<<1, 32>>>(out, cyc, iters);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h, cyc, 8, cudaMemcpyDeviceToHost));
        printf("[mma latency] dependent m16n8k8 tf32: %.1f cycles per MMA\n", (double)h[0] / (iters * 16));
    }
    // 1b throughput
    auto run_thr = [&](int warps, auto kern, int nacc) {
        const int iters = 400;
        kern<<<nsm, warps * 32>>>(out, cyc, iters);
        CK(cudaDeviceSynchronize());
        kern<<<nsm, warps * 32>>>(out, cyc, iters);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h, cyc, nsm * 8, cudaMemcpyDeviceToHost));
        double mx = 0;
        for (int i = 0; i < nsm; ++i) mx = h[i] > mx ? h[i] : mx;
        const double mmas = (double)warps * nacc * iters * 8;
        printf("[mma throughput] 1 CTA/SM, %2d warps x %d acc: %.3f MMA/clk/SM  (%.1f cycles per MMA per SM)\n", warps, nacc, mmas / mx, mx / mmas);
    };
    for (int w : {1, 2, 4, 8, 16}) run_thr(w, thr_kernel<1>, 1);
    for (int w : {1, 2, 4, 8}) run_thr(w, thr_kernel<2>, 2);
    for (int w : {1, 2, 4, 8}) run_thr(w, thr_kernel<4>, 4);
    for (int w : {1, 2, 4, 8}) run_thr(w, thr_kernel<8>, 8);

    // 2 dot3 k-steps
    float* src; CK(cudaMalloc(&src, 1 << 20)); CK(cudaMemset(src, 0, 1 << 20));
    auto run_dot3 = [&](int warps, int nt, auto kern) {
        const int ksteps = 16, reps = 100;
        kern<<<nsm, warps * 32>>>(src, out, cyc, ksteps, reps);
        CK(cudaDeviceSynchronize());
        kern<<<nsm, warps * 32>>>(src, out, cyc, ksteps, reps);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h, cyc, nsm * 8, cudaMemcpyDeviceToHost));
        double mx = 0;
        for (int i = 0; i < nsm; ++i) mx = h[i] > mx ? h[i] : mx;
        const double kst = (double)warps * nt * ksteps * reps;   // tile k-steps (3 MMAs each) per SM
        printf("[dot3 kstep] 1 CTA/SM, %d warps x %d tiles: %.1f cycles per tile-kstep per SM  (%.2f MMA/clk/SM); chain per kstep per warp %.1f cycles\n",
               warps, nt, mx / kst, 3 * kst / mx, mx / (ksteps * reps));
    };
    for (int w : {1, 4, 8, 16}) run_dot3(w, 1, dot3_kernel<1>);
    for (int w : {1, 4, 8, 16}) run_dot3(w, 2, dot3_kernel<2>);
    for (int w : {1, 4, 8}) run_dot3(w, 4, dot3_kernel<4>);

    // 3 bulk copy
    {
        char* buf; CK(cudaMalloc(&buf, 256 << 20)); CK(cudaMemset(buf, 1, 256 << 20));
        for (int bytes : {16384, 65536, 98304, 131072}) {
            const int reps = 200;
            const size_t smem = bytes;
            CK(cudaFuncSetAttribute(bulk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
            for (int ctas_per_sm : {1}) {
                bulk_kernel<<<nsm * ctas_per_sm, 32, smem>>>(buf, cyc, bytes, reps, (size_t)bytes, sink);
                CK(cudaDeviceSynchronize());
                bulk_kernel<<<nsm * ctas_per_sm, 32, smem>>>(buf, cyc, bytes, reps, (size_t)bytes, sink);
                CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(h, cyc, nsm * 8, cudaMemcpyDeviceToHost));
                double mx = 0, mn = 1e30;
                for (int i = 0; i < nsm; ++i) { mx = h[i] > mx ? h[i] : mx; mn = h[i] < mn ? h[i] : mn; }
                printf("[bulk copy] %6d B per copy, all %d SMs, L2-resident (4 x %d B footprint): %.1f B/clk/SM (min %.1f)  -> %.2f us per copy\n", bytes, nsm,
                       bytes, (double)bytes * reps / mx, (double)bytes * reps / mn, mx / reps / (clockRate / 1000.0));
            }
        }
        // single CTA (no contention)
        const int bytes = 65536, reps = 200;
        bulk_kernel<<<1, 32, bytes>>>(buf, cyc, bytes, reps, (size_t)bytes, sink);
        CK(cudaDeviceSynchronize());
        bulk_kernel<<<1, 32, bytes>>>(buf, cyc, bytes, reps, (size_t)bytes, sink);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h, cyc, 8, cudaMemcpyDeviceToHost));
        printf("[bulk copy] single CTA, 65536 B: %.1f B/clk, %.0f cycles per copy\n", (double)bytes * reps / h[0], (double)h[0] / reps);
    }

    // 4 ex2
    {
        const int iters = 2000;
        for (int w : {8, 16}) {
            ex2_kernel<<<nsm, w * 32>>>(out, cyc, iters);
            CK(cudaDeviceSynchronize());
            ex2_kernel<<<nsm, w * 32>>>(out, cyc, iters);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(h, cyc, nsm * 8, cudaMemcpyDeviceToHost));
            double mx = 0;
            for (int i = 0; i < nsm; ++i) mx = h[i] > mx ? h[i] : mx;
            printf("[ex2] %d warps/SM: %.2f ex2 per clk per SM (thread-ops)\n", w, (double)w * 32 * iters * 8 / mx);
        }
    }
    return 0;
}
