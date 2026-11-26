"""mma.sync m16n8k16 bf16 throughput of one SM as a function of the number of warps and of independent accumulators."""
import torch
from torch.utils.cpp_extension import load_inline

src = r"""
#include <cstdint>
#include <cuda_runtime.h>
template <int NACC>
__global__ void k(float* out, long long* cyc, int iters) {
    uint32_t a0 = threadIdx.x * 3u + 1u, a1 = a0 + 7u, a2 = a0 + 11u, a3 = a0 + 13u, b0 = a0 ^ 0x3f803f80u, b1 = a0 ^ 0x3f003f00u;
    float acc[NACC][4];
    #pragma unroll
    for (int i = 0; i < NACC; ++i) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }
    long long t0 = clock64();
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int i = 0; i < NACC; ++i)
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[i][0]), "+f"(acc[i][1]), "+f"(acc[i][2]), "+f"(acc[i][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    long long t1 = clock64();
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < NACC; ++i) s += acc[i][0] + acc[i][1] + acc[i][2] + acc[i][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
    if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}
void run(torch::Tensor out, torch::Tensor cyc, int blocks, int threads, int nacc, int iters) {
    auto o = out.data_ptr<float>(); auto c = reinterpret_cast<long long*>(cyc.data_ptr<int64_t>());
    switch (nacc) {
        case 1: k<1><<<blocks, threads>>>(o, c, iters); break;
        case 2: k<2><<<blocks, threads>>>(o, c, iters); break;
        case 4: k<4><<<blocks, threads>>>(o, c, iters); break;
        case 8: k<8><<<blocks, threads>>>(o, c, iters); break;
        case 16: k<16><<<blocks, threads>>>(o, c, iters); break;
    }
}
"""
mod = load_inline("mmarate", cpp_sources="#include <torch/extension.h>\nvoid run(torch::Tensor, torch::Tensor, int, int, int, int);",
                  cuda_sources=src, functions=["run"], extra_cuda_cflags=["-O3", "-gencode", "arch=compute_103a,code=sm_103a"], verbose=False)
dev = "cuda"
for warps in (1, 2, 4, 8, 16):
    for nacc in (1, 4, 8, 16):
        out = torch.zeros(1024 * 1024, device=dev); cyc = torch.zeros(1024, dtype=torch.int64, device=dev)
        iters = 2000
        mod.run(out, cyc, 1, 32 * warps, nacc, iters); torch.cuda.synchronize()
        mod.run(out, cyc, 1, 32 * warps, nacc, iters); torch.cuda.synchronize()
        c = int(cyc[0]); n_mma = warps * nacc * iters
        print(f"[mma-rate] warps={warps:2d} acc/warp={nacc:2d}: {c / n_mma * 1.0:6.2f} SM-cycles per HMMA  ({n_mma * 2048 / c:7.1f} MAC/clk/SM)", flush=True)
