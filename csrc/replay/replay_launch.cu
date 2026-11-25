#include "replay/replay.h"
#include "replay/replay_kernel.cuh"

namespace flash_kda_replay {

namespace {
constexpr int kThreads = 256;

__global__ void _replay_copy_rows(cutlass::bfloat16_t* __restrict__ dst, cutlass::bfloat16_t const* __restrict__ src,
                                  int64_t const* __restrict__ dst_start, int64_t const* __restrict__ src_start,
                                  int const* __restrict__ count, int64_t row_elems) {
    const int n = blockIdx.x;
    const int i = blockIdx.y;
    if (i >= count[n]) return;
    cutlass::bfloat16_t* d = dst + (dst_start[n] + i) * row_elems;
    cutlass::bfloat16_t const* s = src + (src_start[n] + i) * row_elems;
    for (int64_t e = threadIdx.x; e < row_elems; e += blockDim.x) d[e] = s[e];
}
}  // namespace

int replay_smem_bytes() {
    return int(sizeof(ReplaySharedStorage<kHeadDim, kChunk>));
}

void launch_replay(ReplayParams const& p, int num_entries, cudaStream_t stream) {
    if (num_entries <= 0 || p.H <= 0) return;
    auto kernel = _flash_kda_replay<kHeadDim, kChunk, kThreads>;
    const int smem = replay_smem_bytes();
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kernel<<<dim3(num_entries, p.H), kThreads, smem, stream>>>(p);
}

void launch_copy_rows(cutlass::bfloat16_t* dst, cutlass::bfloat16_t const* src, int64_t const* dst_start,
                      int64_t const* src_start, int const* count, int num_entries, int max_rows, int64_t row_elems,
                      cudaStream_t stream) {
    if (num_entries <= 0 || max_rows <= 0 || row_elems <= 0) return;
    _replay_copy_rows<<<dim3(num_entries, max_rows), 256, 0, stream>>>(dst, src, dst_start, src_start, count, row_elems);
}

}  // namespace flash_kda_replay
