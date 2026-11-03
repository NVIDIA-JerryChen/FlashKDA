// CF-BI KDA: device-side call planning (no host sync, CUDA-graph safe).
//
// For every sequence of the call:
//   head rows  h = 0 if pos == 0 else min(len, 64 - pos)    -> row engine (finishes the open chunk)
//   then       f = (len - h) / 64 full chunks, tail = (len - h) % 64 rows -> K1/K2
//   new pos = (pos + len) % 64
// The chunk grid is anchored at the sequence start, so the head rows are exactly the rows that bring the
// open chunk to its boundary; everything after them starts on a grid line.
// mode 1 (decode) sends every row to the row engine.
#pragma once

#include <cub/block/block_scan.cuh>

#include "cfbi/cfbi_common.cuh"

namespace cfbi {

constexpr int PLAN_THREADS = 1024;

__global__ void __launch_bounds__(PLAN_THREADS)
plan_kernel(const int64_t* cu, int fixed_len, const int* pos_state, int N, int all_rows, SeqPlan* plan) {
    using Scan = cub::BlockScan<int, PLAN_THREADS>;
    __shared__ typename Scan::TempStorage tmp;
    __shared__ int carry;
    if (threadIdx.x == 0) carry = 0;
    __syncthreads();
    for (int base = 0; base < N; base += PLAN_THREADS) {
        const int n = base + threadIdx.x;
        SeqPlan sp = {0, 0, 0, 0, 0, 0, 0, 0};
        int nch = 0;
        if (n < N) {
            int tok0, len;
            if (cu != nullptr) {
                tok0 = (int)cu[n];
                len = (int)(cu[n + 1] - cu[n]);
            } else {
                tok0 = n * fixed_len;
                len = fixed_len;
            }
            const int pos0 = pos_state != nullptr ? pos_state[n] : 0;
            int hrows, nfull, tail;
            if (all_rows) {
                hrows = len; nfull = 0; tail = 0;
            } else {
                hrows = pos0 == 0 ? 0 : min(len, BT - pos0);
                const int rest = len - hrows;
                nfull = rest / BT;
                tail = rest % BT;
            }
            nch = nfull + (tail > 0 ? 1 : 0);
            sp.tok0 = tok0; sp.len = len; sp.pos0 = pos0; sp.hrows = hrows; sp.nfull = nfull; sp.tail = tail;
        }
        int excl, total;
        Scan(tmp).ExclusiveSum(nch, excl, total);
        if (n < N) {
            sp.cbase = carry + excl;
            plan[n] = sp;
        }
        __syncthreads();
        if (threadIdx.x == 0) carry += total;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        SeqPlan last = {0, 0, 0, 0, 0, 0, carry, 0};
        plan[N] = last;
    }
}

// pos is per sequence but every head CTA reads it, so it is only advanced after all kernels of the call.
__global__ void finalize_pos_kernel(const SeqPlan* plan, int* pos, int N) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) {
        const SeqPlan sp = plan[n];
        pos[n] = (int)(((int64_t)sp.pos0 + sp.len) % BT);
    }
}

}  // namespace cfbi
