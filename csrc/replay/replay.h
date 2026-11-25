#pragma once
// Host-visible interface of the open-chunk replay kernel (see replay_kernel.cuh for what it computes).
#include <cuda_runtime.h>
#include <cstdint>

#include <cutlass/bfloat16.h>

namespace flash_kda_replay {

constexpr int kChunk = 16;   // flash_kda.fwd's chunk size; the replay contract is tied to it
constexpr int kHeadDim = 128;

struct ReplayParams {
    // token inputs, packed [T_total, H, D] (beta [T_total, H]), exactly what flash_kda.fwd reads
    cutlass::bfloat16_t const* q;
    cutlass::bfloat16_t const* k;
    cutlass::bfloat16_t const* v;
    cutlass::bfloat16_t const* g;
    cutlass::bfloat16_t const* beta;
    // outputs: row (out_row[n] + i) of an [R, H, D] tensor receives the output of token i of entry n
    cutlass::bfloat16_t* out;
    // per entry n of the launch (grid.x): which tokens to append to which state slot
    int64_t const* tok_start;   // [N] first token row in the packed inputs (nullptr: n * tokens_per_entry)
    int const* tok_count;       // [N] number of tokens, <= 0: nothing to do (nullptr: tokens_per_entry)
    int tokens_per_entry;       // used when tok_start / tok_count are nullptr (decode: inputs [N, T, H, D])
    int64_t const* out_row;     // [N] first output row (nullptr: same as tok_start)
    int const* slot_ids;        // [N] state slot of entry n (nullptr: slot n; negative: skip the entry; >= P traps)
    int num_slots;              // P
    // persistent state pools, indexed by slot
    cutlass::bfloat16_t* state;       // [P, H, D, D] bf16, S at the start of the open chunk ([V, K] like fwd states)
    cutlass::bfloat16_t* rows_k;      // [P, H, CHUNK, D] raw k of the open chunk's rows
    cutlass::bfloat16_t* rows_v;      // [P, H, CHUNK, D] raw v
    cutlass::bfloat16_t* rows_g;      // [P, H, CHUNK, D] raw g (before the gate activation)
    cutlass::bfloat16_t* rows_beta;   // [P, H, CHUNK]    raw beta logits
    int* pos;                         // [P, H] rows in the open chunk (0 .. CHUNK-1), read and advanced per (slot, head)
    // per-head parameters, as flash_kda.fwd takes them
    float const* A_log;    // [H]
    float const* dt_bias;  // [H, D]
    float scale;
    float gate_scale;      // float(lower_bound * log2(e)), computed exactly as flash_kda.cpp does
    int H;
};

// Appends tok_count[n] tokens to state slot slot_ids[n] for every entry n < num_entries (grid = num_entries x H).
void launch_replay(ReplayParams const& p, int num_entries, cudaStream_t stream);

// dst[dst_start[n] + i, :] = src[src_start[n] + i, :] for i < count[n] (rows of row_elems bf16 elements).
void launch_copy_rows(cutlass::bfloat16_t* dst, cutlass::bfloat16_t const* src, int64_t const* dst_start,
                      int64_t const* src_start, int const* count, int num_entries, int max_rows, int64_t row_elems,
                      cudaStream_t stream);

int replay_smem_bytes();

}  // namespace flash_kda_replay
