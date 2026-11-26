// Host-visible interface of the chunk-64 kernels (prefill fwd64 and the open-chunk replay).
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace c64 {

constexpr int kChunk64 = 64;
constexpr int kHeadDim64 = 128;

// ---- prefill: K1 (token-parallel operands) + K2 (serial recurrence) over a table of segments
struct FwdParams {
    // token inputs, packed [T_total, H, 128] (beta [T_total, H]), bf16 bits
    const uint16_t* q;
    const uint16_t* k;
    const uint16_t* v;
    const uint16_t* g;
    const uint16_t* beta;
    uint16_t* out;
    // segment table: segment n owns tokens seg_tok0[n] .. +seg_len[n]-1 of the packed inputs and state slot seg_slot[n]
    int N;
    int H;
    int total_tiles;            // >= sum of ceil(seg_len / 64): K1 grid.x, also the workspace stride
    const int64_t* seg_tok0;    // [N]
    const int* seg_len;         // [N] (<= 0: nothing to do)
    const int* seg_slot;        // [N]
    int* tile_prefix;           // [N+1] scratch (written by the launcher)
    const float* A_log;         // [H]
    const float* dt_bias;       // [H, 128]
    float scale;
    float gate_scale;           // float(lower_bound * log2(e))
    // workspace (K1 -> K2)
    uint16_t* ws_kd;
    uint16_t* ws_qd;
    uint16_t* ws_kr;
    float* ws_gt;
    uint16_t* ws_inv;
    uint16_t* ws_mq;
    // state pools [P, H, 128, 128] bf16 ([V, K] like flash_kda.fwd); nullptr: zero state in / no state out
    const uint16_t* st_in;
    uint16_t* st_out;
    long long* prof;            // optional (profiling): phase timestamps of K1 CTA (tile 7, head 0) / K2 CTA (0, 0, 0)
};

// bytes of workspace for `total_tiles` tile slots, H heads and N segments (K1 -> K2 operands + tile prefix)
size_t fwd64_workspace_bytes(int total_tiles, int H, int N);
// carve the workspace pointers out of one buffer (same layout the size function counts)
void fwd64_bind_workspace(FwdParams& p, void* ws);
void launch_fwd64(FwdParams const& p, int k2_cfg, cudaStream_t stream);

// ---- open-chunk replay (decode, chunk heads / tails of prefill calls, final_state)
struct ReplayParams {
    const uint16_t* q;
    const uint16_t* k;
    const uint16_t* v;
    const uint16_t* g;
    const uint16_t* beta;
    uint16_t* out;              // row (out_row[n] + i) of [R, H, 128] receives the output of token i of entry n
    const int64_t* tok_start;   // [N] first token row (nullptr: n * tokens_per_entry)
    const int* tok_count;       // [N] tokens to append (<= 0: nothing to do; nullptr: tokens_per_entry)
    int tokens_per_entry;
    const int64_t* out_row;     // [N] (nullptr: same as tok_start)
    const int* slot_ids;        // [N] state slot of entry n (nullptr: slot n; negative: skip; >= num_slots traps)
    int num_slots;
    // persistent state pools, indexed by slot
    uint16_t* state;            // [P, H, 128, 128] bf16: S at the start of the open chunk
    uint16_t* rows_k;           // [P, H, 64, 128]
    uint16_t* rows_v;           // [P, H, 64, 128]
    uint16_t* rows_g;           // [P, H, 64, 128]
    uint16_t* rows_beta;        // [P, H, 64]
    int* pos;                   // [P, H] rows in the open chunk (0..63)
    const float* A_log;
    const float* dt_bias;
    float scale;
    float gate_scale;
    int H;
    // peek mode: close the open rows as a short chunk into final_out [num_entries, H, 128, 128] instead of consuming
    // tokens (state / rows / pos stay untouched); entry n reads slot_ids[n]
    uint16_t* final_out;
    long long* prof;            // optional (profiling): phase timestamps of CTA (0, 0)
};

void launch_replay64(ReplayParams const& p, int num_entries, cudaStream_t stream);
int replay64_smem_bytes();
int fwd64_k1_smem_bytes();
int fwd64_k2_smem_bytes(int cfg);

}  // namespace c64
