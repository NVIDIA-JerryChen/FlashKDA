// CF-BI KDA: device-side parameter block, layouts and small helpers shared by the kernels.
#pragma once

#include "cfbi/cfbi_launch.h"
#include "cfbi/cfbi_prep.cuh"
#include "cfbi/cfbi_primitives.cuh"

#include <type_traits>

namespace cfbi {

// ---- layouts ---------------------------------------------------------------------------------
constexpr int ROW_PLANE_FLOATS = BT * HEAD_K;                       // one [64][128] fp32 plane
constexpr int ROWS_PER_HEAD_FLOATS = NPLANES * ROW_PLANE_FLOATS;    // row cache per (seq, head): 160 KiB
constexpr int E_FLOATS = HEAD_V * HEAD_K;                           // state per (seq, head): 64 KiB

// K1 -> K2 workspace of the reference kernels (csrc/cfbi/ref), offsets in floats inside one (chunk, head) block
constexpr int WS_WG = 0;                       // [128][128]: rows 0..63 = W, rows 64..127 = GQ
constexpr int WS_UT = WS_WG + 128 * 128;       // [64][128]
constexpr int WS_AQK = WS_UT + 64 * 128;       // [64][64]   (row t, col j)
constexpr int WS_KD = WS_AQK + 64 * 64;        // [128][64]  (k, j)
constexpr int WS_GEND = WS_KD + 128 * 64;      // [128]
constexpr int WS_FLOATS = WS_GEND + 128;
static_assert(WS_FLOATS == WS_FLOATS_PER_CHUNK_HEAD, "workspace layout out of sync with cfbi_launch.h");
static_assert((WS_FLOATS * 4) % 16 == 0, "workspace blocks must stay 16-byte aligned");

// K1 -> K2 workspace of the optimised kernels: the same values, stored "fragment native" so that K2 can bulk-copy a
// whole operand and every lane reads its mma.sync fragment as one 16-byte (A) / 8-byte (B) shared load.
//   A fragment of a 16x8 tile at (row tile rt, k-step ks), lane = 4*g + t:  float4 {X[16rt+g][8ks+t], X[16rt+g+8][8ks+t],
//   X[16rt+g][8ks+t+4], X[16rt+g+8][8ks+t+4]}  (registers a0..a3 of mma.m16n8k8).
// K2 splits the 128 value columns of a head over CTAs of 8, 16 or 32 columns (chosen per call and per (sequence, head), see K2Cfg in
// cfbi_k2.cuh); the UT part of the workspace is stored in slices of 16 columns whatever the K2 tile is: a K2 CTA reads the slice(s) that
// hold its columns, so K1 does not depend on the K2 tiling.
constexpr int WS2_W = 0;                       // W  A-fragments   [rt 4][ks 16][lane 32][4]
constexpr int WS2_GQ = WS2_W + 8192;           // GQ A-fragments   [rt 4][ks 16][lane 32][4]
constexpr int WS2_UT = WS2_GQ + 8192;          // UT, C-fragment order of the P1 tiles: [16-column slice 8][rt 4][nt 2][lane 32][4]
constexpr int WS2_AQF = WS2_UT + 8192;         // Aqk cross-block A-fragments of sub-blocks 1..3: [12][lane 32][4]
constexpr int WS2_AQD = WS2_AQF + 12 * 128;    // Aqk diagonal 16x16 blocks, row stride 17: [4][16][17]
constexpr int WS2_KD = WS2_AQD + 4 * 16 * 17;  // KD A-fragments   [mt 8][ks 8][lane 32][4]
constexpr int WS2_GEND = WS2_KD + 8192;        // GEND [128]
constexpr int WS2_FLOATS = WS2_GEND + 128;
static_assert(WS2_FLOATS <= WS_FLOATS_PER_CHUNK_HEAD, "workspace v2 must fit the allocation");
static_assert((WS2_FLOATS * 4) % 16 == 0 && (WS2_AQD * 4) % 16 == 0 && (WS2_KD * 4) % 16 == 0, "16-byte alignment");
constexpr int AQ_NFRAG = 12;                   // fragments of sub-blocks 1..3: s*(s-1)+ks, ks < 2s

// The pos word of a state slot (one int32 per slot) holds the fill of the open chunk, 0 .. BT - 1, in its low bits; outside of a call it is exactly that number.
// While the single-launch decode kernel (rows_kernel<true>, cfbi_rows.cuh) runs, the bits from POS_COUNT_SHIFT up count the head CTAs of the slot's sequence that have
// arrived (read the word); the last one to arrive stores the plain new value.
constexpr int POS_COUNT_SHIFT = 8;
constexpr int POS_MASK = BT - 1;
static_assert(POS_MASK < (1 << POS_COUNT_SHIFT), "the fill of the open chunk must fit below the arrival counter");

// Token window and state slot of sequence n of a call: the integers the plan kernel and the single-launch decode kernel both start from (one function, so that the two
// read the same values). slot < 0 (a padding entry of a fixed-shape batch) has no tokens and no state access.
struct SeqWindow {
    int tok0;   // first token of the sequence on the flattened token axis
    int len;    // tokens in this call (0 for a padding entry)
    int slot;   // state slot (index into E / rows / pos), -1 = skipped
};
__device__ __forceinline__ SeqWindow seq_window(const int64_t* cu, int fixed_len, const int* slot_ids, int n) {
    SeqWindow w;
    if (cu != nullptr) {
        w.tok0 = (int)cu[n];
        w.len = (int)(cu[n + 1] - cu[n]);
    } else {
        w.tok0 = n * fixed_len;
        w.len = fixed_len;
    }
    w.slot = slot_ids != nullptr ? slot_ids[n] : n;
    if (w.slot < 0) w.len = 0;   // padding entry of a fixed-shape batch: no tokens, no state access
    return w;
}

// Per-sequence schedule, computed on the device by the plan kernel (no host sync).
struct SeqPlan {
    int tok0;   // first token of the sequence on the flattened token axis
    int len;    // tokens in this call
    int pos0;   // open-chunk fill at call start (snapshot; the pos array is updated at the very end)
    int hrows;  // head rows: handled by the row engine (finish the open chunk, or all rows in mode 1)
    int nfull;  // full 64-row chunks handled by K1/K2
    int tail;   // trailing rows (< 64) handled by K1/K2; they stay in the open chunk afterwards
    int cbase;  // index of this sequence's first K1/K2 chunk (exclusive prefix sum); plan[N].cbase = total
    int slot;   // state slot of the sequence (index into E / rows / pos), -1 = skipped (padding entry)
};
static_assert(sizeof(SeqPlan) == PLAN_INTS_PER_SEQ * sizeof(int), "SeqPlan size");

struct Params {
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const __nv_bfloat16* g;
    const __nv_bfloat16* beta;
    __nv_bfloat16* out;
    float* out_f32;
    const float* A_log;
    const float* dt_bias;
    float scale;
    float scale2;
    int H;
    int N;
    float* E;     // NOT const/__restrict__: written and re-read inside kernels (no .nc loads)
    float* rows;
    int* pos;
    const int* slot_ids;   // [N] state slot of each sequence of the call (nullptr: slot = sequence index)
    int fuse_pos;          // 1: the row engine writes the new pos itself (mode 1), no finalize kernel
    SeqPlan* plan;
    float* ws;
    // K2 value-tile widths (chosen by the launcher from N, H and the SM count): the (sequence, head) pairs p = h * N + n with p < tile_x32
    // use 32-column tiles (4 CTAs each), the other pairs tile_bvb-column tiles (8, 16 or 32; 128 / tile_bvb CTAs each). Only K2 reads them:
    // K1 stores the UT part of the workspace in 16-column slices whatever the tiles are.
    int tile_x32;
    int tile_bvb;
};
// Params is the first argument of every kernel, and a kernel with a further argument behind it (k2_dyn_kernel's schedule) finds that argument at the offset sizeof(Params) of the
// constant bank: a field added here moves it, which is another machine code for a kernel that the change does not own. The layout is therefore fixed; what only one launch path
// needs goes into a type of its own (DecodeParams).
static_assert(sizeof(Params) == 152, "Params is the argument block of every kernel: a change of its size changes the machine code of the kernels that take a further argument behind it");

// The arguments of the single-launch decode kernel (rows_kernel<true>, cfbi_rows.cuh), which has no plan kernel in front of it: the call's Params and the token windows of the
// call (the plan kernel's own arguments): sequence n owns the tokens [cu_seqlens[n], cu_seqlens[n + 1]) or, without cu_seqlens, [n * fixed_len, (n + 1) * fixed_len).
struct DecodeParams : Params {
    const int64_t* cu_seqlens;
    int fixed_len;
};
static_assert(sizeof(DecodeParams) == sizeof(Params) + 16, "the decode arguments (a pointer and an int, padded) follow the common block: no offset of Params moves");
static_assert(std::is_trivially_copyable<DecodeParams>::value, "a kernel argument is copied byte by byte");

// 8 consecutive bf16 (16-byte aligned) -> 8 fp32 (exact).
__device__ __forceinline__ void load8_bf16(const __nv_bfloat16* p, float (&x)[8]) {
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        x[2 * i] = __uint_as_float(w[i] << 16);
        x[2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000u);
    }
}

}  // namespace cfbi
