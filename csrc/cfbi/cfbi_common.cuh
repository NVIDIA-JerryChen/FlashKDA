// CF-BI KDA: device-side parameter block, layouts and small helpers shared by the kernels.
#pragma once

#include "cfbi/cfbi_launch.h"
#include "cfbi/cfbi_prep.cuh"
#include "cfbi/cfbi_primitives.cuh"

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
#ifndef CFBI_K2_BV
#define CFBI_K2_BV 16
#endif
constexpr int K2_BV = CFBI_K2_BV;                  // value columns per K2 CTA
constexpr int K2_NT = K2_BV / 8;           // 8-column MMA tiles per CTA
constexpr int K2_NSLICE = HEAD_V / K2_BV;  // CTAs per (sequence, head)
constexpr int WS2_W = 0;                       // W  A-fragments   [rt 4][ks 16][lane 32][4]
constexpr int WS2_GQ = WS2_W + 8192;           // GQ A-fragments   [rt 4][ks 16][lane 32][4]
constexpr int WS2_UT = WS2_GQ + 8192;          // UT, C-fragment order of the P1 tiles: [slice][rt 4][nt NT][lane 32][4]
constexpr int WS2_AQF = WS2_UT + 8192;         // Aqk cross-block A-fragments of sub-blocks 1..3: [12][lane 32][4]
constexpr int WS2_AQD = WS2_AQF + 12 * 128;    // Aqk diagonal 16x16 blocks, row stride 17: [4][16][17]
constexpr int WS2_KD = WS2_AQD + 4 * 16 * 17;  // KD A-fragments   [mt 8][ks 8][lane 32][4]
constexpr int WS2_GEND = WS2_KD + 8192;        // GEND [128]
constexpr int WS2_FLOATS = WS2_GEND + 128;
static_assert(WS2_FLOATS <= WS_FLOATS_PER_CHUNK_HEAD, "workspace v2 must fit the allocation");
static_assert((WS2_FLOATS * 4) % 16 == 0 && (WS2_AQD * 4) % 16 == 0 && (WS2_KD * 4) % 16 == 0, "16-byte alignment");
constexpr int AQ_NFRAG = 12;                   // fragments of sub-blocks 1..3: s*(s-1)+ks, ks < 2s

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
    SeqPlan* plan;
    float* ws;
};

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
