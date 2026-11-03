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

// K1 -> K2 workspace, offsets in floats inside one (chunk, head) block
constexpr int WS_WG = 0;                       // [128][128]: rows 0..63 = W, rows 64..127 = GQ
constexpr int WS_UT = WS_WG + 128 * 128;       // [64][128]
constexpr int WS_AQK = WS_UT + 64 * 128;       // [64][64]   (row t, col j)
constexpr int WS_KD = WS_AQK + 64 * 64;        // [128][64]  (k, j)
constexpr int WS_GEND = WS_KD + 128 * 64;      // [128]
constexpr int WS_FLOATS = WS_GEND + 128;
static_assert(WS_FLOATS == WS_FLOATS_PER_CHUNK_HEAD, "workspace layout out of sync with cfbi_launch.h");
static_assert((WS_FLOATS * 4) % 16 == 0, "workspace blocks must stay 16-byte aligned");

// Per-sequence schedule, computed on the device by the plan kernel (no host sync).
struct SeqPlan {
    int tok0;   // first token of the sequence on the flattened token axis
    int len;    // tokens in this call
    int pos0;   // open-chunk fill at call start (snapshot; the pos array is updated at the very end)
    int hrows;  // head rows: handled by the row engine (finish the open chunk, or all rows in mode 1)
    int nfull;  // full 64-row chunks handled by K1/K2
    int tail;   // trailing rows (< 64) handled by K1/K2; they stay in the open chunk afterwards
    int cbase;  // index of this sequence's first K1/K2 chunk (exclusive prefix sum); plan[N].cbase = total
    int pad_;
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
