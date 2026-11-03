// CF-BI KDA: the row engine = decode = "one more row of the chunk program" (optimised).
//
// Same arithmetic as csrc/cfbi/ref/ref_rows.cuh (tests/cfbi/test_vs_ref.py compares bits). One CTA (8 warps) per
// (sequence, head) appends the sequence's head rows one by one to the open chunk:
//   A prep -> B Akk/Aqk row vs cached rows -> C UT/W row (+ the cross-block part of O) -> D P1/P2 with E -> E U -> F O
//   -> G chunk end.
// What changed against the reference (v1 measured: Akk/Aqk phase 14-21K cycles, P1/P2 19-41K of ~45K per row, because
// every k-step of every dot3 waited for its own global load):
//   * every global load that does not depend on the row's prep is issued at the top of the row (the cached rows of
//     the UT / W / U recursions, the previous G2 row, v) and the E / cache lines are prefetched into L2 up front;
//   * the K loops are fully unrolled with all operand loads hoisted in front of the MMAs;
//   * rows that share a B operand share one MMA tile: ks and qs (Akk / Aqk, B = kjr) sit in A-row slots s and s^8,
//     W and GQ (P1 / P2, B = E^T) likewise. The tile-position / neighbour independence of a tensor-core output
//     element (the contract's hardware premise, tests/cfbi/test_mma_probe.py) makes every result bit-identical to
//     the one-row-per-tile evaluation; it halves the MMAs of those two phases;
//   * the cross-block part of O (M2 = dot3(Aqk row, cached U rows)) does not need the new U, so it is computed with
//     the UT / W cross-block products; what is left after P1 is one subtraction and a short FMA chain over
//     preloaded cached rows;
//   * the cache append of khat / G2 and the reference-row copy no longer cost a barrier.
// The cache planes (khat, G2, UT, W, U) and E live in global memory (fp32); plain (non-const, non-restrict)
// pointers are used so the compiler never turns reads of freshly written rows into .nc loads.
#pragma once

#include "cfbi/cfbi_common.cuh"

#include <cstdio>

namespace cfbi {

#ifdef CFBI_ROWS_PROFILE
#define ROWS_TS(i) do { if (threadIdx.x == 0) tp[i] = clock64(); } while (0)
#else
#define ROWS_TS(i) do {} while (0)
#endif

struct RowSmem {
    float k[HEAD_K];      // khat of the current row
    float q[HEAD_K];      // qhat of the current row
    float g2[HEAD_K];     // G2 of the current row
    float gref[HEAD_K];   // G2 of the reference row r(pos)
    float gq[HEAD_K];     // GQ = Gamma * qhat
    float gend[HEAD_K];   // GEND (chunk end only)
    float x[2][HEAD_K];   // UT_t, W_t of the current row
    float m[3][HEAD_V];   // cross-sub-block products: UT, W, and U (for O)
    float p1[HEAD_V], p2[HEAD_V], u[HEAD_V];
    float akk[BT], aqk[BT];
    float beta;
};

__device__ __forceinline__ void prefetch_l2(const void* ptr) { asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr)); }

// Cross-sub-block products of the current row for KS = 2s k-steps: M[which][v] = dot3(A row, X[which][0:16s][v]) with
// which = 0: akk row x UT rows, 1: akk row x W rows, 2: aqk row x U rows. 48 column tiles, 6 per warp, all B fragments
// loaded up front. The A row sits in A-row slot `slot`.
template <int KS>
__device__ __forceinline__ void rows_m_tiles(RowSmem& S, const float* Rut, const float* Rw, const float* Ru, int warp,
                                             const LaneIdx& L, int slot) {
    float bv[6][KS][2];
#pragma unroll
    for (int q = 0; q < 6; ++q) {
        const float* X = (q >> 1) == 0 ? Rut : ((q >> 1) == 1 ? Rw : Ru);
        const int nt = warp + 8 * (q & 1);
#pragma unroll
        for (int ks = 0; ks < KS; ++ks)
#pragma unroll
            for (int ii = 0; ii < 2; ++ii) bv[q][ks][ii] = X[(8 * ks + L.t + 4 * ii) * HEAD_K + 8 * nt + L.g];
    }
    float big[6][4], sml[6][4];
#pragma unroll
    for (int q = 0; q < 6; ++q)
#pragma unroll
        for (int i = 0; i < 4; ++i) { big[q][i] = neg_zero(); sml[q][i] = neg_zero(); }
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
        uint32_t kh[4], kl[4], qh[4], ql[4];
#pragma unroll
        for (int ii = 0; ii < 4; ++ii) {
            const int row = L.g + 8 * (ii & 1), kk = 8 * ks + L.t + 4 * (ii >> 1);
            const float av = S.akk[kk], aq = S.aqk[kk];   // unconditional loads, select (no divergent branch)
            split_tf32(row == slot ? av : 0.f, kh[ii], kl[ii]);
            split_tf32(row == slot ? aq : 0.f, qh[ii], ql[ii]);
        }
#pragma unroll
        for (int q = 0; q < 6; ++q) {
            uint32_t bh[2], bl[2];
#pragma unroll
            for (int ii = 0; ii < 2; ++ii) split_tf32(bv[q][ks][ii], bh[ii], bl[ii]);
            const bool is_aqk = (q >> 1) == 2;
            mma_tf32_m16n8k8(sml[q], is_aqk ? ql : kl, bh);
            mma_tf32_m16n8k8(sml[q], is_aqk ? qh : kh, bl);
            mma_tf32_m16n8k8(big[q], is_aqk ? qh : kh, bh);
        }
    }
#pragma unroll
    for (int q = 0; q < 6; ++q) {
        const int which = q >> 1, nt = warp + 8 * (q & 1);
#pragma unroll
        for (int ii = 0; ii < 4; ++ii)
            if (c_row(L, ii) == slot) S.m[which][8 * nt + c_col(L, ii)] = fadd(big[q][ii], sml[q][ii]);
    }
}

#ifndef CFBI_ROWS_MINB
#define CFBI_ROWS_MINB 2
#endif
__global__ void __launch_bounds__(NTHREADS, CFBI_ROWS_MINB) rows_kernel(const Params p) {
    __shared__ RowSmem S;
    const int n = blockIdx.x, h = blockIdx.y, H = p.H;
    const SeqPlan sp = p.plan[n];
    const int nrows = sp.hrows;
    if (nrows == 0) return;
    const int tid = threadIdx.x, warp = tid >> 5;
    const LaneIdx L;
    const int g = L.g, t = L.t;

    float* R = p.rows + ((int64_t)sp.slot * H + h) * ROWS_PER_HEAD_FLOATS;
    float* Rk = R + PLANE_KHAT * ROW_PLANE_FLOATS;
    float* Rg = R + PLANE_G2 * ROW_PLANE_FLOATS;
    float* Rut = R + PLANE_UT * ROW_PLANE_FLOATS;
    float* Rw = R + PLANE_W * ROW_PLANE_FLOATS;
    float* Ru = R + PLANE_U * ROW_PLANE_FLOATS;
    float* Eg = p.E + ((int64_t)sp.slot * H + h) * E_FLOATS;
    const float ale = gate_a_log_exp(p.A_log[h]);
    const int which = tid >> 7, col = tid & 127;   // role in the UT | W recursion (tid < 128: also channel / value column)
    float* const Xg = which ? Rw : Rut;
    const float dtb = p.dt_bias[h * HEAD_K + col];

#ifdef CFBI_ROWS_PROFILE
    long long tp[9];
#endif
    int pos = sp.pos0;
    for (int i = 0; i < nrows; ++i) {
        ROWS_TS(0);
        const int64_t tok = (int64_t)sp.tok0 + i;
        const int64_t off = (tok * H + h) * HEAD_K;   // q/k/v/g/out row offset
        const int s = pos >> 4, b = pos & ~15, slot = pos & 15;
        const int slot_hi = slot >> 3;   // the current row is in A/C rows g (0) or g+8 (1) of the lanes with g == slot & 7
        const bool mine = g == (slot & 7);

        // ---- early loads: nothing here depends on this row's prep ------------------------------------
        if (i == 0) {   // pull the whole state this CTA will touch towards L2 so later phases do not pay HBM latency
            const char* eb = reinterpret_cast<const char*>(Eg);
            prefetch_l2(eb + tid * 128);
            prefetch_l2(eb + (tid + NTHREADS) * 128);
            const int lines = pos * 4;   // 512 B per row = 4 lines of 128 B, five planes
            for (int l = tid; l < 5 * lines; l += NTHREADS) prefetch_l2(R + (l / lines) * ROW_PLANE_FLOATS + (l % lines) * 32);
        }
        float xpre[16], upre[16];   // cached rows b..pos-1 of UT | W (role column) and of U (value column tid < 128)
#pragma unroll
        for (int jj = 0; jj < 16; ++jj) xpre[jj] = (b + jj < pos) ? Xg[(b + jj) * HEAD_K + col] : 0.f;
        float gprev = 0.f, gref_pre = 0.f, vraw = 0.f;
#pragma unroll
        for (int jj = 0; jj < 16; ++jj) upre[jj] = 0.f;
        if (tid < HEAD_V) {
#pragma unroll
            for (int jj = 0; jj < 16; ++jj) upre[jj] = (b + jj < pos) ? Ru[(b + jj) * HEAD_V + tid] : 0.f;
            if (pos > 0) gprev = Rg[(pos - 1) * HEAD_K + tid];
            if (b < pos) gref_pre = Rg[b * HEAD_K + tid];
            vraw = bf16_to_f32(p.v[off + tid]);
        }

        // ---- A. token-local prep -------------------------------------------------------------
        if (warp == 0) {   // lanes 0-15: k, lanes 16-31: q (channels 8*(lane&15) ..+7)
            const bool is_q = L.lane >= 16;
            const int c0 = 8 * (L.lane & 15);
            float x[8];
            load8_bf16((is_q ? p.q : p.k) + off + c0, x);
            const float inv = l2norm_inv(x);
            float* dst = is_q ? S.q : S.k;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float y = fmul(x[j], inv);
                dst[c0 + j] = is_q ? fmul(y, p.scale) : y;
            }
        }
        if (tid < HEAD_K) {
            const float a2 = gate_a2(bf16_to_f32(p.g[off + tid]), dtb, ale, p.scale2);
            const float g2v = (pos == 0) ? a2 : fadd(gprev, a2);
            S.g2[tid] = g2v;
            S.gref[tid] = (b == pos) ? g2v : gref_pre;
        }
        if (tid == HEAD_K) S.beta = beta_act(bf16_to_f32(p.beta[tok * H + h]));
        __syncthreads();
        ROWS_TS(1);

        // ---- B. Akk / Aqk row against the cached rows (and the row itself): one 8-column tile per warp, ks and qs
        //         packed into one A tile (rows slot and slot ^ 8), B = kjr shared ----------------------------------
        if (warp <= (pos >> 3)) {
            const int j0 = 8 * warp, j = j0 + g;
            float rk[16][2], rgv[16][2];
            const int jc = (j < pos) ? j : 0;   // clamped row: every lane loads, the select below discards unused values
#pragma unroll
            for (int ks = 0; ks < 16; ++ks)
#pragma unroll
                for (int ii = 0; ii < 2; ++ii) {
                    const int ch = 8 * ks + t + 4 * ii;
                    const float ak = Rk[jc * HEAD_K + ch], bg = Rg[jc * HEAD_K + ch];
                    const float sk = S.k[ch], sg = S.g2[ch];
                    rk[ks][ii] = (j < pos) ? ak : ((j == pos) ? sk : 0.f);
                    rgv[ks][ii] = (j < pos) ? bg : ((j == pos) ? sg : 0.f);
                }
            float big[4], sml[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) { big[i] = neg_zero(); sml[i] = neg_zero(); }
#pragma unroll
            for (int ks = 0; ks < 16; ++ks) {
                uint32_t ah[4], al[4];
#pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    const int ch = 8 * ks + t + 4 * (ii >> 1);
                    const float f = ex2_ftz(fsub(S.g2[ch], S.gref[ch]));
                    const float ksv = fmul(S.k[ch], f), qsv = fmul(S.q[ch], f);
                    const float v = mine ? (((ii & 1) == slot_hi) ? ksv : qsv) : 0.f;
                    split_tf32(v, ah[ii], al[ii]);
                }
                uint32_t bh[2], bl[2];
#pragma unroll
                for (int ii = 0; ii < 2; ++ii) {
                    const int ch = 8 * ks + t + 4 * ii;
                    const float kv = fmul(rk[ks][ii], ex2_ftz(fsub(S.gref[ch], rgv[ks][ii])));
                    split_tf32(j <= pos ? kv : 0.f, bh[ii], bl[ii]);
                }
                mma_tf32_m16n8k8(sml, al, bh);
                mma_tf32_m16n8k8(sml, ah, bl);
                mma_tf32_m16n8k8(big, ah, bh);
            }
            if (mine) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int c = j0 + 2 * t + e;
                    S.akk[c] = fmul(S.beta, fadd(big[2 * slot_hi + e], sml[2 * slot_hi + e]));
                    S.aqk[c] = fadd(big[2 * (1 - slot_hi) + e], sml[2 * (1 - slot_hi) + e]);
                }
            }
        }
        __syncthreads();
        ROWS_TS(2);

        // ---- C. cross-sub-block products (UT, W rows for the recursion; U rows for O), then the UT / W row -------
        if (s > 0) {
            if (s == 1) rows_m_tiles<2>(S, Rut, Rw, Ru, warp, L, slot);
            else if (s == 2) rows_m_tiles<4>(S, Rut, Rw, Ru, warp, L, slot);
            else rows_m_tiles<6>(S, Rut, Rw, Ru, warp, L, slot);
            __syncthreads();
        }
        ROWS_TS(3);
        {   // right-hand side, cross-block subtraction, in-sub-block FMA chain over the cached rows; thread = (UT|W, column)
            float rhs;
            if (which == 0) rhs = fmul(S.beta, vraw);
            else rhs = fmul(S.beta, fmul(ex2_ftz(S.g2[col]), S.k[col]));
            float x = (s > 0) ? fsub(rhs, S.m[which][col]) : rhs;
#pragma unroll
            for (int jj = 0; jj < 16; ++jj) {
                const float av = S.akk[b + jj];
#ifdef CFBI_NEG_CONTROL
                // negative control for the test harness: two roundings instead of one fma (deliberately WRONG)
                const float xn = fsub(x, fmul(av, xpre[jj]));
#else
                const float xn = ffma(-av, xpre[jj], x);
#endif
                x = (b + jj < pos) ? xn : x;
            }
            Xg[pos * HEAD_K + col] = x;
            S.x[which][col] = x;
            if (which == 0) S.gq[col] = fmul(ex2_ftz(S.g2[col]), S.q[col]);
            if (tid < HEAD_K) {   // append khat / G2 of this row to the cache (later rows read them)
                Rk[pos * HEAD_K + tid] = S.k[tid];
                Rg[pos * HEAD_K + tid] = S.g2[tid];
            }
        }
        __syncthreads();
        ROWS_TS(4);

        // ---- D. P1 = dot3(W_t, E^T), P2 = dot3(GQ_t, E^T): W in A-row slot, GQ in slot ^ 8, two V tiles per warp ----
        {
            float eb[16][2][2];
#pragma unroll
            for (int ks = 0; ks < 16; ++ks)
#pragma unroll
                for (int tl = 0; tl < 2; ++tl)
#pragma unroll
                    for (int ii = 0; ii < 2; ++ii) eb[ks][tl][ii] = Eg[(8 * (warp + 8 * tl) + g) * HEAD_K + 8 * ks + t + 4 * ii];
            float big[2][4], sml[2][4];
#pragma unroll
            for (int tl = 0; tl < 2; ++tl)
#pragma unroll
                for (int i = 0; i < 4; ++i) { big[tl][i] = neg_zero(); sml[tl][i] = neg_zero(); }
#pragma unroll
            for (int ks = 0; ks < 16; ++ks) {
                uint32_t ah[4], al[4];
#pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    const int kk = 8 * ks + t + 4 * (ii >> 1);
                    const float wv = S.x[1][kk], gv = S.gq[kk];
                    const float v = mine ? (((ii & 1) == slot_hi) ? wv : gv) : 0.f;
                    split_tf32(v, ah[ii], al[ii]);
                }
#pragma unroll
                for (int tl = 0; tl < 2; ++tl) {
                    uint32_t bh[2], bl[2];
#pragma unroll
                    for (int ii = 0; ii < 2; ++ii) split_tf32(eb[ks][tl][ii], bh[ii], bl[ii]);
                    mma_tf32_m16n8k8(sml[tl], al, bh);
                    mma_tf32_m16n8k8(sml[tl], ah, bl);
                    mma_tf32_m16n8k8(big[tl], ah, bh);
                }
            }
            if (mine) {
#pragma unroll
                for (int tl = 0; tl < 2; ++tl)
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int c = 8 * (warp + 8 * tl) + 2 * t + e;
                        S.p1[c] = fadd(big[tl][2 * slot_hi + e], sml[tl][2 * slot_hi + e]);
                        S.p2[c] = fadd(big[tl][2 * (1 - slot_hi) + e], sml[tl][2 * (1 - slot_hi) + e]);
                    }
            }
        }
        __syncthreads();
        ROWS_TS(5);

        // ---- E, F. U = UT - P1, O = P2 (+ M2) then the FMA chain over j = b..t -------------------------------
        if (tid < HEAD_V) {
            const float u = fsub(S.x[0][tid], S.p1[tid]);
            S.u[tid] = u;
            Ru[pos * HEAD_V + tid] = u;
            float o = S.p2[tid];
            if (s > 0) o = fadd(o, S.m[2][tid]);
#pragma unroll
            for (int jj = 0; jj < 16; ++jj) {
                const int j = b + jj;
                const float on = ffma(S.aqk[j], (j == pos) ? u : upre[jj], o);
                o = (j <= pos) ? on : o;
            }
            p.out[off + tid] = f32_to_bf16_rn(o);
            if (p.out_f32 != nullptr) p.out_f32[off + tid] = o;
        }
        ROWS_TS(6);

        // ---- G. chunk end: E' = E * GEND + dot3(KD, U) ----------------------------------------------
        if (pos == BT - 1) {
            if (tid < HEAD_K) S.gend[tid] = ex2_ftz(S.g2[tid]);
            __syncthreads();
            for (int jb = warp; jb < 128; jb += 8) {   // 8 key tiles x 16 value tiles
                const int mt = jb >> 4, nt = jb & 15;
                auto load_kd = [&](int ks, float (&a)[4]) {
#pragma unroll
                    for (int ii = 0; ii < 4; ++ii) {
                        const int k = 16 * mt + L.g + 8 * (ii & 1), j = 8 * ks + L.t + 4 * (ii >> 1);
                        a[ii] = fmul(Rk[j * HEAD_K + k], ex2_ftz(fsub(S.g2[k], Rg[j * HEAD_K + k])));
                    }
                };
                auto load_u = [&](int ks, float (&bb)[2]) {
#pragma unroll
                    for (int ii = 0; ii < 2; ++ii) bb[ii] = Ru[(8 * ks + L.t + 4 * ii) * HEAD_V + 8 * nt + L.g];
                };
                float out[4];
                dot3(BT / 8, load_kd, load_u, out);
#pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    const int k = 16 * mt + c_row(L, ii), v = 8 * nt + c_col(L, ii);
                    float* e = &Eg[v * HEAD_K + k];
                    *e = fadd(fmul(*e, S.gend[k]), out[ii]);
                }
            }
            pos = 0;
        } else {
            pos += 1;
        }
        __syncthreads();
#ifdef CFBI_ROWS_PROFILE
        ROWS_TS(7);
        if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && (i < 3 || pos % 16 == 4))
            printf("rows pos=%d: prep %lld | akk/aqk %lld | M tiles %lld | UT/W chain %lld | P1/P2 %lld | U,O %lld | chunk-end/sync %lld | total %lld\n", pos,
                   tp[1] - tp[0], tp[2] - tp[1], tp[3] - tp[2], tp[4] - tp[3], tp[5] - tp[4], tp[6] - tp[5], tp[7] - tp[6], tp[7] - tp[0]);
#endif
    }
}

}  // namespace cfbi
