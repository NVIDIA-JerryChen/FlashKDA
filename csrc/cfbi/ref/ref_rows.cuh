// FROZEN REFERENCE (v1). This is the correctness-first implementation of the contract, kept verbatim as the
// bitwise oracle for the optimised kernels (cfbi_rows.cuh): the tests run both and demand identical bits.
// Do not optimise or "fix" this file; a change here is a change of the numerical contract.
// CF-BI KDA: the row engine = decode = "one more row of the chunk program".
//
// One CTA (8 warps) per (sequence, head). It appends the sequence's head rows one by one to the open chunk:
//   A prep -> B Akk/Aqk row vs cached rows -> C UT/W row -> D P1/P2 with E -> E U -> F O -> G chunk end.
// Every tensor-core product goes through dot3<> with the same tile geometry as K1/K2: the valid row of the
// A operand sits in slot (pos & 15), the other 15 slots are zero.
// The cache planes (khat, G2, UT, W, U) and E live in global memory (fp32); plain (non-const, non-restrict)
// pointers are used so the compiler never turns reads of freshly written rows into .nc loads.
#pragma once

#include "cfbi/cfbi_common.cuh"

namespace cfbi {
namespace ref {

struct RowSmem {
    float k[HEAD_K];      // khat of the current row
    float q[HEAD_K];      // qhat of the current row
    float g2[HEAD_K];     // G2 of the current row
    float gref[HEAD_K];   // G2 of the reference row r(pos)
    float gq[HEAD_K];     // GQ = Gamma * qhat
    float gend[HEAD_K];   // GEND (chunk end only)
    float x[2][HEAD_K];   // UT_t, W_t of the current row
    float m[2][HEAD_V];   // cross-sub-block products (UT / W; later O)
    float p1[HEAD_V], p2[HEAD_V], u[HEAD_V];
    float akk[BT], aqk[BT];
    float beta;
};

__global__ void __launch_bounds__(NTHREADS) rows_kernel(const Params p) {
    __shared__ RowSmem S;
    const int n = blockIdx.x, h = blockIdx.y, H = p.H;
    const SeqPlan sp = p.plan[n];
    const int nrows = sp.hrows;
    if (nrows == 0) return;
    const int tid = threadIdx.x, warp = tid >> 5;
    const LaneIdx L;

    float* R = p.rows + ((int64_t)n * H + h) * ROWS_PER_HEAD_FLOATS;
    float* Rk = R + PLANE_KHAT * ROW_PLANE_FLOATS;
    float* Rg = R + PLANE_G2 * ROW_PLANE_FLOATS;
    float* Rut = R + PLANE_UT * ROW_PLANE_FLOATS;
    float* Rw = R + PLANE_W * ROW_PLANE_FLOATS;
    float* Ru = R + PLANE_U * ROW_PLANE_FLOATS;
    float* Eg = p.E + ((int64_t)n * H + h) * E_FLOATS;
    const float ale = gate_a_log_exp(p.A_log[h]);

    int pos = sp.pos0;
    for (int i = 0; i < nrows; ++i) {
        const int64_t tok = (int64_t)sp.tok0 + i;
        const int64_t off = (tok * H + h) * HEAD_K;   // q/k/v/g/out row offset

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
            const float a2 = gate_a2(bf16_to_f32(p.g[off + tid]), p.dt_bias[h * HEAD_K + tid], ale, p.scale2);
            S.g2[tid] = (pos == 0) ? a2 : fadd(Rg[(pos - 1) * HEAD_K + tid], a2);
        }
        if (tid == HEAD_K) S.beta = beta_act(bf16_to_f32(p.beta[tok * H + h]));
        __syncthreads();
        if (tid < HEAD_K) {   // append khat / G2 to the cache; fetch the reference row's G2
            Rk[pos * HEAD_K + tid] = S.k[tid];
            Rg[pos * HEAD_K + tid] = S.g2[tid];
            const int r = pos & ~15;
            S.gref[tid] = (r == pos) ? S.g2[tid] : Rg[r * HEAD_K + tid];
        }
        __syncthreads();

        // ---- B. Akk / Aqk row against the cached rows (and the row itself for Aqk) -----------------
        const int slot = pos & 15;
        if (warp <= (pos >> 3)) {   // one 8-column tile per warp
            const int j0 = 8 * warp;
            auto load_k = [&](int ks, float (&a)[4]) {
#pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    const int row = L.g + 8 * (ii & 1), ch = 8 * ks + L.t + 4 * (ii >> 1);
                    a[ii] = (row == slot) ? fmul(S.k[ch], ex2_ftz(fsub(S.g2[ch], S.gref[ch]))) : 0.f;
                }
            };
            auto load_q = [&](int ks, float (&a)[4]) {
#pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    const int row = L.g + 8 * (ii & 1), ch = 8 * ks + L.t + 4 * (ii >> 1);
                    a[ii] = (row == slot) ? fmul(S.q[ch], ex2_ftz(fsub(S.g2[ch], S.gref[ch]))) : 0.f;
                }
            };
            auto load_kjr = [&](int ks, float (&b)[2]) {
                const int j = j0 + L.g;
#pragma unroll
                for (int ii = 0; ii < 2; ++ii) {
                    const int ch = 8 * ks + L.t + 4 * ii;
                    float val = 0.f;
                    if (j < pos) val = fmul(Rk[j * HEAD_K + ch], ex2_ftz(fsub(S.gref[ch], Rg[j * HEAD_K + ch])));
                    else if (j == pos) val = fmul(S.k[ch], ex2_ftz(fsub(S.gref[ch], S.g2[ch])));
                    b[ii] = val;
                }
            };
            float ok[4], oq[4];
            dot3(HEAD_K / 8, load_k, load_kjr, ok);
            dot3(HEAD_K / 8, load_q, load_kjr, oq);
#pragma unroll
            for (int ii = 0; ii < 4; ++ii) {
                if (c_row(L, ii) == slot) {
                    const int col = j0 + c_col(L, ii);
                    S.akk[col] = fmul(S.beta, ok[ii]);
                    S.aqk[col] = oq[ii];
                }
            }
        }
        __syncthreads();

        // ---- C. UT / W row: block forward substitution -------------------------------------------
        const int s = pos >> 4, b = pos & ~15;
        if (s > 0) {   // M = dot3(Akk[t, 0:b], X[0:b]) for X = UT and W (32 column tiles over 8 warps)
            for (int jb = warp; jb < 32; jb += 8) {
                const int which = jb >> 4, nt = jb & 15;
                const float* X = which ? Rw : Rut;
                auto load_a = [&](int ks, float (&a)[4]) {
#pragma unroll
                    for (int ii = 0; ii < 4; ++ii) {
                        const int row = L.g + 8 * (ii & 1);
                        a[ii] = (row == slot) ? S.akk[8 * ks + L.t + 4 * (ii >> 1)] : 0.f;
                    }
                };
                auto load_x = [&](int ks, float (&bb)[2]) {
#pragma unroll
                    for (int ii = 0; ii < 2; ++ii) bb[ii] = X[(8 * ks + L.t + 4 * ii) * HEAD_K + 8 * nt + L.g];
                };
                float out[4];
                dot3(2 * s, load_a, load_x, out);
#pragma unroll
                for (int ii = 0; ii < 4; ++ii)
                    if (c_row(L, ii) == slot) S.m[which][8 * nt + c_col(L, ii)] = out[ii];
            }
            __syncthreads();
        }
        {   // right-hand side, cross-block subtraction, in-sub-block FMA chain; thread = (UT|W, column)
            const int which = tid >> 7, col = tid & 127;
            float rhs;
            if (which == 0) rhs = fmul(S.beta, bf16_to_f32(p.v[off + col]));
            else rhs = fmul(S.beta, fmul(ex2_ftz(S.g2[col]), S.k[col]));
            float x = (s > 0) ? fsub(rhs, S.m[which][col]) : rhs;
            float* X = which ? Rw : Rut;
#ifdef CFBI_NEG_CONTROL
            // negative control for the test harness: two roundings instead of one fma (deliberately WRONG)
            for (int j = b; j < pos; ++j) x = fsub(x, fmul(S.akk[j], X[j * HEAD_K + col]));
#else
            for (int j = b; j < pos; ++j) x = ffma(-S.akk[j], X[j * HEAD_K + col], x);
#endif
            X[pos * HEAD_K + col] = x;
            S.x[which][col] = x;
            if (which == 0) S.gq[col] = fmul(ex2_ftz(S.g2[col]), S.q[col]);
        }
        __syncthreads();

        // ---- D. P1 = dot3(W_t, E^T), P2 = dot3(GQ_t, E^T) ----------------------------------------
        for (int jb = warp; jb < 32; jb += 8) {
            const int which = jb >> 4, nt = jb & 15;
            const float* av = which ? S.gq : S.x[1];
            auto load_a = [&](int ks, float (&a)[4]) {
#pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    const int row = L.g + 8 * (ii & 1);
                    a[ii] = (row == slot) ? av[8 * ks + L.t + 4 * (ii >> 1)] : 0.f;
                }
            };
            auto load_e = [&](int ks, float (&bb)[2]) {
#pragma unroll
                for (int ii = 0; ii < 2; ++ii) bb[ii] = Eg[(8 * nt + L.g) * HEAD_K + 8 * ks + L.t + 4 * ii];
            };
            float out[4];
            dot3(HEAD_K / 8, load_a, load_e, out);
#pragma unroll
            for (int ii = 0; ii < 4; ++ii)
                if (c_row(L, ii) == slot) (which ? S.p2 : S.p1)[8 * nt + c_col(L, ii)] = out[ii];
        }
        __syncthreads();

        // ---- E. U = UT - P1 ----------------------------------------------------------------------
        if (tid < HEAD_V) {
            const float u = fsub(S.x[0][tid], S.p1[tid]);
            S.u[tid] = u;
            Ru[pos * HEAD_V + tid] = u;
        }
        __syncthreads();

        // ---- F. O = P2 + dot3(Aqk[t, 0:b], U[0:b]) then FMA chain over j = b..t -----------------
        if (s > 0) {
            for (int nt = warp; nt < 16; nt += 8) {
                auto load_a = [&](int ks, float (&a)[4]) {
#pragma unroll
                    for (int ii = 0; ii < 4; ++ii) {
                        const int row = L.g + 8 * (ii & 1);
                        a[ii] = (row == slot) ? S.aqk[8 * ks + L.t + 4 * (ii >> 1)] : 0.f;
                    }
                };
                auto load_u = [&](int ks, float (&bb)[2]) {
#pragma unroll
                    for (int ii = 0; ii < 2; ++ii) bb[ii] = Ru[(8 * ks + L.t + 4 * ii) * HEAD_V + 8 * nt + L.g];
                };
                float out[4];
                dot3(2 * s, load_a, load_u, out);
#pragma unroll
                for (int ii = 0; ii < 4; ++ii)
                    if (c_row(L, ii) == slot) S.m[0][8 * nt + c_col(L, ii)] = out[ii];
            }
            __syncthreads();
        }
        if (tid < HEAD_V) {
            float o = S.p2[tid];
            if (s > 0) o = fadd(o, S.m[0][tid]);
            for (int j = b; j <= pos; ++j) o = ffma(S.aqk[j], (j == pos) ? S.u[tid] : Ru[j * HEAD_V + tid], o);
            p.out[off + tid] = f32_to_bf16_rn(o);
            if (p.out_f32 != nullptr) p.out_f32[off + tid] = o;
        }

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
    }
}

}  // namespace ref
}  // namespace cfbi
