// FROZEN REFERENCE (v1). This is the correctness-first implementation of the contract, kept verbatim as the
// bitwise oracle for the optimised kernels (cfbi_k1.cuh): the tests run both and demand identical bits.
// Do not optimise or "fix" this file; a change here is a change of the numerical contract.
// CF-BI KDA K1: everything of a chunk that does not depend on the entry state E.
//
// One CTA per (chunk, head): prep (l2norm / gate / cumsum / beta), Akk and Aqk (three-term TF32 over K=128),
// block forward substitution for UT and W (cross-sub-block dot3 + in-sub-block FMA chain), then the
// workspace WG=[W;GQ], UT, Aqk and (full chunks) KD, GEND for K2. A trailing partial chunk additionally
// stores khat, G2, UT, W of its rows in the open-chunk row cache for the next call.
//
// smem (floats): khat/qhat/G2 [64][132], UT/W [64][136], Akk/Aqk [64][68], M [2][16][132], beta [64]
#pragma once

#include "cfbi/cfbi_common.cuh"

namespace cfbi {
namespace ref {

constexpr int K1_SKH = 132;  // khat / qhat / G2 row stride: conflict-free A-fragment style reads
constexpr int K1_SX = 136;   // UT / W row stride: conflict-free B-fragment style reads
constexpr int K1_SA = 68;    // Akk / Aqk row stride
constexpr int K1_SM = 132;
constexpr int K1_SMEM_FLOATS = 3 * 64 * K1_SKH + 2 * 64 * K1_SX + 2 * 64 * K1_SA + 2 * 16 * K1_SM + 64;
constexpr size_t K1_SMEM_BYTES = (size_t)K1_SMEM_FLOATS * sizeof(float);

__global__ void __launch_bounds__(NTHREADS, 1) k1_kernel(const Params p) {
    extern __shared__ __align__(16) float smem[];
    float* sK = smem;                    // khat   [64][132]
    float* sQ = sK + 64 * K1_SKH;        // qhat   [64][132]
    float* sG = sQ + 64 * K1_SKH;        // G2     [64][132]
    float* sUT = sG + 64 * K1_SKH;       // UT     [64][136]
    float* sW = sUT + 64 * K1_SX;        // W      [64][136]
    float* sAkk = sW + 64 * K1_SX;       // Akk    [64][68]
    float* sAqk = sAkk + 64 * K1_SA;     // Aqk    [64][68]
    float* sM = sAqk + 64 * K1_SA;       // M      [2][16][132]
    float* sBeta = sM + 2 * 16 * K1_SM;  // beta   [64]

    const int c = blockIdx.x, h = blockIdx.y, H = p.H;
    if (c >= p.plan[p.N].cbase) return;
    // sequence of this chunk: largest n with plan[n].cbase <= c
    int lo = 0, hi = p.N - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (p.plan[mid].cbase <= c) lo = mid; else hi = mid - 1;
    }
    const int n = lo;
    const SeqPlan sp = p.plan[n];
    const int row0 = sp.hrows + BT * (c - sp.cbase);   // first row of the chunk among the sequence's call rows
    const int valid = min(BT, sp.len - row0);
    const int nsb = (valid + SBR - 1) / SBR;
    const bool full = valid == BT;
    const int64_t tok0 = (int64_t)sp.tok0 + row0;
    const int tid = threadIdx.x, warp = tid >> 5;
    const LaneIdx L;

    for (int i = tid; i < 64 * K1_SA; i += NTHREADS) { sAkk[i] = 0.f; sAqk[i] = 0.f; }
    for (int i = tid; i < 64 * K1_SX; i += NTHREADS) { sUT[i] = 0.f; sW[i] = 0.f; }

    // ---- 1. prep ---------------------------------------------------------------------------------
    {   // l2norm of k (vec 0..63) and q (vec 64..127): 16-lane groups, 8 channels per lane
        const int grp = tid >> 4, lj = tid & 15;
#pragma unroll 1
        for (int r = 0; r < 8; ++r) {
            const int vec = grp + 16 * r;
            const bool isq = vec >= 64;
            const int row = isq ? vec - 64 : vec;
            float x[8];
            if (row < valid) {
                load8_bf16((isq ? p.q : p.k) + ((tok0 + row) * H + h) * HEAD_K + 8 * lj, x);
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) x[j] = 0.f;
            }
            const float inv = l2norm_inv(x);   // all lanes take part in the shuffles
            float* dst = (isq ? sQ : sK) + row * K1_SKH + 8 * lj;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float y = fmul(x[j], inv);
                dst[j] = isq ? fmul(y, p.scale) : y;
            }
        }
    }
    {   // a2 per (row, channel) and beta
        const int ch = tid & 127, half = tid >> 7;
        const float ale = gate_a_log_exp(p.A_log[h]);
        const float dtb = p.dt_bias[h * HEAD_K + ch];
        for (int row = half; row < 64; row += 2) {
            float a2 = 0.f;
            if (row < valid)
                a2 = gate_a2(bf16_to_f32(p.g[((tok0 + row) * H + h) * HEAD_K + ch]), dtb, ale, p.scale2);
            sG[row * K1_SKH + ch] = a2;
        }
        if (tid < 64) sBeta[tid] = tid < valid ? beta_act(bf16_to_f32(p.beta[(tok0 + tid) * H + h])) : 0.f;
    }
    __syncthreads();
    if (tid < 128) {   // sequential cumsum over rows: G2[0] = a2[0], G2[t] = G2[t-1] + a2[t]
        float acc = 0.f;
        for (int row = 0; row < 64; ++row) {
            const float a = sG[row * K1_SKH + tid];
            acc = (row == 0) ? a : fadd(acc, a);
            sG[row * K1_SKH + tid] = acc;
        }
    }
    __syncthreads();

    // ---- 2. Akk / Aqk: one (sub-block s, 8-column tile) job per warp turn -------------------------
    {
        int job = 0;
        for (int s = 0; s < nsb; ++s) {
            for (int nt = 0; nt < 2 * s + 2; ++nt, ++job) {
                if ((job & 7) != warp) continue;
                const int r0 = 16 * s, j0 = 8 * nt;
                auto load_scaled = [&](const float* src, int ks, float (&a)[4]) {
#pragma unroll
                    for (int ii = 0; ii < 4; ++ii) {
                        const int row = r0 + L.g + 8 * (ii & 1), ch = 8 * ks + L.t + 4 * (ii >> 1);
                        a[ii] = (row < valid)
                                    ? fmul(src[row * K1_SKH + ch],
                                           ex2_ftz(fsub(sG[row * K1_SKH + ch], sG[r0 * K1_SKH + ch])))
                                    : 0.f;
                    }
                };
                auto load_kjr = [&](int ks, float (&b)[2]) {
                    const int j = j0 + L.g;
#pragma unroll
                    for (int ii = 0; ii < 2; ++ii) {
                        const int ch = 8 * ks + L.t + 4 * ii;
                        b[ii] = (j < valid)
                                    ? fmul(sK[j * K1_SKH + ch],
                                           ex2_ftz(fsub(sG[r0 * K1_SKH + ch], sG[j * K1_SKH + ch])))
                                    : 0.f;
                    }
                };
                float ok[4], oq[4];
                dot3(HEAD_K / 8, [&](int ks, float (&a)[4]) { load_scaled(sK, ks, a); }, load_kjr, ok);
                dot3(HEAD_K / 8, [&](int ks, float (&a)[4]) { load_scaled(sQ, ks, a); }, load_kjr, oq);
#pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    const int row = r0 + c_row(L, ii), col = j0 + c_col(L, ii);
                    sAkk[row * K1_SA + col] = fmul(sBeta[row], ok[ii]);
                    sAqk[row * K1_SA + col] = oq[ii];
                }
            }
        }
    }
    __syncthreads();

    // ---- 3. block forward substitution for UT and W ----------------------------------------------
    for (int s = 0; s < nsb; ++s) {
        const int b = 16 * s;
        if (s > 0) {   // M = dot3(Akk[b.., 0:b], X[0:b]) for X = UT, W: 32 column tiles over 8 warps
            for (int jb = warp; jb < 32; jb += 8) {
                const int which = jb >> 4, nt = jb & 15;
                const float* X = which ? sW : sUT;
                auto load_a = [&](int ks, float (&a)[4]) {
#pragma unroll
                    for (int ii = 0; ii < 4; ++ii)
                        a[ii] = sAkk[(b + L.g + 8 * (ii & 1)) * K1_SA + 8 * ks + L.t + 4 * (ii >> 1)];
                };
                auto load_x = [&](int ks, float (&bb)[2]) {
#pragma unroll
                    for (int ii = 0; ii < 2; ++ii) bb[ii] = X[(8 * ks + L.t + 4 * ii) * K1_SX + 8 * nt + L.g];
                };
                float out[4];
                dot3(2 * s, load_a, load_x, out);
#pragma unroll
                for (int ii = 0; ii < 4; ++ii)
                    sM[(which * 16 + c_row(L, ii)) * K1_SM + 8 * nt + c_col(L, ii)] = out[ii];
            }
            __syncthreads();
        }
        {   // thread = (UT | W, column): rhs, subtract M, FMA chain over the sub-block's earlier rows
            const int which = tid >> 7, col = tid & 127;
            float xr[16];
#pragma unroll
            for (int tt = 0; tt < 16; ++tt) {
                const int row = b + tt;
                float rhs;
                if (which == 0) {
                    const float vv = row < valid ? bf16_to_f32(p.v[((tok0 + row) * H + h) * HEAD_V + col]) : 0.f;
                    rhs = fmul(sBeta[row], vv);
                } else {
                    rhs = fmul(sBeta[row], fmul(ex2_ftz(sG[row * K1_SKH + col]), sK[row * K1_SKH + col]));
                }
                float x = (s > 0) ? fsub(rhs, sM[(which * 16 + tt) * K1_SM + col]) : rhs;
#pragma unroll
                for (int j = 0; j < tt; ++j) x = ffma(-sAkk[row * K1_SA + b + j], xr[j], x);
                xr[tt] = x;
                (which ? sW : sUT)[row * K1_SX + col] = x;
            }
        }
        __syncthreads();
    }

    // ---- 4. workspace for K2 (and the row cache of a trailing partial chunk) ----------------------
    float* wsb = p.ws + ((int64_t)c * H + h) * WS_FLOATS;
    const int rows_done = nsb * 16;
    for (int idx = tid; idx < 128 * 32; idx += NTHREADS) {   // WG: rows 0..63 = W, 64..127 = GQ
        const int row = idx >> 5, c4 = idx & 31;
        float4 val;
        if (row < 64) {
            val = row < rows_done ? *reinterpret_cast<const float4*>(&sW[row * K1_SX + 4 * c4])
                                  : make_float4(0.f, 0.f, 0.f, 0.f);
        } else {
            const int r = row - 64;
            val.x = fmul(ex2_ftz(sG[r * K1_SKH + 4 * c4 + 0]), sQ[r * K1_SKH + 4 * c4 + 0]);
            val.y = fmul(ex2_ftz(sG[r * K1_SKH + 4 * c4 + 1]), sQ[r * K1_SKH + 4 * c4 + 1]);
            val.z = fmul(ex2_ftz(sG[r * K1_SKH + 4 * c4 + 2]), sQ[r * K1_SKH + 4 * c4 + 2]);
            val.w = fmul(ex2_ftz(sG[r * K1_SKH + 4 * c4 + 3]), sQ[r * K1_SKH + 4 * c4 + 3]);
        }
        *reinterpret_cast<float4*>(&wsb[WS_WG + row * 128 + 4 * c4]) = val;
    }
    for (int idx = tid; idx < 64 * 32; idx += NTHREADS) {    // UT
        const int row = idx >> 5, c4 = idx & 31;
        *reinterpret_cast<float4*>(&wsb[WS_UT + row * 128 + 4 * c4]) =
            row < rows_done ? *reinterpret_cast<const float4*>(&sUT[row * K1_SX + 4 * c4])
                            : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    for (int idx = tid; idx < 64 * 16; idx += NTHREADS) {    // Aqk
        const int row = idx >> 4, c4 = idx & 15;
        *reinterpret_cast<float4*>(&wsb[WS_AQK + row * 64 + 4 * c4]) =
            *reinterpret_cast<const float4*>(&sAqk[row * K1_SA + 4 * c4]);
    }
    if (full) {
        {   // KD[k][j] = khat[j][k] * ex2(G2e[k] - G2[j][k]); thread = (k, half of the j range)
            const int k = tid & 127, jh = tid >> 7;
            const float g2e = sG[63 * K1_SKH + k];
#pragma unroll
            for (int j4 = 0; j4 < 8; ++j4) {
                const int j = jh * 32 + 4 * j4;
                float4 val;
                val.x = fmul(sK[(j + 0) * K1_SKH + k], ex2_ftz(fsub(g2e, sG[(j + 0) * K1_SKH + k])));
                val.y = fmul(sK[(j + 1) * K1_SKH + k], ex2_ftz(fsub(g2e, sG[(j + 1) * K1_SKH + k])));
                val.z = fmul(sK[(j + 2) * K1_SKH + k], ex2_ftz(fsub(g2e, sG[(j + 2) * K1_SKH + k])));
                val.w = fmul(sK[(j + 3) * K1_SKH + k], ex2_ftz(fsub(g2e, sG[(j + 3) * K1_SKH + k])));
                *reinterpret_cast<float4*>(&wsb[WS_KD + k * 64 + j]) = val;
            }
        }
        if (tid < 128) wsb[WS_GEND + tid] = ex2_ftz(sG[63 * K1_SKH + tid]);
    } else if (p.rows != nullptr) {
        float* R = p.rows + ((int64_t)n * H + h) * ROWS_PER_HEAD_FLOATS;
        for (int idx = tid; idx < valid * 32; idx += NTHREADS) {
            const int row = idx >> 5, c4 = idx & 31;
            *reinterpret_cast<float4*>(&R[PLANE_KHAT * ROW_PLANE_FLOATS + row * 128 + 4 * c4]) =
                *reinterpret_cast<const float4*>(&sK[row * K1_SKH + 4 * c4]);
            *reinterpret_cast<float4*>(&R[PLANE_G2 * ROW_PLANE_FLOATS + row * 128 + 4 * c4]) =
                *reinterpret_cast<const float4*>(&sG[row * K1_SKH + 4 * c4]);
            *reinterpret_cast<float4*>(&R[PLANE_UT * ROW_PLANE_FLOATS + row * 128 + 4 * c4]) =
                *reinterpret_cast<const float4*>(&sUT[row * K1_SX + 4 * c4]);
            *reinterpret_cast<float4*>(&R[PLANE_W * ROW_PLANE_FLOATS + row * 128 + 4 * c4]) =
                *reinterpret_cast<const float4*>(&sW[row * K1_SX + 4 * c4]);
        }
    }
}

}  // namespace ref
}  // namespace cfbi
