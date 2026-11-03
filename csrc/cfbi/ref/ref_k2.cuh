// FROZEN REFERENCE (v1). This is the correctness-first implementation of the contract, kept verbatim as the
// bitwise oracle for the optimised kernels (cfbi_k2.cuh): the tests run both and demand identical bits.
// Do not optimise or "fix" this file; a change here is a change of the numerical contract.
// CF-BI KDA K2: the E-dependent part, a serial scan over a sequence's chunks.
//
// One CTA per (V-slice of 16 columns, sequence, head). It keeps its E slice [16][128] in shared memory and,
// for every chunk: P1/P2 = dot3([W;GQ], E^T) -> U = UT - P1 -> O (cross-sub-block dot3 + FMA chain, bf16 out)
// -> for a full chunk the state update E' = E*GEND + dot3(KD, U). A trailing partial chunk only produces O
// and stores its U rows in the open-chunk row cache. All coefficients come from the K1 workspace.
#pragma once

#include "cfbi/cfbi_common.cuh"

namespace cfbi {
namespace ref {

constexpr int K2_BV = 16;   // value columns per CTA
constexpr int K2_SE = 132;  // sE row stride
constexpr int K2_SU = 24;   // sU / sP row stride (conflict-free B-fragment reads)

__global__ void __launch_bounds__(NTHREADS) k2_kernel(const Params p) {
    __shared__ __align__(16) float sE[K2_BV * K2_SE];
    __shared__ __align__(16) float sU[64 * K2_SU];
    __shared__ __align__(16) float sP[2 * 64 * K2_SU];   // [0] = P1, [1] = P2

    const int v0 = blockIdx.x * K2_BV, n = blockIdx.y, h = blockIdx.z, H = p.H;
    const SeqPlan sp = p.plan[n];
    const int nch = sp.nfull + (sp.tail > 0 ? 1 : 0);
    if (nch == 0) return;
    const int tid = threadIdx.x, warp = tid >> 5;
    const LaneIdx L;

    float* Eg = p.E != nullptr ? p.E + ((int64_t)n * H + h) * E_FLOATS + (int64_t)v0 * HEAD_K : nullptr;
    for (int idx = tid; idx < K2_BV * 32; idx += NTHREADS) {
        const int v = idx >> 5, c4 = idx & 31;
        *reinterpret_cast<float4*>(&sE[v * K2_SE + 4 * c4]) =
            Eg != nullptr ? *reinterpret_cast<const float4*>(&Eg[v * HEAD_K + 4 * c4]) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    __syncthreads();

    for (int lc = 0; lc < nch; ++lc) {
        const bool full = lc < sp.nfull;
        const int valid = full ? BT : sp.tail;
        const int nsb = (valid + SBR - 1) / SBR;
        const int64_t tok0 = (int64_t)sp.tok0 + sp.hrows + (int64_t)BT * lc;
        const float* wsb = p.ws + ((int64_t)(sp.cbase + lc) * H + h) * WS_FLOATS;
        const float* wg = wsb + WS_WG;
        const float* ut = wsb + WS_UT;
        const float* aqk = wsb + WS_AQK;
        const float* kd = wsb + WS_KD;
        const float* gend = wsb + WS_GEND;

        // ---- 1. P1 = dot3(W, E^T), P2 = dot3(GQ, E^T): (W|GQ) x row tile x 8-column tile ------------------
        for (int jb = warp; jb < 16; jb += 8) {
            const int nt = jb & 1, rt = (jb >> 1) & 3, which = jb >> 3;
            if (rt >= nsb) continue;
            const float* a_rows = wg + (int64_t)(which * 64 + 16 * rt) * 128;
            auto load_a = [&](int ks, float (&a)[4]) {
#pragma unroll
                for (int ii = 0; ii < 4; ++ii) a[ii] = a_rows[(L.g + 8 * (ii & 1)) * 128 + 8 * ks + L.t + 4 * (ii >> 1)];
            };
            auto load_e = [&](int ks, float (&bb)[2]) {
#pragma unroll
                for (int ii = 0; ii < 2; ++ii) bb[ii] = sE[(8 * nt + L.g) * K2_SE + 8 * ks + L.t + 4 * ii];
            };
            float out[4];
            dot3(HEAD_K / 8, load_a, load_e, out);
#pragma unroll
            for (int ii = 0; ii < 4; ++ii)
                sP[which * 64 * K2_SU + (16 * rt + c_row(L, ii)) * K2_SU + 8 * nt + c_col(L, ii)] = out[ii];
        }
        __syncthreads();

        // ---- 2. U = UT - P1 (and the open-chunk cache of a partial chunk) ---------------------------------
        for (int idx = tid; idx < nsb * 16 * K2_BV; idx += NTHREADS) {
            const int row = idx >> 4, col = idx & 15;
            const float u = fsub(ut[row * 128 + v0 + col], sP[row * K2_SU + col]);
            sU[row * K2_SU + col] = u;
            if (!full && p.rows != nullptr && row < valid)
                p.rows[((int64_t)n * H + h) * ROWS_PER_HEAD_FLOATS + PLANE_U * ROW_PLANE_FLOATS + row * 128 + v0 + col] = u;
        }
        __syncthreads();

        // ---- 3. O = P2 + dot3(Aqk[b.., 0:b], U[0:b]) then FMA chain over j = b..t -----------------------
        {
            const int s = warp >> 1, nt = warp & 1;
            if (s < nsb) {
                const int b = 16 * s;
                float m[4] = {0.f, 0.f, 0.f, 0.f};
                if (s > 0) {
                    auto load_a = [&](int ks, float (&a)[4]) {
#pragma unroll
                        for (int ii = 0; ii < 4; ++ii)
                            a[ii] = aqk[(b + L.g + 8 * (ii & 1)) * 64 + 8 * ks + L.t + 4 * (ii >> 1)];
                    };
                    auto load_u = [&](int ks, float (&bb)[2]) {
#pragma unroll
                        for (int ii = 0; ii < 2; ++ii) bb[ii] = sU[(8 * ks + L.t + 4 * ii) * K2_SU + 8 * nt + L.g];
                    };
                    dot3(2 * s, load_a, load_u, m);
                }
#pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    const int row = b + c_row(L, ii), col = 8 * nt + c_col(L, ii);
                    float o = sP[64 * K2_SU + row * K2_SU + col];
                    if (s > 0) o = fadd(o, m[ii]);
                    for (int j = b; j <= row; ++j) o = ffma(aqk[row * 64 + j], sU[j * K2_SU + col], o);
                    if (row < valid) {
                        const int64_t oi = ((tok0 + row) * H + h) * HEAD_V + v0 + col;
                        p.out[oi] = f32_to_bf16_rn(o);
                        if (p.out_f32 != nullptr) p.out_f32[oi] = o;
                    }
                }
            }
        }

        // ---- 4. state update (full chunks): E' = E*GEND + dot3(KD, U) ---------------------------------------
        if (full) {
            for (int jb = warp; jb < 16; jb += 8) {   // 8 key tiles x 2 value tiles
                const int mt = jb >> 1, nt = jb & 1;
                auto load_kd = [&](int ks, float (&a)[4]) {
#pragma unroll
                    for (int ii = 0; ii < 4; ++ii)
                        a[ii] = kd[(16 * mt + L.g + 8 * (ii & 1)) * 64 + 8 * ks + L.t + 4 * (ii >> 1)];
                };
                auto load_u = [&](int ks, float (&bb)[2]) {
#pragma unroll
                    for (int ii = 0; ii < 2; ++ii) bb[ii] = sU[(8 * ks + L.t + 4 * ii) * K2_SU + 8 * nt + L.g];
                };
                float out[4];
                dot3(BT / 8, load_kd, load_u, out);
#pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    const int k = 16 * mt + c_row(L, ii), v = 8 * nt + c_col(L, ii);
                    float* e = &sE[v * K2_SE + k];
                    *e = fadd(fmul(*e, gend[k]), out[ii]);
                }
            }
        }
        __syncthreads();
    }

    if (Eg != nullptr) {
        for (int idx = tid; idx < K2_BV * 32; idx += NTHREADS) {
            const int v = idx >> 5, c4 = idx & 31;
            *reinterpret_cast<float4*>(&Eg[v * HEAD_K + 4 * c4]) = *reinterpret_cast<const float4*>(&sE[v * K2_SE + 4 * c4]);
        }
    }
}

}  // namespace ref
}  // namespace cfbi
