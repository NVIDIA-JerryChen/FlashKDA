// Chunk-64 KDA, K2 arithmetic: one chunk of the recurrence from the bf16 state S [V][K] (shared memory).
//
//   P  = bf16(Kd  S^T)     O0 = bf16(Qd S^T)                         (K = 128: 8 k-steps of 16, ascending)
//   W' = bf16(bf16(v - P) * bf16(sigmoid(beta)))
//   U  = bf16(INV W')                                                (sub-block m = 0..b ascending)
//   out = bf16(O0 + bf16(Mqk U))
//   S' = bf16(fma(S, exp2(G_end), Kr^T U))                           (only when the chunk closes; b ascending)
//
// k2_step() is used by the prefill K2 kernel (4 warps, operands double-buffered in shared memory) and by the replay
// kernel (8 warps, operands straight out of K1's buffers); each warp owns NB = 8 / NWARP blocks of 16 value columns
// and never touches another warp's columns of S, so no barrier is needed inside a step. Sub-blocks b >= ceil(nvalid/16)
// are skipped (their rows are zero in a short chunk).
#pragma once

#include "c64_common.cuh"

namespace c64 {

struct K2Stage {
    const uint16_t* kd;    // [64][RS] bf16
    const uint16_t* qd;    // [64][RS]
    const uint16_t* kr;    // [64][RS]
    const uint16_t* v;     // [64][ldv] (this CTA's value columns only)
    const uint16_t* inv;   // [64][MS]
    const uint16_t* mq;    // [64][MS]
    const float* gt;       // [128] exp2(G_end)
    const uint16_t* braw;  // [64] raw beta logits
    int ldv;               // row stride of v
};

// emit(row, vcol, packed): the two bf16 outputs (vcol, vcol+1) of chunk row `row`; only called for row < nvalid
// NWARP warps x NB 16-column blocks = the value columns of this CTA (sS holds exactly those rows of S, row stride RS)
template <int NWARP, int NB, class Emit>
__device__ __forceinline__ void k2_step(const K2Stage& st, uint16_t* __restrict__ sS, int nvalid, bool close, Emit&& emit) {
    static_assert(NB == 1 || NB == 2, "NB is 1 or 2");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int vbase = warp * (16 * NB);
    const int nbv = (nvalid + 15) >> 4;

    float accP[4][NB][2][4], accO[4][NB][2][4];
    #pragma unroll
    for (int b = 0; b < 4; ++b)
        #pragma unroll
        for (int nb = 0; nb < NB; ++nb)
            #pragma unroll
            for (int u = 0; u < 2; ++u)
                #pragma unroll
                for (int e = 0; e < 4; ++e) { accP[b][nb][u][e] = 0.f; accO[b][nb][u][e] = 0.f; }

    // ---- phase 1: P = Kd S^T and O0 = Qd S^T
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        uint32_t bf[NB][4];
        #pragma unroll
        for (int nb = 0; nb < NB; ++nb) ldsm_x4(bf[nb], b_addr(sS + (vbase + 16 * nb) * RS + 16 * ks, RS, lane));
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            if (b < nbv) {
                uint32_t ak[4], aq[4];
                ldsm_x4(ak, a_addr(st.kd + (16 * b) * RS + 16 * ks, RS, lane));
                ldsm_x4(aq, a_addr(st.qd + (16 * b) * RS + 16 * ks, RS, lane));
                #pragma unroll
                for (int nb = 0; nb < NB; ++nb) {
                    #pragma unroll
                    for (int u = 0; u < 2; ++u) {
                        mma16816(accP[b][nb][u], ak, bf[nb][2 * u], bf[nb][2 * u + 1]);
                        mma16816(accO[b][nb][u], aq, bf[nb][2 * u], bf[nb][2 * u + 1]);
                    }
                }
            }
        }
    }

    // ---- phase 2: W' = bf16(bf16(v - P) * beta) as B fragments; O0 rounded to bf16
    uint32_t wB[4][NB][2][2];
    uint32_t o0[4][NB][2][2];
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        if (b < nbv) {
            const float sg0 = sigmoid_f(bf2f(st.braw[16 * b + g])), sg1 = sigmoid_f(bf2f(st.braw[16 * b + g + 8]));
            const uint32_t bt0 = pack_bf16x2(sg0, sg0), bt1 = pack_bf16x2(sg1, sg1);   // bf16(sigmoid(beta)) of rows g, g+8
            #pragma unroll
            for (int nb = 0; nb < NB; ++nb) {
                #pragma unroll
                for (int u = 0; u < 2; ++u) {
                    const int vc = vbase + 16 * nb + 8 * u + 2 * t;
                    const uint32_t v0 = ld_u32(&st.v[(16 * b + g) * st.ldv + vc]);
                    const uint32_t v1 = ld_u32(&st.v[(16 * b + g + 8) * st.ldv + vc]);
                    const float* P = accP[b][nb][u];
                    const uint32_t w0 = hmul2(hsub2(v0, pack_bf16x2(P[0], P[1])), bt0);
                    const uint32_t w1 = hmul2(hsub2(v1, pack_bf16x2(P[2], P[3])), bt1);
                    wB[b][nb][u][0] = movm_t(w0);
                    wB[b][nb][u][1] = movm_t(w1);
                    o0[b][nb][u][0] = pack_bf16x2(accO[b][nb][u][0], accO[b][nb][u][1]);
                    o0[b][nb][u][1] = pack_bf16x2(accO[b][nb][u][2], accO[b][nb][u][3]);
                }
            }
        }
    }

    // ---- phase 3: U = INV W' (block lower triangular, m ascending), kept as B fragments
    uint32_t uB[4][NB][2][2];
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        if (b < nbv) {
            float accU[NB][2][4];
            #pragma unroll
            for (int nb = 0; nb < NB; ++nb)
                #pragma unroll
                for (int u = 0; u < 2; ++u)
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) accU[nb][u][e] = 0.f;
            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                if (m <= b) {
                    uint32_t ai[4];
                    ldsm_x4(ai, a_addr(st.inv + (16 * b) * MS + 16 * m, MS, lane));
                    #pragma unroll
                    for (int nb = 0; nb < NB; ++nb)
                        #pragma unroll
                        for (int u = 0; u < 2; ++u) mma16816(accU[nb][u], ai, wB[m][nb][u][0], wB[m][nb][u][1]);
                }
            }
            #pragma unroll
            for (int nb = 0; nb < NB; ++nb) {
                #pragma unroll
                for (int u = 0; u < 2; ++u) {
                    uB[b][nb][u][0] = movm_t(pack_bf16x2(accU[nb][u][0], accU[nb][u][1]));
                    uB[b][nb][u][1] = movm_t(pack_bf16x2(accU[nb][u][2], accU[nb][u][3]));
                }
            }
        }
    }

    // ---- phase 4: out = bf16(O0 + bf16(Mqk U))
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        if (b < nbv) {
            float accM[NB][2][4];
            #pragma unroll
            for (int nb = 0; nb < NB; ++nb)
                #pragma unroll
                for (int u = 0; u < 2; ++u)
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) accM[nb][u][e] = 0.f;
            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                if (m <= b) {
                    uint32_t ai[4];
                    ldsm_x4(ai, a_addr(st.mq + (16 * b) * MS + 16 * m, MS, lane));
                    #pragma unroll
                    for (int nb = 0; nb < NB; ++nb)
                        #pragma unroll
                        for (int u = 0; u < 2; ++u) mma16816(accM[nb][u], ai, uB[m][nb][u][0], uB[m][nb][u][1]);
                }
            }
            #pragma unroll
            for (int nb = 0; nb < NB; ++nb) {
                #pragma unroll
                for (int u = 0; u < 2; ++u) {
                    const int vc = vbase + 16 * nb + 8 * u + 2 * t;
                    const uint32_t r0 = hadd2(o0[b][nb][u][0], pack_bf16x2(accM[nb][u][0], accM[nb][u][1]));
                    const uint32_t r1 = hadd2(o0[b][nb][u][1], pack_bf16x2(accM[nb][u][2], accM[nb][u][3]));
                    if (16 * b + g < nvalid) emit(16 * b + g, vc, r0);
                    if (16 * b + g + 8 < nvalid) emit(16 * b + g + 8, vc, r1);
                }
            }
        }
    }

    // ---- phase 6: S' = bf16(fma(S, exp2(G_end), Kr^T U)) for this warp's columns, all 128 k
    if (close) {
        #pragma unroll NB == 1 ? 2 : 1
        for (int mk = 0; mk < 8; ++mk) {
            float accS[NB][2][4];
            #pragma unroll
            for (int nb = 0; nb < NB; ++nb)
                #pragma unroll
                for (int u = 0; u < 2; ++u)
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) accS[nb][u][e] = 0.f;
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                if (b < nbv) {
                    uint32_t ak[4];
                    ldsm_x4_t(ak, at_addr(st.kr + (16 * b) * RS + 16 * mk, RS, lane));
                    #pragma unroll
                    for (int nb = 0; nb < NB; ++nb)
                        #pragma unroll
                        for (int u = 0; u < 2; ++u) mma16816(accS[nb][u], ak, uB[b][nb][u][0], uB[b][nb][u][1]);
                }
            }
            const float gk0 = st.gt[16 * mk + g], gk1 = st.gt[16 * mk + g + 8];
            #pragma unroll
            for (int nb = 0; nb < NB; ++nb) {
                // S tile (16 v x 16 k) in C layout: regs = (u0,k=g) (u0,k=g+8) (u1,k=g) (u1,k=g+8)
                uint16_t* tile = sS + (vbase + 16 * nb + (lane >> 4) * 8 + (lane & 7)) * RS + 16 * mk + ((lane >> 3) & 1) * 8;
                uint32_t sr[4];
                ldsm_x4_t(sr, tile);
                uint32_t nw[4];
                #pragma unroll
                for (int u = 0; u < 2; ++u) {
                    nw[2 * u] = pack_bf16x2(ffma(bf_lo(sr[2 * u]), gk0, accS[nb][u][0]),
                                            ffma(bf_hi(sr[2 * u]), gk0, accS[nb][u][1]));
                    nw[2 * u + 1] = pack_bf16x2(ffma(bf_lo(sr[2 * u + 1]), gk1, accS[nb][u][2]),
                                                ffma(bf_hi(sr[2 * u + 1]), gk1, accS[nb][u][3]));
                }
                stsm_x4_t(tile, nw[0], nw[1], nw[2], nw[3]);
            }
        }
    }
}

}  // namespace c64
