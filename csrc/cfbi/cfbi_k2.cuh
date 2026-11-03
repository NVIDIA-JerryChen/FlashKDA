// CF-BI KDA K2: the E-dependent part, a serial scan over a sequence's chunks (optimised).
//
// Same arithmetic as csrc/cfbi/ref/ref_k2.cuh (every output element sees the identical sequence of roundings;
// tests/cfbi/test_vs_ref.py compares bits), organised for the hardware:
//   * K1 stores every operand "fragment native" (cfbi_common.cuh, WS2_*): one cp.async.bulk + mbarrier per
//     operand moves it into shared memory and each lane reads its mma.sync fragment as one LDS.128 / LDS.64;
//   * E lives in shared memory twice: fp32 (for E*GEND) and as pre-split TF32 hi/lo B-fragments, so the P1/P2
//     loops do no ALU work on B; U is stored the same way for P3 and the cross-block part of O;
//   * one CTA = (V slice of K2_BV columns, sequence, head), 8 compute warps in lock step per chunk:
//         phase 1   warps 0-3: P1 = W.E^T (K=128) and U = UT - P1       | warps 4-7: P2 = GQ.E^T (K=128)
//         S1        (CTA barrier; copies of the next chunk's W / GQ / UT are issued)
//         phase 3   warps 0-3: P3 = KD.U (K=64) and E' = E*GEND + P3    | warps 4-7: O (cross-block dot3 + FMA chain)
//         S2        (CTA barrier; copies of the next chunk's Aqk / KD are issued)
//     Each operand buffer is single-buffered and refilled as soon as its last reader has passed a barrier, so the
//     next chunk's operands stream in behind the current chunk's tensor-core work.
//   * independent accumulator chains (all V tiles of a warp's row tile) are issued interleaved so that the
//     SMALL/BIG dependency chains of the pinned dot3 order (20.7 cycles per mma) overlap.
#pragma once

#include <cstdio>

#include "cfbi/cfbi_async.cuh"
#include "cfbi/cfbi_common.cuh"

namespace cfbi {

constexpr int K2_SE = 132;                              // fp32 E row stride (conflict-free scattered updates)
constexpr int K2_AQ_FLOATS = 12 * 128 + 4 * 16 * 17;     // AQF + AQD
constexpr int K2_KD_FLOATS = 8192 + 128;                 // KD fragments + GEND

// Tiling and shared-memory layout of a K2 CTA that owns BV value columns (BV = 8: more CTAs for a single short
// sequence, BV = 16: fewer redundant operand copies for large batches).
template <int BV>
struct K2Cfg {
    static constexpr int NT = BV / 8;                    // 8-column MMA tiles per CTA
    static constexpr int NSLICE = HEAD_V / BV;           // CTAs per (sequence, head)
    static constexpr int SU = BV + 8;                    // fp32 U row stride (conflict-free float2 stores from the C fragments)
    static constexpr int UT_FLOATS = 4 * NT * 128;       // UT slice of one chunk
    static constexpr int EHL_WORDS = NT * 16 * 32 * 2;   // one plane (hi or lo) of E B-fragments
    static constexpr int UHL_WORDS = NT * 8 * 32 * 2;    // one plane of U B-fragments
    // byte offsets inside the dynamic shared memory block
    static constexpr int OFF_W = 0;
    static constexpr int OFF_GQ = OFF_W + 8192 * 4;
    static constexpr int OFF_KD = OFF_GQ + 8192 * 4;
    static constexpr int OFF_AQ = OFF_KD + K2_KD_FLOATS * 4;
    static constexpr int OFF_UT = OFF_AQ + K2_AQ_FLOATS * 4;
    static constexpr int OFF_EHL = OFF_UT + UT_FLOATS * 4;
    static constexpr int OFF_U32 = OFF_EHL + 2 * EHL_WORDS * 4;
    static constexpr int OFF_UHL = OFF_U32 + 64 * SU * 4;
    static constexpr int OFF_BAR = OFF_UHL + 2 * UHL_WORDS * 4;
    static constexpr int SMEM_BYTES = OFF_BAR + 64;
    static_assert(OFF_AQ % 16 == 0 && OFF_UT % 16 == 0 && OFF_EHL % 16 == 0 && OFF_U32 % 16 == 0 &&
                      OFF_UHL % 16 == 0 && OFF_BAR % 8 == 0, "shared memory alignment");
};

// word index of E[v][k] (v = value column inside the slice, k = key channel) in a hi/lo plane of B-fragments
// [nt][ks 16][lane 32][2]: the B fragment of the 8x8 (k-step, n-tile) block is b0 = (k = t, n = g), b1 = (k = t+4, n = g)
__device__ __forceinline__ int efrag_word(int v, int k) {
    return ((((v >> 3) * 16 + (k >> 3)) * 32 + ((v & 7) * 4 + (k & 3))) << 1) + ((k >> 2) & 1);
}
// same for U[j][v] (j = token row 0..63 is the K index): [nt][ks 8][lane 32][2]
__device__ __forceinline__ int ufrag_word(int j, int v) {
    return ((((v >> 3) * 8 + (j >> 3)) * 32 + ((v & 7) * 4 + (j & 3))) << 1) + ((j >> 2) & 1);
}

// P2 / P1 style product: rows of one 16-row tile (A fragments of K = 128, 16 k-steps) times the E fragments,
// NT independent 16x8 tiles. Result = fadd_rn(BIG, SMALL) per element, C-fragment order.
template <int K2_NT>
__device__ __forceinline__ void k2_dot3_e(const float4* a, const uint2* eh, const uint2* el, float (&res)[K2_NT][4]) {
    float big[K2_NT][4], sml[K2_NT][4];
#pragma unroll
    for (int nt = 0; nt < K2_NT; ++nt)
#pragma unroll
        for (int i = 0; i < 4; ++i) { big[nt][i] = neg_zero(); sml[nt][i] = neg_zero(); }
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
        const float4 av = a[ks * 32];
        uint32_t ah[4], al[4];
        split_tf32(av.x, ah[0], al[0]);
        split_tf32(av.y, ah[1], al[1]);
        split_tf32(av.z, ah[2], al[2]);
        split_tf32(av.w, ah[3], al[3]);
#pragma unroll
        for (int nt = 0; nt < K2_NT; ++nt) {
            const uint2 bh = eh[(nt * 16 + ks) * 32], bl = el[(nt * 16 + ks) * 32];
            const uint32_t bhv[2] = {bh.x, bh.y}, blv[2] = {bl.x, bl.y};
            mma_tf32_m16n8k8(sml[nt], al, bhv);
            mma_tf32_m16n8k8(sml[nt], ah, blv);
            mma_tf32_m16n8k8(big[nt], ah, bhv);
        }
    }
#pragma unroll
    for (int nt = 0; nt < K2_NT; ++nt)
#pragma unroll
        for (int i = 0; i < 4; ++i) res[nt][i] = fadd(big[nt][i], sml[nt][i]);
}

// cross-sub-block part of O: M = dot3(Aqk[16s.., 0:16s], U[0:16s]) for KS = 2s k-steps, A fragments `a`
// (the sub-block's AQF fragments), B fragments from the U hi/lo planes.
template <int KS, int K2_NT>
__device__ __forceinline__ void k2_cross_block(const float4* a, const uint2* uh, const uint2* ul, float (&res)[K2_NT][4]) {
    float big[K2_NT][4], sml[K2_NT][4];
#pragma unroll
    for (int nt = 0; nt < K2_NT; ++nt)
#pragma unroll
        for (int i = 0; i < 4; ++i) { big[nt][i] = neg_zero(); sml[nt][i] = neg_zero(); }
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
        const float4 av = a[ks * 32];
        uint32_t ah[4], al[4];
        split_tf32(av.x, ah[0], al[0]);
        split_tf32(av.y, ah[1], al[1]);
        split_tf32(av.z, ah[2], al[2]);
        split_tf32(av.w, ah[3], al[3]);
#pragma unroll
        for (int nt = 0; nt < K2_NT; ++nt) {
            const uint2 bh = uh[(nt * 8 + ks) * 32], bl = ul[(nt * 8 + ks) * 32];
            const uint32_t bhv[2] = {bh.x, bh.y}, blv[2] = {bl.x, bl.y};
            mma_tf32_m16n8k8(sml[nt], al, bhv);
            mma_tf32_m16n8k8(sml[nt], ah, blv);
            mma_tf32_m16n8k8(big[nt], ah, bhv);
        }
    }
#pragma unroll
    for (int nt = 0; nt < K2_NT; ++nt)
#pragma unroll
        for (int i = 0; i < 4; ++i) res[nt][i] = fadd(big[nt][i], sml[nt][i]);
}

template <int BV>
__global__ void __launch_bounds__(NTHREADS, 1) k2_kernel(const Params p) {
    using C = K2Cfg<BV>;
    constexpr int K2_BV = BV, K2_NT = C::NT, K2_SU = C::SU, K2_UT_FLOATS = C::UT_FLOATS;
    constexpr int K2_EHL_WORDS = C::EHL_WORDS, K2_UHL_WORDS = C::UHL_WORDS;
    constexpr int K2_OFF_W = C::OFF_W, K2_OFF_GQ = C::OFF_GQ, K2_OFF_KD = C::OFF_KD, K2_OFF_AQ = C::OFF_AQ, K2_OFF_UT = C::OFF_UT;
    constexpr int K2_OFF_EHL = C::OFF_EHL, K2_OFF_U32 = C::OFF_U32, K2_OFF_UHL = C::OFF_UHL, K2_OFF_BAR = C::OFF_BAR;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    float* sW = reinterpret_cast<float*>(smem_raw + K2_OFF_W);
    float* sGQ = reinterpret_cast<float*>(smem_raw + K2_OFF_GQ);
    float* sKD = reinterpret_cast<float*>(smem_raw + K2_OFF_KD);   // KD fragments, then GEND[128]
    float* sAQ = reinterpret_cast<float*>(smem_raw + K2_OFF_AQ);   // AQF [12][32][4], then AQD [4][16][17]
    float* sUT = reinterpret_cast<float*>(smem_raw + K2_OFF_UT);
    uint32_t* sEhi = reinterpret_cast<uint32_t*>(smem_raw + K2_OFF_EHL);
    uint32_t* sElo = sEhi + K2_EHL_WORDS;
    float* sU = reinterpret_cast<float*>(smem_raw + K2_OFF_U32);
    uint32_t* sUhi = reinterpret_cast<uint32_t*>(smem_raw + K2_OFF_UHL);
    uint32_t* sUlo = sUhi + K2_UHL_WORDS;
    uint64_t* bar = reinterpret_cast<uint64_t*>(smem_raw + K2_OFF_BAR);   // 0 W, 1 GQ, 2 UT, 3 AQ, 4 KD

    const int vs = blockIdx.x, n = blockIdx.y, h = blockIdx.z, H = p.H;
    const SeqPlan sp = p.plan[n];
    const int nch = sp.nfull + (sp.tail > 0 ? 1 : 0);
    if (nch == 0) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int v0 = vs * K2_BV;

    if (tid == 0) {
#pragma unroll
        for (int i = 0; i < 5; ++i) mbar_init(&bar[i], 1);
        mbar_init_fence();
    }
    __syncthreads();

    const float* ws_base = p.ws + ((int64_t)sp.cbase * H + h) * WS2_FLOATS;   // chunk lc: + lc * H * WS2_FLOATS
    auto ws_chunk = [&](int lc) { return ws_base + (int64_t)lc * H * WS2_FLOATS; };
    // copies of the operands that P1 / P2 / U consume (W, GQ, UT slice) and of those that O / P3 consume (Aqk, KD)
    auto issue_front = [&](int lc) {
        const float* wsb = ws_chunk(lc);
        fence_proxy_async_smem();
        mbar_arrive_expect_tx(&bar[0], 8192 * 4);
        bulk_g2s(sW, wsb + WS2_W, 8192 * 4, &bar[0]);
        mbar_arrive_expect_tx(&bar[1], 8192 * 4);
        bulk_g2s(sGQ, wsb + WS2_GQ, 8192 * 4, &bar[1]);
        mbar_arrive_expect_tx(&bar[2], K2_UT_FLOATS * 4);
        bulk_g2s(sUT, wsb + WS2_UT + vs * K2_UT_FLOATS, K2_UT_FLOATS * 4, &bar[2]);
    };
    auto issue_back = [&](int lc) {
        const float* wsb = ws_chunk(lc);
        fence_proxy_async_smem();
        mbar_arrive_expect_tx(&bar[3], K2_AQ_FLOATS * 4);
        bulk_g2s(sAQ, wsb + WS2_AQF, K2_AQ_FLOATS * 4, &bar[3]);
        if (lc < sp.nfull) {
            mbar_arrive_expect_tx(&bar[4], K2_KD_FLOATS * 4);
            bulk_g2s(sKD, wsb + WS2_KD, K2_KD_FLOATS * 4, &bar[4]);
        }
    };
    if (tid == 0) {
        issue_front(0);
        issue_back(0);
    }

    // E slice [K2_BV][128]: pre-split hi/lo B fragments in shared memory; the fp32 values live in the registers of the
    // warps that update them (a thread owns the same E elements for the whole scan)
    float* Eg = p.E != nullptr ? p.E + ((int64_t)sp.slot * H + h) * E_FLOATS + (int64_t)v0 * HEAD_K : nullptr;
    for (int idx = tid; idx < K2_BV * HEAD_K; idx += NTHREADS) {
        const int v = idx >> 7, k = idx & 127;
        const float e = Eg != nullptr ? Eg[v * HEAD_K + k] : 0.f;
        uint32_t hi, lo;
        split_tf32(e, hi, lo);
        sEhi[efrag_word(v, k)] = hi;
        sElo[efrag_word(v, k)] = lo;
    }
    __syncthreads();

    const bool isA = warp < 4;
    const int w = warp & 3;
    float ereg[2][K2_NT][4];   // fp32 E elements of this thread (A warps): key tiles w, w + 4, value tiles 0..NT-1
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int nt = 0; nt < K2_NT; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int k = 16 * (w + 4 * mi) + g + 8 * (i >> 1), v = 8 * nt + 2 * t + (i & 1);
                ereg[mi][nt][i] = (isA && Eg != nullptr) ? Eg[v * HEAD_K + k] : 0.f;
            }   // row tile (A: P1 / U, B: P2 / sub-block of O); P3 key tiles are w and w + 4
#ifdef CFBI_K2_PROFILE
    long long acc_wait = 0, acc_p12 = 0, acc_s1 = 0, acc_p3o = 0, acc_s2 = 0, tt;
#define K2_TS() (threadIdx.x == 0 ? clock64() : 0)
#else
#define K2_TS() 0
#endif

    for (int lc = 0; lc < nch; ++lc) {
        const bool full = lc < sp.nfull;
        const int valid = full ? BT : sp.tail;
        const int nsb = (valid + SBR - 1) / SBR;
        const int64_t tok0 = (int64_t)sp.tok0 + sp.hrows + (int64_t)BT * lc;
        const uint32_t par = lc & 1;
        float p2[K2_NT][4];
#ifdef CFBI_K2_PROFILE
        tt = K2_TS();
#endif

        // ---- phase 1 ---------------------------------------------------------------------------------
        if (isA) {
            if (w < nsb) {
                mbar_wait(&bar[0], par);
                float p1[K2_NT][4];
                k2_dot3_e<K2_NT>(reinterpret_cast<const float4*>(sW) + w * 16 * 32 + lane, reinterpret_cast<const uint2*>(sEhi) + lane,
                          reinterpret_cast<const uint2*>(sElo) + lane, p1);
                mbar_wait(&bar[2], par);   // UT slice
#pragma unroll
                for (int nt = 0; nt < K2_NT; ++nt) {
                    const float4 ut = reinterpret_cast<const float4*>(sUT)[(w * K2_NT + nt) * 32 + lane];
                    const float utv[4] = {ut.x, ut.y, ut.z, ut.w};
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float u = fsub(utv[i], p1[nt][i]);
                        const int row = 16 * w + g + 8 * (i >> 1), col = 8 * nt + 2 * t + (i & 1);
                        sU[row * K2_SU + col] = u;
                        uint32_t hi, lo;
                        split_tf32(u, hi, lo);
                        sUhi[ufrag_word(row, col)] = hi;
                        sUlo[ufrag_word(row, col)] = lo;
                        if (!full && p.rows != nullptr && row < valid)
                            p.rows[((int64_t)sp.slot * H + h) * ROWS_PER_HEAD_FLOATS + PLANE_U * ROW_PLANE_FLOATS + row * 128 + v0 + col] = u;
                    }
                }
            }
        } else if (w < nsb) {
            mbar_wait(&bar[1], par);   // GQ
            k2_dot3_e<K2_NT>(reinterpret_cast<const float4*>(sGQ) + w * 16 * 32 + lane, reinterpret_cast<const uint2*>(sEhi) + lane,
                      reinterpret_cast<const uint2*>(sElo) + lane, p2);
        }
#ifdef CFBI_K2_PROFILE
        if (tid == 0) { long long n2 = K2_TS(); acc_p12 += n2 - tt; tt = n2; }
#endif
        __syncthreads();   // S1: U visible, P1 / P2 done with E, W / GQ / UT free
        if (tid == 0 && lc + 1 < nch) issue_front(lc + 1);
#ifdef CFBI_K2_PROFILE
        if (tid == 0) { long long n2 = K2_TS(); acc_s1 += n2 - tt; tt = n2; }
#endif

        // ---- phase 3 ---------------------------------------------------------------------------------
        if (isA) {
            if (full) {   // P3 = dot3(KD, U) over K = 64 for the key tiles w and w + 4, then E' = E*GEND + P3
                mbar_wait(&bar[4], par);
                float big[2][K2_NT][4], sml[2][K2_NT][4];
#pragma unroll
                for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                    for (int nt = 0; nt < K2_NT; ++nt)
#pragma unroll
                        for (int i = 0; i < 4; ++i) { big[mi][nt][i] = neg_zero(); sml[mi][nt][i] = neg_zero(); }
                const float4* kd = reinterpret_cast<const float4*>(sKD) + lane;
                const uint2* uh = reinterpret_cast<const uint2*>(sUhi) + lane;
                const uint2* ul = reinterpret_cast<const uint2*>(sUlo) + lane;
#pragma unroll
                for (int ks = 0; ks < 8; ++ks) {
                    uint32_t ah[2][4], al[2][4];
#pragma unroll
                    for (int mi = 0; mi < 2; ++mi) {
                        const float4 av = kd[((w + 4 * mi) * 8 + ks) * 32];
                        split_tf32(av.x, ah[mi][0], al[mi][0]);
                        split_tf32(av.y, ah[mi][1], al[mi][1]);
                        split_tf32(av.z, ah[mi][2], al[mi][2]);
                        split_tf32(av.w, ah[mi][3], al[mi][3]);
                    }
#pragma unroll
                    for (int nt = 0; nt < K2_NT; ++nt) {
                        const uint2 bh = uh[(nt * 8 + ks) * 32], bl = ul[(nt * 8 + ks) * 32];
                        const uint32_t bhv[2] = {bh.x, bh.y}, blv[2] = {bl.x, bl.y};
#pragma unroll
                        for (int mi = 0; mi < 2; ++mi) {
                            mma_tf32_m16n8k8(sml[mi][nt], al[mi], bhv);
                            mma_tf32_m16n8k8(sml[mi][nt], ah[mi], blv);
                            mma_tf32_m16n8k8(big[mi][nt], ah[mi], bhv);
                        }
                    }
                }
                const float* gend = sKD + 8192;
#pragma unroll
                for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                    for (int nt = 0; nt < K2_NT; ++nt)
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int k = 16 * (w + 4 * mi) + g + 8 * (i >> 1), v = 8 * nt + 2 * t + (i & 1);
                            const float en = fadd(fmul(ereg[mi][nt][i], gend[k]), fadd(big[mi][nt][i], sml[mi][nt][i]));
                            ereg[mi][nt][i] = en;
                            uint32_t hi, lo;
                            split_tf32(en, hi, lo);
                            sEhi[efrag_word(v, k)] = hi;
                            sElo[efrag_word(v, k)] = lo;
                        }
            }
        } else if (w < nsb) {   // O for sub-block s = w
            const int s = w;
            mbar_wait(&bar[3], par);
            float o[K2_NT][4];
            if (s == 0) {
#pragma unroll
                for (int nt = 0; nt < K2_NT; ++nt)
#pragma unroll
                    for (int i = 0; i < 4; ++i) o[nt][i] = p2[nt][i];
            } else {
                float m[K2_NT][4];
                const float4* a = reinterpret_cast<const float4*>(sAQ) + s * (s - 1) * 32 + lane;
                const uint2* uh = reinterpret_cast<const uint2*>(sUhi) + lane;
                const uint2* ul = reinterpret_cast<const uint2*>(sUlo) + lane;
                if (s == 1) k2_cross_block<2, K2_NT>(a, uh, ul, m);
                else if (s == 2) k2_cross_block<4, K2_NT>(a, uh, ul, m);
                else k2_cross_block<6, K2_NT>(a, uh, ul, m);
#pragma unroll
                for (int nt = 0; nt < K2_NT; ++nt)
#pragma unroll
                    for (int i = 0; i < 4; ++i) o[nt][i] = fadd(p2[nt][i], m[nt][i]);
            }
            const float* aqd = sAQ + 12 * 128 + s * 272;
#pragma unroll
            for (int jj = 0; jj < 16; ++jj) {   // O[t] = fma(Aqk[t][j], U[j], O[t]) for j = 16s .. t, ascending
                float uj[K2_NT][2];
#pragma unroll
                for (int nt = 0; nt < K2_NT; ++nt) {
                    const float2 u2 = *reinterpret_cast<const float2*>(&sU[(16 * s + jj) * K2_SU + 8 * nt + 2 * t]);
                    uj[nt][0] = u2.x;
                    uj[nt][1] = u2.y;
                }
#pragma unroll
                for (int hi = 0; hi < 2; ++hi) {
                    const int rr = g + 8 * hi;
                    const float a = aqd[rr * 17 + jj];
#pragma unroll
                    for (int nt = 0; nt < K2_NT; ++nt)
#pragma unroll
                        for (int e = 0; e < 2; ++e) {
                            const float nv = ffma(a, uj[nt][e], o[nt][2 * hi + e]);
                            o[nt][2 * hi + e] = (jj <= rr) ? nv : o[nt][2 * hi + e];
                        }
                }
            }
#pragma unroll
            for (int hi = 0; hi < 2; ++hi) {
                const int row = 16 * s + g + 8 * hi;
                if (row < valid) {
#pragma unroll
                    for (int nt = 0; nt < K2_NT; ++nt) {
                        const int64_t oi = ((tok0 + row) * H + h) * HEAD_V + v0 + 8 * nt + 2 * t;
                        *reinterpret_cast<__nv_bfloat162*>(&p.out[oi]) = __floats2bfloat162_rn(o[nt][2 * hi], o[nt][2 * hi + 1]);
                        if (p.out_f32 != nullptr)
                            *reinterpret_cast<float2*>(&p.out_f32[oi]) = make_float2(o[nt][2 * hi], o[nt][2 * hi + 1]);
                    }
                }
            }
        }
#ifdef CFBI_K2_PROFILE
        if (tid == 0) { long long n2 = K2_TS(); acc_p3o += n2 - tt; tt = n2; }
#endif
        __syncthreads();   // S2: E' visible, O done, KD / Aqk free
        if (tid == 0 && lc + 1 < nch) issue_back(lc + 1);
#ifdef CFBI_K2_PROFILE
        if (tid == 0) { long long n2 = K2_TS(); acc_s2 += n2 - tt; tt = n2; }
#endif
    }

    if (Eg != nullptr && isA) {
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int nt = 0; nt < K2_NT; ++nt)
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int k = 16 * (w + 4 * mi) + g + 8 * (i >> 1), v = 8 * nt + 2 * t + (i & 1);
                    Eg[v * HEAD_K + k] = ereg[mi][nt][i];
                }
    }
#ifdef CFBI_K2_PROFILE
    if (tid == 0 && vs == 0 && n == 0 && h == 0 && nch >= 8)
        printf("K2prof chunks=%d: per chunk cycles: phase1 (wait+P1/P2+U) %lld | S1 %lld | phase3 (P3/E-update | O) %lld | S2 %lld | total %lld\n", nch,
               acc_p12 / nch, acc_s1 / nch, acc_p3o / nch, acc_s2 / nch, (acc_p12 + acc_s1 + acc_p3o + acc_s2) / nch);
#endif
}

}  // namespace cfbi
