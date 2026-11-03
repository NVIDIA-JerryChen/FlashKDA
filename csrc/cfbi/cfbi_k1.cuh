// CF-BI KDA K1 (optimised): everything of a chunk that does not depend on the entry state E.
//
// Same arithmetic as csrc/cfbi/ref/ref_k1.cuh (tests/cfbi/test_vs_ref.py compares bits). One CTA per (chunk, head):
//   0. cp.async the raw bf16 q / k / v / g rows of the chunk into shared memory (all 4096 16-byte pieces issued
//      up front, every warp request fully coalesced) - v1 spent 45 % of its time on exposed global-load latency;
//   1. prep: l2norm inverse of every k / q row (warps 4-7), gate a2 + sequential cumsum G2 and beta (warps 0-3).
//      khat / qhat are NOT stored: they are recomputed as fmul(x, inv) [* scale] wherever they are used, which is
//      the same value bit for bit (a pure function of the same two operands);
//   2. Akk / Aqk per sub-block: warps 0-3 run the MMA jobs (two 8-column tiles per warp, k and q products
//      sharing the B fragments) on A operands (ks, qs pre-split into TF32 hi/lo) that warps 4-7 build into a
//      second staging buffer one sub-block ahead; in the shadow of this phase warps 4-7 also compute and write the
//      parts of the K2 workspace that only depend on prep (GQ, KD, GEND);
//   3. block forward substitution for UT and W: the cross-sub-block dot3 for 4 column tiles per warp shares the
//      A fragments, the in-sub-block FMA chain gets its right-hand sides from shared memory (v1 loaded v from
//      global inside the dependent chain);
//   4. the rest of the workspace (W, UT, Aqk) in fragment-native layout (cfbi_common.cuh); a trailing partial chunk
//      also stores khat, G2, UT, W of its rows in the open-chunk row cache.
//
// smem: raw k/q/v/g [64][136] bf16 (the g buffer is reused for M), G2 [64][132], UT/W [64][136] (the pair is reused
// as two Akk/Aqk A-operand staging buffers), Akk/Aqk [64][68], invK/invQ/beta/beta-logit [64].
#pragma once

#include <cstdio>

#include "cfbi/cfbi_async.cuh"
#include "cfbi/cfbi_common.cuh"

namespace cfbi {

constexpr int K1_RS = 136;   // raw bf16 row stride (elements, 272 B): conflict-free fragment-style element reads
constexpr int K1_SG = 132;   // G2 row stride
constexpr int K1_SX = 136;   // UT / W row stride
constexpr int K1_SA = 68;    // Akk / Aqk row stride
constexpr int K1_SM = 132;   // M row stride

constexpr int K1_OFF_KR = 0;
constexpr int K1_OFF_QR = K1_OFF_KR + 64 * K1_RS * 2;
constexpr int K1_OFF_VR = K1_OFF_QR + 64 * K1_RS * 2;
constexpr int K1_OFF_GR = K1_OFF_VR + 64 * K1_RS * 2;   // raw g; dead after the gate phase, then used for M
constexpr int K1_OFF_G2 = K1_OFF_GR + 64 * K1_RS * 2;
constexpr int K1_OFF_UT = K1_OFF_G2 + 64 * K1_SG * 4;
constexpr int K1_OFF_W = K1_OFF_UT + 64 * K1_SX * 4;
constexpr int K1_OFF_AKK = K1_OFF_W + 64 * K1_SX * 4;
constexpr int K1_OFF_AQK = K1_OFF_AKK + 64 * K1_SA * 4;
constexpr int K1_OFF_SC = K1_OFF_AQK + 64 * K1_SA * 4;   // invK[64] invQ[64] beta[64] betaLogit[64]
constexpr size_t K1_SMEM_BYTES = K1_OFF_SC + 4 * 64 * 4;
constexpr int K1_STAGE_UINT4 = 2 * 16 * 2 * 32;          // one A-operand staging buffer: [which 2][ks 16][hi,lo][lane 32] uint4
static_assert(2 * K1_STAGE_UINT4 * 16 <= 2 * 64 * K1_SX * 4, "two A-operand staging buffers must fit the UT/W area");
static_assert(2 * 16 * K1_SM * 4 <= 64 * K1_RS * 2, "M must fit the raw-g area");
static_assert(K1_SMEM_BYTES <= 232448, "K1 shared memory");
static_assert(K1_OFF_AQK == K1_OFF_AKK + 64 * K1_SA * 4, "Akk and Aqk are zeroed as one array");

#ifdef CFBI_K1_PROFILE
#define K1_TS(i) do { if (threadIdx.x == 0) tp[i] = clock64(); } while (0)
#else
#define K1_TS(i) do {} while (0)
#endif

__device__ __forceinline__ float bf16raw_to_f32(const uint16_t* p) { return __uint_as_float(static_cast<uint32_t>(*p) << 16); }

// cross-sub-block products M = dot3(Akk[b.., 0:b], X[0:b]) for the four column tiles of this warp
// (which = UT | W, nt = warp, warp + 8): the A fragments are shared by the four tiles.
template <int KS>
__device__ __forceinline__ void k1_m_phase(const float* sAkk, const float* sUT, const float* sW, float* sM, int b, int warp,
                                           const LaneIdx& L) {
    float big[4][4], sml[4][4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int i = 0; i < 4; ++i) { big[q][i] = neg_zero(); sml[q][i] = neg_zero(); }
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
        uint32_t ah[4], al[4];
#pragma unroll
        for (int ii = 0; ii < 4; ++ii)
            split_tf32(sAkk[(b + L.g + 8 * (ii & 1)) * K1_SA + 8 * ks + L.t + 4 * (ii >> 1)], ah[ii], al[ii]);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float* X = (q >> 1) ? sW : sUT;
            const int nt = warp + 8 * (q & 1);
            uint32_t bh[2], bl[2];
#pragma unroll
            for (int ii = 0; ii < 2; ++ii) split_tf32(X[(8 * ks + L.t + 4 * ii) * K1_SX + 8 * nt + L.g], bh[ii], bl[ii]);
            mma_tf32_m16n8k8(sml[q], al, bh);
            mma_tf32_m16n8k8(sml[q], ah, bl);
            mma_tf32_m16n8k8(big[q], ah, bh);
        }
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int which = q >> 1, nt = warp + 8 * (q & 1);
#pragma unroll
        for (int ii = 0; ii < 4; ++ii)
            sM[(which * 16 + c_row(L, ii)) * K1_SM + 8 * nt + c_col(L, ii)] = fadd(big[q][ii], sml[q][ii]);
    }
}

// Akk / Aqk of sub-block s for NTL 8-column history tiles (nt0, nt0 + 4, ...): k and q products share the B
// fragments (kjr) and the A fragments come pre-split from the staging buffer (lane already added to `stg`).
template <int NTL>
__device__ __forceinline__ void k1_akk_tiles(const uint4* stg, const uint16_t* sKr, const float* sG, const float* sInvK,
                                             const float* sBeta, float* sAkk, float* sAqk, int s, int nt0, int valid,
                                             const LaneIdx& L) {
    const int r0 = 16 * s;
    float bk[NTL][4], sk[NTL][4], bq[NTL][4], sq[NTL][4];
    float ikj[NTL];
    bool jv[NTL];
    int jj[NTL];
#pragma unroll
    for (int i = 0; i < NTL; ++i) {
        jj[i] = 8 * (nt0 + 4 * i) + L.g;
        jv[i] = jj[i] < valid;
        ikj[i] = sInvK[jj[i]];
#pragma unroll
        for (int e = 0; e < 4; ++e) { bk[i][e] = neg_zero(); sk[i][e] = neg_zero(); bq[i][e] = neg_zero(); sq[i][e] = neg_zero(); }
    }
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
        const uint4 kh = stg[((0 * 16 + ks) * 2 + 0) * 32], kl = stg[((0 * 16 + ks) * 2 + 1) * 32];
        const uint4 qh = stg[((1 * 16 + ks) * 2 + 0) * 32], ql = stg[((1 * 16 + ks) * 2 + 1) * 32];
        const uint32_t akh[4] = {kh.x, kh.y, kh.z, kh.w}, akl[4] = {kl.x, kl.y, kl.z, kl.w};
        const uint32_t aqh[4] = {qh.x, qh.y, qh.z, qh.w}, aql[4] = {ql.x, ql.y, ql.z, ql.w};
#pragma unroll
        for (int i = 0; i < NTL; ++i) {
            uint32_t bh[2], bl[2];
#pragma unroll
            for (int ii = 0; ii < 2; ++ii) {
                const int ch = 8 * ks + L.t + 4 * ii;
                const float kv = fmul(fmul(bf16raw_to_f32(sKr + jj[i] * K1_RS + ch), ikj[i]),
                                      ex2_ftz(fsub(sG[r0 * K1_SG + ch], sG[jj[i] * K1_SG + ch])));
                split_tf32(jv[i] ? kv : 0.f, bh[ii], bl[ii]);
            }
            mma_tf32_m16n8k8(sk[i], akl, bh);
            mma_tf32_m16n8k8(sq[i], aql, bh);
            mma_tf32_m16n8k8(sk[i], akh, bl);
            mma_tf32_m16n8k8(sq[i], aqh, bl);
            mma_tf32_m16n8k8(bk[i], akh, bh);
            mma_tf32_m16n8k8(bq[i], aqh, bh);
        }
    }
#pragma unroll
    for (int i = 0; i < NTL; ++i)
#pragma unroll
        for (int ii = 0; ii < 4; ++ii) {
            const int row = r0 + c_row(L, ii), col = 8 * (nt0 + 4 * i) + c_col(L, ii);
            sAkk[row * K1_SA + col] = fmul(sBeta[row], fadd(bk[i][ii], sk[i][ii]));
            sAqk[row * K1_SA + col] = fadd(bq[i][ii], sq[i][ii]);
        }
}

__global__ void __launch_bounds__(NTHREADS, 1) k1_kernel(const Params p) {
    extern __shared__ __align__(128) unsigned char smem_raw[];
    uint16_t* sKr = reinterpret_cast<uint16_t*>(smem_raw + K1_OFF_KR);
    uint16_t* sQr = reinterpret_cast<uint16_t*>(smem_raw + K1_OFF_QR);
    uint16_t* sVr = reinterpret_cast<uint16_t*>(smem_raw + K1_OFF_VR);
    uint16_t* sGr = reinterpret_cast<uint16_t*>(smem_raw + K1_OFF_GR);
    float* sM = reinterpret_cast<float*>(smem_raw + K1_OFF_GR);   // after the gate phase
    float* sG = reinterpret_cast<float*>(smem_raw + K1_OFF_G2);
    float* sUT = reinterpret_cast<float*>(smem_raw + K1_OFF_UT);
    float* sW = reinterpret_cast<float*>(smem_raw + K1_OFF_W);
    uint4* sStage0 = reinterpret_cast<uint4*>(sUT);   // A-operand staging of the Akk/Aqk phase (UT/W are written later)
    uint4* sStage1 = sStage0 + K1_STAGE_UINT4;
    float* sAkk = reinterpret_cast<float*>(smem_raw + K1_OFF_AKK);
    float* sAqk = reinterpret_cast<float*>(smem_raw + K1_OFF_AQK);
    float* sInvK = reinterpret_cast<float*>(smem_raw + K1_OFF_SC);
    float* sInvQ = sInvK + 64;
    float* sBeta = sInvQ + 64;
    float* sBetaLogit = sBeta + 64;

#ifdef CFBI_K1_PROFILE
    long long tp[8];
    K1_TS(0);
#endif
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

    // ---- 0. async copies of the raw rows (4096 16-byte pieces; a warp request covers two whole rows) ----
#pragma unroll
    for (int it = 0; it < 16; ++it) {
        const int id = it * NTHREADS + tid;
        const int which = id >> 10, row = (id >> 4) & 63, cc = id & 15;   // 4 tensors x 64 rows x 16 pieces
        const __nv_bfloat16* src = which == 0 ? p.k : (which == 1 ? p.q : (which == 2 ? p.v : p.g));
        uint16_t* dst = (which == 0 ? sKr : (which == 1 ? sQr : (which == 2 ? sVr : sGr))) + row * K1_RS + 8 * cc;
        if (row < valid) cp_async_16(dst, src + ((tok0 + row) * H + h) * HEAD_K + 8 * cc);
        else *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
    }
    cp_async_commit();
    if (tid < 64) sBetaLogit[tid] = tid < valid ? bf16_to_f32(p.beta[(tok0 + tid) * H + h]) : 0.f;   // strided 2-byte elements
    {   // Akk and Aqk are contiguous: zero both with 16-byte stores
        float4* z = reinterpret_cast<float4*>(sAkk);
        for (int i = tid; i < 2 * 64 * K1_SA / 4; i += NTHREADS) z[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    cp_async_wait_all();
    __syncthreads();
    K1_TS(1);

    // ---- 1. prep ---------------------------------------------------------------------------------------
    if (tid >= 128) {   // l2norm inverse of the 64 k rows and 64 q rows: 16-lane groups, 8 channels per lane
        const int t2 = tid - 128, grp = t2 >> 4, lj = t2 & 15;
#pragma unroll 4
        for (int r = 0; r < 16; ++r) {
            const int vec = grp + 8 * r;
            const bool isq = vec >= 64;
            const int row = isq ? vec - 64 : vec;
            const uint4 u = *reinterpret_cast<const uint4*>((isq ? sQr : sKr) + row * K1_RS + 8 * lj);
            const uint32_t wv[4] = {u.x, u.y, u.z, u.w};
            float x[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                x[2 * i] = __uint_as_float(wv[i] << 16);
                x[2 * i + 1] = __uint_as_float(wv[i] & 0xFFFF0000u);
            }
            const float inv = l2norm_inv(x);   // all lanes take part in the shuffles
            if (lj == 0) (isq ? sInvQ : sInvK)[row] = inv;
        }
    } else {   // gate a2 and the sequential cumsum over rows for channel `tid`, beta
        const float ale = gate_a_log_exp(p.A_log[h]);
        const float dtb = p.dt_bias[h * HEAD_K + tid];
        float acc = 0.f;
#pragma unroll
        for (int r8 = 0; r8 < 8; ++r8) {
            float a2[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {   // independent: the sequential dependency is only the fadd chain below
                const int row = 8 * r8 + i;
                a2[i] = 0.f;
                if (row < valid)
                    a2[i] = gate_a2(__uint_as_float(static_cast<uint32_t>(sGr[row * K1_RS + tid]) << 16), dtb, ale, p.scale2);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = 8 * r8 + i;
                acc = (row == 0) ? a2[i] : fadd(acc, a2[i]);
                sG[row * K1_SG + tid] = acc;
            }
        }
        if (tid < 64) sBeta[tid] = tid < valid ? beta_act(sBetaLogit[tid]) : 0.f;
    }
    __syncthreads();
    K1_TS(2);

    // khat / qhat of (row, channel): the same fmul(x, inv) [* scale] as the stored values of the reference
    auto khat = [&](int row, int ch) { return fmul(bf16raw_to_f32(sKr + row * K1_RS + ch), sInvK[row]); };
    auto qhat = [&](int row, int ch) { return fmul(fmul(bf16raw_to_f32(sQr + row * K1_RS + ch), sInvQ[row]), p.scale); };

    float* wsb = p.ws + ((int64_t)c * H + h) * WS2_FLOATS;
    const int rows_done = nsb * 16;

    // A operands ks / qs of sub-block s, split into TF32 hi/lo, fragment order [which][ks][hi,lo][lane]. One entry =
    // (ks, lane): the k and the q operand share the exponent factor ex2(G2[row] - G2[r0]); `ltid` / `nthr` describe the
    // helper threads that run it.
    auto stage_fn = [&](int s, uint4* stage, int ltid, int nthr) {
        const int r0 = 16 * s;
        for (int e = ltid; e < 16 * 32; e += nthr) {
            const int lane = e & 31, ks = e >> 5;
            const int g = lane >> 2, t = lane & 3;
            uint32_t hk[4], lk[4], hq[4], lq[4];
#pragma unroll
            for (int ii = 0; ii < 4; ++ii) {
                const int row = r0 + g + 8 * (ii & 1), ch = 8 * ks + t + 4 * (ii >> 1);
                const float f = ex2_ftz(fsub(sG[row * K1_SG + ch], sG[r0 * K1_SG + ch]));
                const float kv = fmul(khat(row, ch), f), qv = fmul(qhat(row, ch), f);
                split_tf32(row < valid ? kv : 0.f, hk[ii], lk[ii]);
                split_tf32(row < valid ? qv : 0.f, hq[ii], lq[ii]);
            }
            stage[((0 * 16 + ks) * 2 + 0) * 32 + lane] = make_uint4(hk[0], hk[1], hk[2], hk[3]);
            stage[((0 * 16 + ks) * 2 + 1) * 32 + lane] = make_uint4(lk[0], lk[1], lk[2], lk[3]);
            stage[((1 * 16 + ks) * 2 + 0) * 32 + lane] = make_uint4(hq[0], hq[1], hq[2], hq[3]);
            stage[((1 * 16 + ks) * 2 + 1) * 32 + lane] = make_uint4(lq[0], lq[1], lq[2], lq[3]);
        }
    };
    // workspace entries that depend on prep only: GQ A-fragments (2048 float4) and, for full chunks, KD (2048); unit u
    auto early_unit = [&](int u) {
        if (u < 2048) {   // GQ [rt][ks][lane]
            const int lane = u & 31, ks = (u >> 5) & 15, rt = u >> 9;
            const int g = lane >> 2, t = lane & 3;
            float4 val;
            float* vv = &val.x;
#pragma unroll
            for (int ii = 0; ii < 4; ++ii) {
                const int row = 16 * rt + g + 8 * (ii & 1), col = 8 * ks + t + 4 * (ii >> 1);
                vv[ii] = fmul(ex2_ftz(sG[row * K1_SG + col]), qhat(row, col));
            }
            *reinterpret_cast<float4*>(&wsb[WS2_GQ + u * 4]) = val;
        } else {          // KD [mt][ks][lane]: khat[j][k] * ex2(G2e[k] - G2[j][k])
            const int idx = u - 2048;
            const int lane = idx & 31, ks = (idx >> 5) & 7, mt = idx >> 8;
            const int g = lane >> 2, t = lane & 3;
            float4 val;
            float* vv = &val.x;
#pragma unroll
            for (int ii = 0; ii < 4; ++ii) {
                const int k = 16 * mt + g + 8 * (ii & 1), j = 8 * ks + t + 4 * (ii >> 1);
                vv[ii] = fmul(khat(j, k), ex2_ftz(fsub(sG[63 * K1_SG + k], sG[j * K1_SG + k])));
            }
            *reinterpret_cast<float4*>(&wsb[WS2_KD + idx * 4]) = val;
        }
    };
    const int early_units = full ? 4096 : 2048;

    // ---- 2. Akk / Aqk per sub-block -------------------------------------------------------------------
    stage_fn(0, sStage0, tid, NTHREADS);
    __syncthreads();
    for (int s = 0; s < nsb; ++s) {
        const uint4* cur = (s & 1) ? sStage1 : sStage0;
        uint4* nxt = (s & 1) ? sStage0 : sStage1;
        if (warp < 4) {   // MMA jobs: tiles nt = warp and warp + 4 of the 2s + 2 history tiles
            const uint4* stg = cur + L.lane;
            if (warp + 4 < 2 * s + 2) k1_akk_tiles<2>(stg, sKr, sG, sInvK, sBeta, sAkk, sAqk, s, warp, valid, L);
            else if (warp < 2 * s + 2) k1_akk_tiles<1>(stg, sKr, sG, sInvK, sBeta, sAkk, sAqk, s, warp, valid, L);
        } else {          // helpers: stage the next sub-block, then this sub-block's share of the early workspace writes
            if (s + 1 < nsb) stage_fn(s + 1, nxt, tid - 128, 128);
            const int u0 = early_units * s / nsb, u1 = early_units * (s + 1) / nsb;
            for (int u = u0 + (tid - 128); u < u1; u += 128) early_unit(u);
            if (s == nsb - 1 && full && tid < 256 && tid >= 128) wsb[WS2_GEND + (tid - 128)] = ex2_ftz(sG[63 * K1_SG + (tid - 128)]);
        }
        __syncthreads();
    }
    K1_TS(3);

    // ---- 3. block forward substitution for UT and W ---------------------------------------------------
    for (int s = 0; s < nsb; ++s) {
        const int b = 16 * s;
        if (s > 0) {
            if (s == 1) k1_m_phase<2>(sAkk, sUT, sW, sM, b, warp, L);
            else if (s == 2) k1_m_phase<4>(sAkk, sUT, sW, sM, b, warp, L);
            else k1_m_phase<6>(sAkk, sUT, sW, sM, b, warp, L);
            __syncthreads();
        }
        {   // thread = (UT | W, column): right-hand sides, subtract M, FMA chain over the sub-block's earlier rows
            const int which = tid >> 7, col = tid & 127;
            float rhs[16];
#pragma unroll
            for (int tt = 0; tt < 16; ++tt) {
                const int row = b + tt;
                if (which == 0) rhs[tt] = fmul(sBeta[row], bf16raw_to_f32(sVr + row * K1_RS + col));
                else rhs[tt] = fmul(sBeta[row], fmul(ex2_ftz(sG[row * K1_SG + col]), khat(row, col)));
            }
            float* X = which ? sW : sUT;
            float xr[16];
#pragma unroll
            for (int tt = 0; tt < 16; ++tt) {
                const int row = b + tt;
                float x = (s > 0) ? fsub(rhs[tt], sM[(which * 16 + tt) * K1_SM + col]) : rhs[tt];
                const float4* arow = reinterpret_cast<const float4*>(&sAkk[row * K1_SA + b]);   // 16-byte aligned
#pragma unroll
                for (int j4 = 0; j4 < (tt + 3) / 4; ++j4) {
                    const float4 av = arow[j4];
                    const float a4[4] = {av.x, av.y, av.z, av.w};
#pragma unroll
                    for (int jl = 0; jl < 4; ++jl)
                        if (4 * j4 + jl < tt) x = ffma(-a4[jl], xr[4 * j4 + jl], x);
                }
                xr[tt] = x;
                X[row * K1_SX + col] = x;
            }
        }
        __syncthreads();
    }
    K1_TS(4);

    // ---- 4. the rest of the workspace for K2 (fragment-native layout, see cfbi_common.cuh) -------------
    for (int idx = tid; idx < 4 * 16 * 32; idx += NTHREADS) {   // W A-fragments: [rt][ks][lane]
        const int lane = idx & 31, ks = (idx >> 5) & 15, rt = idx >> 9;
        const int g = lane >> 2, t = lane & 3;
        float4 val;
        float* vv = &val.x;
#pragma unroll
        for (int ii = 0; ii < 4; ++ii) {
            const int row = 16 * rt + g + 8 * (ii & 1), col = 8 * ks + t + 4 * (ii >> 1);
            vv[ii] = row < rows_done ? sW[row * K1_SX + col] : 0.f;
        }
        *reinterpret_cast<float4*>(&wsb[WS2_W + idx * 4]) = val;
    }
    for (int idx = tid; idx < K2_NSLICE * 4 * K2_NT * 32; idx += NTHREADS) {   // UT in P1 C-fragment order: [slice][rt][nt][lane]
        const int lane = idx & 31, nt = (idx >> 5) % K2_NT, rt = ((idx >> 5) / K2_NT) & 3, slice = idx / (32 * K2_NT * 4);
        const int g = lane >> 2, t = lane & 3;
        const int row = 16 * rt + g, col = slice * K2_BV + 8 * nt + 2 * t;
        float4 val = make_float4(0.f, 0.f, 0.f, 0.f);
        if (row < rows_done) { val.x = sUT[row * K1_SX + col]; val.y = sUT[row * K1_SX + col + 1]; }
        if (row + 8 < rows_done) { val.z = sUT[(row + 8) * K1_SX + col]; val.w = sUT[(row + 8) * K1_SX + col + 1]; }
        *reinterpret_cast<float4*>(&wsb[WS2_UT + idx * 4]) = val;
    }
    for (int idx = tid; idx < AQ_NFRAG * 32; idx += NTHREADS) {   // Aqk cross-block A-fragments: f = s*(s-1) + ks
        const int lane = idx & 31, f = idx >> 5;
        const int s = f < 2 ? 1 : (f < 6 ? 2 : 3), ks = f - s * (s - 1);
        const int g = lane >> 2, t = lane & 3;
        float4 val;
        float* vv = &val.x;
#pragma unroll
        for (int ii = 0; ii < 4; ++ii)
            vv[ii] = sAqk[(16 * s + g + 8 * (ii & 1)) * K1_SA + 8 * ks + t + 4 * (ii >> 1)];
        *reinterpret_cast<float4*>(&wsb[WS2_AQF + idx * 4]) = val;
    }
    for (int idx = tid; idx < 4 * 16 * 16; idx += NTHREADS) {   // Aqk diagonal blocks (row stride 17)
        const int s = idx >> 8, r = (idx >> 4) & 15, cc = idx & 15;
        wsb[WS2_AQD + s * 272 + r * 17 + cc] = sAqk[(16 * s + r) * K1_SA + 16 * s + cc];
    }
    if (!full && p.rows != nullptr) {
        float* R = p.rows + ((int64_t)sp.slot * H + h) * ROWS_PER_HEAD_FLOATS;
        for (int idx = tid; idx < valid * 32; idx += NTHREADS) {
            const int row = idx >> 5, c4 = idx & 31;
            float4 kv;
            kv.x = khat(row, 4 * c4 + 0);
            kv.y = khat(row, 4 * c4 + 1);
            kv.z = khat(row, 4 * c4 + 2);
            kv.w = khat(row, 4 * c4 + 3);
            *reinterpret_cast<float4*>(&R[PLANE_KHAT * ROW_PLANE_FLOATS + row * 128 + 4 * c4]) = kv;
            *reinterpret_cast<float4*>(&R[PLANE_G2 * ROW_PLANE_FLOATS + row * 128 + 4 * c4]) =
                *reinterpret_cast<const float4*>(&sG[row * K1_SG + 4 * c4]);
            *reinterpret_cast<float4*>(&R[PLANE_UT * ROW_PLANE_FLOATS + row * 128 + 4 * c4]) =
                *reinterpret_cast<const float4*>(&sUT[row * K1_SX + 4 * c4]);
            *reinterpret_cast<float4*>(&R[PLANE_W * ROW_PLANE_FLOATS + row * 128 + 4 * c4]) =
                *reinterpret_cast<const float4*>(&sW[row * K1_SX + 4 * c4]);
        }
    }
#ifdef CFBI_K1_PROFILE
    __syncthreads();
    K1_TS(5);
    if (tid == 0 && blockIdx.y == 0 && (blockIdx.x == 0 || blockIdx.x == 37 || blockIdx.x == 300))
        printf("K1prof c=%d: load %lld | prep %lld | akk/aqk %lld | subst %lld | ws-write %lld | total %lld\n", c, tp[1] - tp[0],
               tp[2] - tp[1], tp[3] - tp[2], tp[4] - tp[3], tp[5] - tp[4], tp[5] - tp[0]);
#endif
}

}  // namespace cfbi
