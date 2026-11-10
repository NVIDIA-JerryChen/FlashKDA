// CF-BI KDA K1 (optimised): everything of a chunk that does not depend on the entry state E.
//
// Same arithmetic as csrc/cfbi/ref/ref_k1.cuh (tests/cfbi/test_vs_ref.py compares bits): only the thread mapping and the schedule
// differ. Persistent kernel, one 512-thread CTA per SM (shared-memory limited), CTA b works through the (chunk, head) items
// b, b + G, b + 2G, ... (G = grid size). For one item:
//   0. the raw bf16 q / k / v / g rows of the chunk are already in shared memory: the previous item (or the prologue) issued their
//      cp.async, so the global-load latency that is exposed once per CTA of a one-CTA-per-SM kernel overlaps useful work;
//   1. prep: gate a2 of every row, l2norm inverse of every k / q row, beta, then the sequential cumsum G2. khat / qhat are NOT
//      stored: they are recomputed as fmul(x, inv) [* scale] wherever they are used, which is the same value bit for bit (a pure
//      function of the same two operands);
//   2. Akk / Aqk per sub-block: warps 0-3 run the MMA jobs (two 8-column tiles per warp, k and q products sharing the B fragments,
//      software pipelined) on A operands (ks, qs pre-split into TF32 hi/lo) that warps 4-11 build into a second staging buffer
//      one sub-block ahead; in the shadow of this phase warps 4-11 also compute and write the parts of the K2 workspace that only
//      depend on prep (GQ, KD, GEND);
//   3. block forward substitution for UT and W: the cross-sub-block dot3 for 4 column tiles per warp shares the A fragments; the
//      in-sub-block FMA chain issues all loads of a sub-block before its first store (see the comment there). One idle warp
//      decodes the next work item meanwhile;
//   4. the raw rows of the next item are fetched into the buffers that are dead by now, and the rest of the workspace (W, UT, Aqk)
//      is written in fragment-native layout (cfbi_common.cuh); a trailing partial chunk also stores khat, G2, UT, W of its rows
//      in the open-chunk row cache.
//
// smem: raw k/q/v/g [64][136] bf16 (the g buffer is reused for M), G2 [64][132], UT/W [64][136] (the pair is reused
// as two Akk/Aqk A-operand staging buffers), Akk/Aqk [64][68], invK/invQ/beta/beta-logit [64], the current / next work item.
#pragma once

#include <cstdio>

#include "cfbi/cfbi_async.cuh"
#include "cfbi/cfbi_common.cuh"

namespace cfbi {

constexpr int K1_THREADS = 512;   // 16 warps: twice the warps per scheduler hides latency (K1 is latency-bound); needs <= 128 registers per thread
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

// One (chunk, head) work item of the persistent K1.
struct K1Item {
    int c, h;      // chunk (index among all K1/K2 chunks of the call), head
    int valid;     // rows of the chunk (<= 64)
    int tok0;      // first token of the chunk on the flattened token axis
    int slot;      // state slot of the chunk's sequence (open-chunk row cache)
    int pad[3];
};
constexpr int K1_OFF_ITEM = K1_OFF_SC + 4 * 64 * 4;      // [2] K1Item: the work item being computed / the next one
constexpr size_t K1_SMEM_BYTES = K1_OFF_ITEM + 2 * sizeof(K1Item);
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

// The same cross-sub-block products for 16 warps: warp = column tile nt (0..15), the two tiles are UT and W.
template <int KS>
__device__ __forceinline__ void k1_m_phase16(const float* sAkk, const float* sUT, const float* sW, float* sM, int b, int warp,
                                             const LaneIdx& L) {
    float big[2][4], sml[2][4];
#pragma unroll
    for (int q = 0; q < 2; ++q)
#pragma unroll
        for (int i = 0; i < 4; ++i) { big[q][i] = neg_zero(); sml[q][i] = neg_zero(); }
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
        uint32_t ah[4], al[4];
#pragma unroll
        for (int ii = 0; ii < 4; ++ii)
            split_tf32(sAkk[(b + L.g + 8 * (ii & 1)) * K1_SA + 8 * ks + L.t + 4 * (ii >> 1)], ah[ii], al[ii]);
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const float* X = q ? sW : sUT;
            uint32_t bh[2], bl[2];
#pragma unroll
            for (int ii = 0; ii < 2; ++ii) split_tf32(X[(8 * ks + L.t + 4 * ii) * K1_SX + 8 * warp + L.g], bh[ii], bl[ii]);
            mma_tf32_m16n8k8(sml[q], al, bh);
            mma_tf32_m16n8k8(sml[q], ah, bl);
            mma_tf32_m16n8k8(big[q], ah, bh);
        }
    }
#pragma unroll
    for (int q = 0; q < 2; ++q)
#pragma unroll
        for (int ii = 0; ii < 4; ++ii)
            sM[(q * 16 + c_row(L, ii)) * K1_SM + 8 * warp + c_col(L, ii)] = fadd(big[q][ii], sml[q][ii]);
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
    // Software pipeline: everything a k-step needs (its four A fragments from the staging buffer and the TF32-split B
    // operands kjr built from raw k, G2 and invK) is fetched two steps ahead of the MMAs that consume it, so the shared-memory
    // and ex2 latencies of steps ks+1 and ks+2 overlap the MMAs of step ks instead of adding to them. Same operations on the
    // same operands in the same per-accumulator order: only the issue order of independent work changes.
    struct Step {
        uint4 kh, kl, qh, ql;
        uint32_t bh[NTL][2], bl[NTL][2];
    };
    auto fetch = [&](int ks) {
        Step st;
        st.kh = stg[((0 * 16 + ks) * 2 + 0) * 32];
        st.kl = stg[((0 * 16 + ks) * 2 + 1) * 32];
        st.qh = stg[((1 * 16 + ks) * 2 + 0) * 32];
        st.ql = stg[((1 * 16 + ks) * 2 + 1) * 32];
#pragma unroll
        for (int i = 0; i < NTL; ++i) {
#pragma unroll
            for (int ii = 0; ii < 2; ++ii) {
                const int ch = 8 * ks + L.t + 4 * ii;
                const float kv = fmul(fmul(bf16raw_to_f32(sKr + jj[i] * K1_RS + ch), ikj[i]),
                                      ex2_ftz(fsub(sG[r0 * K1_SG + ch], sG[jj[i] * K1_SG + ch])));
                split_tf32(jv[i] ? kv : 0.f, st.bh[i][ii], st.bl[i][ii]);
            }
        }
        return st;
    };
    Step cur = fetch(0), nxt = fetch(1);
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
        Step nn;
        if (ks + 2 < 16) nn = fetch(ks + 2);
        const uint32_t akh[4] = {cur.kh.x, cur.kh.y, cur.kh.z, cur.kh.w}, akl[4] = {cur.kl.x, cur.kl.y, cur.kl.z, cur.kl.w};
        const uint32_t aqh[4] = {cur.qh.x, cur.qh.y, cur.qh.z, cur.qh.w}, aql[4] = {cur.ql.x, cur.ql.y, cur.ql.z, cur.ql.w};
#pragma unroll
        for (int i = 0; i < NTL; ++i) {
            mma_tf32_m16n8k8(sk[i], akl, cur.bh[i]);
            mma_tf32_m16n8k8(sq[i], aql, cur.bh[i]);
            mma_tf32_m16n8k8(sk[i], akh, cur.bl[i]);
            mma_tf32_m16n8k8(sq[i], aqh, cur.bl[i]);
            mma_tf32_m16n8k8(bk[i], akh, cur.bh[i]);
            mma_tf32_m16n8k8(bq[i], aqh, cur.bh[i]);
        }
        cur = nxt;
        nxt = nn;
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

// Work item w = head * nchunk + chunk -> its chunk, head, rows and sequence (the dependent loads of the plan search are latency-bound:
// the kernel decodes the next item in an otherwise idle warp while the current one is computed).
__device__ __forceinline__ K1Item k1_decode(const Params& p, int w, int nchunk) {
    K1Item item;
    item.h = w / nchunk;
    item.c = w - item.h * nchunk;
    // sequence of this chunk: largest n with plan[n].cbase <= c
    int lo = 0, hi = p.N - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (p.plan[mid].cbase <= item.c) lo = mid; else hi = mid - 1;
    }
    const SeqPlan sp = p.plan[lo];
    const int row0 = sp.hrows + BT * (item.c - sp.cbase);   // first row of the chunk among the sequence's call rows
    item.valid = min(BT, sp.len - row0);
    item.tok0 = sp.tok0 + row0;
    item.slot = sp.slot;
    item.pad[0] = item.pad[1] = item.pad[2] = 0;
    return item;
}

template <int BV>
__global__ void __launch_bounds__(K1_THREADS, 1) k1_kernel(const Params p) {
    constexpr int K2_BV = BV, K2_NT = BV / 8, K2_NSLICE = HEAD_V / BV;   // the UT workspace is stored per K2 slice
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

    const int tid = threadIdx.x, warp = tid >> 5;
    const LaneIdx L;

    // Persistent kernel: CTA b works through the work items b, b + G, b + 2G, ... (item w = head * nchunk + chunk, chunk fastest,
    // the order of the former (blockIdx.x, blockIdx.y) grid). The raw q / k / v / g rows of the next item are fetched with cp.async
    // while the current item writes its workspace, so the global-load latency that used to be exposed at the start of every CTA
    // (one CTA per SM: nothing else to run) overlaps useful work.
    const int nchunk = p.plan[p.N].cbase;
    const int total = nchunk * p.H;
    const int G = gridDim.x;
    if (static_cast<int>(blockIdx.x) >= total) return;
    K1Item* sItem = reinterpret_cast<K1Item*>(smem_raw + K1_OFF_ITEM);   // [it & 1]: current item, [(it + 1) & 1]: next item

    // cp.async of the raw bf16 q / k / v / g rows of an item (4096 16-byte pieces, every warp request covers two whole rows);
    // which_mask bits: 1 = k, 2 = q, 4 = v, 8 = g. Rows past the end of a trailing partial chunk are zeroed.
    auto issue_raw = [&](const K1Item& item, int which_mask) {
#pragma unroll
        for (int i = 0; i < 4096 / K1_THREADS; ++i) {
            const int id = i * K1_THREADS + tid;
            const int which = id >> 10, row = (id >> 4) & 63, cc = id & 15;   // 4 tensors x 64 rows x 16 pieces
            if (!((which_mask >> which) & 1)) continue;
            const __nv_bfloat16* src = which == 0 ? p.k : (which == 1 ? p.q : (which == 2 ? p.v : p.g));
            uint16_t* dst = (which == 0 ? sKr : (which == 1 ? sQr : (which == 2 ? sVr : sGr))) + row * K1_RS + 8 * cc;
            if (row < item.valid) cp_async_16(dst, src + (((int64_t)item.tok0 + row) * p.H + item.h) * HEAD_K + 8 * cc);
            else *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
    };
    // per-item scalars, loaded ahead of their use into registers: head decay, dt_bias of this thread's channel, beta logit of row tid
    float a_log_h, dtb_c, beta_logit = 0.f;
    auto load_scalars = [&](const K1Item& item) {
        a_log_h = p.A_log[item.h];
        dtb_c = p.dt_bias[item.h * HEAD_K + (tid & 127)];
        if (tid < 64) beta_logit = tid < item.valid ? bf16_to_f32(p.beta[((int64_t)item.tok0 + tid) * p.H + item.h]) : 0.f;   // strided 2-byte elements
    };
    {   // the first item of this CTA: every thread decodes it (identical loads), thread 0 publishes it
        const K1Item item0 = k1_decode(p, blockIdx.x, nchunk);
        issue_raw(item0, 15);
        cp_async_commit();
        load_scalars(item0);
        if (tid == 0) sItem[0] = item0;
    }

    for (int w = blockIdx.x, it = 0; w < total; w += G, ++it) {
#ifdef CFBI_K1_PROFILE
        long long tp[12];
        long long t_job, t_help, t_z, t_stage0 = 0;
        K1_TS(0);
#endif
        // ---- 0. the raw rows of this item are in flight (issued by the previous iteration / the prologue) ----
        if (tid < 64) sBetaLogit[tid] = beta_logit;
        {   // Akk and Aqk are contiguous: zero both with 16-byte stores
            float4* z = reinterpret_cast<float4*>(sAkk);
            for (int i = tid; i < 2 * 64 * K1_SA / 4; i += K1_THREADS) z[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        cp_async_wait_all();
        __syncthreads();
        K1_TS(1);
        const K1Item cur = sItem[it & 1];
        const int c = cur.c, h = cur.h, valid = cur.valid, H = p.H;
        const int nsb = (valid + SBR - 1) / SBR;
        const bool full = valid == BT;

        // ---- 1. prep ---------------------------------------------------------------------------------------
        // 1a. all threads: gate a2 of one channel for half of the rows (independent), l2norm inverse of 8 of the 128 k / q
        //     rows (16-lane groups, 8 channels per lane, all 32 lanes take part in the shuffles), beta
        {
            const float ale = gate_a_log_exp(a_log_h);
            const int c = tid & 127, quad = tid >> 7;   // 512 threads: channel c, rows quad, quad + 4, ... (16 rows per thread)
            const float dtb = dtb_c;
#pragma unroll
            for (int r2 = 0; r2 < 2; ++r2) {
                float a2v[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {   // 8 independent rows in flight; unconditional work + select (no divergent branch)
                    const int row = 4 * (8 * r2 + i) + quad;
                    a2v[i] = gate_a2(__uint_as_float(static_cast<uint32_t>(sGr[row * K1_RS + c]) << 16), dtb, ale, p.scale2);
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int row = 4 * (8 * r2 + i) + quad;
                    sG[row * K1_SG + c] = row < valid ? a2v[i] : 0.f;
                }
            }
            K1_TS(6);
            const int grp = tid >> 4, lj = tid & 15;
            float invv[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {   // 32 groups of 16 lanes, 4 of the 128 k / q rows each
                const int vec = grp + 32 * r;
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
                invv[r] = l2norm_inv(x);
            }
            // the stores come after all four reductions: a store between them would keep the compiler from overlapping the
            // shuffle chains of the four vectors (it cannot prove the stores do not alias the next vector's loads)
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int vec = grp + 32 * r;
                const bool isq = vec >= 64;
                const int row = isq ? vec - 64 : vec;
                if (lj == 0) (isq ? sInvQ : sInvK)[row] = invv[r];
            }
            K1_TS(7);
            if (tid < 64) sBeta[tid] = tid < valid ? beta_act(sBetaLogit[tid]) : 0.f;
        }
        __syncthreads();
        K1_TS(8);
        // 1b. sequential cumsum over the rows for every channel, in place: G2[0] = a2[0], G2[t] = G2[t-1] + a2[t]
        if (tid < 128) {
            float acc = 0.f;
#pragma unroll
            for (int r8 = 0; r8 < 8; ++r8) {
                float a[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) a[i] = sG[(8 * r8 + i) * K1_SG + tid];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc = (r8 == 0 && i == 0) ? a[i] : fadd(acc, a[i]);
                    sG[(8 * r8 + i) * K1_SG + tid] = acc;
                }
            }
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
#ifdef CFBI_K1_PROFILE
        t_job = 0; t_help = 0;
        if (tid == 0) t_z = clock64();
#endif
        stage_fn(0, sStage0, tid, K1_THREADS);
        __syncthreads();
#ifdef CFBI_K1_PROFILE
        if (tid == 0) t_stage0 = clock64() - t_z;
#endif
        for (int s = 0; s < nsb; ++s) {
            const uint4* cur = (s & 1) ? sStage1 : sStage0;
            uint4* nxt = (s & 1) ? sStage0 : sStage1;
#ifdef CFBI_K1_PROFILE
            if (tid == 0 || tid == 128) t_z = clock64();
#endif
            if (warp < 4) {   // MMA jobs: tiles nt = warp and warp + 4 of the 2s + 2 history tiles. One tile warp per scheduler: every tile
                              // warp re-reads the whole 32 KiB A operand, so more tile warps only add shared-memory traffic
                const uint4* stg = cur + L.lane;
                if (warp + 4 < 2 * s + 2) k1_akk_tiles<2>(stg, sKr, sG, sInvK, sBeta, sAkk, sAqk, s, warp, valid, L);
                else if (warp < 2 * s + 2) k1_akk_tiles<1>(stg, sKr, sG, sInvK, sBeta, sAkk, sAqk, s, warp, valid, L);
#ifdef CFBI_K1_PROFILE
                if (tid == 0) t_job += clock64() - t_z;
#endif
            } else if (warp < 4 + 8) {   // helpers (warps 4-11): stage the next sub-block, then this sub-block's share of the early workspace writes
                const int htid = tid - 128, hn = 32 * 8;
                if (s + 1 < nsb) stage_fn(s + 1, nxt, htid, hn);
                const int u0 = early_units * s / nsb, u1 = early_units * (s + 1) / nsb;
                for (int u = u0 + htid; u < u1; u += hn) early_unit(u);
                if (s == nsb - 1 && full && htid < 128) wsb[WS2_GEND + htid] = ex2_ftz(sG[63 * K1_SG + htid]);
#ifdef CFBI_K1_PROFILE
                if (tid == 128) t_help += clock64() - t_z;
#endif
            }
            __syncthreads();
        }
        K1_TS(3);

        // ---- 3. block forward substitution for UT and W ---------------------------------------------------
#ifdef CFBI_K1_PROFILE
        long long t_m = 0, t_ch = 0, t_y;
#endif
        for (int s = 0; s < nsb; ++s) {
            const int b = 16 * s;
#ifdef CFBI_K1_PROFILE
            if (tid == 0) t_y = clock64();
#endif
            if (s > 0) {
                if (warp < 8) {   // 8 warps x 4 column tiles share each A fragment (16 warps would split Akk twice as often)
                    if (s == 1) k1_m_phase<2>(sAkk, sUT, sW, sM, b, warp, L);
                    else if (s == 2) k1_m_phase<4>(sAkk, sUT, sW, sM, b, warp, L);
                    else k1_m_phase<6>(sAkk, sUT, sW, sM, b, warp, L);
                }
                __syncthreads();
            }
#ifdef CFBI_K1_PROFILE
            if (tid == 0) { long long tn = clock64(); t_m += tn - t_y; t_y = tn; }
#endif
            if (tid < 256) {   // threads 0..255 (the other warps wait at the barrier): thread = (UT | W, column): right-hand sides, subtract M, FMA chain over the sub-block's earlier rows
                const int which = tid >> 7, col = tid & 127;
                // xr[tt] starts as the right-hand side of row b + tt minus its cross-sub-block product M, and ends as the solution.
                // Nothing is stored to shared memory until all 16 rows are solved: the compiler cannot prove such stores do not
                // alias the later rows' loads (rhs operands, M, Akk row), so it kept every row's loads behind the previous row's
                // store and the whole sub-block ran as one long chain of load latency + dependent FMAs. `which` is warp-uniform:
                // the branch sits outside the row loop so the 16 rows' loads are independent and are issued together.
                float xr[16];
                if (which == 0) {
#pragma unroll
                    for (int tt = 0; tt < 16; ++tt)
                        xr[tt] = fmul(sBeta[b + tt], bf16raw_to_f32(sVr + (b + tt) * K1_RS + col));
                } else {
#pragma unroll
                    for (int tt = 0; tt < 16; ++tt)
                        xr[tt] = fmul(sBeta[b + tt], fmul(ex2_ftz(sG[(b + tt) * K1_SG + col]), khat(b + tt, col)));
                }
                if (s > 0) {
#pragma unroll
                    for (int tt = 0; tt < 16; ++tt) xr[tt] = fsub(xr[tt], sM[(which * 16 + tt) * K1_SM + col]);
                }
#pragma unroll
                for (int tt = 1; tt < 16; ++tt) {   // row 0 has no earlier row in its sub-block
                    const float4* arow = reinterpret_cast<const float4*>(&sAkk[(b + tt) * K1_SA + b]);   // 16-byte aligned
                    float x = xr[tt];
#pragma unroll
                    for (int j4 = 0; j4 < (tt + 3) / 4; ++j4) {
                        const float4 av = arow[j4];
                        const float a4[4] = {av.x, av.y, av.z, av.w};
#pragma unroll
                        for (int jl = 0; jl < 4; ++jl)
                            if (4 * j4 + jl < tt) x = ffma(-a4[jl], xr[4 * j4 + jl], x);
                    }
                    xr[tt] = x;
                }
                float* X = which ? sW : sUT;
#pragma unroll
                for (int tt = 0; tt < 16; ++tt) X[(b + tt) * K1_SX + col] = xr[tt];
            } else if (s == 0 && tid == K1_THREADS - 1 && w + G < total) {
                // warps 8-15 have nothing to do here: decode the next work item meanwhile (a chain of dependent plan loads)
                sItem[(it + 1) & 1] = k1_decode(p, w + G, nchunk);
            }
            __syncthreads();
#ifdef CFBI_K1_PROFILE
            if (tid == 0) t_ch += clock64() - t_y;
#endif
        }
        K1_TS(4);

        // ---- 3b. next work item: its raw rows go into the buffers that are dead from here on (k / q / v / g are not read again,
        //          except k by the open-chunk row cache of a trailing partial chunk below, which is fetched after it) ----
        const bool has_next = w + G < total;
        const bool keep_k = !full && p.rows != nullptr;
        K1Item nx;
        if (has_next) {
            nx = sItem[(it + 1) & 1];   // written by the decode in phase 3 (published by its barriers)
            issue_raw(nx, keep_k ? 14 : 15);
            cp_async_commit();
            load_scalars(nx);
        }

        // ---- 4. the rest of the workspace for K2 (fragment-native layout, see cfbi_common.cuh) -------------
        for (int idx = tid; idx < 4 * 16 * 32; idx += K1_THREADS) {   // W A-fragments: [rt][ks][lane]
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
        for (int idx = tid; idx < K2_NSLICE * 4 * K2_NT * 32; idx += K1_THREADS) {   // UT in P1 C-fragment order: [slice][rt][nt][lane]
            const int lane = idx & 31, nt = (idx >> 5) % K2_NT, rt = ((idx >> 5) / K2_NT) & 3, slice = idx / (32 * K2_NT * 4);
            const int g = lane >> 2, t = lane & 3;
            const int row = 16 * rt + g, col = slice * K2_BV + 8 * nt + 2 * t;
            float4 val = make_float4(0.f, 0.f, 0.f, 0.f);
            if (row < rows_done) { val.x = sUT[row * K1_SX + col]; val.y = sUT[row * K1_SX + col + 1]; }
            if (row + 8 < rows_done) { val.z = sUT[(row + 8) * K1_SX + col]; val.w = sUT[(row + 8) * K1_SX + col + 1]; }
            *reinterpret_cast<float4*>(&wsb[WS2_UT + idx * 4]) = val;
        }
        for (int idx = tid; idx < AQ_NFRAG * 32; idx += K1_THREADS) {   // Aqk cross-block A-fragments: f = s*(s-1) + ks
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
        for (int idx = tid; idx < 4 * 16 * 16; idx += K1_THREADS) {   // Aqk diagonal blocks (row stride 17)
            const int s = idx >> 8, r = (idx >> 4) & 15, cc = idx & 15;
            wsb[WS2_AQD + s * 272 + r * 17 + cc] = sAqk[(16 * s + r) * K1_SA + 16 * s + cc];
        }
        if (!full && p.rows != nullptr) {
            float* R = p.rows + ((int64_t)cur.slot * H + h) * ROWS_PER_HEAD_FLOATS;
            for (int idx = tid; idx < valid * 32; idx += K1_THREADS) {
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
        // end of the item: all reads of this item's shared memory are done before the next iteration overwrites it; a partial chunk
        // that kept its k rows for the row cache fetches the next item's k only now
        __syncthreads();
        if (has_next && keep_k) {
            issue_raw(nx, 1);
            cp_async_commit();
        }
#ifdef CFBI_K1_PROFILE
        K1_TS(5);
        if (tid == 0 && h == 0 && (c == 0 || c == 37 || c == 300))
            printf("K1prof c=%d: load %lld | prep %lld (a2 %lld, l2norm %lld, barrier %lld, cumsum %lld) | akk/aqk %lld (stage0 %lld, jobs %lld, helpers %lld) | subst %lld (M %lld, chain %lld) | ws-write %lld | total %lld\n", c, tp[1] - tp[0],
                   tp[2] - tp[1], tp[6] - tp[1], tp[7] - tp[6], tp[8] - tp[7], tp[2] - tp[8], tp[3] - tp[2], t_stage0, t_job, t_help, tp[4] - tp[3], t_m, t_ch, tp[5] - tp[4], tp[5] - tp[0]);
#endif
    }   // work item loop
}

}  // namespace cfbi
