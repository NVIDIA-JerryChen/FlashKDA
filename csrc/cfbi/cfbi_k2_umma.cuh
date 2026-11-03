// CF-BI KDA K2 on the Blackwell tensor core path (tcgen05 / UMMA): the big GEMMs of the chunk scan.
//
// Same arithmetic as csrc/cfbi/cfbi_k2.cuh and csrc/cfbi/ref/ref_k2.cuh (tests/cfbi/test_umma_k2.py compares bits):
// every tensor-core product is the pinned three-term TF32 dot3, issued as tcgen05.mma K = 8 per instruction in the
// order SMALL += lo*hi; SMALL += hi*lo; BIG += hi*hi (accumulate = 0 on the first K-step; benchmarks/micro/umma_probe.cu
// established that this is bit-identical to the mma.sync dot3, for M in {64,128}, N in 8..128, K <= 128).
//
// One CTA per (V slice of KU_BV = 64 columns, sequence, head), 13 warps:
//   warp 0     MMA issuer (one thread) + TMEM allocation. TMEM: 4 x 64 columns = {BIG, SMALL} of P1/P2 and of P3.
//   warps 1-4  transform: fp32 fragment-native workspace operands (W|GQ, KD) -> TF32 hi/lo planes in the UMMA canonical
//              K-major layout, streamed through a KU_NS-stage smem ring (one stage = one K = 8 step of M = 128 rows).
//   warps 5-8  epilogue: thread = TMEM lane. P1/P2 rows: U = UT - P1 (fp32, hi/lo planes = the B operand of P3), P2;
//              P3 rows (lane = key channel k): E'[v][k] = E*GEND + P3 with the E^T row of the thread kept in registers,
//              hi/lo planes of E' = the B operand of the next chunk's P1/P2.
//   warps 9-12 O: cross-sub-block dot3 (mma.sync, unchanged arithmetic) + in-sub-block FMA chain + bf16 store.
// Per chunk:  P1/P2 = [W;GQ] (M=128) x E^T (N=64, K=128: 16 steps x 3 UMMAs)  ->  epilogue 1  ->
//             P3    = KD (M=128 keys) x U (N=64, K=64: 8 steps x 3 UMMAs)     ->  epilogue 2 (E update).
// All hand-offs are mbarriers (see the BAR_* table); every bit-relevant operation outside the UMMAs is the same explicit
// fsub / fadd / fmul / ffma as in the reference.
#pragma once

#include "cfbi/cfbi_async.cuh"
#include "cfbi/cfbi_common.cuh"
#include "cfbi/cfbi_tcgen05.cuh"

namespace cfbi {

constexpr int KU_BV = 64;                    // value columns per CTA
constexpr int KU_NT = KU_BV / 8;             // 8-column tiles
constexpr int KU_NSLICE = HEAD_V / KU_BV;    // CTAs per (sequence, head)
constexpr int KU_NS = 4;                     // A-operand ring stages
constexpr int KU_WARPS = 13;
constexpr int KU_THREADS = KU_WARPS * 32;

// ---- shared memory layout (bytes) ------------------------------------------------------------------
// B operands (no swizzle, K-major canonical layout, core matrices 8 rows x 16 B; the pitch between the 16-byte chunks of
// a row group is padded to 144 B so that the one-word-per-thread column writes of the epilogues are bank-conflict free)
constexpr int KU_EPITCH = 144;                   // LBO of the B planes
constexpr int KU_E_SBO = 32 * KU_EPITCH;         // E planes: N = 64 rows (v) x K = 128 (32 chunks): 8-row group stride
constexpr int KU_U_SBO = 16 * KU_EPITCH;         // U planes: N = 64 rows (v) x K = 64 (16 chunks)
constexpr int KU_E_PLANE = 8 * KU_E_SBO;         // 36864
constexpr int KU_U_PLANE = 8 * KU_U_SBO;         // 18432
// A operand stages: M = 128 rows x K = 8: 16 row groups, core matrices at g*320 + kc*160 (pitch chosen so that the
// transform's one-word-per-lane stores and the UMMA reads are conflict free)
constexpr int KU_A_LBO = 160, KU_A_SBO = 320;
constexpr int KU_A_PLANE = 16 * KU_A_SBO;        // 5120
constexpr int KU_A_STAGE = 2 * KU_A_PLANE;       // hi + lo
constexpr int KU_SU = 66;                        // fp32 row stride of sU / sP2 (even: float2 accesses)
constexpr int KU_AQ_FLOATS = 12 * 128 + 4 * 16 * 17;   // AQF + AQD of the workspace

constexpr int KU_OFF_EH = 0;
constexpr int KU_OFF_EL = KU_OFF_EH + KU_E_PLANE;
constexpr int KU_OFF_UH = KU_OFF_EL + KU_E_PLANE;
constexpr int KU_OFF_UL = KU_OFF_UH + KU_U_PLANE;
constexpr int KU_OFF_A = KU_OFF_UL + KU_U_PLANE;
constexpr int KU_OFF_SU = KU_OFF_A + KU_NS * KU_A_STAGE;
constexpr int KU_OFF_SP2 = KU_OFF_SU + 64 * KU_SU * 4;
constexpr int KU_OFF_AQ = KU_OFF_SP2 + 64 * KU_SU * 4;
constexpr int KU_OFF_BAR = KU_OFF_AQ + KU_AQ_FLOATS * 4;
constexpr int KU_SMEM_BYTES = KU_OFF_BAR + 256;
static_assert(KU_SMEM_BYTES <= 232448, "UMMA K2 shared memory");
static_assert(KU_OFF_UH % 16 == 0 && KU_OFF_A % 16 == 0 && KU_OFF_SU % 16 == 0 && KU_OFF_AQ % 16 == 0 && KU_OFF_BAR % 8 == 0, "alignment");

// mbarriers: BAR_FULL[s] / BAR_EMPTY[s] (A ring), then the per-chunk hand-offs
constexpr int BAR_FULL = 0;
constexpr int BAR_EMPTY = KU_NS;
constexpr int BAR_EREADY = 2 * KU_NS;       // E planes of the chunk are in smem (epilogue 2 of the previous chunk)   -> MMA
constexpr int BAR_P12 = BAR_EREADY + 1;     // P1/P2 accumulators complete (tcgen05.commit)                          -> epilogue
constexpr int BAR_P3 = BAR_EREADY + 2;      // P3 accumulators complete (tcgen05.commit)                             -> epilogue
constexpr int BAR_UREADY = BAR_EREADY + 3;  // U (fp32 + planes) and P2 of the chunk are in smem                     -> MMA, O
constexpr int BAR_ODONE = BAR_EREADY + 4;   // O is done reading U / P2 / U planes / AQ of the chunk                  -> epilogue
constexpr int BAR_AQ = BAR_EREADY + 5;      // Aqk of the chunk copied into smem (bulk copy)                         -> O
constexpr int KU_NBAR = BAR_EREADY + 6;
static_assert(KU_NBAR * 8 <= 256, "barrier area");

// tensor-memory columns (allocation: 256)
constexpr int TM_BIG12 = 0, TM_SMALL12 = 64, TM_BIG3 = 128, TM_SMALL3 = 192, TM_COLS = 256;

// word index (4-byte units) of element (row v, k) in a B plane
__device__ __forceinline__ int ku_eword(int v, int k) { return (v >> 3) * (KU_E_SBO / 4) + (k >> 2) * (KU_EPITCH / 4) + (v & 7) * 4 + (k & 3); }
__device__ __forceinline__ int ku_uword(int v, int j) { return (v >> 3) * (KU_U_SBO / 4) + (j >> 2) * (KU_EPITCH / 4) + (v & 7) * 4 + (j & 3); }

// cross-sub-block part of O: dot3(Aqk[16s.., 0:16s], U[0:16s]) for KS = 2s k-steps and NTP 8-column tiles starting at tile
// ntb. A fragments `a` come from the Aqk workspace (fp32, split here), B fragments from the U hi/lo planes.
template <int KS, int NTP>
__device__ __forceinline__ void ku_cross_block(const float4* a, const uint32_t* uh, const uint32_t* ul, int ntb, int lane, float (&res)[NTP][4]) {
    float big[NTP][4], sml[NTP][4];
#pragma unroll
    for (int nt = 0; nt < NTP; ++nt)
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
        for (int nt = 0; nt < NTP; ++nt) {
            const int wi = (ntb + nt) * (KU_U_SBO / 4) + (2 * ks) * (KU_EPITCH / 4) + lane;   // b0 = (j = 8ks + t, v = 8nt + g)
            const uint32_t bhv[2] = {uh[wi], uh[wi + KU_EPITCH / 4]};                         // b1 = (j = 8ks + t + 4)
            const uint32_t blv[2] = {ul[wi], ul[wi + KU_EPITCH / 4]};
            mma_tf32_m16n8k8(sml[nt], al, bhv);
            mma_tf32_m16n8k8(sml[nt], ah, blv);
            mma_tf32_m16n8k8(big[nt], ah, bhv);
        }
    }
#pragma unroll
    for (int nt = 0; nt < NTP; ++nt)
#pragma unroll
        for (int i = 0; i < 4; ++i) res[nt][i] = fadd(big[nt][i], sml[nt][i]);
}

__global__ void __launch_bounds__(KU_THREADS, 1) k2u_kernel(const Params p) {
#ifdef CFBI_HAVE_TCGEN05
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    __shared__ uint32_t tmem_slot;
    uint64_t* bar = reinterpret_cast<uint64_t*>(smem_raw + KU_OFF_BAR);

    const int vs = blockIdx.x, n = blockIdx.y, h = blockIdx.z, H = p.H;
    const SeqPlan sp = p.plan[n];
    const int nfull = sp.nfull, nch = nfull + (sp.tail > 0 ? 1 : 0);
    if (nch == 0) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int v0 = vs * KU_BV;

    const float* ws_base = p.ws + ((int64_t)sp.cbase * H + h) * WS2_FLOATS;   // chunk lc: + lc * H * WS2_FLOATS
    auto ws_chunk = [&](int lc) { return ws_base + (int64_t)lc * H * WS2_FLOATS; };

    float* sU = reinterpret_cast<float*>(smem_raw + KU_OFF_SU);
    float* sP2 = reinterpret_cast<float*>(smem_raw + KU_OFF_SP2);
    float* sAQ = reinterpret_cast<float*>(smem_raw + KU_OFF_AQ);
    uint32_t* ehi = reinterpret_cast<uint32_t*>(smem_raw + KU_OFF_EH);
    uint32_t* elo = reinterpret_cast<uint32_t*>(smem_raw + KU_OFF_EL);
    uint32_t* uhi = reinterpret_cast<uint32_t*>(smem_raw + KU_OFF_UH);
    uint32_t* ulo = reinterpret_cast<uint32_t*>(smem_raw + KU_OFF_UL);

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < KU_NS; ++s) {
            mbar_init(&bar[BAR_FULL + s], 128);
            mbar_init(&bar[BAR_EMPTY + s], 1);
        }
        mbar_init(&bar[BAR_EREADY], 128);
        mbar_init(&bar[BAR_P12], 1);
        mbar_init(&bar[BAR_P3], 1);
        mbar_init(&bar[BAR_UREADY], 128);
        mbar_init(&bar[BAR_ODONE], 128);
        mbar_init(&bar[BAR_AQ], 1);
        mbar_init_fence();
    }
    if (warp == 0) tmem_alloc(&tmem_slot, TM_COLS);
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const uint32_t tmem = tmem_slot;

    auto issue_aq = [&](int lc) {   // one thread: Aqk fragments + diagonal blocks of chunk lc -> sAQ
        fence_proxy_async_smem();
        mbar_arrive_expect_tx(&bar[BAR_AQ], KU_AQ_FLOATS * 4);
        bulk_g2s(sAQ, ws_chunk(lc) + WS2_AQF, KU_AQ_FLOATS * 4, &bar[BAR_AQ]);
    };
    if (tid == 0) issue_aq(0);

    if (warp == 0) {
        // =============================== MMA issuer =====================================================
        if (lane == 0) {
            const uint32_t idesc = make_idesc(128, KU_BV);
            int i = 0;   // global A-stage counter (same sequence as the transform warps)
            for (int lc = 0; lc < nch; ++lc) {
                const bool full = lc < nfull;
                mbar_wait_bounded(&bar[BAR_EREADY], lc & 1, 1);
                tc_fence_after();
                for (int ks = 0; ks < 16; ++ks, ++i) {   // P1/P2: A = [W;GQ] stage (hi, lo), B = E planes k-step ks
                    const int s = i % KU_NS;
                    mbar_wait_bounded(&bar[BAR_FULL + s], (i / KU_NS) & 1, 2);
                    tc_fence_after();
                    const uint32_t ab = smem_u32(smem_raw + KU_OFF_A + s * KU_A_STAGE);
                    const uint64_t ah = make_smem_desc(ab, KU_A_LBO, KU_A_SBO), al = make_smem_desc(ab + KU_A_PLANE, KU_A_LBO, KU_A_SBO);
                    const uint64_t bh = make_smem_desc(smem_u32(smem_raw + KU_OFF_EH) + ks * 2 * KU_EPITCH, KU_EPITCH, KU_E_SBO);
                    const uint64_t bl = make_smem_desc(smem_u32(smem_raw + KU_OFF_EL) + ks * 2 * KU_EPITCH, KU_EPITCH, KU_E_SBO);
                    const uint32_t acc0 = ks > 0 ? 1u : 0u;
#ifdef CFBI_UMMA_NEG_NOLO   // negative control (tests only): a wrong program must be caught by the bitwise tests
                    umma_tf32(tmem + TM_SMALL12, ah, bl, idesc, acc0);
#else
                    umma_tf32(tmem + TM_SMALL12, al, bh, idesc, acc0);
                    umma_tf32(tmem + TM_SMALL12, ah, bl, idesc, 1u);
#endif
                    umma_tf32(tmem + TM_BIG12, ah, bh, idesc, acc0);
                    umma_commit(&bar[BAR_EMPTY + s]);
                }
                umma_commit(&bar[BAR_P12]);
                if (full) {
                    mbar_wait_bounded(&bar[BAR_UREADY], lc & 1, 3);
                    tc_fence_after();
                    for (int ks = 0; ks < 8; ++ks, ++i) {   // P3: A = KD stage, B = U planes k-step ks
                        const int s = i % KU_NS;
                        mbar_wait_bounded(&bar[BAR_FULL + s], (i / KU_NS) & 1, 4);
                        tc_fence_after();
                        const uint32_t ab = smem_u32(smem_raw + KU_OFF_A + s * KU_A_STAGE);
                        const uint64_t ah = make_smem_desc(ab, KU_A_LBO, KU_A_SBO), al = make_smem_desc(ab + KU_A_PLANE, KU_A_LBO, KU_A_SBO);
                        const uint64_t bh = make_smem_desc(smem_u32(smem_raw + KU_OFF_UH) + ks * 2 * KU_EPITCH, KU_EPITCH, KU_U_SBO);
                        const uint64_t bl = make_smem_desc(smem_u32(smem_raw + KU_OFF_UL) + ks * 2 * KU_EPITCH, KU_EPITCH, KU_U_SBO);
                        const uint32_t acc0 = ks > 0 ? 1u : 0u;
#ifdef CFBI_UMMA_NEG_NOLO
                        umma_tf32(tmem + TM_SMALL3, ah, bl, idesc, acc0);
#else
                        umma_tf32(tmem + TM_SMALL3, al, bh, idesc, acc0);
                        umma_tf32(tmem + TM_SMALL3, ah, bl, idesc, 1u);
#endif
                        umma_tf32(tmem + TM_BIG3, ah, bh, idesc, acc0);
                        umma_commit(&bar[BAR_EMPTY + s]);
                    }
                    umma_commit(&bar[BAR_P3]);
                }
            }
        }
        __syncwarp();
    } else if (warp <= 4) {
        // =============================== transform warps ================================================
        const int w = warp - 1;
        const int total = 24 * nfull + (sp.tail > 0 ? 16 : 0);   // A stages of the whole call: per full chunk 16 (W|GQ) + 8 (KD)
        constexpr int PF = 4;
        float4 pre[PF][2];
        auto load_stage = [&](int i, float4 (&dst)[2]) {
            const int lcs = (i < 24 * nfull) ? i / 24 : nfull;
            const int j = i - 24 * lcs;
            const float* wsb = ws_chunk(lcs);
            const float* p0;
            const float* p1;
            if (j < 16) {
                p0 = wsb + WS2_W + ((w * 16 + j) * 32 + lane) * 4;
                p1 = wsb + WS2_GQ + ((w * 16 + j) * 32 + lane) * 4;
            } else {
                const int ks = j - 16;
                p0 = wsb + WS2_KD + ((w * 8 + ks) * 32 + lane) * 4;
                p1 = wsb + WS2_KD + (((w + 4) * 8 + ks) * 32 + lane) * 4;
            }
            dst[0] = __ldcg(reinterpret_cast<const float4*>(p0));
            dst[1] = __ldcg(reinterpret_cast<const float4*>(p1));
        };
#pragma unroll
        for (int q = 0; q < PF; ++q)
            if (q < total) load_stage(q, pre[q]);
        for (int i0 = 0; i0 < total; i0 += PF) {
#pragma unroll
            for (int q = 0; q < PF; ++q) {
                const int i = i0 + q;
                if (i < total) {
                    float4 f0 = pre[q][0], f1 = pre[q][1];
                    if (i + PF < total) load_stage(i + PF, pre[q]);
                    const int s = i % KU_NS;
                    mbar_wait_bounded(&bar[BAR_EMPTY + s], ((i / KU_NS) & 1) ^ 1, 10);
                    const int lcs = (i < 24 * nfull) ? i / 24 : nfull;
                    const int j = i - 24 * lcs;
                    if (j < 16 && lcs >= nfull && w >= (sp.tail + 15) / 16) {   // rows of sub-blocks past the tail: exact zeros
                        f0 = make_float4(0.f, 0.f, 0.f, 0.f);
                        f1 = f0;
                    }
                    uint32_t* hi = reinterpret_cast<uint32_t*>(smem_raw + KU_OFF_A + s * KU_A_STAGE);
                    uint32_t* lo = hi + KU_A_PLANE / 4;
                    // fragment (lane = 4g + t) of a 16x8 tile at row groups gi0, gi0+1: x = (g, k=t), y = (g+8, t), z = (g, t+4), w = (g+8, t+4);
                    // every store instruction writes one whole 128-byte core matrix (lane = word)
                    auto put = [&](float x, int word) {
                        uint32_t hh, ll;
                        split_tf32(x, hh, ll);
                        hi[word + lane] = hh;
                        lo[word + lane] = ll;
                    };
                    const int gA = 2 * w, gB = 8 + 2 * w;   // row groups of the first / second 16-row tile of this warp
                    put(f0.x, gA * 80);
                    put(f0.y, (gA + 1) * 80);
                    put(f0.z, gA * 80 + 40);
                    put(f0.w, (gA + 1) * 80 + 40);
                    put(f1.x, gB * 80);
                    put(f1.y, (gB + 1) * 80);
                    put(f1.z, gB * 80 + 40);
                    put(f1.w, (gB + 1) * 80 + 40);
                    fence_proxy_async_smem();
                    mbar_arrive(&bar[BAR_FULL + s]);
                }
            }
        }
    } else if (warp <= 8) {
        // =============================== epilogue warps (thread = TMEM lane) ============================
        const int q = warp & 3;
        const int r = 32 * q + lane;   // TMEM lane = row of the accumulators
        const uint32_t lane_off = (uint32_t)(q * 32) << 16;
        float* Eg = p.E != nullptr ? p.E + ((int64_t)sp.slot * H + h) * E_FLOATS + (int64_t)v0 * HEAD_K : nullptr;
        float ereg[KU_BV];   // E[v0 + v][k = r]: the thread owns row k = r of E^T for the whole scan
#pragma unroll
        for (int v = 0; v < KU_BV; ++v) {
            const float e = Eg != nullptr ? Eg[(int64_t)v * HEAD_K + r] : 0.f;
            ereg[v] = e;
            uint32_t hh, ll;
            split_tf32(e, hh, ll);
            ehi[ku_eword(v, r)] = hh;
            elo[ku_eword(v, r)] = ll;
        }
        fence_proxy_async_smem();
        mbar_arrive(&bar[BAR_EREADY]);

        const int rt = r >> 4, gg = r & 7, hb = (r >> 3) & 1;
        const int ut_row = rt * 256 + gg * 16 + 2 * hb;   // offset of UT[r][v] inside the (slice-sliced) fragment-native UT block
        for (int lc = 0; lc < nch; ++lc) {
            const bool full = lc < nfull;
            const int valid = full ? BT : sp.tail;
            const float* wsb = ws_chunk(lc);
            const float* utp = wsb + WS2_UT + vs * 4096 + ut_row;
            const float gend_k = full ? __ldcg(&wsb[WS2_GEND + r]) : 0.f;
            if (lc > 0) mbar_wait_bounded(&bar[BAR_ODONE], (lc - 1) & 1, 20);   // O of the previous chunk is done with sU / sP2 / U planes
            mbar_wait_bounded(&bar[BAR_P12], lc & 1, 21);
            tc_fence_after();
            // ---- epilogue 1: P1 rows (lane < 64): U = UT - P1; P2 rows: keep P2 -------------------------------------
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                float utv[16];
                if (r < 64) {
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const int v = 16 * c + i;
                        utv[i] = __ldcg(utp + (v >> 4) * 1024 + ((v >> 3) & 1) * 128 + ((v >> 1) & 3) * 4 + (v & 1));
                    }
                }
                uint32_t bg[16], sm[16];
                tmem_ld16(tmem + TM_BIG12 + lane_off + 16 * c, bg);
                tmem_ld16(tmem + TM_SMALL12 + lane_off + 16 * c, sm);
                tmem_wait_ld();
                if (r < 64) {
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const int v = 16 * c + i;
                        const float p1 = fadd(__uint_as_float(bg[i]), __uint_as_float(sm[i]));
                        const float u = fsub(utv[i], p1);
                        sU[r * KU_SU + v] = u;
                        uint32_t hh, ll;
                        split_tf32(u, hh, ll);
                        uhi[ku_uword(v, r)] = hh;
                        ulo[ku_uword(v, r)] = ll;
                        if (!full && p.rows != nullptr && r < valid)
                            p.rows[((int64_t)sp.slot * H + h) * ROWS_PER_HEAD_FLOATS + PLANE_U * ROW_PLANE_FLOATS + r * 128 + v0 + v] = u;
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < 16; ++i)
                        sP2[(r - 64) * KU_SU + 16 * c + i] = fadd(__uint_as_float(bg[i]), __uint_as_float(sm[i]));
                }
            }
            fence_proxy_async_smem();
            tc_fence_before();
            mbar_arrive(&bar[BAR_UREADY]);
            // ---- epilogue 2 (full chunks): E' = E*GEND + P3, new B planes ----------------------------------------------
            if (full) {
                mbar_wait_bounded(&bar[BAR_P3], lc & 1, 22);
                tc_fence_after();
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    uint32_t bg[16], sm[16];
                    tmem_ld16(tmem + TM_BIG3 + lane_off + 16 * c, bg);
                    tmem_ld16(tmem + TM_SMALL3 + lane_off + 16 * c, sm);
                    tmem_wait_ld();
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const int v = 16 * c + i;
                        const float en = fadd(fmul(ereg[v], gend_k), fadd(__uint_as_float(bg[i]), __uint_as_float(sm[i])));
                        ereg[v] = en;
                        uint32_t hh, ll;
                        split_tf32(en, hh, ll);
                        ehi[ku_eword(v, r)] = hh;
                        elo[ku_eword(v, r)] = ll;
                    }
                }
                fence_proxy_async_smem();
                tc_fence_before();
                mbar_arrive(&bar[BAR_EREADY]);
            }
        }
        if (Eg != nullptr) {
#pragma unroll
            for (int v = 0; v < KU_BV; ++v) Eg[(int64_t)v * HEAD_K + r] = ereg[v];
        }
    } else {
        // =============================== O warps: sub-block s = warp - 9 ================================
        const int s = warp - 9;
        const int g = lane >> 2, t = lane & 3;
        const int otid = tid - 9 * 32;
        for (int lc = 0; lc < nch; ++lc) {
            const bool full = lc < nfull;
            const int valid = full ? BT : sp.tail;
            const int nsb = (valid + SBR - 1) / SBR;
            const int64_t tok0 = (int64_t)sp.tok0 + sp.hrows + (int64_t)BT * lc;
            if (s < nsb) {
                mbar_wait_bounded(&bar[BAR_AQ], lc & 1, 30);
                mbar_wait_bounded(&bar[BAR_UREADY], lc & 1, 31);
                const float* aqd = sAQ + 12 * 128 + s * 272;
#pragma unroll
                for (int pass = 0; pass < 2; ++pass) {   // two passes of 4 eight-column tiles (register budget)
                    const int ntb = pass * 4;
                    float o[4][4];
#pragma unroll
                    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                        for (int i = 0; i < 4; ++i)
                            o[nt][i] = sP2[(16 * s + g + 8 * (i >> 1)) * KU_SU + 8 * (ntb + nt) + 2 * t + (i & 1)];
                    if (s > 0) {
                        float m[4][4];
                        const float4* a = reinterpret_cast<const float4*>(sAQ) + s * (s - 1) * 32 + lane;
                        if (s == 1) ku_cross_block<2, 4>(a, uhi, ulo, ntb, lane, m);
                        else if (s == 2) ku_cross_block<4, 4>(a, uhi, ulo, ntb, lane, m);
                        else ku_cross_block<6, 4>(a, uhi, ulo, ntb, lane, m);
#pragma unroll
                        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                            for (int i = 0; i < 4; ++i) o[nt][i] = fadd(o[nt][i], m[nt][i]);
                    }
#pragma unroll
                    for (int jj = 0; jj < 16; ++jj) {   // O[t] = fma(Aqk[t][j], U[j], O[t]) for j = 16s .. t, ascending
                        float uj[4][2];
#pragma unroll
                        for (int nt = 0; nt < 4; ++nt) {
                            const float2 u2 = *reinterpret_cast<const float2*>(&sU[(16 * s + jj) * KU_SU + 8 * (ntb + nt) + 2 * t]);
                            uj[nt][0] = u2.x;
                            uj[nt][1] = u2.y;
                        }
#pragma unroll
                        for (int hi = 0; hi < 2; ++hi) {
                            const int rr = g + 8 * hi;
                            const float a = aqd[rr * 17 + jj];
#pragma unroll
                            for (int nt = 0; nt < 4; ++nt)
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
                            for (int nt = 0; nt < 4; ++nt) {
                                const int64_t oi = ((tok0 + row) * H + h) * HEAD_V + v0 + 8 * (ntb + nt) + 2 * t;
                                *reinterpret_cast<__nv_bfloat162*>(&p.out[oi]) = __floats2bfloat162_rn(o[nt][2 * hi], o[nt][2 * hi + 1]);
                                if (p.out_f32 != nullptr)
                                    *reinterpret_cast<float2*>(&p.out_f32[oi]) = make_float2(o[nt][2 * hi], o[nt][2 * hi + 1]);
                            }
                        }
                    }
                }
            }
            mbar_arrive(&bar[BAR_ODONE]);
            if (otid == 0 && lc + 1 < nch) {   // sAQ is free once every O thread has arrived: fetch the next chunk's Aqk
                mbar_wait_bounded(&bar[BAR_ODONE], lc & 1, 32);
                issue_aq(lc + 1);
            }
        }
    }

    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    if (warp == 0) tmem_dealloc(tmem, TM_COLS);
#endif
}

}  // namespace cfbi
