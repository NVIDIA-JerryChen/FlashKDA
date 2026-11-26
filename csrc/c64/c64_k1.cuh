// Chunk-64 KDA, K1 arithmetic: everything of a chunk that does not depend on the entry state.
//
// k1_core() turns the raw bf16 rows of one chunk (k, q, g, beta, in shared memory) into the operands of the recurrence:
//   Kd, Qd, Kr  [64][128] bf16 (handed to a sink), gt = exp2(G_end) [128] fp32, INV and Mqk [64][64] bf16 (in smem).
// It is called by the prefill K1 kernel (sink = global stores) and by the replay kernel (sink = shared stores), so
// both execute the same instructions on the same operands.
//
// Contract (docs/20261125-replay-c64.md): 16-row sub-blocks b with reference row R_b = G[16b-1] (R_0 = 0); every exp
// factor is <= 1 except the in-sub-block ones (<= 2^127 for |lower_bound| <= 5.5):
//   kd_loc_i = bf16(khat * bf16(ex2(G_i - R_b))), w_ij = bf16(khat_j * bf16(ex2(R_i - G_j)))  (j <= i),
//   L_ij = sigmoid(beta_i) * (kd_loc_i . w_ij^T), Mqk_ij = bf16(qd_loc_i . w_ij^T),
//   INV = (I + L)^-1: 16x16 diagonal blocks by 8x8 fp32 forward substitution + a bf16 merge, then the off-diagonal
//   blocks INV_ij = -INV_ii . bf16(sum_m bf16(L_im) . INV_mj) by block forward substitution.
// Row r of every output depends only on rows 0..r (L, Mqk, INV are lower triangular; the gate cumsum is a sequential
// fold), which is what lets the replay kernel re-run this on the open chunk.
#pragma once

#include "c64_common.cuh"

namespace c64 {

struct K1Smem {
    uint16_t k[CHUNK * RS];      // raw k -> l2-normalised khat -> (replay kernel) Kd
    uint16_t q[CHUNK * RS];      // raw q -> qhat -> (replay kernel) Qd
    union {
        uint16_t gr[CHUNK * RS]; // raw g (bf16); dead after the gate phase
        float L[CHUNK * LS];     // fp32 L (strictly lower); alive until the inverse is done   (64*68*4 == 64*136*2)
        uint16_t kr[CHUNK * RS]; // (replay kernel) Kr, written after the inverse
    };
    float G[CHUNK * GS];         // cumulative gate (log2 domain), fp32
    uint16_t mq[CHUNK * MS];     // Mqk, bf16
    uint16_t inv[CHUNK * MS];    // INV, bf16
    uint16_t scr[4 * 16 * 24];   // per-warp 16x16 scratch tile of the diagonal inverses (warps 0-3)
    float beta[CHUNK];           // sigmoid(beta) in fp32 for rows < nvalid
    float dt[D];                 // dt_bias of this head
    float gend[D];               // G at the last valid row (the total of the chunk's gates)
    float gt[D];                 // exp2(G_end)
    uint16_t braw[CHUNK];        // raw beta logits (bf16)
};
static_assert(sizeof(K1Smem) % 16 == 0, "K1Smem layout");
static_assert(sizeof(float) * CHUNK * LS == sizeof(uint16_t) * CHUNK * RS, "L aliases gr / kr exactly");

struct GateParams {
    float a_log_exp;   // exp(A_log[h])
    float gate_scale;  // lower_bound * log2(e)
    float scale_b;     // bf16-rounded q scale, as fp32
};

__device__ __forceinline__ float2 ld_f2(const float* p) { return *reinterpret_cast<const float2*>(p); }

// ---------------------------------------------------------------------------------------------------------------
// step 1: gate + cumulative sum (threads 0-127: one column each, sequential over the rows), beta sigmoid and l2norm of
// the 64 k rows and 64 q rows (threads 128-255: 8 groups of 16 lanes, 8 channels per lane)
// ---------------------------------------------------------------------------------------------------------------
template <int NT>
__device__ __forceinline__ void k1_prep(K1Smem& s, GateParams gp, int nvalid) {
    const int tid = threadIdx.x;
    const int nrows = ((nvalid + 15) >> 4) * 16;      // rows of the valid sub-blocks; the others are never read
    if (tid < 128) {
        // 1a. gate a2 = gate_scale * sigmoid(exp(A_log) * (g + dt_bias)) and its sequential cumulative sum per column:
        //     G[0] = 0 + a2[0], G[r] = G[r-1] + a2[r] (rows >= nvalid add 0)
        const int col = tid;
        const float dtc = s.dt[col];
        float sum = 0.f;
        for (int r8 = 0; r8 < nrows; r8 += 8) {        // 8 rows at a time: loads, then gates, then the fold (same order)
            float gv[8], a2[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) gv[i] = bf2f(s.gr[(r8 + i) * RS + col]);
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                a2[i] = (r8 + i < nvalid) ? fmul(gp.gate_scale, sigmoid_f(fmul(gp.a_log_exp, fadd(gv[i], dtc)))) : 0.f;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                sum = fadd(sum, a2[i]);
                s.G[(r8 + i) * GS + col] = sum;
            }
        }
        s.gend[col] = sum;
        s.gt[col] = ex2_ftz(sum);
    } else if (tid < 256) {
        if (tid - 128 < CHUNK) s.beta[tid - 128] = (tid - 128 < nvalid) ? sigmoid_f(bf2f(s.braw[tid - 128])) : 0.f;
        // 1c. l2norm of the 64 k rows and 64 q rows: 8 groups of 16 lanes, 8 channels per lane
        const int t2 = tid - 128;
        const int grp = t2 >> 4, l = t2 & 15;
        #pragma unroll 4
        for (int p = 0; p < 16; ++p) {
            const int vec = p * 8 + grp;                       // 0..127: k rows then q rows (8 rows per pass)
            const int row = vec & (CHUNK - 1);
            if (row >= nrows) continue;                        // uniform per warp: both 16-lane groups share the 8-row block
            uint16_t* base = (vec < CHUNK) ? s.k : s.q;
            uint16_t* ptr = base + row * RS + 8 * l;
            const uint4 u = ld_smem_u4(ptr);
            const uint32_t w[4] = {u.x, u.y, u.z, u.w};
            float x[8];
            float acc = 0.f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                x[2 * i] = bf_lo(w[i]);
                x[2 * i + 1] = bf_hi(w[i]);
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i) acc = ffma(x[i], x[i], acc);
            #pragma unroll
            for (int d = 8; d >= 1; d >>= 1) acc = fadd(acc, __shfl_xor_sync(0xFFFFFFFFu, acc, d));
            const float inv = rsqrt_ftz(fadd(acc, 1e-6f));
            uint4 o;
            o.x = pack_bf16x2(fmul(x[0], inv), fmul(x[1], inv));
            o.y = pack_bf16x2(fmul(x[2], inv), fmul(x[3], inv));
            o.z = pack_bf16x2(fmul(x[4], inv), fmul(x[5], inv));
            o.w = pack_bf16x2(fmul(x[6], inv), fmul(x[7], inv));
            st_smem_u4(ptr, o);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// step 2: one (i, j) pair of sub-blocks (j <= i), one warp. 16x16 outputs L_ij (fp32, beta applied, masked) and Mqk_ij
// (bf16, masked), contraction over the 128 channels in 8 k-steps; the A operands (kd_loc, qd_loc) and the B operand
// (w) are built from khat / qhat / G on the fly, so nothing but the fragments is ever materialised.
// ---------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void k1_pair(K1Smem& s, int i, int j, float scale_b, int lane) {
    const int g = lane >> 2, t = lane & 3;
    float accL[2][4], accQ[2][4];
    #pragma unroll
    for (int u = 0; u < 2; ++u)
        #pragma unroll
        for (int e = 0; e < 4; ++e) { accL[u][e] = 0.f; accQ[u][e] = 0.f; }
    const uint32_t scale2 = pack_bf16x2(scale_b, scale_b);
    const int ra = 16 * i + g, rb = ra + 8;
    const float* Rr = (i > 0) ? &s.G[(16 * i - 1) * GS] : nullptr;
    #pragma unroll
    for (int st = 0; st < 8; ++st) {
        const int c0 = 16 * st + 2 * t;
        float2 R0 = make_float2(0.f, 0.f), R1 = make_float2(0.f, 0.f);
        if (i > 0) { R0 = ld_f2(Rr + c0); R1 = ld_f2(Rr + c0 + 8); }
        uint32_t aK[4], aQ[4];
        #pragma unroll
        for (int h = 0; h < 4; ++h) {
            const int row = (h & 1) ? rb : ra;
            const int col = c0 + ((h >> 1) ? 8 : 0);
            const float2 Rv = (h >> 1) ? R1 : R0;
            const float2 Gv = ld_f2(&s.G[row * GS + col]);
            const uint32_t kk = ld_u32(&s.k[row * RS + col]);
            const uint32_t qq = ld_u32(&s.q[row * RS + col]);
            const uint32_t ee = pack_bf16x2(ex2_ftz(fsub(Gv.x, Rv.x)), ex2_ftz(fsub(Gv.y, Rv.y)));   // bf16 exp factors
            aK[h] = hmul2(kk, ee);
            aQ[h] = hmul2(hmul2(qq, ee), scale2);
        }
        uint32_t bw[2][2];
        #pragma unroll
        for (int u = 0; u < 2; ++u) {
            const int rn = 16 * j + 8 * u + g;
            #pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int col = c0 + hh * 8;
                const float2 Rv = hh ? R1 : R0;
                const float2 Gv = ld_f2(&s.G[rn * GS + col]);
                const uint32_t kk = ld_u32(&s.k[rn * RS + col]);
                bw[u][hh] = hmul2(kk, pack_bf16x2(ex2_ftz(fsub(Rv.x, Gv.x)), ex2_ftz(fsub(Rv.y, Gv.y))));
            }
        }
        #pragma unroll
        for (int u = 0; u < 2; ++u) {
            mma16816(accL[u], aK, bw[u][0], bw[u][1]);
            mma16816(accQ[u], aQ, bw[u][0], bw[u][1]);
        }
    }
    #pragma unroll
    for (int u = 0; u < 2; ++u) {
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int rr = g + 8 * (e >> 1), cc = 8 * u + 2 * t + (e & 1);
            float v = fmul(accL[u][e], s.beta[16 * i + rr]);
            if (i == j && !(rr > cc)) v = 0.f;
            s.L[(16 * i + rr) * LS + 16 * j + cc] = v;
        }
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int rr = g + 8 * h, cc = 8 * u + 2 * t;
            float lo = accQ[u][2 * h], hi = accQ[u][2 * h + 1];
            if (i == j) {
                if (!(rr >= cc)) lo = 0.f;
                if (!(rr >= cc + 1)) hi = 0.f;
            }
            st_u32(&s.mq[(16 * i + rr) * MS + 16 * j + cc], pack_bf16x2(lo, hi));
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// step 3a: (I + L_bb)^-1 of the strictly lower 16x16 diagonal block b (fp32 in s.L) -> bf16 block of s.inv.
// 8x8 fp32 forward substitution (one row per lane, utils.cuh's fused routine), then the bf16 merge
//   P = [Ai 0; 0 Bi], dc = P @ [0 0; C 0], INV = P + bf16(bf16(-dc) @ P)    (C = L's lower-left 8x8 block)
// ---------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void k1_inv_diag(K1Smem& s, int b, int lane) {
    uint16_t* scr = &s.scr[b * 16 * 24];
    const int i = lane & 7;
    const int row0 = lane & 8;
    const int base = 16 * b;
    float inv[8];
    #pragma unroll
    for (int p = 0; p < 8; ++p)
        inv[p] = (p < i) ? s.L[(base + row0 + i) * LS + base + row0 + p] : (p == i ? 1.0f : 0.0f);
    #pragma unroll
    for (int st = 0; st < 7; ++st) {
        const float row_scale = -inv[st];
        #pragma unroll
        for (int p = 0; p < st; ++p) {
            const float pivot = __shfl_sync(0xFFFFFFFFu, inv[p], (lane & ~7) | st);
            if (i > st) inv[p] = ffma(row_scale, pivot, inv[p]);
        }
        if (i > st) inv[st] = row_scale;
    }
    if (lane < 8) {            // rows 0..7 of P: [Ai | 0]
        #pragma unroll
        for (int j = 0; j < 8; ++j) scr[lane * 24 + j] = f2bf(inv[j]);
        #pragma unroll
        for (int j = 8; j < 16; ++j) scr[lane * 24 + j] = 0;
    } else if (lane < 16) {    // rows 8..15 of P: [0 | Bi]
        #pragma unroll
        for (int j = 0; j < 8; ++j) scr[lane * 24 + j] = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) scr[lane * 24 + 8 + j] = f2bf(inv[j]);
    }
    __syncwarp();
    const int g = lane >> 2, t = lane & 3;
    uint32_t pa[4], pb[4];
    ldsm_x4(pa, a_addr(scr, 24, lane));        // P as the A operand
    ldsm_x4_t(pb, bt_addr(scr, 24, lane));     // P as the B operand ([k][n] row-major)
    // dc = P @ M, M = [0 0; C 0] with C = bf16(L[base+8.., base+0..]): only the k >= 8, n < 8 part of B is non-zero
    const uint32_t mb1 = pack_bf16x2(s.L[(base + 8 + 2 * t) * LS + base + g], s.L[(base + 8 + 2 * t + 1) * LS + base + g]);
    float dc[4] = {0.f, 0.f, 0.f, 0.f};
    mma16816(dc, pa, 0u, mb1);
    // o = bf16(-dc) @ P: negdc is non-zero only in columns 0..7, o only in rows 8..15 x columns 0..7
    const uint32_t na[4] = {pack_bf16x2(-dc[0], -dc[1]), pack_bf16x2(-dc[2], -dc[3]), 0u, 0u};
    float oc[4] = {0.f, 0.f, 0.f, 0.f};
    mma16816(oc, na, pb[0], pb[1]);
    __syncwarp();
    if (lane < 16) {   // INV block = P (rows 8..15 get their lower-left 8x8 from o below)
        const uint4* src = reinterpret_cast<const uint4*>(scr + lane * 24);
        uint4* dst = reinterpret_cast<uint4*>(&s.inv[(base + lane) * MS + base]);
        dst[0] = src[0];
        dst[1] = src[1];
    }
    __syncwarp();
    st_u32(&s.inv[(base + 8 + g) * MS + base + 2 * t], pack_bf16x2(oc[2], oc[3]));
}

// ---------------------------------------------------------------------------------------------------------------
// step 3b: off-diagonal block INV_ij (i > j): Y = sum_{m=j}^{i-1} bf16(L_im) . INV_mj (fp32 accumulate, m ascending),
// INV_ij = bf16(-(INV_ii . bf16(Y)))
// ---------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void k1_inv_off(K1Smem& s, int i, int j, int lane) {
    const int g = lane >> 2, t = lane & 3;
    float y[2][4];
    #pragma unroll
    for (int u = 0; u < 2; ++u)
        #pragma unroll
        for (int e = 0; e < 4; ++e) y[u][e] = 0.f;
    #pragma unroll
    for (int m = 0; m < 3; ++m) {
        if (m >= j && m < i) {
            const float* L0 = &s.L[(16 * i + g) * LS + 16 * m + 2 * t];
            const float* L1 = &s.L[(16 * i + g + 8) * LS + 16 * m + 2 * t];
            const float2 x0 = ld_f2(L0), x1 = ld_f2(L1), x2 = ld_f2(L0 + 8), x3 = ld_f2(L1 + 8);
            const uint32_t a[4] = {pack_bf16x2(x0.x, x0.y), pack_bf16x2(x1.x, x1.y), pack_bf16x2(x2.x, x2.y),
                                   pack_bf16x2(x3.x, x3.y)};
            uint32_t bb[4];
            ldsm_x4_t(bb, bt_addr(&s.inv[(16 * m) * MS + 16 * j], MS, lane));
            mma16816(y[0], a, bb[0], bb[1]);
            mma16816(y[1], a, bb[2], bb[3]);
        }
    }
    uint32_t yb[2][2];
    #pragma unroll
    for (int u = 0; u < 2; ++u) {
        yb[u][0] = movm_t(pack_bf16x2(y[u][0], y[u][1]));
        yb[u][1] = movm_t(pack_bf16x2(y[u][2], y[u][3]));
    }
    uint32_t a2[4];
    ldsm_x4(a2, a_addr(&s.inv[(16 * i) * MS + 16 * i], MS, lane));
    float z[2][4];
    #pragma unroll
    for (int u = 0; u < 2; ++u) {
        #pragma unroll
        for (int e = 0; e < 4; ++e) z[u][e] = 0.f;
        mma16816(z[u], a2, yb[u][0], yb[u][1]);
    }
    #pragma unroll
    for (int u = 0; u < 2; ++u) {
        st_u32(&s.inv[(16 * i + g) * MS + 16 * j + 8 * u + 2 * t], pack_bf16x2(-z[u][0], -z[u][1]));
        st_u32(&s.inv[(16 * i + g + 8) * MS + 16 * j + 8 * u + 2 * t], pack_bf16x2(-z[u][2], -z[u][3]));
    }
}

// ---------------------------------------------------------------------------------------------------------------
// step 4: K2 operands Kd = bf16(khat bf16(ex2(G))), Qd = bf16(bf16(qhat bf16(ex2(G))) scale), Kr = bf16(khat
// bf16(ex2(G_end - G))), 16 bytes (8 channels) at a time, handed to the sink: sink(row, c8, kd, qd, kr)
// ---------------------------------------------------------------------------------------------------------------
template <int NT, class Sink>
__device__ __forceinline__ void k1_operands(K1Smem& s, GateParams gp, int nrows, bool want_kr, Sink&& sink) {
    const int tid = threadIdx.x;
    const uint32_t scale2 = pack_bf16x2(gp.scale_b, gp.scale_b);
    for (int vec = tid; vec < nrows * 16; vec += NT) {        // nrows x 16 vectors of 8 channels
        const int row = vec >> 4, c8 = (vec & 15) * 8;
        const uint4 kk = ld_smem_u4(&s.k[row * RS + c8]);
        const uint4 qq = ld_smem_u4(&s.q[row * RS + c8]);
        const float4 ga = *reinterpret_cast<const float4*>(&s.G[row * GS + c8]);
        const float4 gb = *reinterpret_cast<const float4*>(&s.G[row * GS + c8 + 4]);
        const float G8[8] = {ga.x, ga.y, ga.z, ga.w, gb.x, gb.y, gb.z, gb.w};
        const uint32_t kw[4] = {kk.x, kk.y, kk.z, kk.w};
        const uint32_t qw[4] = {qq.x, qq.y, qq.z, qq.w};
        uint32_t okd[4], oqd[4], okr[4] = {0u, 0u, 0u, 0u};
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t ee = pack_bf16x2(ex2_ftz(G8[2 * i]), ex2_ftz(G8[2 * i + 1]));
            okd[i] = hmul2(kw[i], ee);
            oqd[i] = hmul2(hmul2(qw[i], ee), scale2);
        }
        if (want_kr) {
            const float4 ea = *reinterpret_cast<const float4*>(&s.gend[c8]);
            const float4 eb = *reinterpret_cast<const float4*>(&s.gend[c8 + 4]);
            const float E8[8] = {ea.x, ea.y, ea.z, ea.w, eb.x, eb.y, eb.z, eb.w};
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                okr[i] = hmul2(kw[i], pack_bf16x2(ex2_ftz(fsub(E8[2 * i], G8[2 * i])), ex2_ftz(fsub(E8[2 * i + 1], G8[2 * i + 1]))));
        }
        sink(row, c8, make_uint4(okd[0], okd[1], okd[2], okd[3]), make_uint4(oqd[0], oqd[1], oqd[2], oqd[3]),
             make_uint4(okr[0], okr[1], okr[2], okr[3]));
    }
}

// The whole K1 arithmetic of one chunk, 256 threads. Precondition: s.k, s.q, s.gr, s.braw hold the raw rows (rows >=
// nvalid zero) and a __syncthreads() separates that from this call. No barrier at the end: the caller synchronises
// before it consumes s.inv, s.mq, s.gt or what the sink wrote.
// tp (optional, profiling only): thread 0 writes clock64() after every phase into tp[0..6]
#define C64_TS(i) do { if (tp != nullptr && threadIdx.x == 0) tp[i] = clock64(); } while (0)
// NT = threads of the CTA (256 or 320; with 320 every one of the 10 sub-block pairs gets its own warp). s.dt must hold
// dt_bias of the head (loaded by the caller with the rows).
template <int NT, class Sink>
__device__ __forceinline__ void k1_core(K1Smem& s, GateParams gp, int nvalid, bool want_kr, Sink&& sink, long long* tp = nullptr) {
    static_assert(NT == 256 || NT == 320, "K1 runs on 8 or 10 warps");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nbv = (nvalid + 15) >> 4;
    C64_TS(0);
    k1_prep<NT>(s, gp, nvalid);
    __syncthreads();
    C64_TS(1);
    for (int p = warp; p < 10; p += NT / 32) {          // pairs (0,0) (1,0) (1,1) (2,0) (2,1) (2,2) (3,0) (3,1) (3,2) (3,3)
        const int i = (p >= 6) ? 3 : (p >= 3) ? 2 : (p >= 1) ? 1 : 0;
        const int j = p - (i * (i + 1)) / 2;
        if (16 * i < nvalid) k1_pair(s, i, j, gp.scale_b, lane);
    }
    __syncthreads();
    C64_TS(2);
    if (warp < 4 && warp < nbv) k1_inv_diag(s, warp, lane);
    __syncthreads();
    C64_TS(3);
    #pragma unroll
    for (int d = 1; d < 4; ++d) {
        const int i = warp + d;
        if (i < 4 && i < nbv) k1_inv_off(s, i, warp, lane);
        __syncthreads();
    }
    C64_TS(4);
    k1_operands<NT>(s, gp, nbv * 16, want_kr, sink);
    C64_TS(5);
}

}  // namespace c64
