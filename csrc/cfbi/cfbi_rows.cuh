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
//   * the cache append of khat / G2 and the reference-row copy no longer cost a barrier;
//   * schedule trims (no operand and no operation changes): the plan entry arrives in one round trip, the tokens of a row are loaded before the cached rows and the L2 prefetch (which walks one
//     plane at a time and divides nothing), the lines of the first row's tokens are prefetched into L1 in front of the arrival atomic of the single launch, the accumulators of the B and D epilogues
//     are picked with selects (no local memory), and the last row of a call has no end-of-row barrier.
// The cache planes (khat, G2, UT, W, U) and E live in global memory (fp32); plain (non-const, non-restrict)
// pointers are used so the compiler never turns reads of freshly written rows into .nc loads.
//
// rows_kernel<false> runs behind the plan kernel (head rows of a prefill call, decode with IMPL_DECODE_PLAN); rows_kernel<true> is the same body for decode without
// a plan kernel (one launch per decode call, cfbi_launch.h decode_single_launch): it derives its plan entry itself and advances pos by an arrival count, see rows_entry.
#pragma once

#include "cfbi/cfbi_common.cuh"

#include <cstdio>

namespace cfbi {

#ifdef CFBI_ROWS_PROFILE
// Row-engine profile (profile build only, read back by the host through rows_profile_read in cfbi_launch.cu): clock64 stamps of every row of every CTA relative to the
// stamp taken at kernel entry. Slot [row position][CTA = n * H + h]: entries 0..7 = ROWS_TS(0..7) of the row, 8 = kernel exit (written to the slot of the last row of the
// CTA), 9 = SM id + 1 (0: the slot holds no row).
constexpr int ROWS_PROF_POS = 64;
constexpr int ROWS_PROF_CTAS = 4096;
constexpr int ROWS_PROF_STAMPS = 10;
__device__ unsigned g_rows_prof[ROWS_PROF_POS][ROWS_PROF_CTAS][ROWS_PROF_STAMPS];
#define ROWS_TS(i) do { if (threadIdx.x == 0) tp[i] = clock64(); } while (0)
#else
#define ROWS_TS(i) do {} while (0)
#endif

// Test hooks of the row engine (variant builds only: the default build defines none of these macros and compiles to the same machine code, which Gate X shows).
// ROWS_SYNC(n) is barrier n of a row (1 .. 6) and ROWS_SYNC_END() the barrier that ends it (7); ROWS_DELAY(n) is delay site n: 0 = kernel entry, before the first loads, 1 .. 7 = before
// barrier n, 8 = right after barrier 6 (the start of the chunk-end phase, which reads S.g2 and S.gend: the window that the barrier ending the row protects against the prologue of the
// next row), and the four sites of the arrival protocol of the single-launch decode (rows_entry, thread 0 of every participating CTA): 9 = before the arrival, 10 = after the arrival of the
// FIRST CTA of its sequence to arrive, 11 = after the arrival of the LAST one and before it stores the new fill, 12 = after the arrival of every CTA (ROWS_DECODE_ARRIVED: the first and the last
// are told by the count that the arrival atomic returned, never by the head). CFBI_ROWS_DELAY_NS compiles the delay hook (its value is the default length of a delay in ns; the host chooses
// sites, warp and CTAs at run time through rows_test_delay_config in cfbi_launch.cu and reads how many delays ran through rows_test_delay_hits). CFBI_ROWS_TEST_NO_END_BARRIER removes the
// barrier that ends a row (a racy negative control: a warp that sleeps in the chunk-end phase (site 8) then lets the others start the next row and rewrite S.g2 under it),
// CFBI_ROWS_TEST_POS_OOB moves the accesses to a pos word that the host selects (rows_test_pos_offset takes an offset and an access mask: bit POS_ACCESS_FUSED_STORE = the fused store of the
// two-launch row engine, POS_ACCESS_ARRIVAL = the arrival atomic and POS_ACCESS_ELECTED_STORE = the elected store of the single launch) to pos[slot + offset] (out-of-bounds controls for
// memcheck) and CFBI_ROWS_TEST_COMPILE_ERROR stops the compilation (the control that shows that a variant that does not build fails its test).
#ifdef CFBI_ROWS_DELAY_NS
struct RowsDelayConfig {
    unsigned ns;           // length of a delay (0: CFBI_ROWS_DELAY_NS)
    int warp;              // the warp that sleeps (-1: every warp)
    int cta_mod, cta_rem;  // the CTAs that sleep: cta % cta_mod == cta_rem with cta = n * H + h (cta_mod 0: every CTA)
    unsigned sites;        // bit n: delay site n is active (n = 0 .. 12)
};
__device__ RowsDelayConfig g_rows_delay;
__device__ unsigned long long g_rows_delay_hits;
__device__ __forceinline__ void rows_delay(int site) {
    const RowsDelayConfig c = g_rows_delay;
    if (!((c.sites >> site) & 1u)) return;
    const int cta = blockIdx.x * gridDim.y + blockIdx.y;
    if (c.cta_mod > 0 && cta % c.cta_mod != c.cta_rem) return;
    if (c.warp >= 0 && (int)(threadIdx.x >> 5) != c.warp) return;
    if ((threadIdx.x & 31) == 0) atomicAdd(&g_rows_delay_hits, 1ull);
    __nanosleep(c.ns ? c.ns : (unsigned)CFBI_ROWS_DELAY_NS);
    __threadfence_block();   // the loads that follow the site stay behind the sleep (the compiler may not hoist them above it)
}
// The delays after the arrival of the single-launch decode (rows_entry): site 12 for every CTA, site 10 for the first CTA of its sequence to arrive and site 11 for the last. ``old`` is the word
// that the arrival atomic returned (the number of CTAs of the sequence that arrived before this one is above POS_COUNT_SHIFT).
__device__ __forceinline__ void rows_delay_arrived(int old, int heads) {
    rows_delay(12);
    if ((old >> POS_COUNT_SHIFT) == 0) rows_delay(10);
    if ((old >> POS_COUNT_SHIFT) == heads - 1) rows_delay(11);
}
#define ROWS_DELAY(n) rows_delay(n)
#define ROWS_DECODE_ARRIVED(old, heads) rows_delay_arrived(old, heads)
#else
#define ROWS_DELAY(n) do {} while (0)
#define ROWS_DECODE_ARRIVED(old, heads) do {} while (0)
#endif
#define ROWS_SYNC(n) do { ROWS_DELAY(n); __syncthreads(); } while (0)
#ifdef CFBI_ROWS_TEST_NO_END_BARRIER
#define ROWS_SYNC_END() ROWS_DELAY(7)
#else
#define ROWS_SYNC_END() ROWS_SYNC(7)
#endif
#ifdef CFBI_ROWS_TEST_COMPILE_ERROR
#error "CFBI_ROWS_TEST_COMPILE_ERROR: a deliberate compile error (the control that shows that a variant that does not build fails its test)"
#endif
// The accesses to a pos word that the out-of-bounds control can move (bit n of the access mask of rows_test_pos_offset): the fused store of the two-launch row engine, and the arrival atomic and
// the elected store of the single launch.
constexpr int POS_ACCESS_FUSED_STORE = 0, POS_ACCESS_ARRIVAL = 1, POS_ACCESS_ELECTED_STORE = 2;
#ifdef CFBI_ROWS_TEST_POS_OOB
__device__ int g_rows_test_pos_offset;
__device__ unsigned g_rows_test_pos_accesses;
#define ROWS_POS_INDEX(access, slot) ((slot) + (((g_rows_test_pos_accesses >> (access)) & 1u) ? g_rows_test_pos_offset : 0))
#else
#define ROWS_POS_INDEX(access, slot) (slot)
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
// The static shared memory of rows_kernel is this structure plus, for rows_kernel<true>, one int (the fill of the open chunk). The need that cfbi_policy.h states is the compiler's
// figure for it: at least the layout, and less than one allocation unit (1 KiB) more.
static_assert(sizeof(RowSmem) + sizeof(int) <= ROWS_STATIC_SMEM_BYTES, "ROWS_STATIC_SMEM_BYTES (cfbi_policy.h) must cover the static shared memory of the row engine");
static_assert(ROWS_STATIC_SMEM_BYTES - (sizeof(RowSmem) + sizeof(int)) < 1024, "ROWS_STATIC_SMEM_BYTES (cfbi_policy.h) is more than an allocation unit above the layout: take the compiler's figure");

__device__ __forceinline__ void prefetch_l2(const void* ptr) { asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr)); }
__device__ __forceinline__ void prefetch_l1(const void* ptr) { asm volatile("prefetch.global.L1 [%0];" ::"l"(ptr)); }
#ifndef CFBI_ROWS_PF_L1
#define CFBI_ROWS_PF_L1 0
#endif
__device__ __forceinline__ void prefetch_state(const void* ptr) {
#if CFBI_ROWS_PF_L1
    prefetch_l1(ptr);
#else
    prefetch_l2(ptr);
#endif
}

// 8 consecutive bf16 held in one uint4 -> 8 fp32 (exact): the unpacking half of load8_bf16 (cfbi_common.cuh), so that the 16-byte load can be issued early and converted where it is used.
__device__ __forceinline__ void unpack8_bf16(const uint4& u, float (&x)[8]) {
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        x[2 * i] = __uint_as_float(w[i] << 16);
        x[2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000u);
    }
}

// The global loads of a row that need only its token, its head and the role of the thread, none of the state and none of pos: warp 0 holds 16 bytes of k (lanes 0-15) or of q (lanes 16-31),
// channels 8 * (lane & 15) ..+7; thread tid < HEAD_K the g of channel tid, thread HEAD_K the beta logit, thread tid < HEAD_V the v of channel tid. They stay raw (the bf16 pairs of k / q are
// unpacked where phase A uses them). The arguments are the tensors, not the parameter block: only the entry of the kernel is handed the block (the protocol rules of the single launch).
struct RowTokens {
    uint4 kq;
    float g, beta, v;
};
__device__ __forceinline__ RowTokens rows_load_tokens(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* g, const __nv_bfloat16* beta, const __nv_bfloat16* v, int64_t tok,
                                                      int h, int H, int tid, int lane) {
    const int64_t off = (tok * H + h) * HEAD_K;   // q/k/v/g row offset
    RowTokens t;
    t.kq = make_uint4(0u, 0u, 0u, 0u);
    t.g = 0.f;
    t.beta = 0.f;
    t.v = 0.f;
    if (tid < 32) t.kq = *reinterpret_cast<const uint4*>((lane >= 16 ? q : k) + off + 8 * (lane & 15));
    if (tid < HEAD_K) t.g = bf16_to_f32(g[off + tid]);
    if (tid == HEAD_K) t.beta = bf16_to_f32(beta[tok * H + h]);
    if (tid < HEAD_V) t.v = bf16_to_f32(v[off + tid]);
    return t;
}
// The L1 prefetch of the lines that rows_load_tokens loads (the same addresses): a prefetch holds no register, the loaded values would.
__device__ __forceinline__ void rows_prefetch_tokens(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* g, const __nv_bfloat16* beta, const __nv_bfloat16* v, int64_t tok, int h,
                                                     int H, int tid, int lane) {
    const int64_t off = (tok * H + h) * HEAD_K;
    if (tid < 32) prefetch_l1((lane >= 16 ? q : k) + off + 8 * (lane & 15));
    if (tid < HEAD_K) prefetch_l1(g + off + tid);
    if (tid == HEAD_K) prefetch_l1(beta + tok * H + h);
    if (tid < HEAD_V) prefetch_l1(v + off + tid);
}

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

// A relaxed (strong, device scope) store: the one store to a pos word that other CTAs of the call may be racing with (see rows_entry), so it must be a morally strong
// operation like the atomics next to it.
__device__ __forceinline__ void st_relaxed_gpu(int* ptr, int v) {
    asm volatile("st.relaxed.gpu.global.b32 [%0], %1;" ::"l"(ptr), "r"(v) : "memory");
}
#ifdef CFBI_DECODE_TEST_SNAPSHOT_AFTER_ARRIVAL
__device__ __forceinline__ int ld_relaxed_gpu(const int* ptr) {   // a wrong protocol (tests only): a read of the pos word of its own
    int v;
    asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(ptr) : "memory");
    return v;
}
#endif

// The sequence entry (what the plan kernel writes to p.plan[n]) of this CTA.
//
// DECODE = false: the plan kernel ran in front of this kernel; read its entry.
//
// DECODE = true: single-launch decode (mode 1, cfbi_launch.h decode_single_launch): there is no plan kernel. The entry is derived here from the call's arguments:
// seq_window is the function the plan kernel uses for the token window and the slot, and hrows = len (every row goes through the row engine). What the plan kernel
// is also there for is the SNAPSHOT of pos: the H head CTAs of a sequence all start from the same fill of the open chunk, and one of them must not advance the
// pos word before the others have read it. Here the word is read and COUNTED by one atomic: every CTA of a participating sequence starts with
//     old = atomicAdd(&pos[slot], 1 << POS_COUNT_SHIFT)
// whose result is the word as it was just before the add: the fill of the open chunk in the low bits (& POS_MASK: the counter is never part of it) and the number of CTAs
// that arrived before this one above. All H CTAs of a sequence take the same decision (len and slot do not depend on the head), so either all of them arrive or none does.
// The arrival that sees H - 1 before it is the last: every other CTA of the sequence has done its atomic, i.e. has read pos and will never touch the word again, so it stores
// the plain new value (pos0 + len) % BT, which also clears the counter; the other CTAs leave the word alone. The last CTA to ARRIVE is not the last to FINISH: the others may still be
// in their row loops when it stores, and the store is not a completion signal for E, rows or the output (nothing reads pos to learn that the call is done; the next call reads it after
// this kernel, in the same stream or behind an event). Between the first arrival and that store the word is not a plain pos, and nothing may read it as one while the call runs (a
// precondition of the state anyway: the slots of a call are distinct and no other call works on the state concurrently). A plain pos at rest is not a proof that a call finished
// either: an interrupted call can leave arrival bits in the word, or a plain pos with half updated E and rows. Every access to the word is an atomic or a relaxed store (no data
// race in the memory model), and after a call that ran to the end it is plain again: pos, E and rows are what the two-launch decode leaves, bit for bit. A skipped CTA (slot < 0, or an
// empty sequence) never touches the word.
// Thread 0 does one arrival atomic per participating CTA, and the elected CTA one store per sequence; it hands the fill to the CTA through shared memory; all threads call this function
// (it has a barrier), and the early exit (len == 0) is uniform over the CTA.
// Test builds only (the default build has none of these macros): the delay sites 9 .. 12 (ROWS_DELAY, ROWS_DECODE_ARRIVED: before the arrival, after the arrival of the first CTA, after
// the arrival of the last one before its store, after the arrival of every CTA) make the late starters of a grid that does not fit one wave on demand; ROWS_POS_INDEX moves the arrival
// atomic and the elected store out of bounds; and the wrong protocols that the harness must catch: CFBI_DECODE_TEST_EVERY_CTA_WRITES (no counting: every CTA reads the word and stores the
// new value at the end of its row loop, the plan-less version of the two-launch scheme without its snapshot), CFBI_DECODE_TEST_FIRST_ARRIVER (the first CTA to arrive, not the last,
// stores), CFBI_DECODE_TEST_SNAPSHOT_AFTER_ARRIVAL (the fill is read from the word after the arrival, not taken from what the atomic returned) and CFBI_DECODE_TEST_NO_ELECTED_STORE (the
// elected CTA does not store: every arrival bit stays in the word after the call, which the at-rest check of the harness must see).
template <bool DECODE>
using RowsParams = std::conditional_t<DECODE, DecodeParams, Params>;   // DecodeParams: the call's Params and its token windows (cfbi_common.cuh)

template <bool DECODE>
__device__ __forceinline__ SeqPlan rows_entry(const RowsParams<DECODE>& p, int n, int& pos_smem) {
    if constexpr (!DECODE) {
        return p.plan[n];
    } else {
        const SeqWindow w = seq_window(p.cu_seqlens, p.fixed_len, p.slot_ids, n);
        if (w.len == 0) return SeqPlan{0, 0, 0, 0, 0, 0, 0, w.slot};   // padding entry or empty sequence (uniform over the CTA): no state access, the caller returns
        if (threadIdx.x == 0) {
            ROWS_DELAY(9);   // test hook: before the arrival
#ifdef CFBI_DECODE_TEST_EVERY_CTA_WRITES
            const int old = atomicAdd(p.pos + ROWS_POS_INDEX(POS_ACCESS_ARRIVAL, w.slot), 0);   // a plain read of the word (an add of zero): no counting
            pos_smem = old & POS_MASK;
#else
            const int old = atomicAdd(p.pos + ROWS_POS_INDEX(POS_ACCESS_ARRIVAL, w.slot), 1 << POS_COUNT_SHIFT);
            ROWS_DECODE_ARRIVED(old, p.H);   // test hook: delay sites 10 (first arriver), 11 (last arriver, before its store) and 12 (every CTA), after the arrival
#ifdef CFBI_DECODE_TEST_SNAPSHOT_AFTER_ARRIVAL
            pos_smem = ld_relaxed_gpu(p.pos + w.slot) & POS_MASK;   // wrong protocol (tests only): the fill from a read of its own after the arrival, which the elected store may have rewritten
#else
            pos_smem = old & POS_MASK;
#endif
#ifdef CFBI_DECODE_TEST_FIRST_ARRIVER
            if ((old >> POS_COUNT_SHIFT) == 0) st_relaxed_gpu(p.pos + ROWS_POS_INDEX(POS_ACCESS_ELECTED_STORE, w.slot), (int)(((int64_t)(old & POS_MASK) + w.len) % BT));
#else
#ifndef CFBI_DECODE_TEST_NO_ELECTED_STORE
            if ((old >> POS_COUNT_SHIFT) == p.H - 1) st_relaxed_gpu(p.pos + ROWS_POS_INDEX(POS_ACCESS_ELECTED_STORE, w.slot), (int)(((int64_t)(old & POS_MASK) + w.len) % BT));
#endif
#endif
#endif
        }
        __syncthreads();
        return SeqPlan{w.tok0, w.len, pos_smem, w.len, 0, 0, 0, w.slot};
    }
}

#ifndef CFBI_ROWS_MINB
#define CFBI_ROWS_MINB 2
#endif
// DECODE = false: the head rows / decode rows of a call whose plan kernel has run. DECODE = true: the single-launch decode, see rows_entry; same body, same arithmetic.
template <bool DECODE>
__global__ void __launch_bounds__(NTHREADS, CFBI_ROWS_MINB) rows_kernel(const RowsParams<DECODE> p) {
#ifdef CFBI_ROWS_PROFILE
    const long long prof_entry = clock64();   // the profile starts at kernel entry: before the plan load and the rest of the prologue
#endif
    ROWS_DELAY(0);   // test hook: kernel entry, before the plan load and the first loads
    __shared__ RowSmem S;
    __shared__ int dec_pos0;   // DECODE only: the fill of the open chunk, from thread 0 (unused, hence not allocated, with DECODE = false)
    const int n = blockIdx.x, h = blockIdx.y, H = p.H;
    const int tid = threadIdx.x, warp = tid >> 5;
    const LaneIdx L;
    if constexpr (DECODE) {
        // What the first row loads and needs only the slot's token start and the head, nothing of the fill of the open chunk, is requested here, in front of the arrival atomic of rows_entry: the
        // lines are prefetched into L1, so their round trip overlaps the atomic's and the loads of the row, right behind the entry, hit L1. A prefetch holds no register across the entry; the loaded
        // values would (the first row's tokens are 7 registers, and the single-launch instantiation then spills where it did not). What depends on the fill (the cached rows b..pos-1 and the
        // previous G2 row, whose addresses and masks are functions of pos) stays behind the entry: the atomic is the only read of pos. The entry derives the same window (the inputs are read-only
        // during a call, as the protocol already needs for the heads of a sequence to agree).
        const SeqWindow w0 = seq_window(p.cu_seqlens, p.fixed_len, p.slot_ids, n);
        if (w0.len != 0) rows_prefetch_tokens(p.q, p.k, p.g, p.beta, p.v, (int64_t)w0.tok0, h, H, tid, L.lane);   // uniform over the CTA: the entry returns for an empty window
    }
    const SeqPlan sp = rows_entry<DECODE>(p, n, dec_pos0);
    // DECODE = false: the whole plan entry in one global round trip. Left alone, the compiler sinks every load of the entry but `hrows` below the exit that follows, so the fields that every address
    // below needs (tok0, pos0, slot) cost a second dependent round trip. The empty volatile asm pins their loads in front of the test; it emits no instruction.
    if constexpr (!DECODE) asm volatile("" ::"r"(sp.tok0), "r"(sp.len), "r"(sp.pos0), "r"(sp.slot));
    const int nrows = sp.hrows;
    if (nrows == 0) return;
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
    int prof_last = -1;   // position of the last row of this CTA that was recorded
#endif
    int pos = sp.pos0;
    // cached rows b..pos-1 of UT | W (role column) and of U (value column, tid < 128) of the current sub-block, and the
    // reference row's G2: loaded from global for the first row of the call, afterwards carried in registers
    // (a row's own result is appended to them), so consecutive head rows do not wait for their predecessor's stores.
    float xpre[16], upre[16];
    float gref_pre = 0.f;
#pragma unroll
    for (int jj = 0; jj < 16; ++jj) { xpre[jj] = 0.f; upre[jj] = 0.f; }
    for (int i = 0; i < nrows; ++i) {
        ROWS_TS(0);
#ifdef CFBI_ROWS_PROFILE
        const int prof_pos = pos;   // the position of this row (pos is advanced at its end)
#endif
        const int64_t tok = (int64_t)sp.tok0 + i;
        const int64_t off = (tok * H + h) * HEAD_K;   // q/k/v/g/out row offset
        const int s = pos >> 4, b = pos & ~15, slot = pos & 15;
        const int slot_hi = slot >> 3;   // the current row is in A/C rows g (0) or g+8 (1) of the lanes with g == slot & 7
        const bool mine = g == (slot & 7);

        // ---- early loads: nothing here depends on this row's prep ------------------------------------
        // Issue order: the loads of the critical path first (the row's own tokens, the cached rows of the recursion and the previous G2 row), then the L2 prefetch of the state: a run of
        // instructions that only has to be issued before the state is needed, not in front of the loads that the first phases wait for.
        const RowTokens tk = rows_load_tokens(p.q, p.k, p.g, p.beta, p.v, tok, h, H, tid, L.lane);
        float gprev = 0.f;
        if (i == 0) {
#pragma unroll
            for (int jj = 0; jj < 16; ++jj) xpre[jj] = (b + jj < pos) ? Xg[(b + jj) * HEAD_K + col] : 0.f;
            if (tid < HEAD_V) {
#pragma unroll
                for (int jj = 0; jj < 16; ++jj) upre[jj] = (b + jj < pos) ? Ru[(b + jj) * HEAD_V + tid] : 0.f;
                if (pos > 0) gprev = Rg[(pos - 1) * HEAD_K + tid];
                if (b < pos) gref_pre = Rg[b * HEAD_K + tid];
            }
            // pull the whole state this CTA will touch towards L2 so later phases do not pay HBM latency
            const char* eb = reinterpret_cast<const char*>(Eg);
            prefetch_state(eb + tid * 128);
            prefetch_state(eb + (tid + NTHREADS) * 128);
            const int lines = pos * 4;   // 512 B per row = 4 lines of 128 B per plane, five planes: one plane at a time, so that no index is divided
            // (not unrolled: five copies of the loop hold registers that the row loop does not have)
#pragma unroll 1
            for (int pl = 0; pl < NPLANES; ++pl)
                for (int l = tid; l < lines; l += NTHREADS) prefetch_state(R + pl * ROW_PLANE_FLOATS + l * 32);
        } else if (tid < HEAD_V) {
            gprev = S.g2[tid];   // the previous row's G2 (written by this very thread last iteration)
        }

        // ---- A. token-local prep -------------------------------------------------------------
        if (warp == 0) {   // lanes 0-15: k, lanes 16-31: q (channels 8*(lane&15) ..+7)
            const bool is_q = L.lane >= 16;
            const int c0 = 8 * (L.lane & 15);
            float x[8];
            unpack8_bf16(tk.kq, x);
            const float inv = l2norm_inv(x);
            float* dst = is_q ? S.q : S.k;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float y = fmul(x[j], inv);
                dst[c0 + j] = is_q ? fmul(y, p.scale) : y;
            }
        }
        if (tid < HEAD_K) {
            const float a2 = gate_a2(tk.g, dtb, ale, p.scale2);
            const float g2v = (pos == 0) ? a2 : fadd(gprev, a2);
            S.g2[tid] = g2v;
            S.gref[tid] = (b == pos) ? g2v : gref_pre;
            gref_pre = (b == pos) ? g2v : gref_pre;
        }
        if (tid == HEAD_K) S.beta = beta_act(tk.beta);
        ROWS_SYNC(1);
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
                    // tile row slot holds ks (-> Akk), row slot ^ 8 holds qs (-> Aqk). Selects, not an index 2 * slot_hi + e into big / sml: a run-time index puts both arrays into local
                    // memory (a store / load round trip per row)
                    const float bk = slot_hi ? big[2 + e] : big[e], sk = slot_hi ? sml[2 + e] : sml[e];
                    const float bq = slot_hi ? big[e] : big[2 + e], sq = slot_hi ? sml[e] : sml[2 + e];
                    S.akk[c] = fmul(S.beta, fadd(bk, sk));
                    S.aqk[c] = fadd(bq, sq);
                }
            }
        }
        ROWS_SYNC(2);
        ROWS_TS(2);

        // ---- C. cross-sub-block products (UT, W rows for the recursion; U rows for O), then the UT / W row -------
        if (s > 0) {
            if (s == 1) rows_m_tiles<2>(S, Rut, Rw, Ru, warp, L, slot);
            else if (s == 2) rows_m_tiles<4>(S, Rut, Rw, Ru, warp, L, slot);
            else rows_m_tiles<6>(S, Rut, Rw, Ru, warp, L, slot);
            ROWS_SYNC(3);
        }
        ROWS_TS(3);
        float x_row = 0.f, u_row = 0.f;
        {   // right-hand side, cross-block subtraction, in-sub-block FMA chain over the cached rows; thread = (UT|W, column)
            float rhs;
            if (which == 0) rhs = fmul(S.beta, tk.v);
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
            x_row = x;
            if (which == 0) S.gq[col] = fmul(ex2_ftz(S.g2[col]), S.q[col]);
            if (tid < HEAD_K) {   // append khat / G2 of this row to the cache (later rows read them)
                Rk[pos * HEAD_K + tid] = S.k[tid];
                Rg[pos * HEAD_K + tid] = S.g2[tid];
            }
        }
        ROWS_SYNC(4);
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
#ifdef CFBI_ROWS_TEST_ALTER_SPLIT
                    split_tf32(v * 1.0000001f, ah[ii], al[ii]);   // negative control for the test harness: an altered input of the split (deliberately WRONG)
#else
                    split_tf32(v, ah[ii], al[ii]);
#endif
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
                        // W in tile row slot (-> P1), GQ in row slot ^ 8 (-> P2): selects, see phase B
                        const float b1 = slot_hi ? big[tl][2 + e] : big[tl][e], s1 = slot_hi ? sml[tl][2 + e] : sml[tl][e];
                        const float b2 = slot_hi ? big[tl][e] : big[tl][2 + e], s2 = slot_hi ? sml[tl][e] : sml[tl][2 + e];
                        S.p1[c] = fadd(b1, s1);
                        S.p2[c] = fadd(b2, s2);
                    }
            }
        }
        ROWS_SYNC(5);
        ROWS_TS(5);

        // ---- E, F. U = UT - P1, O = P2 (+ M2) then the FMA chain over j = b..t -------------------------------
        if (tid < HEAD_V) {
            const float u = fsub(S.x[0][tid], S.p1[tid]);
            u_row = u;
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
        if (i + 1 < nrows) {   // append this row to the cached-row registers of the next row (same sub-block)
            const int jprev = pos - b;
#pragma unroll
            for (int jj = 0; jj < 16; ++jj) {
                xpre[jj] = (jj == jprev) ? x_row : xpre[jj];
                upre[jj] = (jj == jprev) ? u_row : upre[jj];
            }
        }

        // ---- G. chunk end: E' = E * GEND + dot3(KD, U), P3 over K = 64. Warp w owns the value tiles w and w + 8 for all
        //         8 key tiles; the KD fragments of a key tile are shared by its two value tiles, operands are loaded in
        //         two batches of four k-steps in front of the MMAs --------------------------------------------------
        if (pos == BT - 1) {
            if (tid < HEAD_K) S.gend[tid] = ex2_ftz(S.g2[tid]);
            ROWS_SYNC(6);
            ROWS_DELAY(8);
            for (int mt = 0; mt < 8; ++mt) {
                float big[2][4], sml[2][4], eold[2][4];
#pragma unroll
                for (int tl = 0; tl < 2; ++tl)
#pragma unroll
                    for (int i2 = 0; i2 < 4; ++i2) {
                        big[tl][i2] = neg_zero();
                        sml[tl][i2] = neg_zero();
                        // this thread is the only owner of these E elements: read them now, use them after the MMAs
                        eold[tl][i2] = Eg[(8 * (warp + 8 * tl) + c_col(L, i2)) * HEAD_K + 16 * mt + c_row(L, i2)];
                    }
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    float rkA[4][4], rgA[4][4], ub[4][2][2];
#pragma unroll
                    for (int kl = 0; kl < 4; ++kl) {
                        const int ks = 4 * half + kl;
#pragma unroll
                        for (int ii = 0; ii < 4; ++ii) {
                            const int k = 16 * mt + g + 8 * (ii & 1), j = 8 * ks + t + 4 * (ii >> 1);
                            rkA[kl][ii] = Rk[j * HEAD_K + k];
                            rgA[kl][ii] = Rg[j * HEAD_K + k];
                        }
#pragma unroll
                        for (int tl = 0; tl < 2; ++tl)
#pragma unroll
                            for (int ii = 0; ii < 2; ++ii) ub[kl][tl][ii] = Ru[(8 * ks + t + 4 * ii) * HEAD_V + 8 * (warp + 8 * tl) + g];
                    }
#pragma unroll
                    for (int kl = 0; kl < 4; ++kl) {
                        uint32_t ah[4], al[4];
#pragma unroll
                        for (int ii = 0; ii < 4; ++ii) {
                            const int k = 16 * mt + g + 8 * (ii & 1);
                            split_tf32(fmul(rkA[kl][ii], ex2_ftz(fsub(S.g2[k], rgA[kl][ii]))), ah[ii], al[ii]);
                        }
#pragma unroll
                        for (int tl = 0; tl < 2; ++tl) {
                            uint32_t bh[2], bl[2];
#pragma unroll
                            for (int ii = 0; ii < 2; ++ii) split_tf32(ub[kl][tl][ii], bh[ii], bl[ii]);
                            mma_tf32_m16n8k8(sml[tl], al, bh);
                            mma_tf32_m16n8k8(sml[tl], ah, bl);
                            mma_tf32_m16n8k8(big[tl], ah, bh);
                        }
                    }
                }
#pragma unroll
                for (int tl = 0; tl < 2; ++tl)
#pragma unroll
                    for (int ii = 0; ii < 4; ++ii) {
                        const int k = 16 * mt + c_row(L, ii), v = 8 * (warp + 8 * tl) + c_col(L, ii);
                        Eg[v * HEAD_K + k] = fadd(fmul(eold[tl][ii], S.gend[k]), fadd(big[tl][ii], sml[tl][ii]));
                    }
            }
            pos = 0;
        } else {
            pos += 1;
        }
        // The barrier keeps the next row's prep (it overwrites S.k / S.q / S.g2 ...) and its global loads of what this row stored behind this row's reads and stores. With no next row (decode: always)
        // nothing follows but a global store of thread 0, so the last row has no barrier. The condition is uniform: every thread of the CTA executes the same number of end-of-row barriers.
        if (i + 1 < nrows) ROWS_SYNC_END();
#ifdef CFBI_ROWS_PROFILE
        ROWS_TS(7);
        prof_last = prof_pos;
        if (threadIdx.x == 0 && n * H + h < ROWS_PROF_CTAS) {   // after the last stamp of the row: the buffer write is outside every measured phase
            unsigned smid;
            asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
            unsigned* r = g_rows_prof[prof_pos][n * H + h];
#pragma unroll
            for (int k = 0; k < 8; ++k) r[k] = (unsigned)(tp[k] - prof_entry);
            r[9] = smid + 1;
        }
#endif
    }
    if (DECODE) {
        // single launch: the last CTA of the sequence to arrive has already advanced pos (rows_entry); nothing to store here
#ifdef CFBI_DECODE_TEST_EVERY_CTA_WRITES
        if (tid == 0) p.pos[sp.slot] = (int)(((int64_t)sp.pos0 + sp.len) % BT);   // negative control (tests only): every CTA stores, from the fill it read at its own start
#endif
    } else if (p.fuse_pos && tid == 0) {
        p.pos[ROWS_POS_INDEX(POS_ACCESS_FUSED_STORE, sp.slot)] = (int)(((int64_t)sp.pos0 + sp.len) % BT);   // same value from every head CTA (sp.pos0 is the plan kernel's snapshot)
    }
#ifdef CFBI_ROWS_PROFILE
    if (threadIdx.x == 0 && prof_last >= 0 && n * H + h < ROWS_PROF_CTAS)
        g_rows_prof[prof_last][n * H + h][8] = (unsigned)(clock64() - prof_entry);   // after the last global store of the kernel
#ifdef CFBI_ROWS_PROFILE_PERTURB   // negative control of the bit-identity check of the profile build: one output bit of one CTA differs
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) reinterpret_cast<unsigned short*>(p.out)[(int64_t)sp.tok0 * H * HEAD_K] ^= 1;
#endif
#endif
}

}  // namespace cfbi
