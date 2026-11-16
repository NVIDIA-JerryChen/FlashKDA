// CF-BI KDA: K2 driven by a schedule built on the device (implementation bit IMPL_K2_DYN).
//
// K2 is a serial scan per (sequence, head, value slice): the kernel takes as long as the busiest SM, and one CTA fills an SM (shared memory), so
// the ORDER of the CTAs and the value-tile WIDTH of every (sequence, head) pair decide the time. The host cannot see the sequence lengths (no host
// synchronisation), the device can: k2_sched_kernel (one CTA) reads the plan, chooses the widths and the order and writes one packed int per
// item (pair, value slice, width) into the plan buffer; k2_dyn_kernel is an ordinary 1-D grid whose CTA b runs item b of the table through the
// unchanged k2_tile<BV>. Every (pair, slice) is still one CTA, chunk by chunk, with the same arithmetic: the bits cannot change. The policy is
// specified by the host reference tests/cfbi/k2_policy.py (same integer arithmetic; tests compare the two).
//
// Names matter to the benchmark tooling: the kernels start with "k2" (K2 stage), the K2 proper is "k2_dyn_kernel".
#define CFBI_K2_TILE_ONLY
#include <cub/block/block_scan.cuh>

#include "cfbi/cfbi_k2.cuh"
#include "cfbi/cfbi_k2_dyn.h"

namespace cfbi {

constexpr int SCHED_THREADS = 1024;       // CTA size of the builder for up to K2_SCHED_MAX_SEQUENCES sequences; a call of at most SCHED_THREADS_SMALL sequences uses the small CTA
constexpr int SCHED_THREADS_SMALL = 128;  // (cheaper barriers and scans; it still holds the SCHED_CANDS candidate evaluations)
constexpr int SCHED_MAXN = K2_SCHED_MAX_SEQUENCES;
constexpr int GRID_INTERVALS = 16;                        // split positions x = (i * pairs) / GRID_INTERVALS, i = 0 .. GRID_INTERVALS
constexpr int SCHED_CANDS = 3 + 6 * (GRID_INTERVALS + 1);   // homogeneous 8, 16, 32, then the six mixed width pairs
constexpr int ACCEPT_PCT = 97;                            // a candidate must beat the reference schedule's bound by this percentage ...
constexpr int REORDER_PCT_FEW = 90;                       // ... (with few waves: the old order is replaced only if the sorted order beats it by this)
constexpr int MANY_WAVES = 3;                             // more items than MANY_WAVES * SMs: bounds are compared; else candidates must be certified
constexpr int ITEM_NONE = -1;                             // table entry beyond the item count

// Time of one CTA per 64-token chunk at the tile width code 0 / 1 / 2 = 8 / 16 / 32 columns, units of 0.1 K cycles: the measurement of the earlier K2
// work on B200 / B300 (docs/20261103-cfbi-perf.md; 3.4 : 4.7 : 7.8 K cycles, the constants of k2_grid_cost in cfbi_launch.cu). Not calibrated for Hopper.
__host__ __device__ __forceinline__ long long k2_chunk_time(int wc) { return wc == 0 ? 34 : (wc == 1 ? 47 : 78); }
__host__ __device__ __forceinline__ int k2_items_per_pair(int wc) { return 16 >> wc; }

// ---- the closed form of the fixed-length launcher, on the device ------------------------------------------------------------------------------------
// Busiest SM's time per chunk (units of 0.1 K cycles) for nh equal pairs, the first x32 with 32-column tiles, the rest bvb-column tiles (8 or 16);
// identical to k2_grid_cost in cfbi_launch.cu and to grid_cost of tests/cfbi/k2_policy.py.
__device__ __forceinline__ long long grid_cost_dev(long long nh, long long x32, int bvb, long long sms_ll) {
    // every quantity stays below 2^32 for the capacity of the builder (at most 16896 pairs: 4 * x32 <= 67584, times <= 78 * (67584 / sms + 135168)),
    // so the divisions are 32-bit (64-bit divisions are emulated and slow); products that can exceed 2^32 are formed in 64 bit
    const unsigned sms = static_cast<unsigned>(sms_ll);
    const unsigned tl = 78, ts = bvb == 8 ? 34 : (bvb == 16 ? 47 : 78);
    const unsigned nl = 4u * static_cast<unsigned>(x32), ns = static_cast<unsigned>(nh - x32) * static_cast<unsigned>(HEAD_V / bvb);
    const unsigned a = nl / sms, r = nl - a * sms;
    const unsigned f0 = a * tl, f1 = (a + 1) * tl;
    const unsigned busy_long = r > 0 ? f1 : f0;
    if (ns == 0) return busy_long;
    unsigned lo = f0, hi = f1 + ns * ts;
    while (lo < hi) {
        const unsigned mid = (lo + hi) >> 1;
        const unsigned long long slots = static_cast<unsigned long long>(sms - r) * (mid >= f0 ? (mid - f0) / ts + 1 : 0u) +
                                         static_cast<unsigned long long>(r) * (mid >= f1 ? (mid - f1) / ts + 1 : 0u);
        if (slots >= ns) hi = mid; else lo = mid + 1;
    }
    const unsigned last_end = lo + ts;
    return last_end > busy_long ? last_end : busy_long;
}

struct SchedShared {
    int cs[SCHED_MAXN];            // chunk count per sequence (0: inactive)
    int csr[SCHED_MAXN + 1];       // chunk counts in rank order (c descending, sequence ascending); csr[N] = 0
    int srt[SCHED_MAXN];           // sequence of rank j
    int lead[SCHED_MAXN];          // first rank of the group of equal chunk count that contains rank j
    int gsz[SCHED_MAXN];           // size of that group
    int actp[SCHED_MAXN + 1];      // active sequences before sequence n (index order)
    int actseq[SCHED_MAXN];        // the k-th active sequence (index order)
    long long pre[SCHED_MAXN + 1]; // exclusive prefix sums of csr over the ranks
    long long cub[SCHED_CANDS];    // upper bound of candidate q
    long long cpig[SCHED_CANDS];   // lower bound of candidate q
    unsigned long long red;
    long long hcost[3];
    int nact, kind, wa, wb, x;
    long long refW, refdmax;
};

// chunk count of the pair at position i of the sorted pair list: the sorted list repeats the rank-sorted sequence list H times per rank
__device__ __forceinline__ int c_at(const SchedShared& sh, int H, int i) { return sh.csr[i / H]; }

// sum of the chunk counts of the first x pairs of the sorted pair list
__device__ __forceinline__ long long c_prefix(const SchedShared& sh, int H, int x) {
    const int j0 = x / H;
    if (j0 >= sh.nact) return (long long)H * sh.pre[sh.nact];
    const int gt = sh.lead[j0];
    return (long long)H * sh.pre[gt] + (long long)(x - gt * H) * sh.csr[j0];
}

// pair id p = h * N + n of the pair at position i of the sorted pair list (c descending, p ascending)
__device__ __forceinline__ int pair_at(const SchedShared& sh, int N, int H, int i) {
    const int j0 = i / H;
    const int gt = sh.lead[j0], g = sh.gsz[j0];
    const int off = i - gt * H;
    const int h = off / g, r = off - h * g;
    return h * N + sh.srt[gt + r];
}

// ---- candidates ------------------------------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void cand_decode(int q, int pairs, int& wa, int& wb, int& x) {
    if (q < 3) { wa = wb = q; x = 0; return; }
    const int m = q - 3, pi = m / (GRID_INTERVALS + 1), i = m % (GRID_INTERVALS + 1);
    // (a, b) over a != b in the order (8, 16) (8, 32) (16, 8) (16, 32) (32, 8) (32, 16)
    wa = pi < 2 ? 0 : (pi < 4 ? 1 : 2);
    wb = pi == 0 ? 1 : (pi == 1 ? 2 : (pi == 2 ? 0 : (pi == 3 ? 2 : (pi == 4 ? 0 : 1))));
    x = static_cast<int>((static_cast<long long>(i) * pairs) / GRID_INTERVALS);
}

// W, longest item, S-th and (S+1)-th longest item of the candidate (first x pairs width code wa, the others wb); the last two are 0 when the
// candidate has fewer than S (S + 1) items. The items of the two groups form two non-increasing lists (A: x * ma items, B: (pairs - x) * mb items) whose
// entries are read directly from the rank-sorted chunk counts, so the S-th largest of their union is a binary search on how many of the top S come from A.
__device__ void cand_stats(const SchedShared& sh, int H, int pairs, int S, int wa, int wb, int x, long long& W, long long& dmax,
                           long long& dS, long long& dS1) {
    const long long ta = k2_chunk_time(wa), tb = k2_chunk_time(wb);
    const int ma = k2_items_per_pair(wa), mb = k2_items_per_pair(wb);
    const long long call = c_prefix(sh, H, pairs), ca = c_prefix(sh, H, x);
    W = ma * ta * ca + mb * tb * (call - ca);
    dmax = max(x > 0 ? ta * sh.csr[0] : 0LL, x < pairs ? tb * c_at(sh, H, x) : 0LL);
    // item counts and indices are below 16 * 16896: 32-bit arithmetic and divisions
    const int LA = ma * x, LB = mb * (pairs - x);
    const unsigned dA = static_cast<unsigned>(ma * H), dB = static_cast<unsigned>(mb * H);
    const long long INF = 0x7fffffffffffffffLL;
    auto A = [&](int r) -> long long { return r < 0 ? INF : (r >= LA ? 0 : ta * sh.csr[static_cast<unsigned>(r) / dA]); };
    auto B = [&](int r) -> long long { return r < 0 ? INF : (r >= LB ? 0 : tb * sh.csr[static_cast<unsigned>(x * mb + r) / dB]); };
    if (LA + LB < S) { dS = 0; dS1 = 0; return; }
    int lo = S - LB > 0 ? S - LB : 0, hi = S < LA ? S : LA;
    while (lo < hi) {   // largest i with A[i - 1] >= B[S - i]: i of the S longest items come from A
        const int i = (lo + hi + 1) >> 1;
        if (A(i - 1) >= B(S - i)) lo = i; else hi = i - 1;
    }
    const int i = lo, j = S - lo;
    const long long a1 = A(i - 1), b1 = B(j - 1);
    dS = a1 < b1 ? a1 : b1;
    const long long an = A(i), bn = B(j);
    dS1 = (LA + LB < S + 1) ? 0 : (an > bn ? an : bn);
}

// floor((S - 1) * d / S) = d - ceil(d / S), by a 32-bit division when d fits (d below 78 * 2^25)
__device__ __forceinline__ long long scaled_sm1(long long d, int S) {
    if (d < (1LL << 32) - 65536) return d - static_cast<long long>((static_cast<unsigned>(d) + static_cast<unsigned>(S) - 1u) / static_cast<unsigned>(S));
    return ((S - 1) * d) / S;
}
__device__ __forceinline__ long long bound_ub(long long S, long long W, long long dmax, long long dS1) {
    const long long q = W / S;
    const long long v = q + (W - q * S != 0) + scaled_sm1(dS1, static_cast<int>(S));
    return v > dmax ? v : dmax;
}
__device__ __forceinline__ long long bound_pig(long long S, long long W, long long dmax, long long dS, long long dS1) {
    const long long q = W / S;
    long long v = q + (W - q * S != 0);
    if (dS + dS1 > v) v = dS + dS1;
    return v > dmax ? v : dmax;
}

// The closed-form choice of the fixed-length launcher for nh equal pairs, evaluated by the whole block (the mixed-grid candidates in parallel).
// Returns the same (x32, bvb) as choose_k2_tiles of cfbi_launch.cu: the first x32 pairs 32-column tiles, the others bvb columns. All threads call it.
__device__ void choose_equal_block(SchedShared& sh, long long nh, int sms, int& x32_out, int& bvb_out) {
    if (threadIdx.x == 0) sh.red = ~0ull;
    if (threadIdx.x >= 32 && threadIdx.x < 35) sh.hcost[threadIdx.x - 32] = threadIdx.x == 32 ? grid_cost_dev(nh, 0, 8, sms) : (threadIdx.x == 33 ? grid_cost_dev(nh, 0, 16, sms) : grid_cost_dev(nh, nh, 16, sms));
    __syncthreads();
    unsigned long long loc = ~0ull;
    for (long long k = 1 + threadIdx.x; 4 * nh > k * sms; k += blockDim.x) {
        long long x = k * sms / 4;
        if (x > nh - 1) x = nh - 1;
        if (x >= 1) {
            const unsigned long long key = (static_cast<unsigned long long>(grid_cost_dev(nh, x, 16, sms)) << 24) | static_cast<unsigned long long>(k);
            if (key < loc) loc = key;
        }
    }
    if (loc != ~0ull) atomicMin(&sh.red, loc);
    __syncthreads();
    long long best = sh.hcost[0];
    int bx = 0, bw = 8;
    if (sh.hcost[1] < best) { best = sh.hcost[1]; bx = 0; bw = 16; }
    if (sh.hcost[2] < best) { best = sh.hcost[2]; bx = static_cast<int>(nh); bw = 16; }
    if (sh.red != ~0ull) {
        const long long c = static_cast<long long>(sh.red >> 24), k = static_cast<long long>(sh.red & 0xFFFFFFull);
        long long x = k * sms / 4;
        if (x > nh - 1) x = nh - 1;
        if (c < best && x > 0 && c * 100 <= best * 97) { bx = static_cast<int>(x); bw = 16; }
    }
    x32_out = bx;
    bvb_out = bw;
    __syncthreads();
}

// does the pair at sorted position j (width code wc) come before the pair (key, p) in the total order: key descending, pair id ascending
__device__ __forceinline__ bool precedes(const SchedShared& sh, int N, int H, int j, int wc, long long key, int p) {
    const long long kj = static_cast<long long>(c_at(sh, H, j)) * k2_chunk_time(wc);
    return kj > key || (kj == key && pair_at(sh, N, H, j) < p);
}

// ---- the schedule builder: one CTA ------------------------------------------------------------------------------------------------------------------
// plan: the (N + 1) SeqPlan entries of the call; sched: the schedule region behind them (K2_SCHED_HEADER_INTS + 16 * N * H ints); S: SMs; fw: 0 = policy,
// 8 / 16 / 32 = forced width of every pair; call_fixed: no cu_seqlens (the old schedule is then the closed form of the fixed-length launcher over all N * H pair
// positions, and the schedule of a list of unequal active chunk counts, which a fixed-length call has with warm positions and skipped slots, is that schedule).
template <int THREADS>
__global__ void __launch_bounds__(THREADS) k2_sched_kernel(const SeqPlan* plan, int N, int H, int S, int fw, int call_fixed, int* sched) {
    __shared__ SchedShared sh;
    __shared__ typename cub::BlockScan<unsigned long long, THREADS>::TempStorage scan64;
    const int tid = threadIdx.x;

    // chunk counts and the rank of every sequence (c descending, sequence index ascending) by counting
    int cn = 0;
    if (tid < N) {
        const SeqPlan sp = plan[tid];
        cn = sp.nfull + (sp.tail > 0 ? 1 : 0);
        sh.cs[tid] = cn;
    }
    __syncthreads();
    const int nact = __syncthreads_count(tid < N && cn > 0);
    if (tid < N) {
        int gt = 0, eq = 0, lt = 0;
        for (int m = 0; m < N; ++m) {
            const int cm = sh.cs[m];
            gt += cm > cn;
            eq += cm == cn;
            lt += (cm == cn) & (m < tid);
        }
        const int rank = gt + lt;
        sh.srt[rank] = tid;
        sh.csr[rank] = cn;
        sh.lead[rank] = gt;
        sh.gsz[rank] = eq;
    }
    if (tid == 0) sh.csr[N] = 0;
    __syncthreads();
    // one scan for both prefix sums: the high part = chunk counts in rank order (pre), the low 11 bits = active flags in sequence order (actp)
    const int act = (tid < N && cn > 0) ? 1 : 0;
    unsigned long long ex, tot;
    cub::BlockScan<unsigned long long, THREADS>(scan64).ExclusiveSum(
        ((tid < nact ? static_cast<unsigned long long>(sh.csr[tid]) : 0ull) << 11) | static_cast<unsigned long long>(act), ex, tot);
    if (tid < N) {
        sh.pre[tid] = static_cast<long long>(ex >> 11);
        sh.actp[tid] = static_cast<int>(ex & 2047u);
        if (act) sh.actseq[ex & 2047u] = tid;
    }
    if (tid == 0) {
        sh.pre[N] = static_cast<long long>(tot >> 11);
        sh.actp[N] = static_cast<int>(tot & 2047u);
    }
    if (tid == 0) sh.nact = nact;
    __syncthreads();
    const int Pa = nact * H;

    // the decision: kind 0 = the old schedule in natural order (the first x active pairs width code wa, the others wb: x = 0 and one width for a variable-length
    // call, the closed form of the fixed-length launcher over all N * H pair positions for a fixed-length call), kind 1 = sorted (first x pairs width code wa,
    // the others wb)
    int label = 1;   // kind written to the header (an equal-length schedule that is the old schedule reports 0; its table is built by the sorted writer)
    bool natural = false;
    const int w0 = (static_cast<long long>(N) * H * 16 <= S) ? 8 : 16;
    if (Pa == 0) {
        if (tid == 0) { sh.kind = 0; sh.wa = sh.wb = 0; sh.x = 0; }
        label = 0;
    } else if (fw != 0) {
        if (tid == 0) { sh.wa = sh.wb = fw == 8 ? 0 : (fw == 16 ? 1 : 2); sh.x = 0; }
    } else if (sh.csr[0] == sh.csr[nact - 1]) {   // equal chunk counts: the closed form is exact
        int x32, bvb;
        choose_equal_block(sh, Pa, S, x32, bvb);
        int x32o = 0, bvbo = w0;
        if (call_fixed) {
            if (nact == N) { x32o = x32; bvbo = bvb; }
            else choose_equal_block(sh, static_cast<long long>(N) * H, S, x32o, bvbo);
        }
        // active pairs among the first x32o pair positions p = h * N + n
        const int hq = x32o / N, nq = x32o % N;
        const int ax = hq * nact + sh.actp[nq];
        if (tid == 0) { sh.wa = 2; sh.wb = bvb == 8 ? 0 : (bvb == 16 ? 1 : 2); sh.x = x32; }
        label = (x32 == ax && bvb == bvbo) ? 0 : 1;
    } else if (call_fixed) {
        // A fixed-length call with unequal active chunk counts (warm open-chunk positions of the sequences; skipped slots are absent): the exact old schedule of
        // the call type, the closed form of the fixed-length launcher over all N * H pair positions in natural order, 32 columns for the pairs at positions
        // p < x32o and bvbo columns for the others (k2_policy.build_schedule). The candidate search below is for variable-length calls.
        int x32o, bvbo;
        choose_equal_block(sh, static_cast<long long>(N) * H, S, x32o, bvbo);
        const int ax = (x32o / N) * nact + sh.actp[x32o % N];   // active pairs among the first x32o pair positions p = h * N + n
        if (tid == 0) { sh.kind = 0; sh.wa = 2; sh.wb = bvbo == 8 ? 0 : (bvbo == 16 ? 1 : 2); sh.x = ax; }
        natural = true;
        label = 0;
    } else {
        // candidates of an unequal list of a variable-length call
        if (tid < SCHED_CANDS) {
            int wa, wb, x;
            cand_decode(tid, Pa, wa, wb, x);
            long long W, dmax, dS, dS1;
            cand_stats(sh, H, Pa, S, wa, wb, x, W, dmax, dS, dS1);
            sh.cub[tid] = bound_ub(S, W, dmax, dS1);
            sh.cpig[tid] = bound_pig(S, W, dmax, dS, dS1);
            if (tid == (w0 == 8 ? 0 : 1)) { sh.refW = W; sh.refdmax = dmax; }
        }
        __syncthreads();
        const int refq = w0 == 8 ? 0 : 1;
        const long long ub_ref = sh.cub[refq], pig_ref = sh.cpig[refq];
        const bool many = static_cast<long long>(Pa) * (16 >> refq) > static_cast<long long>(MANY_WAVES) * S;
        if (tid < 32) {   // best accepted candidate: (lower bound, upper bound, index) ascending
            long long bp = 0x7fffffffffffffffLL, bu = 0x7fffffffffffffffLL;
            int bq = SCHED_CANDS;
            for (int q = tid; q < SCHED_CANDS; q += 32) {
                const long long u = sh.cub[q], p = sh.cpig[q];
                const bool ok = many ? (100 * u <= ACCEPT_PCT * ub_ref) : (u <= pig_ref);
                if (ok && (p < bp || (p == bp && (u < bu || (u == bu && q < bq))))) { bp = p; bu = u; bq = q; }
            }
            for (int off = 16; off > 0; off >>= 1) {
                const long long op = __shfl_xor_sync(0xffffffffu, bp, off), ou = __shfl_xor_sync(0xffffffffu, bu, off);
                const int oq = __shfl_xor_sync(0xffffffffu, bq, off);
                if (op < bp || (op == bp && (ou < bu || (ou == bu && oq < bq)))) { bp = op; bu = ou; bq = oq; }
            }
            if (tid == 0) {
                int wa, wb, x;
                if (bq < SCHED_CANDS) {
                    cand_decode(bq, Pa, wa, wb, x);
                    sh.kind = 1;
                } else {
                    const long long ub_old = (sh.refW + S - 1) / S + ((S - 1) * sh.refdmax) / S;   // Graham bound of the old order (same items as the reference)
                    if (100 * ub_ref <= (many ? ACCEPT_PCT : REORDER_PCT_FEW) * ub_old) {
                        wa = wb = refq; x = 0;
                        sh.kind = 1;
                    } else {
                        wa = wb = refq; x = 0;
                        sh.kind = 0;   // the old schedule
                    }
                }
                sh.wa = wa; sh.wb = wb; sh.x = x;
            }
        }
        __syncthreads();
        natural = sh.kind == 0;
        label = sh.kind;
    }
    __syncthreads();
    const int wa = sh.wa, wb = sh.wb, x = sh.x;
    const int ma = k2_items_per_pair(wa), mb = k2_items_per_pair(wb);
    const long long count = static_cast<long long>(ma) * x + static_cast<long long>(mb) * (Pa - x);
    int* items = sched + K2_SCHED_HEADER_INTS;
    const long long cap = static_cast<long long>(K2_SCHED_ITEMS_PER_PAIR) * N * H;

    if (Pa > 0 && natural) {
        // the old schedule: the active pairs in natural order (head, then sequence index); the first x of them (the pairs at positions below x32o of a
        // fixed-length call, none for a variable-length call) at width wa, the others at width wb
        for (int idx = tid; idx < Pa; idx += THREADS) {
            const int h = idx / nact, ia = idx - h * nact;
            const int p = h * N + sh.actseq[ia];
            const bool in_a = idx < x;
            const int wc = in_a ? wa : wb, m = in_a ? ma : mb;
            const int off = in_a ? idx * ma : x * ma + (idx - x) * mb;
            for (int s = 0; s < m; ++s) items[off + s] = k2_item_pack(p, s, wc);
        }
    } else if (Pa > 0) {
        const long long ta = k2_chunk_time(wa), tb = k2_chunk_time(wb);
        for (int i = tid; i < Pa; i += THREADS) {
            const bool in_a = i < x;
            const int wc = in_a ? wa : wb;
            const int p = pair_at(sh, N, H, i);
            const long long key = static_cast<long long>(c_at(sh, H, i)) * (in_a ? ta : tb);
            int off;
            if (in_a) {                       // B pairs that precede this pair in the total order
                int lo = x, hi = Pa;
                while (lo < hi) {
                    const int mid = (lo + hi) >> 1;
                    if (precedes(sh, N, H, mid, wb, key, p)) lo = mid + 1; else hi = mid;
                }
                off = i * ma + (lo - x) * mb;
            } else {                          // A pairs that precede it
                int lo = 0, hi = x;
                while (lo < hi) {
                    const int mid = (lo + hi) >> 1;
                    if (precedes(sh, N, H, mid, wa, key, p)) lo = mid + 1; else hi = mid;
                }
                off = (i - x) * mb + lo * ma;
            }
            const int m = k2_items_per_pair(wc);
            for (int s = 0; s < m; ++s) items[off + s] = k2_item_pack(p, s, wc);
        }
    }
    // the table beyond the item count holds a sentinel (a second line of defence: the K2 CTAs b >= count exit before they read the table, and it is what the
    // no-range-check test variant would read)
#ifdef CFBI_DYN_TEST_PAD_ITEM_ZERO   // negative control (tests only): the entries beyond the count point at a VALID item (item 0)
    __syncthreads();
    const int pad_value = count > 0 ? items[0] : ITEM_NONE;
#else
    const int pad_value = ITEM_NONE;
#endif
    for (long long t = count + tid; t < cap; t += THREADS) items[t] = pad_value;
    if (tid < K2_SCHED_HEADER_INTS) {
        int v = 0;
        if (tid == K2_SCHED_COUNT) v = static_cast<int>(count);
        else if (tid == K2_SCHED_KIND) v = label;
        else if (tid == K2_SCHED_PAIRS) v = Pa;
        else if (tid == K2_SCHED_CHOICE_A) v = (natural && x == 0) ? 0 : (8 << wa);
        else if (tid == K2_SCHED_CHOICE_B) v = 8 << wb;
        else if (tid == K2_SCHED_CHOICE_X) v = x;
        else if (tid == K2_SCHED_SEQUENCES) v = nact;
        sched[tid] = v;
    }
}

// ---- the K2 kernel ---------------------------------------------------------------------------------------------------------------------------------
// CTA b runs item b of the table (sched == nullptr: the old schedule decoded in closed form, 16 slices of 8 columns per pair in natural order). All CTAs
// get the shared memory of the widest tile; a CTA runs only the body of its own width. CTAs beyond the item count return before any shared-memory or
// barrier access and before they read an entry of the table (the only global access is the item count of the table header).
#ifdef CFBI_DYN_TEST_START_LOG   // start-order probe (tests only): when and where every CTA that runs an item starts
constexpr int START_LOG_MAX = 1 << 19;
__device__ unsigned long long g_dyn_start_time[START_LOG_MAX];
__device__ unsigned g_dyn_start_smid[START_LOG_MAX];
#endif

#ifdef CFBI_DYN_TEST_PERMUTE_SEED
// Test hook (tests only): CTA b runs item perm(b), a bijection of 0 .. count - 1 given by the seed. Whole items keep their serial chunk order inside
// the CTA; only the order in which CTAs take items changes. With CFBI_DYN_TEST_PERMUTE_BREAK the map is NOT a bijection (some items run twice, some
// never): the negative control that shows the hook is live.
__device__ __forceinline__ int test_gcd(int a, int b) {
    while (b != 0) { const int t = a % b; a = b; b = t; }
    return a;
}
__device__ __forceinline__ int test_permute(int b, int count) {
    if (count <= 1) return 0;
    const unsigned seed = CFBI_DYN_TEST_PERMUTE_SEED;
    int a = 3 + 2 * static_cast<int>((seed * 7u) % 11u);
    while (test_gcd(a, count) != 1) a += 2;
    const int c = static_cast<int>((seed * 2654435761u) % static_cast<unsigned>(count));
#ifdef CFBI_DYN_TEST_PERMUTE_BREAK
    return static_cast<int>((static_cast<long long>(a) * b + c) % (count - 1));
#else
    return static_cast<int>((static_cast<long long>(a) * b + c) % count);
#endif
}
#endif

__global__ void __launch_bounds__(NTHREADS, 1) k2_dyn_kernel(const Params p, const int* sched) {
    const int b = static_cast<int>(blockIdx.x);
    int pair, slice, wc;
    if (sched != nullptr) {
        const int count = sched[K2_SCHED_COUNT];
#ifndef CFBI_DYN_TEST_NO_RANGE_CHECK
        if (b >= count) return;         // CTA-uniform range check: before any entry of the table is read (the entries beyond the count hold a sentinel as well)
#endif
        int src = b;
#ifdef CFBI_DYN_TEST_PERMUTE_SEED
        if (b < count) src = test_permute(b, count);
#endif
#ifdef CFBI_DYN_TEST_DUP_ITEM       // negative control (tests only): CTA k takes the entry of CTA k + 1: item k is lost, item k + 1 runs twice
        if (b == CFBI_DYN_TEST_DUP_ITEM && count > 1) src = (b + 1) % count;
#endif
        const int it = sched[K2_SCHED_HEADER_INTS + src];
#ifdef CFBI_DYN_TEST_SKIP_ITEM      // negative control (tests only): CTA k does nothing: item k is lost
        if (b == CFBI_DYN_TEST_SKIP_ITEM) return;
#endif
#ifndef CFBI_DYN_TEST_NO_RANGE_CHECK
        if (it < 0) return;             // an entry inside the count that holds the sentinel (a table that was not written): run nothing
#else                               // negative control (tests only): no range check; the builder pads with a valid item, the late CTAs wait until it is done
        if (b >= count) {
            for (int i = 0; i < CFBI_DYN_TEST_EXTRA_DELAY_MS; ++i) __nanosleep(1000000);
        }
#endif
        pair = it & ((1 << K2_ITEM_SLICE_SHIFT) - 1);
        slice = (it >> K2_ITEM_SLICE_SHIFT) & 15;
        wc = (it >> K2_ITEM_WIDTH_SHIFT) & 3;
    } else {
        pair = b >> 4;
        slice = b & 15;
        wc = 0;
    }
#ifdef CFBI_DYN_TEST_START_LOG
    if (threadIdx.x == 0 && b < START_LOG_MAX) {
        unsigned long long t;
        unsigned sm;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
        g_dyn_start_time[b] = t;
        g_dyn_start_smid[b] = sm;
    }
#endif
#ifdef CFBI_DYN_TEST_DELAY_NS       // stress (tests only): every CTA lingers for 0 .. 3 x the delay before it starts its item
    __nanosleep(static_cast<unsigned>(CFBI_DYN_TEST_DELAY_NS) * ((static_cast<unsigned>(b) * 2654435761u) >> 30));
#endif
    const int n = pair % p.N, h = pair / p.N;
    if (wc == 0) k2_tile<8>(p, slice, n, h);
    else if (wc == 1) k2_tile<16>(p, slice, n, h);
    else k2_tile<32>(p, slice, n, h);
}

// ---- host entry points -----------------------------------------------------------------------------------------------------------------------------
int k2_dyn_test_log(unsigned long long* times, unsigned* smids, int n) {
#ifdef CFBI_DYN_TEST_START_LOG
    if (n > START_LOG_MAX) n = START_LOG_MAX;
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(times, g_dyn_start_time, sizeof(unsigned long long) * n);
    cudaMemcpyFromSymbol(smids, g_dyn_start_smid, sizeof(unsigned) * n);
    return n;
#else
    (void)times; (void)smids; (void)n;
    return 0;
#endif
}

void k2_sched_build(SeqPlan* plan, int N, int H, int sms, int k2_bv, bool call_fixed, cudaStream_t stream) {
    int* sched = reinterpret_cast<int*>(plan) + static_cast<int64_t>(N + 1) * PLAN_INTS_PER_SEQ;
    if (N <= SCHED_THREADS_SMALL) k2_sched_kernel<SCHED_THREADS_SMALL><<<1, SCHED_THREADS_SMALL, 0, stream>>>(plan, N, H, sms, k2_bv, call_fixed ? 1 : 0, sched);
    else k2_sched_kernel<SCHED_THREADS><<<1, SCHED_THREADS, 0, stream>>>(plan, N, H, sms, k2_bv, call_fixed ? 1 : 0, sched);
}

void launch_k2_dyn(const Params& p, const ForwardArgs& a, int sms, cudaStream_t stream) {
    const int64_t N = a.N, H = a.H;
    // Everything fits one wave at 8 columns (the old rule's choice): all CTAs start at once, nothing to order or to choose; the kernel decodes the old
    // schedule itself and no builder runs. A forced width always builds a schedule (the device ordering is part of what it forces).
    const bool trivial = a.k2_bv == 0 && N * H * 16 <= sms;
    const int* sched = nullptr;
    if (!trivial) {
        k2_sched_build(p.plan, a.N, a.H, sms, a.k2_bv, a.cu_seqlens == nullptr, stream);
        sched = reinterpret_cast<const int*>(p.plan) + (N + 1) * PLAN_INTS_PER_SEQ;
    }
    cudaFuncSetAttribute(k2_dyn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, K2Cfg<32>::SMEM_BYTES);
    k2_dyn_kernel<<<static_cast<unsigned>(N * H * K2_SCHED_ITEMS_PER_PAIR), NTHREADS, K2Cfg<32>::SMEM_BYTES, stream>>>(p, sched);
}

}  // namespace cfbi
