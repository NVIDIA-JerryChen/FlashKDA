// Host-side declarations of the CF-BI launchers (plain C++: the definitions live in cfbi_launch.cu /
// cfbi_probe.cu and are called from cfbi_binding.cpp).
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace cfbi {

int version();

// Workspace layout between K1 and K2, in floats per (chunk, head):
//   WG [128][128] (W rows then GQ rows) | UT [64][128] | Aqk [64][64] | KD [128][64] | GEND [128]
constexpr int64_t WS_FLOATS_PER_CHUNK_HEAD = 128 * 128 + 64 * 128 + 64 * 64 + 128 * 64 + 128;  // 36992
constexpr int PLAN_INTS_PER_SEQ = 8;   // sizeof(SeqPlan) / sizeof(int)

// The plan buffer of a call also holds the schedule of the dynamic-schedule K2 (IMPL_K2_DYN, cfbi_k2_dyn.cu) behind the (N + 1) SeqPlan entries:
// K2_SCHED_HEADER_INTS header ints (K2_SCHED_* fields), then one packed int per item (a (sequence, head) pair and value slice, k2_item_pack), at most
// K2_SCHED_ITEMS_PER_PAIR items per pair (16 slices of 8 columns). The builder handles up to K2_SCHED_MAX_SEQUENCES sequences and
// K2_SCHED_MAX_PAIRS (sequence, head) pairs per call; larger calls run the old K2 kernels (the host knows N and H without a synchronisation).
constexpr int K2_SCHED_HEADER_INTS = 16;
constexpr int K2_SCHED_ITEMS_PER_PAIR = 16;
constexpr int K2_SCHED_MAX_SEQUENCES = 1024;
constexpr int K2_SCHED_MAX_PAIRS = 16896;   // 512 sequences of 33 heads
// most sequences the schedule builder takes for H heads (0 for H < 1)
inline int64_t k2_sched_capacity(int64_t H) {
    return H < 1 ? 0 : (K2_SCHED_MAX_PAIRS / H < K2_SCHED_MAX_SEQUENCES ? K2_SCHED_MAX_PAIRS / H : K2_SCHED_MAX_SEQUENCES);
}
// ints of the schedule behind the SeqPlan entries for a call of N sequences and H heads (0: beyond the capacity, no schedule)
inline int64_t k2_sched_ints(int64_t N, int64_t H) {
    return N >= 1 && N <= k2_sched_capacity(H) ? K2_SCHED_HEADER_INTS + K2_SCHED_ITEMS_PER_PAIR * N * H : 0;
}
// ints of the whole plan buffer of a call (what workspace_sizes reports)
inline int64_t plan_ints(int64_t N, int64_t H) { return (N + 1) * PLAN_INTS_PER_SEQ + k2_sched_ints(N, H); }
// header fields of the schedule (ints)
constexpr int K2_SCHED_COUNT = 0;      // number of items: the CTAs b >= count of the K2 kernel exit at once
constexpr int K2_SCHED_KIND = 1;       // 0: the old schedule (natural order: pair, then value slice, ascending), 1: sorted (longest CTA first, see cfbi_k2_dyn.cu)
constexpr int K2_SCHED_PAIRS = 2;      // active (sequence, head) pairs
constexpr int K2_SCHED_CHOICE_A = 3;   // sorted candidate: width of the first CHOICE_X pairs of the sorted pair list (the longest sequences); the old schedule of a
                                       // fixed-length call with unequal chunk counts: width of its first CHOICE_X active pairs in natural order (32 columns; 0 when there is none)
constexpr int K2_SCHED_CHOICE_B = 4;   // width of the other pairs
constexpr int K2_SCHED_CHOICE_X = 5;
constexpr int K2_SCHED_SEQUENCES = 6;  // sequences with at least one chunk
// a packed item: pair p = h * N + n (bits 0..24), value slice (bits 25..28), width code (bits 29..30: 0 = 8, 1 = 16, 2 = 32 columns)
constexpr int K2_ITEM_SLICE_SHIFT = 25;
constexpr int K2_ITEM_WIDTH_SHIFT = 29;
#ifdef __CUDACC__
#define CFBI_HOST_DEVICE __host__ __device__
#else
#define CFBI_HOST_DEVICE
#endif
CFBI_HOST_DEVICE inline int k2_item_pack(int pair, int slice, int width_code) {
    return pair | (slice << K2_ITEM_SLICE_SHIFT) | (width_code << K2_ITEM_WIDTH_SHIFT);
}

// Everything one forward call needs, with raw pointers (bf16 tensors as void*).
struct ForwardArgs {
    const void* q;     // bf16 [T_total, H, 128]
    const void* k;     // bf16 [T_total, H, 128]
    const void* v;     // bf16 [T_total, H, 128]
    const void* g;     // bf16 [T_total, H, 128]
    const void* beta;  // bf16 [T_total, H]  (raw logits)
    void* out;         // bf16 [T_total, H, 128]
    float* out_f32;    // optional fp32 copy of the output before the bf16 rounding (tests / accuracy), or null
    const float* A_log;    // [H]
    const float* dt_bias;  // [H, 128]
    float scale;
    float scale2;      // fp32(lower_bound * log2 e), computed in double
    int H;
    int N;             // number of sequences
    int fixed_len;     // tokens per sequence when cu_seqlens == null
    int64_t T_total;
    const int64_t* cu_seqlens;  // [N+1] or null
    float* E;          // [S,H,128,128] fp32 state pool, or null (training: zero initial state, no write-back)
    float* rows;       // [S,H,5,64,128] fp32 open-chunk row cache, or null
    int* pos;          // [S] int32 fill of the open chunk per state slot, or null
    const int* slot_ids;   // [N] int32 state slot of each sequence of the call (-1: skip the entry), or null (slot = n)
    int* plan;         // scratch [(N+1) * PLAN_INTS_PER_SEQ] ints, plus the schedule ints of k2_sched_ints(N, H) when the dynamic-schedule K2 runs
    float* ws;         // scratch [(ceil(T_total/64) + N) * H * WS_FLOATS_PER_CHUNK_HEAD] floats (mode 0 only)
    int mode;          // 0: prefill / training schedule (row engine for head rows, K1+K2 for the rest)
                       // 1: row engine for every row (decode; also a differential test of K1/K2); the caller maps
                       //    mode 0 + IMPL_ROWS_ONLY to 1 (resolve_mode) and supplies a state (E, rows, pos) then
    int phase;         // PHASE_* bits (3 = everything in one call)
    int k2_bv;         // value columns per K2 CTA: 0 = choose from the grid size (a homogeneous width or a 32/16 mix), 8, 16 or 32 = forced
                       // (tests); 1000 + x = the first x (sequence, head) pairs with 32-column tiles, the rest with 16 (tests)
    int impl;          // IMPL_* bits as returned by resolve_impl (the launcher does not re-resolve)
};

// Implementation selector: the frozen v1 kernels in csrc/cfbi/ref/ are the bitwise oracle of the optimised ones.
// K1 and K2 share one bit because their workspace layouts differ between the implementations. 0 = the optimised
// kernels where the device can run them (see resolve_impl: otherwise the reference kernels are used).
constexpr int IMPL_REF_ROWS = 1;    // row engine (decode, head rows)
constexpr int IMPL_REF_CHUNK = 2;   // K1 + K2 (prefill / training forward)
constexpr int IMPL_ROWS_ONLY = 4;   // no K1/K2: every row goes through the row engine (mode 1 semantics even for mode 0)
constexpr int IMPL_K2_UMMA = 8;     // optimised K1<16> + the tcgen05 (UMMA) K2 (Blackwell sm_100/103 only; ignored with IMPL_REF_CHUNK)
// optimised K1<16> + a K2 driven by a schedule that a builder kernel makes on the device from the plan (cfbi_k2_dyn.cu): the order of the K2 CTAs
// (longest sequence first) and the value-tile width of every (sequence, head) pair (8, 16 or 32 columns) follow the sequence lengths, which the host
// does not know (no host synchronisation). Bit-identical to the other K2 kernels (every (pair, value slice) is still one CTA, chunk by chunk, with
// the same arithmetic). Requested explicitly it runs for every call; impl 0 sets it only for the calls that k2_dyn_default_route selects (below).
// Dropped (resolve_impl clears it) with IMPL_REF_CHUNK, IMPL_ROWS_ONLY, IMPL_K2_UMMA (the UMMA K2 wins when both are requested) and below sm_90;
// ignored by mode 1 (decode never runs K2) and, for calls beyond k2_sched_capacity(H) sequences, by the launcher (the old K2 kernels run).
// k2_bv 8 / 16 / 32 forces that width for every pair (the device ordering stays), 0 = the device policy.
constexpr int IMPL_K2_DYN = 16;
// The old K2 kernels (natural grid order, tile widths from the host rule), also where the default routing would run the device-scheduled K2: what a
// comparison against the old behaviour asks for. It changes nothing else (resolve_impl drops it) and excludes IMPL_K2_DYN.
constexpr int IMPL_K2_OLD = 32;
constexpr int IMPL_MAX = 63;        // largest accepted impl value (all bits above)

// Default routing (impl 0, nothing else requested): a variable-length (cu_seqlens) prefill call on a Blackwell device (compute capability 10: the cost model
// of the schedule policy was measured on B200 / B300; Hopper and older run the old kernels unless impl 16 is requested) runs the device-scheduled K2 when it is
// predicted to repay the builder (5 .. 30 us) and the extra time of the table-driven kernel on ANY batch of that size. The host knows only N, H and the number
// of tokens, not the lengths. For equal lengths (the worst case of the device policy: skew only helps) the policy is the closed form of the fixed-length launcher
// (choose_k2_tiles), so the rule compares the grid of that closed form with the old rule's grid (16-column tiles) through k2_grid_cost:
//   * N * H * K2_SCHED_ITEMS_PER_PAIR > SMs: up to one wave of SMs at 8 columns there is nothing to order or choose (no builder runs, the old schedule is decoded),
//   * T_total >= K2_ROUTE_MIN_TOKENS_PER_SEQ * N: with fewer chunks per sequence the CTAs gain less than the table read costs,
//   * the closed form picks another grid than the old rule (it does not between one and two waves: such calls lose 3 - 13 % on equal lengths), and
//     (co - cd) / co * stage - K2_ROUTE_OVERHEAD_PCT % * stage >= builder, with co and cd the costs of the old and of the chosen grid, stage the estimated old K2 stage
//     (co * (K2_ROUTE_UNIT_X10NS * tokens per sequence / 64 + K2_ROUTE_CTA_X10NS), times in 0.1 ns) and the builder an upper envelope of its measured time (by N and by N * H),
//   * N <= K2_ROUTE_MAX_SEQUENCES (the builder itself takes up to k2_sched_capacity(H): its time at 1024 sequences, 68 us, loses the saving), and no forced tile width (k2_bv 0).
// Lists with skew gain more than the equal-length estimate says, so some of them (a long sequence among short ones, few sequences) stay on the old kernels: impl 16
// is for them. The measurements behind the constants are in docs/20261103-cfbi-perf.md; tests/cfbi/k2_policy.py (route_default) is the specification and the
// tests compare the two. A caller that wants another choice passes impl 16 or impl 32.
constexpr int64_t K2_ROUTE_MIN_TOKENS_PER_SEQ = 256;
constexpr int64_t K2_ROUTE_OVERHEAD_PCT = 8;           // allowance, percent of the old K2 stage: the table-driven kernel (2 - 5 % slower on the same grid) and the model error (about 3 points)
constexpr int64_t K2_ROUTE_MAX_SEQUENCES = 512;
constexpr int64_t K2_ROUTE_UNIT_X10NS = 526;           // one unit of k2_grid_cost (0.1 K cycles) per chunk at the K2 clock of B200 / B300: 52.6 ns, in 0.1 ns
constexpr int64_t K2_ROUTE_CTA_X10NS = 1064;           // one CTA round (about 5 us) per unit of cost 47 (a 16-column CTA per chunk), in 0.1 ns per unit
constexpr int64_t K2_ROUTE_MAX_TOKENS_PER_SEQ = int64_t(1) << 20;   // the estimate saturates here (more tokens never switch the routing off; 64-bit arithmetic)
bool k2_dyn_default_route(int64_t N, int64_t H, int64_t T_total, int device = -1);   // device < 0: the current one

// What a device can run. The optimised kernels need sm_90+ (bulk copies, mbarrier transaction counts); K1 needs 204 KiB
// (optimised) / 218 KiB (reference) of dynamic shared memory per CTA, while the opt-in limit is 227 KiB on sm_90 / 100 /
// 103, but 163 KiB on sm_80 and 99 KiB on sm_86 / 89 / 120: the chunk kernels cannot run there.
struct DeviceCaps {
    int major, minor;
    int smem_optin;   // cudaDevAttrMaxSharedMemoryPerBlockOptin, bytes
};
DeviceCaps device_caps(int device);   // device < 0: the current device

// The implementation bits that actually run for `requested` on `device` (< 0: current): compute capability < 9 forces
// the reference kernels (IMPL_REF_ROWS | IMPL_REF_CHUNK); a device whose shared-memory opt-in limit is below the
// chunk kernels' need gets IMPL_ROWS_ONLY | IMPL_REF_ROWS (the reference row engine). Idempotent. The row engine evaluates the same program row by row, so
// the bits do not depend on this choice, only the speed does (tests/cfbi/test_vs_ref.py, test_arch.py).
int resolve_impl(int requested, int device = -1);
// Schedule that runs: mode 0 becomes mode 1 (row engine for every row) when IMPL_ROWS_ONLY is set.
inline int resolve_mode(int mode, int impl) { return (mode == 0 && (impl & IMPL_ROWS_ONLY)) ? 1 : mode; }
// Dynamic shared memory per CTA of the K1 kernel of the implementation selected by `impl`.
size_t k1_smem_bytes(int impl);

// Phases of a mode-0 call (context parallelism): K1 (plan + E-independent chunk coefficients) can run on every rank at
// once, K2 (the serial scan that needs the entry state E) has to wait for the previous rank's E. PHASE_K1 skips the
// row engine (shards must start on a 64-row grid line, i.e. pos == 0), PHASE_K2 reuses the plan and workspace.
constexpr int PHASE_K1 = 1;
constexpr int PHASE_K2 = 2;

// Runs plan -> row engine -> K1 -> K2 -> pos update on `stream`.
void launch_forward(const ForwardArgs& a, cudaStream_t stream);

// ---- probes (tests only) ---------------------------------------------------------------------
// D[tile] (16x8) = dot3(A[tile] (16xK, row-major), B[tile] (Kx8, row-major)), one warp per tile.
// reverse_k != 0 walks the K-steps in descending order (negative control).
void launch_probe_dot3(const float* A, const float* B, float* D, int ntiles, int K, int reverse_k,
                       cudaStream_t stream);

// y[i] = approx(x[i]); which: 0 = ex2.approx.ftz, 1 = tanh.approx, 2 = rsqrt.approx.ftz, 3 = rcp.approx.ftz.
void launch_probe_transc(const float* x, float* y, int64_t n, int which, cudaStream_t stream);

// Per-token prep for `rows` tokens of one head (q,k,g: bf16 [rows,128]; beta: bf16 [rows]).
void launch_probe_prep(const void* q, const void* k, const void* g, const void* beta,
                       const float* A_log, const float* dt_bias, float scale, float scale2,
                       float* khat, float* qhat, float* a2, float* beta_out, int rows,
                       cudaStream_t stream);

// The dynamic K2 schedule of a call (tests only): runs the plan kernel on the given inputs (cu_seqlens, fixed_len, pos, slot_ids as in ForwardArgs) and the
// schedule builder, writing the plan and the schedule into `plan` (plan_ints(N, H) ints). sms: the SM count the builder assumes (0 = the current device's;
// the production launch always uses the device's). k2_bv: 0 = the device policy, 8 / 16 / 32 = forced width. Fixed-length call type iff cu_seqlens is null.
struct K2SchedProbeArgs {
    const int64_t* cu_seqlens;
    int fixed_len;
    int N;
    int H;
    const int* pos;
    const int* slot_ids;
    int* plan;
    int sms;
    int k2_bv;
};
void launch_probe_k2_schedule(const K2SchedProbeArgs& a, cudaStream_t stream);

// Test hook of the start-order probe (cfbi_k2_dyn.cu): copies the start time (globaltimer, ns) and the SM of the first n CTAs of the last device-scheduled K2
// launch of a build compiled with CFBI_DYN_TEST_START_LOG into the host arrays and returns n; any other build has no log and returns 0.
int k2_dyn_test_log(unsigned long long* times, unsigned* smids, int n);

}  // namespace cfbi
