// CF-BI launch policy in plain C++ (no CUDA header: the host compiler alone builds this file, which is how the tests call it with any SM count and compute capability): the
// implementation bits of a forward call, the shared-memory need of the row-engine launch and the pure functions that decide a launch from numbers. cfbi_launch.h includes it; a
// wrapper that reads the device (cfbi_launch.cu) and the binding (cfbi_binding.cpp) call the functions below and nothing else decides the same questions.
#pragma once

#include <cstdint>

namespace cfbi {

// ---- implementation bits (the `impl` argument of a forward call) -----------------------------------------------------------------------------------------------
// Implementation selector: the frozen v1 kernels in csrc/cfbi/ref/ are the bitwise oracle of the optimised ones.
// K1 and K2 share one bit because their workspace layouts differ between the implementations. 0 = the optimised
// kernels where the device can run them (see resolve_impl in cfbi_launch.h: otherwise the reference kernels are used).
constexpr int IMPL_REF_ROWS = 1;    // row engine (decode, head rows)
constexpr int IMPL_REF_CHUNK = 2;   // K1 + K2 (prefill / training forward)
constexpr int IMPL_ROWS_ONLY = 4;   // no K1/K2: every row goes through the row engine (mode 1 semantics even for mode 0)
constexpr int IMPL_K2_UMMA = 8;     // optimised K1<16> + the tcgen05 (UMMA) K2 (Blackwell sm_100/103 only; ignored with IMPL_REF_CHUNK)
// optimised K1<16> + a K2 driven by a schedule that a builder kernel makes on the device from the plan (cfbi_k2_dyn.cu): the order of the K2 CTAs
// (longest sequence first) and the value-tile width of every (sequence, head) pair (8, 16 or 32 columns) follow the sequence lengths, which the host
// does not know (no host synchronisation). Bit-identical to the other K2 kernels (every (pair, value slice) is still one CTA, chunk by chunk, with
// the same arithmetic). Requested explicitly it runs for every call; impl 0 sets it only for the calls that k2_dyn_default_route selects (cfbi_launch.h).
// Dropped (resolve_impl clears it) with IMPL_REF_CHUNK, IMPL_ROWS_ONLY, IMPL_K2_UMMA (the UMMA K2 wins when both are requested) and below sm_90;
// ignored by mode 1 (decode never runs K2) and, for calls beyond k2_sched_capacity(H) sequences, by the launcher (the old K2 kernels run).
// k2_bv 8 / 16 / 32 forces that width for every pair (the device ordering stays), 0 = the device policy.
constexpr int IMPL_K2_DYN = 16;
// The old K2 kernels (natural grid order, tile widths from the host rule), also where the default routing would run the device-scheduled K2: what a
// comparison against the old behaviour asks for. It changes nothing else (resolve_impl drops it) and excludes IMPL_K2_DYN.
constexpr int IMPL_K2_OLD = 32;
// Decode (mode 1) on the optimised row engine can be ONE kernel launch (rows_kernel<true>, see decode_single_launch_policy below): no plan kernel runs, every head CTA derives its
// sequence's token window, slot and open-chunk fill itself, and the head CTA that is the last of a sequence's H CTAs to arrive advances pos. This bit keeps the former
// two launches instead (plan kernel, then the row engine fed by the plan): the A/B of the benchmark and a differential test target. It changes no bit of any result and is
// ignored by every call that is not a decode call of the optimised row engine (mode 0 prefill / training forward, IMPL_REF_ROWS).
constexpr int IMPL_DECODE_PLAN = 64;
// Bit 128 is reserved for the operand-swapped tiles of the row engine (IMPL_ROWS_SWAP; not part of this tree). Until it is assigned it is rejected like every value above IMPL_MAX.
constexpr int IMPL_ROWS_SWAP_RESERVED = 128;
constexpr int IMPL_MAX = 127;       // largest accepted impl value (all bits up to IMPL_DECODE_PLAN)
static_assert(IMPL_MAX == IMPL_ROWS_SWAP_RESERVED - 1, "every bit below the reserved one is assigned: the accepted range is exactly 0 .. IMPL_MAX");
static_assert(IMPL_MAX == 2 * IMPL_DECODE_PLAN - 1, "the highest assigned bit is IMPL_DECODE_PLAN");

// ---- shared memory of the row-engine launch ------------------------------------------------------------------------------------------------------------------------
// Bytes per CTA of the optimised row engine (rows_kernel, both instantiations): STATIC is what the compiler allocates for its static __shared__ variables, the figure of the code
// generation gate (ptxas reports 8,192 B on sm_100a / sm_103a and 7,808 B on sm_90a); it is larger than the structure of the source (cfbi_rows.cuh static_asserts that the
// structure fits and that the difference stays below one allocation unit of 1 KiB), so the need is the compiler's figure, not the structure's. DYNAMIC is the dynamic shared
// memory of the launch (the launch sites and the occupancy hook read this constant). The per-CTA reserve of the driver (1 KiB) comes on top of both and is a device figure.
// The dynamic part of the optimised engine is the staged copy of the state E (cfbi_rows.cuh): 128 value rows of ROWS_E_STRIDE floats, a row being the 128 key columns of E and a pad of 4
// floats (a row stride of 4 banks (mod 32): the fragment reads and the chunk-end updates of the kernel are then free of bank conflicts, and every row starts on a 16-byte boundary for the
// async copies). The kernel static_asserts that these numbers are its layout; the literals are kept as numbers because the tests and the record checker read them from this file.
constexpr int ROWS_STATIC_SMEM_BYTES = 8192;
constexpr int ROWS_E_STRIDE = 132;
constexpr int ROWS_E_TILE_FLOATS = 16896;
constexpr int ROWS_DYNAMIC_SMEM_BYTES = 67584;
constexpr int ROWS_SMEM_NEED_BYTES = ROWS_STATIC_SMEM_BYTES + ROWS_DYNAMIC_SMEM_BYTES;
static_assert(ROWS_E_TILE_FLOATS == 128 * ROWS_E_STRIDE && ROWS_DYNAMIC_SMEM_BYTES == ROWS_E_TILE_FLOATS * static_cast<int>(sizeof(float)), "the dynamic shared memory of the row engine is the E tile: 128 rows of ROWS_E_STRIDE floats");
// The frozen reference row engine (csrc/cfbi/ref, the oracle) stages nothing: its launch has no dynamic shared memory, whatever the optimised engine's has (its own launch constant, its own
// occupancy figure and its own binding in the decision record).
constexpr int ROWS_REF_DYNAMIC_SMEM_BYTES = 0;
// The budget of the row engine, the resource gate of the optimisation steps: ROWS_BUDGET_CTAS_PER_SM CTAs, each with its static and dynamic shared memory and the reserve of the driver, fit
// the shared memory of one SM of the devices of the gate (228 KiB on B200 and B300; the occupancy records say 233,472 B). A tile that does not fit is a build error, not a halved occupancy.
constexpr int ROWS_BUDGET_CTAS_PER_SM = 2;
constexpr int ROWS_BUDGET_SM_SHARED_BYTES = 228 * 1024;
constexpr int ROWS_BUDGET_CTA_RESERVE_BYTES = 1024;
static_assert(ROWS_BUDGET_CTAS_PER_SM * (ROWS_SMEM_NEED_BYTES + ROWS_BUDGET_CTA_RESERVE_BYTES) <= ROWS_BUDGET_SM_SHARED_BYTES,
              "the shared memory of the optimised row engine (static + dynamic + the driver's reserve) must leave room for ROWS_BUDGET_CTAS_PER_SM CTAs per SM");

// ---- which implementation runs on a device ----------------------------------------------------------------------------------------------------------------------------
// What the launches of a forward call ask of a device, per CTA, in bytes of shared memory (static plus dynamic, without the driver's per-CTA reserve: the opt-in limit of the device,
// cudaDevAttrMaxSharedMemoryPerBlockOptin, bounds exactly this figure): K1 of the optimised and of the reference chunk kernels (dynamic) and the optimised row engine (ROWS_SMEM_NEED_BYTES).
struct ImplNeeds {
    int64_t k1;
    int64_t k1_ref;
    int64_t rows;
};

// The pure part of resolve_impl (cfbi_launch.h documents the bits): the implementation bits that run for `requested` on a device of compute capability `cc` (major * 10 + minor) whose
// opt-in limit of shared memory per block is `smem_optin`. Every input is an argument, so the tests call it with any device and any need. Idempotent.
//   * the tcgen05 K2 exists in the sm_100a / sm_103a code only (dropped on every other major version); below sm_90 the optimised kernels are compiled as traps: the reference kernels run;
//   * a device whose opt-in limit is below the need of K1 of the selected family runs every row through the row engine, the reference row engine (the one validated on such devices);
//   * the optimised row engine needs `needs.rows`: a device whose limit is below it runs the reference row engine, WHATEVER was requested, an explicit rows-only request (IMPL_ROWS_ONLY,
//     which skips the K1 check above) included, and keeps the optimised one from exactly that need on;
//   * the dynamic-schedule K2 runs on the optimised K1: the reference chunk kernels, the row-engine-only schedule and a request for the UMMA K2 (which wins, also where it is dropped) clear it.
constexpr int resolve_impl_for(int requested, int cc, int64_t smem_optin, const ImplNeeds& needs) {
    int impl = requested & (IMPL_REF_ROWS | IMPL_REF_CHUNK | IMPL_ROWS_ONLY | IMPL_K2_UMMA | IMPL_K2_DYN | IMPL_DECODE_PLAN);
    const int major = cc / 10;
    if (major != 10) impl &= ~IMPL_K2_UMMA;
    if (major < 9) impl |= IMPL_REF_ROWS | IMPL_REF_CHUNK;
    if (!(impl & IMPL_ROWS_ONLY) && smem_optin < ((impl & IMPL_REF_CHUNK) ? needs.k1_ref : needs.k1)) impl |= IMPL_ROWS_ONLY | IMPL_REF_ROWS;
    if (!(impl & IMPL_REF_ROWS) && smem_optin < needs.rows) impl |= IMPL_REF_ROWS;
    if ((impl & (IMPL_REF_CHUNK | IMPL_ROWS_ONLY)) || (requested & IMPL_K2_UMMA)) impl &= ~IMPL_K2_DYN;
    return impl;
}

// ---- single-launch decode -----------------------------------------------------------------------------------------------------------------------------------------------
// Whether a call that runs mode 1 (`mode` and `impl` as resolve_mode / resolve_impl return them: the caller maps mode 0 + IMPL_ROWS_ONLY to 1 and resolves the device's bits) of N
// sequences and H heads is ONE kernel launch of the optimised row engine (rows_kernel<true>, cfbi_rows.cuh) instead of the plan kernel followed by the row engine. The plan kernel
// is one more dependent launch on every decode call (its recorded cost is in the performance document); what the single launch pays instead is one more
// dependent global access per CTA (slot_ids, then the pos word) that every CTA of a multi-wave grid repeats, so the rule keeps the plan kernel for grids of more than
// DECODE_SINGLE_MAX_CTAS_PER_SM CTAs per SM (N * H CTAs; ten waves of the two CTAs per SM that fit). The limit and the routing set are the result of an interleaved A/B against
// IMPL_DECODE_PLAN: the limit is the largest grid, in CTAs per SM, below which no cell that was timed with the calibration control is slower than the two launches, and the
// performance document records the cells, with the ones that straddle the limit. The single launch is correct on every device that runs the optimised row engine; the routing set
// only says where it is the default.
constexpr int64_t DECODE_SINGLE_MAX_CTAS_PER_SM = 20;
constexpr int DECODE_SINGLE_ROUTING_CC_MAJOR = 10;   // B200 (cc 100) and B300 (cc 103); Hopper (cc 90) and the others keep the plan kernel by default

// The part of the rule that depends on neither the device nor the size of the grid: mode 1 on the optimised row engine (not IMPL_REF_ROWS: the frozen oracle keeps the plan kernel,
// resolve_impl sets the bit below sm_90, where the optimised kernels cannot run), without IMPL_DECODE_PLAN, for a grid that exists. A calibration control that forces the single
// launch (cfbi_launch.cu, hidden) applies this and nothing else.
constexpr bool decode_single_launch_eligible(int mode, int impl, int64_t N, int64_t H) {
    return mode == 1 && (impl & (IMPL_REF_ROWS | IMPL_DECODE_PLAN)) == 0 && N >= 1 && H >= 1;
}

// The routing set: the compute capabilities (major * 10 + minor: 90, 100, 103, 120) on which the single launch is the default.
constexpr bool decode_single_launch_routed(int cc) { return cc / 10 == DECODE_SINGLE_ROUTING_CC_MAJOR; }

// The only policy function of this launch decision and a pure one: the arguments are all the inputs (`sms` the SM count of the device, `cc` its compute capability as major * 10 + minor).
// True only for an eligible call whose grid, N * H CTAs, holds at most DECODE_SINGLE_MAX_CTAS_PER_SM CTAs per SM on a device of the routing set; an SM count that is not positive
// (a failed device query) keeps the plan kernel, the safe schedule. N * H is compared as N <= limit / H (the same for positive numbers, no overflow).
constexpr bool decode_single_launch_policy(int mode, int impl, int64_t N, int64_t H, int sms, int cc) {
    return decode_single_launch_eligible(mode, impl, N, H) && sms > 0 && decode_single_launch_routed(cc) && N <= (DECODE_SINGLE_MAX_CTAS_PER_SM * sms) / H;
}

}  // namespace cfbi
