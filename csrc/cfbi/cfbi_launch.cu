// CF-BI KDA launcher: plan -> row engine -> K1 -> K2 -> pos update, all on one stream, no host sync.
#include "cfbi/cfbi_common.cuh"
#include "cfbi/cfbi_k1.cuh"
#include "cfbi/cfbi_k2.cuh"
#include "cfbi/cfbi_k2_dyn.h"
#include "cfbi/cfbi_k2_umma.cuh"
#include "cfbi/cfbi_launch.h"
#include "cfbi/cfbi_plan.cuh"
#include "cfbi/cfbi_rows.cuh"
#include "cfbi/ref/ref_k1.cuh"
#include "cfbi/ref/ref_k2.cuh"
#include "cfbi/ref/ref_rows.cuh"

namespace cfbi {

int version() { return 1; }

// The tcgen05 (UMMA) K2: one CTA per (64 value columns, sequence, head); needs K1<16>'s workspace (16-column UT slices).
static void launch_k2_umma(const Params& p, const ForwardArgs& a, cudaStream_t stream) {
    cudaFuncSetAttribute(k2u_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, KU_SMEM_BYTES);
    k2u_kernel<<<dim3(KU_NSLICE, a.N, a.H), KU_THREADS, KU_SMEM_BYTES, stream>>>(p);
#ifdef CFBI_UMMA_PROFILE
    cudaStreamSynchronize(stream);
    long long t[128];
    cudaMemcpyFromSymbol(t, g_ku_prof, sizeof(t));
    printf("PROF chunk %d (CTA 0,0,0), cycles since scan start\n", KU_PROF_CHUNK);
    printf("PROF mma : start %lld eready %lld full_k0 %lld full_k15 %lld p12_issued %lld uready %lld p3_issued %lld | k5: full_ok %lld umma3_issued %lld commit_done %lld | T0 j5: empty_ok %lld full %lld | T0 j9: empty_ok %lld full %lld | k9: full_ok %lld commit_done %lld\n", t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[12], t[13], t[10], t[11], t[14], t[15]);
    for (int w = 0; w < 4; ++w)
        printf("PROF T%d  : empty_j0 %lld full_j0 %lld empty_j15 %lld full_j15 %lld empty_j23 %lld full_j23 %lld\n", w, t[16 + 8 * w], t[17 + 8 * w], t[18 + 8 * w], t[19 + 8 * w], t[20 + 8 * w], t[21 + 8 * w]);
    for (int e = 0; e < 4; ++e)
        printf("PROF E%d  : start %lld odone_ok %lld p12_ok %lld epi1_done %lld p3_ok %lld epi2_done %lld | epi1 batch0: tmem_ld_done %lld batch_done %lld\n", e, t[48 + 8 * e], t[49 + 8 * e], t[50 + 8 * e], t[51 + 8 * e], t[52 + 8 * e], t[53 + 8 * e], t[54 + 8 * e], t[55 + 8 * e]);
    for (int o = 0; o < 4; ++o)
        printf("PROF O%d  : start %lld ur_ok %lld end %lld\n", o, t[80 + 8 * o], t[81 + 8 * o], t[82 + 8 * o]);
    fflush(stdout);
#endif
}

// K2 value-tile widths of a call (see Params::tile_x32): the first x32 (sequence, head) pairs use 32-column tiles, the others bvb-column
// tiles. K2 is a serial scan over a sequence's chunks and one CTA fills an SM (shared memory), so the kernel takes about
// chunks * (time of the busiest SM), and one CTA needs T(BV) per chunk: measured on B200 / B300 (grids of one round and saturated grids
// agree) T(8) : T(16) : T(32) = 3.4 : 4.7 : 7.8 K cycles. A wider tile is cheaper per value column (the A operands W / GQ / KD / Aqk are
// loaded once per CTA and chunk) but quantises the grid more coarsely, and a mix of 32- and 16-column pairs fills the last round of an
// awkwardly sized grid. cost = the busiest SM's time per chunk (units of 0.1 K cycles) with CTAs dispatched in grid order (32-column
// tiles first) to the SM that frees first.
static int64_t k2_grid_cost(int64_t nh, int64_t x32, int bvb, int64_t sms) {
    const int64_t tl = 78, ts = bvb == 8 ? 34 : (bvb == 16 ? 47 : 78);
    const int64_t nl = 4 * x32, ns = (nh - x32) * (HEAD_V / bvb);
    const int64_t a = nl / sms, r = nl % sms;                // r SMs run a + 1 long CTAs (free at f1), the other sms - r run a (free at f0)
    const int64_t f0 = a * tl, f1 = (a + 1) * tl;
    const int64_t busy_long = r > 0 ? f1 : f0;
    if (ns == 0) return busy_long;
    auto slots = [&](int64_t t) {                            // CTA start slots that are free at or before time t
        return (sms - r) * (t >= f0 ? (t - f0) / ts + 1 : 0) + r * (t >= f1 ? (t - f1) / ts + 1 : 0);
    };
    int64_t lo = f0, hi = f1 + ns * ts;                      // earliest t with slots(t) >= ns: the start of the last short CTA
    while (lo < hi) {
        const int64_t mid = (lo + hi) / 2;
        if (slots(mid) >= ns) hi = mid; else lo = mid + 1;
    }
    const int64_t last_end = lo + ts;
    return last_end > busy_long ? last_end : busy_long;
}

static void choose_k2_tiles(int k2_bv, int64_t nh, int sms, bool equal_len, int* x32, int* bvb) {
    if (k2_bv >= 1000) {   // testing hook: a mixed grid with exactly k2_bv - 1000 pairs of 32-column tiles, the rest 16-column tiles
        *x32 = static_cast<int>(std::min<int64_t>(k2_bv - 1000, nh));
        *bvb = 16;
        return;
    }
    if (k2_bv == 8 || k2_bv == 16) { *x32 = 0; *bvb = k2_bv; return; }
    if (k2_bv == 32) { *x32 = static_cast<int>(nh); *bvb = 16; return; }
    // The cost model below assumes sequences of equal length, which the host knows only for a fixed-length call (no cu_seqlens). With
    // cu_seqlens the lengths live on the device (no host sync): a wide tile lengthens the critical path of the longest sequence (the CTA of
    // a 32-column tile needs 1.66x the time per chunk of a 16-column one), so such calls keep the rule of the original launcher.
    if (!equal_len) {
        *x32 = 0;
        *bvb = nh * (HEAD_V / 8) <= sms ? 8 : 16;
        return;
    }
    // automatic: the cheapest homogeneous width (ties: the narrower tile) ...
    int64_t best = k2_grid_cost(nh, 0, 8, sms);
    int bx = 0, bw = 8;
    const int64_t c16 = k2_grid_cost(nh, 0, 16, sms), c32 = k2_grid_cost(nh, nh, 16, sms);
    if (c16 < best) { best = c16; bx = 0; bw = 16; }
    if (c32 < best) { best = c32; bx = static_cast<int>(nh); bw = 16; }
    // ... or 32-column tiles for the first x pairs and 16-column tiles for the rest, with x filling whole rounds of the SMs (4 CTAs per pair);
    // taken only if it is predicted to be at least 3 % faster than the best homogeneous grid (the model is not exact)
    int64_t mixed = best;
    int mx = -1;
    for (int64_t k = 1; 4 * nh > k * sms; ++k) {
        const int64_t x = std::min<int64_t>(nh - 1, k * sms / 4);
        if (x < 1) continue;
        const int64_t c = k2_grid_cost(nh, x, 16, sms);
        if (c < mixed) { mixed = c; mx = static_cast<int>(x); }
    }
    if (mx > 0 && mixed * 100 <= best * 97) { bx = mx; bw = 16; }
    *x32 = bx;
    *bvb = bw;
}

template <int BV>
static void launch_k2_homogeneous(const Params& p, const ForwardArgs& a, cudaStream_t stream) {
    cudaFuncSetAttribute(k2_kernel<BV>, cudaFuncAttributeMaxDynamicSharedMemorySize, K2Cfg<BV>::SMEM_BYTES);
    k2_kernel<BV><<<dim3(K2Cfg<BV>::NSLICE, a.N, a.H), NTHREADS, K2Cfg<BV>::SMEM_BYTES, stream>>>(p);
}

static void launch_chunk_kernels(const Params& p, const ForwardArgs& a, cudaStream_t stream, bool run_k1, bool run_k2,
                                 int64_t max_chunks, bool k2_dyn, int sms_k2) {
    if (run_k1) {
        cudaFuncSetAttribute(k1_kernel<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)K1_SMEM_BYTES);
        // persistent: one CTA per SM works through the (chunk, head) items; the kernel reads the exact item count from the plan
        // (max_chunks * H is only an upper bound known on the host)
        int dev = 0, sms = 148;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        const int64_t items = max_chunks * a.H;
        const unsigned grid = (unsigned)(items < sms ? items : sms);
        k1_kernel<16><<<grid, K1_THREADS, K1_SMEM_BYTES, stream>>>(p);   // stores the UT part in 16-column slices whatever K2 tiles are used
    }
    if (run_k2) {
        if (k2_dyn) {
            launch_k2_dyn(p, a, sms_k2, stream);
        } else if (a.impl & IMPL_K2_UMMA) {
            launch_k2_umma(p, a, stream);
        } else {
            const int64_t nh = static_cast<int64_t>(a.N) * a.H;
            if (p.tile_x32 > 0 && p.tile_x32 < nh) {   // two tile widths in one grid
                const int64_t ctas = 4 * static_cast<int64_t>(p.tile_x32) + (nh - p.tile_x32) * (HEAD_V / p.tile_bvb);
                cudaFuncSetAttribute(k2_kernel_mixed, cudaFuncAttributeMaxDynamicSharedMemorySize, K2Cfg<32>::SMEM_BYTES);
                k2_kernel_mixed<<<static_cast<unsigned>(ctas), NTHREADS, K2Cfg<32>::SMEM_BYTES, stream>>>(p);
            } else {                                   // one tile width: CTA (value slice, sequence, head)
                const int bv = p.tile_x32 > 0 ? 32 : p.tile_bvb;
                if (bv == 8) launch_k2_homogeneous<8>(p, a, stream);
                else if (bv == 32) launch_k2_homogeneous<32>(p, a, stream);
                else launch_k2_homogeneous<16>(p, a, stream);
            }
        }
    }
}

DeviceCaps device_caps(int device) {
    if (device < 0) cudaGetDevice(&device);
    DeviceCaps c{0, 0, 0};
    cudaDeviceGetAttribute(&c.major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&c.minor, cudaDevAttrComputeCapabilityMinor, device);
    cudaDeviceGetAttribute(&c.smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    return c;
}

size_t k1_smem_bytes(int impl) { return (impl & IMPL_REF_CHUNK) ? ref::K1_SMEM_BYTES : K1_SMEM_BYTES; }

int resolve_impl(int requested, int device) {
    const DeviceCaps c = device_caps(device);
    int impl = requested & (IMPL_REF_ROWS | IMPL_REF_CHUNK | IMPL_ROWS_ONLY | IMPL_K2_UMMA | IMPL_K2_DYN);
    // The tcgen05 K2 only exists in the sm_100a / sm_103a code (cfbi_tcgen05.cuh): drop the request everywhere else
    // (the mma.sync K2 computes the same bits).
    if (c.major != 10) impl &= ~IMPL_K2_UMMA;
    // The optimised kernels use sm_90+ instructions (cfbi_async.cuh); below that they are compiled as traps.
    if (c.major < 9) impl |= IMPL_REF_ROWS | IMPL_REF_CHUNK;
    // K1 needs 204 KiB (optimised) / 218 KiB (reference) of dynamic shared memory per CTA; where the device cannot
    // provide it the row engine (which needs 7 KiB) evaluates every row - the same program, just row by row. That
    // fallback uses the reference row engine, the one validated on such devices (sm_80 / 86 / 89).
    if (!(impl & IMPL_ROWS_ONLY) && static_cast<size_t>(c.smem_optin) < k1_smem_bytes(impl)) impl |= IMPL_ROWS_ONLY | IMPL_REF_ROWS;
    // The dynamic-schedule K2 runs on top of the optimised K1: the reference chunk kernels (also what sm < 90 gets) and the row-engine-only schedule
    // have no K2 stage of this kind, and when the UMMA K2 is requested as well it wins (also where it is dropped: the request alone clears the bit).
    if ((impl & (IMPL_REF_CHUNK | IMPL_ROWS_ONLY)) || (requested & IMPL_K2_UMMA)) impl &= ~IMPL_K2_DYN;
    return impl;
}

// Duration of the schedule builder (units of 0.1 ns) for N sequences and `pairs` = N * H pairs: an upper envelope of the medians measured on B200 / B300, the larger of
// an estimate by N and one by the number of pairs (k2_policy.route_builder_x10ns).
static int64_t k2_route_builder_x10ns(int64_t N, int64_t pairs) {
    const int64_t by_sequences = N <= 4 ? 55000 : N <= 16 ? 70000 : N <= 32 ? 95000 : N <= 128 ? 120000 : N <= 256 ? 160000 : 300000;
    const int64_t by_pairs = pairs <= 24 ? 55000 : pairs <= 128 ? 75000 : pairs <= 192 ? 95000 : pairs <= 768 ? 120000 : pairs <= 1536 ? 160000 : 300000;
    return std::max(by_sequences, by_pairs);
}

// The routing rule of cfbi_launch.h in integer arithmetic; tests/cfbi/k2_policy.py (route_default) is the specification of every line.
bool k2_dyn_default_route(int64_t N, int64_t H, int64_t T_total, int device) {
    if (device < 0) cudaGetDevice(&device);
    if (device_caps(device).major != 10) return false;   // the schedule policy is calibrated on B200 / B300 only
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    if (N < 1 || H < 1 || N > K2_ROUTE_MAX_SEQUENCES || N > k2_sched_capacity(H) || T_total < K2_ROUTE_MIN_TOKENS_PER_SEQ * N) return false;
    const int64_t nh = N * H;
    if (nh * K2_SCHED_ITEMS_PER_PAIR <= sms) return false;            // one wave at 8 columns: the old schedule is decoded, no builder runs
    int x32 = 0, bvb = 0;
    choose_k2_tiles(0, nh, sms, true, &x32, &bvb);                    // the device policy for equal lengths: the closed form of the fixed-length launcher
    if (x32 == 0 && bvb == 16) return false;                          // the old rule's grid (16-column tiles): nothing to gain
    const int64_t co = k2_grid_cost(nh, 0, 16, sms), cd = k2_grid_cost(nh, x32, bvb, sms);
    const int64_t tokens_per_seq = std::min<int64_t>(T_total / N, K2_ROUTE_MAX_TOKENS_PER_SEQ);
    const int64_t stage = co * (K2_ROUTE_UNIT_X10NS * tokens_per_seq / 64 + K2_ROUTE_CTA_X10NS);
    return (100 * (co - cd) - K2_ROUTE_OVERHEAD_PCT * co) * stage >= 100 * co * k2_route_builder_x10ns(N, nh);
}

void launch_forward(const ForwardArgs& a, cudaStream_t stream) {
    Params p;
    p.q = static_cast<const __nv_bfloat16*>(a.q);
    p.k = static_cast<const __nv_bfloat16*>(a.k);
    p.v = static_cast<const __nv_bfloat16*>(a.v);
    p.g = static_cast<const __nv_bfloat16*>(a.g);
    p.beta = static_cast<const __nv_bfloat16*>(a.beta);
    p.out = static_cast<__nv_bfloat16*>(a.out);
    p.out_f32 = a.out_f32;
    p.A_log = a.A_log;
    p.dt_bias = a.dt_bias;
    p.scale = a.scale;
    p.scale2 = a.scale2;
    p.H = a.H;
    p.N = a.N;
    p.E = a.E;
    p.rows = a.rows;
    p.pos = a.pos;
    p.slot_ids = a.slot_ids;
    p.plan = reinterpret_cast<SeqPlan*>(a.plan);
    p.ws = a.ws;
    p.tile_x32 = 0;
    p.tile_bvb = 16;

    const int mode = resolve_mode(a.mode, a.impl);   // the binding has already done this and supplied a state if needed
    p.fuse_pos = mode == 1 ? 1 : 0;
    const bool run_k1 = (a.phase & PHASE_K1) != 0, run_k2 = (a.phase & PHASE_K2) != 0;
    if (run_k1) plan_kernel<<<1, PLAN_THREADS, 0, stream>>>(a.cu_seqlens, a.fixed_len, a.pos, a.slot_ids, a.N, mode == 1 ? 1 : 0, p.plan);
    if (run_k1 && run_k2 && a.E != nullptr) {
        // head rows (finishing the open chunk) or, in mode 1, every row
        if (a.impl & IMPL_REF_ROWS) ref::rows_kernel<<<dim3(a.N, a.H), NTHREADS, 0, stream>>>(p);
        else rows_kernel<<<dim3(a.N, a.H), NTHREADS, 0, stream>>>(p);
    }
    if (mode == 0) {
        const int64_t max_chunks = (a.T_total + BT - 1) / BT + a.N;   // upper bound; extra CTAs exit at once
        if (a.impl & IMPL_REF_CHUNK) {
            cudaFuncSetAttribute(ref::k1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ref::K1_SMEM_BYTES);
            if (run_k1) ref::k1_kernel<<<dim3((unsigned)max_chunks, a.H), NTHREADS, ref::K1_SMEM_BYTES, stream>>>(p);
            if (run_k2) ref::k2_kernel<<<dim3(HEAD_V / ref::K2_BV, a.N, a.H), NTHREADS, 0, stream>>>(p);
        } else {
            // K2 value tiles: chosen from N, H and the SM count unless forced by a.k2_bv (K1 does not depend on them: it always stores the
            // UT part of the workspace in 16-column slices, which the UMMA K2 consumes as well).
            int dev = 0, sms = 148;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
            const int64_t nh = static_cast<int64_t>(a.N) * a.H;
            // beyond the capacity of the schedule builder (host-known N and H) the old K2 kernels run
            const bool k2_dyn = (a.impl & IMPL_K2_DYN) != 0 && !(a.impl & IMPL_K2_UMMA) && a.N <= k2_sched_capacity(a.H);
            if (k2_dyn || (a.impl & IMPL_K2_UMMA)) { p.tile_x32 = 0; p.tile_bvb = 16; }
            else choose_k2_tiles(a.k2_bv, nh, sms, a.cu_seqlens == nullptr, &p.tile_x32, &p.tile_bvb);
            launch_chunk_kernels(p, a, stream, run_k1, run_k2, max_chunks, k2_dyn, sms);
        }
    }
    if (run_k2 && a.pos != nullptr && mode != 1) finalize_pos_kernel<<<(a.N + 255) / 256, 256, 0, stream>>>(p.plan, a.pos, a.N);
}

// The dynamic K2 schedule of a call (tests only, see cfbi_launch.h).
void launch_probe_k2_schedule(const K2SchedProbeArgs& a, cudaStream_t stream) {
    int sms = a.sms;
    if (sms <= 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    }
    SeqPlan* plan = reinterpret_cast<SeqPlan*>(a.plan);
    plan_kernel<<<1, PLAN_THREADS, 0, stream>>>(a.cu_seqlens, a.fixed_len, a.pos, a.slot_ids, a.N, 0, plan);
    k2_sched_build(plan, a.N, a.H, sms, a.k2_bv, a.cu_seqlens == nullptr, stream);
}

}  // namespace cfbi
