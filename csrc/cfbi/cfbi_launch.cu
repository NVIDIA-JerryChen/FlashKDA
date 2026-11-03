// CF-BI KDA launcher: plan -> row engine -> K1 -> K2 -> pos update, all on one stream, no host sync.
#include "cfbi/cfbi_common.cuh"
#include "cfbi/cfbi_k1.cuh"
#include "cfbi/cfbi_k2.cuh"
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
        printf("PROF E%d  : start %lld odone_ok %lld p12_ok %lld epi1_done %lld p3_ok %lld epi2_done %lld\n", e, t[48 + 8 * e], t[49 + 8 * e], t[50 + 8 * e], t[51 + 8 * e], t[52 + 8 * e], t[53 + 8 * e]);
    for (int o = 0; o < 4; ++o)
        printf("PROF O%d  : start %lld ur_ok %lld end %lld\n", o, t[80 + 8 * o], t[81 + 8 * o], t[82 + 8 * o]);
    fflush(stdout);
#endif
}

template <int BV>
static void launch_chunk_kernels(const Params& p, const ForwardArgs& a, cudaStream_t stream, bool run_k1, bool run_k2,
                                 int64_t max_chunks) {
    if (run_k1) {
        cudaFuncSetAttribute(k1_kernel<BV>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)K1_SMEM_BYTES);
        k1_kernel<BV><<<dim3((unsigned)max_chunks, a.H), NTHREADS, K1_SMEM_BYTES, stream>>>(p);
    }
    if (run_k2) {
        if (BV == 16 && (a.impl & IMPL_K2_UMMA)) {
            launch_k2_umma(p, a, stream);
        } else {
            cudaFuncSetAttribute(k2_kernel<BV>, cudaFuncAttributeMaxDynamicSharedMemorySize, K2Cfg<BV>::SMEM_BYTES);
            k2_kernel<BV><<<dim3(K2Cfg<BV>::NSLICE, a.N, a.H), NTHREADS, K2Cfg<BV>::SMEM_BYTES, stream>>>(p);
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
    int impl = requested & (IMPL_REF_ROWS | IMPL_REF_CHUNK | IMPL_ROWS_ONLY | IMPL_K2_UMMA);
    // The tcgen05 K2 only exists in the sm_100a / sm_103a code (cfbi_tcgen05.cuh): drop the request everywhere else
    // (the mma.sync K2 computes the same bits).
    if (c.major != 10) impl &= ~IMPL_K2_UMMA;
    // The optimised kernels use sm_90+ instructions (cfbi_async.cuh); below that they are compiled as traps.
    if (c.major < 9) impl |= IMPL_REF_ROWS | IMPL_REF_CHUNK;
    // K1 needs 204 KiB (optimised) / 218 KiB (reference) of dynamic shared memory per CTA; where the device cannot
    // provide it the row engine (which needs 7 KiB) evaluates every row - the same program, just row by row. That
    // fallback uses the reference row engine, the one validated on such devices (sm_80 / 86 / 89).
    if (!(impl & IMPL_ROWS_ONLY) && static_cast<size_t>(c.smem_optin) < k1_smem_bytes(impl)) impl |= IMPL_ROWS_ONLY | IMPL_REF_ROWS;
    return impl;
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
            int bv = a.k2_bv;   // value columns per K2 CTA: 8 when the whole (N x H x 16 slices) grid fits in one wave of SMs
            if (a.impl & IMPL_K2_UMMA) bv = 16;   // the UMMA K2 consumes the 16-column UT slices of K1<16> (checked by the binding)
            if (bv == 0) {
                int dev = 0, sms = 148;
                cudaGetDevice(&dev);
                cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
                bv = ((int64_t)a.N * a.H * (HEAD_V / 8) <= sms) ? 8 : 16;
            }
            if (bv == 8) launch_chunk_kernels<8>(p, a, stream, run_k1, run_k2, max_chunks);
            else launch_chunk_kernels<16>(p, a, stream, run_k1, run_k2, max_chunks);
        }
    }
    if (run_k2 && a.pos != nullptr && mode != 1) finalize_pos_kernel<<<(a.N + 255) / 256, 256, 0, stream>>>(p.plan, a.pos, a.N);
}

}  // namespace cfbi
