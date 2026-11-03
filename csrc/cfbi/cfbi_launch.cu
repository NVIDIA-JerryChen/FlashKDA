// CF-BI KDA launcher: plan -> row engine -> K1 -> K2 -> pos update, all on one stream, no host sync.
#include "cfbi/cfbi_common.cuh"
#include "cfbi/cfbi_k1.cuh"
#include "cfbi/cfbi_k2.cuh"
#include "cfbi/cfbi_launch.h"
#include "cfbi/cfbi_plan.cuh"
#include "cfbi/cfbi_rows.cuh"
#include "cfbi/ref/ref_k1.cuh"
#include "cfbi/ref/ref_k2.cuh"
#include "cfbi/ref/ref_rows.cuh"

namespace cfbi {

int version() { return 1; }

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
    int impl = requested & (IMPL_REF_ROWS | IMPL_REF_CHUNK | IMPL_ROWS_ONLY);
    // The optimised kernels use sm_90+ instructions (cfbi_async.cuh); below that they are compiled as traps.
    if (c.major < 9) impl |= IMPL_REF_ROWS | IMPL_REF_CHUNK;
    // K1 needs ~218 KiB of dynamic shared memory per CTA; where the device cannot provide it the row engine (which
    // needs 7 KiB) evaluates every row - the same program, just row by row.
    if (!(impl & IMPL_ROWS_ONLY) && static_cast<size_t>(c.smem_optin) < k1_smem_bytes(impl)) impl |= IMPL_ROWS_ONLY;
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
    p.plan = reinterpret_cast<SeqPlan*>(a.plan);
    p.ws = a.ws;

    const int mode = resolve_mode(a.mode, a.impl);   // the binding has already done this and supplied a state if needed
    plan_kernel<<<1, PLAN_THREADS, 0, stream>>>(a.cu_seqlens, a.fixed_len, a.pos, a.N, mode == 1 ? 1 : 0, p.plan);
    if (a.E != nullptr) {
        // head rows (finishing the open chunk) or, in mode 1, every row
        if (a.impl & IMPL_REF_ROWS) ref::rows_kernel<<<dim3(a.N, a.H), NTHREADS, 0, stream>>>(p);
        else rows_kernel<<<dim3(a.N, a.H), NTHREADS, 0, stream>>>(p);
    }
    if (mode == 0) {
        const int64_t max_chunks = (a.T_total + BT - 1) / BT + a.N;   // upper bound; extra CTAs exit at once
        if (a.impl & IMPL_REF_CHUNK) {
            cudaFuncSetAttribute(ref::k1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ref::K1_SMEM_BYTES);
            ref::k1_kernel<<<dim3((unsigned)max_chunks, a.H), NTHREADS, ref::K1_SMEM_BYTES, stream>>>(p);
            ref::k2_kernel<<<dim3(HEAD_V / ref::K2_BV, a.N, a.H), NTHREADS, 0, stream>>>(p);
        } else {
            cudaFuncSetAttribute(k1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)K1_SMEM_BYTES);
            cudaFuncSetAttribute(k2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, K2_SMEM_BYTES);
            k1_kernel<<<dim3((unsigned)max_chunks, a.H), NTHREADS, K1_SMEM_BYTES, stream>>>(p);
            k2_kernel<<<dim3(K2_NSLICE, a.N, a.H), NTHREADS, K2_SMEM_BYTES, stream>>>(p);
        }
    }
    if (a.pos != nullptr) finalize_pos_kernel<<<(a.N + 255) / 256, 256, 0, stream>>>(p.plan, a.pos, a.N);
}

}  // namespace cfbi
