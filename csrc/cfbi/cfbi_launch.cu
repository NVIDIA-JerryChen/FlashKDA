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

template <int BV>
static void launch_chunk_kernels(const Params& p, const ForwardArgs& a, cudaStream_t stream, bool run_k1, bool run_k2,
                                 int64_t max_chunks) {
    if (run_k1) {
        cudaFuncSetAttribute(k1_kernel<BV>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)K1_SMEM_BYTES);
        k1_kernel<BV><<<dim3((unsigned)max_chunks, a.H), NTHREADS, K1_SMEM_BYTES, stream>>>(p);
    }
    if (run_k2) {
        cudaFuncSetAttribute(k2_kernel<BV>, cudaFuncAttributeMaxDynamicSharedMemorySize, K2Cfg<BV>::SMEM_BYTES);
        k2_kernel<BV><<<dim3(K2Cfg<BV>::NSLICE, a.N, a.H), NTHREADS, K2Cfg<BV>::SMEM_BYTES, stream>>>(p);
    }
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
    p.fuse_pos = a.mode == 1 ? 1 : 0;
    p.plan = reinterpret_cast<SeqPlan*>(a.plan);
    p.ws = a.ws;

    const bool run_k1 = (a.phase & PHASE_K1) != 0, run_k2 = (a.phase & PHASE_K2) != 0;
    if (run_k1) plan_kernel<<<1, PLAN_THREADS, 0, stream>>>(a.cu_seqlens, a.fixed_len, a.pos, a.slot_ids, a.N, a.mode == 1 ? 1 : 0, p.plan);
    if (run_k1 && run_k2 && a.E != nullptr) {
        // head rows (finishing the open chunk) or, in mode 1, every row
        if (a.impl & IMPL_REF_ROWS) ref::rows_kernel<<<dim3(a.N, a.H), NTHREADS, 0, stream>>>(p);
        else rows_kernel<<<dim3(a.N, a.H), NTHREADS, 0, stream>>>(p);
    }
    if (a.mode == 0) {
        const int64_t max_chunks = (a.T_total + BT - 1) / BT + a.N;   // upper bound; extra CTAs exit at once
        if (a.impl & IMPL_REF_CHUNK) {
            cudaFuncSetAttribute(ref::k1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ref::K1_SMEM_BYTES);
            if (run_k1) ref::k1_kernel<<<dim3((unsigned)max_chunks, a.H), NTHREADS, ref::K1_SMEM_BYTES, stream>>>(p);
            if (run_k2) ref::k2_kernel<<<dim3(HEAD_V / ref::K2_BV, a.N, a.H), NTHREADS, 0, stream>>>(p);
        } else {
            int bv = a.k2_bv;   // value columns per K2 CTA: 8 when the whole (N x H x 16 slices) grid fits in one wave of SMs
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
    if (run_k2 && a.pos != nullptr && a.mode != 1) finalize_pos_kernel<<<(a.N + 255) / 256, 256, 0, stream>>>(p.plan, a.pos, a.N);
}

}  // namespace cfbi
