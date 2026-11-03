// CF-BI KDA probes: tiny kernels that expose the pinned primitives to tests.
//   * probe_dot3   - the hardware premise: an output element of dot3 depends only on its own
//                    (row, column) operand vectors, not on the slot they sit in nor on other rows/columns.
//   * probe_transc - the three approximate instructions (ex2 / tanh / rsqrt) as elementwise kernels, so that
//                    PyTorch emulations can use the *same* approximations.
//   * probe_prep   - the shared per-token prep (l2norm, gate, beta).
#include "cfbi/cfbi_launch.h"
#include "cfbi/cfbi_prep.cuh"
#include "cfbi/cfbi_primitives.cuh"

namespace cfbi {

__global__ void probe_dot3_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ D, int K, int reverse_k) {
    const int tile = blockIdx.x;
    LaneIdx L;
    const float* a = A + (int64_t)tile * 16 * K;
    const float* b = B + (int64_t)tile * K * 8;
    const int nks = K / 8;
    auto load_a = [&](int ks, float (&r)[4]) {
        const int k0 = 8 * (reverse_k ? (nks - 1 - ks) : ks);
        r[0] = a[(L.g)     * K + k0 + L.t];
        r[1] = a[(L.g + 8) * K + k0 + L.t];
        r[2] = a[(L.g)     * K + k0 + L.t + 4];
        r[3] = a[(L.g + 8) * K + k0 + L.t + 4];
    };
    auto load_b = [&](int ks, float (&r)[2]) {
        const int k0 = 8 * (reverse_k ? (nks - 1 - ks) : ks);
        r[0] = b[(k0 + L.t) * 8 + L.g];
        r[1] = b[(k0 + L.t + 4) * 8 + L.g];
    };
    float out[4];
    dot3(nks, load_a, load_b, out);
    float* d = D + (int64_t)tile * 16 * 8;
#pragma unroll
    for (int i = 0; i < 4; ++i) d[c_row(L, i) * 8 + c_col(L, i)] = out[i];
}

void launch_probe_dot3(const float* A, const float* B, float* D, int ntiles, int K, int reverse_k,
                       cudaStream_t stream) {
    probe_dot3_kernel<<<ntiles, 32, 0, stream>>>(A, B, D, K, reverse_k);
}

__global__ void probe_transc_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n, int which) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    y[i] = which == 0 ? ex2_ftz(v) : (which == 1 ? tanh_approx(v) : rsqrt_ftz(v));
}

void launch_probe_transc(const float* x, float* y, int64_t n, int which, cudaStream_t stream) {
    probe_transc_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(x, y, n, which);
}

// One warp per token: lanes 0-15 normalise k, lanes 16-31 normalise q (channels 8*(lane&15) ..+7), then
// every lane computes the gate for channels lane, lane+32, lane+64, lane+96.
__global__ void probe_prep_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                                  const __nv_bfloat16* __restrict__ g, const __nv_bfloat16* __restrict__ beta,
                                  const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                                  float scale, float scale2, float* __restrict__ khat,
                                  float* __restrict__ qhat, float* __restrict__ a2,
                                  float* __restrict__ beta_out, int rows) {
    const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    const bool is_q = lane >= 16;
    const int c0 = 8 * (lane & 15);
    const __nv_bfloat16* src = (is_q ? q : k) + (int64_t)row * HEAD_K + c0;
    float x[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) x[i] = bf16_to_f32(src[i]);
    const float inv = l2norm_inv(x);
    float* dst = (is_q ? qhat : khat) + (int64_t)row * HEAD_K + c0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float y = __fmul_rn(x[i], inv);
        dst[i] = is_q ? __fmul_rn(y, scale) : y;
    }
    const float ale = gate_a_log_exp(A_log[0]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = lane + 32 * i;
        a2[(int64_t)row * HEAD_K + c] =
            gate_a2(bf16_to_f32(g[(int64_t)row * HEAD_K + c]), dt_bias[c], ale, scale2);
    }
    if (lane == 0) beta_out[row] = beta_act(bf16_to_f32(beta[row]));
}

void launch_probe_prep(const void* q, const void* k, const void* g, const void* beta,
                       const float* A_log, const float* dt_bias, float scale, float scale2,
                       float* khat, float* qhat, float* a2, float* beta_out, int rows,
                       cudaStream_t stream) {
    const int warps = 8;
    probe_prep_kernel<<<(rows + warps - 1) / warps, warps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(k),
        static_cast<const __nv_bfloat16*>(g), static_cast<const __nv_bfloat16*>(beta), A_log, dt_bias,
        scale, scale2, khat, qhat, a2, beta_out, rows);
}

}  // namespace cfbi
