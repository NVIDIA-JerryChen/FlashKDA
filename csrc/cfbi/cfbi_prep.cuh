// CF-BI KDA: per-token prep (l2norm, gate, beta). One shared set of device functions: K1, the row
// engine and the probes all call these, so the per-token numbers are identical everywhere.
//
// Formulas follow master's tests/torch_ref.py (the bit-exact emulation of master's prep), except that
// khat / qhat stay in fp32 (master rounds them to bf16).
#pragma once

#include "cfbi/cfbi_primitives.cuh"

namespace cfbi {

// sigmoid(x) = fma(tanh.approx(x * 0.5), 0.5, 0.5)
__device__ __forceinline__ float sigmoid_tanh(float x) {
    return __fmaf_rn(tanh_approx(__fmul_rn(x, 0.5f)), 0.5f, 0.5f);
}

// Per-head gate scalar: ex2(A_log * LOG2E)
__device__ __forceinline__ float gate_a_log_exp(float A_log) { return ex2_ftz(__fmul_rn(A_log, LOG2E_F)); }

// log2-decay of one (token, channel):  a2 = scale2 * sigmoid(a_log_exp * (g + dt_bias)),
// scale2 = float(lower_bound * LOG2E) (computed in double on the host, like torch_ref / master).
__device__ __forceinline__ float gate_a2(float g, float dt_bias, float a_log_exp, float scale2) {
    return __fmul_rn(scale2, sigmoid_tanh(__fmul_rn(a_log_exp, __fadd_rn(g, dt_bias))));
}

// beta = sigmoid(logit)
__device__ __forceinline__ float beta_act(float logit) { return sigmoid_tanh(logit); }

// L2 inverse norm of a 128-vector spread over a 16-lane group: lane j holds channels 8j .. 8j+7.
//   partial_j = 8 sequential FMAs; butterfly add over xor offsets 8,4,2,1; rsqrt.approx.ftz(sum + 1e-6).
// All 32 lanes of the warp must execute this (full-mask shuffles); the two 16-lane halves are independent.
__device__ __forceinline__ float l2norm_inv(const float (&x)[8]) {
    float p = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) p = __fmaf_rn(x[i], x[i], p);
#pragma unroll
    for (int off = 8; off >= 1; off >>= 1) p = __fadd_rn(p, __shfl_xor_sync(0xffffffffu, p, off));
    return rsqrt_ftz(__fadd_rn(p, 1e-6f));
}

}  // namespace cfbi
