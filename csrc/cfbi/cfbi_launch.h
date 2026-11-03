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
    float* E;          // [N,H,128,128] fp32 state, or null (training: zero initial state, no write-back)
    float* rows;       // [N,H,5,64,128] fp32 open-chunk row cache, or null
    int* pos;          // [N] int32 fill of the open chunk, or null
    int* plan;         // scratch [(N+1) * PLAN_INTS_PER_SEQ] ints
    float* ws;         // scratch [(ceil(T_total/64) + N) * H * WS_FLOATS_PER_CHUNK_HEAD] floats (mode 0 only)
    int mode;          // 0: prefill / training schedule (row engine for head rows, K1+K2 for the rest)
                       // 1: row engine for every row (decode; also a differential test of K1/K2); the caller maps
                       //    mode 0 + IMPL_ROWS_ONLY to 1 (resolve_mode) and supplies a state (E, rows, pos) then
    int impl;          // IMPL_* bits as returned by resolve_impl (the launcher does not re-resolve)
};

// Implementation selector: the frozen v1 kernels in csrc/cfbi/ref/ are the bitwise oracle of the optimised ones.
// K1 and K2 share one bit because their workspace layouts differ between the implementations. 0 = the optimised
// kernels where the device can run them (see resolve_impl: otherwise the reference kernels are used).
constexpr int IMPL_REF_ROWS = 1;    // row engine (decode, head rows)
constexpr int IMPL_REF_CHUNK = 2;   // K1 + K2 (prefill / training forward)
constexpr int IMPL_ROWS_ONLY = 4;   // no K1/K2: every row goes through the row engine (mode 1 semantics even for mode 0)

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

}  // namespace cfbi
