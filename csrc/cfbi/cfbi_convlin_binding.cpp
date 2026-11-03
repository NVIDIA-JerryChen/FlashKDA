// Python binding of the CF-BI conv / linear kernels (extension flash_kda_cfbi_conv_C).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include "cfbi/cfbi_convlin.h"

namespace {

cudaStream_t cur_stream() { return at::cuda::getCurrentCUDAStream().stream(); }

bool aligned(const torch::Tensor& t, uintptr_t a) { return reinterpret_cast<uintptr_t>(t.data_ptr()) % a == 0; }

// ---- short causal depthwise conv --------------------------------------------------------------------------
// x: bf16 [B, T, D] (or [T, D] = one sequence), channels contiguous, tokens at a constant row stride.
// weight: [D, W] (or [D, 1, W]) fp32 or bf16; bias: [D] fp32 or bf16 or None; state: bf16 [N, D, W-1] or None.
// cu_seqlens: int64 [N+1] with B == 1 (packed sequences). out: bf16, same shape as x, contiguous (written).
void conv_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor out, std::optional<torch::Tensor> bias,
                  std::optional<torch::Tensor> state, std::optional<torch::Tensor> cu_seqlens,
                  std::optional<int64_t> max_seqlen, bool silu, std::optional<torch::Tensor> out_f32) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16, "x must be a bf16 CUDA tensor");
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "x must be [B, T, D] or [T, D]");
    TORCH_CHECK(x.stride(-1) == 1 || x.size(-1) == 1, "x must have contiguous channels (stride(-1) == 1)");
    TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kBFloat16 && out.is_contiguous() && out.sizes() == x.sizes(),
                "out must be a contiguous bf16 tensor shaped like x");
    TORCH_CHECK(x.numel() == 0 || out.data_ptr() != x.data_ptr(), "out must not alias x (the kernel is not in-place)");
    const int64_t B = x.dim() == 3 ? x.size(0) : 1;
    const int64_t T = x.dim() == 3 ? x.size(1) : x.size(0);
    const int64_t D = x.size(-1);
    TORCH_CHECK(D < (int64_t(1) << 30), "D too large");

    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() &&
                    (weight.dtype() == torch::kFloat32 || weight.dtype() == torch::kBFloat16),
                "weight must be a contiguous fp32 or bf16 CUDA tensor");
    TORCH_CHECK((weight.dim() == 2 && weight.size(0) == D) || (weight.dim() == 3 && weight.size(0) == D && weight.size(1) == 1),
                "weight must be [D, W] or [D, 1, W]");
    const int64_t W = weight.size(-1);
    TORCH_CHECK(W >= 2 && W <= 8, "kernel width must be in 2..8");
    bool bias_bf16 = false;
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda() && bias->is_contiguous() && bias->numel() == D &&
                        (bias->dtype() == torch::kFloat32 || bias->dtype() == torch::kBFloat16),
                    "bias must be a contiguous fp32 or bf16 CUDA tensor with D elements");
        bias_bf16 = bias->dtype() == torch::kBFloat16;
    }
    int64_t N = B;
    const int64_t* cu_ptr = nullptr;
    if (cu_seqlens.has_value()) {
        TORCH_CHECK(B == 1, "B must be 1 when cu_seqlens is given");
        TORCH_CHECK(cu_seqlens->is_cuda() && cu_seqlens->dtype() == torch::kInt64 && cu_seqlens->dim() == 1 &&
                        cu_seqlens->numel() >= 2 && cu_seqlens->is_contiguous(),
                    "cu_seqlens must be a contiguous int64 CUDA vector with >= 2 entries");
        N = cu_seqlens->numel() - 1;
        cu_ptr = cu_seqlens->data_ptr<int64_t>();
    }
    TORCH_CHECK(N < (int64_t(1) << 30), "too many sequences");
    if (state.has_value()) {
        TORCH_CHECK(state->is_cuda() && state->dtype() == torch::kBFloat16 && state->is_contiguous(),
                    "state must be a contiguous bf16 CUDA tensor");
        TORCH_CHECK(state->dim() == 3 && state->size(0) == N && state->size(1) == D && state->size(2) == W - 1,
                    "state must be [N, D, W-1]");
    }
    if (out_f32.has_value()) {
        TORCH_CHECK(out_f32->is_cuda() && out_f32->dtype() == torch::kFloat32 && out_f32->is_contiguous() &&
                        out_f32->sizes() == x.sizes(),
                    "out_f32 must be a contiguous fp32 tensor shaped like x");
    }

    // token row stride of x (elements). With B > 1 the batch stride must be T rows.
    int64_t xs;
    if (x.dim() == 3) {
        xs = (T > 1) ? x.stride(1) : (B > 1 ? x.stride(0) : D);
        if (B > 1) TORCH_CHECK(x.stride(0) == T * xs, "x must be laid out as B sequences of T tokens at one row stride");
    } else {
        xs = (T > 1) ? x.stride(0) : D;
    }
    TORCH_CHECK(xs >= D, "x row stride must be at least D");

    c10::cuda::CUDAGuard guard(x.device());
    int64_t tokens = B * T;
    if (tokens == 0 || D == 0) return;   // nothing to compute; the state stays as it is

    int64_t max_len = max_seqlen.has_value() ? *max_seqlen : (cu_seqlens.has_value() ? tokens : T);
    TORCH_CHECK(max_len >= 0, "max_seqlen must be >= 0");
    const int64_t tiles = std::max<int64_t>(1, (max_len + cfbi::CONV_TILE_TOKENS - 1) / cfbi::CONV_TILE_TOKENS);
    TORCH_CHECK(tiles < (int64_t(1) << 30), "max_seqlen too large");

    cfbi::ConvLaunch a{};
    a.x = x.data_ptr();
    a.y = out.data_ptr();
    a.y_f32 = out_f32.has_value() ? out_f32->data_ptr<float>() : nullptr;
    a.weight = weight.data_ptr();
    a.bias = bias.has_value() ? bias->data_ptr() : nullptr;
    a.state = state.has_value() ? state->data_ptr() : nullptr;
    a.cu_seqlens = cu_ptr;
    a.x_stride = xs;
    a.y_stride = D;
    a.T = T;
    a.D = (int)D;
    a.W = (int)W;
    a.N = (int)N;
    a.tiles = (int)tiles;
    a.silu = silu ? 1 : 0;
    a.w_bf16 = weight.dtype() == torch::kBFloat16 ? 1 : 0;
    a.bias_bf16 = bias_bf16 ? 1 : 0;
    a.vec = (D % 2 == 0 && xs % 2 == 0 && aligned(x, 4) && aligned(out, 4)) ? 2 : 1;
    cfbi::launch_conv(a, cur_stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- batch-invariant linear -----------------------------------------------------------------------------------
// x: bf16 [M, K], weight: bf16 [N, K], bias: [N] fp32/bf16 or None, out: bf16 [M, N] (written). K % 16 == 0.
// cfg < 0 chooses the tile configuration by shape; 0..4 forces one (tests: every config gives the same bits).
void linear_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor out, std::optional<torch::Tensor> bias,
                    int64_t cfg, std::optional<torch::Tensor> out_f32) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16 && x.dim() == 2, "x must be a bf16 CUDA matrix [M, K]");
    TORCH_CHECK(weight.is_cuda() && weight.dtype() == torch::kBFloat16 && weight.dim() == 2, "weight must be bf16 [N, K]");
    TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kBFloat16 && out.dim() == 2, "out must be bf16 [M, N]");
    const int64_t M = x.size(0), K = x.size(1), N = weight.size(0);
    TORCH_CHECK(weight.size(1) == K, "weight must be [N, K] with the same K as x");
    TORCH_CHECK(out.size(0) == M && out.size(1) == N, "out must be [M, N]");
    TORCH_CHECK(K % 16 == 0 && K > 0, "K must be a positive multiple of 16 (the program is a chain of k16 MMAs)");
    TORCH_CHECK(M < (int64_t(1) << 30) && N < (int64_t(1) << 30) && K < (int64_t(1) << 30), "problem too large");
    TORCH_CHECK(x.stride(1) == 1 && weight.stride(1) == 1 && out.stride(1) == 1, "last dimensions must be contiguous");
    TORCH_CHECK(x.stride(0) % 8 == 0 && weight.stride(0) % 8 == 0, "row strides of x and weight must be multiples of 8");
    TORCH_CHECK(aligned(x, 16) && aligned(weight, 16), "x and weight must be 16-byte aligned");
    TORCH_CHECK(out.stride(0) >= N, "out row stride must be at least N");
    TORCH_CHECK(x.numel() == 0 || out.data_ptr() != x.data_ptr(), "out must not alias x");
    bool bias_bf16 = false;
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda() && bias->is_contiguous() && bias->numel() == N &&
                        (bias->dtype() == torch::kFloat32 || bias->dtype() == torch::kBFloat16),
                    "bias must be a contiguous fp32 or bf16 CUDA tensor with N elements");
        bias_bf16 = bias->dtype() == torch::kBFloat16;
    }
    TORCH_CHECK(cfg >= -1 && cfg < cfbi::LINEAR_NUM_CFG, "cfg must be -1 or 0..4");
    if (out_f32.has_value()) {
        TORCH_CHECK(out_f32->is_cuda() && out_f32->dtype() == torch::kFloat32 && out_f32->is_contiguous() &&
                        out_f32->dim() == 2 && out_f32->size(0) == M && out_f32->size(1) == N,
                    "out_f32 must be a contiguous fp32 [M, N] tensor");
    }
    c10::cuda::CUDAGuard guard(x.device());
    if (M == 0 || N == 0) return;
    cfbi::LinearLaunch a{};
    a.x = x.data_ptr();
    a.w = weight.data_ptr();
    a.bias = bias.has_value() ? bias->data_ptr() : nullptr;
    a.out = out.data_ptr();
    a.out_f32 = out_f32.has_value() ? out_f32->data_ptr<float>() : nullptr;
    a.M = (int)M;
    a.N = (int)N;
    a.K = (int)K;
    a.lda = x.stride(0);
    a.ldb = weight.stride(0);
    a.ldc = out.stride(0);
    a.bias_bf16 = bias_bf16 ? 1 : 0;
    a.vec_out = (out.stride(0) % 2 == 0 && aligned(out, 4)) ? 1 : 0;
    cfbi::launch_linear(a, (int)cfg, cur_stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t linear_pick_cfg(int64_t M, int64_t N) {
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    return cfbi::linear_pick_cfg((int)M, (int)N, sms);
}

// A: bf16 [ntiles, 16, K], B: bf16 [ntiles, K, 8] -> fp32 [ntiles, 16, 8]
torch::Tensor probe_mma_bf16(torch::Tensor A, torch::Tensor B, bool reverse_k) {
    TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kBFloat16 && A.is_contiguous() && A.dim() == 3 && A.size(1) == 16,
                "A must be contiguous bf16 [ntiles, 16, K]");
    const int64_t n = A.size(0), K = A.size(2);
    TORCH_CHECK(K % 16 == 0 && K > 0, "K must be a positive multiple of 16");
    TORCH_CHECK(B.is_cuda() && B.dtype() == torch::kBFloat16 && B.is_contiguous() && B.dim() == 3 && B.size(0) == n &&
                    B.size(1) == K && B.size(2) == 8,
                "B must be contiguous bf16 [ntiles, K, 8]");
    c10::cuda::CUDAGuard guard(A.device());
    auto D = torch::empty({n, 16, 8}, A.options().dtype(torch::kFloat32));
    if (n > 0) cfbi::launch_probe_mma_bf16(A.data_ptr(), B.data_ptr(), D.data_ptr<float>(), (int)n, (int)K, reverse_k ? 1 : 0, cur_stream());
    return D;
}

// y = ex2.approx.ftz(x) (which = 0) or rcp.approx.ftz(x) (which = 1), elementwise, fp32.
torch::Tensor probe_unary(torch::Tensor x, int64_t which) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32 && x.is_contiguous(), "x must be contiguous fp32 CUDA");
    TORCH_CHECK(which == 0 || which == 1, "which must be 0 (ex2) or 1 (rcp)");
    c10::cuda::CUDAGuard guard(x.device());
    auto y = torch::empty_like(x);
    if (x.numel() > 0) cfbi::launch_probe_unary(x.data_ptr<float>(), y.data_ptr<float>(), x.numel(), (int)which, cur_stream());
    return y;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_forward", &conv_forward, "CF-BI short causal conv + SiLU (in place state)", py::arg("x"), py::arg("weight"),
          py::arg("out"), py::arg("bias") = py::none(), py::arg("state") = py::none(), py::arg("cu_seqlens") = py::none(),
          py::arg("max_seqlen") = py::none(), py::arg("silu") = true, py::arg("out_f32") = py::none());
    m.def("linear_forward", &linear_forward, "CF-BI batch-invariant bf16 linear", py::arg("x"), py::arg("weight"),
          py::arg("out"), py::arg("bias") = py::none(), py::arg("cfg") = -1, py::arg("out_f32") = py::none());
    m.def("linear_pick_cfg", &linear_pick_cfg, "tile configuration chosen for (M, N)", py::arg("M"), py::arg("N"));
    m.def("probe_mma_bf16", &probe_mma_bf16, "m16n8k16 bf16 MMA chain probe", py::arg("A"), py::arg("B"),
          py::arg("reverse_k") = false);
    m.def("probe_unary", &probe_unary, "ex2 / rcp approximations", py::arg("x"), py::arg("which"));
}
