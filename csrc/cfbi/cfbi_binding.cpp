#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include "cfbi/cfbi_launch.h"

namespace {

cudaStream_t cur_stream() { return at::cuda::getCurrentCUDAStream().stream(); }

void check_f32_cuda(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

torch::Tensor probe_dot3(torch::Tensor A, torch::Tensor B, bool reverse_k) {
    check_f32_cuda(A, "A");
    check_f32_cuda(B, "B");
    TORCH_CHECK(A.dim() == 3 && A.size(1) == 16, "A must be [ntiles, 16, K]");
    const int64_t n = A.size(0), K = A.size(2);
    TORCH_CHECK(K % 8 == 0 && K > 0, "K must be a positive multiple of 8");
    TORCH_CHECK(B.dim() == 3 && B.size(0) == n && B.size(1) == K && B.size(2) == 8, "B must be [ntiles, K, 8]");
    c10::cuda::CUDAGuard guard(A.device());
    auto D = torch::empty({n, 16, 8}, A.options());
    cfbi::launch_probe_dot3(A.data_ptr<float>(), B.data_ptr<float>(), D.data_ptr<float>(), (int)n, (int)K,
                            reverse_k ? 1 : 0, cur_stream());
    return D;
}

torch::Tensor probe_transc(torch::Tensor x, int64_t which) {
    check_f32_cuda(x, "x");
    TORCH_CHECK(which >= 0 && which <= 2, "which must be 0 (ex2), 1 (tanh) or 2 (rsqrt)");
    c10::cuda::CUDAGuard guard(x.device());
    auto y = torch::empty_like(x);
    cfbi::launch_probe_transc(x.data_ptr<float>(), y.data_ptr<float>(), x.numel(), (int)which, cur_stream());
    return y;
}

// q,k,g: bf16 [rows,128]; beta: bf16 [rows]; A_log: fp32 [1]; dt_bias: fp32 [128].
std::vector<torch::Tensor> probe_prep(torch::Tensor q, torch::Tensor k, torch::Tensor g, torch::Tensor beta,
                                      torch::Tensor A_log, torch::Tensor dt_bias, double scale, double lower_bound) {
    for (const auto* t : {&q, &k, &g, &beta}) {
        TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->dtype() == torch::kBFloat16, "q,k,g,beta must be contiguous bf16 CUDA");
    }
    check_f32_cuda(A_log, "A_log");
    check_f32_cuda(dt_bias, "dt_bias");
    TORCH_CHECK(q.dim() == 2 && q.size(1) == 128, "q must be [rows, 128]");
    const int64_t rows = q.size(0);
    TORCH_CHECK(k.sizes() == q.sizes() && g.sizes() == q.sizes(), "k,g must match q");
    TORCH_CHECK(beta.numel() == rows, "beta must be [rows]");
    TORCH_CHECK(A_log.numel() == 1 && dt_bias.numel() == 128, "A_log [1], dt_bias [128]");
    c10::cuda::CUDAGuard guard(q.device());
    auto f32 = q.options().dtype(torch::kFloat32);
    auto khat = torch::empty({rows, 128}, f32), qhat = torch::empty({rows, 128}, f32);
    auto a2 = torch::empty({rows, 128}, f32), b = torch::empty({rows}, f32);
    const float scale_f = static_cast<float>(scale);
    const float scale2 = static_cast<float>(lower_bound * 1.4426950408889634);
    cfbi::launch_probe_prep(q.data_ptr(), k.data_ptr(), g.data_ptr(), beta.data_ptr(), A_log.data_ptr<float>(),
                            dt_bias.data_ptr<float>(), scale_f, scale2, khat.data_ptr<float>(),
                            qhat.data_ptr<float>(), a2.data_ptr<float>(), b.data_ptr<float>(), (int)rows,
                            cur_stream());
    return {khat, qhat, a2, b};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("version", &cfbi::version, "CF-BI extension version");
    m.def("probe_dot3", &probe_dot3, "dot3 tile probe", py::arg("A"), py::arg("B"), py::arg("reverse_k") = false);
    m.def("probe_transc", &probe_transc, "ex2/tanh/rsqrt approximations", py::arg("x"), py::arg("which"));
    m.def("probe_prep", &probe_prep, "per-token prep probe", py::arg("q"), py::arg("k"), py::arg("g"),
          py::arg("beta"), py::arg("A_log"), py::arg("dt_bias"), py::arg("scale"), py::arg("lower_bound"));
}
