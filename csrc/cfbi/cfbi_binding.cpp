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
    TORCH_CHECK(which >= 0 && which <= 3, "which must be 0 (ex2), 1 (tanh), 2 (rsqrt) or 3 (rcp)");
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

void check_bf16_cuda(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// One entry point for prefill, the training forward and decode.
//   q,k,v,g: bf16 [B,T,H,128]; beta: bf16 [B,T,H] (raw logits); out: bf16 [B,T,H,128] (written in place).
//   State (E fp32 [N,H,128,128], rows fp32 [N,H,5,64,128], pos int32 [N]) is all-or-none: given => prefill/decode
//   (state updated in place); absent => training forward from the empty state (nothing is persisted).
//   cu_seqlens (int64 [N+1], B must be 1) selects varlen mode; without it every batch entry is one sequence.
//   mode 0: row engine for the head rows (finishing the open chunk) + K1/K2 for the rest; mode 1: row engine
//   for every row (decode, and a differential test of K1/K2).
void forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, double scale,
             torch::Tensor out, torch::Tensor A_log, torch::Tensor dt_bias, double lower_bound,
             std::optional<torch::Tensor> E, std::optional<torch::Tensor> rows, std::optional<torch::Tensor> pos,
             std::optional<torch::Tensor> cu_seqlens, int64_t mode, std::optional<torch::Tensor> out_f32,
             int64_t impl) {
    for (const auto* t : {&q, &k, &v, &g, &out}) check_bf16_cuda(*t, "q/k/v/g/out");
    check_bf16_cuda(beta, "beta");
    check_f32_cuda(A_log, "A_log");
    check_f32_cuda(dt_bias, "dt_bias");
    TORCH_CHECK(q.dim() == 4 && q.size(3) == 128, "q must be [B, T, H, 128]");
    TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes() && g.sizes() == q.sizes() && out.sizes() == q.sizes(),
                "k, v, g, out must match q");
    const int64_t B = q.size(0), T = q.size(1), H = q.size(2);
    TORCH_CHECK(beta.dim() == 3 && beta.size(0) == B && beta.size(1) == T && beta.size(2) == H,
                "beta must be [B, T, H]");
    TORCH_CHECK(A_log.dim() == 1 && A_log.size(0) == H, "A_log must be [H]");
    TORCH_CHECK(dt_bias.dim() == 2 && dt_bias.size(0) == H && dt_bias.size(1) == 128, "dt_bias must be [H, 128]");
    TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 or 1");
    TORCH_CHECK(impl >= 0 && impl <= 3, "impl must be 0..3 (bit 0: reference row engine, bit 1: reference K1/K2)");
    TORCH_CHECK(lower_bound >= -5.0 && lower_bound <= 0.0, "lower_bound must be in [-5, 0]");
    const int64_t T_total = B * T;
    TORCH_CHECK(T_total < (int64_t(1) << 31) && H < 65536, "problem too large for the int32 token index / grid");
    // the kernels use 16-byte vector accesses on q/k (8 bf16) and on the fp32 state
    auto aligned16 = [](const torch::Tensor& t) { return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0; };
    TORCH_CHECK(aligned16(q) && aligned16(k), "q and k must be 16-byte aligned");

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
    TORCH_CHECK(N < 65536, "too many sequences for the grid");

    const bool has_state = E.has_value();
    TORCH_CHECK(has_state == rows.has_value() && has_state == pos.has_value(),
                "E, rows and pos must be given together");
    if (has_state) {
        check_f32_cuda(*E, "E");
        check_f32_cuda(*rows, "rows");
        TORCH_CHECK(pos->is_cuda() && pos->dtype() == torch::kInt32 && pos->is_contiguous(), "pos must be int32 CUDA");
        TORCH_CHECK(E->dim() == 4 && E->size(0) == N && E->size(1) == H && E->size(2) == 128 && E->size(3) == 128,
                    "E must be [N, H, 128, 128]");
        TORCH_CHECK(rows->dim() == 5 && rows->size(0) == N && rows->size(1) == H && rows->size(2) == 5 &&
                        rows->size(3) == 64 && rows->size(4) == 128,
                    "rows must be [N, H, 5, 64, 128]");
        TORCH_CHECK(pos->dim() == 1 && pos->size(0) == N, "pos must be [N]");
        TORCH_CHECK(aligned16(*E) && aligned16(*rows), "E and rows must be 16-byte aligned");
    } else {
        TORCH_CHECK(mode == 0, "decode (mode 1) needs a state");
    }
    if (out_f32.has_value()) {
        check_f32_cuda(*out_f32, "out_f32");
        TORCH_CHECK(out_f32->sizes() == q.sizes(), "out_f32 must match q");
    }
    c10::cuda::CUDAGuard guard(q.device());
    if (T_total == 0) return;   // nothing to compute; the state stays as it is

    auto plan = torch::empty({(N + 1) * cfbi::PLAN_INTS_PER_SEQ}, q.options().dtype(torch::kInt32));
    torch::Tensor ws;
    if (mode == 0) {
        const int64_t max_chunks = (T_total + 63) / 64 + N;
        ws = torch::empty({max_chunks * H * cfbi::WS_FLOATS_PER_CHUNK_HEAD}, q.options().dtype(torch::kFloat32));
    }

    cfbi::ForwardArgs a{};
    a.q = q.data_ptr(); a.k = k.data_ptr(); a.v = v.data_ptr(); a.g = g.data_ptr(); a.beta = beta.data_ptr();
    a.out = out.data_ptr();
    a.out_f32 = out_f32.has_value() ? out_f32->data_ptr<float>() : nullptr;
    a.A_log = A_log.data_ptr<float>();
    a.dt_bias = dt_bias.data_ptr<float>();
    a.scale = static_cast<float>(scale);
    a.scale2 = static_cast<float>(lower_bound * 1.4426950408889634);
    a.H = (int)H;
    a.N = (int)N;
    a.fixed_len = (int)T;
    a.T_total = T_total;
    a.cu_seqlens = cu_ptr;
    a.E = has_state ? E->data_ptr<float>() : nullptr;
    a.rows = has_state ? rows->data_ptr<float>() : nullptr;
    a.pos = has_state ? pos->data_ptr<int>() : nullptr;
    a.plan = plan.data_ptr<int>();
    a.ws = mode == 0 ? ws.data_ptr<float>() : nullptr;
    a.mode = (int)mode;
    a.impl = (int)impl;
    cfbi::launch_forward(a, cur_stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("version", &cfbi::version, "CF-BI extension version");
    m.def("forward", &forward, "CF-BI forward (prefill / training forward / decode)", py::arg("q"), py::arg("k"),
          py::arg("v"), py::arg("g"), py::arg("beta"), py::arg("scale"), py::arg("out"), py::arg("A_log"),
          py::arg("dt_bias"), py::arg("lower_bound"), py::arg("E") = py::none(), py::arg("rows") = py::none(),
          py::arg("pos") = py::none(), py::arg("cu_seqlens") = py::none(), py::arg("mode") = 0,
          py::arg("out_f32") = py::none(), py::arg("impl") = 0);
    m.def("probe_dot3", &probe_dot3, "dot3 tile probe", py::arg("A"), py::arg("B"), py::arg("reverse_k") = false);
    m.def("probe_transc", &probe_transc, "ex2/tanh/rsqrt approximations", py::arg("x"), py::arg("which"));
    m.def("probe_prep", &probe_prep, "per-token prep probe", py::arg("q"), py::arg("k"), py::arg("g"),
          py::arg("beta"), py::arg("A_log"), py::arg("dt_bias"), py::arg("scale"), py::arg("lower_bound"));
}
