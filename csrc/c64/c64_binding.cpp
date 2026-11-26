// torch binding of the chunk-64 kernels (module flash_kda_C64).
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "c64_params.h"

namespace {

using at::BFloat16;

void check_t(torch::Tensor const& t, char const* name, c10::ScalarType dtype, torch::Device dev) {
    TORCH_CHECK(t.device() == dev, name, " must be on ", dev);
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == dtype, name, " must be ", dtype);
}
void check_aligned(torch::Tensor const& t, char const* name) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, " must be 16-byte aligned");
}
uint16_t const* bf(torch::Tensor const& t) { return reinterpret_cast<uint16_t const*>(t.data_ptr()); }
uint16_t* bfm(torch::Tensor const& t) { return reinterpret_cast<uint16_t*>(t.data_ptr()); }

}  // namespace

int64_t fwd64_ws_size(int64_t total_tiles, int64_t H, int64_t N) {
    return int64_t(c64::fwd64_workspace_bytes(int(total_tiles), int(H), int(N)));
}

// One fwd64 call over a segment table. q, k, v, g, out: [T_total, H, 128] bf16; beta [T_total, H] bf16.
void fwd64(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, double scale,
           torch::Tensor out, torch::Tensor A_log, torch::Tensor dt_bias, double lower_bound,
           torch::Tensor seg_tok0, torch::Tensor seg_len, torch::Tensor seg_slot, int64_t total_tiles,
           torch::Tensor workspace, std::optional<torch::Tensor> state_in, std::optional<torch::Tensor> state_out,
           int64_t k2_cfg, std::optional<torch::Tensor> prof) {
    using namespace c64;
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    const auto dev = q.device();
    const at::cuda::OptionalCUDAGuard device_guard(dev);
    for (auto* t : {&q, &k, &v, &g, &out}) {
        check_t(*t, "q/k/v/g/out", torch::kBFloat16, dev);
        TORCH_CHECK(t->dim() == 3 && t->sizes() == q.sizes(), "q/k/v/g/out must be [T_total, H, 128] with one shape");
        check_aligned(*t, "q/k/v/g/out");
    }
    const int64_t T_total = q.size(0), H = q.size(1);
    TORCH_CHECK(q.size(2) == kHeadDim64, "only D == 128 is supported");
    check_t(beta, "beta", torch::kBFloat16, dev);
    TORCH_CHECK(beta.dim() == 2 && beta.size(0) == T_total && beta.size(1) == H, "beta must be [T_total, H]");
    check_t(A_log, "A_log", torch::kFloat32, dev);
    TORCH_CHECK(A_log.dim() == 1 && A_log.size(0) == H, "A_log must be [H]");
    check_t(dt_bias, "dt_bias", torch::kFloat32, dev);
    TORCH_CHECK(dt_bias.dim() == 2 && dt_bias.size(0) == H && dt_bias.size(1) == kHeadDim64, "dt_bias must be [H, 128]");
    const int64_t N = seg_len.numel();
    check_t(seg_tok0, "seg_tok0", torch::kInt64, dev);
    check_t(seg_len, "seg_len", torch::kInt32, dev);
    check_t(seg_slot, "seg_slot", torch::kInt32, dev);
    TORCH_CHECK(seg_tok0.numel() == N && seg_slot.numel() == N, "segment tables must have N entries");
    TORCH_CHECK(total_tiles >= 0 && total_tiles <= 2147483647 && N <= 2147483647, "too many tiles / segments");
    check_t(workspace, "workspace", torch::kUInt8, dev);
    check_aligned(workspace, "workspace");
    TORCH_CHECK(int64_t(workspace.numel()) >= fwd64_ws_size(total_tiles, H, N), "workspace is too small");
    if (state_in.has_value()) {
        check_t(*state_in, "state_in", torch::kBFloat16, dev);
        TORCH_CHECK(state_in->dim() == 4 && state_in->size(1) == H && state_in->size(2) == kHeadDim64 &&
                    state_in->size(3) == kHeadDim64, "state_in must be [P, H, 128, 128] bf16");
        check_aligned(*state_in, "state_in");
    }
    if (state_out.has_value()) {
        check_t(*state_out, "state_out", torch::kBFloat16, dev);
        TORCH_CHECK(state_out->dim() == 4 && state_out->size(1) == H && state_out->size(2) == kHeadDim64 &&
                    state_out->size(3) == kHeadDim64, "state_out must be [P, H, 128, 128] bf16");
        check_aligned(*state_out, "state_out");
    }

    FwdParams p{};
    p.q = bf(q); p.k = bf(k); p.v = bf(v); p.g = bf(g); p.beta = bf(beta); p.out = bfm(out);
    p.N = int(N); p.H = int(H); p.total_tiles = int(total_tiles);
    p.seg_tok0 = seg_tok0.data_ptr<int64_t>();
    p.seg_len = seg_len.data_ptr<int>();
    p.seg_slot = seg_slot.data_ptr<int>();
    p.A_log = A_log.data_ptr<float>();
    p.dt_bias = dt_bias.data_ptr<float>();
    p.scale = float(scale);
    p.gate_scale = float(lower_bound * 1.4426950408889634);
    fwd64_bind_workspace(p, workspace.data_ptr());
    p.st_in = state_in.has_value() ? bf(*state_in) : nullptr;
    p.st_out = state_out.has_value() ? bfm(*state_out) : nullptr;
    p.prof = prof.has_value() ? reinterpret_cast<long long*>(prof->data_ptr<int64_t>()) : nullptr;
    launch_fwd64(p, int(k2_cfg), at::cuda::getCurrentCUDAStream().stream());
}

void replay64(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, double scale,
              torch::Tensor out, torch::Tensor A_log, torch::Tensor dt_bias, double lower_bound,
              torch::Tensor state, torch::Tensor rows_k, torch::Tensor rows_v, torch::Tensor rows_g,
              torch::Tensor rows_beta, torch::Tensor pos, int64_t num_entries, int64_t tokens_per_entry,
              std::optional<torch::Tensor> tok_start, std::optional<torch::Tensor> tok_count,
              std::optional<torch::Tensor> out_row, std::optional<torch::Tensor> slot_ids,
              std::optional<torch::Tensor> final_out, std::optional<torch::Tensor> prof) {
    using namespace c64;
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    const auto dev = q.device();
    const at::cuda::OptionalCUDAGuard device_guard(dev);
    for (auto* t : {&q, &k, &v, &g}) {
        check_t(*t, "q/k/v/g", torch::kBFloat16, dev);
        TORCH_CHECK(t->dim() == 3 && t->sizes() == q.sizes(), "q/k/v/g must be [T_total, H, 128] with one shape");
        check_aligned(*t, "q/k/v/g");
    }
    const int64_t T_total = q.size(0), H = q.size(1), D = q.size(2);
    TORCH_CHECK(D == kHeadDim64, "only D == 128 is supported");
    check_t(beta, "beta", torch::kBFloat16, dev);
    TORCH_CHECK(beta.dim() == 2 && beta.size(0) == T_total && beta.size(1) == H, "beta must be [T_total, H]");
    check_t(out, "out", torch::kBFloat16, dev);
    TORCH_CHECK(out.dim() == 3 && out.size(1) == H && out.size(2) == D, "out must be [R, H, 128]");
    check_aligned(out, "out");
    check_t(A_log, "A_log", torch::kFloat32, dev);
    TORCH_CHECK(A_log.dim() == 1 && A_log.size(0) == H, "A_log must be [H]");
    check_t(dt_bias, "dt_bias", torch::kFloat32, dev);
    TORCH_CHECK(dt_bias.dim() == 2 && dt_bias.size(0) == H && dt_bias.size(1) == D, "dt_bias must be [H, 128]");
    check_t(state, "state", torch::kBFloat16, dev);
    TORCH_CHECK(state.dim() == 4 && state.size(1) == H && state.size(2) == D && state.size(3) == D,
                "state must be [P, H, 128, 128] bf16");
    check_aligned(state, "state");
    const int64_t P = state.size(0);
    for (auto* t : {&rows_k, &rows_v, &rows_g}) {
        check_t(*t, "rows_k/rows_v/rows_g", torch::kBFloat16, dev);
        TORCH_CHECK(t->dim() == 4 && t->size(0) == P && t->size(1) == H && t->size(2) == kChunk64 && t->size(3) == D,
                    "rows_k/rows_v/rows_g must be [P, H, 64, 128]");
        check_aligned(*t, "rows_k/rows_v/rows_g");
    }
    check_t(rows_beta, "rows_beta", torch::kBFloat16, dev);
    TORCH_CHECK(rows_beta.dim() == 3 && rows_beta.size(0) == P && rows_beta.size(1) == H && rows_beta.size(2) == kChunk64,
                "rows_beta must be [P, H, 64]");
    check_t(pos, "pos", torch::kInt32, dev);
    TORCH_CHECK(pos.dim() == 2 && pos.size(0) == P && pos.size(1) == H, "pos must be [P, H] int32");

    const int64_t N = num_entries;
    TORCH_CHECK(N >= 0 && N <= 2147483647, "num_entries out of range");
    TORCH_CHECK(tok_start.has_value() == tok_count.has_value(), "tok_start and tok_count go together");
    const bool peek = final_out.has_value();
    if (tok_start.has_value()) {
        check_t(*tok_start, "tok_start", torch::kInt64, dev);
        TORCH_CHECK(tok_start->dim() == 1 && tok_start->size(0) == N, "tok_start must be [N] int64");
        check_t(*tok_count, "tok_count", torch::kInt32, dev);
        TORCH_CHECK(tok_count->dim() == 1 && tok_count->size(0) == N, "tok_count must be [N] int32");
    } else if (!peek) {
        TORCH_CHECK(tokens_per_entry >= 0 && N * tokens_per_entry <= T_total,
                    "without tok_start the call must hold N x tokens_per_entry tokens");
        TORCH_CHECK(out.size(0) >= N * tokens_per_entry || out_row.has_value(), "out has too few rows");
    }
    if (out_row.has_value()) {
        check_t(*out_row, "out_row", torch::kInt64, dev);
        TORCH_CHECK(out_row->dim() == 1 && out_row->size(0) == N, "out_row must be [N] int64");
    }
    if (slot_ids.has_value()) {
        check_t(*slot_ids, "slot_ids", torch::kInt32, dev);
        TORCH_CHECK(slot_ids->dim() == 1 && slot_ids->size(0) == N, "slot_ids must be [N] int32");
    } else {
        TORCH_CHECK(N <= P, "without slot_ids entry n uses slot n, so N must be <= P");
    }
    if (peek) {
        check_t(*final_out, "final_out", torch::kBFloat16, dev);
        TORCH_CHECK(final_out->dim() == 4 && final_out->size(0) == N && final_out->size(1) == H &&
                    final_out->size(2) == D && final_out->size(3) == D, "final_out must be [N, H, 128, 128] bf16");
        check_aligned(*final_out, "final_out");
    }

    ReplayParams p{};
    p.q = bf(q); p.k = bf(k); p.v = bf(v); p.g = bf(g); p.beta = bf(beta); p.out = bfm(out);
    p.tok_start = tok_start.has_value() ? tok_start->data_ptr<int64_t>() : nullptr;
    p.tok_count = tok_count.has_value() ? tok_count->data_ptr<int>() : nullptr;
    p.tokens_per_entry = int(tokens_per_entry);
    p.out_row = out_row.has_value() ? out_row->data_ptr<int64_t>() : nullptr;
    p.slot_ids = slot_ids.has_value() ? slot_ids->data_ptr<int>() : nullptr;
    p.num_slots = int(P);
    p.state = bfm(state); p.rows_k = bfm(rows_k); p.rows_v = bfm(rows_v); p.rows_g = bfm(rows_g);
    p.rows_beta = bfm(rows_beta);
    p.pos = pos.data_ptr<int>();
    p.A_log = A_log.data_ptr<float>();
    p.dt_bias = dt_bias.data_ptr<float>();
    p.scale = float(scale);
    p.gate_scale = float(lower_bound * 1.4426950408889634);
    p.H = int(H);
    p.final_out = peek ? bfm(*final_out) : nullptr;
    p.prof = prof.has_value() ? reinterpret_cast<long long*>(prof->data_ptr<int64_t>()) : nullptr;
    launch_replay64(p, int(N), at::cuda::getCurrentCUDAStream().stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd64", &fwd64, "chunk-64 bf16 prefill over a segment table (CUDA)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"), py::arg("beta"), py::arg("scale"), py::arg("out"),
          py::arg("A_log"), py::arg("dt_bias"), py::arg("lower_bound"),
          py::arg("seg_tok0"), py::arg("seg_len"), py::arg("seg_slot"), py::arg("total_tiles"), py::arg("workspace"),
          py::arg("state_in") = py::none(), py::arg("state_out") = py::none(), py::arg("k2_cfg") = 0, py::arg("prof") = py::none());
    m.def("fwd64_ws_size", &fwd64_ws_size, "workspace bytes of fwd64", py::arg("total_tiles"), py::arg("H"), py::arg("N"));
    m.def("replay64", &replay64, "chunk-64 open-chunk replay (CUDA)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"), py::arg("beta"), py::arg("scale"), py::arg("out"),
          py::arg("A_log"), py::arg("dt_bias"), py::arg("lower_bound"),
          py::arg("state"), py::arg("rows_k"), py::arg("rows_v"), py::arg("rows_g"), py::arg("rows_beta"), py::arg("pos"),
          py::arg("num_entries"), py::arg("tokens_per_entry") = 0,
          py::arg("tok_start") = py::none(), py::arg("tok_count") = py::none(),
          py::arg("out_row") = py::none(), py::arg("slot_ids") = py::none(), py::arg("final_out") = py::none(),
          py::arg("prof") = py::none());
    m.def("smem_bytes", []() {
        return py::make_tuple(c64::fwd64_k1_smem_bytes(), c64::fwd64_k2_smem_bytes(0), c64::replay64_smem_bytes());
    }, "dynamic shared memory of (fwd64 K1, fwd64 K2 cfg 0, replay64)");
}
