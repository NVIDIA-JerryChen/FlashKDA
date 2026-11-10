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

// Tests: the K2 schedule of a call. plan: int32 CUDA [plan_ints(N, H)]; cu_seqlens (int64 [N + 1]) or fixed_len tokens per sequence for N sequences;
// pos (int32, one per state slot) and slot_ids (int32 [N]) as in forward. Fills `plan` with the SeqPlan entries and, behind them, the schedule.
void probe_k2_schedule(torch::Tensor plan, int64_t H, int64_t N, int64_t fixed_len, std::optional<torch::Tensor> cu_seqlens,
                       std::optional<torch::Tensor> pos, std::optional<torch::Tensor> slot_ids, int64_t sms, int64_t k2_bv) {
    TORCH_CHECK(plan.is_cuda() && plan.dtype() == torch::kInt32 && plan.is_contiguous(), "plan must be a contiguous int32 CUDA tensor");
    TORCH_CHECK(H >= 1 && N >= 1 && N <= cfbi::k2_sched_capacity(H), "N must be in 1 .. the capacity of the schedule builder for H heads");
    TORCH_CHECK(plan.numel() >= cfbi::plan_ints(N, H), "plan too small: need plan_ints(N, H) ints");
    TORCH_CHECK(k2_bv == 0 || k2_bv == 8 || k2_bv == 16 || k2_bv == 32, "k2_bv must be 0, 8, 16 or 32");
    TORCH_CHECK(sms >= 0 && sms < 65536, "sms out of range");
    c10::cuda::CUDAGuard guard(plan.device());
    cfbi::K2SchedProbeArgs a{};
    if (cu_seqlens.has_value()) {
        TORCH_CHECK(cu_seqlens->is_cuda() && cu_seqlens->dtype() == torch::kInt64 && cu_seqlens->numel() == N + 1 && cu_seqlens->is_contiguous(),
                    "cu_seqlens must be a contiguous int64 CUDA vector with N + 1 entries");
        a.cu_seqlens = cu_seqlens->data_ptr<int64_t>();
    }
    if (pos.has_value()) {
        TORCH_CHECK(pos->is_cuda() && pos->dtype() == torch::kInt32 && pos->is_contiguous(), "pos must be a contiguous int32 CUDA tensor");
        a.pos = pos->data_ptr<int>();
    }
    if (slot_ids.has_value()) {
        TORCH_CHECK(slot_ids->is_cuda() && slot_ids->dtype() == torch::kInt32 && slot_ids->numel() == N && slot_ids->is_contiguous(),
                    "slot_ids must be a contiguous int32 CUDA vector with N entries");
        a.slot_ids = slot_ids->data_ptr<int>();
    }
    a.fixed_len = static_cast<int>(fixed_len);
    a.N = static_cast<int>(N);
    a.H = static_cast<int>(H);
    a.plan = plan.data_ptr<int>();
    a.sms = static_cast<int>(sms);
    a.k2_bv = static_cast<int>(k2_bv);
    cfbi::launch_probe_k2_schedule(a, cur_stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Tests: the start time (ns) and SM of the first n CTAs of the last device-scheduled K2 launch; only an extension built with CFBI_DYN_TEST_START_LOG has a
// log (the others return a zero count).
std::vector<torch::Tensor> dyn_start_log(int64_t n) {
    TORCH_CHECK(n >= 0 && n <= (1 << 19), "n out of range");
    auto t = torch::zeros({n}, torch::dtype(torch::kInt64));
    auto s = torch::zeros({n}, torch::dtype(torch::kInt32));
    const int got = cfbi::k2_dyn_test_log(reinterpret_cast<unsigned long long*>(t.data_ptr<int64_t>()), reinterpret_cast<unsigned*>(s.data_ptr<int>()), static_cast<int>(n));
    return {t, s, torch::tensor({static_cast<int64_t>(got)}, torch::dtype(torch::kInt64))};
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
             int64_t impl, std::optional<torch::Tensor> slot_ids, std::optional<torch::Tensor> ws_in,
             std::optional<torch::Tensor> plan_in, int64_t phase, int64_t k2_bv) {
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
    TORCH_CHECK(impl >= 0 && impl <= cfbi::IMPL_MAX,
                "impl must be 0..63 (bit 0: reference row engine, bit 1: reference K1/K2, bit 2: row engine for every row, "
                "bit 3: UMMA K2, bit 4: K2 driven by a schedule built on the device, bit 5: the old K2 kernels also where the default routing "
                "would choose the device-scheduled one)");
    TORCH_CHECK(!((impl & cfbi::IMPL_K2_DYN) && (impl & cfbi::IMPL_K2_OLD)), "impl bits 4 (device-scheduled K2) and 5 (old K2 kernels) exclude each other");
    TORCH_CHECK(phase >= 1 && phase <= 3, "phase must be 1 (K1), 2 (K2) or 3 (both)");
    TORCH_CHECK(k2_bv == 0 || k2_bv == 8 || k2_bv == 16 || k2_bv == 32 || (k2_bv >= 1000 && k2_bv <= 1000 + 65536),
                "k2_bv must be 0 (automatic), 8, 16, 32, or 1000 + x (testing: the first x (sequence, head) pairs use 32-column tiles, "
                "the others 16-column tiles)");
    TORCH_CHECK(phase == 3 || (mode == 0 && ws_in.has_value() && plan_in.has_value()),
                "a split call (phase 1 or 2) is mode 0 and needs the workspace and plan buffers");
    TORCH_CHECK(!(impl & cfbi::IMPL_K2_UMMA) || k2_bv == 0 || k2_bv == 16, "the UMMA K2 works on 16-column value slices (k2_bv 0 or 16)");
    TORCH_CHECK(!(impl & cfbi::IMPL_K2_DYN) || k2_bv < 1000,
                "the K2 driven by a device-built schedule (impl bit 4) takes k2_bv 0 (device policy), 8, 16 or 32 (forced width of every pair), not 1000 + x");
    TORCH_CHECK(lower_bound >= -5.0 && lower_bound <= 0.0, "lower_bound must be in [-5, 0]");
    const int64_t T_total = B * T;
    TORCH_CHECK(T_total < (int64_t(1) << 31) && H < 65536, "problem too large for the int32 token index / grid");
    // the kernels use 16-byte vector accesses on q/k (8 bf16) and on the fp32 state
    auto aligned16 = [](const torch::Tensor& t) { return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0; };
    TORCH_CHECK(aligned16(q) && aligned16(k) && aligned16(v) && aligned16(g), "q, k, v and g must be 16-byte aligned");

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
    // The state is a pool of S slots; sequence n of the call uses slot slot_ids[n] (slot_ids absent: slot n, S == N).
    int64_t S = N;
    const int* slot_ptr = nullptr;
    if (slot_ids.has_value()) {
        TORCH_CHECK(has_state, "slot_ids needs a state (E, rows, pos)");
        TORCH_CHECK(slot_ids->is_cuda() && slot_ids->dtype() == torch::kInt32 && slot_ids->dim() == 1 &&
                        slot_ids->numel() == N && slot_ids->is_contiguous(),
                    "slot_ids must be a contiguous int32 CUDA vector with one entry per sequence");
        slot_ptr = slot_ids->data_ptr<int>();
    }
    // Which kernels run on this device (sm < 90: the reference kernels; no room for K1's shared memory: the row
    // engine for every row). Bits do not depend on it, see cfbi_launch.h.
    c10::cuda::CUDAGuard guard(q.device());
    int eff_impl = cfbi::resolve_impl((int)impl);
    const int eff_mode = cfbi::resolve_mode((int)mode, eff_impl);
    // Default routing (see cfbi_launch.h): only a call that asks for nothing (impl 0, no forced tile width) and that the optimised kernels run as they are.
    // A plan buffer supplied by the caller that is too small for the schedule keeps the old kernels (an explicit impl 16 is an error instead, below).
    if (impl == 0 && k2_bv == 0 && eff_impl == 0 && eff_mode == 0 && cu_seqlens.has_value() &&
        (!plan_in.has_value() || plan_in->numel() >= cfbi::plan_ints(N, H)) && cfbi::k2_dyn_default_route(N, H, T_total, q.get_device())) {
        eff_impl |= cfbi::IMPL_K2_DYN;
    }
    if (has_state) {
        check_f32_cuda(*E, "E");
        check_f32_cuda(*rows, "rows");
        TORCH_CHECK(pos->is_cuda() && pos->dtype() == torch::kInt32 && pos->is_contiguous(), "pos must be int32 CUDA");
        if (slot_ids.has_value()) S = E->size(0);
        TORCH_CHECK(E->dim() == 4 && E->size(0) == S && E->size(1) == H && E->size(2) == 128 && E->size(3) == 128,
                    "E must be [N, H, 128, 128] (a pool [S, H, 128, 128] with slot_ids)");
        TORCH_CHECK(rows->dim() == 5 && rows->size(0) == S && rows->size(1) == H && rows->size(2) == 5 &&
                        rows->size(3) == 64 && rows->size(4) == 128,
                    "rows must be [N, H, 5, 64, 128] (a pool [S, ...] with slot_ids)");
        TORCH_CHECK(pos->dim() == 1 && pos->size(0) == S, "pos must be [N] (a pool [S] with slot_ids)");
        TORCH_CHECK(aligned16(*E) && aligned16(*rows), "E and rows must be 16-byte aligned");
    } else {
        TORCH_CHECK(mode == 0, "decode (mode 1) needs a state");
    }
    if (out_f32.has_value()) {
        check_f32_cuda(*out_f32, "out_f32");
        TORCH_CHECK(out_f32->sizes() == q.sizes(), "out_f32 must match q");
    }
    if (T_total == 0) return;   // nothing to compute; the state stays as it is

    // Training forward (no state) on the row-engine-only path: the row engine needs a state, so it gets a zeroed
    // temporary one (the empty initial state; nothing is persisted).
    torch::Tensor tmp_E, tmp_rows, tmp_pos;
    if (!has_state && eff_mode == 1) {
        tmp_E = torch::zeros({N, H, 128, 128}, q.options().dtype(torch::kFloat32));
        tmp_rows = torch::zeros({N, H, 5, 64, 128}, q.options().dtype(torch::kFloat32));
        tmp_pos = torch::zeros({N}, q.options().dtype(torch::kInt32));
    }
    TORCH_CHECK(phase == 3 || eff_mode == 0, "the context-parallel split (phase 1 / 2) needs the chunk kernels, which this device or "
                                             "implementation cannot run (row-engine-only schedule)");

    const int64_t max_chunks_alloc = (T_total + 63) / 64 + N;
    // the plan buffer also holds the schedule of the dynamic-schedule K2 when that K2 runs (cfbi::plan_ints, what workspace_sizes reports); a call beyond
    // the capacity of the schedule builder runs the old K2 kernels and needs the plan entries only
    const bool dyn_runs = eff_mode == 0 && (eff_impl & cfbi::IMPL_K2_DYN) && N <= cfbi::k2_sched_capacity(H);
    const int64_t plan_need = (N + 1) * cfbi::PLAN_INTS_PER_SEQ + (dyn_runs ? cfbi::k2_sched_ints(N, H) : 0);
    torch::Tensor plan = plan_in.has_value() ? *plan_in : torch::empty({plan_need}, q.options().dtype(torch::kInt32));
    TORCH_CHECK(plan.is_cuda() && plan.dtype() == torch::kInt32 && plan.is_contiguous() && plan.numel() >= plan_need,
                "plan buffer: contiguous int32 CUDA, workspace_sizes(...)[1] entries ((N + 1) * 8 plan entries, plus the schedule of the device-scheduled K2)");
    torch::Tensor ws;
    if (eff_mode == 0) {
        if (ws_in.has_value()) {
            ws = *ws_in;
            TORCH_CHECK(ws.is_cuda() && ws.dtype() == torch::kFloat32 && ws.is_contiguous() &&
                            ws.numel() >= max_chunks_alloc * H * cfbi::WS_FLOATS_PER_CHUNK_HEAD,
                        "workspace buffer too small: need (ceil(T/64) + N) * H * ", cfbi::WS_FLOATS_PER_CHUNK_HEAD, " floats");
        } else {
            ws = torch::empty({max_chunks_alloc * H * cfbi::WS_FLOATS_PER_CHUNK_HEAD}, q.options().dtype(torch::kFloat32));
        }
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
    a.E = has_state ? E->data_ptr<float>() : (tmp_E.defined() ? tmp_E.data_ptr<float>() : nullptr);
    a.rows = has_state ? rows->data_ptr<float>() : (tmp_rows.defined() ? tmp_rows.data_ptr<float>() : nullptr);
    a.pos = has_state ? pos->data_ptr<int>() : (tmp_pos.defined() ? tmp_pos.data_ptr<int>() : nullptr);
    a.slot_ids = slot_ptr;
    a.plan = plan.data_ptr<int>();
    a.ws = eff_mode == 0 ? ws.data_ptr<float>() : nullptr;
    a.mode = eff_mode;
    a.impl = eff_impl;
    a.phase = (int)phase;
    a.k2_bv = (int)k2_bv;
    cfbi::launch_forward(a, cur_stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Implementation bits that run for `impl` on `device` (< 0: the current one), see cfbi_launch.h.
int64_t resolve_impl_py(int64_t impl, int64_t device) {
    TORCH_CHECK(impl >= 0 && impl <= cfbi::IMPL_MAX, "impl must be 0..63");
    return cfbi::resolve_impl((int)impl, (int)device);
}

// {compute capability major, minor, shared-memory opt-in limit per block (bytes), K1 shared memory of the optimised
// kernels, K1 shared memory of the reference kernels} of `device`.
std::vector<int64_t> device_caps_py(int64_t device) {
    const cfbi::DeviceCaps c = cfbi::device_caps((int)device);
    return {c.major, c.minor, c.smem_optin, (int64_t)cfbi::k1_smem_bytes(0),
            (int64_t)cfbi::k1_smem_bytes(cfbi::IMPL_REF_CHUNK)};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("version", &cfbi::version, "CF-BI extension version");
    m.def("resolve_impl", &resolve_impl_py, "implementation bits that run on a device", py::arg("impl") = 0,
          py::arg("device") = -1);
    m.def("device_caps", &device_caps_py, "[cc major, cc minor, smem opt-in, K1 smem (optimised), K1 smem (reference)]",
          py::arg("device") = -1);
    m.def("forward", &forward, "CF-BI forward (prefill / training forward / decode)", py::arg("q"), py::arg("k"),
          py::arg("v"), py::arg("g"), py::arg("beta"), py::arg("scale"), py::arg("out"), py::arg("A_log"),
          py::arg("dt_bias"), py::arg("lower_bound"), py::arg("E") = py::none(), py::arg("rows") = py::none(),
          py::arg("pos") = py::none(), py::arg("cu_seqlens") = py::none(), py::arg("mode") = 0,
          py::arg("out_f32") = py::none(), py::arg("impl") = 0, py::arg("slot_ids") = py::none(), py::arg("ws") = py::none(), py::arg("plan") = py::none(),
          py::arg("phase") = 3, py::arg("k2_bv") = 0);
    m.def("workspace_sizes", [](int64_t T_total, int64_t N, int64_t H) {
        return std::make_pair(((T_total + 63) / 64 + N) * H * cfbi::WS_FLOATS_PER_CHUNK_HEAD, cfbi::plan_ints(N, H));
    }, "(workspace floats, plan ints) of a call; the plan buffer also holds the schedule of the device-scheduled K2", py::arg("T_total"), py::arg("N"), py::arg("H"));
    m.def("probe_k2_schedule", &probe_k2_schedule,
          "run the plan kernel and the K2 schedule builder (tests): fills `plan` (workspace_sizes(...)[1] ints); sms 0 = this device's SM count",
          py::arg("plan"), py::arg("H"), py::arg("N"), py::arg("fixed_len"), py::arg("cu_seqlens") = py::none(), py::arg("pos") = py::none(),
          py::arg("slot_ids") = py::none(), py::arg("sms") = 0, py::arg("k2_bv") = 0);
    m.def("dyn_start_log", &dyn_start_log, "(start ns, SM, [count]) of the first n CTAs of the last device-scheduled K2 (start-order probe build only)", py::arg("n"));
    m.def("sched_capacity", [](int64_t H) { return cfbi::k2_sched_capacity(H); }, "most sequences the K2 schedule builder takes for H heads", py::arg("H"));
    m.def("k2_route", &cfbi::k2_dyn_default_route, "whether the default implementation (impl 0) runs the device-scheduled K2 for a cu_seqlens call of N sequences, "
          "H heads and T_total tokens on a device (< 0: the current one)", py::arg("N"), py::arg("H"), py::arg("T_total"), py::arg("device") = -1);
    m.def("probe_dot3", &probe_dot3, "dot3 tile probe", py::arg("A"), py::arg("B"), py::arg("reverse_k") = false);
    m.def("probe_transc", &probe_transc, "ex2/tanh/rsqrt approximations", py::arg("x"), py::arg("which"));
    m.def("probe_prep", &probe_prep, "per-token prep probe", py::arg("q"), py::arg("k"), py::arg("g"),
          py::arg("beta"), py::arg("A_log"), py::arg("dt_bias"), py::arg("scale"), py::arg("lower_bound"));
}
