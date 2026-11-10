"""Non-finite and extreme inputs are outside the contract's value range, but the optimised kernels evaluate the same
operations as the reference, so even NaN / Inf propagation must be bit-identical (a mismatch would mean an optimisation
changed an operation order or a masked-out lane leaked into a valid one)."""
import pytest
import torch

from flash_kda.cfbi import alloc_state, decode, prefill, state_equal, state_mismatch

from helpers import make_case, n_diff


def _poison(case, kind, rng_seed=0):
    g = torch.Generator(device="cuda")
    g.manual_seed(rng_seed)
    T, H = case["q"].shape[1], case["q"].shape[2]

    def sprinkle(t, val, n=6):
        t = t.clone()
        for _ in range(n):
            t[0, int(torch.randint(0, T, (1,), device="cuda", generator=g)), int(torch.randint(0, H, (1,), device="cuda", generator=g)),
              int(torch.randint(0, 128, (1,), device="cuda", generator=g))] = val
        return t

    out = dict(case)
    if kind == "inf_q":
        out["q"] = sprinkle(case["q"], float("inf"))
    elif kind == "nan_k":
        out["k"] = sprinkle(case["k"], float("nan"))
    elif kind == "nan_v":
        out["v"] = sprinkle(case["v"], float("nan"))
    elif kind == "huge_g":
        out["g"] = sprinkle(case["g"], 3e38)
    elif kind == "neg_inf_g":
        out["g"] = sprinkle(case["g"], float("-inf"))
    elif kind == "zero_rows":
        out["k"] = case["k"].clone()
        out["k"][:, 5:40] = 0
        out["q"] = case["q"].clone()
        out["q"][:, 10:30] = 0
    return out


@pytest.mark.parametrize("kind", ["inf_q", "nan_k", "nan_v", "huge_g", "neg_inf_g", "zero_rows"])
def test_poisoned_inputs_match_reference(ext, kind):
    H, T = 2, 200
    case = _poison(make_case(T, H, "authors", seed=5), kind)
    res = {}
    for impl in (0, 3, 16):                       # 16: the K2 driven by a schedule built on the device
        if impl == 16 and not ext.resolve_impl(16, -1) & 16:
            continue
        st = alloc_state(1, H)
        out = torch.zeros_like(case["q"])
        o32 = torch.zeros(case["q"].shape, dtype=torch.float32, device="cuda")
        prefill(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], out, case["A_log"], case["dt_bias"],
                case["lower_bound"], st, out_fp32=o32, _impl=impl)
        # then a few decode steps from the poisoned state (row engine, packed tiles)
        for t in range(3):
            q1 = {k: case[k][:, t:t + 1].contiguous() for k in ("q", "k", "v", "g", "beta")}
            o1 = torch.zeros_like(q1["q"])
            decode(q1["q"], q1["k"], q1["v"], q1["g"], q1["beta"], case["scale"], o1, case["A_log"], case["dt_bias"],
                   case["lower_bound"], st, _impl=impl)
            out = torch.cat([out, o1], dim=1)
        res[impl] = (out, o32, st)
    for impl in res:
        if impl == 3:
            continue
        assert n_diff(res[impl][0], res[3][0]) == 0, f"{kind}/{impl}: bf16 outputs differ"
        assert n_diff(res[impl][1], res[3][1]) == 0, f"{kind}/{impl}: fp32 outputs differ"
        assert state_equal(res[impl][2], res[3][2]), f"{kind}/{impl}: {state_mismatch(res[impl][2], res[3][2])}"
