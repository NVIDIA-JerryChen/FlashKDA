"""Architecture handling: the runtime hardware self-check, the kernel selection per device, the row-engine-only path.

* ``verify_hardware`` / ``_guard``: the empirical battery of the tensor-core premise passes on this device, is cached,
  and (with a fake probe that violates the premise, or one that cannot tell two programs apart) makes ``_guard`` raise;
  the two environment switches behave as documented.
* ``resolve_impl``: below sm_90 the reference kernels run (the optimised ones use bulk copies / mbarriers); where K1's
  shared memory does not fit (sm_80: 163 KiB, sm_86 / 89 / 120: 99 KiB) every row goes through the row engine.
* the row-engine-only path (``_impl`` bit 2, what Ampere runs) gives the same bits as the chunk kernels.
"""
import pytest
import torch

from flash_kda.cfbi import alloc_state, api, forward_train, prefill, state_equal, state_mismatch, verify, verify_hardware

from helpers import make_case, n_diff

REF, ROWS_ONLY = 3, 4


@pytest.fixture()
def fresh_caches(monkeypatch):
    """Isolated verdict caches (the real verdicts are restored afterwards), and no env switches set."""
    monkeypatch.setattr(verify, "_RESULTS", {})
    monkeypatch.setattr(api, "_GUARDED", {})
    monkeypatch.delenv("FLASH_KDA_CFBI_ALLOW_UNVALIDATED_ARCH", raising=False)
    monkeypatch.delenv("FLASH_KDA_CFBI_SKIP_VERIFY", raising=False)


class FakeExt:
    """The real extension with a sabotaged ``probe_dot3``: what a device that violates the premise would look like."""

    def __init__(self, real, kind):
        self.real, self.kind = real, kind

    def resolve_impl(self, impl, device):
        return self.real.resolve_impl(impl, device)

    def probe_dot3(self, A, B, reverse_k=False):
        if self.kind == "insensitive":   # ignores the K order: the negative control cannot tell programs apart
            reverse_k = False
        D = self.real.probe_dot3(A, B, reverse_k)
        if self.kind == "slot":   # the result depends on the row slot of the tile (1 ulp for slots >= 8)
            D[:, 8:, :] = (D[:, 8:, :].contiguous().view(torch.int32) + 1).view(torch.float32)
        elif self.kind == "neighbour":   # the result depends on the data of the next row of the tile
            D = D + 1e-3 * A[:, :, 0].roll(-1, dims=1)[:, :, None]
        return D


# --------------------------------------------------------------------------------------------------------
# verify_hardware / _guard
# --------------------------------------------------------------------------------------------------------
def test_verify_hardware_passes_here(ext, fresh_caches):
    res = verify_hardware(force=True)
    print(f"\n[verify_hardware] {res}")
    assert res["ok"], res
    assert res["premise_ok"] and res["negative_control_ok"] and res["accuracy_ok"] and res["reason"] == ""
    assert res["mismatches"] == 0 and res["placements_checked"] >= 10000
    assert res["k_order_diff_fraction"] > 0.05 and res["max_rel_err"] < 2e-6
    assert res["elapsed_s"] < 1.0, "the battery is meant to take well under a second"
    assert res["capability"] == tuple(torch.cuda.get_device_capability())


def test_verify_hardware_is_cached(ext, fresh_caches, monkeypatch):
    first = verify_hardware()
    monkeypatch.setattr(verify, "_run_battery", lambda *a, **k: pytest.fail("the battery ran twice"))
    assert verify_hardware() == first
    assert verify_hardware(torch.device("cuda", torch.cuda.current_device())) == first


def test_guard_passes_here_and_caches(ext, fresh_caches, monkeypatch):
    dev = torch.device("cuda", torch.cuda.current_device())
    api._guard(dev)
    assert api._GUARDED[dev.index]
    monkeypatch.setattr(api, "verify_hardware", lambda *a, **k: pytest.fail("verified twice"))
    api._guard(dev)


@pytest.mark.parametrize("kind", ["slot", "neighbour"])
def test_fake_premise_violation_is_detected_and_guard_raises(ext, fresh_caches, monkeypatch, kind):
    monkeypatch.setattr(verify, "load_ext", lambda: FakeExt(ext, kind))
    dev = torch.device("cuda", torch.cuda.current_device())
    with pytest.raises(RuntimeError, match="premise could not be validated") as e:
        api._guard(dev)
    assert "changed with its tile slot" in str(e.value)
    res = verify_hardware(dev)   # the failing verdict is cached
    assert not res["ok"] and not res["premise_ok"] and res["mismatches"] > 0
    assert res["negative_control_ok"], "the sabotage must not look like a control failure"
    with pytest.raises(RuntimeError):   # and not silently cached as passed
        api._guard(dev)


def test_fake_insensitive_probe_is_rejected(ext, fresh_caches, monkeypatch):
    """A probe that cannot tell the K order apart cannot prove anything: the verdict is "not ok" even if the placements agree."""
    monkeypatch.setattr(verify, "load_ext", lambda: FakeExt(ext, "insensitive"))
    res = verify_hardware(force=True)
    assert res["premise_ok"] and not res["negative_control_ok"] and not res["ok"]
    assert "vacuous" in res["reason"]
    with pytest.raises(RuntimeError, match="vacuous"):
        api._guard(torch.device("cuda", torch.cuda.current_device()))


def test_fake_clean_probe_passes(ext, fresh_caches, monkeypatch):
    """The fake machinery itself does not break a healthy probe (so the failures above are caused by the sabotage)."""
    monkeypatch.setattr(verify, "load_ext", lambda: FakeExt(ext, "clean"))
    assert verify_hardware(force=True)["ok"]


def test_skip_verify_switch(ext, fresh_caches, monkeypatch):
    """FLASH_KDA_CFBI_SKIP_VERIFY=1: the battery does not even run (not even with a probe that would fail it)."""
    dev = torch.device("cuda", torch.cuda.current_device())
    monkeypatch.setattr(verify, "load_ext", lambda: FakeExt(ext, "slot"))
    monkeypatch.setattr(verify, "_run_battery", lambda *a, **k: pytest.fail("battery ran with SKIP_VERIFY=1"))
    monkeypatch.setenv("FLASH_KDA_CFBI_SKIP_VERIFY", "1")
    api._guard(dev)
    assert api._GUARDED[dev.index]


def test_allow_unvalidated_arch_switch(ext, fresh_caches, monkeypatch):
    """FLASH_KDA_CFBI_ALLOW_UNVALIDATED_ARCH=1 (the old switch): a failing verdict only warns, the verdict stays honest."""
    dev = torch.device("cuda", torch.cuda.current_device())
    monkeypatch.setattr(verify, "load_ext", lambda: FakeExt(ext, "slot"))
    monkeypatch.setenv("FLASH_KDA_CFBI_ALLOW_UNVALIDATED_ARCH", "1")
    with pytest.warns(RuntimeWarning, match="premise could not be validated"):
        api._guard(dev)
    assert api._GUARDED[dev.index] and not verify_hardware(dev)["ok"]


def test_graph_capture_needs_a_warm_verdict(ext, fresh_caches):
    """The battery synchronises, so it cannot run inside a CUDA graph capture: a clear error there, and no problem once the
    verdict is cached (every cfbi entry point calls _guard, also during a capture)."""
    dev = torch.device("cuda", torch.cuda.current_device())
    x = torch.zeros(4, device=dev)
    side = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side), torch.cuda.graph(graph, stream=side):
        x.add_(1)
        with pytest.raises(RuntimeError, match="CUDA graph capture"):
            api._guard(dev)
    torch.cuda.synchronize()
    verify_hardware()   # warm
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side), torch.cuda.graph(graph, stream=side):
        x.add_(1)
        api._guard(dev)
    torch.cuda.synchronize()


def test_pre_ampere_is_rejected_without_building(ext, fresh_caches, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 5))
    monkeypatch.setattr(verify, "load_ext", lambda: pytest.fail("must not build for sm_75"))
    res = verify_hardware(force=True)
    assert not res["ok"] and "TF32" in res["reason"]
    with pytest.raises(RuntimeError, match="TF32"):
        api._guard(torch.device("cuda", torch.cuda.current_device()))


# --------------------------------------------------------------------------------------------------------
# kernel selection
# --------------------------------------------------------------------------------------------------------
def test_resolve_impl_policy(ext):
    major, minor, smem, k1_opt, k1_ref = ext.device_caps(-1)
    assert (major, minor) == torch.cuda.get_device_capability()
    r0, r3 = ext.resolve_impl(0, -1), ext.resolve_impl(REF, -1)
    print(f"\n[resolve_impl] sm_{major}{minor}: smem opt-in {smem} B, K1 needs {k1_opt} (optimised) / {k1_ref} (reference); "
          f"impl 0 -> {r0}, impl 3 -> {r3}")
    if major < 9:
        assert r0 & REF == REF and r3 & REF == REF, "below sm_90 only the reference kernels can run"
    else:
        assert r0 & REF == 0, "sm_90+ runs the optimised kernels by default"
        assert r3 & REF == REF, "an explicit reference request must be honoured"
    for r in (r0, r3):
        need = k1_ref if r & 2 else k1_opt
        assert bool(r & ROWS_ONLY) == (smem < need), "row-engine-only exactly when K1's shared memory does not fit"
        assert ext.resolve_impl(r, -1) == r, "resolve_impl is idempotent"
    assert ext.resolve_impl(ROWS_ONLY, -1) & ROWS_ONLY, "an explicit row-engine-only request must be honoured"


def test_every_impl_value_is_accepted_and_same_bits(ext):
    """All eight ``impl`` values (resolved per device) give the same bits for a call with state and for the training
    forward: optimised / reference / row-engine-only are one program. On Ampere they all resolve to impl 7."""
    H = 3
    lens = [150, 1, 64, 129, 33, 0, 70]
    case = make_case(sum(lens), H, "heavy_tail", seed=13)
    cu = [0]
    for L in lens:
        cu.append(cu[-1] + L)
    cu_t = torch.tensor(cu, dtype=torch.int64, device="cuda")
    seen, results = set(), []
    for impl in range(8):
        eff = ext.resolve_impl(impl, -1)
        if eff in seen:
            continue
        seen.add(eff)
        st = alloc_state(len(lens), H)
        out = torch.zeros_like(case["q"])
        out32 = torch.zeros(case["q"].shape, dtype=torch.float32, device="cuda")
        prefill(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], out, case["A_log"], case["dt_bias"],
                case["lower_bound"], st, cu_t, out_fp32=out32, _impl=impl)
        ot32 = torch.zeros(case["q"].shape, dtype=torch.float32, device="cuda")
        ot = forward_train(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], case["A_log"],
                           case["dt_bias"], case["lower_bound"], cu_t, out_fp32=ot32, _impl=impl)
        results.append((eff, out, out32, st, ot, ot32))
    print(f"\n[impl matrix] distinct effective implementations on this device: {sorted(seen)}")
    e0, o0, f0, s0, t0, tf0 = results[0]
    for eff, o, f, s, t, tf in results[1:]:
        what = f"impl {eff} vs {e0}"
        assert n_diff(o0, o) == 0 and n_diff(f0, f) == 0, f"{what}: outputs differ"
        assert state_equal(s0, s), f"{what}: state differs {state_mismatch(s0, s)}"
        assert n_diff(t0, t) == 0 and n_diff(tf0, tf) == 0, f"{what}: training forward differs"
    assert n_diff(o0, t0) == 0 and n_diff(f0, tf0) == 0, "training forward != prefill"


def test_training_forward_batched_rows_only_path(ext):
    """No ``cu_seqlens`` (every batch entry one sequence) on the row-engine-only path, which needs a temporary state."""
    H, B, T = 2, 3, 130
    case = make_case(B * T, H, "authors", seed=14)
    case = {k: (v.reshape(B, T, *v.shape[2:]) if k in ("q", "k", "v", "g", "beta") else v) for k, v in case.items()}
    outs = []
    for impl in (0, ROWS_ONLY | REF):
        o32 = torch.zeros(case["q"].shape, dtype=torch.float32, device="cuda")
        o = forward_train(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], case["A_log"],
                          case["dt_bias"], case["lower_bound"], out_fp32=o32, _impl=impl)
        outs.append((o, o32))
    assert n_diff(outs[0][0], outs[1][0]) == 0 and n_diff(outs[0][1], outs[1][1]) == 0
