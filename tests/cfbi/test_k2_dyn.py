"""The K2 driven by a schedule built on the device (impl bit 16): wiring, the kernels that launch, the default routing, bit identity with the frozen
reference kernels, split K1 / K2 calls and CUDA graphs.

A schedule-only change must not move a single bit: every (sequence, head, value slice) is still one CTA, chunk by chunk, with the same arithmetic.
Every scenario therefore compares ``_impl=16`` with ``_impl=3`` (the frozen reference kernels) bit for bit: bf16 and fp32 outputs, the state E, the valid
row-cache planes and ``pos``. A comparison is only meaningful if the new kernel really ran, so every device run is wrapped in a profiler check that
rejects a run in which ``k2_dyn_kernel`` did not launch (a silently dropped bit would otherwise pass vacuously).
"""
import collections
import itertools
import random

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from flash_kda.cfbi import alloc_state, decode, forward_train, prefill, prefill_k1, prefill_k2, state_equal, state_mismatch

import k2_policy as kp
from helpers import make_case, n_diff
from test_k2_policy import B200_ROUTING_MEASUREMENTS
from test_k2_sched import assert_matches, assert_old_layout_header, probe
from test_vs_ref import assert_same_results, random_takes, run_chain
from varlen_corpus import CORPUS_V

REF, UMMA, DYN, OLD, ROWS_ONLY, REF_CHUNK = 3, 8, 16, 32, 4, 2
OLD_K2 = ("k2_kernel", "k2_kernel_mixed")


@pytest.fixture(autouse=True)
def _dyn_runs_here(ext):
    if not ext.resolve_impl(DYN, -1) & DYN:
        pytest.skip("the device cannot run the device-scheduled K2 (impl bit 16 is dropped here)")


def _sms():
    return torch.cuda.get_device_properties(0).multi_processor_count


# ---- observing the launches --------------------------------------------------------------------------------------------------------------------------
def _base(name):
    """``void cfbi::k2_kernel<16>(cfbi::Params)`` -> ``k2_kernel`` (template arguments stripped, as the benchmark tool does)."""
    name = name.split("(")[0].split("<")[0].strip()
    if name.startswith("void "):
        name = name[5:]
    return name.replace("cfbi::", "", 1)


def launched(fn, warm=True):
    """Kernel launch counts of one call of ``fn`` (kernel base name -> launches); ``fn`` is run once before to take JIT / attribute set-up out of the profile."""
    if warm:
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    cnt = collections.Counter()
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            cnt[_base(ev.name)] += 1
    return cnt


def require_dyn(cnt, builder=True):
    """The device-scheduled K2 launched and none of the old K2 kernels did (a comparison without it is vacuous)."""
    assert cnt["k2_dyn_kernel"] >= 1, "the device-scheduled K2 did not launch: %s" % dict(cnt)
    assert not any(cnt[k] for k in OLD_K2), "an old K2 kernel launched next to it: %s" % dict(cnt)
    assert bool(cnt["k2_sched_kernel"]) == builder, "schedule builder launches: %s" % dict(cnt)


def require_old(cnt):
    assert cnt["k2_dyn_kernel"] == 0 and cnt["k2_sched_kernel"] == 0, dict(cnt)
    assert sum(cnt[k] for k in OLD_K2) >= 1, dict(cnt)


def _cu(lens):
    return torch.tensor([0] + list(itertools.accumulate(lens)), dtype=torch.int64, device="cuda")


def varlen_fn(lens, H, impl, k2_bv=0, seed=1):
    """A call that prefills a fresh state with the sequences ``lens`` (one varlen call)."""
    case = make_case(sum(lens), H, "authors", seed=seed)
    cu = _cu(lens)
    out = torch.zeros_like(case["q"])

    def fn():
        prefill(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], out, case["A_log"], case["dt_bias"], case["lower_bound"],
                alloc_state(len(lens), H), cu, _impl=impl, _k2_bv=k2_bv)
    return fn


def fixed_fn(B, L, H, impl, seed=1):
    """A fixed-length call (no cu_seqlens): B sequences of L tokens."""
    case = make_case(B * L, H, "authors", seed=seed)
    q, k, v, g, beta = (case[x].reshape(B, L, *case[x].shape[2:]) for x in ("q", "k", "v", "g", "beta"))
    out = torch.zeros_like(q)

    def fn():
        prefill(q, k, v, g, beta, case["scale"], out, case["A_log"], case["dt_bias"], case["lower_bound"], alloc_state(B, H), None, _impl=impl)
    return fn


def check_dyn_ran(cnt):
    """A comparison is rejected as vacuous unless the device-scheduled K2 launched (and no old K2 kernel)."""
    assert cnt["k2_dyn_kernel"] >= 1 and not any(cnt[k] for k in OLD_K2), "vacuous comparison: %s" % dict(cnt)


def run_dyn(case, lens, takes, H, k2_bv=0):
    """``run_chain`` with ``_impl=16``, rejected as vacuous unless the device-scheduled K2 launched."""
    box = []
    check_dyn_ran(launched(lambda: box.append(run_chain(case, lens, takes, H, DYN, k2_bv=k2_bv)), warm=False))
    return box[-1]


# ---- wiring: the bit, the precedence, the error cases ----------------------------------------------------------------------------------------------
def test_resolve_impl_precedence(ext):
    assert ext.resolve_impl(0, -1) & DYN == 0, "impl 0 never sets the bit"
    assert ext.resolve_impl(DYN, -1) & DYN, "the bit stays on a device that can run it"
    assert ext.resolve_impl(DYN | 1, -1) & DYN, "the reference row engine does not matter to K2"
    for dropped in (DYN | REF_CHUNK, DYN | REF, DYN | ROWS_ONLY, DYN | UMMA, DYN | UMMA | 1, 31):
        assert ext.resolve_impl(dropped, -1) & DYN == 0, dropped
    if torch.cuda.get_device_capability()[0] == 10:
        assert ext.resolve_impl(DYN | UMMA, -1) & UMMA, "when both are requested the UMMA K2 runs"
    for v in range(32):
        r = ext.resolve_impl(v, -1)
        assert ext.resolve_impl(r, -1) == r, v
        assert ext.resolve_impl(v | OLD, -1) == r, "bit 5 (the old K2 kernels) only matters to the default routing, not to what resolve_impl returns"


def test_invalid_values_raise(ext):
    fn = lambda impl, bv=0: varlen_fn([100, 70], 2, impl, bv)()
    for bad in (64, 63, DYN | OLD, -1):                # beyond bit 5, and the two K2 choices of bits 4 and 5 together
        with pytest.raises((RuntimeError, ValueError)):
            fn(bad)
    with pytest.raises((RuntimeError, ValueError)):
        ext.resolve_impl(64, -1)
    fn(OLD)                                          # the old K2 kernels, explicitly
    fn(OLD, 1003)                                    # ... which keep their test hook
    for bv in (1000, 1001, 1037):
        with pytest.raises((RuntimeError, ValueError), match="1000"):
            fn(DYN, bv)
    for bv in (0, 8, 16, 32):
        fn(DYN, bv)                                 # the forced widths are accepted
    fn(0, 1003)                                      # the test hook of the old kernels is untouched without the bit


def test_a_comparison_without_a_dyn_launch_is_rejected_as_vacuous():
    lens = [900, 300, 700, 1300, 5, 199, 1000, 450, 640, 77]
    check_dyn_ran(launched(varlen_fn(lens, 6, DYN)))
    with pytest.raises(AssertionError, match="vacuous"):
        check_dyn_ran(launched(varlen_fn(lens, 6, OLD)))                            # impl 32: the old kernels ran, not the new one
    with pytest.raises(AssertionError, match="vacuous"):
        check_dyn_ran(launched(varlen_fn(lens, 6, DYN | UMMA)))                      # the request was dropped (UMMA wins)


# ---- which kernels launch ----------------------------------------------------------------------------------------------------------------------------
def test_launched_kernel_set_on_the_named_shapes():
    sms = _sms()
    # an explicit bit launches the schedule builder and the device-scheduled K2, on fixed-length and on cu_seqlens calls, and none of the old kernels
    require_dyn(launched(fixed_fn(8, 8192, 6, DYN)))
    require_old(launched(fixed_fn(8, 8192, 6, 0)))                                  # impl 0 on a fixed-length call: the old kernels only
    for name in ("V1", "V2"):
        e = CORPUS_V[name]
        require_dyn(launched(varlen_fn(e.lens, e.H, DYN)))
        require_old(launched(varlen_fn(e.lens, e.H, OLD)))                          # impl 32: the old kernels also where the default would route
    e = CORPUS_V["V7c"]
    require_dyn(launched(varlen_fn(e.lens, e.H, DYN)))                              # 64 sequences: many waves of CTAs
    # everything fits one wave at 8 columns: nothing to order or choose, the kernel decodes the old schedule itself, no builder
    e = CORPUS_V["V7a"]
    assert e.H * 16 <= sms
    require_dyn(launched(varlen_fn(e.lens, e.H, DYN)), builder=False)
    require_dyn(launched(varlen_fn(e.lens, e.H, DYN, k2_bv=16)))                    # a forced width always builds the schedule
    # the forced widths keep the device ordering
    for bv in (8, 16, 32):
        require_dyn(launched(varlen_fn(CORPUS_V["V4"].lens, 6, DYN, k2_bv=bv)))


# ---- the default routing -----------------------------------------------------------------------------------------------------------------------------
def _cc_major():
    return torch.cuda.get_device_capability()[0]


def test_default_route_matches_the_specification(ext):
    """``k2_dyn_default_route`` (csrc/cfbi/cfbi_launch.cu, ``ext.k2_route``) and ``route_default`` of tests/cfbi/k2_policy.py (the specification) agree on a grid of
    calls around every condition of the rule, on the lists that the rule was measured on, and on the named corpus entries."""
    sms, cc = _sms(), _cc_major()
    cases = []
    for H in (1, 2, 3, 6, 10, 12, 20, 32, 40):
        for N in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256, 512, 1024, 1025):
            for per in (1, 63, 64, 100, 255, 256, 257, 384, 512, 768, 1000, 1024, 1536, 2048, 4096, 16384):
                cases.append((N, H, N * per))
    for N in range(1, 12):                                                          # around the one-wave threshold N * H * 16 against the SM count, and the next wave
        H = max(1, sms // (16 * N))
        for dh in (0, 1, 2, 4):
            for per in (512, 4096):
                cases.append((N, H + dh, per * N))
    for N, H, T, _ in B200_ROUTING_MEASUREMENTS.values():                            # the measured lists, and the tokens next to them
        for dt in (-N, 0, N, 64 * N):
            cases.append((N, H, max(1, T + dt)))
    for e in CORPUS_V.values():
        cases.append((len(e.lens), e.H, sum(e.lens)))
    for N, H, T in cases:
        assert ext.k2_route(N, H, T) == kp.route_default(N, H, T, sms, cc), (N, H, T)
    for name, e in CORPUS_V.items():                                                 # the named entries: routed on Blackwell except the one-wave list and the two skewed sequences
        want = cc == 10 and name not in ("V7a", "V7b")
        assert ext.k2_route(len(e.lens), e.H, sum(e.lens)) == want, name


@pytest.mark.parametrize("name", ["V1", "V2", "V3", "V6", "V7a", "V7b", "V7c", "V7d"])
def test_default_impl_runs_the_scheduled_k2_exactly_where_the_rule_says(ext, name):
    e = CORPUS_V[name]
    routed = ext.k2_route(len(e.lens), e.H, sum(e.lens))
    cnt = launched(varlen_fn(e.lens, e.H, 0))
    (require_dyn if routed else require_old)(cnt)
    require_old(launched(varlen_fn(e.lens, e.H, OLD)))                               # bit 5 keeps the old kernels where the default would route
    require_dyn(launched(varlen_fn(e.lens, e.H, DYN)), builder=len(e.lens) * e.H * 16 > _sms())


def test_default_impl_is_never_routed_for_other_call_kinds(ext):
    e = CORPUS_V["V2"]
    if not ext.k2_route(len(e.lens), e.H, sum(e.lens)):
        pytest.skip("this device does not route V2 by default")
    require_dyn(launched(varlen_fn(e.lens, e.H, 0)))                                 # the routed shape ...
    require_old(launched(fixed_fn(8, 8192, 6, 0)))                                   # ... but a call without cu_seqlens is never routed
    for bv in (8, 16, 32, 1003):
        require_old(launched(varlen_fn(e.lens, e.H, 0, k2_bv=bv)))                   # a forced tile width (and the test hook) belongs to the old kernels
    for impl in (1, REF_CHUNK, REF, ROWS_ONLY, UMMA):                                # any other request than the default is taken as it is
        cnt = launched(varlen_fn(e.lens, e.H, impl))
        assert cnt["k2_dyn_kernel"] == 0 and cnt["k2_sched_kernel"] == 0, (impl, dict(cnt))
    case = make_case(sum(e.lens), e.H, "authors", seed=3)
    box = []
    cnt = launched(lambda: box.append(forward_train(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], case["A_log"], case["dt_bias"],
                                                    case["lower_bound"], _cu(e.lens), _impl=0)), warm=True)
    require_dyn(cnt)                                                                 # the training forward with cu_seqlens follows the same rule


def test_default_impl_keeps_the_old_kernels_when_the_supplied_plan_buffer_is_too_small(ext):
    """A caller that sized the plan buffer for the old kernels (the SeqPlan entries only) must not start to fail because the default now routes: the call
    runs the old kernels. An explicit impl 16 needs the room and says so."""
    e = CORPUS_V["V2"]
    if not ext.k2_route(len(e.lens), e.H, sum(e.lens)):
        pytest.skip("this device does not route V2 by default")
    N, H, T = len(e.lens), e.H, sum(e.lens)
    case = make_case(T, H, "authors", seed=9)
    ws_n, plan_n = ext.workspace_sizes(T, N, H)
    assert plan_n > (N + 1) * 8

    def call(impl, plan_ints):
        st = alloc_state(N, H)
        out = torch.zeros_like(case["q"])
        ws = torch.empty(ws_n, dtype=torch.float32, device="cuda")
        plan = torch.empty(plan_ints, dtype=torch.int32, device="cuda")
        ext.forward(case["q"], case["k"], case["v"], case["g"], case["beta"], float(case["scale"]), out, case["A_log"], case["dt_bias"], float(case["lower_bound"]),
                    st.E, st.rows, st.pos, _cu(e.lens), 0, None, impl, None, ws, plan, 3, 0)
        return out

    require_old(launched(lambda: call(0, (N + 1) * 8)))                              # too small for the schedule: the old kernels, no error
    require_dyn(launched(lambda: call(0, plan_n)))                                   # the size that workspace_sizes reports: routed
    with pytest.raises((RuntimeError, ValueError)):
        call(DYN, (N + 1) * 8)


@pytest.mark.parametrize("name", ["V1", "V2", "V4", "V7b", "V7c", "V7d"])
def test_routed_default_calls_are_bit_identical_to_the_reference(ext, name):
    e = CORPUS_V[name]
    case = make_case(sum(e.lens), e.H, "authors", seed=11)
    ref = run_chain(case, e.lens, [e.lens], e.H, REF)
    box = []
    cnt = launched(lambda: box.append(run_chain(case, e.lens, [e.lens], e.H, 0)), warm=False)
    (require_dyn if ext.k2_route(len(e.lens), e.H, sum(e.lens)) else require_old)(cnt)
    assert_same_results(ref, box[-1], name)
    rng = random.Random(5)                                                           # a warm chain (head rows, open chunks) through the default implementation
    takes = random_takes(e.lens, 3, rng)
    assert_same_results(run_chain(case, e.lens, takes, e.H, REF), run_chain(case, e.lens, takes, e.H, 0), (name, "warm"))


def test_bits_that_clear_the_new_bit_launch_no_new_kernel(ext):
    lens = [900, 300, 700, 1300, 5, 199, 1000, 450, 640, 77]
    for impl in (DYN | REF_CHUNK, DYN | ROWS_ONLY, DYN | UMMA):
        cnt = launched(varlen_fn(lens, 6, impl))
        assert cnt["k2_dyn_kernel"] == 0 and cnt["k2_sched_kernel"] == 0, (impl, dict(cnt))
    case = make_case(8, 3, "authors", seed=2)                                        # decode never runs K2
    st = alloc_state(8, 3)

    def dec():
        o = torch.zeros_like(case["q"][0])
        decode(case["q"][0], case["k"][0], case["v"][0], case["g"][0], case["beta"][0], case["scale"], o, case["A_log"], case["dt_bias"], case["lower_bound"], st, _impl=DYN)
    cnt = launched(dec)
    assert not any(k.startswith("k2") for k in cnt), dict(cnt)


@pytest.mark.parametrize("H", [1, 6, 33])
def test_capacity_boundary_launches_the_new_kernel_up_to_the_capacity_and_the_old_ones_beyond(ext, H):
    cap = ext.sched_capacity(H)
    assert cap >= 512
    for N in (cap, cap + 1):
        lens = [70] * N                                                               # short sequences: one full chunk and a tail
        cnt = launched(varlen_fn(lens, H, DYN))
        (require_dyn if N == cap else require_old)(cnt)
        case = make_case(sum(lens), H, "authors", seed=4)
        assert_same_results(run_chain(case, lens, [lens], H, REF), run_dyn(case, lens, [lens], H) if N == cap else run_chain(case, lens, [lens], H, DYN), (H, N))


# ---- bit identity ------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(CORPUS_V.keys()))
def test_corpus_v_is_bit_identical_for_every_width_setting(name):
    e = CORPUS_V[name]
    case = make_case(sum(e.lens), e.H, "authors", seed=7)
    ref = run_chain(case, e.lens, [e.lens], e.H, REF)
    for bv in (0, 8, 16, 32):
        assert_same_results(ref, run_dyn(case, e.lens, [e.lens], e.H, bv), (name, bv))


@pytest.mark.parametrize("name", ["V1", "V4", "V7b"])
def test_repeated_launches_give_the_same_bits(name):
    e = CORPUS_V[name]
    case = make_case(sum(e.lens), e.H, "heavy_tail", seed=8)
    ref = run_chain(case, e.lens, [e.lens], e.H, REF)
    for rep in range(5):
        assert_same_results(ref, run_dyn(case, e.lens, [e.lens], e.H), (name, rep))


@pytest.mark.parametrize("regime", ["authors", "heavy_tail", "saturated", "one_hot"])
@pytest.mark.parametrize("bv", [0, 8, 32])
def test_edge_lengths_cold(regime, bv):
    H = 3
    rng = random.Random(1)
    lens = [rng.choice([1, 15, 16, 17, 63, 64, 65, 128, 130, 200, 257, rng.randint(1, 300)]) for _ in range(9)] + [0, 64, 1]
    case = make_case(sum(lens), H, regime, seed=5)
    assert_same_results(run_chain(case, lens, [lens], H, REF), run_dyn(case, lens, [lens], H, bv), f"cold/{regime}/{bv}")


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("bv", [0, 16])
def test_warm_call_chains(seed, bv):
    H = 2
    rng = random.Random(10 + seed)
    lens = [rng.randint(60, 420) for _ in range(6)] + [64, 1, 130]
    case = make_case(sum(lens), H, "authors", seed=20 + seed)
    takes = random_takes(lens, 4, rng)
    assert_same_results(run_chain(case, lens, takes, H, REF), run_dyn(case, lens, takes, H, bv), f"warm/{seed}/{bv}")


# the grid sizes of the K2 stage, from below one wave of the SMs to many waves, and up to 512 sequences
SHAPES = [(2, 3), (9, 2), (7, 3), (13, 6), (24, 6), (40, 6), (100, 6), (257, 2), (512, 1)]


@pytest.mark.parametrize("N,H", SHAPES)
def test_waves_from_below_one_to_many_and_up_to_512_sequences(N, H):
    rng = random.Random(N * 100 + H)
    lens = [max(1, int(rng.lognormvariate(5.0, 1.0))) for _ in range(N)]
    lens = [min(n, 700) for n in lens]
    case = make_case(sum(lens), H, "authors", seed=N)
    ref = run_chain(case, lens, [lens], H, REF)
    for bv in (0, 8, 16, 32):
        assert_same_results(ref, run_dyn(case, lens, [lens], H, bv), (N, H, bv))


def test_pool_with_nonconsecutive_slots_and_padding_entries():
    H, S = 3, 24
    ids = [17, 3, -1, 8, 12, 0, -1, 5, 19, 2, 14]
    N = len(ids)
    sl = torch.tensor(ids, dtype=torch.int32, device="cuda")

    def go(impl):
        pool = alloc_state(S, H)
        outs = []
        for call in range(3):
            lens = [rng_len(call, n) for n in range(N)]
            case = make_case(sum(lens), H, "authors", seed=40 + call)
            o = torch.zeros_like(case["q"])
            o32 = torch.zeros(case["q"].shape, dtype=torch.float32, device="cuda")
            prefill(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], o, case["A_log"], case["dt_bias"], case["lower_bound"], pool, _cu(lens),
                    out_fp32=o32, slot_ids=sl, _impl=impl)
            outs.append((o, o32, pool.clone()))
        return outs

    def rng_len(call, n):
        return random.Random(1000 * call + n).choice([0, 1, 40, 64, 65, 130, 200, 333])

    ref = go(REF)
    box = []
    cnt = launched(lambda: box.append(go(DYN)), warm=False)
    assert cnt["k2_dyn_kernel"] >= 1 and not any(cnt[k] for k in OLD_K2), dict(cnt)
    assert_same_results(ref, box[-1], "pool")
    # the padding entries produced no output rows
    assert ref[0][0].shape == box[-1][0][0].shape


def _warm_pool(H, S, slots, fills, seed):
    """A pool of S slots in which the slots ``slots`` hold sequences whose open chunks are at the fills ``fills`` (one varlen call of the reference kernels)."""
    lens = [64 * (1 + i % 3) + f for i, f in enumerate(fills)]
    case = make_case(sum(lens), H, "authors", seed=seed)
    pool = alloc_state(S, H)
    prefill(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], torch.zeros_like(case["q"]), case["A_log"], case["dt_bias"], case["lower_bound"], pool,
            _cu(lens), slot_ids=torch.tensor(slots, dtype=torch.int32, device="cuda"), _impl=REF)
    return pool


def _fixed_call(pool0, slots, B, L, H, impl, seed):
    """A fixed-length call (no cu_seqlens) of B sequences of L tokens on a copy of ``pool0`` through ``slots`` (-1: padding entry)."""
    case = make_case(B * L, H, "authors", seed=seed)
    q, k, v, g, beta = (case[x].reshape(B, L, *case[x].shape[2:]) for x in ("q", "k", "v", "g", "beta"))
    out = torch.zeros_like(q)
    o32 = torch.zeros(q.shape, dtype=torch.float32, device="cuda")
    pool = pool0.clone()
    prefill(q, k, v, g, beta, case["scale"], out, case["A_log"], case["dt_bias"], case["lower_bound"], pool, None, out_fp32=o32,
            slot_ids=torch.tensor(slots, dtype=torch.int32, device="cuda"), _impl=impl)
    return out, o32, pool


# name: (entries B, heads H, skipped entries, open-chunk fill of every entry); with L = 424 tokens (L % 64 == 40) the fills 5, 17, 1, 12, 20 leave 6 chunks and
# 40, 0, 33, 63, 50 leave 7: the unequal lists a fixed-length call has on a warm pool
FIXED_WARM = {
    "unequal-mixed": (8, 6, (), (5, 40, 17, 0, 33, 1, 63, 12)),
    "unequal-skipped-mixed": (8, 6, (2, 5), (5, 40, 17, 0, 33, 1, 63, 12)),
    "equal-skipped-mixed": (8, 6, (1, 6), (17,) * 8),
    "unequal-skipped-all32": (10, 6, (0, 7, 9), (5, 40, 17, 0, 33, 1, 63, 12, 20, 50)),
}


@pytest.mark.parametrize("name", list(FIXED_WARM))
def test_explicit_impl_16_on_a_warm_fixed_length_call_is_bit_identical(ext, name):
    """A fixed-length call on a warm pool (open chunks at unequal fills, skipped slots, slots neither contiguous nor in order) with the device-scheduled K2: the
    builder lays out the old schedule of the fixed-length launcher over all N * H pair positions (unequal chunk counts) or the closed form of the active pairs
    (equal counts); every bit equals the frozen reference kernels and the old K2 kernels."""
    B, H, skipped, fills = FIXED_WARM[name]
    L, S = 424, 32
    equal = name.startswith("equal")
    warm_slots = random.Random(7).sample(range(S), B)
    pool0 = _warm_pool(H, S, warm_slots, fills, seed=61)
    assert pool0.pos[warm_slots].tolist() == list(fills)                                 # the warm-up left the intended open chunks
    slots = [-1 if i in skipped else s for i, s in enumerate(warm_slots)]
    counts = [0 if s < 0 else kp.chunk_count(L, f) for s, f in zip(slots, fills)]
    assert len({c for c in counts if c}) == (1 if equal else 2) and all(counts) == (not skipped), counts   # the intended list
    assert B * H * 16 > _sms()                                                           # the builder runs (more than one wave at 8 columns)
    x32, _ = kp.old_rule(B, H, _sms(), True)
    if _sms() in (132, 144, 148):
        assert (0 < x32 < B * H) == ("mixed" in name) and (x32 == B * H) == name.endswith("all32"), (name, x32)
    # the schedule this call gets (the same inputs through the test probe): the reference's, and for unequal chunk counts the old layout with the 32-column
    # positions counted over all N * H positions
    d = probe(ext, None, H, N=B, fixed_len=L, pos=pool0.pos, slot_ids=torch.tensor(slots, dtype=torch.int32, device="cuda"))
    sched = kp.build_schedule(counts, H, _sms(), call_fixed=True)
    assert_matches(d, counts, sched, H, name)
    if not equal:
        assert d.kind == "old"
        assert_old_layout_header(d, counts, B, H, _sms())
    ref = _fixed_call(pool0, slots, B, L, H, REF, 62)
    box = []
    cnt = launched(lambda: box.append(_fixed_call(pool0, slots, B, L, H, DYN, 62)))
    require_dyn(cnt)                                                                     # the builder and the device-scheduled K2 ran, no old K2 kernel
    assert_same_results([ref], [box[-1]], name)
    require_old(launched(lambda: box.append(_fixed_call(pool0, slots, B, L, H, OLD, 62))))
    assert_same_results([ref], [box[-1]], (name, "old kernels"))


def test_training_forward_with_cu_seqlens():
    H = 3
    lens = [150, 1, 64, 129, 33, 0, 70, 400, 640, 77, 900]
    case = make_case(sum(lens), H, "heavy_tail", seed=13)
    cu = _cu(lens)

    def go(impl):
        o32 = torch.zeros(case["q"].shape, dtype=torch.float32, device="cuda")
        o = forward_train(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], case["A_log"], case["dt_bias"], case["lower_bound"], cu, out_fp32=o32, _impl=impl)
        return o, o32

    ref = go(REF)
    box = []
    cnt = launched(lambda: box.append(go(DYN)), warm=False)
    assert cnt["k2_dyn_kernel"] >= 1, dict(cnt)
    assert n_diff(ref[0], box[-1][0]) == 0 and n_diff(ref[1], box[-1][1]) == 0


# ---- split K1 / K2 calls and context-parallel shards -----------------------------------------------------------------------------------------------
def test_split_k1_k2_varlen_call_equals_one_call_of_the_reference():
    H = 2
    lens = [64 * 7 + 5, 64 * 5, 64 * 9 + 63, 64 * 20, 64 * 3, 64 * 14 + 9, 64 * 11, 64 * 6 + 1]
    case = make_case(sum(lens), H, "authors", seed=8)
    cu = _cu(lens)
    ref_state = alloc_state(len(lens), H)
    ref_out = torch.zeros_like(case["q"])
    ref_o32 = torch.zeros(case["q"].shape, dtype=torch.float32, device="cuda")
    prefill(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], ref_out, case["A_log"], case["dt_bias"], case["lower_bound"], ref_state, cu,
            out_fp32=ref_o32, _impl=REF)
    st = alloc_state(len(lens), H)
    h1 = prefill_k1(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], case["A_log"], case["dt_bias"], case["lower_bound"], st, cu, _impl=DYN)
    assert h1.plan.numel() == (len(lens) + 1) * 8 + 16 + 16 * len(lens) * H, "the plan buffer of a split call carries the schedule"
    out = torch.zeros_like(case["q"])
    o32 = torch.zeros(case["q"].shape, dtype=torch.float32, device="cuda")
    cnt = launched(lambda: prefill_k2(h1, out, st, out_fp32=o32), warm=False)
    require_dyn(cnt)                                                                 # the K2 phase builds its schedule from the plan of the K1 phase
    assert n_diff(out, ref_out) == 0 and n_diff(o32, ref_o32) == 0 and state_equal(st, ref_state), state_mismatch(st, ref_state)


@pytest.mark.parametrize("cuts", [[192, 448], [64], [640, 704, 768]])
def test_context_parallel_shards_with_the_new_k2(cuts):
    H, T = 3, 64 * 12 + 21
    case = make_case(T, H, "authors", seed=7)

    def sl(lo, hi):
        return {k: case[k][:, lo:hi].contiguous() for k in ("q", "k", "v", "g", "beta")}

    ref_state = alloc_state(1, H)
    ref_out = torch.zeros_like(case["q"])
    prefill(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], ref_out, case["A_log"], case["dt_bias"], case["lower_bound"], ref_state, _impl=REF)
    bounds = [0] + cuts + [T]
    shards = [sl(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    states = [alloc_state(1, H) for _ in shards]
    h1 = [prefill_k1(s["q"], s["k"], s["v"], s["g"], s["beta"], case["scale"], case["A_log"], case["dt_bias"], case["lower_bound"], st, _impl=DYN)
          for s, st in zip(shards, states)]
    outs = []
    for i, (s, st) in enumerate(zip(shards, states)):
        if i > 0:
            st.E.copy_(states[i - 1].E)
        o = torch.zeros_like(s["q"])
        prefill_k2(h1[i], o, st)
        outs.append(o)
    assert n_diff(torch.cat(outs, dim=1), ref_out) == 0
    assert state_equal(states[-1], ref_state), state_mismatch(states[-1], ref_state)


# ---- CUDA graphs ----------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("impl,mult", [(DYN, 1), (0, 8)])
def test_cuda_graph_replay_recomputes_the_schedule_from_rewritten_cu_seqlens(ext, impl, mult):
    """The launch configuration depends on N, H and the total token count, never on the lengths: a captured call replays bit-identically, also after the
    contents of cu_seqlens were rewritten in place (same N, H, total tokens), because the builder runs inside the graph on the device. ``impl`` 16 with short
    sequences, and the default implementation with sequences eight times longer (a call that the default routing selects on a Blackwell device)."""
    H = 3
    lens_a = [mult * n for n in (70, 5, 130, 300, 64, 1, 77, 200, 640, 900, 33, 12)]
    lens_b = [mult * n for n in (900, 640, 33, 12, 70, 5, 130, 300, 64, 1, 77, 200)]
    assert sum(lens_a) == sum(lens_b) and len(lens_a) == len(lens_b)
    lens_c = [mult] * 11 + [sum(lens_a) - 11 * mult]                                  # an extreme rewrite: one long sequence
    N, T = len(lens_a), sum(lens_a)
    if impl == 0 and _cc_major() == 10:
        assert ext.k2_route(N, H, T), "the default implementation must route this call"
    case = make_case(T, H, "authors", seed=51)
    cu_t = _cu(lens_a)
    init = alloc_state(N, H)
    warm = make_case(N * 3, H, "authors", seed=52)
    prefill(warm["q"], warm["k"], warm["v"], warm["g"], warm["beta"], warm["scale"], torch.zeros_like(warm["q"]), warm["A_log"], warm["dt_bias"], warm["lower_bound"], init,
            _cu([3] * N), _impl=0)

    def eager(lens, impl):
        st = init.clone()
        o = torch.zeros_like(case["q"])
        o32 = torch.zeros(case["q"].shape, dtype=torch.float32, device="cuda")
        prefill(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], o, case["A_log"], case["dt_bias"], case["lower_bound"], st, _cu(lens), out_fp32=o32, _impl=impl)
        return o, o32, st

    g_state = init.clone()
    out = torch.zeros_like(case["q"])
    out32 = torch.zeros(case["q"].shape, dtype=torch.float32, device="cuda")

    def launch():
        prefill(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], out, case["A_log"], case["dt_bias"], case["lower_bound"], g_state, cu_t,
                out_fp32=out32, _impl=impl)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):                                                      # warm-up (allocations, attribute set-up) off the capture
        keep, g_state = g_state, init.clone()
        launch()
        g_state = keep
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    def replay(lens):
        cu_t.copy_(_cu(lens))                                                         # rewrite the contents in place
        g_state.E.copy_(init.E), g_state.rows.copy_(init.rows), g_state.pos.copy_(init.pos)
        out.zero_(), out32.zero_()
        graph.replay()
        torch.cuda.synchronize()
        ref = eager(lens, REF)
        assert_same_results([(ref[0], ref[1], ref[2])], [(out, out32, g_state)], "graph replay %s" % lens[:3])

    replay(lens_a)
    replay(lens_a)                                                                     # replayed twice
    replay(lens_b)                                                                     # after the contents of cu_seqlens were rewritten
    replay(lens_c)
