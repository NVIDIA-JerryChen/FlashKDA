"""Single-launch decode (``rows_kernel<true>``, ``rows_entry`` in csrc/cfbi/cfbi_rows.cuh): decode on the optimised row engine is ONE kernel launch, bit-identical to
the two-launch decode (plan kernel + row engine) and to the frozen reference.

What changed is only where a sequence's token window, slot and open-chunk fill come from and who advances ``pos``: no plan kernel runs; every head CTA derives its entry
itself (``seq_window``, the plan kernel's own function) and the H head CTAs of a sequence count their arrivals in the high bits of the ``pos`` word, the last one to arrive
stores the plain new fill. No floating-point operation moved, so every scenario compares bits with zero tolerance:

* the launch sets (a profiler check: one ``rows_kernel<true>`` and nothing else where the policy says so or the hidden calibration control is on; the plan kernel plus
  ``rows_kernel<false>`` for ``_impl=64``; the frozen reference keeps its plan kernel; mode 0 is untouched) and the policy on the grid size and the device
  (``decode_single_launch``, specified in ``decode_policy.py``),
* decode chains from every chunk phase on grids of MORE than two waves of CTAs (the late starters that the arrival count exists for), with a state pool, padding entries
  (``slot_ids == -1``) and chunk ends inside the chain, against the reference kernels and against the two-launch optimised decode,
* mode 1 with several tokens per sequence (fixed length and ``cu_seqlens``, empty sequences, padding entries), also on a big grid and across chunk ends,
* a CUDA graph replayed with rewritten ``slot_ids``, on the single launch and on the two-launch path.

Every equality test names the path of each of its sides (``rows_workload.Path``): the single launch is ``_impl=0`` with the hidden calibration control on (the ``forced`` fixture), the two launches
are ``_impl=64``, the oracle is ``_impl=3``, and every call is run under the profiler and must launch exactly the kernels of its own path (``rows_workload.launched_exactly``): a test that runs one
path on both sides, because of a routing default, a device or the ``FLASH_KDA_CFBI_IMPL`` of Suite Sd, fails its own launch assertion (controls below), and every ``decode`` / ``prefill`` call of this
module passes ``_impl`` explicitly (``test_decode_routing_env.py`` checks that from the source). The grids are sized from the occupancy of the instantiations that the compared paths launch.
The late-start race itself is stressed in ``test_decode_single_launch_variants.py`` (delayed CTAs, wrong protocols that must be caught); the logic of the protocol is checked
exhaustively in ``test_decode_pos_protocol_model.py``.
"""
import random

import pytest
import torch

from flash_kda.cfbi import alloc_state, decode, prefill, state_equal, state_mismatch

import decode_policy as dp
import helpers
import live_grid
import rows_workload as rw
from helpers import make_case, n_diff, recording, skip_or_fail
from test_vs_ref import _call_inputs, _cu

REF_ROWS, REF, ROWS_ONLY, DECODE_PLAN = 1, 3, 4, 64


def _sms():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def _cc():
    return dp.cc_number(torch.cuda.get_device_capability())


def spec_single_launch(mode, impl, N, H, sms):
    """The policy of the device that runs the test (``decode_policy.py``: the one specification of ``decode_single_launch_policy``), ``impl`` as resolve_impl returns it."""
    return dp.policy(mode, impl, N, H, sms, _cc())


def active_n(ext, H, paths=(rw.SINGLE, rw.TWO_LAUNCH)):
    """The number of ACTIVE sequences (live in every launch, H CTAs each) that make a decode launch at least ``WAVES`` waves of the CTAs that are resident at once, for every row-engine
    instantiation that the compared ``paths`` launch (the occupancy API of the extension answers, no constant). Padding entries, empty sequences and entries without a token in a launch come ON
    TOP of this number (``live_grid`` planners), and ``assert_live_floor`` asserts what the planned launches really put on the device."""
    return max(live_grid.active_quota(helpers.resident_ctas(ext, p.instance), H) for p in paths)


def assert_live_floor(ext, launches, H, paths=(rw.SINGLE, rw.TWO_LAUNCH), start_pos=None):
    """Before any GPU work: every planned launch (``[(lens, slots)]``: a padding entry, an empty sequence and an entry without a token are not live) puts at least ``ceil(2.5 x resident CTAs)``
    live CTAs on the device, for every instantiation that the compared paths launch. Returns the coverage of the plan (``rows_workload.launch_coverage``)."""
    cov = rw.launch_coverage(launches, H, start_pos)
    caps = {p.instance: helpers.resident_ctas(ext, p.instance) for p in paths}
    problems = rw.beyond_waves_problems(cov["live_ctas"], caps)
    assert problems == [], problems
    return cov


def require_single_launch(ext, N, H):
    """The call that follows is ONE launch: by the policy of the device or because the control is on (``forced``). A device that cannot run it this way is a skip, except on B200 and
    B300, where it is a failure."""
    impl = ext.resolve_impl(0, -1)
    if not ext.decode_single_launch(1, impl, N, H):
        skip_or_fail("this device does not run decode as one launch here (impl 0 resolves to %d, grid %d x %d)" % (impl, N, H))


# ---- observing the launches --------------------------------------------------------------------------------------------------------------------------
def kernel_counts(fn, warm=True):
    """The CF-BI kernels one call of ``fn`` launches: name (argument list removed, template arguments kept: ``rows_kernel<true>``) -> launches. ``fn`` runs once before
    (JIT build, attribute set-up, the hardware check of the first call) unless ``warm`` is false."""
    if warm:
        fn()
    with recording() as rec:
        fn()
    return rec.counts


def decode_fn(N, H, impl, seed=1):
    """A call that decodes one token per sequence from a zero state of N slots."""
    case = make_case(N, H, "authors", seed=seed)
    st = alloc_state(N, H)
    q, k, v, g, beta = (case[x][0] for x in ("q", "k", "v", "g", "beta"))
    out = torch.zeros_like(q)

    def fn():
        decode(q, k, v, g, beta, case["scale"], out, case["A_log"], case["dt_bias"], case["lower_bound"], st, _impl=impl)
    return fn


# ---- which kernels launch ----------------------------------------------------------------------------------------------------------------------------
def expected_decode_launches(ext, impl, N, H):
    """The launch set of a decode call of N sequences and H heads that the specification predicts for this device (the control off): one ``rows_kernel<true>`` and nothing else, or
    the plan kernel and the row engine behind it (the frozen reference where ``impl`` resolves to it)."""
    eff = ext.resolve_impl(impl, -1)
    if spec_single_launch(1, eff, N, H, _sms()):
        return {"rows_kernel<true>": 1}
    return {"plan_kernel": 1, ("ref::rows_kernel" if eff & REF_ROWS else "rows_kernel<false>"): 1}


@pytest.mark.parametrize("N, H", [(1, 12), (4, 6), (16, 3), (40, 12)])
def test_the_default_launch_set_is_what_the_specification_predicts_for_this_device(ext, N, H):
    assert dict(kernel_counts(decode_fn(N, H, 0))) == expected_decode_launches(ext, 0, N, H)    # no plan kernel, no finalize kernel where the policy says one launch


@pytest.mark.parametrize("N, H", [(1, 12), (4, 6), (16, 3), (40, 12)])
def test_the_control_makes_decode_one_launch(ext, forced, N, H):
    require_single_launch(ext, N, H)
    assert dict(kernel_counts(decode_fn(N, H, 0))) == {"rows_kernel<true>": 1}


def test_the_control_forces_a_grid_beyond_the_limit_and_nothing_else(ext, forced):
    """Above the limit the policy keeps the plan kernel; the control takes the limit and the routing set away, and nothing else: the request for two launches (bit 64) and the reference
    row engine keep theirs."""
    sms = _sms()
    N, H = dp.MAX_CTAS_PER_SM * sms // 12 + 1, 12
    require_single_launch(ext, N, H)
    assert dict(kernel_counts(decode_fn(N, H, 0))) == {"rows_kernel<true>": 1}
    for impl in (DECODE_PLAN, REF_ROWS, REF):
        assert dict(kernel_counts(decode_fn(4, 6, impl))) == expected_decode_launches(ext, impl, 4, 6) != {"rows_kernel<true>": 1}


def test_the_other_requests_keep_the_plan_kernel(ext):
    N, H = 4, 6
    for impl in (DECODE_PLAN, REF_ROWS, REF, REF | DECODE_PLAN, ROWS_ONLY | DECODE_PLAN):
        eff = ext.resolve_impl(impl, -1)
        rows = "ref::rows_kernel" if eff & REF_ROWS else "rows_kernel<false>"
        assert dict(kernel_counts(decode_fn(N, H, impl))) == {"plan_kernel": 1, rows: 1}, (impl, eff)


def test_the_gate_keeps_the_plan_kernel_for_grids_beyond_it(ext):
    H, sms = 12, _sms()
    N = (dp.MAX_CTAS_PER_SM * sms) // H + 1               # one sequence more than the gate takes
    assert not spec_single_launch(1, ext.resolve_impl(0, -1), N, H, sms)
    assert dict(kernel_counts(decode_fn(N, H, 0))) == expected_decode_launches(ext, 0, N, H) == {"plan_kernel": 1, "rows_kernel<false>": 1}
    N = (dp.MAX_CTAS_PER_SM * sms) // H                   # the largest grid the gate takes (N * H <= 8 * SMs), on a device of the routing set
    assert spec_single_launch(1, ext.resolve_impl(0, -1), N, H, sms) == (_cc() // 10 == dp.ROUTING_CC_MAJOR)
    assert dict(kernel_counts(decode_fn(N, H, 0))) == expected_decode_launches(ext, 0, N, H)


@pytest.mark.parametrize("impl", [0, DECODE_PLAN], ids=["default", "bit64"])
def test_prefill_keeps_its_launches(ext, forced, impl):
    """Mode 0 is untouched, with or without bit 64 and with the calibration control on: plan kernel first, then the row engine behind it (rows_kernel<false>), K1 and K2 as before; never the
    single-launch kernel."""
    lens = [150, 1, 64, 129, 33, 0, 70]
    H = 3
    case = make_case(sum(lens), H, "authors", seed=3)
    cu = torch.tensor(_cu(lens), dtype=torch.int64, device="cuda")
    st = alloc_state(len(lens), H)
    out = torch.zeros_like(case["q"])

    def fn():
        prefill(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], out, case["A_log"], case["dt_bias"], case["lower_bound"], st, cu, _impl=impl)
    cnt = kernel_counts(fn)
    assert cnt["plan_kernel"] == 1 and cnt["rows_kernel<true>"] == 0, dict(cnt)
    assert cnt["rows_kernel<false>"] + cnt["ref::rows_kernel"] == 1, dict(cnt)


def test_the_gate_matches_the_specification(ext):
    sms = _sms()
    shapes = [(N, H) for N in (1, 2, 7, 64, 65, 255, 1024) for H in (1, 3, 12, 33, 64)]
    for H in (1, 6, 12, 32):                                   # the boundary N * H == 8 * SMs and its neighbours
        for dn in (-1, 0, 1):
            shapes.append((max(1, (dp.MAX_CTAS_PER_SM * sms) // H + dn), H))
    shapes.append((dp.MAX_CTAS_PER_SM * sms, 1))
    shapes.append((dp.MAX_CTAS_PER_SM * sms + 1, 1))
    for mode in (0, 1):
        for impl in (0, 1, 2, 3, 4, 5, 8, 16, 32, 64, 65, 66, 67, 68, 127):
            eff = ext.resolve_impl(impl, -1)
            for N, H in shapes:
                assert ext.decode_single_launch(mode, eff, N, H) == spec_single_launch(mode, eff, N, H, sms), (mode, impl, eff, N, H)
    assert not ext.decode_single_launch(1, 0, 0, 12) and not ext.decode_single_launch(1, 0, 4, 0), "no sequences or no heads: nothing to launch"


def test_the_decode_plan_bit_is_wired(ext):
    assert ext.resolve_impl(DECODE_PLAN, -1) & DECODE_PLAN, "the bit stays (it is a request of its own)"
    for v in range(128):
        r = ext.resolve_impl(v, -1)
        assert ext.resolve_impl(r, -1) == r, v
    with pytest.raises((RuntimeError, ValueError)):
        ext.resolve_impl(128, -1)
    fn = decode_fn(2, 2, 127)                                    # every bit: only bits 4 and 5 exclude each other, and that is rejected by the forward call
    with pytest.raises((RuntimeError, ValueError)):
        fn()
    decode_fn(2, 2, 64 | 15)()                                   # the new bit with the others is accepted


# ---- bit identity ------------------------------------------------------------------------------------------------------------------------------------
def step_inputs(case, cu, first, done):
    """Token ``first[n] + done[n]`` of every sequence n as [N, H, 128] decode inputs (the sequences start at ``cu[n]`` of the flattened case)."""
    idx = torch.tensor([cu[n] + first[n] + done[n] for n in range(len(first))], dtype=torch.int64, device="cuda")
    return {k: case[k][0].index_select(0, idx).contiguous() for k in ("q", "k", "v", "g", "beta")}


def decode_with(case, ins, st, ids, impl):
    """One decode call on ``st`` (updated in place); the outputs are pre-filled with a sentinel, so rows a call does not write are visible."""
    out = torch.full_like(ins["q"], 3.0)
    o32 = torch.full(ins["q"].shape, 3.0, dtype=torch.float32, device="cuda")
    decode(ins["q"], ins["k"], ins["v"], ins["g"], ins["beta"], case["scale"], out, case["A_log"], case["dt_bias"], case["lower_bound"], st, out_fp32=o32, slot_ids=ids, _impl=impl)
    return out, o32


def decode_on(path, case, ins, st, ids):
    """One decode call on ``st`` on ``path``: run under the profiler, it must launch exactly the kernels of that path."""
    return rw.launched_exactly(path, lambda: decode_with(case, ins, st, ids, path.impl))


def prefilled(H, N, S, first, steps, ids, seed):
    """A case of ``first[n] + steps`` tokens per sequence, and a state pool of S slots in which sequence n has run its first ``first[n]`` tokens (through the reference)."""
    lens = [f + steps for f in first]
    case = make_case(sum(lens), H, "authors", seed=seed)
    cu = _cu(lens)
    sub, cu_call = _call_inputs(case, cu, [0] * N, first)
    pool = alloc_state(S, H)
    prefill(sub["q"], sub["k"], sub["v"], sub["g"], sub["beta"], case["scale"], torch.zeros_like(sub["q"]), case["A_log"], case["dt_bias"], case["lower_bound"], pool, cu_call,
            slot_ids=ids, _impl=REF)
    return case, cu, pool


def plain_pos(st):
    """No arrival counter is left in the pos words: every entry is a fill 0 .. 63."""
    return bool(((st.pos >= 0) & (st.pos < 64)).all())


OTHERS = [rw.ORACLE, rw.TWO_LAUNCH]                                  # the sides that the single launch is compared with: the frozen reference and the two-launch decode
OTHER_IDS = ["reference", "two_launch"]


@pytest.mark.parametrize("other", OTHERS, ids=OTHER_IDS)
@pytest.mark.parametrize("pool", [False, True], ids=["compact", "pool_and_padding"])
def test_decode_chain_beyond_two_waves_is_bit_identical(ext, forced, other, pool):
    """Prefill every sequence to a different chunk phase (61, 62, 63, 0, 1, 2 mod 64: the chain crosses chunk ends), then decode on the single launch and on ``other`` from copies of the
    same state: identical bf16 and fp32 outputs, E, valid row-cache planes and pos after every step, on a grid of at least 2.5 waves of every instantiation the two sides launch. Every call
    of both sides asserts its own launch set."""
    H, steps = 12, 5
    N, skip = live_grid.padded_batch(active_n(ext, H), 7, 3) if pool else (active_n(ext, H), [])      # the padding entries of the decode batch (no tokens, no state access) come on top of the quota
    require_single_launch(ext, N, H)
    rng = random.Random(11)
    S = N + 9 if pool else N
    slots = rng.sample(range(S), N) if pool else None
    assert_live_floor(ext, [([1] * N, [-1 if i in skip else i for i in range(N)])] * steps, H)
    first = [125 + (n % 6) for n in range(N)]
    ids = torch.tensor(slots, dtype=torch.int32, device="cuda") if pool else None
    case, cu, st = prefilled(H, N, S, first, steps, ids, seed=21)
    st_a, st_b = st.clone(), st.clone()
    ids_call = None
    if pool:
        ids_call = ids.clone()
        ids_call[skip] = -1
    for t in range(steps):
        ins = step_inputs(case, cu, first, [t] * N)
        oa, fa = decode_on(rw.SINGLE, case, ins, st_a, ids_call)
        ob, fb = decode_on(other, case, ins, st_b, ids_call)
        assert n_diff(oa, ob) == 0 and n_diff(fa, fb) == 0, f"step {t}: outputs differ"
        assert state_equal(st_a, st_b), f"step {t}: state differs {state_mismatch(st_a, st_b)}"
        assert plain_pos(st_a), f"step {t}: an arrival counter is left in pos"
        if skip:
            assert bool((oa[skip] == 3.0).all()) and bool((fa[skip] == 3.0).all()), f"step {t}: a padding entry wrote its outputs"
    # the chain really crossed chunk ends: the fills after `steps` tokens are not the fills before
    assert not torch.equal(st_a.pos, st.pos)


@pytest.mark.parametrize("other", OTHERS, ids=OTHER_IDS)
def test_decode_from_every_phase_on_a_big_grid_is_bit_identical(ext, forced, other):
    """The scenario of ``test_vs_ref.test_decode_from_every_phase`` (every chunk phase, 70 single-token steps across a chunk end) on a grid of at least 2.5 resident waves, single launch
    against ``other``: sequence n starts at phase n mod 64, so all 64 phases are among the sequences; what the chain executes is derived from its plan (``launch_coverage``) and must
    reach every position and a chunk end of every sequence; after every step outputs, E, valid rows and pos are identical and no arrival bit is left, and the final positions are exactly
    the starts advanced by 70."""
    H, steps = 12, 70
    N = max(active_n(ext, H), rw.BT)
    require_single_launch(ext, N, H)
    first = [2 * rw.BT + n % rw.BT for n in range(N)]            # two whole chunks, then phase n mod 64
    cov = assert_live_floor(ext, [([1] * N, list(range(N)))] * steps, H, start_pos={n: first[n] % rw.BT for n in range(N)})      # every sequence is live in every step
    assert cov["positions"] == list(range(rw.BT)), cov["positions"]                    # a single-token chain has no row after a chunk end inside one launch (the multi-token tests have)
    assert cov["chunk_end_rows"] >= N and cov["cross_subblock_rows"] > 0, cov           # every sequence finishes a chunk within the 70 steps
    case, cu, st = prefilled(H, N, N, first, steps, None, seed=41)
    st_a, st_b = st.clone(), st.clone()
    for t in range(steps):
        ins = step_inputs(case, cu, first, [t] * N)
        oa, fa = decode_on(rw.SINGLE, case, ins, st_a, None)
        ob, fb = decode_on(other, case, ins, st_b, None)
        assert n_diff(oa, ob) == 0 and n_diff(fa, fb) == 0, f"step {t}: outputs differ"
        assert state_equal(st_a, st_b), f"step {t}: state differs {state_mismatch(st_a, st_b)}"
        assert plain_pos(st_a), f"step {t}: an arrival counter is left in pos"
    assert st_a.pos.cpu().tolist() == [(f + steps) % rw.BT for f in first], "the final positions are not the start phases advanced by the steps"


@pytest.mark.parametrize("other", OTHERS, ids=OTHER_IDS)
def test_mode1_calls_with_several_tokens_per_sequence_across_chunks_on_a_big_grid(ext, forced, other):
    """The scenario of ``test_vs_ref.test_mode1_row_engine_everything`` on a big grid: ``cu_seqlens`` calls of mode 1 with lengths up to 130 (empty sequences among them), every sequence run
    in three successive calls so that its open chunk carries from call to call and its rows cross chunk ends, single launch against ``other`` from the same state, a grid of at least 2.5
    resident waves of every instantiation the two sides launch; the executed rows (derived from the plan of the calls) reach every position, a chunk end and rows after it."""
    H = 3
    lens, takes = live_grid.mode1_plan(active_n(ext, H))      # the active sequences take tokens in every call; the empty and the one- and two-token entries come on top of them
    N = len(lens)
    require_single_launch(ext, N, H)
    launches = live_grid.launches(takes)
    assert_live_floor(ext, launches, H)                       # every one of the three calls is beyond 2.5 live waves, derived from the planned takes before any GPU work
    case = make_case(sum(lens), H, "authors", seed=51)
    cu = _cu(lens)
    st_a, st_b = alloc_state(N, H), alloc_state(N, H)
    done = [0] * N
    for call in range(3):
        take = takes[call]
        sub, cu_call = _call_inputs(case, cu, done, take)
        oa, fa = mode1_on(rw.SINGLE, case, sub, cu_call, st_a, None)
        ob, fb = mode1_on(other, case, sub, cu_call, st_b, None)
        assert n_diff(oa, ob) == 0 and n_diff(fa, fb) == 0, f"call {call}: outputs differ"
        assert state_equal(st_a, st_b), f"call {call}: state differs {state_mismatch(st_a, st_b)}"
        assert plain_pos(st_a), f"call {call}: an arrival counter is left in pos"
        done = [d + k for d, k in zip(done, take)]
    assert done == lens
    assert st_a.pos.cpu().tolist() == [L % rw.BT for L in lens]
    cov = rw.launch_coverage(launches, H)
    assert rw.coverage_problems(cov) == [], cov


@pytest.mark.parametrize("N, H", [(1, 1), (1, 12), (3, 7), (24, 12), (75, 12), (5, 33)])
def test_one_token_from_a_zero_state_matches_the_reference_on_every_grid_shape(ext, forced, N, H):
    require_single_launch(ext, N, H)
    ref = None
    for path in (rw.ORACLE, rw.TWO_LAUNCH, rw.SINGLE):
        case = make_case(N, H, "heavy_tail", seed=5)
        st = alloc_state(N, H)
        ins = {k: case[k][0].contiguous() for k in ("q", "k", "v", "g", "beta")}
        out, o32 = decode_on(path, case, ins, st, None)
        if ref is None:
            ref = (out, o32, st)
        else:
            assert n_diff(out, ref[0]) == 0 and n_diff(o32, ref[1]) == 0, f"{path.name}: outputs differ"
            assert state_equal(st, ref[2]), f"{path.name}: state differs {state_mismatch(st, ref[2])}"
            assert plain_pos(st)


# ---- mode 1 with several tokens per sequence ---------------------------------------------------------------------------------------------------------
def _mode1_call(case, sub, cu_call, st, ids, impl):
    out = torch.full_like(sub["q"], 3.0)
    o32 = torch.full(sub["q"].shape, 3.0, dtype=torch.float32, device="cuda")
    prefill(sub["q"], sub["k"], sub["v"], sub["g"], sub["beta"], case["scale"], out, case["A_log"], case["dt_bias"], case["lower_bound"], st, cu_call, out_fp32=o32,
            slot_ids=ids, _mode=1, _impl=impl)
    return out, o32


def mode1_on(path, case, sub, cu_call, st, ids):
    """One mode-1 call on ``path``: it must launch exactly the kernels of that path."""
    return rw.launched_exactly(path, lambda: _mode1_call(case, sub, cu_call, st, ids, path.impl))


# the three requests of the fixed-length call: the default row engine, ``IMPL_ROWS_ONLY`` (the single launch too), and ``IMPL_ROWS_ONLY`` with bit 64 (two launches)
FIXED_PATHS = [rw.SINGLE, rw.Path("single_rows_only", ROWS_ONLY, helpers.ROWS_DECODE, rw.SINGLE.launches), rw.Path("two_launch_rows_only", DECODE_PLAN | ROWS_ONLY, helpers.ROWS_OPTIMISED, rw.TWO_LAUNCH.launches)]


@pytest.mark.parametrize("path", FIXED_PATHS, ids=["mode1", "rows_only", "rows_only_plan"])
def test_fixed_length_mode1_call_matches_the_reference(ext, forced, path):
    """No ``cu_seqlens``: every batch entry is one sequence of T tokens (``fixed_len`` = T), run through the row engine from a warm state; 60 + 7 tokens cross a chunk end. Each side asserts its
    own launch set."""
    N, H, T0, T1 = 40, 3, 60, 7
    if path.launches == rw.SINGLE.launches:
        require_single_launch(ext, N, H)
    case = make_case(N * (T0 + T1), H, "authors", seed=8)
    q, k, v, g, beta = (case[x].reshape(N, T0 + T1, *case[x].shape[2:]) for x in ("q", "k", "v", "g", "beta"))
    warm = {"q": q[:, :T0], "k": k[:, :T0], "v": v[:, :T0], "g": g[:, :T0], "beta": beta[:, :T0]}
    call = {"q": q[:, T0:], "k": k[:, T0:], "v": v[:, T0:], "g": g[:, T0:], "beta": beta[:, T0:]}
    warm = {x: t.contiguous() for x, t in warm.items()}
    call = {x: t.contiguous() for x, t in call.items()}
    st = alloc_state(N, H)
    prefill(warm["q"], warm["k"], warm["v"], warm["g"], warm["beta"], case["scale"], torch.zeros_like(warm["q"]), case["A_log"], case["dt_bias"], case["lower_bound"], st,
            _impl=REF)
    st_a, st_b = st.clone(), st.clone()
    oa, fa = mode1_on(path, case, call, None, st_a, None)
    ob, fb = mode1_on(rw.ORACLE, case, call, None, st_b, None)
    assert n_diff(oa, ob) == 0 and n_diff(fa, fb) == 0 and state_equal(st_a, st_b), state_mismatch(st_a, st_b)
    assert plain_pos(st_a)


@pytest.mark.parametrize("path", [rw.SINGLE, rw.TWO_LAUNCH], ids=["mode1", "two_launch"])
def test_varlen_mode1_call_with_empty_sequences_and_padding_entries(ext, forced, path):
    """``cu_seqlens`` with empty sequences and entries whose slot is -1 (they own tokens that are skipped), over a state pool warmed to different fills. Each side asserts its own launch set."""
    H, S = 2, 9
    lens = [5, 0, 70, 1, 0, 130, 64, 33]
    N = len(lens)
    if path is rw.SINGLE:
        require_single_launch(ext, N, H)
    ids_warm = torch.tensor([3, 6, 0, 7, 1, 5, 8, 2], dtype=torch.int32, device="cuda")
    ids_call = torch.tensor([3, 6, 0, -1, 1, 5, -1, 2], dtype=torch.int32, device="cuda")
    warm = [(n * 13) % 70 + 1 for n in range(N)]                # tokens every sequence has run before the call (fills 1 .. 70)
    total = [w + L for w, L in zip(warm, lens)]
    case = make_case(sum(total), H, "authors", seed=9)
    cu = _cu(total)
    sub0, cu0 = _call_inputs(case, cu, [0] * N, warm)
    pool = alloc_state(S, H)
    prefill(sub0["q"], sub0["k"], sub0["v"], sub0["g"], sub0["beta"], case["scale"], torch.zeros_like(sub0["q"]), case["A_log"], case["dt_bias"], case["lower_bound"], pool, cu0,
            slot_ids=ids_warm, _impl=REF)
    sub1, cu1 = _call_inputs(case, cu, warm, lens)               # the call: the next ``lens[n]`` tokens of every sequence
    st_a, st_b = pool.clone(), pool.clone()
    oa, fa = mode1_on(path, case, sub1, cu1, st_a, ids_call)
    ob, fb = mode1_on(rw.ORACLE, case, sub1, cu1, st_b, ids_call)
    assert n_diff(oa, ob) == 0 and n_diff(fa, fb) == 0, "outputs differ"
    assert state_equal(st_a, st_b), state_mismatch(st_a, st_b)
    assert plain_pos(st_a)
    cu_l = _cu(lens)
    for n in (3, 6):                                           # the padding entries wrote nothing
        assert bool((oa[:, cu_l[n]:cu_l[n + 1]] == 3.0).all())
    assert torch.equal(st_a.pos[7], pool.pos[7]) and torch.equal(st_a.pos[8], pool.pos[8]), "the slots of the padding entries were touched"


# ---- controls: the equality checks and the launch assertions can fail ----------------------------------------------------------------------------
@pytest.mark.parametrize("what", ["out", "out32", "E", "row", "pos"])
def test_the_comparisons_of_this_module_see_a_corrupted_reference(ext, forced, what):
    """One flipped bit in the reference's bf16 output, fp32 output, E, a valid cache row or pos makes the comparison that the chain tests use (``n_diff``, ``state_equal``) fail, and only
    that one: the equality checks above are not vacuous."""
    H = 12
    N = active_n(ext, H)
    require_single_launch(ext, N, H)
    assert_live_floor(ext, [([1] * N, list(range(N)))], H)
    first = [61 + n % 6 for n in range(N)]
    case, cu, st = prefilled(H, N, N, first, 2, None, seed=31)
    st_a, st_b = st.clone(), st.clone()
    ins = step_inputs(case, cu, first, [0] * N)
    oa, fa = decode_on(rw.SINGLE, case, ins, st_a, None)
    ob, fb = decode_on(rw.ORACLE, case, ins, st_b, None)
    assert n_diff(oa, ob) == 0 and n_diff(fa, fb) == 0 and state_equal(st_a, st_b), "the honest comparison must pass first"
    bad = rw.corrupt(rw.Outcome(ob, fb, st_b), what)
    differs = {"out": n_diff(oa, bad.out) > 0, "out32": n_diff(fa, bad.out32) > 0, "state": not state_equal(st_a, bad.state)}
    assert differs == {"out": what == "out", "out32": what == "out32", "state": what in ("E", "row", "pos")}, (what, differs)
    if what in ("E", "row", "pos"):
        field = {"E": "E", "row": "rows", "pos": "pos"}[what]
        m = state_mismatch(st_a, bad.state)
        assert m.get(field, 0) > 0, m


def test_a_launch_assertion_fails_for_a_call_of_another_path(ext, forced):
    """The assertion of every side is path-sensitive: a call of one path run as another path fails it, so a test that silently runs the single launch on both sides (or the two-launch path on
    both) fails its own assertion."""
    N, H = 4, 6
    require_single_launch(ext, N, H)
    for path in rw.PATHS.values():
        rw.launched_exactly(path, decode_fn(N, H, path.impl))                        # a call of the path passes ...
        for other in rw.PATHS.values():
            if other is not path:
                with pytest.raises(AssertionError) as e:                                 # ... a call of another path does not
                    rw.launched_exactly(path, decode_fn(N, H, other.impl))
                assert path.name in str(e.value) and "launched" in str(e.value), str(e.value)


def test_the_default_request_is_not_a_path_the_tests_may_rely_on(ext):
    """Without the calibration control the same ``_impl=0`` call takes the single launch or the plan kernel by the policy of the device and the grid (a grid above the limit always takes the plan
    kernel): what a test means must not depend on that, which is why every side above names its path and asserts it."""
    H, sms = 12, _sms()
    N = (dp.MAX_CTAS_PER_SM * sms) // H + 1                                         # one sequence above the limit of the policy
    with pytest.raises(AssertionError):
        rw.launched_exactly(rw.SINGLE, decode_fn(N, H, 0))                         # the control is off here: the plan kernel stays
    rw.launched_exactly(rw.TWO_LAUNCH, decode_fn(N, H, 0))


def test_a_variant_that_never_stores_from_the_elected_cta_leaves_arrival_bits_and_fails_the_at_rest_check(ext, forced):
    """The build whose elected (last-arriving) CTA does not store (``CFBI_DECODE_TEST_NO_ELECTED_STORE``; the default build has no such macro) on a big grid: every arrival bit stays in the pos
    word (the old fill plus H arrivals above the counter shift), the at-rest check of the workloads reports it and the comparator refuses a state that is not at rest, while outputs, E and
    the valid rows are untouched (the missing store is the only difference). The default build, run the same way, is at rest."""
    H = 12
    mod = rw.build_variant("dec_nostore", ("CFBI_DECODE_TEST_NO_ELECTED_STORE",))
    mod.decode_single_force(True)                                   # a variant is a module of its own, with its own switch
    fam = rw.Family("no_store_control", fixed=1, pool=False, fills="mixed", steps=1)
    rng = random.Random(5)
    par = rw.make_params(H, 5)
    step, S = rw.draw_step(ext, fam, H, par, rng, 50, instance=helpers.ROWS_DECODE)
    caps = {"default build": helpers.resident_ctas(ext, rw.SINGLE.instance), "variant": helpers.resident_ctas(mod, rw.SINGLE.instance)}
    assert not rw.beyond_waves_problems([step.live_ctas()], caps), rw.beyond_waves_problems([step.live_ctas()], caps)      # beyond 2.5 live waves of the instantiation of BOTH builds, before any launch
    warm = rw.warm_pool(ext, S, H, rw.draw_fills(rng, S, fam.fills), par, 5)
    good = rw.run_arm(ext, rw.SINGLE, warm, step)
    bad = rw.run_arm(mod, rw.SINGLE, warm, step)                    # the variant launches the single launch too
    assert rw.at_rest_problems(step, warm, good) == []
    problems = rw.at_rest_problems(step, warm, bad)
    assert any("a position left [0, 63]" in p for p in problems), problems
    pos0, pos = warm.pos.cpu().tolist(), bad.state.pos.cpu().tolist()
    live = [step.slot(n) for n in range(step.N) if step.live(n)]
    assert live and all(pos[s] == pos0[s] + (H << 8) for s in live), "every live sequence must keep its old fill and all H arrivals above it"
    with pytest.raises(AssertionError) as e:
        rw.mismatch(bad, good)
    assert "a position left" in str(e.value), str(e.value)
    assert n_diff(bad.out, good.out) == 0 and n_diff(bad.out32, good.out32) == 0
    assert torch.equal(helpers.bits(bad.state.E), helpers.bits(good.state.E)) and torch.equal(helpers.bits(bad.state.rows), helpers.bits(good.state.rows))   # the whole tensors: stale rows too
    assert int((bad.state.pos != good.state.pos).sum()) == len(live), "pos differs exactly where a sequence ran"


# ---- memory safety at the edges on the single launch --------------------------------------------------------------------------------------------------
def test_the_single_launch_stays_inside_its_buffers_beyond_two_waves(ext, forced):
    """The guard-zone check of ``test_guard_zones.py`` on the single launch and a big grid: every tensor of the call (inputs, outputs, ``slot_ids``, E, the cache rows and pos) sits inside a
    larger buffer whose margins hold a NaN pattern; after three calls the margins are untouched (the arrival atomic and the elected store stay inside their slot), the inputs are unchanged and
    no arrival bit is left in pos. Padding entries (slot -1) and a pool with unused slots are part of the call."""
    import test_guard_zones as gz

    H = 12
    N, n_pad = live_grid.fraction_batch(active_n(ext, H), 8)          # the N // 8 padding entries come on top of the quota
    require_single_launch(ext, N, H)
    S = N + 9
    rng = random.Random(3)
    slots = rng.sample(range(S), N)
    for n in rng.sample(range(N), n_pad):
        slots[n] = -1
    assert_live_floor(ext, [([1] * N, slots)] * 3, H)
    case = make_case(N, H, "authors", seed=71)
    pool, pg = gz._state(S, H)
    gslots = gz.Guarded(torch.tensor(slots, dtype=torch.int32, device="cuda"))
    bufs = {k: gz.Guarded(case[k][:, :N].reshape(N, 1, *case[k].shape[2:]).contiguous()) for k in ("q", "k", "v", "g", "beta")}
    o1, o132 = gz._zeros_guarded(bufs["q"].t.shape, torch.bfloat16), gz._zeros_guarded(bufs["q"].t.shape, torch.float32)

    def call():
        decode(bufs["q"].t, bufs["k"].t, bufs["v"].t, bufs["g"].t, bufs["beta"].t, case["scale"], o1.t, case["A_log"], case["dt_bias"], case["lower_bound"], pool,
               out_fp32=o132.t, slot_ids=gslots.t, _impl=rw.SINGLE.impl)
    rw.launched_exactly(rw.SINGLE, call)
    call()
    call()
    torch.cuda.synchronize()
    gz._check({**{f"dec_{k}": v for k, v in bufs.items()}, "slots": gslots, "out": o1, "out32": o132, "E": pg[0], "rows": pg[1], "pos": pg[2]}, "single-launch decode")
    for k, v in bufs.items():
        assert v.content_unchanged(), f"decode modified its input '{k}'"
    assert plain_pos(pool), "an arrival counter is left in pos"
    used = [s for s in slots if s >= 0]
    assert all(int(pool.pos[s]) == 3 for s in used), "a sequence that ran three calls is not at position 3"
    assert all(int(pool.pos[s]) == 0 for s in range(S) if s not in used), "a slot that no sequence used was touched"


# ---- CUDA graph --------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("path", [rw.SINGLE, rw.TWO_LAUNCH], ids=["single", "two_launch"])
def test_cuda_graph_replay_with_rewritten_slot_ids_beyond_two_waves(ext, forced, path):
    """A fixed-shape decode batch of at least 2.5 waves of CTAs captured once on ``path`` and replayed with the slot assignment rewritten every step (entries change slots, some are
    padding); every replay equals the reference decode on a copy of the pool, bit for bit, and no arrival counter is left behind. ``_impl`` is passed explicitly (0 for the single launch, 64
    for the two-launch graph): under ``FLASH_KDA_CFBI_IMPL=64`` (Suite Sd) a call without it would capture the plan kernel and fail the assertion of the single-launch graph. The graph
    holds exactly the kernels of its path (the scratch launch before the capture and every replay are run under the profiler); no plan buffer exists in the single-launch graph."""
    H, steps = 12, 4
    N, n_pad = live_grid.fraction_batch(active_n(ext, H), 8)          # the N // 8 padding entries of every replay come on top of the quota
    if path is rw.SINGLE:
        require_single_launch(ext, N, H)
    assert_live_floor(ext, [([1] * N, [-1] * n_pad + list(range(N - n_pad)))] * steps, H, paths=(path,))     # the graph runs on one path: its instantiation only
    S = N + 9
    first = [125 + (n % 6) for n in range(N)]
    rng = random.Random(17)
    slots = rng.sample(range(S), N)                            # sequence n -> pool slot (fixed for its lifetime)
    ids0 = torch.tensor(slots, dtype=torch.int32, device="cuda")
    case, cu, pool = prefilled(H, N, S, first, steps, ids0, seed=23)
    ref_pool = pool.clone()
    bufs = {k: torch.zeros(N, *case[k].shape[2:], dtype=case[k].dtype, device="cuda") for k in ("q", "k", "v", "g", "beta")}
    out = torch.zeros_like(bufs["q"])
    o32 = torch.zeros(bufs["q"].shape, dtype=torch.float32, device="cuda")
    ids = torch.full((N,), -1, dtype=torch.int32, device="cuda")

    def launch():
        decode(bufs["q"], bufs["k"], bufs["v"], bufs["g"], bufs["beta"], case["scale"], out, case["A_log"], case["dt_bias"], case["lower_bound"], pool, out_fp32=o32, slot_ids=ids,
               _impl=path.impl)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):                               # warm-up on a scratch pool (attribute set-up, the hardware check of the first call)
        saved = pool
        pool = alloc_state(S, H)
        launch()
        pool = saved
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    saved = pool
    pool = alloc_state(S, H)
    cnt = kernel_counts(launch)                                 # on a scratch pool (every entry is padding): the kernels of the path, so the graph holds exactly those nodes
    pool = saved
    assert dict(cnt) == path.launches, dict(cnt)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    done = [0] * N
    for t in range(steps):
        order = list(range(N))
        rng.shuffle(order)                                      # entry i serves sequence order[i]
        pad = set(rng.sample(range(N), n_pad))                  # entries that are padding this step
        for i, n in enumerate(order):
            if i in pad:
                ids[i] = -1
                for k in bufs:
                    bufs[k][i].zero_()
            else:
                ids[i] = slots[n]
                idx = cu[n] + first[n] + done[n]
                for k in bufs:
                    bufs[k][i] = case[k][0, idx]
        out.fill_(3.0)
        o32.fill_(3.0)
        with recording() as rec:
            graph.replay()
        assert dict(rec.counts) == path.launches, f"replay {t}: the graph launched {dict(rec.counts)}, expected {path.launches}"
        ins = {k: bufs[k].clone() for k in bufs}
        ids_ref = ids.clone()
        ro, rf = decode_on(rw.ORACLE, case, ins, ref_pool, ids_ref)
        torch.cuda.synchronize()
        assert n_diff(out, ro) == 0 and n_diff(o32, rf) == 0, f"replay {t}: outputs differ from the reference"
        assert state_equal(pool, ref_pool), f"replay {t}: state differs {state_mismatch(pool, ref_pool)}"
        assert plain_pos(pool), f"replay {t}: an arrival counter is left in pos"
        for i, n in enumerate(order):
            if i not in pad:
                done[n] += 1
