"""Decode families of the row engine on grids beyond 2.5 resident waves, bit for bit against the frozen reference (``_impl=3``).

The tests that could pass before only reached grids that fit in one or two waves of resident CTAs, one kind of state and one kind of call. These families reach the conditions where a change
of the row engine could break and a small test would not notice (a flag, a spin or a schedule that works while every CTA is resident; a pool slot that is permuted, unused or padding; an open
chunk at every position, the chunk end included; rows that follow a chunk end inside one launch):

* the grid is sized from the device: ``helpers.decode_sequences`` = SMs times the CTAs per SM that the occupancy API reports (the hidden binding), so the *live* CTAs (not the padding or
  empty entries, which exit at once) fill at least 2.5 waves; the numbers are asserted, not assumed;
* mode 1 with a fixed length (one row = decode, five rows = a multi-row launch that runs over the chunk end) and with ``cu_seqlens`` (lengths 1 .. 8, empty sequences);
* a compact state and a pool with permuted, non-contiguous slots, padding entries (slot -1) and slots that no sequence uses; a warm state whose open chunks sit at every kind of position
  (made by the reference kernels, so that both arms start from the oracle's own state), carried through three successive launches (positions advance and wrap);
* after every launch the optimised arm is checked on its own (positions stay in [0, 63] and advance by the rows that ran, a slot that no sequence used is untouched bit for bit, padding and
  empty rows keep their sentinel, rows that ran are finite) and against the reference (bf16 and fp32 output, E, the valid rows of the cache, pos: identical bits).

Controls that must fail, so that the comparison can fail: one flipped bit in each compared object is reported (and a flipped bit in a stale cache row, which is not part of the state, is
not); a launch from a stale state differs; the build with the sabotaged fma (``CFBI_NEG_CONTROL``) differs from the reference on these workloads; a variant that does not build fails its
test instead of skipping it.

Every family runs on BOTH decode paths, named explicitly (``rows_workload.Path``): ``single`` is ``_impl=0`` with the hidden calibration control on (one ``rows_kernel<true>``), ``two_launch`` is
``_impl=64`` (the plan kernel and ``rows_kernel<false>``); the oracle is ``_impl=3``. Every launch of every step is run under the profiler and must show exactly the kernels of its own path, and the
grid is sized from the occupancy of the instantiation that the path launches, so no device, routing default or environment variable decides what a family covers. The ``phases`` families start
their (at least 64) live sequences at the positions n mod 64 in a shuffled order: the positions that their rows ran at are derived from the executed steps and must be all 64.

Every group of cases that launches kernels (the families of one head count and path, the controls) runs in a child process under ``GROUP_TIMEOUT_S`` (``rows_bounded``): a hang fails the test and kills
the child, and every case keeps its id and its assertions (the group functions below return the outcome of each case; the tests assert the one they are named after).
"""
import pytest

import rows_bounded as rb
import rows_workload as rw
from helpers import WAVES, resident_ctas, skip_or_fail
from rows_workload import Family

FAMILIES = [
    Family("decode_compact", fixed=1, pool=False, fills="mixed"),
    Family("decode_pool_permuted", fixed=1, pool=True, fills="mixed"),
    Family("decode_pool_padding", fixed=1, pool=True, pad=0.3, fills="mixed"),
    Family("decode_open_chunks", fixed=1, pool=True, pad=0.2, fills="open"),
    Family("multirow_compact", fixed=5, pool=False, fills="crossing"),
    Family("multirow_pool_padding", fixed=5, pool=True, pad=0.3, fills="crossing"),
    Family("varlen_compact", lens=[1, 2, 3, 5, 8], pool=False, empty=0.25, fills="crossing"),
    Family("varlen_pool", lens=[1, 2, 3, 5, 8], pool=True, pad=0.25, empty=0.25, fills="mixed"),
    Family("decode_all_phases", fixed=1, pool=False, fills="phases", min_live=rw.BT),
    Family("multirow_all_phases", fixed=5, pool=False, fills="phases", min_live=rw.BT),
]
BY_NAME = {f.name: f for f in FAMILIES}
HEADS = [12, 3]
PATH_NAMES = ["single", "two_launch"]                                                              # the paths that the families run on (rows_workload.PATHS)
FLIPS = [("out", "out"), ("out32", "out32"), ("E", "E"), ("row", "rows"), ("pos", "pos")]       # the object that is corrupted, the field that must report it
GROUP_TIMEOUT_S = 600     # per group: the families of one head count and path take under a minute and the controls about 2 minutes on a B200 (a first run adds the JIT builds); a hang ends here


@pytest.fixture(autouse=True)
def _optimised_kernels_run_here(ext):
    """Below sm_90 ``_impl=0`` resolves to the reference kernels: the comparison would be a build with itself."""
    if ext.resolve_impl(rw.OPTIMISED, -1) == ext.resolve_impl(rw.REFERENCE, -1):
        skip_or_fail("the optimised kernels do not run on this device (_impl=0 resolves to the reference kernels)")


def expectations(fam, cov, ext, instance):
    """The conditions that the family must have reached on a path that launches the row-engine instantiation ``instance``: the problems found (an empty list = every condition reached)."""
    probs = []
    cap = resident_ctas(ext, instance)
    if cov["live_ctas"] < WAVES * cap:
        probs.append("live grid %d CTAs is below %.1f waves of %d" % (cov["live_ctas"], WAVES, cap))
    if cov["grid_ctas"] < WAVES * cap:
        probs.append("grid %d CTAs is below %.1f waves" % (cov["grid_ctas"], WAVES))
    if cov["chunk_end_rows"] == 0:
        probs.append("no row finished a chunk (position 63)")
    if fam.pool and not cov["permuted"]:
        probs.append("the slots were not permuted")
    if fam.pool and cov["pool_unused"] < rw.POOL_SPARE:
        probs.append("the pool had only %d unused slots" % cov["pool_unused"])
    if fam.pad and cov["padding"] == 0:
        probs.append("no padding entry")
    if fam.empty and cov["empty"] == 0:
        probs.append("no empty sequence")
    if fam.fixed and fam.fixed > 1 and cov["rows_after_chunk_end"] == 0:
        probs.append("no launch ran on past a chunk end")
    if fam.fixed is None and cov["rows_after_chunk_end"] == 0:
        probs.append("no launch ran on past a chunk end")
    if fam.fills == "open" and cov["open_chunk_sequences"] < cov["live_sequences"] * 0.9:
        probs.append("only %d of %d sequences continued an open chunk" % (cov["open_chunk_sequences"], cov["live_sequences"]))
    if (fam.fixed or 0) <= 1 and fam.fixed is not None and cov["max_rows"] != 1:
        probs.append("a decode family ran more than one row of a sequence")
    if fam.fills == "phases":
        if cov["positions"] != list(range(rw.BT)):
            missing = sorted(set(range(rw.BT)) - set(cov["positions"]))
            probs.append("the rows ran at %d of the 64 positions only (missing %s ..)" % (len(cov["positions"]), missing[:6]))
        if cov["live_sequences"] < fam.steps * rw.BT:
            probs.append("fewer than %d live sequences per step (%d in %d steps)" % (rw.BT, cov["live_sequences"], cov["steps"]))
    return probs


# ---- the cases (they run in the child process of their group) -------------------------------------------------------------------------------------------
def family_case(ext, fam, H, path_name):
    path = rw.PATHS[path_name]
    results = rw.run_family(ext, fam, H, seed=11, path=path)
    cov = rw.coverage(results)
    assert cov["steps"] == fam.steps
    assert expectations(fam, cov, ext, path.instance) == [], cov


def flip_case(two_steps, what, field):
    step, opt, ref, warm = two_steps[-1]
    assert all(v == 0 for v in rw.mismatch(opt, ref).values())
    m = rw.mismatch(rw.corrupt(opt, what), ref)
    assert m == {k: (1 if k == field else 0) for k in m}, (what, m)


def stale_row_case(two_steps):
    step, opt, ref, warm = two_steps[-1]
    assert all(v == 0 for v in rw.mismatch(rw.corrupt(opt, "stale_row"), ref).values())


def stale_state_case(ext, two_steps):
    (step1, opt1, ref1, warm1), (step2, opt2, ref2, warm2) = two_steps[0], two_steps[1]
    stale = rw.run_arm(ext, rw.SINGLE, warm1, step2)           # the state before the previous launch: the oracle ran from the state after it
    m = rw.mismatch(stale, ref2)
    assert sum(m.values()) > 0, m


def sabotaged_case(ext, path_name):
    """The row engine built with the sabotaged fma (a module of its own: its calibration control is switched on here), run on the path ``path_name``, differs from the oracle on a decode
    family."""
    path = rw.PATHS[path_name]
    bad = rw.build_variant("negctl", ("CFBI_NEG_CONTROL=1",))
    bad.decode_single_force(True)
    results = rw.run_family(ext, BY_NAME["decode_pool_padding"], 12, seed=31, path=path)
    step, opt, ref, warm = results[0]
    sab = rw.run_arm(bad, path, warm, step)
    m = rw.mismatch(sab, ref)
    assert m["out32"] > 0 or m["out"] > 0 or m["E"] > 0 or m["rows"] > 0, "the sabotaged build matches the reference: the workload cannot see a changed row engine (%s)" % m


# ---- the groups (one child process each) ------------------------------------------------------------------------------------------------------------------
def group_families(ext, heads, path):
    """Every family at ``heads`` heads on the path ``path`` (a name of ``rows_workload.PATHS``): the outcome of each (None = passed). The calibration control is process-wide state: the child
    process of the group switches it on, so ``single`` is ``_impl=0`` forced and the other launches of the group show that it overrides neither bit 64 nor the reference."""
    ext.decode_single_force(True)
    return {"cases": {repr(fam): rb.capture(family_case, ext, fam, heads, path) for fam in FAMILIES}, "reps": 1}


def group_controls(ext):
    """The controls that make the comparison fail on purpose; they share two successive launches of a pool family with padding and empty sequences on the single launch (both arms equal,
    asserted by ``run_family``)."""
    ext.decode_single_force(True)
    two_steps = rw.run_family(ext, BY_NAME["varlen_pool"], 12, seed=21, path=rw.SINGLE)
    assert len(two_steps) >= 2
    cases = {"flipped:" + what: rb.capture(flip_case, two_steps, what, field) for what, field in FLIPS}
    cases["stale_row"] = rb.capture(stale_row_case, two_steps)
    cases["stale_state"] = rb.capture(stale_state_case, ext, two_steps)
    for name in PATH_NAMES:
        cases["sabotaged_build:" + name] = rb.capture(sabotaged_case, ext, name)
    return {"cases": cases, "reps": 1}


@pytest.fixture(scope="module")
def family_groups(ext):
    """The result of the family group of a head count and path, run once in its own bounded child process."""
    cache = {}

    def get(heads, path):
        if (heads, path) not in cache:
            cache[(heads, path)] = rb.run_group("test_rows_decode_families", "group_families", {"heads": heads, "path": path}, GROUP_TIMEOUT_S)
        return cache[(heads, path)]

    return get


@pytest.fixture(scope="module")
def control_group(ext):
    return rb.run_group("test_rows_decode_families", "group_controls", {}, GROUP_TIMEOUT_S)


@pytest.mark.parametrize("path", PATH_NAMES)
@pytest.mark.parametrize("H", HEADS)
@pytest.mark.parametrize("fam", FAMILIES, ids=repr)
def test_the_family_is_bitwise_the_reference_beyond_2_5_waves(family_groups, fam, H, path):
    outcome = family_groups(H, path)["cases"][repr(fam)]
    assert outcome is None, outcome


# ---- controls: the comparison can fail ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("what,field", FLIPS)
def test_a_single_flipped_bit_is_reported_in_the_object_that_holds_it(control_group, what, field):
    outcome = control_group["cases"]["flipped:" + what]
    assert outcome is None, outcome


def test_a_flipped_bit_in_a_stale_cache_row_is_not_part_of_the_state(control_group):
    assert control_group["cases"]["stale_row"] is None, control_group["cases"]["stale_row"]


def test_a_launch_from_a_stale_state_differs(control_group):
    assert control_group["cases"]["stale_state"] is None, control_group["cases"]["stale_state"]


@pytest.mark.parametrize("path", PATH_NAMES)
def test_the_sabotaged_row_engine_build_differs_from_the_reference_on_these_workloads(control_group, path):
    outcome = control_group["cases"]["sabotaged_build:" + path]
    assert outcome is None, outcome


def test_a_variant_that_does_not_build_fails_instead_of_skipping():
    with pytest.raises(pytest.fail.Exception) as e:
        rw.build_variant("rowsbroken", ("CFBI_ROWS_TEST_COMPILE_ERROR=1",))
    assert "does not build" in str(e.value) and "deliberate compile error" in str(e.value), str(e.value)[-500:]
