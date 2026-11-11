"""The live grid of the beyond-waves tests, planned and asserted from what the launches execute (plain Python: no GPU; ``PYTHONPATH=. pytest --noconftest tests/cfbi/test_decode_live_grid.py``).

The mode-1 test of the single-launch module (round 7) sized N x H = 741 CTAs, then made every eleventh entry empty and split the lengths 1 and 2 into ``[0, 0, L]``: at 148 SMs x 2 resident
CTAs its three calls had 495, 495 and 675 live CTAs (1.67, 1.67 and 2.28 waves), and the guard that existed, ``rows_workload.coverage_problems`` (which only asks for one live CTA), passed. The
pool chain (padding every seventh entry), the graph replays and the guard-zone test (``N // 8`` padding entries) had the same defect on a smaller scale. Here:

* the round-7 planner is kept as a literal control: it reaches 495 / 495 / 675 live CTAs, the old guard passes it and the per-launch floor (``rows_workload.beyond_waves_problems``) rejects it;
* the planners of ``live_grid`` reserve the active quota and add the extras on top: every planned launch reaches ``ceil(2.5 x resident CTAs)`` live CTAs for 2 and 3 resident CTAs per SM and for 3, 12
  and 33 heads, and the extras (empty entries, zero-take entries, padding) are still there;
* the bounded racecheck workloads stay small by design: ``coverage_problems`` is NOT given a floor (a control shows that its workload passes it and fails the floor);
* every test that sizes a workload with ``active_n`` asserts ``assert_live_floor`` (an AST rule that fails on a copy of the source without it).
"""
import ast
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import helpers  # noqa: E402
import live_grid as lg  # noqa: E402
import rows_workload as rw  # noqa: E402

TESTS = ROOT / "tests" / "cfbi"
SMS = (132, 148)
PER_SM = (2, 3)
HEADS = (3, 12, 33)
CAPACITIES = [sms * per for sms in SMS for per in PER_SM]
GPU_MODULES = ("test_decode_single_launch.py", "test_decode_single_launch_variants.py")


def round7_mode1(N, seed=7):
    """The planner of the round-7 mode-1 test, literally: N entries sized from the nominal grid, every eleventh empty, lengths 1 and 2 split into ``[0, 0, L]``."""
    rng = random.Random(seed)
    lens = [0 if n % 11 == 5 else rng.choice([1, 2, 5, 33, 64, 65, 97, 130]) for n in range(N)]
    takes = [[L // 3 if c < 2 else L - 2 * (L // 3) for L in lens] for c in range(3)]
    return lens, takes


def test_the_constants_agree_with_the_helpers():
    assert lg.WAVES == helpers.WAVES == rw.WAVES
    for cap in CAPACITIES:
        for H in HEADS + (1, 6, 64):
            assert lg.active_quota(cap, H) == helpers.sequences_beyond_waves(cap, H), (cap, H)
            assert lg.active_quota(cap, H) * H >= lg.live_floor(cap) > (lg.active_quota(cap, H) - 1) * H


# ---- the round-7 planner: the literal counterexample --------------------------------------------------------------------------------------------------
def test_the_round_7_mode_1_planner_has_495_495_675_live_ctas_and_the_old_guard_passed_it():
    H, cap = 3, 296                                                   # 148 SMs x 2 resident CTAs of the optimised row engine
    N = helpers.sequences_beyond_waves(cap, H)
    assert N == 247 and N * H == 741 >= lg.live_floor(cap) == 740       # the nominal grid is beyond 2.5 waves ...
    lens, takes = round7_mode1(N)
    cov = rw.launch_coverage(lg.launches(takes), H)
    assert cov["live_ctas"] == [495, 495, 675]                         # ... and the live grid of the three calls is not
    assert rw.coverage_problems(cov) == []                             # the guard that existed asks for one live CTA only
    probs = rw.beyond_waves_problems(cov["live_ctas"], {"rows_kernel<true>": cap})
    assert len(probs) == 3 and "launch 0: 495 live CTAs are 1.67 waves" in probs[0] and "launch 2: 675 live CTAs are 2.28 waves" in probs[2], probs


def test_a_nominally_large_batch_whose_active_part_is_small_fails_the_floor():
    """The padding of the pool chain (every seventh entry) and of the graph and guard tests (N // 8 entries) taken out of a nominal grid: 636 and 660 live CTAs at 12 heads."""
    H, cap = 12, 296
    N = helpers.sequences_beyond_waves(cap, H)
    assert N == 62
    pool_chain = [([1] * N, [-1 if i % 7 == 3 else i for i in range(N)])]
    graph = [([1] * N, [-1] * (N // 8) + list(range(N - N // 8)))]
    for plan, live in ((pool_chain, 636), (graph, 660)):
        cov = rw.launch_coverage(plan, H)
        assert cov["live_ctas"] == [live]
        assert rw.beyond_waves_problems(cov["live_ctas"], {"rows_kernel<false>": cap}) != []


def test_every_instantiation_of_the_compared_paths_is_held_to_its_own_capacity():
    cov = rw.launch_coverage([([1] * 70, list(range(70)))], 12)           # 840 live CTAs
    assert rw.beyond_waves_problems(cov["live_ctas"], {"a": 296, "b": 296}) == []
    probs = rw.beyond_waves_problems(cov["live_ctas"], {"a": 296, "b": 400})   # 400 resident CTAs need 1000 live ones
    assert len(probs) == 1 and "rows_kernel" not in probs[0] and " b " in probs[0] + " ", probs
    assert rw.beyond_waves_problems([], {"a": 296}) == []


# ---- the repaired planners ----------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cap", CAPACITIES)
@pytest.mark.parametrize("H", HEADS)
def test_the_mode_1_planner_reaches_the_floor_in_every_call_and_keeps_the_extras(cap, H):
    active = lg.active_quota(cap, H)
    lens, takes = lg.mode1_plan(active)
    cov = rw.launch_coverage(lg.launches(takes), H)
    assert rw.beyond_waves_problems(cov["live_ctas"], {"rows_kernel<true>": cap}) == [], cov["live_ctas"]
    assert min(cov["live_ctas"]) >= lg.live_floor(cap)
    assert sum(1 for L in lens if L >= min(lg.MODE1_LENGTHS)) >= active and all(sum(t[n] for t in takes) == lens[n] for n in range(len(lens)))   # every length is taken apart exactly
    assert sum(1 for L in lens if L == 0) > 0 and any(L in (1, 2) for L in lens)                 # the extras are still in the workload ...
    assert any(takes[0][n] == 0 and takes[2][n] > 0 for n in range(len(lens)))                     # ... entries that take no token in the first two calls too
    assert all(takes[c][n] > 0 for n, L in enumerate(lens) if L >= min(lg.MODE1_LENGTHS) for c in range(3))     # the active ones take tokens in every call
    assert rw.coverage_problems(cov) == [], cov                                                   # positions 0..63, a chunk end, rows after it, rows with a next row: unchanged


@pytest.mark.parametrize("cap", CAPACITIES)
@pytest.mark.parametrize("H", HEADS)
@pytest.mark.parametrize("every, residue", [(7, 3), (5, 0), (4, 3)])
def test_a_padded_batch_leaves_the_active_quota_live(cap, H, every, residue):
    active = lg.active_quota(cap, H)
    N, padding = lg.padded_batch(active, every, residue)
    assert N - len(padding) == active and padding and all(i % every == residue for i in padding)
    prev_live = (N - 1) - len([i for i in padding if i < N - 1])
    assert prev_live < active                                          # the smallest batch with that many live entries
    cov = rw.launch_coverage([([1] * N, [-1 if i in set(padding) else i for i in range(N)])], H)
    assert rw.beyond_waves_problems(cov["live_ctas"], {"i": cap}) == []


@pytest.mark.parametrize("cap", CAPACITIES)
@pytest.mark.parametrize("H", HEADS)
def test_a_fraction_batch_leaves_the_active_quota_live(cap, H):
    active = lg.active_quota(cap, H)
    N, n_pad = lg.fraction_batch(active, 8)
    assert N - n_pad >= active and n_pad == N // 8 and (N - 1) - (N - 1) // 8 < active
    cov = rw.launch_coverage([([1] * N, [-1] * n_pad + list(range(N - n_pad)))] * 4, H)
    assert rw.beyond_waves_problems(cov["live_ctas"], {"i": cap}) == []


# ---- the bounded racecheck workloads stay small: the floor is not part of coverage_problems -----------------------------------------------------------
def test_the_bounded_racecheck_workload_passes_coverage_problems_and_is_not_beyond_waves():
    bounded = [([1, 66, 3, 5], [3, 7, -1, 0]), ([1, 1, 1], [7, 3, 0])]       # the workload of tests/cfbi/test_rows_sanitizer.py: four CTAs, one of them runs 66 rows
    cov = rw.launch_coverage(bounded, 1)
    assert rw.coverage_problems(cov) == []
    assert rw.beyond_waves_problems(cov["live_ctas"], {"rows_kernel<false>": 296}) != []
    for live, ok in (([1, 1], True), ([10 ** 6], True), ([5, 0], False), ([], False)):       # the one live CTA per launch is all that coverage_problems asks: no floor in waves
        probs = rw.coverage_problems(dict(cov, live_ctas=live))
        assert (probs == []) == ok, (live, probs)
        assert ok or probs == ["a launch has no live CTA (every entry is a padding or empty entry)"], probs


# ---- every user of the quota asserts the floor ---------------------------------------------------------------------------------------------------------
def floor_problems(source):
    """The functions that size a workload with ``active_n`` but never call ``assert_live_floor``."""
    out = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef):
            names = {c.func.id for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
            if "active_n" in names and "assert_live_floor" not in names:
                out.append(node.name)
    return out


@pytest.mark.parametrize("module", GPU_MODULES)
def test_every_test_that_sizes_a_workload_with_the_active_quota_asserts_the_floor(module):
    source = (TESTS / module).read_text()
    assert "active_n(" in source and "big_n" not in source                 # the retired name (a nominal grid) is gone
    assert floor_problems(source) == []


def test_the_floor_rule_fails_on_a_copy_without_the_assertion():
    source = (TESTS / "test_decode_single_launch.py").read_text()
    head, sep, tail = source.partition("def test_mode1_calls_with_several_tokens_per_sequence_across_chunks_on_a_big_grid")
    assert sep
    mutant = head + sep + tail.replace("assert_live_floor(ext, launches, H)", "pass", 1)
    assert floor_problems(mutant) == ["test_mode1_calls_with_several_tokens_per_sequence_across_chunks_on_a_big_grid"]
    assert len(floor_problems(source.replace("assert_live_floor(", "unchecked("))) >= 6
