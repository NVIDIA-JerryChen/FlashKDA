"""compute-sanitizer over the row engine: memcheck and synccheck over decode beyond two resident waves, racecheck over a bounded workload, and an out-of-bounds control.

The driver (a separate process under the tool) runs mode-1 launches of the optimised row engine on a pool through ``rows_workload``: a ``cu_seqlens`` launch of 1 .. 70 rows per sequence
(open chunks at every position, chunk ends crossed, empty sequences), then a decode launch and a five-row launch with new permuted slots and padding entries (slot -1), every grid beyond
2.5 resident waves of live CTAs (sized from the occupancy binding). The runs are UNFILTERED (no ``--kernel-name``) and each is bounded by ``TIMEOUT_S``; a run that exceeds it fails.

* memcheck and synccheck report no error. racecheck covers shared-memory hazards, which is why its workload is bounded: four CTAs, but one of them runs 66 rows from position 0, so it
  executes every position 0 .. 63 (every sub-block, the cross-sub-block phase, the chunk-end phase) and two rows after the chunk end (the window that the barrier ending a row protects).
  What the workload executes is derived from its plan (``rows_workload.launch_coverage``), printed by the driver before the tool starts, required to cover the row engine
  (``coverage_problems``) and checked after the run against the final positions of the pool; a workload whose long sequence sits on a padding slot is rejected by a CPU rule. A racecheck
  run that cannot analyse the kernel FAILS here (there is no skip). initcheck is not a gate and is not run by the suite.
* The control: a variant that stores the fused position of a sequence at ``pos[slot + S + 2^24]`` (``CFBI_ROWS_TEST_POS_OOB``, a separate build) is reported by memcheck as an invalid
  global write in ``rows_kernel`` (with the caching allocator off so that the pool's ``pos`` is its own allocation); the same variant with offset 0 is clean, so the report comes from the
  offset and not from the build.

A missing compute-sanitizer fails on B200 / B300 (``helpers.skip_or_fail``), a variant that does not build fails (``rows_workload.build_variant``).
"""
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import rows_workload as rw
from helpers import skip_or_fail

TIMEOUT_S = 1800
REPO = Path(__file__).resolve().parents[2]
TESTS = REPO / "tests" / "cfbi"
OOB = ("CFBI_ROWS_TEST_POS_OOB=1",)
BOUNDED_H, BOUNDED_POOL = 1, 8
BOUNDED_LAUNCHES = [([1, 66, 3, 5], [3, 7, -1, 0]),     # a 66-row sequence on the live slot 7 (positions 0 .. 63 and two rows after the chunk end), a padding entry, permuted slots
                    ([1, 1, 1], [7, 3, 0])]              # a decode launch on the state that the first launch left (slot 7 at position 2)
OOB_OFFSET_BEYOND = 1 << 24
DONE = "ROWS_SANITIZER_WORKLOAD_DONE"

PRELUDE = textwrap.dedent('''
    import random, sys
    sys.path.insert(0, "{tests}")
    sys.path.insert(0, "{repo}")
    import torch
    import helpers
    import rows_workload as rw
    from flash_kda.cfbi import alloc_state, load_ext
    from rows_workload import Family, Step
''')

WAVES_DRIVER = PRELUDE + textwrap.dedent('''
    ext = load_ext()
    H = 12
    rng = random.Random(5)
    par = rw.make_params(H, 5)
    fams = [Family("san_varlen", lens=[1, 7, 33, 62, 63, 64, 65, 70], pool=True, pad=0.3, empty=0.2),
            Family("san_decode", fixed=1, pool=True, pad=0.3),
            Family("san_multirow", fixed=5, pool=True, pad=0.3)]
    live = helpers.sequences_beyond_waves(helpers.resident_ctas(ext), H)
    S = int(live * 1.5) + 19
    pool = alloc_state(S, H)
    for i, fam in enumerate(fams):
        step, S2 = rw.draw_step(ext, fam, H, par, rng, 50 + i, pool_size=S)
        assert step.live_ctas() >= helpers.WAVES * helpers.resident_ctas(ext), "the grid is not beyond %.1f waves" % helpers.WAVES
        rw.run_step(ext, 0, pool, step)
    torch.cuda.synchronize()
    print("{done} live_ctas", live * H, "pool", S)
''')

BOUNDED_DRIVER = PRELUDE + textwrap.dedent('''
    import json
    H, S = {heads}, {pool}
    launches = {launches}
    cov = rw.launch_coverage(launches, H)
    print("ROWS_SANITIZER_COVERAGE", json.dumps(cov, sort_keys=True), flush=True)       # what the workload executes, derived from its plan, BEFORE the tool has anything to say
    problems = rw.coverage_problems(cov)
    assert not problems, "the bounded workload does not cover the row engine: %s" % problems
    ext = load_ext()
    par = rw.make_params(H, 6)
    pool = alloc_state(S, H)
    for i, (lens, slots) in enumerate(launches):
        rw.run_step(ext, 0, pool, Step(par, H, lens, slots, len(set(lens)) == 1, 60 + i))
    torch.cuda.synchronize()
    final = pool.pos.cpu().tolist()
    print("ROWS_SANITIZER_FINAL_POS", json.dumps(final), flush=True)                    # the positions that the rows really left behind
    expected = [cov["final_pos"].get(s, 0) for s in range(S)]
    assert final == expected, "the launches left the positions %s, the plan says %s: the long sequence did not run" % (final, expected)
    print("{done} bounded")
''')

OOB_DRIVER = PRELUDE + textwrap.dedent('''
    mod = rw.build_variant("rowsoob", ("CFBI_ROWS_TEST_POS_OOB=1",))
    H, S = 2, 12
    par = rw.make_params(H, 7)
    pool = alloc_state(S, H)
    mod.rows_test_pos_offset({offset})
    step = Step(par, H, [1, 1, 1, 1], [3, 7, 1, 9], True, 71)
    rw.run_step(mod, 64, pool, step)      # _impl 64: the plan kernel and the row engine behind it, the path whose fused position store the variant offsets (a decode call of the single launch has no such store)
    torch.cuda.synchronize()
    print("{done} offset", {offset})
''')


def sanitizer():
    exe = shutil.which("compute-sanitizer")
    if exe is None:
        skip_or_fail("compute-sanitizer is not installed")
    return exe


def run_driver(tool, tmp_path, driver, extra=(), env=None, offset="S + %d" % OOB_OFFSET_BEYOND):
    exe = sanitizer()
    script = tmp_path / "driver.py"
    script.write_text(driver.format(tests=str(TESTS), repo=str(REPO), done=DONE, offset=offset, heads=BOUNDED_H, pool=BOUNDED_POOL, launches=repr(BOUNDED_LAUNCHES)))
    # application-only: the tool does not follow child processes (a driver that had to rebuild the extension would run ninja and nvcc under it, which hangs)
    cmd = [exe, "--tool", tool, "--error-exitcode", "99", "--target-processes", "application-only", *extra, sys.executable, str(script)]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, env=dict(os.environ, **(env or {})))
    except subprocess.TimeoutExpired:
        pytest.fail("compute-sanitizer %s exceeded the bound of %d s" % (tool, TIMEOUT_S), pytrace=False)


@pytest.fixture(autouse=True)
def _optimised_kernels_run_here(ext):
    if ext.resolve_impl(rw.OPTIMISED, -1) == ext.resolve_impl(rw.REFERENCE, -1):
        skip_or_fail("the optimised kernels do not run on this device (_impl=0 resolves to the reference kernels)")


@pytest.mark.parametrize("tool", ["memcheck", "synccheck"])
def test_sanitizer_reports_no_error_over_decode_beyond_two_waves_with_padding_slots(tool, tmp_path):
    r = run_driver(tool, tmp_path, WAVES_DRIVER)
    text = r.stdout + r.stderr
    assert DONE in text, text[-3000:]
    assert r.returncode == 0 and "ERROR SUMMARY: 0 errors" in text, text[-3000:]


def printed(text, label):
    """The JSON that the driver printed after ``label`` (the last such line)."""
    lines = [ln for ln in text.splitlines() if ln.startswith(label + " ")]
    assert lines, "the driver did not print %s: %s" % (label, text[-1500:])
    return json.loads(lines[-1][len(label) + 1:])


def test_racecheck_reports_no_hazard_over_a_bounded_workload_that_crosses_a_chunk_end(tmp_path):
    t0 = time.time()
    r = run_driver("racecheck", tmp_path, BOUNDED_DRIVER, extra=("--racecheck-report", "hazard"))
    seconds = time.time() - t0
    text = r.stdout + r.stderr
    assert DONE in text, "racecheck could not run the workload: " + text[-3000:]
    cov = printed(text, "ROWS_SANITIZER_COVERAGE")
    plan = json.loads(json.dumps(rw.launch_coverage(BOUNDED_LAUNCHES, BOUNDED_H), sort_keys=True))
    assert cov == plan, "the driver executed another workload than the plan of the test"
    assert rw.coverage_problems(cov) == [], rw.coverage_problems(cov)
    final = printed(text, "ROWS_SANITIZER_FINAL_POS")
    assert final == [plan["final_pos"].get(str(s), 0) for s in range(BOUNDED_POOL)], final
    assert r.returncode == 0 and "RACECHECK SUMMARY: 0 hazards" in text, text[-3000:]
    print("\nbounded racecheck: %d rows at all %d positions (%d chunk-end rows, %d after a chunk end, %d beyond the first sub-block) in %d CTAs of the first launch, %.1f s, 0 hazards" % (
        cov["rows"], len(cov["positions"]), cov["chunk_end_rows"], cov["rows_after_chunk_end"], cov["cross_subblock_rows"], cov["live_ctas"][0], seconds))


# ---- the control: an out-of-bounds store of the position is reported ----------------------------------------------------------------------------------------------
def test_memcheck_reports_the_position_store_beyond_the_pool(tmp_path):
    rw.build_variant("rowsoob", OOB)                       # built here, outside the tool; the driver loads the cached build
    r = run_driver("memcheck", tmp_path, OOB_DRIVER, env={"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"})
    text = r.stdout + r.stderr
    assert r.returncode == 99, "memcheck did not report the out-of-bounds position store (exit %d): %s" % (r.returncode, text[-2000:])
    assert "Invalid __global__ write" in text and "rows_kernel" in text, text[-3000:]


def test_the_same_variant_with_offset_zero_is_clean_under_memcheck(tmp_path):
    rw.build_variant("rowsoob", OOB)
    r = run_driver("memcheck", tmp_path, OOB_DRIVER, env={"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"}, offset="0")
    text = r.stdout + r.stderr
    assert DONE in text and r.returncode == 0 and "ERROR SUMMARY: 0 errors" in text, text[-3000:]
