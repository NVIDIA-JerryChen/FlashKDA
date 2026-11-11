"""compute-sanitizer over the row engine on BOTH paths of the decode: memcheck and synccheck over decode beyond two resident waves, racecheck over a bounded workload, and out-of-bounds controls.

Two paths, each named, forced and asserted: the single launch (``rows_kernel<true>``, one kernel per call; the hidden calibration control on in the driver, ``_impl=0``) and the two launches (the
plan kernel and ``rows_kernel<false>``, ``_impl=64``). The profiler cannot attach under the tool, so the launch set of a path is asserted WITHOUT it: the test runs the very workload that the
driver runs (``sanitizer_workloads``) once in its own process, every launch asserted to be the kernels of the path (``rows_workload.launched_exactly``), and compares the digest of the plan that it
executed with the digest that the driver printed under the tool (sizes, live CTAs of every launch, checksums of the slots and lengths). A path that did not run what it names fails here.

The driver (a separate process under the tool) runs mode-1 launches on a pool: a ``cu_seqlens`` launch of 1 .. 70 rows per sequence (open chunks at every position, chunk ends crossed, empty
sequences), a decode launch and a five-row launch with new permuted slots and padding entries (slot -1), every grid beyond 2.5 resident waves of LIVE CTAs of the instantiation that the path
launches (sized from the occupancy binding, asserted before the launch). The runs are UNFILTERED (no ``--kernel-name``) and each is bounded by ``TIMEOUT_S``; a run that exceeds it fails.

* memcheck and synccheck report no error. racecheck covers shared-memory hazards (among them the hand-over of the fill from thread 0 to the CTA in the single launch), which is why its workload is
  bounded: a few CTAs, but one of them runs 66 rows from position 0, so it executes every position 0 .. 63 (every sub-block, the cross-sub-block phase, the chunk-end phase) and two rows after the
  chunk end (the window that the barrier ending a row protects). What the workload executes is derived from its plan (``rows_workload.launch_coverage``), printed by the driver before the tool
  starts, required to cover the row engine (``coverage_problems``) and checked after the run against the final positions of the pool. A racecheck run that cannot analyse the kernel FAILS here
  (there is no skip). initcheck is not a gate and is not run by the suite. None of the tools is a race detector for the protocol of the global ``pos`` word between CTAs (that is what the
  protocol model, the source pins and the delay variants are for).
* The controls (a variant build, ``CFBI_ROWS_TEST_POS_OOB``, with the caching allocator off so that the pool's ``pos`` is its own allocation) move chosen accesses to ``pos[slot + S + 2^24]``
  (``rows_test_pos_offset(offset, accesses)``): on the single launch the ARRIVAL ATOMIC alone is reported as an invalid atomic and the ELECTED STORE alone as an invalid write, both inside
  ``rows_kernel<true>`` (one report need not hold both: an invalid atomic returns no valid count, so the store may not run); on the two launches the fused store is reported as an invalid write
  inside ``rows_kernel<false>``. The same build with offset 0 is clean, and so is the build whose offset moves only accesses that the path does not execute (the mask isolates the access).

A missing compute-sanitizer fails on B200 / B300 (``helpers.skip_or_fail``), a variant that does not build fails (``rows_workload.build_variant``).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import rows_workload as rw
import sanitizer_workloads as sw
from helpers import skip_or_fail

TIMEOUT_S = 1800
REPO = Path(__file__).resolve().parents[2]
TESTS = REPO / "tests" / "cfbi"
OOB = ("CFBI_ROWS_TEST_POS_OOB=1",)
PATHS = ("single", "two_launch")
BOUNDED_H, BOUNDED_POOL = 1, 8
BOUNDED_LAUNCHES = [([1, 66, 3, 5], [3, 7, -1, 0]),     # a 66-row sequence on the live slot 7 (positions 0 .. 63 and two rows after the chunk end), a padding entry, permuted slots
                    ([1, 1, 1], [7, 3, 0])]              # a decode launch on the state that the first launch left (slot 7 at position 2)
OOB_OFFSET_BEYOND = sw.OOB_SLOTS + (1 << 24)
ACC_FUSED, ACC_ARRIVAL, ACC_ELECTED = 1, 2, 4            # bits of the access mask of rows_test_pos_offset
DONE = "ROWS_SANITIZER_WORKLOAD_DONE"

PRELUDE = textwrap.dedent('''
    import random, sys
    sys.path.insert(0, "{tests}")
    sys.path.insert(0, "{repo}")
    import torch
    import helpers
    import rows_workload as rw
    import sanitizer_workloads as sw
    from flash_kda.cfbi import alloc_state, load_ext
    from rows_workload import Family, Step
''')

WAVES_DRIVER = PRELUDE + textwrap.dedent('''
    import json
    ext = load_ext()
    path = rw.PATHS["{path}"]
    ext.decode_single_force(path is rw.SINGLE)      # the single launch is what is under test on that path (the control is process-wide); bit 64 asks for the two launches
    print("ROWS_SANITIZER_PATH", path.name, "impl", path.impl, "force", path is rw.SINGLE, flush=True)
    digest = sw.waves_workload(ext, path)
    print("ROWS_SANITIZER_DIGEST", json.dumps(digest, sort_keys=True), flush=True)
    print("{done} live_ctas", min(l["live_ctas"] for l in digest["launches"]))
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
    path = rw.PATHS["{path}"]
    ext.decode_single_force(path is rw.SINGLE)
    print("ROWS_SANITIZER_PATH", path.name, "impl", path.impl, "force", path is rw.SINGLE, flush=True)
    final, digest = sw.bounded_workload(ext, path, launches, H, S)
    print("ROWS_SANITIZER_DIGEST", json.dumps(digest, sort_keys=True), flush=True)
    print("ROWS_SANITIZER_FINAL_POS", json.dumps(final), flush=True)                    # the positions that the rows really left behind
    expected = [cov["final_pos"].get(s, 0) for s in range(S)]
    assert final == expected, "the launches left the positions %s, the plan says %s: the long sequence did not run" % (final, expected)
    print("{done} bounded")
''')

OOB_DRIVER = PRELUDE + textwrap.dedent('''
    import json
    mod = rw.build_variant("rowsoob", ("CFBI_ROWS_TEST_POS_OOB=1",))
    path = rw.PATHS["{path}"]
    mod.decode_single_force(path is rw.SINGLE)      # a variant is a module of its own, with its own switch
    print("ROWS_SANITIZER_PATH", path.name, "impl", path.impl, "force", path is rw.SINGLE, flush=True)
    digest = sw.oob_workload(mod, path, {offset}, {accesses})
    print("ROWS_SANITIZER_DIGEST", json.dumps(digest, sort_keys=True), flush=True)
    print("{done} offset", {offset}, "accesses", {accesses})
''')


def sanitizer():
    exe = shutil.which("compute-sanitizer")
    if exe is None:
        skip_or_fail("compute-sanitizer is not installed")
    return exe


def run_driver(tool, tmp_path, driver, extra=(), env=None, offset=OOB_OFFSET_BEYOND, path="single", accesses=ACC_ARRIVAL | ACC_ELECTED | ACC_FUSED):
    exe = sanitizer()
    script = tmp_path / "driver.py"
    script.write_text(driver.format(tests=str(TESTS), repo=str(REPO), done=DONE, offset=offset, accesses=accesses, path=path, heads=BOUNDED_H, pool=BOUNDED_POOL, launches=repr(BOUNDED_LAUNCHES)))
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


def printed(text, label):
    """The JSON that the driver printed after ``label`` (the last such line)."""
    lines = [ln for ln in text.splitlines() if ln.startswith(label + " ")]
    assert lines, "the driver did not print %s: %s" % (label, text[-1500:])
    return json.loads(lines[-1][len(label) + 1:])


def on_path(mod, path_name, fn):
    """Run ``fn()`` with the single-launch control of ``mod`` as the path needs it (on for the single launch, off for the two launches), and put it back."""
    previous = mod.decode_single_force(path_name == "single")
    try:
        return fn()
    finally:
        mod.decode_single_force(previous)


def assert_path_named(text, path_name):
    path = rw.PATHS[path_name]
    assert "ROWS_SANITIZER_PATH %s impl %d force %s" % (path.name, path.impl, path_name == "single") in text, text[-1500:]


# ---- memcheck / synccheck over decode beyond two waves with padding slots, on both paths ------------------------------------------------------------------------------
@pytest.mark.parametrize("tool", ["memcheck", "synccheck"])
@pytest.mark.parametrize("path_name", PATHS)
def test_sanitizer_reports_no_error_over_decode_beyond_two_waves_with_padding_slots(ext, tool, path_name, tmp_path):
    path = rw.PATHS[path_name]
    expected = json.loads(json.dumps(on_path(ext, path_name, lambda: sw.waves_workload(ext, path, assert_launches=True)), sort_keys=True))     # without the tool: every launch is the path's
    assert min(l["live_ctas"] for l in expected["launches"]) >= 2.5 * expected["resident_ctas"]
    r = run_driver(tool, tmp_path, WAVES_DRIVER, path=path_name)
    text = r.stdout + r.stderr
    assert DONE in text, text[-3000:]
    assert_path_named(text, path_name)
    assert printed(text, "ROWS_SANITIZER_DIGEST") == expected, "the tool ran another workload than the one whose launches were asserted"
    assert r.returncode == 0 and "ERROR SUMMARY: 0 errors" in text, text[-3000:]


# ---- racecheck over a bounded workload, on both paths -----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("path_name", PATHS)
def test_racecheck_reports_no_hazard_over_a_bounded_workload_that_crosses_a_chunk_end(ext, path_name, tmp_path):
    path = rw.PATHS[path_name]
    plan = json.loads(json.dumps(rw.launch_coverage(BOUNDED_LAUNCHES, BOUNDED_H), sort_keys=True))
    final_pos, expected = on_path(ext, path_name, lambda: sw.bounded_workload(ext, path, BOUNDED_LAUNCHES, BOUNDED_H, BOUNDED_POOL, assert_launches=True))
    assert final_pos == [plan["final_pos"].get(str(s), 0) for s in range(BOUNDED_POOL)], "the workload (without the tool) did not leave the positions of its plan"
    t0 = time.time()
    r = run_driver("racecheck", tmp_path, BOUNDED_DRIVER, extra=("--racecheck-report", "hazard"), path=path_name)
    seconds = time.time() - t0
    text = r.stdout + r.stderr
    assert DONE in text, "racecheck could not run the workload: " + text[-3000:]
    assert_path_named(text, path_name)
    cov = printed(text, "ROWS_SANITIZER_COVERAGE")
    assert cov == plan, "the driver executed another workload than the plan of the test"
    assert rw.coverage_problems(cov) == [], rw.coverage_problems(cov)
    assert printed(text, "ROWS_SANITIZER_DIGEST") == json.loads(json.dumps(expected, sort_keys=True))
    assert printed(text, "ROWS_SANITIZER_FINAL_POS") == final_pos
    assert r.returncode == 0 and "RACECHECK SUMMARY: 0 hazards" in text, text[-3000:]
    print("\nbounded racecheck on the %s path: %d rows at all %d positions (%d chunk-end rows, %d after a chunk end, %d beyond the first sub-block) in %d CTAs of the first launch, %.1f s, 0 hazards" % (
        path_name, cov["rows"], len(cov["positions"]), cov["chunk_end_rows"], cov["rows_after_chunk_end"], cov["cross_subblock_rows"], cov["live_ctas"][0], seconds))


# ---- the controls: an out-of-bounds position access is reported, the one that was moved and no other ---------------------------------------------------------------------
OOB_ENV = {"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"}      # every tensor is its own allocation: an offset beyond the pool's pos leaves it
OOB_CTAS = sw.OOB_SEQUENCES * sw.OOB_HEADS
# (id, path, accesses moved, what memcheck must report first, the number of errors it must report (None: not asserted))
OOB_REPORTED = [
    ("single_arrival_atomic", "single", ACC_ARRIVAL, "Invalid __global__ atomic", OOB_CTAS),           # every CTA of the launch arrives once
    ("single_elected_store", "single", ACC_ELECTED, "Invalid __global__ write", sw.OOB_SEQUENCES),       # exactly one CTA per sequence stores
    ("single_both", "single", ACC_ARRIVAL | ACC_ELECTED, "Invalid __global__ atomic", None),            # an invalid atomic returns no valid count: whether the store then runs is not asserted
    ("two_launch_fused_store", "two_launch", ACC_FUSED, "Invalid __global__ write", OOB_CTAS),          # every head CTA stores the fused position
]
# the build with an offset, but with an access mask that selects nothing that the path executes
OOB_ISOLATED = [("single_with_only_the_fused_store_moved", "single", ACC_FUSED), ("two_launch_with_only_the_single_launch_accesses_moved", "two_launch", ACC_ARRIVAL | ACC_ELECTED)]


def memcheck_oob(ext, tmp_path, path_name, offset, accesses):
    """Run the control driver under memcheck after the same variant run once without the tool (offset 0, nothing moved) to assert the path's launches; returns the text of the tool."""
    mod = rw.build_variant("rowsoob", OOB)                       # built here, outside the tool; the driver loads the cached build
    path = rw.PATHS[path_name]
    on_path(mod, path_name, lambda: sw.oob_workload(mod, path, 0, 0, assert_launches=True))
    r = run_driver("memcheck", tmp_path, OOB_DRIVER, env=OOB_ENV, offset=offset, path=path_name, accesses=accesses)
    return r, r.stdout + r.stderr


def kernel_lines(text):
    return [ln.strip() for ln in text.splitlines() if "rows_kernel" in ln]


def kernel_instances(text):
    """The instantiations of the row engine that the report names: ``1`` is ``rows_kernel<true>`` (the single launch), ``0`` is ``rows_kernel<false>`` (behind the plan kernel). The parameter type in the
    demangled name is ``std::conditional<..., DecodeParams, Params>::type`` for both, so the template argument is what tells them apart."""
    return sorted(set(re.findall(r"rows_kernel<\(bool\)(\d)>", text)))


@pytest.mark.parametrize("case,path_name,accesses,report,count", OOB_REPORTED, ids=[c[0] for c in OOB_REPORTED])
def test_memcheck_reports_the_position_access_that_was_moved_beyond_the_pool(ext, tmp_path, case, path_name, accesses, report, count):
    r, text = memcheck_oob(ext, tmp_path, path_name, OOB_OFFSET_BEYOND, accesses)
    assert r.returncode == 99, "memcheck did not report the out-of-bounds access (exit %d): %s" % (r.returncode, text[-2000:])
    assert_path_named(text, path_name)
    assert report in text, text[-3000:]
    lines = kernel_lines(text)
    assert lines, "the report names no row-engine kernel: " + text[-2000:]
    assert kernel_instances(text) == (["1"] if path_name == "single" else ["0"]), "the report is about rows_kernel<%s>, the path %s launches rows_kernel<%s>: %s" % (
        "/".join(kernel_instances(text)), path_name, "true" if path_name == "single" else "false", lines[:2])
    errors = [ln.strip() for ln in text.splitlines() if "Invalid __global__" in ln]
    print("\n%s: %d error(s), first: %s | %s" % (case, len(errors), errors[0], lines[0]))
    assert report in errors[0], errors[:2]
    if count is not None:
        assert len(errors) == count, "memcheck reported %d invalid accesses, the launch makes %d of this kind: %s" % (len(errors), count, errors[:3])


@pytest.mark.parametrize("path_name", PATHS)
def test_the_same_variant_with_offset_zero_is_clean_under_memcheck(ext, tmp_path, path_name):
    r, text = memcheck_oob(ext, tmp_path, path_name, 0, ACC_ARRIVAL | ACC_ELECTED | ACC_FUSED)
    assert DONE in text and r.returncode == 0 and "ERROR SUMMARY: 0 errors" in text, text[-3000:]
    assert_path_named(text, path_name)


@pytest.mark.parametrize("case,path_name,accesses", OOB_ISOLATED, ids=[c[0] for c in OOB_ISOLATED])
def test_an_offset_that_moves_only_accesses_the_path_does_not_make_is_clean(ext, tmp_path, case, path_name, accesses):
    """The report comes from the access that was moved, not from the build or from the offset being set: with the same offset and a mask of accesses that this path never executes nothing is reported."""
    r, text = memcheck_oob(ext, tmp_path, path_name, OOB_OFFSET_BEYOND, accesses)
    assert DONE in text and r.returncode == 0 and "ERROR SUMMARY: 0 errors" in text, text[-3000:]
