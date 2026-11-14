"""The negative controls of the staged state tile of the row engine, on a device (B200, B300): a tile that is WRONG must make the comparison with the frozen reference fail.

The tile (``csrc/cfbi/cfbi_rows.cuh``: every warp copies the 16 value rows of E that it owns into dynamic shared memory once per call, waits for its copies in front of the barrier that ends phase C
and reads its fragments from the tile) is a verbatim copy of what the row engine used to read from global memory, so the end-to-end gate stays the bit comparison with ``csrc/cfbi/ref`` (Gate G,
Suite S, the decode families). A comparison that has never failed proves nothing, so two builds of the row engine with a deliberately wrong tile are run on the decode families of the single launch and
of the two-launch path and must differ from the oracle:

* ``CFBI_ROWS_NEG_E_ROW``: the rows of E are staged shifted by one (the tile holds another row than its index says);
* ``CFBI_ROWS_NEG_E_LATE``: the copies are issued right in front of their first read and never waited for (the read cannot see them: no ordering of results can show a missing wait when the copy is
  issued at the top of the row, because it has always landed by phase D, so the control moves the issue).

The ordering of the real kernel (the one wait in front of the barrier, after the staging, in front of the first read) is a source rule with mutants (``test_rows_e_tile.py``), the occupancy of the
tile's launch (two CTAs per SM with 67,584 B of dynamic shared memory) is read from the occupancy API (``test_rows_occupancy.py``), and an over-budget tile does not build (``test_rows_e_tile.py``).

Every group that launches kernels runs in a child process under ``GROUP_TIMEOUT_S`` (``rows_bounded``): a hang fails the test and kills the child; a variant that does not build fails its test
(``rows_workload.build_variant``), it never skips it.
"""
import pytest

import rows_bounded as rb
import rows_workload as rw
from helpers import skip_or_fail
from rows_workload import Family

GROUP_TIMEOUT_S = 1200      # two JIT builds of a variant (a few minutes each on a first run) and eight launches of a family
CONTROLS = {"shifted_row": ("CFBI_ROWS_NEG_E_ROW=1",), "late_issue": ("CFBI_ROWS_NEG_E_LATE=1",)}
# The sequences must start where E matters: the cumulative decay of the gates (the contract's regime) takes the E terms of a row (GQ = Gamma * q, W) and the old E of a chunk end (GEND) below the
# range of fp32 at the late positions of a chunk, so a wrong tile there changes no bit of any result (the first version of this module started every sequence of its multi-row family at a
# position of 57 .. 63 and both controls matched the reference on all of them). The fills of the mixed kind include the early positions.
FAMILIES = [Family("e_decode", fixed=1, pool=True, pad=0.3, fills="mixed", steps=1),                  # one row per sequence: the production decode launch
            Family("e_multirow", fixed=5, pool=True, pad=0.2, fills="mixed", steps=1)]                # five rows per sequence: the tile is read again by every row after the first
PATH_NAMES = ["single", "two_launch"]
HEADS = 12


@pytest.fixture(autouse=True)
def _optimised_kernels_run_here(ext):
    """Below sm_90 ``_impl=0`` resolves to the reference kernels: the comparison would be a build with itself."""
    if ext.resolve_impl(rw.OPTIMISED, -1) == ext.resolve_impl(rw.REFERENCE, -1):
        skip_or_fail("the optimised kernels do not run here (_impl=0 resolves to the reference kernels)")


# ---- the cases (they run in the child process of their group) -------------------------------------------------------------------------------------------
def control_case(ext, control, fam, path_name):
    """The clean build against the oracle on a warm state (asserted equal by ``run_family``), then the control build from the same warm state and step: it must differ."""
    path = rw.PATHS[path_name]
    bad = rw.build_variant("rowse_" + control, CONTROLS[control])
    bad.decode_single_force(True)
    results = rw.run_family(ext, fam, HEADS, seed=41, path=path)
    step, opt, ref, warm = results[0]
    assert all(v == 0 for v in rw.mismatch(opt, ref).values())
    wrong = rw.run_arm(bad, path, warm, step)
    m = rw.mismatch(wrong, ref)
    assert m["out32"] > 0 and m["out"] > 0, "the %s build matches the reference on %s (%s): the workload cannot see a wrong state tile (%s)" % (control, fam, path_name, m)


# ---- the group (one child process) ------------------------------------------------------------------------------------------------------------------------
def group_controls(ext):
    ext.decode_single_force(True)
    cases = {}
    for control in sorted(CONTROLS):
        for fam in FAMILIES:
            for path in PATH_NAMES:
                cases["%s:%s:%s" % (control, repr(fam), path)] = rb.capture(control_case, ext, control, fam, path)
    return {"cases": cases, "reps": 1}


@pytest.fixture(scope="module")
def control_group(ext):
    return rb.run_group("test_rows_e_controls", "group_controls", {}, GROUP_TIMEOUT_S)


@pytest.mark.parametrize("path", PATH_NAMES)
@pytest.mark.parametrize("fam", FAMILIES, ids=repr)
@pytest.mark.parametrize("control", sorted(CONTROLS))
def test_a_wrong_state_tile_differs_from_the_reference(control_group, control, fam, path):
    outcome = control_group["cases"]["%s:%s:%s" % (control, repr(fam), path)]
    assert outcome is None, outcome
