"""The shared-memory layout of the row engine with its staged A fragments, as the compiler sees it (host compile of a unit that includes ``csrc/cfbi/cfbi_rows.cuh``: its ``static_assert``s are
evaluated, no kernel is instantiated; plain Python, nvcc on the login node, no GPU).

Two layouts exist. The production build keeps ``RowSmem`` without the two buffers that only fed the phase-D fragments (6660 B) and the zero-first ``FragSmem`` (7808 B); the test build that
carries the fragment-level check (``CFBI_ROWS_TEST_AFRAG_CHECK``) keeps those two buffers again, because it recomputes the unstaged fragments from them (7684 B). Each build has its own need in
the policy header and its own asserts (the layout is covered by the constant, and the constant is less than one allocation unit above the layout): the production constant is never loosened
for the test build, and the test build's figure is never taken for production evidence.

* both layouts compile, and their sizes and offsets are the numbers of the lane model (``afrag_model.py``), so the model and the structure cannot drift apart;
* for each layout a constant one byte below it, and one more than an allocation unit above it, fail the build with the assert that names the rule;
* a structure with another zero-page phase, another alignment, a zero page smaller than a tile or a moved tile fails the build.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import afrag_model as am

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))

import codegen_gate as cg  # noqa: E402

CSRC = ROOT / "csrc" / "cfbi"
ORACLE = "CFBI_ROWS_TEST_AFRAG_CHECK"
NEED = {False: 15360, True: 16384}                   # the static shared memory that each build asks of the device (Blackwell allocation unit 1 KiB), the policy header's constants
RAW = {False: am.STATIC_F, True: am.STATIC_F + 1024}  # the raw layout of each build: RowSmem rounded to 128 + FragSmem + the int of the decode kernel (the oracle build keeps 1024 B more of RowSmem)


@pytest.fixture(scope="module")
def tc():
    try:
        return cg.find_toolchain()
    except cg.Unusable as e:
        pytest.skip("no toolchain for the compile-only checks on this machine: %s" % e)


def make_tree(tmp, edits=()):
    """A copy of the sources under ``csrc/cfbi`` with each ``(file, old, new)`` applied once."""
    tree = Path(tempfile.mkdtemp(dir=str(tmp)))
    shutil.copytree(CSRC, tree / "csrc" / "cfbi")
    for name, old, new in edits:
        target = tree / "csrc" / "cfbi" / name
        text = target.read_text()
        assert text.count(old) == 1, (name, old, text.count(old))
        target.write_text(text.replace(old, new))
    return tree


def compile_unit(tc, tree, oracle=False, extra=""):
    """``(exit status, text)`` of the host compile of the unit that includes the row engine (and ``extra``) for sm_90a, with the test macro of the oracle build when asked."""
    (tree / "check_rows.cu").write_text('#include "cfbi/cfbi_rows.cuh"\n' + extra)
    cmd = cg.compile_command(tc, "check_rows.cu", ("sm_90a",), (ORACLE,) if oracle else (), str(tree / "check_rows.o"))
    r = subprocess.run(cmd, cwd=str(tree), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    return r.returncode, r.stdout


def model_asserts(oracle):
    """The layout of the lane model as asserts about the compiled structures."""
    rowsmem = am.ROWSMEM_T if oracle else am.ROWSMEM_F
    return "".join('static_assert(%s, "the layout of the lane model: %s");\n' % (cond, why) for cond, why in (
        ("sizeof(cfbi::RowSmem) == %d" % rowsmem, "RowSmem"), ("sizeof(cfbi::FragSmem) == %d" % am.FRAG_SIZE, "FragSmem size"), ("alignof(cfbi::FragSmem) == 128", "FragSmem alignment"),
        ("offsetof(cfbi::FragSmem, zero) == %d" % am.ZERO_OFF, "zero page"), ("offsetof(cfbi::FragSmem, ab) == %d" % am.AB_OFF, "AB tile"),
        ("offsetof(cfbi::FragSmem, am) == %d" % am.AM_OFF, "AM tile"), ("offsetof(cfbi::FragSmem, ad) == %d" % am.AD_OFF, "AD tile"),
        ("sizeof(cfbi::FragSmem::ab) == %d" % (4 * am.AB_WORDS), "AB size"), ("sizeof(cfbi::FragSmem::am) == %d" % (4 * am.AM_WORDS), "AM size"),
        ("sizeof(cfbi::FragSmem::ad) == %d" % (4 * am.AD_WORDS), "AD size"), ("sizeof(cfbi::FragSmem::zero) == %d" % (4 * am.ZERO_WORDS), "zero size"),
        ("cfbi::ROWS_BUILD_STATIC_SMEM_BYTES == %d" % NEED[oracle], "the need of the build"), ("cfbi::ROWS_SMEM_NEED_BYTES == %d" % NEED[oracle], "the need that the launcher reads")))


def test_the_raw_layouts_are_what_the_needs_are_judged_against():
    assert RAW == {False: 14596, True: 15620}
    for oracle in (False, True):
        assert RAW[oracle] <= NEED[oracle] and NEED[oracle] - RAW[oracle] < 1024                    # covered, and less than one allocation unit above (the margin rule of the sources)
        assert NEED[oracle] == -(-RAW[oracle] // 1024) * 1024                                        # the need is the layout rounded up to the allocation unit


@pytest.mark.parametrize("oracle", [False, True], ids=["production", "oracle"])
def test_each_layout_compiles_and_is_the_one_of_the_lane_model(tmp_path, tc, oracle):
    status, text = compile_unit(tc, make_tree(tmp_path), oracle, model_asserts(oracle))
    assert status == 0, text[-2500:]


@pytest.mark.parametrize("oracle,constant,value,expect", [
    (False, "ROWS_STATIC_SMEM_BYTES = 15360;", RAW[False] - 1, "must cover the static shared memory of the row engine"),
    (False, "ROWS_STATIC_SMEM_BYTES = 15360;", RAW[False] + 1024, "is more than an allocation unit above the layout"),
    (True, "ROWS_ORACLE_STATIC_SMEM_BYTES = 16384;", RAW[True] - 1, "must cover the static shared memory of the row engine"),
    (True, "ROWS_ORACLE_STATIC_SMEM_BYTES = 16384;", RAW[True] + 1024, "is more than an allocation unit above the layout"),
], ids=["production-below", "production-above", "oracle-below", "oracle-above"])
def test_a_need_that_does_not_match_its_layout_fails_the_build_of_that_layout(tmp_path, tc, oracle, constant, value, expect):
    name, number = constant.split(" = ")
    tree = make_tree(tmp_path, [("cfbi_policy.h", constant, "%s = %d;" % (name, value))])
    status, text = compile_unit(tc, tree, oracle)
    assert status != 0 and expect in text, (constant, value, text[-1500:])


def test_the_need_of_one_build_does_not_cover_the_layout_of_the_other(tmp_path, tc):
    """The production constant does not cover the oracle layout, so the oracle build can only be built with its own need (and the production build is never loosened for it)."""
    tree = make_tree(tmp_path, [("cfbi_policy.h", "ROWS_ORACLE_STATIC_SMEM_BYTES = 16384;", "ROWS_ORACLE_STATIC_SMEM_BYTES = 15360;")])
    status, text = compile_unit(tc, tree, True)
    assert status != 0 and "must cover the static shared memory of the row engine" in text, text[-1500:]
    status, text = compile_unit(tc, make_tree(tmp_path, [("cfbi_policy.h", "ROWS_ORACLE_STATIC_SMEM_BYTES = 16384;", "ROWS_ORACLE_STATIC_SMEM_BYTES = 15360;")]), False)
    assert status == 0, text[-1500:]                                                                  # the production build does not read the oracle constant


@pytest.mark.parametrize("name,old,new,expect", [
    ("a zero page at another phase", "uint4 pad0_[4];", "uint4 pad0_[3];", "the zero twin of a 64-byte group lies in the other half of the banks"),
    ("a weaker alignment", "struct alignas(128) FragSmem {", "struct alignas(64) FragSmem {", "FragSmem is aligned to 128 bytes"),
    ("a zero page smaller than a tile", "uint4 zero[16][2][4];", "uint4 zero[8][2][4];", "one immediate offset serves every lane"),
    ("a tile that moved", "uint4 am[6][2][2][4];", "uint4 am[5][2][2][4];", "the layout of the lane model"),
    ("a pad that unaligns the tiles", "uint4 pad1_[4];", "uint4 pad1_[5];", "the staged tiles are 128-byte aligned"),
])
def test_a_structure_that_breaks_the_layout_rules_fails_the_build(tmp_path, tc, name, old, new, expect):
    status, text = compile_unit(tc, make_tree(tmp_path, [("cfbi_rows.cuh", old, new)]), False, model_asserts(False))
    assert status != 0 and expect in text, (name, text[-1800:])
