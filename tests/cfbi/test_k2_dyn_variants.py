"""Compile-time test variants of the device-scheduled K2 (impl bit 16): the harness can fail, and the result does not depend on the order or timing of the CTAs.

Variants are separate JIT builds of the extension with test macros (``load_variant``); the default build contains none of them.

* Mutation variants must be CAUGHT by the same bit comparison that the other tests use (a nonzero difference count against the frozen reference): an item
  that is skipped, an item that is replaced by its neighbour, a CTA that reads the wrong 16-column slice of the K1 output, and a missing range check for
  the CTAs beyond the item count while the table beyond the count points at a valid item. The outputs are pre-filled with a sentinel, so a skipped item
  is a certain mismatch.
* Stress variants must leave every bit identical to the reference: the CTAs take the items in a permuted order (seeds 1, 2, 3; the items keep their
  serial chunk order inside the CTA) and linger for per-CTA delays of 2000 and 20000 ns; every case runs five times. A variant whose permutation is not a
  bijection (negative control of the hook itself) must be caught.
"""
import itertools
import re
from pathlib import Path

import pytest
import torch

from flash_kda.cfbi import alloc_state, state_mismatch

from helpers import make_case, n_diff
from varlen_corpus import CORPUS_V

REF, DYN = 3, 16
LENS = [900, 300, 700, 1300, 5, 199, 1000, 450, 640, 77]
H = 6
REPEATS = 5


@pytest.fixture(autouse=True)
def _dyn_runs_here(ext):
    if not ext.resolve_impl(DYN, -1) & DYN:
        pytest.skip("the device cannot run the device-scheduled K2 (impl bit 16 is dropped here)")


def build_variant(name, defines):
    from flash_kda.cfbi._ext import load_variant

    try:
        return load_variant(name, tuple(defines))
    except Exception as e:   # pragma: no cover - the variants need a JIT build
        pytest.skip(f"cannot build the variant {name}: {e}")


def run(mod, impl, lens, heads, k2_bv=0, seed=3):
    """One prefill of a fresh state through ``mod.forward`` (outputs pre-filled with a sentinel); returns (bf16 out, fp32 out, state)."""
    case = make_case(sum(lens), heads, "authors", seed=seed)
    cu = torch.tensor([0] + list(itertools.accumulate(lens)), dtype=torch.int64, device="cuda")
    st = alloc_state(len(lens), heads)
    out = torch.full_like(case["q"], 3.0)
    o32 = torch.full(case["q"].shape, 3.0, dtype=torch.float32, device="cuda")
    mod.forward(case["q"], case["k"], case["v"], case["g"], case["beta"], float(case["scale"]), out, case["A_log"], case["dt_bias"], float(case["lower_bound"]),
                st.E, st.rows, st.pos, cu, 0, o32, impl, None, None, None, 3, k2_bv)
    torch.cuda.synchronize()
    return out, o32, st


def differences(a, b):
    """Number of differing elements of the bf16 outputs, the fp32 outputs and the state (E, valid rows, pos), against a reference run."""
    sm = state_mismatch(a[2], b[2])
    return {"out": n_diff(a[0], b[0]), "out32": n_diff(a[1], b[1]), "E": sm["E"], "rows": sm["rows"], "pos": sm["pos"]}


def total(d):
    return sum(d.values())


# ---- the default build has no test macro --------------------------------------------------------------------------------------------------------------
def test_the_default_build_contains_no_test_macro():
    from flash_kda.cfbi import _ext

    assert not any("TEST" in f for f in _ext.NVCC_FLAGS)
    root = Path(__file__).resolve().parents[2] / "csrc" / "cfbi"
    for f in root.rglob("*"):
        if f.suffix in (".cu", ".cuh", ".h", ".cpp"):
            for ln, line in enumerate(f.read_text().splitlines(), 1):
                assert not re.match(r"\s*#\s*define\s+CFBI_(DYN|K2)_TEST", line), f"{f.name}:{ln}: a test macro must only come from a -D flag"


def test_the_unmodified_build_equals_the_reference_on_the_variant_workload(ext):
    ref = run(ext, REF, LENS, H)
    assert total(differences(ref, run(ext, DYN, LENS, H))) == 0


# ---- mutation variants are caught -----------------------------------------------------------------------------------------------------------------
MUTATIONS = [
    ("dyn_skip", ("CFBI_DYN_TEST_SKIP_ITEM=0",)),                                    # the longest item is never run
    ("dyn_dup", ("CFBI_DYN_TEST_DUP_ITEM=2",)),                                      # item 2 is replaced by item 3
    ("dyn_wrong_ut", ("CFBI_K2_TEST_WRONG_UT_SLICE=1",)),                            # every CTA reads the next 16-column slice
    ("dyn_no_range_check", ("CFBI_DYN_TEST_NO_RANGE_CHECK=1", "CFBI_DYN_TEST_PAD_ITEM_ZERO=1", "CFBI_DYN_TEST_EXTRA_DELAY_MS=5")),
]


@pytest.mark.parametrize("name,defines", MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_a_mutated_variant_is_caught_by_the_comparison(ext, name, defines):
    ref = run(ext, REF, LENS, H)
    assert total(differences(ref, run(ext, DYN, LENS, H))) == 0, "the unmutated build must match, or the comparison proves nothing"
    bad = differences(ref, run(build_variant(name, defines), DYN, LENS, H))
    assert total(bad) > 0, f"the harness missed the variant {name}"
    if name in ("dyn_skip", "dyn_dup"):
        assert bad["out"] > 0 and bad["E"] > 0, bad          # a lost item leaves its sentinel outputs and its state slice untouched


def test_the_range_check_variant_really_launches_ctas_beyond_the_item_count(ext):
    """The missing-range-check negative control is only meaningful if the grid has CTAs beyond the count: the device policy chooses wide tiles here."""
    from test_k2_sched import probe

    d = probe(ext, LENS, H)
    assert d.count < 16 * len(LENS) * H and (d.pad == -1).all()


# ---- stress variants stay identical -----------------------------------------------------------------------------------------------------------------
STRESS = [
    ("dyn_perm1_d2000", ("CFBI_DYN_TEST_PERMUTE_SEED=1", "CFBI_DYN_TEST_DELAY_NS=2000")),
    ("dyn_perm2_d20000", ("CFBI_DYN_TEST_PERMUTE_SEED=2", "CFBI_DYN_TEST_DELAY_NS=20000")),
    ("dyn_perm3_d2000", ("CFBI_DYN_TEST_PERMUTE_SEED=3", "CFBI_DYN_TEST_DELAY_NS=2000")),
]
CASES = [("V2", CORPUS_V["V2"].lens, 6, 0), ("V4", CORPUS_V["V4"].lens, 6, 0), ("V7b", CORPUS_V["V7b"].lens, 6, 0), ("edge", [150, 1, 64, 129, 33, 0, 70, 700, 65, 63], 3, 0),
         ("V4 8", CORPUS_V["V4"].lens, 6, 8), ("V4 32", CORPUS_V["V4"].lens, 6, 32)]


@pytest.mark.parametrize("name,defines", STRESS, ids=[s[0] for s in STRESS])
def test_permuted_item_order_and_delays_leave_every_bit_identical(ext, name, defines):
    mod = build_variant(name, defines)
    for case_name, lens, heads, bv in CASES:
        ref = run(ext, REF, lens, heads, bv)
        for rep in range(REPEATS):
            d = differences(ref, run(mod, DYN, lens, heads, bv, seed=3))
            assert total(d) == 0, (name, case_name, rep, d)


def test_a_permutation_that_is_not_a_bijection_is_caught(ext):
    """Negative control of the permutation hook: the same hook with a broken map must produce wrong bits (otherwise the hook is not live)."""
    mod = build_variant("dyn_perm_break", ("CFBI_DYN_TEST_PERMUTE_SEED=1", "CFBI_DYN_TEST_PERMUTE_BREAK=1"))
    ref = run(ext, REF, LENS, H)
    assert total(differences(ref, run(mod, DYN, LENS, H))) > 0
