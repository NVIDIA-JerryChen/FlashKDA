"""The lane / offset model of the staged A fragments (``afrag_model.py``; plain Python and numpy, no GPU).

The row engine builds the three families of TF32 hi / lo A fragments (the ks / qs rows of phase B, the akk / aqk rows of the cross-sub-block products, the W / GQ rows of phase D) once per row
into the shared tiles of ``FragSmem``; the model executes the unstaged kernel's own expressions and the staging protocol (producers, persistent poisoned tiles, consumers that read the tile or
the zero page) and compares every register word as uint32. What these tests pin:

* the mapping: every word that a consumer loads equals the word that the unstaged kernel builds, for every position 0..63, the wrap at the chunk end and special values (signed zeros, NaN,
  infinities); every staged word is written once per row by one thread, every read is ordered after its write by the barrier plan, no tile is rewritten before the row before it has read it;
* the geometry: tile offsets, the zero page, the 16-byte and 128-byte alignment, the bank wavefronts of the loads (the minimum, 4 per warp) and the bound of the producers' stores;
* the controls: each of the 19 deliberate defects is rejected, and the model says which of them an output comparison could never see (a word that differs only in a row that no output reads),
  because those need a fragment-level check on the device.

The layout numbers of the model are the ones that ``test_rows_afrag_layout.py`` asserts of the compiled structure, so the two cannot drift apart.
"""
import numpy as np
import pytest

import afrag_model as am

CONTROLS = [(mut, fam) for mut, fams in am.MUTATIONS.items() for fam in fams]
CLASSES = {("shift_lane", "AB"): "visible", ("shift_lane", "AM"): "visible", ("shift_lane", "AD"): "visible",
           ("wrong_parity", "AB"): "visible", ("wrong_parity", "AM"): "visible", ("wrong_parity", "AD"): "visible",
           ("missing_word", "AB"): "visible", ("missing_word", "AM"): "visible", ("missing_word", "AD"): "visible",
           ("double_writer", "AB"): "log-only", ("double_writer", "AM"): "log-only", ("double_writer", "AD"): "log-only",
           ("missing_barrier", "AB"): "log-only", ("missing_barrier", "AM"): "log-only", ("missing_barrier", "AD"): "log-only",
           ("stale_zero", "AM"): "invisible",
           ("nonmine_tile", "AB"): "invisible", ("nonmine_tile", "AM"): "invisible", ("nonmine_tile", "AD"): "invisible"}


def run_control(mut, fam, seed=17):
    stats = {}
    problems = am.run_sequence(am.CONTROL_ROWS, np.random.default_rng(seed), special_every=99, mut=(mut, fam), stats=stats)
    return problems, stats


def test_the_staged_words_equal_the_unstaged_words_for_every_position_and_the_chunk_wrap():
    problems = []
    for seed in (16, 99):                                                    # two random states, special values in every other row; 64 positions, the wrap, 16 more
        problems += am.run_sequence(list(range(64)) + list(range(16)), np.random.default_rng(seed))
    assert problems == [], problems[:3]


def test_the_geometry_and_the_reuse_of_the_tiles_hold():
    assert am.alignment_problems() == []
    assert am.check_reuse() == []


def test_the_layout_numbers_of_the_model():
    assert (am.ZERO_OFF, am.AB_OFF, am.AM_OFF, am.AD_OFF, am.END_OFF, am.FRAG_SIZE) == (64, 2176, 4224, 5760, 7808, 7808)
    assert (am.ZERO_OFF + 4 * am.ZERO_WORDS, am.ARENA) == (2112, (2176, 7808))
    assert (am.ROWSMEM_F, am.STATIC_F) == (6660, 14596)                      # RowSmem without its two dead buffers, and the worst-case static shared memory with the int of the decode kernel
    assert -(-am.STATIC_F // 1024) * 1024 == 15360 and -(-am.STATIC_F // 128) * 128 == 14720


@pytest.mark.parametrize("mut,fam", CONTROLS)
def test_every_deliberate_defect_of_the_staging_is_rejected(mut, fam):
    problems, _ = run_control(mut, fam)
    assert problems, (mut, fam, "the model accepted a defective staging")


@pytest.mark.parametrize("mut,fam", CONTROLS)
def test_the_model_says_which_defects_an_output_comparison_could_see(mut, fam):
    """visible: words differ in the A rows that reach an output; invisible: they differ only in rows that no output reads (the non-mine lanes, the unused parity of a one-row tile), so no comparison of
    the outputs can ever reject the defect and the device needs a check of the fragments themselves; log-only: no word differs, the protocol checks see it (on the device it is a race)."""
    _, stats = run_control(mut, fam)
    all_diff, consumed = stats.get(fam, [0, 0])
    assert ("visible" if consumed else ("invisible" if all_diff else "log-only")) == CLASSES[(mut, fam)], (mut, fam, all_diff, consumed)


def test_a_zero_word_that_is_not_rewritten_is_stale_only_after_the_parity_flips():
    """stale_zero: the data compare passes for the rows before the flip of the row parity (position 24 after 23) and fails from it on; the coverage check rejects it from the first row."""
    rng = np.random.default_rng(3)
    tiles = am.Smem()
    tiles.w["AM"][:] = 0
    before, after = [], []
    for pos in (21, 22, 23):
        stats = {}
        am.run_row(tiles, am.make_state(rng, False), pos, ("stale_zero", "AM"), stats)
        before.append(stats["AM"][0])
    stats = {}
    am.run_row(tiles, am.make_state(rng, False), 24, ("stale_zero", "AM"), stats)
    after.append(stats["AM"][0])
    assert before == [0, 0, 0] and after[0] > 0


def test_the_loads_have_the_minimum_wavefronts_and_the_stores_are_bounded():
    stats = am.bank_stats()
    assert (stats["AB_load"], stats["AM_load"], stats["AD_load"]) == (4, 4, 4)       # a 128-bit load of a warp is four quarter-warp requests: nothing is lost to a bank conflict
    assert stats["AM_store"] == 1 and stats["AB_store"] <= 4 and stats["AD_store"] <= 4
    extra = (stats["AB_store"] - 1) * 4 * 4 + (stats["AD_store"] - 1) * 8 * 2
    assert extra <= 96                                                       # the producers' conflict costs at most this many shared-memory wavefronts per row and CTA


def test_the_model_does_not_compare_a_word_with_itself():
    """The staged side must observe its reads (the count of tile reads is asserted per row) and a poisoned tile that a row does not rewrite is read as the poison word."""
    rng = np.random.default_rng(5)
    problems, stats = run_control("missing_word", "AB")
    assert any("a5a5a5a5" in p for p in problems), problems[:3]
    tiles = am.Smem()
    assert all(int(t[0]) == 0xA5A5A5A5 for t in tiles.w.values())
    am.run_row(tiles, am.make_state(rng, False), 5)
    assert tiles.reads and {r[0] for r in tiles.reads} == {"AB", "AD"}            # position 5: no cross-sub-block rows, so no AM tile is read
