"""Source rules of the test hooks of the row engine (plain Python, no GPU: ``pytest --noconftest tests/cfbi/test_rows_hooks.py``), each with mutants that break it.

The hooks (``ROWS_SYNC``, ``ROWS_SYNC_END``, ``ROWS_DELAY``, ``ROWS_POS_INDEX`` and the host setters; ``cfbi_rows.cuh``, ``cfbi_launch.cu``) let the device tests delay a chosen warp at the
barriers of a row, remove the barrier that ends a row, write the position at an out-of-bounds index and fail a build on purpose. No comparison of results can show that the *default* build
has none of this, nor that a barrier added later has a delay site, so the rules are pinned as text:

* sites: no raw ``__syncthreads`` is left in ``rows_kernel`` (every barrier is a numbered ``ROWS_SYNC`` macro and so has a delay site; the order of the numbers is pinned together with the
  clock-profile stamps in ``test_rows_profile.py``); the only direct delay calls are site 0 (kernel entry, after the profile entry stamp and before the plan load), site 14 (the window of
  barrier 13 of the staged A fragments, in front of the stores of the AB producers: between barrier 1 and ``ROWS_SYNC_AB()``) and site 8 (right after barrier 6, inside the chunk-end phase);
* default build: read with every test macro undefined, the hook macros are what the unhooked kernel contained (``ROWS_SYNC(n)`` is a barrier, ``ROWS_SYNC_END()`` is ``ROWS_SYNC(7)``,
  ``ROWS_DELAY`` and the other hooks are empty, ``ROWS_POS_INDEX(slot)`` is the slot) and none of the machinery of the variants is left; the host setters only return an error;
* macro guard: no source defines a test macro, no build flag of the default build carries one, and every macro that the sources test is classified (a test macro of a known prefix or a
  tunable), so that a test macro with a new prefix cannot slip past the guard; a planted ``#define`` or ``-D`` of any prefix, old or new, is caught;
* policy: the new GPU test modules never ``pytest.skip`` (a skipped test proves nothing): conditions that would skip fail on B200 / B300 (``helpers.skip_or_fail``), a variant that does not
  build fails (``rows_workload.build_variant``).
"""
import json
import re
from pathlib import Path

import pytest

import helpers
import rows_workload as rw
import test_rows_profile as trp
from protocol_source import HASH, TEST_PREFIXES, default_text, is_test_macro, logical_lines, macros_in  # noqa: F401
from test_rows_profile import mutate

ROOT = Path(__file__).resolve().parents[2]
CSRC = ROOT / "csrc" / "cfbi"
TESTS = ROOT / "tests" / "cfbi"

# the macros of the variant builds, by prefix (the first three are the hooks of the row engine): protocol_source.TEST_PREFIXES, the one list that the reader of the pins uses too
TUNABLES = ("CFBI_ROWS_MINB", "CFBI_ROWS_PF_L1", "CFBI_K2_TILE_ONLY", "CFBI_HAVE_TCGEN05", "CFBI_CONV_UNROLL", "CFBI_CONV_THREADS", "CFBI_SRC_HASH")
GPU_MODULES = ("test_rows_occupancy.py", "test_rows_decode_families.py", "test_rows_delay.py", "test_rows_sanitizer.py", "test_rows_graph.py", "test_rows_profile_gpu.py", "test_k2_dyn_variants.py", "rows_workload.py", "rows_bounded.py",
               "test_decode_single_launch.py", "test_decode_single_launch_variants.py", "test_decode_policy_device.py", "sanitizer_workloads.py")


def sources(csrc=CSRC):
    return {str(f.relative_to(csrc)): f.read_text() for f in sorted(csrc.rglob("*")) if f.suffix in (".cu", ".cuh", ".h", ".cpp")}


@pytest.fixture(scope="module")
def rows_text():
    return (CSRC / "cfbi_rows.cuh").read_text()


@pytest.fixture(scope="module")
def launch_text():
    return (CSRC / "cfbi_launch.cu").read_text()


# ---- sites ------------------------------------------------------------------------------------------------------------------------------------------------------
ENTRY_CALL = "rows_entry<DECODE>("          # the plan load of rows_kernel<false> and the derived entry of rows_kernel<true>: one call, so one site before it
RAW_BARRIER = re.compile(r"\b__syncthreads\b")
BARRIER_MACRO = re.compile(r"\bROWS_SYNC(?:_END|_AB)?\(")
DELAY_CALL = re.compile(r"\bROWS_DELAY\((\d+)\)")


def site_problems(text):
    body = trp.kernel_body(text)
    first, last = trp.row_loop(body)
    probs = []
    for k, (n, l, g) in enumerate(body):
        if l.strip().startswith("//"):
            continue
        if RAW_BARRIER.search(l):
            probs.append("line %d: a raw barrier in rows_kernel has no delay site: %s" % (n, l.strip()))
        if BARRIER_MACRO.search(l) and not first <= k <= last:
            probs.append("line %d: a barrier macro outside the row loop: %s" % (n, l.strip()))
    sites = [(k, int(m.group(1))) for k, (n, l, g) in enumerate(body) if not l.strip().startswith("//") for m in DELAY_CALL.finditer(l)]
    if [s for k, s in sites] != [0, 14, 8]:
        return probs + ["the delay sites called directly in rows_kernel are %s, expected [0, 14, 8] (the others are inside the barrier macros)" % [s for k, s in sites]]
    code = trp.code_lines(body)
    entry, window, chunk = sites[0][0], sites[1][0], sites[2][0]
    barriers = [(k, l) for k, l in code if BARRIER_MACRO.match(l)]
    behind = [l for k, l in barriers if k < window]
    ahead = [l for k, l in barriers if k > window]
    if not behind or not behind[-1].startswith("ROWS_SYNC(1);") or not ahead or not ahead[0].startswith("ROWS_SYNC_AB();"):
        probs.append("site 14 is not in the window between barrier 1 and barrier 13 (ROWS_SYNC_AB)")
    plan = [k for k, l in code if ENTRY_CALL in l]
    if not plan or entry > plan[0]:
        probs.append("site 0 does not come before the plan load (the sequence entry of the kernel)")
    stamp = [k for k, l in code if "prof_entry = clock64()" in l]
    if stamp and stamp[0] > entry:
        probs.append("site 0 comes before the profile entry stamp")
    before = [l for k, l in code if k < chunk]
    if not before or not before[-1].startswith("ROWS_SYNC(6);"):
        probs.append("site 8 does not directly follow barrier 6")
    return probs


def test_every_barrier_of_the_row_engine_has_a_delay_site(rows_text):
    assert site_problems(rows_text) == []


@pytest.mark.parametrize("name,old,new,expect", [
    ("a raw barrier", "        ROWS_SYNC(4);\n", "        __syncthreads();\n", "raw barrier"),
    ("a raw barrier added", "        ROWS_TS(5);\n", "        ROWS_TS(5);\n        __syncthreads();\n", "raw barrier"),
    ("site 8 dropped", "            ROWS_DELAY(8);\n", "", "expected [0, 14, 8]"),
    ("a direct call of site 3", "        ROWS_SYNC(4);\n", "        ROWS_DELAY(3);\n        ROWS_SYNC(4);\n", "expected [0, 14, 8]"),
    ("site 0 behind the plan load", "    ROWS_DELAY(0);   // test hook: kernel entry, before the plan load and the first loads\n", "", "expected [0, 14, 8]"),
    ("site 8 before barrier 6", "            ROWS_SYNC(6);\n            ROWS_DELAY(8);\n", "            ROWS_DELAY(8);\n            ROWS_SYNC(6);\n", "directly follow barrier 6"),
    ("site 14 dropped", "            ROWS_DELAY(14);   // test hook: the window of barrier 13, in front of the stores\n", "", "expected [0, 14, 8]"),
    ("a second call of site 14", "        ROWS_SYNC(4);\n", "        ROWS_DELAY(14);\n        ROWS_SYNC(4);\n", "expected [0, 14, 8]"),
    ("a raw barrier for barrier 13", "        ROWS_SYNC_AB();\n", "        __syncthreads();\n", "raw barrier"),
])
def test_a_missing_or_misplaced_delay_site_is_caught(rows_text, name, old, new, expect):
    probs = site_problems(mutate(rows_text, old, new))
    assert any(expect in p for p in probs), (name, probs)


def test_site_0_after_the_plan_load_is_caught(rows_text):
    moved = mutate(rows_text, "    ROWS_DELAY(0);   // test hook: kernel entry, before the plan load and the first loads\n", "")
    moved = mutate(moved, "    const SeqPlan sp = rows_entry<DECODE>(p, n, dec_pos0);\n", "    const SeqPlan sp = rows_entry<DECODE>(p, n, dec_pos0);\n    ROWS_DELAY(0);\n")
    assert any("before the plan load" in p for p in site_problems(moved))


SITE_14 = "            ROWS_DELAY(14);   // test hook: the window of barrier 13, in front of the stores\n"


@pytest.mark.parametrize("name,anchor,placed", [
    ("before barrier 1", "        ROWS_SYNC(1);\n", "        ROWS_DELAY(14);\n        ROWS_SYNC(1);\n"),
    ("behind barrier 13", "        ROWS_SYNC_AB();\n", "        ROWS_SYNC_AB();\n        ROWS_DELAY(14);\n"),
    ("behind barrier 2", "        ROWS_SYNC(2);\n", "        ROWS_SYNC(2);\n        ROWS_DELAY(14);\n"),
])
def test_site_14_outside_the_window_of_barrier_13_is_caught(rows_text, name, anchor, placed):
    moved = mutate(mutate(rows_text, SITE_14, ""), anchor, placed)
    assert any("not in the window between barrier 1 and barrier 13" in p for p in site_problems(moved)), name


# ---- the default build ------------------------------------------------------------------------------------------------------------------------------------
VARIANT_NAMES = re.compile(r"\bg_rows_delay\w*|\brows_delay\b|\brows_delay_arrived\b|\bRowsDelayConfig\b|\bg_rows_test_pos_\w+|\bld_relaxed_gpu\b|__nanosleep|__threadfence_block|^\s*#\s*error\b"
                           r"|\bg_afrag_\w+|\bafrag_check_\w+|\bafrag_count\b|\bafrag_late_value\b|\bROWS_DELAY_LATE\b", re.M)


DECODE_ENTRY = ("atomicAdd(p.pos + ROWS_POS_INDEX(POS_ACCESS_ARRIVAL, w.slot), 1 << POS_COUNT_SHIFT)", "if ((old >> POS_COUNT_SHIFT) == p.H - 1) st_relaxed_gpu(p.pos + ROWS_POS_INDEX(POS_ACCESS_ELECTED_STORE, w.slot),")
DECODE_WRONG = ("POS_ACCESS_ARRIVAL, w.slot), 0)", "if ((old >> POS_COUNT_SHIFT) == 0) st_relaxed_gpu(", "p.pos[sp.slot] =", "ld_relaxed_gpu")


def default_build_problems(rows_text, launch_text):
    macros = macros_in([rows_text, launch_text])
    rows = default_text(rows_text, macros)
    launch = default_text(launch_text, macros)
    probs = []
    for want in ("#define ROWS_DELAY(n) do {} while (0)", "#define ROWS_SYNC(n) do { ROWS_DELAY(n); __syncthreads(); } while (0)", "#define ROWS_SYNC_END() ROWS_SYNC(7)",
                 "#define ROWS_SYNC_AB() ROWS_SYNC(13)", "#define ROWS_POS_INDEX(access, slot) (slot)", "#define ROWS_DECODE_ARRIVED(old, heads) do {} while (0)"):
        if want not in rows:
            probs.append("the default build does not read %r" % want)
    for m in VARIANT_NAMES.finditer(rows):
        probs.append("cfbi_rows.cuh: the default build still contains %r" % m.group(0).strip())
    for want in DECODE_ENTRY:                    # the single-launch entry as the default build reads it: the counting protocol, no wrong variant
        if want not in rows:
            probs.append("the default build of the decode entry does not read %r" % want)
    for bad in DECODE_WRONG:
        if bad in rows:
            probs.append("the default build of the decode entry still contains the wrong protocol %r" % bad)
    for fn in ("rows_test_delay_config", "rows_test_delay_hits", "rows_test_pos_offset", "rows_test_afrag_reset", "rows_test_afrag_mismatches"):
        body = re.search(r"^int %s\([^)]*\) \{\n(.*?)^\}" % fn, launch, re.M | re.S)
        if body is None:
            probs.append("cfbi_launch.cu: the setter %s is missing" % fn)
        elif "cudaErrorNotSupported" not in body.group(1) or "cudaMemcpy" in body.group(1) or "cudaDeviceSynchronize" in body.group(1):
            probs.append("cfbi_launch.cu: the default build of %s does more than return cudaErrorNotSupported" % fn)
    return probs


def test_the_variant_macros_of_the_row_engine_are_found(rows_text, launch_text):
    found = macros_in([rows_text, launch_text])
    assert {"CFBI_ROWS_DELAY_NS", "CFBI_ROWS_TEST_NO_END_BARRIER", "CFBI_ROWS_TEST_POS_OOB", "CFBI_ROWS_TEST_COMPILE_ERROR", "CFBI_ROWS_PROFILE", "CFBI_NEG_CONTROL"} <= found
    assert {"CFBI_ROWS_TEST_AFRAG_CHECK", "CFBI_ROWS_TEST_NO_AB_BARRIER", "CFBI_ROWS_TEST_AM_STORES_LATE", "CFBI_ROWS_TEST_AD_STORES_LATE", "CFBI_ROWS_TEST_AB_SHIFT_LANE",
            "CFBI_ROWS_TEST_AB_WRONG_PARITY", "CFBI_ROWS_TEST_AM_STALE_ZERO", "CFBI_ROWS_TEST_NONMINE_TILE", "CFBI_ROWS_TEST_ZERO_TILE_WRITE"} <= found      # the oracle build and the controls of the staged A fragments
    assert {"CFBI_DECODE_TEST_EVERY_CTA_WRITES", "CFBI_DECODE_TEST_FIRST_ARRIVER", "CFBI_DECODE_TEST_SNAPSHOT_AFTER_ARRIVAL", "CFBI_DECODE_TEST_NO_ELECTED_STORE"} <= found      # the wrong protocols of the single-launch decode entry
    assert "CFBI_DECODE_TEST_DELAY_NS" not in found      # retired: the delays of the decode entry are the sites 9 .. 12 of the row-engine delay hook


def test_the_default_build_of_the_hooks_is_the_unhooked_kernel(rows_text, launch_text):
    assert default_build_problems(rows_text, launch_text) == []


@pytest.mark.parametrize("name,where,old,new,expect", [
    ("the delay hook compiled in", "rows", "#else\n#define ROWS_DELAY(n) do {} while (0)\n#define ROWS_DECODE_ARRIVED(old, heads) do {} while (0)\n#endif", "#else\n#define ROWS_DELAY(n) rows_delay(n)\n#define ROWS_DECODE_ARRIVED(old, heads) do {} while (0)\n#endif", "ROWS_DELAY(n) do {}"),
    ("the arrival delays compiled in", "rows", "#else\n#define ROWS_DELAY(n) do {} while (0)\n#define ROWS_DECODE_ARRIVED(old, heads) do {} while (0)\n#endif", "#else\n#define ROWS_DELAY(n) do {} while (0)\n#define ROWS_DECODE_ARRIVED(old, heads) rows_delay_arrived(old, heads)\n#endif", "ROWS_DECODE_ARRIVED(old, heads) do {}"),
    ("the end barrier removed by default", "rows", "#else\n#define ROWS_SYNC_END() ROWS_SYNC(7)\n#endif", "#else\n#define ROWS_SYNC_END() ROWS_DELAY(7)\n#endif", "ROWS_SYNC_END() ROWS_SYNC(7)"),
    ("the position offset compiled in", "rows", "#else\n#define ROWS_POS_INDEX(access, slot) (slot)\n#endif", "#else\n#define ROWS_POS_INDEX(access, slot) ((slot) + 1)\n#endif", "ROWS_POS_INDEX(access, slot) (slot)"),
    ("the barrier of the AB tile removed by default", "rows", "#else\n#define ROWS_SYNC_AB() ROWS_SYNC(13)\n#endif", "#else\n#define ROWS_SYNC_AB() ROWS_DELAY(13)\n#endif", "ROWS_SYNC_AB() ROWS_SYNC(13)"),
    ("the fragment counter compiled in", "rows", "#ifdef CFBI_ROWS_TEST_AFRAG_CHECK\n// The fragment-level check (test build only)", "#if 1\n// The fragment-level check (test build only)", "g_afrag_mismatch"),
    ("the late-store delay compiled in", "rows", "#ifdef CFBI_ROWS_TEST_AM_STORES_LATE\n#define ROWS_DELAY_LATE() ROWS_DELAY(14)\n#endif", "#if 1\n#define ROWS_DELAY_LATE() ROWS_DELAY(14)\n#endif", "ROWS_DELAY_LATE"),
    ("the late AD store compiled in", "rows", "#ifdef CFBI_ROWS_TEST_AD_STORES_LATE\n        float afrag_late_value = 0.f;", "#if 1\n        float afrag_late_value = 0.f;", "afrag_late_value"),
    ("a fragment setter that works by default", "launch", "#else\n    return (int)cudaErrorNotSupported;\n#endif\n}\n\nint rows_test_afrag_mismatches(", "#else\n    return (int)cudaDeviceSynchronize();\n#endif\n}\n\nint rows_test_afrag_mismatches(", "rows_test_afrag_reset"),
    ("a sleep outside the macro", "rows", "    ROWS_DELAY(0);   // test hook", "    __nanosleep(1);\n    ROWS_DELAY(0);   // test hook", "__nanosleep"),
    ("the first-arriver election by default", "rows", "#ifndef CFBI_DECODE_TEST_NO_ELECTED_STORE\n            if ((old >> POS_COUNT_SHIFT) == p.H - 1) st_relaxed_gpu(", "#ifndef CFBI_DECODE_TEST_NO_ELECTED_STORE\n            if ((old >> POS_COUNT_SHIFT) == 0) st_relaxed_gpu(", "== p.H - 1) st_relaxed_gpu"),
    ("the elected store missing by default", "rows", "#ifndef CFBI_DECODE_TEST_NO_ELECTED_STORE\n            if ((old >> POS_COUNT_SHIFT) == p.H - 1) st_relaxed_gpu(", "#ifdef CFBI_DECODE_TEST_NO_ELECTED_STORE\n            if ((old >> POS_COUNT_SHIFT) == p.H - 1) st_relaxed_gpu(", "== p.H - 1) st_relaxed_gpu"),
    ("the non-counting read by default", "rows", "#else\n            const int old = atomicAdd(p.pos + ROWS_POS_INDEX(POS_ACCESS_ARRIVAL, w.slot), 1 << POS_COUNT_SHIFT);", "#else\n            const int old = atomicAdd(p.pos + ROWS_POS_INDEX(POS_ACCESS_ARRIVAL, w.slot), 0);", "1 << POS_COUNT_SHIFT"),
    ("the snapshot read by default", "rows", "#ifdef CFBI_DECODE_TEST_SNAPSHOT_AFTER_ARRIVAL\n            pos_smem = ld_relaxed_gpu(", "#if 1\n            pos_smem = ld_relaxed_gpu(", "ld_relaxed_gpu"),
    ("the every-CTA store by default", "rows", "#ifdef CFBI_DECODE_TEST_EVERY_CTA_WRITES\n        if (tid == 0) p.pos[sp.slot]", "#if 1\n        if (tid == 0) p.pos[sp.slot]", "wrong protocol"),
    ("a sleep of the decode entry outside its macro", "rows", "            ROWS_DELAY(9);   // test hook", "            __nanosleep(1);\n            ROWS_DELAY(9);   // test hook", "__nanosleep"),
    ("the compile error outside its macro", "rows", "#ifdef CFBI_ROWS_TEST_COMPILE_ERROR\n#error", "#if 1\n#error", "#error"),
    ("a setter that works by default", "launch", "    (void)offset; (void)accesses;\n    return (int)cudaErrorNotSupported;", "    return (int)cudaMemcpyToSymbol(g_rows_test_pos_offset, &offset, sizeof(offset));", "rows_test_pos_offset"),
])
def test_a_hook_that_leaks_into_the_default_build_is_caught(rows_text, launch_text, name, where, old, new, expect):
    r, l = (mutate(rows_text, old, new), launch_text) if where == "rows" else (rows_text, mutate(launch_text, old, new))
    probs = default_build_problems(r, l)
    assert any(expect in p for p in probs), (name, probs)


def test_the_mini_preprocessor_drops_the_branches_of_test_macros_only():
    text = "a\n#ifdef T\nb\n#else\nc\n#endif\n#ifndef T\nd\n#endif\n#ifdef OTHER\ne\n#ifdef T\nf\n#endif\n#endif\n#if 1\ng\n#endif\n"
    assert default_text(text, {"T"}).split("\n") == ["a", "c", "d", "#ifdef OTHER", "e", "#endif", "#if 1", "g", "#endif"]
    with pytest.raises(ValueError):
        default_text("#ifdef T\n#elif X\n#endif\n", {"T"})


# ---- the macro guard --------------------------------------------------------------------------------------------------------------------------------------
def macro_guard_problems(srcs, flag_texts, nvcc_flags):
    probs = []
    for name, text in srcs.items():
        for ln, line in logical_lines(text):                       # the lines that the preprocessor reads: a spliced or digraph directive is read too
            m = re.match(HASH + r"define[ \t\v\f]+(CFBI_\w+)", line)
            if m and is_test_macro(m.group(1)):
                probs.append("%s:%d defines the test macro %s: a test macro must only come from a -D flag of a variant build" % (name, ln, m.group(1)))
    for name, text in flag_texts.items():
        for m in re.finditer(r"-D\s*(CFBI_\w+)", text):
            if is_test_macro(m.group(1)):
                probs.append("%s carries the flag -D%s" % (name, m.group(1)))
    for f in nvcc_flags:
        m = re.match(r"-D\s*(CFBI_\w+)", f)
        if m and is_test_macro(m.group(1)):
            probs.append("NVCC_FLAGS carries %s" % f)
    classified = set()
    for text in srcs.values():
        code = "\n".join(line for _, line in logical_lines(text))
        classified |= set(re.findall(HASH + r"(?:ifdef|ifndef)[ \t\v\f]+(CFBI_\w+)", code)) | set(re.findall(r"defined\s*\(?\s*(CFBI_\w+)", code))
    for name in sorted(classified):
        if not is_test_macro(name) and name not in TUNABLES:
            probs.append("the macro %s is neither a test macro of a known prefix nor a tunable: add it to the macro guard" % name)
    return probs


def real_guard_inputs():
    from flash_kda.cfbi import _ext

    flags = {"_ext.py": (ROOT / "flash_kda" / "cfbi" / "_ext.py").read_text(), "setup.py": (ROOT / "setup.py").read_text()}
    return sources(), flags, list(_ext.NVCC_FLAGS)


def test_the_default_build_defines_no_test_macro_of_any_prefix():
    srcs, flags, nvcc = real_guard_inputs()
    assert macro_guard_problems(srcs, flags, nvcc) == []


@pytest.mark.parametrize("macro", ["CFBI_ROWS_DELAY_NS", "CFBI_ROWS_TEST_NO_END_BARRIER", "CFBI_ROWS_TEST_POS_OOB", "CFBI_ROWS_TEST_COMPILE_ERROR", "CFBI_ROWS_TEST_ANYTHING_NEW",
                                   "CFBI_ROWS_PROFILE", "CFBI_NEG_CONTROL", "CFBI_DYN_TEST_SKIP_ITEM", "CFBI_UMMA_RACE_DELAY_NS", "CFBI_CONV_NEG_CONTROL",
                                   "CFBI_DECODE_TEST_SNAPSHOT_AFTER_ARRIVAL", "CFBI_DECODE_TEST_EVERY_CTA_WRITES", "CFBI_DECODE_TEST_FIRST_ARRIVER", "CFBI_DECODE_TEST_NO_ELECTED_STORE",
                                   "CFBI_DECODE_TEST_ANYTHING_NEW"])
def test_a_planted_define_of_any_prefix_is_caught(macro):
    srcs, flags, nvcc = real_guard_inputs()
    planted = dict(srcs)
    planted["cfbi_common.cuh"] = planted["cfbi_common.cuh"] + "\n#define %s 1\n" % macro
    assert any(macro in p and "defines the test macro" in p for p in macro_guard_problems(planted, flags, nvcc))
    assert any(macro in p for p in macro_guard_problems(srcs, flags, nvcc + ["-D%s=1" % macro]))
    assert any(macro in p for p in macro_guard_problems(srcs, dict(flags, **{"setup.py": flags["setup.py"] + "\n'-D%s=1'" % macro}), nvcc))
    assert any(macro in p for p in macro_guard_problems(srcs, dict(flags, **{"_ext.py": flags["_ext.py"].replace('"-O3",', '"-O3", "-D%s=1",' % macro)}), nvcc))


def test_planting_the_delay_define_in_a_temporary_copy_of_the_sources_fails_the_guard(tmp_path):
    """The control as the acceptance criterion words it: a ``#define CFBI_ROWS_DELAY_NS`` in a copy of ``csrc/cfbi`` on disk, read back and judged by the guard."""
    import shutil

    copy = tmp_path / "cfbi"
    shutil.copytree(CSRC, copy)
    _, flags, nvcc = real_guard_inputs()
    assert macro_guard_problems(sources(copy), flags, nvcc) == []                      # the copy is as clean as the original
    target = copy / "cfbi_rows.cuh"
    target.write_text("#define CFBI_ROWS_DELAY_NS 2000\n" + target.read_text())
    probs = macro_guard_problems(sources(copy), flags, nvcc)
    assert any("cfbi_rows.cuh:1 defines the test macro CFBI_ROWS_DELAY_NS" in p for p in probs), probs


def test_a_tunable_is_no_test_macro_and_may_be_defined():
    srcs, flags, nvcc = real_guard_inputs()
    planted = dict(srcs)
    planted["cfbi_common.cuh"] = planted["cfbi_common.cuh"] + "\n#define CFBI_ROWS_MINB 1\n"
    assert macro_guard_problems(planted, flags, nvcc + ["-DCFBI_ROWS_MINB=1"]) == []


def test_a_test_macro_with_a_new_prefix_cannot_slip_past_the_guard():
    srcs, flags, nvcc = real_guard_inputs()
    planted = dict(srcs)
    planted["cfbi_rows.cuh"] = planted["cfbi_rows.cuh"] + "\n#ifdef CFBI_ROWS_EXPERIMENT_SPIN\n#endif\n"
    assert any("CFBI_ROWS_EXPERIMENT_SPIN" in p and "add it to the macro guard" in p for p in macro_guard_problems(planted, flags, nvcc))


# ---- the policy: no skip, failing variant builds -----------------------------------------------------------------------------------------------------------
SKIP = re.compile(r"\bpytest\.(?:skip|importorskip|mark\.skip\w*|xfail)\b|\bpytest\.mark\.xfail\b")


def skip_problems(texts):
    return ["%s:%d calls %s" % (name, ln, SKIP.search(line).group(0)) for name, text in texts.items() for ln, line in enumerate(text.splitlines(), 1) if SKIP.search(line)]


def gpu_module_texts():
    return {n: (TESTS / n).read_text() for n in GPU_MODULES if (TESTS / n).exists()}


def test_the_row_engine_tests_never_skip():
    assert skip_problems(gpu_module_texts()) == []


def test_a_skip_in_a_row_engine_test_is_caught():
    texts = gpu_module_texts()
    texts["test_rows_delay.py"] += "\n\ndef test_x():\n    pytest.skip('no')\n"
    texts["rows_workload.py"] += "\n\ndef f():\n    pytest.importorskip('x')\n"
    probs = skip_problems(texts)
    assert any("test_rows_delay.py" in p and "pytest.skip" in p for p in probs) and any("rows_workload.py" in p and "importorskip" in p for p in probs)


@pytest.mark.parametrize("capability,fails", [((10, 0), True), ((10, 3), True), ((9, 0), False), ((8, 9), False), ((12, 0), False)])
def test_a_condition_that_would_skip_fails_on_b200_and_b300_only(capability, fails):
    assert helpers.must_run(capability) is fails
    with pytest.raises(pytest.fail.Exception if fails else pytest.skip.Exception) as e:
        helpers.skip_or_fail("the thing is missing", capability)
    assert "the thing is missing" in str(e.value)


def test_a_variant_that_does_not_build_fails_not_skips():
    def broken(name, defines):
        raise RuntimeError("Error building extension 'x': [1/3] c++ " + "-DFLAG " * 400 + "\nFAILED: cfbi_launch.cuda.o\ncfbi_rows.cuh(78): error: #error directive: deliberate\nninja: build stopped")

    with pytest.raises(pytest.fail.Exception) as e:
        rw.build_variant("x", ("A=1",), loader=broken)
    msg = str(e.value)
    assert "does not build" in msg and "deliberate" in msg and len(msg) < 2500, msg[-300:]
    assert rw.build_variant("ok", ("A=1",), loader=lambda name, defines: (name, defines)) == ("ok", ("A=1",))


def test_a_run_that_requires_a_gpu_fails_without_one_instead_of_skipping(tmp_path):
    """The shared ``ext`` fixture skips without CUDA unless the run says that it needs a GPU (``CFBI_REQUIRE_GPU=1``, set by the runs on allocated nodes). Only meaningful where there is
    no CUDA device (the login node, CI): with one the tests of the subprocess would run."""
    import os
    import subprocess
    import sys

    import torch

    if torch.cuda.is_available():
        return          # nothing to show here; the nodes run with CFBI_REQUIRE_GPU=1 and a device
    target = str(TESTS / "test_rows_occupancy.py") + "::test_the_binding_reports_every_instantiation_of_the_row_engine"
    base = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs", target]
    env = {k: v for k, v in os.environ.items() if k != "CFBI_REQUIRE_GPU"}
    soft = subprocess.run(base, capture_output=True, text=True, env=env, cwd=str(ROOT))
    assert soft.returncode == 0 and "1 skipped" in soft.stdout, soft.stdout[-800:]
    hard = subprocess.run(base, capture_output=True, text=True, env=dict(env, CFBI_REQUIRE_GPU="1"), cwd=str(ROOT))
    assert hard.returncode != 0 and "CFBI_REQUIRE_GPU=1" in hard.stdout and "1 error" in hard.stdout and "skipped" not in hard.stdout, hard.stdout[-800:]    # a failed fixture is an error


def test_a_variant_builds_in_a_directory_of_its_own(monkeypatch):
    """A variant that shares the build directory rewrites its build.ninja: the next load of the default extension rebuilds it (under a tool that hangs)."""
    import os

    seen = []

    def loader(name, defines):
        seen.append(os.environ.get("FLASH_KDA_CFBI_BUILD_DIR"))
        return name

    monkeypatch.setenv("FLASH_KDA_CFBI_BUILD_DIR", "/build/tree")
    assert rw.build_variant("rowsx", ("A=1",), loader=loader) == "rowsx"
    assert seen == ["/build/tree/variants/rowsx"] and os.environ["FLASH_KDA_CFBI_BUILD_DIR"] == "/build/tree"

    def broken(name, defines):
        raise RuntimeError("Error building\nx: error: boom")

    with pytest.raises(pytest.fail.Exception):                       # a failing build restores the directory as well
        rw.build_variant("rowsz", ("A=1",), loader=broken)
    assert os.environ["FLASH_KDA_CFBI_BUILD_DIR"] == "/build/tree"

    monkeypatch.delenv("FLASH_KDA_CFBI_BUILD_DIR")                    # no build directory set: nothing is set for the variant either
    seen.clear()
    rw.build_variant("rowsy", ("A=1",), loader=loader)
    assert seen == [None] and "FLASH_KDA_CFBI_BUILD_DIR" not in os.environ


def test_the_sanitizer_does_not_follow_child_processes(monkeypatch, tmp_path):
    import subprocess
    import types

    import test_rows_sanitizer as ts

    seen = []
    monkeypatch.setattr(ts, "sanitizer", lambda: "compute-sanitizer")
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: seen.append(cmd) or types.SimpleNamespace(returncode=0, stdout="", stderr=""))
    ts.run_driver("memcheck", tmp_path, ts.BOUNDED_DRIVER)
    assert seen and "--target-processes" in seen[0] and seen[0][seen[0].index("--target-processes") + 1] == "application-only", seen


# ---- the bounded racecheck workload executes what it claims (no GPU) ---------------------------------------------------------------------------------------
def test_the_bounded_racecheck_workload_covers_every_position_the_chunk_end_and_the_rows_after_it():
    import test_rows_sanitizer as ts

    cov = rw.launch_coverage(ts.BOUNDED_LAUNCHES, ts.BOUNDED_H)
    assert rw.coverage_problems(cov) == []
    assert cov["positions"] == list(range(64)) and cov["chunk_end_rows"] == 1 and cov["rows_after_chunk_end"] == 2 and cov["cross_subblock_rows"] >= 48
    assert cov["final_pos"] == {0: 6, 3: 2, 7: 3} and cov["live_ctas"] == [3 * ts.BOUNDED_H, 3 * ts.BOUNDED_H]


def bounded_mutants():
    import test_rows_sanitizer as ts

    (lens1, slots1), launch2 = ts.BOUNDED_LAUNCHES
    return [
        ("the round-3 workload: the 66-row sequence sits on a padding slot", [([1, 66, 3, 5], [3, -1, 5, 0]), ([1, 1, 1], [5, 3, 0])], "positions only"),
        ("the long sequence of the current workload is remapped to padding", [(lens1, [slots1[0], -1] + slots1[2:]), launch2], "positions only"),
        ("a long sequence that stops before the chunk end", [([1, 40, 3, 5], slots1), launch2], "no row finishes a chunk"),
        ("a chunk end that is the last row of its launch", [([1, 64, 3, 5], slots1), launch2], "no row follows a chunk end inside its launch"),
        ("a launch of padding entries only", [(lens1, slots1), ([1, 1, 1], [-1, -1, -1])], "no live CTA"),
        ("one row per sequence", [([1, 1, 1, 1], slots1), launch2], "positions only"),
    ]


@pytest.mark.parametrize("name,launches,expect", bounded_mutants(), ids=[m[0] for m in bounded_mutants()])
def test_a_bounded_workload_that_does_not_reach_the_phases_is_rejected(name, launches, expect):
    import test_rows_sanitizer as ts

    probs = rw.coverage_problems(rw.launch_coverage(launches, ts.BOUNDED_H))
    assert any(expect in p for p in probs), (name, probs)


def driver_prefix(launches):
    """The part of the bounded driver that runs before it needs the device: it plans, prints the coverage and asserts it."""
    import test_rows_sanitizer as ts

    script = ts.BOUNDED_DRIVER.format(tests=str(TESTS), repo=str(ROOT), done=ts.DONE, path="single", heads=ts.BOUNDED_H, pool=ts.BOUNDED_POOL, launches=repr(launches))
    assert script.count("ext = load_ext()") == 1
    return script.split("ext = load_ext()")[0]


def test_the_driver_asserts_the_coverage_of_its_plan_before_it_needs_the_device(capsys):
    import test_rows_sanitizer as ts

    exec(compile(driver_prefix(ts.BOUNDED_LAUNCHES), "bounded_driver_prefix", "exec"), {})
    printed_line = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("ROWS_SANITIZER_COVERAGE ")]
    assert len(printed_line) == 1 and json.loads(printed_line[0].split(" ", 1)[1]) == json.loads(json.dumps(rw.launch_coverage(ts.BOUNDED_LAUNCHES, ts.BOUNDED_H), sort_keys=True))
    for name, launches, expect in bounded_mutants():
        with pytest.raises(AssertionError, match="does not cover the row engine"):
            exec(compile(driver_prefix(launches), "bounded_driver_prefix", "exec"), {})


def test_the_driver_compares_the_final_positions_with_its_plan():
    import test_rows_sanitizer as ts

    script = ts.BOUNDED_DRIVER
    assert "ROWS_SANITIZER_FINAL_POS" in script and 'cov["final_pos"]' in script and "assert final == expected" in script
