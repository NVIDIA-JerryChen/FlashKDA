"""Source rules of the test hooks of the row engine (plain Python, no GPU: ``pytest --noconftest tests/cfbi/test_rows_hooks.py``), each with mutants that break it.

The hooks (``ROWS_SYNC``, ``ROWS_SYNC_END``, ``ROWS_DELAY``, ``ROWS_POS_INDEX`` and the host setters; ``cfbi_rows.cuh``, ``cfbi_launch.cu``) let the device tests delay a chosen warp at the
barriers of a row, remove the barrier that ends a row, write the position at an out-of-bounds index and fail a build on purpose. No comparison of results can show that the *default* build
has none of this, nor that a barrier added later has a delay site, so the rules are pinned as text:

* sites: no raw ``__syncthreads`` is left in ``rows_kernel`` (every barrier is a numbered ``ROWS_SYNC`` macro and so has a delay site; the order of the numbers is pinned together with the
  clock-profile stamps in ``test_rows_profile.py``); the only direct delay calls are site 0 (kernel entry, after the profile entry stamp and before the plan load) and site 8 (right after
  barrier 6, inside the chunk-end phase);
* default build: read with every test macro undefined, the hook macros are what the unhooked kernel contained (``ROWS_SYNC(n)`` is a barrier, ``ROWS_SYNC_END()`` is ``ROWS_SYNC(7)``,
  ``ROWS_DELAY`` and the other hooks are empty, ``ROWS_POS_INDEX(slot)`` is the slot) and none of the machinery of the variants is left; the host setters only return an error;
* macro guard: no source defines a test macro, no build flag of the default build carries one, and every macro that the sources test is classified (a test macro of a known prefix or a
  tunable), so that a test macro with a new prefix cannot slip past the guard; a planted ``#define`` or ``-D`` of any prefix, old or new, is caught;
* policy: the new GPU test modules never ``pytest.skip`` (a skipped test proves nothing): conditions that would skip fail on B200 / B300 (``helpers.skip_or_fail``), a variant that does not
  build fails (``rows_workload.build_variant``).
"""
import re
from pathlib import Path

import pytest

import helpers
import rows_workload as rw
import test_rows_profile as trp
from test_rows_profile import mutate

ROOT = Path(__file__).resolve().parents[2]
CSRC = ROOT / "csrc" / "cfbi"
TESTS = ROOT / "tests" / "cfbi"

# the macros of the variant builds, by prefix (the first three are the hooks of the row engine), and the build-time tunables that are no test macros
TEST_PREFIXES = ("CFBI_ROWS_PROFILE", "CFBI_ROWS_DELAY_NS", "CFBI_ROWS_TEST_", "CFBI_NEG_CONTROL", "CFBI_K1_PROFILE", "CFBI_K2_PROFILE", "CFBI_UMMA_PROFILE", "CFBI_DYN_TEST_",
                 "CFBI_K2_TEST_", "CFBI_UMMA_NEG_", "CFBI_UMMA_RACE_", "CFBI_CONV_NEG_", "CFBI_LIN_NEG_")
TUNABLES = ("CFBI_ROWS_MINB", "CFBI_ROWS_PF_L1", "CFBI_K2_TILE_ONLY", "CFBI_HAVE_TCGEN05", "CFBI_CONV_UNROLL", "CFBI_CONV_THREADS", "CFBI_SRC_HASH")
GPU_MODULES = ("test_rows_occupancy.py", "test_rows_decode_families.py", "test_rows_delay.py", "test_rows_sanitizer.py", "test_rows_graph.py", "test_rows_profile_gpu.py", "test_k2_dyn_variants.py", "rows_workload.py")


def is_test_macro(name):
    return any(name.startswith(p) for p in TEST_PREFIXES)


def sources(csrc=CSRC):
    return {str(f.relative_to(csrc)): f.read_text() for f in sorted(csrc.rglob("*")) if f.suffix in (".cu", ".cuh", ".h", ".cpp")}


@pytest.fixture(scope="module")
def rows_text():
    return (CSRC / "cfbi_rows.cuh").read_text()


@pytest.fixture(scope="module")
def launch_text():
    return (CSRC / "cfbi_launch.cu").read_text()


# ---- sites ------------------------------------------------------------------------------------------------------------------------------------------------------
RAW_BARRIER = re.compile(r"\b__syncthreads\b")
BARRIER_MACRO = re.compile(r"\bROWS_SYNC(?:_END)?\(")
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
    if [s for k, s in sites] != [0, 8]:
        return probs + ["the delay sites called directly in rows_kernel are %s, expected [0, 8] (the others are inside the barrier macros)" % [s for k, s in sites]]
    code = trp.code_lines(body)
    entry, chunk = sites[0][0], sites[1][0]
    plan = [k for k, l in code if "p.plan[" in l]
    if not plan or entry > plan[0]:
        probs.append("site 0 does not come before the plan load")
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
    ("site 8 dropped", "            ROWS_DELAY(8);\n", "", "expected [0, 8]"),
    ("a direct call of site 3", "        ROWS_SYNC(4);\n", "        ROWS_DELAY(3);\n        ROWS_SYNC(4);\n", "expected [0, 8]"),
    ("site 0 behind the plan load", "    ROWS_DELAY(0);   // test hook: kernel entry, before the plan load and the first loads\n", "", "expected [0, 8]"),
    ("site 8 before barrier 6", "            ROWS_SYNC(6);\n            ROWS_DELAY(8);\n", "            ROWS_DELAY(8);\n            ROWS_SYNC(6);\n", "directly follow barrier 6"),
])
def test_a_missing_or_misplaced_delay_site_is_caught(rows_text, name, old, new, expect):
    probs = site_problems(mutate(rows_text, old, new))
    assert any(expect in p for p in probs), (name, probs)


def test_site_0_after_the_plan_load_is_caught(rows_text):
    moved = mutate(rows_text, "    ROWS_DELAY(0);   // test hook: kernel entry, before the plan load and the first loads\n", "")
    moved = mutate(moved, "    const SeqPlan sp = p.plan[n];\n", "    const SeqPlan sp = p.plan[n];\n    ROWS_DELAY(0);\n")
    assert any("before the plan load" in p for p in site_problems(moved))


# ---- the default build ------------------------------------------------------------------------------------------------------------------------------------
def macros_in(texts):
    """The macro names of the variants that some text tests (derived, not listed): every ``#ifdef`` / ``#ifndef`` / ``defined()`` of a name with a test prefix."""
    found = set()
    for t in texts:
        found |= set(re.findall(r"#\s*(?:ifdef|ifndef)\s+(CFBI_\w+)", t)) | set(re.findall(r"defined\s*\(?\s*(CFBI_\w+)", t))
    return {m for m in found if is_test_macro(m)}


def default_text(text, macros):
    """The text as a build that defines none of ``macros`` reads it: the branch of ``#ifdef M`` is dropped and its ``#else`` branch kept (``#ifndef M`` the other way round), the
    directive lines of those conditionals are dropped, every other line stays."""
    out, stack = [], []
    for line in text.splitlines():
        m = re.match(r"^\s*#\s*(ifdef|ifndef|if|elif|else|endif)\b\s*(\w+)?", line)
        skipping = any(f[1] for f in stack)
        if m:
            d, name = m.group(1), m.group(2)
            if d in ("ifdef", "ifndef") and name in macros:
                stack.append(["test", d == "ifdef"])
                continue
            if d in ("ifdef", "ifndef", "if"):
                stack.append(["other", False])
            elif d == "endif":
                if stack.pop()[0] == "test":
                    continue
            elif d == "else" and stack and stack[-1][0] == "test":
                stack[-1][1] = not stack[-1][1]
                continue
            elif d == "elif" and stack and stack[-1][0] == "test":
                raise ValueError("#elif of a conditional on a test macro is not read by this preprocessor")
        if not skipping:
            out.append(line)
    assert not stack, "unbalanced preprocessor conditionals"
    return "\n".join(out)


VARIANT_NAMES = re.compile(r"\bg_rows_delay\w*|\brows_delay\b|\bRowsDelayConfig\b|\bg_rows_test_pos_offset\b|__nanosleep|__threadfence_block|^\s*#\s*error\b", re.M)


def default_build_problems(rows_text, launch_text):
    macros = macros_in([rows_text, launch_text])
    rows = default_text(rows_text, macros)
    launch = default_text(launch_text, macros)
    probs = []
    for want in ("#define ROWS_DELAY(n) do {} while (0)", "#define ROWS_SYNC(n) do { ROWS_DELAY(n); __syncthreads(); } while (0)", "#define ROWS_SYNC_END() ROWS_SYNC(7)",
                 "#define ROWS_POS_INDEX(slot) (slot)"):
        if want not in rows:
            probs.append("the default build does not read %r" % want)
    for m in VARIANT_NAMES.finditer(rows):
        probs.append("cfbi_rows.cuh: the default build still contains %r" % m.group(0).strip())
    for fn in ("rows_test_delay_config", "rows_test_delay_hits", "rows_test_pos_offset"):
        body = re.search(r"^int %s\([^)]*\) \{\n(.*?)^\}" % fn, launch, re.M | re.S)
        if body is None:
            probs.append("cfbi_launch.cu: the setter %s is missing" % fn)
        elif "cudaErrorNotSupported" not in body.group(1) or "cudaMemcpy" in body.group(1) or "cudaDeviceSynchronize" in body.group(1):
            probs.append("cfbi_launch.cu: the default build of %s does more than return cudaErrorNotSupported" % fn)
    return probs


def test_the_variant_macros_of_the_row_engine_are_found(rows_text, launch_text):
    found = macros_in([rows_text, launch_text])
    assert {"CFBI_ROWS_DELAY_NS", "CFBI_ROWS_TEST_NO_END_BARRIER", "CFBI_ROWS_TEST_POS_OOB", "CFBI_ROWS_TEST_COMPILE_ERROR", "CFBI_ROWS_PROFILE", "CFBI_NEG_CONTROL"} <= found


def test_the_default_build_of_the_hooks_is_the_unhooked_kernel(rows_text, launch_text):
    assert default_build_problems(rows_text, launch_text) == []


@pytest.mark.parametrize("name,where,old,new,expect", [
    ("the delay hook compiled in", "rows", "#else\n#define ROWS_DELAY(n) do {} while (0)\n#endif", "#else\n#define ROWS_DELAY(n) rows_delay(n)\n#endif", "ROWS_DELAY(n) do {}"),
    ("the end barrier removed by default", "rows", "#else\n#define ROWS_SYNC_END() ROWS_SYNC(7)\n#endif", "#else\n#define ROWS_SYNC_END() ROWS_DELAY(7)\n#endif", "ROWS_SYNC_END() ROWS_SYNC(7)"),
    ("the position offset compiled in", "rows", "#else\n#define ROWS_POS_INDEX(slot) (slot)\n#endif", "#else\n#define ROWS_POS_INDEX(slot) ((slot) + 1)\n#endif", "ROWS_POS_INDEX(slot) (slot)"),
    ("a sleep outside the macro", "rows", "    ROWS_DELAY(0);   // test hook", "    __nanosleep(1);\n    ROWS_DELAY(0);   // test hook", "__nanosleep"),
    ("the compile error outside its macro", "rows", "#ifdef CFBI_ROWS_TEST_COMPILE_ERROR\n#error", "#if 1\n#error", "#error"),
    ("a setter that works by default", "launch", "    (void)offset;\n    return (int)cudaErrorNotSupported;", "    return (int)cudaMemcpyToSymbol(g_rows_test_pos_offset, &offset, sizeof(offset));", "rows_test_pos_offset"),
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
        for ln, line in enumerate(text.splitlines(), 1):
            m = re.match(r"\s*#\s*define\s+(CFBI_\w+)", line)
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
        classified |= set(re.findall(r"#\s*(?:ifdef|ifndef)\s+(CFBI_\w+)", text)) | set(re.findall(r"defined\s*\(?\s*(CFBI_\w+)", text))
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
                                   "CFBI_ROWS_PROFILE", "CFBI_NEG_CONTROL", "CFBI_DYN_TEST_SKIP_ITEM", "CFBI_UMMA_RACE_DELAY_NS", "CFBI_CONV_NEG_CONTROL"])
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
