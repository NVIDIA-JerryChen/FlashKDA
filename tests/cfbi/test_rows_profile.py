"""Self-tests of the row-engine clock profile (``-DCFBI_ROWS_PROFILE=1``): the aggregation of the stamps into a record, and the source rules that keep the profile out of the default
build and out of the measured phases.

Plain Python (no GPU, no torch): ``pytest --noconftest tests/cfbi/test_rows_profile.py``. The device side (the profile build leaves every bit of the default build unchanged, the stamps are
ordered, a perturbed profile build is caught) is in ``test_rows_profile_gpu.py``.

The source rules of ``csrc/cfbi/cfbi_rows.cuh`` cannot be seen by any comparison of results, nor by a bit-identity test, so they are pinned here as text rules, each with mutants that break it
(a rule that cannot fail proves nothing). After every edit of ``rows_kernel`` the rules are read in the source again; a deliberate change of the barrier structure updates ``ROW_SEQUENCE``
together with the stamps:

* entry: the entry stamp is the first statement of ``rows_kernel``, before the plan load and the rest of the prologue;
* exit: the exit stamp comes after the last global store of the kernel (the ``pos`` store of the fused finish);
* order: the per-row stamps ``ROWS_TS(0..7)`` keep their positions relative to the barriers (the sequence below), stamp 0 is the first statement of the row, a stamp that ends a barrier
  interval follows its barrier directly, and the stamp that ends the row follows the last barrier of the row;
* buffer: the profile buffer is written only after the last stamp of the row, so that no write lies inside a measured phase;
* guard: every line that belongs to the profile (stamps, buffer, accessors, perturbation hook) lies inside a ``CFBI_ROWS_PROFILE`` region; the default build compiles none of it.
"""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "benchmarks",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rows_profile_record as rpr  # noqa: E402

CSRC = ROOT / "csrc" / "cfbi"
ROWS_SRC = CSRC / "cfbi_rows.cuh"
LAUNCH_SRC = CSRC / "cfbi_launch.cu"
BINDING_SRC = CSRC / "cfbi_binding.cpp"

GUARD = "CFBI_ROWS_PROFILE"
PERTURB = "CFBI_ROWS_PROFILE_PERTURB"
# what belongs to the profile in the row engine / in the launcher (the stamp macro ROWS_TS is a no-op outside the profile build and may stay unguarded)
ROWS_PROFILE_NAMES = re.compile(r"\bprof_\w+|\bg_rows_prof\b|\bROWS_PROF_\w+|\bclock64\b|%%smid|\btp\b|\bCFBI_ROWS_PROFILE_PERTURB\b")
LAUNCH_PROFILE_NAMES = re.compile(r"\bg_rows_prof\b|\bROWS_PROF_\w+")
KERNEL_HEAD = re.compile(r"^__global__ void __launch_bounds__\(NTHREADS, CFBI_ROWS_MINB\) rows_kernel\(const RowsParams<DECODE> p\) \{")
ENTRY_CALL = "rows_entry<DECODE>("   # the plan load of rows_kernel<false> and the derived entry of rows_kernel<true>
STAMP = re.compile(r"\bROWS_TS\((\d)\)")
BARRIER = re.compile(r"\bROWS_SYNC(?:_END)?\(\d?\);")     # every barrier of the row engine is one of the numbered barrier macros (cfbi_rows.cuh): barrier n is ROWS_SYNC(n), the one that ends a row ROWS_SYNC_END()
EXIT_STAMP = re.compile(r"\[8\] = \(unsigned\)\(clock64\(\) - prof_entry\)")
GLOBAL_STORE = re.compile(r"\b(?:Xg|Rk|Rg|Ru|Rut|Rw|R|Eg|p\.out|p\.out_f32|p\.pos|p\.E|p\.rows)\[[^\]]*\]\s*=[^=]")
ROW_LOOP = re.compile(r"^    for \(int i = 0; i < nrows; \+\+i\) \{")
AFTER_LOOP = "    if (DECODE) {"          # the first statement behind the row loop: the end of the kernel (the position word of the call, then the exit stamp)
POS_STORE = "p.pos[ROWS_POS_INDEX(POS_ACCESS_FUSED_STORE, sp.slot)] ="

# the barriers and stamps of one row in source order (a barrier inside an ``if`` is still one entry; SYNC<n> is barrier n, SYNC7 the barrier that ends the row): changes with the barrier
# structure of the row engine, not with the profile; the delay variant's sites are these barrier numbers (test_rows_hooks.py)
ROW_SEQUENCE = ["TS0", "SYNC1", "TS1", "SYNC2", "TS2", "SYNC3", "TS3", "SYNC4", "TS4", "SYNC5", "TS5", "TS6", "SYNC6", "SYNC7", "TS7"]
ADJACENT_TO_BARRIER = (1, 2, 4, 5, 7)   # stamps that must directly follow a barrier (stamp 3 follows the optional cross-block barrier block, stamp 6 the output stores)


# ---- a small preprocessor-aware reader of the sources ----------------------------------------------------------------------------------------------
def guard_regions(text):
    """For every line, the set of macros X such that the line lies in the ``#ifdef X`` branch of a conditional (the ``#else`` branch of an ``#ifdef X`` is outside X; numeric conditionals guard nothing)."""
    stack, out = [], []
    for line in text.splitlines():
        m = re.match(r"^\s*#\s*(ifdef|ifndef|if|elif|else|endif)\b\s*(\w+)?", line)
        if m:
            d, name = m.group(1), m.group(2)
            if d == "ifdef":
                stack.append([name, True])
            elif d == "ifndef":
                stack.append([name, False])
            elif d == "if":
                stack.append([None, False])
            elif d == "elif":
                stack[-1][1] = False
            elif d == "else":
                stack[-1][1] = not stack[-1][1]
            elif d == "endif":
                stack.pop()
        out.append(frozenset(n for n, active in stack if n and active))
    assert not stack, "unbalanced preprocessor conditionals"
    return out


def kernel_body(text):
    """``[(line number, line, guards)]`` of the body of ``rows_kernel`` (after its opening line, up to the closing brace at column 0)."""
    lines = text.splitlines()
    guards = guard_regions(text)
    start = next(i for i, l in enumerate(lines) if KERNEL_HEAD.match(l))
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "}")
    return [(i + 1, lines[i], guards[i]) for i in range(start + 1, end)]


def code_lines(body):
    """The lines of ``body`` that are neither blank, nor comments, nor preprocessor directives, nor pragmas: ``(index in body, stripped text)``."""
    return [(k, l.strip()) for k, (n, l, g) in enumerate(body) if l.strip() and not l.strip().startswith(("//", "#"))]


def row_loop(body):
    """``(first, last)`` indexes of ``body`` of the row loop: its opening line up to the line before the statement that follows it (the end of the kernel, ``AFTER_LOOP``)."""
    first = next(k for k, (n, l, g) in enumerate(body) if ROW_LOOP.match(l))
    last = next(k for k, (n, l, g) in enumerate(body) if l == AFTER_LOOP) - 1
    return first, last


def unguarded(text, names, guard=GUARD):
    """Lines that mention a profile name outside a ``guard`` region."""
    guards = guard_regions(text)
    return [(i + 1, l.strip()) for i, l in enumerate(text.splitlines()) if names.search(l) and guard not in guards[i]]


# ---- the rules, each as a function that returns the list of its violations -------------------------------------------------------------------------
def entry_problems(text):
    body = kernel_body(text)
    code = code_lines(body)
    probs = []
    if not code or not code[0][1].startswith("const long long prof_entry = clock64();"):
        probs.append("the first statement of rows_kernel is not the entry stamp: %r" % (code[0][1] if code else None))
    else:
        if GUARD not in body[code[0][0]][2]:
            probs.append("the entry stamp is not inside a %s region" % GUARD)
        plan = [k for k, l in code if ENTRY_CALL in l]
        if not plan or plan[0] < code[0][0]:
            probs.append("the plan load does not come after the entry stamp")
    first_clock = next((k for k, (n, l, g) in enumerate(body) if "clock64" in l and not l.strip().startswith("//")), None)
    if first_clock is None or first_clock != code[0][0]:
        probs.append("another clock read precedes the entry stamp")
    return probs


def exit_problems(text):
    body = kernel_body(text)
    stores = [k for k, (n, l, g) in enumerate(body) if GLOBAL_STORE.search(l) and PERTURB not in g]
    pos_store = [k for k, (n, l, g) in enumerate(body) if POS_STORE in l]
    exits = [k for k, (n, l, g) in enumerate(body) if EXIT_STAMP.search(l)]
    probs = []
    if not stores or len(pos_store) != 1 or len(exits) != 1:
        return ["expected global stores, exactly one fused pos store and one exit stamp, found %d, %d and %d" % (len(stores), len(pos_store), len(exits))]
    if exits[0] < max(stores):
        probs.append("a global store (line %d) comes after the exit stamp (line %d)" % (body[max(stores)][0], body[exits[0]][0]))
    if exits[0] < pos_store[0]:
        probs.append("the exit stamp comes before the fused pos store")
    if GUARD not in body[exits[0]][2]:
        probs.append("the exit stamp is not inside a %s region" % GUARD)
    return probs


def row_tokens(body, first, last):
    toks = []
    for k in range(first, last + 1):
        line = body[k][1]
        if line.strip().startswith("//"):
            continue
        for m in re.finditer(r"\bROWS_TS\((\d)\)|\bROWS_SYNC\((\d)\);|\bROWS_SYNC_END\(\);", line):
            toks.append("TS" + m.group(1) if m.group(1) is not None else ("SYNC" + m.group(2) if m.group(2) is not None else "SYNC7"))
    return toks


def order_problems(text):
    body = kernel_body(text)
    first, last = row_loop(body)
    probs = []
    seq = row_tokens(body, first, last)
    if seq != ROW_SEQUENCE:
        probs.append("stamps and barriers of the row are %s, expected %s" % (" ".join(seq), " ".join(ROW_SEQUENCE)))
    code = [(k, t) for k, t in code_lines(body) if first < k <= last]
    if not code or not STAMP.search(code[0][1]) or STAMP.search(code[0][1]).group(1) != "0":
        probs.append("stamp 0 is not the first statement of the row loop")
    for i in ADJACENT_TO_BARRIER:
        at = [j for j, (k, t) in enumerate(code) if re.search(r"\bROWS_TS\(%d\)" % i, t)]
        if len(at) != 1:
            probs.append("stamp %d appears %d times" % (i, len(at)))
        elif at[0] == 0 or not BARRIER.search(code[at[0] - 1][1]):
            probs.append("stamp %d does not directly follow a barrier" % i)
    return probs


def buffer_problems(text):
    body = kernel_body(text)
    first, last = row_loop(body)
    stamp7 = [k for k in range(first, last + 1) if re.search(r"\bROWS_TS\(7\)", body[k][1])]
    writes = [k for k in range(first, last + 1) if re.search(r"\bg_rows_prof\b", body[k][1]) and not body[k][1].strip().startswith("//")]
    before = [k for k in range(0, first) if re.search(r"\bg_rows_prof\b", body[k][1]) and not body[k][1].strip().startswith("//")]
    probs = []
    if len(stamp7) != 1 or not writes:
        return ["expected one stamp 7 and at least one buffer write in the row loop, found %d and %d" % (len(stamp7), len(writes))]
    if before:
        probs.append("the buffer is touched before the row loop")
    if min(writes) < stamp7[0]:
        probs.append("the buffer is written (line %d) before the last stamp of the row (line %d)" % (body[min(writes)][0], body[stamp7[0]][0]))
    return probs


def guard_problems(rows_text, launch_text, binding_text):
    probs = ["cfbi_rows.cuh line %d: %s" % x for x in unguarded(rows_text, ROWS_PROFILE_NAMES)]
    probs += ["cfbi_launch.cu line %d: %s" % x for x in unguarded(launch_text, LAUNCH_PROFILE_NAMES)]
    probs += ["cfbi_binding.cpp line %d: %s" % (i + 1, l.strip()) for i, l in enumerate(binding_text.splitlines()) if LAUNCH_PROFILE_NAMES.search(l)]
    # the perturbation hook only inside its own nested region, which itself lies inside the profile region
    guards = guard_regions(rows_text)
    for i, l in enumerate(rows_text.splitlines()):
        if "reinterpret_cast<unsigned short*>(p.out)" in l and not {GUARD, PERTURB} <= guards[i]:
            probs.append("cfbi_rows.cuh line %d: the perturbation hook is outside %s inside %s" % (i + 1, PERTURB, GUARD))
    return probs


@pytest.fixture(scope="module")
def rows_text():
    return ROWS_SRC.read_text()


@pytest.fixture(scope="module")
def launch_text():
    return LAUNCH_SRC.read_text()


@pytest.fixture(scope="module")
def binding_text():
    return BINDING_SRC.read_text()


def mutate(text, old, new, count=1):
    assert text.count(old) == count, "the mutant needs %d occurrence(s) of %r, found %d" % (count, old, text.count(old))
    out = text.replace(old, new)
    assert out != text
    return out


# ---- source rules of the kernel ---------------------------------------------------------------------------------------------------------------------
def test_the_sources_have_the_shape_that_the_rules_read(rows_text):
    body = kernel_body(rows_text)
    first, last = row_loop(body)
    assert 0 < first < last < len(body) - 1
    assert len([1 for n, l, g in body if GLOBAL_STORE.search(l)]) >= 8        # the stores of the rules below are really found (out, out_f32, Eg, Xg, Rk, Rg, Ru, pos)
    assert len(guard_regions(rows_text)) == len(rows_text.splitlines())


def test_the_entry_stamp_is_the_first_statement_of_the_row_kernel(rows_text):
    assert entry_problems(rows_text) == []


def test_an_entry_stamp_after_the_plan_load_is_caught(rows_text):
    entry = "#ifdef CFBI_ROWS_PROFILE\n    const long long prof_entry = clock64();   // the profile starts at kernel entry: before the plan load and the rest of the prologue\n#endif\n"
    moved = mutate(rows_text, entry, "")
    moved = mutate(moved, "    const SeqPlan sp = rows_entry<DECODE>(p, n, dec_pos0);\n", "    const SeqPlan sp = rows_entry<DECODE>(p, n, dec_pos0);\n" + entry)
    assert any("not the entry stamp" in p for p in entry_problems(moved))
    unguarded_entry = mutate(rows_text, entry, "    const long long prof_entry = clock64();\n")
    assert any("not inside" in p for p in entry_problems(unguarded_entry))


def test_the_exit_stamp_follows_the_last_global_store(rows_text):
    assert exit_problems(rows_text) == []


def test_an_exit_stamp_before_a_store_is_caught(rows_text):
    store = "        p.pos[ROWS_POS_INDEX(POS_ACCESS_FUSED_STORE, sp.slot)] = (int)(((int64_t)sp.pos0 + sp.len) % BT);   // same value from every head CTA (sp.pos0 is the plan kernel's snapshot)\n"
    stamp = "        g_rows_prof[prof_last][n * H + h][8] = (unsigned)(clock64() - prof_entry);   // after the last global store of the kernel\n"
    moved = mutate(mutate(rows_text, store, ""), stamp, stamp + store)       # the fused pos store now follows the exit stamp
    probs = exit_problems(moved)
    assert any("comes after the exit stamp" in p for p in probs) and any("before the fused pos store" in p for p in probs)
    # an extra output store behind the exit stamp (outside the perturbation hook, which is the one store that may follow it)
    late = mutate(rows_text, "#ifdef CFBI_ROWS_PROFILE_PERTURB", "    if (threadIdx.x == 0) p.out_f32[0] = 0.f;\n#ifdef CFBI_ROWS_PROFILE_PERTURB")
    assert any("comes after the exit stamp" in p for p in exit_problems(late))
    unguarded_exit = mutate(rows_text, "#ifdef CFBI_ROWS_PROFILE\n    if (threadIdx.x == 0 && prof_last >= 0", "#if 1\n    if (threadIdx.x == 0 && prof_last >= 0")
    assert any("not inside" in p for p in exit_problems(unguarded_exit))


def test_the_stamps_keep_their_barriers(rows_text):
    assert order_problems(rows_text) == []


@pytest.mark.parametrize("name,old,new", [
    ("stamp 1 before its barrier", "        ROWS_SYNC(1);\n        ROWS_TS(1);\n", "        ROWS_TS(1);\n        ROWS_SYNC(1);\n"),
    ("stamp 2 dropped", "        ROWS_TS(2);\n", ""),
    ("stamp 5 moved behind a statement", "        ROWS_TS(5);\n", "        const int prof_dummy = 0; (void)prof_dummy;\n        ROWS_TS(5);\n"),
    ("stamp 7 before the last barrier", "        ROWS_SYNC_END();\n#ifdef CFBI_ROWS_PROFILE\n        ROWS_TS(7);\n",
     "#ifdef CFBI_ROWS_PROFILE\n        ROWS_TS(7);\n#endif\n        ROWS_SYNC_END();\n#ifdef CFBI_ROWS_PROFILE\n"),
])
def test_a_moved_stamp_is_caught(rows_text, name, old, new):
    assert order_problems(mutate(rows_text, old, new)), name


def test_a_stamp_0_behind_the_first_statement_of_the_row_is_caught(rows_text):
    moved = mutate(rows_text, "        ROWS_TS(0);\n#ifdef CFBI_ROWS_PROFILE\n        const int prof_pos = pos;", "#ifdef CFBI_ROWS_PROFILE\n        const int prof_pos = pos;")
    moved = mutate(moved, "        const int64_t tok = (int64_t)sp.tok0 + i;\n", "        const int64_t tok = (int64_t)sp.tok0 + i;\n        ROWS_TS(0);\n")
    assert any("stamp 0 is not the first statement" in p for p in order_problems(moved))


def test_the_buffer_is_written_after_the_last_stamp_of_the_row(rows_text):
    assert buffer_problems(rows_text) == []


def test_a_buffer_write_inside_a_measured_phase_is_caught(rows_text):
    # the stamp-7 line moves behind the write block: the buffer is written inside the last measured phase
    mutant = mutate(rows_text, "#ifdef CFBI_ROWS_PROFILE\n        ROWS_TS(7);\n        prof_last = prof_pos;\n", "#ifdef CFBI_ROWS_PROFILE\n        prof_last = prof_pos;\n")
    mutant = mutate(mutant, "            r[9] = smid + 1;\n        }\n#endif\n", "            r[9] = smid + 1;\n        }\n        ROWS_TS(7);\n#endif\n")
    assert any("before the last stamp" in p for p in buffer_problems(mutant))
    early = mutate(rows_text, "        ROWS_TS(0);\n#ifdef CFBI_ROWS_PROFILE\n", "        ROWS_TS(0);\n#ifdef CFBI_ROWS_PROFILE\n        g_rows_prof[0][0][0] = 0;\n")
    assert any("before the last stamp" in p for p in buffer_problems(early))


def test_everything_of_the_profile_lies_inside_the_profile_guard(rows_text, launch_text, binding_text):
    assert guard_problems(rows_text, launch_text, binding_text) == []


@pytest.mark.parametrize("which,old,new", [
    ("entry stamp", "#ifdef CFBI_ROWS_PROFILE\n    const long long prof_entry", "#if 1\n    const long long prof_entry"),
    ("exit stamp", "#ifdef CFBI_ROWS_PROFILE\n    if (threadIdx.x == 0 && prof_last >= 0", "#if 1\n    if (threadIdx.x == 0 && prof_last >= 0"),
    ("buffer declaration", "#ifdef CFBI_ROWS_PROFILE\n// Row-engine profile", "#if 1\n// Row-engine profile"),
    ("perturbation hook", "#ifdef CFBI_ROWS_PROFILE_PERTURB", "#if 1"),
])
def test_an_unguarded_profile_line_in_the_row_engine_is_caught(rows_text, launch_text, binding_text, which, old, new):
    assert guard_problems(mutate(rows_text, old, new), launch_text, binding_text), which


def test_an_unguarded_access_in_the_launcher_or_the_binding_is_caught(rows_text, launch_text, binding_text):
    dims = "void rows_profile_dims(int* positions, int* ctas, int* stamps) {\n#ifdef CFBI_ROWS_PROFILE\n"
    assert guard_problems(rows_text, mutate(launch_text, dims, dims.replace("#ifdef CFBI_ROWS_PROFILE", "#if 1")), binding_text)
    assert guard_problems(rows_text, launch_text, binding_text + "\nstatic const int kPos = cfbi::ROWS_PROF_POS;\n")


def test_the_default_build_defines_no_profile_macro():
    """The profile macros come from a -D flag of a variant build only: no source defines them, no build flag of the default build carries them."""
    for f in list(CSRC.rglob("*")) + [ROOT / "setup.py", ROOT / "flash_kda" / "cfbi" / "_ext.py"]:
        if f.suffix in (".cu", ".cuh", ".h", ".cpp", ".py"):
            for ln, line in enumerate(f.read_text().splitlines(), 1):
                assert not re.match(r"\s*#\s*define\s+CFBI_ROWS_PROFILE", line), "%s:%d defines a profile macro" % (f.name, ln)
    ext = (ROOT / "flash_kda" / "cfbi" / "_ext.py").read_text()
    setup = (ROOT / "setup.py").read_text()
    assert "ROWS_PROFILE" not in ext and "ROWS_PROFILE" not in setup


# ---- the aggregation of the stamps ----------------------------------------------------------------------------------------------------------------
def make_slots(N, H, phases_of, prologue_of=lambda pos, cta: 100 + cta, exit_gap=30, sm_of=lambda pos, cta: cta % 4):
    """``slots[pos][cta]`` = 10 stamps: ts0 = prologue, ts1..ts7 = ts0 + the cumulative phase lengths, ts8 = ts7 + exit_gap, ts9 = SM id + 1."""
    slots = []
    for pos in range(rpr.POSITIONS):
        row = []
        for cta in range(N * H):
            ts = [prologue_of(pos, cta)]
            for d in phases_of(pos, cta):
                ts.append(ts[-1] + d)
            row.append(ts + [ts[7] + exit_gap, sm_of(pos, cta) + 1])
        slots.append(row)
    return slots


REGULAR = [10, 20, 30, 40, 50, 60, 70]
CHUNK_END = [10, 20, 30, 40, 50, 60, 700]


def two_kinds(pos, cta):
    return CHUNK_END if pos == rpr.CHUNK_END_POS else REGULAR


def test_a_cell_averages_each_group_over_the_same_rows():
    N, H = 2, 3
    cell = rpr.cell_from_slots(make_slots(N, H, two_kinds), N, H)
    assert cell["key"] == "rows-profile;N=2;H=3;" and cell["ctas"] == 6 and cell["rows_expected"] == 64 * 6 and cell["rows_recorded"] == 64 * 6
    reg, ce, allr = cell["rows"]["regular"], cell["rows"]["chunk_end"], cell["rows"]["all"]
    assert (reg["rows"], ce["rows"], allr["rows"]) == (63 * 6, 6, 64 * 6)
    assert [reg["phases"][k] for k in rpr.PHASES] == [float(x) for x in REGULAR]
    assert ce["phases"]["chunk_end_sync"] == 700.0 and ce["phases"]["prep"] == 10.0
    assert allr["phases"]["chunk_end_sync"] == pytest.approx((63 * 70 + 700) / 64.0)
    assert reg["row_total"] == sum(REGULAR) and ce["row_total"] == sum(CHUNK_END)
    for block in (reg, ce, allr):
        assert sum(block["phases"].values()) == pytest.approx(block["row_total"])
        assert sum(block["shares"].values()) == pytest.approx(1.0)
        assert block["kernel_total"] == pytest.approx(block["prologue"] + block["row_total"] + 30)   # exit stamp = last row stamp + 30; the prologue is ts0
        assert block["epilogue"] == 30.0
    assert reg["prologue"] == pytest.approx(100 + (N * H - 1) / 2.0)
    assert [b["pos"] for b in cell["per_position"]] == list(range(rpr.POSITIONS)) and cell["per_position"][63]["rows"] == 6
    assert cell["sms"] == [0, 1, 2, 3]


def test_the_phase_means_add_up_to_the_row_mean_also_when_the_positions_have_unequal_row_counts():
    """A mean of totals against a sum of per-phase means is only the same number over the same rows: rows without a stamp (SM id 0) are dropped from every column alike."""
    N, H = 3, 2
    slots = make_slots(N, H, lambda pos, cta: [1 + (pos * 7 + cta * 3 + k) % 13 for k in range(7)])
    for pos in (4, 9, 63):
        slots[pos][1] = [0] * rpr.STAMPS
    slots[20] = [[0] * rpr.STAMPS for _ in range(N * H)]
    cell = rpr.cell_from_slots(slots, N, H)
    assert cell["rows_recorded"] == 64 * 6 - 3 - 6 and cell["rows"]["chunk_end"]["rows"] == 5
    assert cell["per_position"][20] == {"rows": 0, "pos": 20}
    for g in rpr.GROUPS:
        b = cell["rows"][g]
        assert sum(b["phases"].values()) == pytest.approx(b["row_total"]) and b["rows"] > 0
        assert b["row_total_p50"] <= b["row_total_p95"] <= b["row_total_max"]


def test_a_slot_without_a_row_or_with_the_wrong_length_is_not_a_row():
    assert rpr.row_measures([0] * 10) is None
    assert rpr.row_measures([1] * 9) is None and rpr.row_measures([1] * 11) is None
    m = rpr.row_measures([5, 7, 11, 13, 17, 19, 23, 29, 31, 3])
    assert m["prologue"] == 5 and m["phases"] == [2, 4, 2, 4, 2, 4, 6] and m["row_total"] == 24 and m["epilogue"] == 2 and m["kernel_total"] == 31 and m["sm"] == 2


def test_the_buffer_shape_is_checked():
    with pytest.raises(ValueError):
        rpr.cell_from_slots(make_slots(1, 1, two_kinds)[:63], 1, 1)
    with pytest.raises(ValueError):
        rpr.cell_from_slots(make_slots(1, 2, two_kinds), 2, 2)
    extra = [row + [[0] * rpr.STAMPS] * 5 for row in make_slots(1, 2, two_kinds)]       # a buffer wider than N * H CTAs: the unused slots are ignored
    assert rpr.cell_from_slots(extra, 1, 2)["rows_recorded"] == 64 * 2


def test_cell_keys_end_with_a_delimiter():
    assert rpr.cell_key(1, 12) == "rows-profile;N=1;H=12;"
    assert not rpr.cell_key(16, 12).startswith(rpr.cell_key(1, 12)[:-5])   # "N=1" is not a prefix match of "N=16"
    assert rpr.cell_key(16, 12).startswith("rows-profile;N=16;")


def test_a_record_survives_a_round_trip(tmp_path):
    cell = rpr.cell_from_slots(make_slots(1, 2, two_kinds), 1, 2)
    meta = {"commit": "abc1234", "dirty": False, "gpu": "NVIDIA B200", "sm_count": 148}
    path = str(tmp_path / "profile.json")
    rpr.write_record(path, meta, [cell])
    got = rpr.load_record(path)
    assert got["schema"] == rpr.SCHEMA and got["meta"] == meta and got["cells"] == json.loads(json.dumps([cell]))
    assert got["_path"] == path


def test_the_tool_prints_a_line_per_group_and_reads_the_profile_build(tmp_path):
    """The command line tool imports without torch; its cell formatter reports every group of a cell."""
    sys.path.insert(0, str(ROOT / "benchmarks" / "micro"))
    try:
        import rows_profile as tool
    finally:
        sys.path.pop(0)
    cell = rpr.cell_from_slots(make_slots(1, 2, two_kinds), 1, 2)
    text = tool.format_cell(cell)
    assert text.count("\n") == 3 and "regular" in text and "chunk_end" in text and "all" in text and "128 of 128 rows recorded" in text
    assert tool.VARIANT == ("CFBI_ROWS_PROFILE=1",)
