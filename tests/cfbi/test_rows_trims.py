"""The trims of the row engine (``csrc/cfbi/cfbi_rows.cuh``): source pins with mutants, and model checks of the two rewrites that have some index algebra (plain Python, no GPU, no torch:
``PYTHONPATH=. pytest --noconftest tests/cfbi/test_rows_trims.py``).

The trims are schedule-only edits of ``rows_kernel``. They change no operand and no operation order of the program, so the end-to-end gate stays the bit comparison with ``csrc/cfbi/ref`` (Gate G,
Suite S: they need a GPU). What no comparison of results can show, and what this module pins as text (read the way a default build reads it, ``protocol_source``), is the schedule itself:

* entry, two-launch instantiation (``rows_kernel<false>``): every field of the plan entry that the row loop needs is loaded in front of the ``hrows == 0`` exit (an empty volatile asm that names
  them pins their loads there);
* entry, single-launch instantiation (``rows_kernel<true>``): the lines that the first row loads and that need only the slot's token start and the head are prefetched into L1 in front of
  ``rows_entry`` (in front of the arrival atomic), for a non-empty window only, at the addresses that the row's own loads use; a prefetch holds no register across the entry (the loaded
  values would, and the single-launch instantiation then spills). Nothing that depends on the fill of the open chunk, nor any access to ``pos``, comes before the entry;
* order of the loads of the first row: the tokens of the row, then the cached rows, then the L2 prefetch, and the prefetch loop divides nothing and is not unrolled over the planes (five
  copies of it hold registers that the row loop does not have: the plan-path instantiation spilled on Blackwell with them);
* the B and D epilogues pick the accumulator of tile row ``slot`` / ``slot ^ 8`` with selects, not with a run-time index (which puts the arrays into local memory);
* the barrier that ends a row is skipped on the last row and only there.

Every pin is shown to FAIL on a copy of the source with one edit (a pin that cannot fail proves nothing) and to hold under harmless edits. The model checks restate the index algebra of the prefetch
loop and of the selects in plain Python and compare the old and the new form on every input; the pins tie the models to the source.
"""
import re

import pytest

import protocol_source as ps
from test_rows_profile import mutate

ROWS = "cfbi_rows.cuh"
KERNEL = "__global__ void __launch_bounds__ ( NTHREADS , CFBI_ROWS_MINB ) rows_kernel ("
EXIT = ps.norm("if (nrows == 0) return;")
ENTRY = ps.norm("const SeqPlan sp = rows_entry<DECODE>(p, n, dec_pos0);")
PIN = ps.norm('if constexpr (!DECODE) asm volatile("" ::"r"(sp.tok0), "r"(sp.len), "r"(sp.pos0), "r"(sp.slot));')
NROWS = ps.norm("const int nrows = sp.hrows;")
HOIST_OPEN = ps.norm("if constexpr (DECODE) {")
HOIST_WINDOW = ps.norm("const SeqWindow w0 = seq_window(p.cu_seqlens, p.fixed_len, p.slot_ids, n);")
HOIST_CALL = ps.norm("if (w0.len != 0) rows_prefetch_tokens(p.q, p.k, p.g, p.beta, p.v, (int64_t)w0.tok0, h, H, tid, L.lane);")
OWN_TOKENS = ps.norm("const RowTokens tk = rows_load_tokens(p.q, p.k, p.g, p.beta, p.v, tok, h, H, tid, L.lane);")
LOAD_HELPER = ps.norm("""
    const int64_t off = (tok * H + h) * HEAD_K;
    RowTokens t;
    t.kq = make_uint4(0u, 0u, 0u, 0u);
    t.g = 0.f;
    t.beta = 0.f;
    t.v = 0.f;
    if (tid < 32) t.kq = *reinterpret_cast<const uint4*>((lane >= 16 ? q : k) + off + 8 * (lane & 15));
    if (tid < HEAD_K) t.g = bf16_to_f32(g[off + tid]);
    if (tid == HEAD_K) t.beta = bf16_to_f32(beta[tok * H + h]);
    if (tid < HEAD_V) t.v = bf16_to_f32(v[off + tid]);
    return t;
""")
PREFETCH_HELPER = ps.norm("""
    const int64_t off = (tok * H + h) * HEAD_K;
    if (tid < 32) prefetch_l1((lane >= 16 ? q : k) + off + 8 * (lane & 15));
    if (tid < HEAD_K) prefetch_l1(g + off + tid);
    if (tid == HEAD_K) prefetch_l1(beta + tok * H + h);
    if (tid < HEAD_V) prefetch_l1(v + off + tid);
""")
XPRE_LOAD = ps.norm("xpre[jj] = (b + jj < pos) ? Xg[(b + jj) * HEAD_K + col] : 0.f;")
PREFETCH = "prefetch_state ("
END_BARRIER = ps.norm("if (i + 1 < nrows) ROWS_SYNC_END();")
SELECTS_B = [ps.norm(s) for s in ("const float bk = slot_hi ? big[2 + e] : big[e], sk = slot_hi ? sml[2 + e] : sml[e];", "const float bq = slot_hi ? big[e] : big[2 + e], sq = slot_hi ? sml[e] : sml[2 + e];",
                                  "S.akk[c] = fmul(S.beta, fadd(bk, sk));", "S.aqk[c] = fadd(bq, sq);")]
SELECTS_D = [ps.norm(s) for s in ("const float b1 = slot_hi ? big[tl][2 + e] : big[tl][e], s1 = slot_hi ? sml[tl][2 + e] : sml[tl][e];",
                                  "const float b2 = slot_hi ? big[tl][e] : big[tl][2 + e], s2 = slot_hi ? sml[tl][e] : sml[tl][2 + e];", "S.p1[c] = fadd(b1, s1);", "S.p2[c] = fadd(b2, s2);")]
RUNTIME_INDEX = [ps.norm(s) for s in ("big[2 * slot_hi + e]", "sml[2 * slot_hi + e]", "big[2 * (1 - slot_hi) + e]", "sml[2 * (1 - slot_hi) + e]", "big[tl][2 * slot_hi + e]", "sml[tl][2 * slot_hi + e]",
                                      "big[tl][2 * (1 - slot_hi) + e]", "sml[tl][2 * (1 - slot_hi) + e]")]


@pytest.fixture(scope="module")
def real():
    return ps.load()


def kernel_text(texts):
    """The tokens of the body of ``rows_kernel`` as the default build reads them, single-spaced (hooks of the variants gone, test-macro branches resolved)."""
    toks = ps.expand_default_hooks(ps.tokens(ps.default_view(texts)[ROWS]))
    body = ps.function_body(toks, KERNEL)
    assert body is not None, "rows_kernel is not found"
    return " ".join(body)


def before(text, a, b):
    """True when the snippet ``a`` is in ``text`` and comes before the first ``b`` that follows it."""
    i = text.find(a)
    return i >= 0 and text.find(b, i) > i


# ---- the pins ----------------------------------------------------------------------------------------------------------------------------------------------------
def entry_problems(texts):
    s = kernel_text(texts)
    probs = []
    for name, snippet in (("the entry call", ENTRY), ("the plan-entry pin", PIN), ("the row count", NROWS), ("the exit", EXIT)):
        if s.count(snippet) != 1:
            probs.append("entry: %s `%s` is not in rows_kernel exactly once" % (name, snippet))
    if probs:
        return probs
    order = [s.index(x) for x in (ENTRY, PIN, NROWS, EXIT)]
    if order != sorted(order):
        probs.append("entry: the plan entry is not loaded and pinned (the volatile asm that names tok0, len, pos0 and slot) between the entry call and the exit test, in front of the exit")
    return probs


def helper_body(texts, head):
    toks = ps.expand_default_hooks(ps.tokens(ps.default_view(texts)[ROWS]))
    body = ps.function_body(toks, head)
    return None if body is None else " ".join(body)


def hoist_problems(texts):
    s = kernel_text(texts)
    probs = []
    for name, snippet in (("the window of the hoist", HOIST_WINDOW), ("the guarded prefetch", HOIST_CALL), ("the own token loads of the row", OWN_TOKENS)):
        if s.count(snippet) != 1:
            probs.append("hoist: %s `%s` is not in rows_kernel exactly once" % (name, snippet))
    if probs or s.count(ENTRY) != 1:
        return probs or ["hoist: the entry call is missing"]
    i_open, i_window, i_call, i_entry = s.index(HOIST_OPEN), s.index(HOIST_WINDOW), s.index(HOIST_CALL), s.index(ENTRY)
    if not (i_open < i_window < i_call < i_entry):
        probs.append("hoist: the prefetch of the first row's lines is not issued in front of the entry (in front of the arrival atomic) in a single-launch branch")
        return probs
    block = s[i_open:i_entry]
    for name in ("Xg", "Rg", "Ru", "Rk", "Rw", "Rut", "R", "Eg", "xpre", "upre", "gprev", "gref_pre", "prefetch_state", "pos", "atomicAdd", "__syncthreads", "ROWS_SYNC", "rows_load_tokens", "tk"):
        if re.search(r"(?<![\w.])%s(?![\w])" % re.escape(name), block):
            probs.append("hoist: the hoisted block uses `%s`: it may only prefetch what needs the slot's token start and the head (no state, no fill of the open chunk, and no loaded value held across the entry)" % name)
    if re.search(r"\bp \. pos\b", block):
        probs.append("hoist: the hoisted block reads or writes the pos word: the arrival atomic is its only reader")
    loads, prefetches = helper_body(texts, "__device__ __forceinline__ RowTokens rows_load_tokens ("), helper_body(texts, "__device__ __forceinline__ void rows_prefetch_tokens (")
    if loads != LOAD_HELPER:
        probs.append("hoist: rows_load_tokens is not the pinned set of loads of a row's tokens: %s" % loads)
    if prefetches != PREFETCH_HELPER:
        probs.append("hoist: rows_prefetch_tokens does not prefetch the lines that rows_load_tokens loads, at the same addresses: %s" % prefetches)
    return probs


PLANE_LOOP_PRAGMA = "#pragma unroll 1\n            for (int pl = 0; pl < NPLANES; ++pl)"


def prefetch_problems(texts):
    s = kernel_text(texts)
    probs = []
    if PLANE_LOOP_PRAGMA not in texts[ROWS]:
        probs.append("prefetch: the loop over the planes of the prefetch is not `#pragma unroll 1` (five unrolled copies of the loop hold registers that the row loop does not have)")
    if s.count(PREFETCH) != 3:
        return ["prefetch: the row loop has %d L2 prefetches, expected 3 (two lines of E and the loop over the lines of the cached rows)" % s.count(PREFETCH)]
    i_tokens, i_xpre, i_first_prefetch = s.index(OWN_TOKENS), s.index(XPRE_LOAD), s.index(PREFETCH)
    if not i_tokens < i_xpre < i_first_prefetch:
        probs.append("prefetch: the loads of the row (its tokens, then the cached rows) do not come in front of the L2 prefetch")
    loop = s[s.index("const int lines = pos * 4 ;"):] if "const int lines = pos * 4 ;" in s else ""
    end = loop.find("} else if ( tid < HEAD_V )")
    body = loop[:end] if end >= 0 else loop
    if not body:
        probs.append("prefetch: the prefetch loop of the cached rows is not found")
    elif re.search(r"[/%]", body):
        probs.append("prefetch: the prefetch loop divides (a 32-bit division per prefetched line)")
    return probs


def select_problems(texts):
    s = kernel_text(texts)
    probs = []
    for token in RUNTIME_INDEX:
        if token in s:
            probs.append("selects: `%s`: a run-time index into the accumulators of the epilogue (they go to local memory)" % token)
    for name, snippets in (("B", SELECTS_B), ("D", SELECTS_D)):
        found = [s.find(x) for x in snippets]
        if min(found) < 0 or found != sorted(found):
            probs.append("selects: the %s epilogue is not the four statements that pick the accumulator of tile row slot / slot ^ 8 with selects, in order" % name)
    return probs


def barrier_problems(texts):
    s = kernel_text(texts)
    probs = []
    if s.count(END_BARRIER) != 1:
        probs.append("barrier: the barrier that ends a row is not `%s` exactly once" % END_BARRIER)
    if len(re.findall(r"ROWS_SYNC_END", s)) != 1:
        probs.append("barrier: rows_kernel has %d barriers that end a row, expected one" % len(re.findall(r"ROWS_SYNC_END", s)))
    guarded = re.findall(r"if \( [^)]*\) ROWS_SYNC \(", s)
    if guarded:
        probs.append("barrier: a numbered barrier of the row is guarded by a condition on its own: %s" % guarded)
    return probs


ALL_PINS = {"entry": entry_problems, "hoist": hoist_problems, "prefetch": prefetch_problems, "selects": select_problems, "barrier": barrier_problems}


@pytest.mark.parametrize("name", sorted(ALL_PINS))
def test_the_real_source_satisfies_the_pin(real, name):
    assert ALL_PINS[name](real) == []


def mutant(real, old, new):
    return dict(real, **{ROWS: mutate(real[ROWS], old, new)})


PIN_LINE = '    if constexpr (!DECODE) asm volatile("" ::"r"(sp.tok0), "r"(sp.len), "r"(sp.pos0), "r"(sp.slot));\n'
MUTANTS = [
    # (pin, name, old, new, expected text of the problem)
    ("entry", "no pin", PIN_LINE, "", "entry"),
    ("entry", "the pin behind the exit", PIN_LINE + "    const int nrows = sp.hrows;\n    if (nrows == 0) return;\n", "    const int nrows = sp.hrows;\n    if (nrows == 0) return;\n" + PIN_LINE, "entry"),
    ("entry", "the pin for the other instantiation", "    if constexpr (!DECODE) asm volatile", "    if constexpr (DECODE) asm volatile", "entry"),
    ("entry", "a field left out of the pin", '"r"(sp.tok0), "r"(sp.len), "r"(sp.pos0), "r"(sp.slot));', '"r"(sp.tok0), "r"(sp.len), "r"(sp.pos0));', "entry"),
    ("hoist", "no hoist", "        if (w0.len != 0) rows_prefetch_tokens(p.q, p.k, p.g, p.beta, p.v, (int64_t)w0.tok0, h, H, tid, L.lane);   // uniform over the CTA: the entry returns for an empty window\n", "", "hoist"),
    ("hoist", "the prefetch of an empty window", "        if (w0.len != 0) rows_prefetch_tokens(", "        rows_prefetch_tokens(", "hoist"),
    ("hoist", "the loaded values held across the entry", "        if (w0.len != 0) rows_prefetch_tokens(p.q, p.k, p.g, p.beta, p.v, (int64_t)w0.tok0, h, H, tid, L.lane);",
     "        if (w0.len != 0) tk0 = rows_load_tokens(p.q, p.k, p.g, p.beta, p.v, (int64_t)w0.tok0, h, H, tid, L.lane);", "hoist"),
    ("hoist", "a prefetch of another line (g)", "    if (tid < HEAD_K) prefetch_l1(g + off + tid);", "    if (tid < HEAD_K) prefetch_l1(g + off);", "hoist"),
    ("hoist", "a prefetch of another token (beta)", "    if (tid == HEAD_K) prefetch_l1(beta + tok * H + h);", "    if (tid == HEAD_K) prefetch_l1(beta + tok * H);", "hoist"),
    ("hoist", "a changed load of the row (v)", "    if (tid < HEAD_V) t.v = bf16_to_f32(v[off + tid]);", "    if (tid < HEAD_V) t.v = bf16_to_f32(v[off + tid + 1]);", "hoist"),
    ("hoist", "a state load in front of the atomic", "        const SeqWindow w0 = seq_window(p.cu_seqlens, p.fixed_len, p.slot_ids, n);\n",
     "        const SeqWindow w0 = seq_window(p.cu_seqlens, p.fixed_len, p.slot_ids, n);\n        xpre[0] = Xg[0];\n", "hoist"),
    ("hoist", "a prefetch in front of the atomic", "        const SeqWindow w0 = seq_window(p.cu_seqlens, p.fixed_len, p.slot_ids, n);\n",
     "        const SeqWindow w0 = seq_window(p.cu_seqlens, p.fixed_len, p.slot_ids, n);\n        prefetch_state(p.E);\n", "hoist"),
    ("hoist", "a second read of pos", "        const SeqWindow w0 = seq_window(p.cu_seqlens, p.fixed_len, p.slot_ids, n);\n", "        const SeqWindow w0 = seq_window(p.cu_seqlens, p.fixed_len, p.slot_ids, n);\n        dec_pos0 = p.pos[w0.slot];\n", "hoist"),
    ("hoist", "the row's own loads dropped", "        const RowTokens tk = rows_load_tokens(p.q, p.k, p.g, p.beta, p.v, tok, h, H, tid, L.lane);\n", "        RowTokens tk{};\n", "hoist"),
    ("prefetch", "the plane loop unrolled", "#pragma unroll 1\n            for (int pl = 0; pl < NPLANES; ++pl)", "#pragma unroll\n            for (int pl = 0; pl < NPLANES; ++pl)", "prefetch"),
    ("prefetch", "the division back", "                for (int l = tid; l < lines; l += NTHREADS) prefetch_state(R + pl * ROW_PLANE_FLOATS + l * 32);",
     "                for (int l = tid; l < lines; l += NTHREADS) prefetch_state(R + (l / lines) * ROW_PLANE_FLOATS + l * 32);", "prefetch"),
    ("prefetch", "the prefetch in front of the cached-row loads", "        float gprev = 0.f;\n        if (i == 0) {\n", "        float gprev = 0.f;\n        if (i == 0) {\n            prefetch_state(Eg);\n            const int lines0 = 0;\n            (void)lines0;\n            prefetch_state(R);\n", "prefetch"),
    ("selects", "the run-time index back (B)", "                    const float bk = slot_hi ? big[2 + e] : big[e], sk = slot_hi ? sml[2 + e] : sml[e];\n",
     "                    const float bk = big[2 * slot_hi + e], sk = sml[2 * slot_hi + e];\n", "selects"),
    ("selects", "the run-time index back (D)", "                        const float b2 = slot_hi ? big[tl][e] : big[tl][2 + e], s2 = slot_hi ? sml[tl][e] : sml[tl][2 + e];\n",
     "                        const float b2 = big[tl][2 * (1 - slot_hi) + e], s2 = sml[tl][2 * (1 - slot_hi) + e];\n", "selects"),
    ("selects", "the arms swapped (B: ks and qs exchanged)", "const float bk = slot_hi ? big[2 + e] : big[e], sk = slot_hi ? sml[2 + e] : sml[e];", "const float bk = slot_hi ? big[e] : big[2 + e], sk = slot_hi ? sml[e] : sml[2 + e];", "selects"),
    ("selects", "the arms swapped (D: W and GQ exchanged)", "const float b1 = slot_hi ? big[tl][2 + e] : big[tl][e], s1 = slot_hi ? sml[tl][2 + e] : sml[tl][e];",
     "const float b1 = slot_hi ? big[tl][e] : big[tl][2 + e], s1 = slot_hi ? sml[tl][e] : sml[tl][2 + e];", "selects"),
    ("selects", "the big and the small accumulator mixed", "S.p2[c] = fadd(b2, s2);", "S.p2[c] = fadd(b2, s1);", "selects"),
    ("barrier", "no skip", "        if (i + 1 < nrows) ROWS_SYNC_END();\n", "        ROWS_SYNC_END();\n", "barrier"),
    ("barrier", "the skip one row early", "if (i + 1 < nrows) ROWS_SYNC_END();", "if (i + 2 < nrows) ROWS_SYNC_END();", "barrier"),
    ("barrier", "the skip on the wrong rows", "if (i + 1 < nrows) ROWS_SYNC_END();", "if (i > 0) ROWS_SYNC_END();", "barrier"),
    ("barrier", "the skip on a position", "if (i + 1 < nrows) ROWS_SYNC_END();", "if (pos != BT - 1) ROWS_SYNC_END();", "barrier"),
    ("barrier", "a second barrier that ends the row", "        if (i + 1 < nrows) ROWS_SYNC_END();\n", "        if (i + 1 < nrows) ROWS_SYNC_END();\n        ROWS_SYNC_END();\n", "barrier"),
    ("barrier", "a numbered barrier with a condition of its own", "        ROWS_SYNC(2);\n", "        if (warp < 7) ROWS_SYNC(2);\n", "barrier"),
]


@pytest.mark.parametrize("pin,name,old,new,expect", MUTANTS, ids=["%s: %s" % (m[0], m[1]) for m in MUTANTS])
def test_a_wrong_schedule_is_caught(real, pin, name, old, new, expect):
    probs = ALL_PINS[pin](mutant(real, old, new))
    assert any(expect in p for p in probs), (name, probs)


@pytest.mark.parametrize("name,old,new", [
    ("a comment", "    // DECODE = false: the whole plan entry in one global round trip.", "    // (a different comment) DECODE = false: the whole plan entry in one global round trip."),
    ("spacing of the pin", PIN_LINE, PIN_LINE.replace('"" ::', '""  ::').replace("(sp.len),", "( sp.len ),")),
    ("a blank line in front of the exit", "    if (nrows == 0) return;\n", "\n    if (nrows == 0) return;\n"),
    ("a delay hook of the variants", "    ROWS_DELAY(0);   // test hook", "    ROWS_DELAY(0);\n    ROWS_DELAY(0);   // test hook"),
])
def test_a_harmless_edit_keeps_every_pin(real, name, old, new):
    text = mutant(real, old, new)
    assert [p for check in ALL_PINS.values() for p in check(text)] == [], name


# ---- the models of the two rewrites that have index algebra ---------------------------------------------------------------------------------------------------------
def constant(real, name):
    """An integer constant of the sources (``constexpr int NAME = <number or product of numbers and constants>;``)."""
    for text in ps.default_view(real).values():
        m = re.search(r"constexpr\s+int\s+%s\s*=\s*([^;]+);" % name, text)
        if m:
            return eval(re.sub(r"[A-Za-z_]\w*", lambda t: str(constant(real, t.group(0))), m.group(1)), {"__builtins__": {}})
    raise KeyError(name)


def lines_of_the_old_loop(pos, nthreads, nplanes, plane_floats):
    """The loop before the trim: l = tid, tid + nthreads, ... < nplanes * lines; plane l // lines, line l % lines (a 32-bit division per prefetch)."""
    lines, out = pos * 4, []
    for tid in range(nthreads):
        l = tid
        while l < nplanes * lines:
            out.append((l // lines) * plane_floats + (l % lines) * 32)
            l += nthreads
    return out


def lines_of_the_new_loop(pos, nthreads, nplanes, plane_floats):
    """The loop after the trim, as the source writes it (``prefetch_loop_form``): no division."""
    lines, out = pos * 4, []
    for tid in range(nthreads):
        for pl in range(nplanes):
            l = tid
            while l < lines:
                out.append(pl * plane_floats + l * 32)
                l += nthreads
    return out


@pytest.mark.parametrize("pos", list(range(64)))
def test_the_new_prefetch_loop_touches_the_same_lines_exactly_once(real, pos):
    nthreads, nplanes, plane = constant(real, "NTHREADS"), constant(real, "NPLANES"), constant(real, "ROW_PLANE_FLOATS")
    old, new = lines_of_the_old_loop(pos, nthreads, nplanes, plane), lines_of_the_new_loop(pos, nthreads, nplanes, plane)
    assert sorted(old) == sorted(new)
    assert len(set(new)) == len(new) == nplanes * pos * 4, "every (plane, 128-byte line) of the cached rows exactly once"


def test_the_prefetch_comparison_can_fail(real):
    """A loop that skips the last plane is not the same set."""
    nthreads, nplanes, plane = constant(real, "NTHREADS"), constant(real, "NPLANES"), constant(real, "ROW_PLANE_FLOATS")
    pos, lines = 33, 33 * 4
    short = [pl * plane + l * 32 for pl in range(nplanes - 1) for l in range(lines)]
    assert sorted(short) != sorted(lines_of_the_old_loop(pos, nthreads, nplanes, plane))


@pytest.mark.parametrize("slot_hi", [0, 1])
@pytest.mark.parametrize("e", [0, 1])
def test_the_epilogue_selects_equal_the_indexed_accumulators(slot_hi, e):
    """Phase B / D epilogue: the tile row ``slot`` (rows g + 8 * slot_hi of the lanes that own it) holds ks | W, the row ``slot ^ 8`` holds qs | GQ. Before: big[2 * slot_hi + e] and
    big[2 * (1 - slot_hi) + e] (a run-time index, local memory); after: selects on slot_hi."""
    big = [100.0 + i for i in range(4)]
    sml = [200.0 + i for i in range(4)]
    for arr in (big, sml):
        assert (arr[2 + e] if slot_hi else arr[e]) == arr[2 * slot_hi + e]
        assert (arr[e] if slot_hi else arr[2 + e]) == arr[2 * (1 - slot_hi) + e]


def test_the_select_comparison_can_fail():
    """Swapping the two rows (ks <-> qs, W <-> GQ) is visible: the selects must not be written the other way round."""
    big = [100.0 + i for i in range(4)]
    for slot_hi in (0, 1):
        for e in (0, 1):
            assert (big[e] if slot_hi else big[2 + e]) != big[2 * slot_hi + e]


@pytest.mark.parametrize("nrows", [1, 2, 5, 64])
def test_every_thread_executes_the_same_number_of_end_of_row_barriers(nrows):
    """``if (i + 1 < nrows) ROWS_SYNC_END();`` depends on uniform values only (``i`` and ``nrows``): all 256 threads execute the same number of end-of-row barriers, ``nrows - 1``, so a decode
    row (``nrows == 1``) has none and a call of several rows keeps one between consecutive rows (the last row has no next row to protect)."""
    counts = {sum(1 for i in range(nrows) if i + 1 < nrows) for _tid in range(256)}
    assert counts == {nrows - 1}
