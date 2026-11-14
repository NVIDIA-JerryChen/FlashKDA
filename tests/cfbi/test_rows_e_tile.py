"""The staged state tile of the row engine (``csrc/cfbi/cfbi_rows.cuh``, ``cfbi_policy.h``, ``cfbi_launch.cu``): source pins with mutants, model checks of the index algebra and the bank
claims, and the budget (plain Python and the host C++ compiler, no GPU, no torch: ``PYTHONPATH=. pytest --noconftest tests/cfbi/test_rows_e_tile.py``).

The tile is a schedule-only edit of ``rows_kernel``: the state E (HEAD_V rows of HEAD_K floats per (sequence, head)) is staged ONCE per call in dynamic shared memory by 16-byte async copies
instead of being gathered from global memory in every row, every fragment holds the same bits as the global load it replaces, and no operand or operation of the program changes. The end-to-end
gate stays the bit comparison with ``csrc/cfbi/ref`` (Gate G, Suite S: they need a GPU). What no comparison of results can show, and what this module pins as text (read the way a default build
reads it, ``protocol_source``), is the protocol of the tile:

* ownership: warp w copies exactly the 16 value rows of E that its P1 / P2 tiles read and that its chunk-end tiles update (``8 * (w + 8 * tl) + gg``, ``tl < 2``, ``gg < 8``: the sets of the 8
  warps partition the 128 rows), every row of E is read and written by its owner warp only, and no other row is copied;
* the staging is issued once per call, in front of the row loop (inside the loop under ``i == 0`` the compiler keeps the addresses of the 16 copies live across the loop and the kernel spills),
  waited for by every thread in front of the barrier that ends phase C (``i == 0`` only), and the first read of the tile comes after that barrier: no ordering of this kind is observable by a
  comparison of results (the copy issued at the top of the row has always landed by phase D), so it is a source rule with mutants;
* phase D reads its B fragments from the tile, the chunk end reads the old values from it and writes the new ones to the tile AND to global memory (the state of the next call), and nothing else
  reads or writes E: no hoisted gather, no prefetch of E; the lane coordinates that the tile addresses are made of come from an opaque copy taken where they are used (otherwise the compiler computes
  the addresses once in front of the row loop and holds them across it: on sm_90a the single-launch instantiation then needs more stack than the base);
* the numbers (stride 132, 128 rows, 67,584 B, the budget of two CTAs per SM) are one set in the CUDA-free header, bound to the layout by ``static_assert``s; an over-budget tile fails the build;
* the launcher opts every optimised instantiation in once per device with checked return values (never remembers a failure), launches with the dynamic size of its own engine (the frozen
  reference engine has none) and the occupancy hook asks each instance with the dynamic size of its own launch.

Every pin is shown to FAIL on a copy of the source with one edit (a pin that cannot fail proves nothing) and to hold under harmless edits. The model checks restate the index algebra and the bank
claims in plain Python on the expressions that they READ from the source, so that the model and the kernel cannot drift apart.
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import protocol_source as ps
from test_rows_profile import mutate

ROOT = Path(__file__).resolve().parents[2]
CSRC = ROOT / "csrc" / "cfbi"
ROWS, LAUNCH, HEADER = "cfbi_rows.cuh", "cfbi_launch.cu", "cfbi_policy.h"
KERNEL = "__global__ void __launch_bounds__ ( NTHREADS , CFBI_ROWS_MINB ) rows_kernel ("
STAGE_HEAD = "__device__ __forceinline__ void rows_stage_e ("
PREPARE_HEAD = "cudaError_t rows_prepare ("


@pytest.fixture(scope="module")
def real():
    return ps.load()


def kernel_text(texts):
    """The tokens of the body of ``rows_kernel`` as the default build reads them, single-spaced."""
    toks = ps.expand_default_hooks(ps.tokens(ps.default_view(texts)[ROWS]))
    body = ps.function_body(toks, KERNEL)
    assert body is not None, "rows_kernel is not found"
    return " ".join(body)


def helper_text(texts, file, head):
    toks = ps.expand_default_hooks(ps.tokens(ps.default_view(texts)[file]))
    body = ps.function_body(toks, head)
    return None if body is None else " ".join(body)


def mutant(texts, file, old, new):
    return dict(texts, **{file: mutate(texts[file], old, new)})


def before(text, a, b):
    """True when the snippet ``a`` is in ``text`` and the first ``b`` after it exists."""
    i = text.find(a)
    return i >= 0 and text.find(b, i + len(a)) > i


# ---- the pins of the kernel ----------------------------------------------------------------------------------------------------------------------------------
STAGE_BODY = ps.norm("""
    for (int r = 0; r < 16; ++r) {
        const int v = 8 * (warp + 8 * (r >> 3)) + (r & 7);
        const int vs = v;
        cp_async_16(sE + v * ROWS_E_STRIDE + 4 * lane, Eg + vs * HEAD_K + 4 * lane);
    }
    cp_async_commit();
""")
STAGE_CALL = ps.norm("rows_stage_e(sE, Eg, warp, L.lane);")
ROW_LOOP = ps.norm("for (int i = 0; i < nrows; ++i) {")
WAIT = ps.norm("if (i == 0) cp_async_wait_all();")
BARRIER_C = ps.norm("ROWS_SYNC(4);")
DYN_SMEM = ps.norm("extern __shared__ __align__(128) unsigned char rows_dyn_smem[];")
TILE_BASE = ps.norm("float* const sE = reinterpret_cast<float*>(rows_dyn_smem);")
D_BASE = ps.norm("const float* const sEw = sE + (8 * warp + rows_opaque(g)) * ROWS_E_STRIDE + rows_opaque(t);")
OPAQUE_DECL = ps.norm("const int go = rows_opaque(g), to = rows_opaque(t);")
OPAQUE_HELPER = "__device__ __forceinline__ int rows_opaque ("
FRAGMENT_HELPERS = {"__device__ __forceinline__ int c_row_of (": ps.norm("return g + 8 * (i >> 1);"), "__device__ __forceinline__ int c_col_of (": ps.norm("return 2 * t + (i & 1);")}
D_READ = ps.norm("split_tf32(sEw[64 * tl * ROWS_E_STRIDE + 8 * ks + 4 * ii], bh[ii], bl[ii]);")
G_READ = ps.norm("eold[tl][i2] = sE[(8 * (warp + 8 * tl) + c_col_of(to, i2)) * ROWS_E_STRIDE + 16 * mt + c_row_of(go, i2)];")
G_INDEX = ps.norm("const int k = 16 * mt + c_row_of(go, ii), v = 8 * (warp + 8 * tl) + c_col_of(to, ii);")
G_WRITES = [ps.norm(s) for s in ("const float en = fadd(fmul(eold[tl][ii], S.gend[k]), fadd(big[tl][ii], sml[tl][ii]));", "sE[v * ROWS_E_STRIDE + k] = en;", "Eg[v * HEAD_K + k] = en;")]


def stage_problems(texts):
    body = helper_text(texts, ROWS, STAGE_HEAD)
    if body != STAGE_BODY:
        return ["stage: rows_stage_e is not the pinned copy of the 16 rows of a warp (8 * (warp + 8 * (r >> 3)) + (r & 7), one 16-byte copy per lane, one group): %s" % body]
    probs = []
    s = kernel_text(texts)
    if s.count(STAGE_CALL) != 1 or len(re.findall(r"\brows_stage_e\b", s)) != 1:
        probs.append("stage: rows_stage_e is not called exactly once in rows_kernel, as `%s`" % STAGE_CALL)
    elif not before(s, STAGE_CALL, ROW_LOOP):
        probs.append("stage: the staging is not issued in front of the row loop (inside it the compiler holds the addresses of the copies across the loop and the kernel spills)")
    return probs


def order_problems(texts):
    s = kernel_text(texts)
    probs = []
    if s.count(WAIT) != 1:
        return ["order: the wait of the copies is not `%s` exactly once" % WAIT]
    if s.count(BARRIER_C) != 1:
        return ["order: the barrier that ends phase C is not `%s` exactly once" % BARRIER_C]
    if not s.index(WAIT) < s.index(BARRIER_C):
        probs.append("order: the wait for the copies does not come in front of the barrier that ends phase C (the barrier publishes the copies of the other lanes: a wait behind it is no wait)")
    if not s.index(WAIT) + len(WAIT) + 1 == s.index(BARRIER_C):
        probs.append("order: something stands between the wait and the barrier that ends phase C")
    if s.index(BARRIER_C) > s.find("sEw [") or s.find("sEw [") < 0:
        probs.append("order: the tile is read in front of the barrier that follows the wait (phase D must read it after it)")
    if re.search(r"\bcp_async_(?:commit|16)\b", s):
        probs.append("order: an async copy is issued in rows_kernel itself: only rows_stage_e issues copies")
    return probs


def tile_problems(texts):
    s = kernel_text(texts)
    probs = []
    for name, snippet in (("the dynamic tile", DYN_SMEM), ("the tile base", TILE_BASE), ("the phase-D row base", D_BASE), ("the phase-D fragment read", D_READ), ("the chunk-end read", G_READ),
                          ("the chunk-end index", G_INDEX), ("the opaque lane coordinates of the chunk end", OPAQUE_DECL)):
        if s.count(snippet) != 1:
            probs.append("tile: %s `%s` is not in rows_kernel exactly once" % (name, snippet))
    for snippet in G_WRITES:
        if s.count(snippet) != 1:
            probs.append("tile: the chunk-end statement `%s` is not in rows_kernel exactly once" % snippet)
    if probs:
        return probs
    if [s.index(x) for x in G_WRITES] != sorted(s.index(x) for x in G_WRITES):
        probs.append("tile: the chunk end does not compute the new value, then write the tile, then global memory")
    uses = re.findall(r"\bEg\b", s)
    if len(uses) != 3:
        probs.append("tile: Eg is named %d times in rows_kernel, expected 3 (its definition, the staging call and the chunk-end store): no other access to the state in global memory" % len(uses))
    for token in ("eb", "prefetch_state ( eb"):
        if re.search(r"(?<![\w.])%s\b" % re.escape(token), s):
            probs.append("tile: `%s` is in rows_kernel: the gather or the prefetch of E that the tile replaces" % token)
    if not before(s, OPAQUE_DECL, "for ( int mt = 0 ; mt < 8 ; ++ mt )"):
        probs.append("tile: the opaque lane coordinates are not taken in front of the loop over the key tiles of the chunk end")
    if len(re.findall(r"\brows_opaque\b", s)) != 4:
        probs.append("tile: rows_opaque is named %d times in rows_kernel, expected 4 (the phase-D base twice, the chunk-end coordinates twice)" % len(re.findall(r"\brows_opaque\b", s)))
    if len(re.findall(r"\bsE\b", s)) != 5:
        probs.append("tile: sE is named %d times in rows_kernel, expected 5 (its definition, the staging call, the phase-D base, the chunk-end read and write)" % len(re.findall(r"\bsE\b", s)))
    return probs


def helper_problems(texts):
    """The helpers that the tile addresses are made of: an opaque copy that is opaque, and the fragment layout of the accumulator helpers of the primitives."""
    probs = []
    body = helper_text(texts, ROWS, OPAQUE_HELPER)
    if body != ps.norm('asm volatile("" : "+r"(x)); return x;'):
        probs.append("helpers: rows_opaque is not an empty volatile asm that takes and gives back its argument: %s" % body)
    for head, golden in FRAGMENT_HELPERS.items():
        got = helper_text(texts, ROWS, head)
        if got != golden:
            probs.append("helpers: %s is not the fragment layout of the accumulator (%s): %s" % (head, golden, got))
    return probs


def variants_problems(texts):
    """The two negative controls exist in the sources as test macros (not in the default build) and are what they say."""
    rows = texts[ROWS]
    probs = []
    shifted = "#ifdef CFBI_ROWS_NEG_E_ROW\n        const int vs = (v + 1) & (HEAD_V - 1);\n#else\n        const int vs = v;\n#endif\n"
    if shifted not in rows:
        probs.append("variants: the control that stages the rows of E shifted by one is not the pinned #ifdef CFBI_ROWS_NEG_E_ROW / #else block of rows_stage_e")
    if "CFBI_ROWS_NEG_E_" in ps.default_view(texts)[ROWS]:
        probs.append("variants: a CFBI_ROWS_NEG_E_ macro is named in the default view of the row engine")
    late = re.findall(r"#ifdef CFBI_ROWS_NEG_E_LATE\n\s*if \(i == 0\) rows_stage_e\(sE, Eg, warp, L\.lane\);[^\n]*\n#endif", rows)
    if len(late) != 1 or not before(rows, "ROWS_SYNC(4);", late[0]) or not before(rows, late[0], "sEw[64"):
        probs.append("variants: the late-issue control is not one `#ifdef CFBI_ROWS_NEG_E_LATE` block with the staging between the barrier that ends phase C and the first read of the tile")
    if len(re.findall(r"#ifndef CFBI_ROWS_NEG_E_LATE\n[^\n]*\n[^\n]*\n[^\n]*rows_stage_e\(sE, Eg, warp, L\.lane\);\n#endif", rows)) != 1:
        probs.append("variants: the staging in front of the row loop is not guarded by `#ifndef CFBI_ROWS_NEG_E_LATE` (the late-issue control would then stage twice)")
    return probs


ALL_PINS = {"stage": stage_problems, "order": order_problems, "tile": tile_problems, "helpers": helper_problems, "variants": variants_problems}


@pytest.mark.parametrize("name", sorted(ALL_PINS))
def test_the_real_source_satisfies_the_pin(real, name):
    assert ALL_PINS[name](real) == []


KERNEL_SOURCE_LINES = {
    "the staging row formula": ("const int v = 8 * (warp + 8 * (r >> 3)) + (r & 7);", "const int v = 8 * (warp + 8 * (r >> 3)) + (r & 3);", "stage"),
    "15 rows copied": ("for (int r = 0; r < 16; ++r) {\n        const int v = 8", "for (int r = 0; r < 15; ++r) {\n        const int v = 8", "stage"),
    "the rows of E shifted by one in the default build": ("#else\n        const int vs = v;\n#endif", "#else\n        const int vs = (v + 1) & (HEAD_V - 1);\n#endif", "stage"),
    "the destination row stride": ("cp_async_16(sE + v * ROWS_E_STRIDE + 4 * lane", "cp_async_16(sE + v * HEAD_K + 4 * lane", "stage"),
    "no commit": ("    cp_async_commit();\n}", "}", "stage"),
    "the staging inside the row loop": ("    rows_stage_e(sE, Eg, warp, L.lane);\n#endif\n", "#endif\n", "stage"),
    "no wait": ("        if (i == 0) cp_async_wait_all();", "", "order"),
    "the wait behind the barrier": ("        if (i == 0) cp_async_wait_all();   // this thread's copies of the E tile have landed; the barrier below publishes them (a row of E was copied by all 32 lanes of its warp)\n        ROWS_SYNC(4);",
                                    "        ROWS_SYNC(4);\n        if (i == 0) cp_async_wait_all();", "order"),
    "a wait for every row": ("        if (i == 0) cp_async_wait_all();", "        cp_async_wait_all();", "order"),
    "phase D reads global memory again": ("split_tf32(sEw[64 * tl * ROWS_E_STRIDE + 8 * ks + 4 * ii], bh[ii], bl[ii]);", "split_tf32(Eg[(8 * (warp + 8 * tl) + g) * HEAD_K + 8 * ks + t + 4 * ii], bh[ii], bl[ii]);", "tile"),
    "phase D reads with the wrong stride": ("sEw[64 * tl * ROWS_E_STRIDE + 8 * ks + 4 * ii]", "sEw[64 * tl * HEAD_K + 8 * ks + 4 * ii]", "tile"),
    "the chunk end reads global memory": ("eold[tl][i2] = sE[(8 * (warp + 8 * tl) + c_col_of(to, i2)) * ROWS_E_STRIDE + 16 * mt + c_row_of(go, i2)];",
                                          "eold[tl][i2] = Eg[(8 * (warp + 8 * tl) + c_col_of(to, i2)) * HEAD_K + 16 * mt + c_row_of(go, i2)];", "tile"),
    "the phase-D base from the plain lane coordinates": ("sE + (8 * warp + rows_opaque(g)) * ROWS_E_STRIDE + rows_opaque(t);", "sE + (8 * warp + g) * ROWS_E_STRIDE + t;", "tile"),
    "the chunk-end coordinates not opaque": ("const int go = rows_opaque(g), to = rows_opaque(t);", "const int go = g, to = t;", "tile"),
    "the chunk-end coordinates taken after the loop head": ("            const int go = rows_opaque(g), to = rows_opaque(t);   // the lane coordinates of the fragment elements below (see rows_opaque)\n", "", "tile"),
    "an opaque copy that is not opaque": ('    asm volatile("" : "+r"(x));\n    return x;', "    return x;", "helpers"),
    "a fragment helper with another layout": ("int c_col_of(int t, int i) { return 2 * t + (i & 1); }", "int c_col_of(int t, int i) { return 2 * t + (i >> 1); }", "helpers"),
    "the chunk end writes the tile only": ("                        Eg[v * HEAD_K + k] = en;          // and the next call from global memory\n", "", "tile"),
    "the chunk end writes global memory only": ("                        sE[v * ROWS_E_STRIDE + k] = en;   // the next row of this call reads E' from the tile\n", "", "tile"),
    "the chunk end writes global memory first": ("                        sE[v * ROWS_E_STRIDE + k] = en;   // the next row of this call reads E' from the tile\n                        Eg[v * HEAD_K + k] = en;          // and the next call from global memory\n",
                                                 "                        Eg[v * HEAD_K + k] = en;          // and the next call from global memory\n                        sE[v * ROWS_E_STRIDE + k] = en;   // the next row of this call reads E' from the tile\n", "tile"),
    "a prefetch of E returns": ("    int pos = sp.pos0;\n", "    int pos = sp.pos0;\n    prefetch_state(reinterpret_cast<const char*>(Eg));\n", "tile"),
    "the control block of the late issue lost its guard": ("#ifndef CFBI_ROWS_NEG_E_LATE\n    // E goes", "#ifdef CFBI_ROWS_NEG_E_LATE\n    // E goes", "variants"),
    "a default build that shifts the control": ("const int vs = (v + 1) & (HEAD_V - 1);", "const int vs = (v + 2) & (HEAD_V - 1);", "variants"),
}


@pytest.mark.parametrize("name", sorted(KERNEL_SOURCE_LINES))
def test_a_mutant_of_the_kernel_fails_its_pin(real, name):
    old, new, pin = KERNEL_SOURCE_LINES[name]
    text = real[ROWS]
    assert text.count(old) == 1, "the mutant needs one occurrence of %r, found %d" % (old, text.count(old))
    assert ALL_PINS[pin](mutant(real, ROWS, old, new)), name


HARMLESS = {
    "a blank line in the helper": ("    cp_async_commit();\n}", "    cp_async_commit();\n\n}"),
    "a comment after the wait": ("        if (i == 0) cp_async_wait_all();", "        if (i == 0) cp_async_wait_all();   // harmless"),
    "spaces in the tile index": ("sE[v * ROWS_E_STRIDE + k] = en;", "sE[ v*ROWS_E_STRIDE + k ] = en;"),
    "a comment between statements of the kernel": ("    if (nrows == 0) return;\n", "    if (nrows == 0) return;   // harmless\n"),
}


@pytest.mark.parametrize("name", sorted(HARMLESS))
def test_a_harmless_edit_keeps_every_pin(real, name):
    old, new = HARMLESS[name]
    edited = mutant(real, ROWS, old, new)
    assert [p for pin in ALL_PINS.values() for p in pin(edited)] == [], name


# ---- the model: the index algebra and the banks, on the expressions that the source holds ---------------------------------------------------------------------------
def header_ints():
    """The integer constants of the header that are a number or a product of numbers (the others are expressions of these)."""
    return {m.group(1): eval(m.group(2), {"__builtins__": {}}) for m in re.finditer(r"constexpr\s+int\s+(\w+)\s*=\s*(\d+(?:\s*\*\s*\d+)*)\s*;", (CSRC / HEADER).read_text())}


def evaluate(expr, **names):
    assert re.fullmatch(r"[\w\s\-+*()<>&]+", expr), expr
    return eval(expr, {"__builtins__": {}}, dict(names, rows_opaque=lambda x: x))                      # the opaque copy is the value itself


def expression(text, pattern):
    m = re.search(pattern, text)
    assert m, pattern
    return m.group(1).strip()


def fragment_names(expr):
    """The accumulator fragment helpers of an expression of the chunk end as names (``c_row_of(go, ii)`` -> ``c_row``): the model gives them the values of the fragment layout."""
    return re.sub(r"c_(row|col)_of \( (?:go|to) , (?:i2|ii) \)", r"c_\1", expr)


@pytest.fixture(scope="module")
def formulas(real):
    s = kernel_text(real)
    stage = helper_text(real, ROWS, STAGE_HEAD)
    return {
        "stage_row": expression(stage, r"const int v = ([^;]+) ;"),
        "d_base": expression(s, r"const float \* const sEw = sE \+ ([^;]+) ;"),
        "d_read": expression(s, r"split_tf32 \( sEw \[ ([^\]]+) \]"),
        "g_read": fragment_names(expression(s, r"eold \[ tl \] \[ i2 \] = sE \[ ([^\]]+) \] ;")),
        "g_write": expression(s, r"sE \[ ([^\]]+) \] = en ;"),
        "g_global": expression(s, r"Eg \[ ([^\]]+) \] = en ;"),
        "g_k": fragment_names(expression(s, r"const int k = (16 \* mt \+ c_row_of \( go , ii \)) , v = ")),
        "g_v": fragment_names(expression(s, r"const int k = 16 \* mt \+ c_row_of \( go , ii \) , v = ([^;]+) ;")),
    }


def c_row(g, i):
    return g + 8 * (i >> 1)


def c_col(t, i):
    return 2 * t + (i & 1)


def lane_names(lane):
    return {"g": lane >> 2, "t": lane & 3}


def chunk_end(formulas, w, tl, mt, i, lane, stride):
    """``(v, k, tile index read, tile index written, global index written)`` of the chunk-end update of fragment element ``i`` of a lane, from the expressions of the source."""
    n = lane_names(lane)
    names = {"warp": w, "tl": tl, "mt": mt, "c_row": c_row(n["g"], i), "c_col": c_col(n["t"], i), "ROWS_E_STRIDE": stride, "HEAD_K": 128}
    k, v = evaluate(formulas["g_k"], **names), evaluate(formulas["g_v"], **names)
    return v, k, evaluate(formulas["g_read"], **names), evaluate(formulas["g_write"], v=v, k=k, **names), evaluate(formulas["g_global"], v=v, k=k, **names)


def test_the_rows_of_the_warps_partition_e_and_are_the_rows_that_their_tiles_use(formulas):
    owned = {w: sorted(evaluate(formulas["stage_row"], warp=w, r=r) for r in range(16)) for w in range(8)}
    assert sorted(v for rows in owned.values() for v in rows) == list(range(128))                    # a partition of the 128 value rows
    stride = header_ints()["ROWS_E_STRIDE"]
    for w in range(8):
        read_d, written_g = set(), set()
        for lane in range(32):
            n = lane_names(lane)
            row0 = evaluate(formulas["d_base"], warp=w, ROWS_E_STRIDE=stride, **n)
            for tl in range(2):
                read_d.add((row0 + evaluate(formulas["d_read"], tl=tl, ks=0, ii=0, ROWS_E_STRIDE=stride)) // stride)          # the tile row that phase D reads: 8 warp + g + 64 tl
                for mt in range(8):
                    for i in range(4):
                        written_g.add(chunk_end(formulas, w, tl, mt, i, lane, stride)[0])
        assert read_d == set(owned[w]) == written_g or (read_d <= set(owned[w]) and written_g == set(owned[w])), (w, sorted(read_d)[:4])
        assert set(owned[w]) == {r for r in range(128) if 8 * w <= r % 64 < 8 * w + 8}                # warp w owns value rows 8 w + gg and 64 + 8 w + gg


def test_the_fragment_helpers_of_the_tile_are_the_accumulator_layout_of_the_primitives(real):
    """``c_row_of`` / ``c_col_of`` (lane coordinates given) equal ``c_row`` / ``c_col`` of the primitives for every lane and element."""
    prim = ps.tokens(ps.default_view(real)["cfbi_primitives.cuh"])
    row, col = ps.function_body(prim, "int c_row ("), ps.function_body(prim, "int c_col (")
    assert row and col
    row_expr = " ".join(row).replace("return ", "").replace(" ;", "").replace("L . g", "g")
    col_expr = " ".join(col).replace("return ", "").replace(" ;", "").replace("L . t", "t")
    for lane in range(32):
        n = lane_names(lane)
        for i in range(4):
            assert evaluate(row_expr, g=n["g"], i=i) == c_row(n["g"], i) and evaluate(col_expr, t=n["t"], i=i) == c_col(n["t"], i), (lane, i)


def test_the_tile_fragments_are_the_elements_that_the_global_gather_read(formulas):
    """The tile index of phase D and of the chunk end is the global index (value row * HEAD_K + key column) with the row stride of the tile instead of HEAD_K, and the chunk end reads and
    writes the tile element of the global element that it updates."""
    stride = header_ints()["ROWS_E_STRIDE"]
    for w in range(8):
        for lane in range(32):
            n = lane_names(lane)
            for tl in range(2):
                for ks in range(16):
                    for ii in range(2):
                        got = evaluate(formulas["d_base"], warp=w, ROWS_E_STRIDE=stride, **n) + evaluate(formulas["d_read"], tl=tl, ks=ks, ii=ii, ROWS_E_STRIDE=stride)
                        assert divmod(got, stride) == (8 * (w + 8 * tl) + n["g"], 8 * ks + n["t"] + 4 * ii), (w, lane, tl, ks, ii)    # the old gather: Eg[(8 (warp + 8 tl) + g) * HEAD_K + 8 ks + t + 4 ii]
                for mt in range(8):
                    for i in range(4):
                        v, k, tile_read, tile_written, global_written = chunk_end(formulas, w, tl, mt, i, lane, stride)
                        assert tile_read == tile_written == v * stride + k and global_written == v * 128 + k and 0 <= k < 128 and 0 <= v < 128, (w, lane, tl, mt, i)


def banks(addresses):
    return sorted(a % 32 for a in addresses)


def test_the_fragment_reads_and_the_chunk_end_updates_are_free_of_bank_conflicts(formulas):
    stride = header_ints()["ROWS_E_STRIDE"]
    for w in range(8):
        for tl in range(2):
            for ks in range(16):
                for ii in range(2):
                    addrs = [evaluate(formulas["d_base"], warp=w, ROWS_E_STRIDE=stride, **lane_names(lane)) + evaluate(formulas["d_read"], tl=tl, ks=ks, ii=ii, ROWS_E_STRIDE=stride) for lane in range(32)]
                    assert banks(addrs) == list(range(32)), (w, tl, ks, ii)                              # the 32 lanes of a warp read 32 different banks
            for mt in range(8):
                for i in range(4):
                    addrs = [chunk_end(formulas, w, tl, mt, i, lane, stride)[3] for lane in range(32)]
                    assert banks(addrs) == list(range(32)), (w, tl, mt, i)                               # and so do the 32 lanes of a chunk-end update
    assert stride % 32 == 4 and stride % 4 == 0                                                           # the claim that the static_assert of the kernel holds


def test_the_copies_are_whole_aligned_rows_and_the_tile_bytes_are_the_header_figure():
    h = header_ints()
    assert h["ROWS_E_STRIDE"] * 4 % 16 == 0                                                              # every row of the tile starts on a 16-byte boundary: the async copy needs it
    assert h["ROWS_E_TILE_FLOATS"] == 128 * h["ROWS_E_STRIDE"] and h["ROWS_DYNAMIC_SMEM_BYTES"] == 4 * h["ROWS_E_TILE_FLOATS"] == 67584
    assert 32 * 16 == 128 * 4                                                                              # one copy per lane (16 bytes): a warp instruction moves one whole row of E (512 bytes)


# ---- the budget -------------------------------------------------------------------------------------------------------------------------------------------------
def test_two_ctas_fit_the_shared_memory_of_an_sm_with_the_numbers_of_the_header():
    h = header_ints()
    need = h["ROWS_STATIC_SMEM_BYTES"] + h["ROWS_DYNAMIC_SMEM_BYTES"]                                       # ROWS_SMEM_NEED_BYTES is this sum in the header
    per_cta = need + h["ROWS_BUDGET_CTA_RESERVE_BYTES"]
    assert need == h["ROWS_STATIC_SMEM_BYTES"] + h["ROWS_DYNAMIC_SMEM_BYTES"] == 8192 + 67584 == 75776
    assert h["ROWS_BUDGET_CTAS_PER_SM"] * per_cta <= h["ROWS_BUDGET_SM_SHARED_BYTES"]
    assert 3 * per_cta <= 233472 and 4 * per_cta > 233472                                                 # a third CTA would fit by shared memory (the registers bound two), a fourth would not


def host_compiler():
    for name in ("g++", "c++"):
        found = shutil.which(name)
        if found:
            return found
    return None


def compile_header(tmp, edits):
    """Compile an empty translation unit that includes the header of a tree whose header had ``edits`` ({old: new}) with the host compiler: ``(exit status, text)``."""
    cxx = host_compiler()
    if cxx is None:
        pytest.skip("no host compiler on this machine")
    csrc = Path(tmp) / "csrc"
    shutil.copytree(CSRC, csrc / "cfbi")
    target = csrc / "cfbi" / HEADER
    text = target.read_text()
    for old, new in edits.items():
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    target.write_text(text)
    (Path(tmp) / "unit.cpp").write_text('#include "cfbi/cfbi_policy.h"\nint main() { return 0; }\n')
    r = subprocess.run([cxx, "-std=c++17", "-I", str(csrc), str(Path(tmp) / "unit.cpp"), "-o", str(Path(tmp) / "unit")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    return r.returncode, r.stdout


def test_a_tile_that_does_not_fit_two_ctas_per_sm_fails_the_build(tmp_path):
    def build(name, edits):
        (tmp_path / name).mkdir()
        return compile_header(tmp_path / name, edits)
    assert build("real", {})[0] == 0                                                                       # the real header builds: the control is not vacuous
    over = {"ROWS_E_STRIDE = 132;": "ROWS_E_STRIDE = 388;", "ROWS_E_TILE_FLOATS = 16896;": "ROWS_E_TILE_FLOATS = 49664;", "ROWS_DYNAMIC_SMEM_BYTES = 67584;": "ROWS_DYNAMIC_SMEM_BYTES = 198656;"}
    rc, text = build("over", over)                                                                         # a consistent tile of 198,656 B: only the budget is wrong
    assert rc != 0 and "leave room for ROWS_BUDGET_CTAS_PER_SM CTAs per SM" in text, text[-1500:]
    rc, text = build("torn", {"ROWS_DYNAMIC_SMEM_BYTES = 67584;": "ROWS_DYNAMIC_SMEM_BYTES = 67585;"})
    assert rc != 0 and "the dynamic shared memory of the row engine is the E tile" in text, text[-1500:]     # numbers of the header that disagree with each other
    rc, text = build("wide", {"ROWS_E_STRIDE = 132;": "ROWS_E_STRIDE = 133;"})
    assert rc != 0 and "the dynamic shared memory of the row engine is the E tile" in text, text[-1500:]


# ---- the launcher -----------------------------------------------------------------------------------------------------------------------------------------------
PREPARE_BODY = ps.norm("""
    int dev = 0;
    cudaError_t e = rows_current_device(&dev);
    if (e != cudaSuccess) return e;
    std::atomic<unsigned char>* kept = (dev >= 0 && dev < kRowsDevices) ? &g_rows_prepared[DECODE ? 1 : 0][dev] : nullptr;
    if (kept != nullptr && kept->load(std::memory_order_acquire) != 0) return cudaSuccess;
    e = cudaFuncSetAttribute(rows_kernel<DECODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, ROWS_DYNAMIC_SMEM_BYTES);
    if (e == cudaSuccess) e = cudaFuncSetAttribute(rows_kernel<DECODE>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    if (e == cudaSuccess && kept != nullptr) kept->store(1, std::memory_order_release);
    return e;
""")
SITES = [ps.norm(s) for s in (
    "if (a.impl & IMPL_REF_ROWS) ref::rows_kernel<<<dim3(a.N, a.H), NTHREADS, ROWS_REF_DYNAMIC_SMEM, stream>>>(p);",
    "else if (one_kernel) { if (rows_prepare<true>() != cudaSuccess) return; const DecodeParams dp{p, a.cu_seqlens, a.fixed_len}; rows_kernel<true><<<dim3(a.N, a.H), NTHREADS, ROWS_DYNAMIC_SMEM, stream>>>(dp); }",
    "else { if (rows_prepare<false>() != cudaSuccess) return; rows_kernel<false><<<dim3(a.N, a.H), NTHREADS, ROWS_DYNAMIC_SMEM, stream>>>(p); }")]
CONSTANTS = [ps.norm(s) for s in ("constexpr size_t ROWS_DYNAMIC_SMEM = ROWS_DYNAMIC_SMEM_BYTES;", "constexpr size_t ROWS_REF_DYNAMIC_SMEM = ROWS_REF_DYNAMIC_SMEM_BYTES;")]
INSTANCES = ps.norm("""const RowsInstance kRowsInstances[] = {{"cfbi::rows_kernel<false>", reinterpret_cast<const void*>(&rows_kernel<false>), ROWS_DYNAMIC_SMEM, &rows_prepare<false>},
    {"cfbi::rows_kernel<true>", reinterpret_cast<const void*>(&rows_kernel<true>), ROWS_DYNAMIC_SMEM, &rows_prepare<true>},
    {"cfbi::ref::rows_kernel", reinterpret_cast<const void*>(&ref::rows_kernel), ROWS_REF_DYNAMIC_SMEM, nullptr}};""")
OCCUPANCY = ps.norm("""
    const RowsInstance& r = kRowsInstances[index];
    cudaError_t e = r.prepare != nullptr ? r.prepare() : cudaSuccess;
    if (e != cudaSuccess) return (int)e;
    cudaFuncAttributes attr;
    e = cudaFuncGetAttributes(&attr, r.fn);
    if (e != cudaSuccess) return (int)e;
    int blocks = 0;
    e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, r.fn, NTHREADS, r.dynamic_smem);
    if (e != cudaSuccess) return (int)e;
    out->name = r.name;
    out->threads = NTHREADS;
    out->static_smem = (int)attr.sharedSizeBytes;
    out->dynamic_smem = (int)r.dynamic_smem;
    out->registers = attr.numRegs;
    out->ctas_per_sm = blocks;
    return 0;
""")


def launcher_problems(texts):
    view = " ".join(ps.tokens(ps.default_view(texts)[LAUNCH]))
    probs = []
    for snippet in CONSTANTS + [INSTANCES]:
        if view.count(snippet) != 1:
            probs.append("launcher: `%s` is not in the launcher exactly once" % snippet[:90])
    prepare = helper_text(texts, LAUNCH, PREPARE_HEAD)
    if prepare != PREPARE_BODY:
        probs.append("launcher: rows_prepare is not the pinned preparation (both attributes, return values checked, only a success remembered): %s" % prepare)
    if not all(view.count(x) == 1 for x in SITES):
        probs.append("launcher: the launch sites are not the three pinned ones (the optimised launches prepare first and pass ROWS_DYNAMIC_SMEM, the reference passes ROWS_REF_DYNAMIC_SMEM)")
    occupancy = helper_text(texts, LAUNCH, "int rows_occupancy (")
    if occupancy is None or not occupancy.startswith(ps.norm("if (index < 0 || index >= rows_occupancy_count() || out == nullptr) return (int)cudaErrorInvalidValue;")) or not occupancy.endswith(OCCUPANCY):
        probs.append("launcher: rows_occupancy does not prepare the instance, ask the occupancy API with the dynamic size of its own launch and report that size")
    if len(re.findall(r"\bcudaFuncSetAttribute \( rows_kernel\b", view)) != 2:
        probs.append("launcher: rows_kernel is given its attributes in other places than rows_prepare")
    return probs


def test_the_real_launcher_satisfies_the_pins(real):
    assert launcher_problems(real) == []


LAUNCHER_MUTANTS = {
    "the reference launched with the tile": ("NTHREADS, ROWS_REF_DYNAMIC_SMEM, stream", "NTHREADS, ROWS_DYNAMIC_SMEM, stream"),
    "the optimised engine launched without the tile": ("rows_kernel<false><<<dim3(a.N, a.H), NTHREADS, ROWS_DYNAMIC_SMEM, stream>>>(p);", "rows_kernel<false><<<dim3(a.N, a.H), NTHREADS, ROWS_REF_DYNAMIC_SMEM, stream>>>(p);"),
    "no preparation before the plan-path launch": ("            if (rows_prepare<false>() != cudaSuccess) return;\n", ""),
    "no preparation before the single launch": ("            if (rows_prepare<true>() != cudaSuccess) return;   // the runtime keeps the error for the launch check of the caller; nothing is launched without the opt-in to the dynamic tile\n", ""),
    "the attribute without the carve-out": ("    if (e == cudaSuccess) e = cudaFuncSetAttribute(rows_kernel<DECODE>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);\n", ""),
    "the size of the reference's tile": ("cudaFuncAttributeMaxDynamicSharedMemorySize, ROWS_DYNAMIC_SMEM_BYTES);", "cudaFuncAttributeMaxDynamicSharedMemorySize, ROWS_REF_DYNAMIC_SMEM_BYTES);"),
    "a failure remembered as a success": ("    if (e == cudaSuccess && kept != nullptr) kept->store(1, std::memory_order_release);", "    if (kept != nullptr) kept->store(1, std::memory_order_release);"),
    "the return value of the first attribute ignored": ("    e = cudaFuncSetAttribute(rows_kernel<DECODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, ROWS_DYNAMIC_SMEM_BYTES);\n    if (e == cudaSuccess) e =",
                                                         "    cudaFuncSetAttribute(rows_kernel<DECODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, ROWS_DYNAMIC_SMEM_BYTES);\n    e ="),
    "the occupancy of an instance asked without its own dynamic size": ("NTHREADS, r.dynamic_smem);", "NTHREADS, ROWS_DYNAMIC_SMEM);"),
    "the occupancy hook reports the optimised figure for every instance": ("out->dynamic_smem = (int)r.dynamic_smem;", "out->dynamic_smem = (int)ROWS_DYNAMIC_SMEM;"),
    "the occupancy hook does not prepare": ("    cudaError_t e = r.prepare != nullptr ? r.prepare() : cudaSuccess;\n    if (e != cudaSuccess) return (int)e;\n    cudaFuncAttributes attr;\n    e = cudaFuncGetAttributes",
                                             "    cudaFuncAttributes attr;\n    cudaError_t e = cudaFuncGetAttributes"),
    "the reference instance with the optimised constant": ("ref::rows_kernel), ROWS_REF_DYNAMIC_SMEM, nullptr}", "ref::rows_kernel), ROWS_DYNAMIC_SMEM, nullptr}"),
    "one constant for both engines": ("constexpr size_t ROWS_REF_DYNAMIC_SMEM = ROWS_REF_DYNAMIC_SMEM_BYTES;", "constexpr size_t ROWS_REF_DYNAMIC_SMEM = ROWS_DYNAMIC_SMEM_BYTES;"),
}


@pytest.mark.parametrize("name", sorted(LAUNCHER_MUTANTS))
def test_a_mutant_of_the_launcher_fails_its_pin(real, name):
    old, new = LAUNCHER_MUTANTS[name]
    assert launcher_problems(mutant(real, LAUNCH, old, new)), name
