"""Source pins of the arrival-count protocol of the single-launch decode, with mutants (plain Python, no GPU: ``PYTHONPATH=. pytest --noconftest tests/cfbi/test_decode_protocol_source.py``).

``rows_entry<true>`` (csrc/cfbi/cfbi_rows.cuh) lets the H head CTAs of a sequence agree on who advances the position word with one atomic and one store. Four rules of its ordering cannot be
seen by any comparison of results (the bits of the output, E, the rows and pos are the same with a plain store, with the atomic behind a barrier, with an extra masked read of the word):

1. the prologue ``atomicAdd(&pos[slot], 1 << POS_COUNT_SHIFT)`` is the ONLY read of the word; the fill of the open chunk comes from ``old & POS_MASK``;
2. the CTA whose arrival saw ``H - 1`` before it stores the plain ``(pos0 + len) % 64`` with a relaxed device-scope store;
3. a skipped CTA (padding entry, empty sequence) returns before the arrival, on a decision that depends on the sequence only (``len`` and ``slot``);
4. no spin-wait, no extra buffer, no fence; ``POS_MASK < (1 << POS_COUNT_SHIFT)`` is a ``static_assert``.

``protocol_source.py`` reads the sources as a default build reads them (comments gone, test-macro branches resolved, default hooks expanded, tokens) and returns the broken rules, ``R0`` being the
whole entry. Every rule is shown here to FAIL on a temporary copy of the sources with one edit (a rule that cannot fail proves nothing), and the pins are shown to hold under harmless edits
(spacing, comments, the numbers of the delay sites, the variant branches, a macro that no build takes). The preprocessor is part of what they read: a macro of a pinned name rewrites it for the
compiler while the tokens stay, so every effective ``#define`` / ``#undef`` and every ``-D`` flag of such a name is a problem (R12). A directive is read as the preprocessor reads it: on logical lines (a backslash before a line break, a comment over lines, a digraph or a form feed do not hide it), and the reader and the real preprocessor (``g++ -E -dM``) are shown to agree on which macros a text defines. The host model (``test_decode_pos_protocol_model.py``) takes its constants and its election from the same reader.
"""
import re

import pytest

import protocol_source as ps
from test_rows_profile import mutate

R, C, P, L, B = "cfbi_rows.cuh", "cfbi_common.cuh", "cfbi_primitives.cuh", "cfbi_launch.cu", "cfbi_binding.cpp"
ELECT = "== p.H - 1) st_relaxed_gpu(p.pos + ROWS_POS_INDEX(POS_ACCESS_ELECTED_STORE, w.slot)"
STORE_LINE = "            if ((old >> POS_COUNT_SHIFT) == p.H - 1) st_relaxed_gpu(p.pos + ROWS_POS_INDEX(POS_ACCESS_ELECTED_STORE, w.slot), (int)(((int64_t)(old & POS_MASK) + w.len) % BT));\n"
STORE_END = "(old & POS_MASK) + w.len) % BT));\n#endif\n#endif\n#endif"
HOOK9 = "            ROWS_DELAY(9);   // test hook: before the arrival\n"
SKIP_LINE = "        if (w.len == 0) return SeqPlan{0, 0, 0, 0, 0, 0, 0, w.slot};"
BARRIER_LINES = "        }\n        __syncthreads();"
ARRIVAL_ALL = "            ROWS_DECODE_ARRIVED(old, p.H);"
HELPER_END = '    asm volatile("st.relaxed.gpu.global.b32 [%0], %1;" ::"l"(ptr), "r"(v) : "memory");\n}\n'
DEFAULT_HOOKS = "#else\n#define ROWS_DELAY(n) do {} while (0)\n#define ROWS_DECODE_ARRIVED(old, heads) do {} while (0)\n#endif"
FLAGS_ANCHOR = '    "-fmad=false",'
PLAIN_STORE_MACRO = "#define st_relaxed_gpu(ptr, v) (*(ptr) = (v))\n"


@pytest.fixture(scope="module")
def real():
    return ps.load()


def copy_with(tmp_path, texts, edits):
    """A copy of the sources on disk with the edits applied (each ``(file, old, new)`` must find ``old`` exactly once), read back through the reader."""
    d = tmp_path / "cfbi"
    d.mkdir()
    flags = {}
    for name, text in texts.items():
        for f, old, new in edits:
            if f == name:
                text = mutate(text, old, new)
        if name.startswith("flags:"):
            flags[name] = text              # the build flags live outside the source directory: they are read from memory
            continue
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(text)
    loaded = ps.load(d)
    loaded.update(flags)
    return loaded


# ---- the real sources satisfy every rule ----------------------------------------------------------------------------------------------------------------------
def test_the_real_sources_satisfy_every_rule(real):
    assert ps.problems(real) == []


def test_the_constants_and_the_election_are_read_from_the_sources(real):
    assert ps.constants(real) == {"BT": 64, "POS_COUNT_SHIFT": 8, "POS_MASK": 63}
    e = ps.election(real)
    for H in (1, 2, 7, 64):
        assert [c for c in range(H + 2) if e(c, H)] == [H - 1]


def test_the_decode_branch_is_what_the_pins_say_it_is(real):
    assert " ".join(ps.decode_branch(real)) == ps.GOLDEN
    assert ps.GOLDEN.count("p . pos") == 2 and ps.GOLDEN.count("atomicAdd") == 1 and ps.GOLDEN.count("st_relaxed_gpu") == 1 and ps.GOLDEN.count("__syncthreads") == 1


# ---- the reader ---------------------------------------------------------------------------------------------------------------------------------------------------
def test_comments_are_not_code_and_strings_are_not_comments():
    assert ps.norm("a = b; // atomicAdd(p.pos, 0)\n/* st_relaxed_gpu( */ c;") == "a = b ; c ;"
    assert ps.norm('asm("ld // x");') == 'asm ( "ld // x" ) ;'
    assert ps.tokens("k<<<g, b>>>(x);") == ["k", "<<", "<", "g", ",", "b", ">>", ">", "(", "x", ")", ";"]


def test_test_macro_branches_are_resolved_the_way_a_default_build_reads_them():
    text = "a\n#ifdef CFBI_DECODE_TEST_X\nb\n#else\nc\n#ifndef CFBI_DECODE_TEST_Y\nd\n#endif\n#endif\n#ifdef TUNABLE\ne\n#endif\n"
    view = ps.default_text(text, ps.macros_in([text]))
    assert view.split("\n") == ["a", "c", "d", "#ifdef TUNABLE", "e", "#endif"]
    with pytest.raises(ValueError):
        ps.default_text("#ifdef CFBI_DECODE_TEST_X\n#elif Y\n#endif\n", {"CFBI_DECODE_TEST_X"})


def test_the_default_hooks_vanish_and_the_position_index_is_the_slot():
    toks = ps.tokens("ROWS_DELAY(9); a = ROWS_POS_INDEX(POS_ACCESS_ARRIVAL, w.slot + 1); ROWS_DECODE_ARRIVED(old, p.H); b;")
    assert " ".join(ps.expand_default_hooks(toks)) == "a = ( w . slot + 1 ) ; b ;"


# ---- mutants: each rule fails on a temporary copy of the sources ----------------------------------------------------------------------------------------------
MUTANTS = [
    # rule 2: the election and the store
    ("the election against p.H", [(R, ELECT, "== p.H) st_relaxed_gpu(p.pos + ROWS_POS_INDEX(POS_ACCESS_ELECTED_STORE, w.slot)")], {"R0", "R2"}),
    ("the election against 0 (the first arriver)", [(R, ELECT, "== 0) st_relaxed_gpu(p.pos + ROWS_POS_INDEX(POS_ACCESS_ELECTED_STORE, w.slot)")], {"R0", "R2"}),
    ("the election against H - 2", [(R, ELECT, "== p.H - 2) st_relaxed_gpu(p.pos + ROWS_POS_INDEX(POS_ACCESS_ELECTED_STORE, w.slot)")], {"R0", "R2"}),
    ("the election by head index", [(R, "if ((old >> POS_COUNT_SHIFT) == p.H - 1) st_relaxed_gpu(p.pos + ROWS_POS_INDEX(POS_ACCESS_ELECTED_STORE",
                                     "if (blockIdx.y == 0) st_relaxed_gpu(p.pos + ROWS_POS_INDEX(POS_ACCESS_ELECTED_STORE")], {"R0", "R2"}),
    ("the election on the word after the add", [(R, "if ((old >> POS_COUNT_SHIFT) == p.H - 1) st_relaxed_gpu(p.pos + ROWS_POS_INDEX(POS_ACCESS_ELECTED_STORE",
                                                  "if (((old >> POS_COUNT_SHIFT) + 1) == p.H) st_relaxed_gpu(p.pos + ROWS_POS_INDEX(POS_ACCESS_ELECTED_STORE")], {"R0", "R2"}),
    ("a plain store instead of the relaxed one", [(R, STORE_LINE, "            if ((old >> POS_COUNT_SHIFT) == p.H - 1) p.pos[ROWS_POS_INDEX(POS_ACCESS_ELECTED_STORE, w.slot)] = (int)(((int64_t)(old & POS_MASK) + w.len) % BT);\n")], {"R0", "R2"}),
    ("the relaxed store helper made a plain global store", [(R, "st.relaxed.gpu.global.b32", "st.global.b32")], {"R6"}),
    ("the relaxed store helper without volatile", [(R, 'asm volatile("st.relaxed.gpu.global.b32', 'asm("st.relaxed.gpu.global.b32')], {"R6"}),
    ("the relaxed store helper without its memory clobber", [(R, '::"l"(ptr), "r"(v) : "memory");', '::"l"(ptr), "r"(v));')], {"R6"}),
    ("the store without the mask", [(R, STORE_END, "(old) + w.len) % BT));\n#endif\n#endif\n#endif")], {"R0", "R2"}),
    ("the store without the modulo", [(R, STORE_END, "(old & POS_MASK) + w.len)));\n#endif\n#endif\n#endif")], {"R0", "R2"}),
    ("the store without the 64-bit promotion", [(R, "(int)(((int64_t)(old & POS_MASK) + w.len) % BT));\n#endif\n#endif\n#endif", "(int)((((old & POS_MASK) + w.len) % BT));\n#endif\n#endif\n#endif")], {"R0", "R2"}),
    ("the store to another slot", [(R, ELECT, "== p.H - 1) st_relaxed_gpu(p.pos + ROWS_POS_INDEX(POS_ACCESS_ELECTED_STORE, w.slot + 1)")], {"R0", "R2"}),
    ("the store of another length", [(R, STORE_END, "(old & POS_MASK) + w.len + 1) % BT));\n#endif\n#endif\n#endif")], {"R0", "R2"}),
    ("no elected store at all", [(R, STORE_LINE, "")], {"R0", "R2"}),
    ("a second store", [(R, STORE_LINE, STORE_LINE + "            st_relaxed_gpu(p.pos + ROWS_POS_INDEX(POS_ACCESS_ELECTED_STORE, w.slot), 0);\n")], {"R0", "R2"}),
    ("the elected store only in a variant branch", [(R, "#ifndef CFBI_DECODE_TEST_NO_ELECTED_STORE\n            if ((old >> POS_COUNT_SHIFT) == p.H - 1)", "#ifdef CFBI_DECODE_TEST_NO_ELECTED_STORE\n            if ((old >> POS_COUNT_SHIFT) == p.H - 1)")], {"R0", "R2"}),
    ("the real election in a comment, a wrong one in the code", [(R, ELECT, "== 0) st_relaxed_gpu(p.pos + ROWS_POS_INDEX(POS_ACCESS_ELECTED_STORE, w.slot)"),
                                                                  (R, STORE_END, "(old & POS_MASK) + w.len) % BT));   // (old >> POS_COUNT_SHIFT) == p.H - 1\n#endif\n#endif\n#endif")], {"R0", "R2"}),
    # rule 1: the atomic is the only read
    ("a barrier in front of the atomic", [(R, "        if (threadIdx.x == 0) {\n            ROWS_DELAY(9);", "        __syncthreads();\n        if (threadIdx.x == 0) {\n            ROWS_DELAY(9);")], {"R0", "R4", "R9"}),
    ("the atomic behind other work", [(R, HOOK9, HOOK9 + "            pos_smem = w.tok0;\n")], {"R0", "R1", "R9"}),
    ("an extra read of the word before the atomic", [(R, HOOK9, HOOK9 + "            const int snap = p.pos[ROWS_POS_INDEX(POS_ACCESS_ARRIVAL, w.slot)];\n")], {"R0", "R1", "R7"}),
    ("an extra read of the word through a helper", [(R, HOOK9, HOOK9 + "            const int snap = load_pos(p.pos, w.slot);\n")], {"R0", "R1", "R7"}),
    ("a read of the word through an alias", [(R, HOOK9, HOOK9 + "            const int* alias = p.pos;\n            const int snap = alias[w.slot];\n")], {"R0", "R1", "R7"}),
    ("a snapshot of the word after the arrival", [(R, ARRIVAL_ALL, ARRIVAL_ALL + "\n            const int snap = ld_relaxed_gpu(p.pos + w.slot);")], {"R0", "R1", "R4", "R7"}),
    ("the fill taken without the mask", [(R, "#else\n            pos_smem = old & POS_MASK;\n#endif", "#else\n            pos_smem = old;\n#endif")], {"R0", "R1"}),
    ("a plain add of zero instead of the counting add", [(R, "ROWS_POS_INDEX(POS_ACCESS_ARRIVAL, w.slot), 1 << POS_COUNT_SHIFT);", "ROWS_POS_INDEX(POS_ACCESS_ARRIVAL, w.slot), 0);")], {"R0", "R1"}),
    ("the every-CTA tail store enabled in the default build", [(R, "#ifdef CFBI_DECODE_TEST_EVERY_CTA_WRITES\n        if (tid == 0) p.pos[sp.slot] = ", "#if 1\n        if (tid == 0) p.pos[sp.slot] = ")], {"R7", "R12"}),
    ("the parameter block handed to another function", [(R, "    const int nrows = sp.hrows;\n", "    const int nrows = sp.hrows;\n    touch(p);\n")], {"R7"}),
    # rule 3: the skip
    ("the skip behind the atomic", [(R, SKIP_LINE + "   // padding entry or empty sequence (uniform over the CTA): no state access, the caller returns\n", ""),
                                    (R, "        __syncthreads();\n        return SeqPlan{w.tok0", "        __syncthreads();\n" + SKIP_LINE + "\n        return SeqPlan{w.tok0")], {"R0", "R3"}),
    ("a skip that depends on the head", [(R, SKIP_LINE, "        if (w.len == 0 || blockIdx.y > 3) return SeqPlan{0, 0, 0, 0, 0, 0, 0, w.slot};")], {"R0", "R3"}),
    ("a window that depends on the head", [(C, "    if (w.slot < 0) w.len = 0;", "    if (w.slot < 0 || blockIdx.y > 3) w.len = 0;")], {"R3"}),
    ("a negative slot that is not skipped", [(C, "    if (w.slot < 0) w.len = 0;", "")], {"R3"}),
    # rule 4: no spin, no fence, one hand-over barrier
    ("a spin-wait on the counter", [(R, HOOK9, HOOK9 + "            while (atomicAdd(p.pos + ROWS_POS_INDEX(POS_ACCESS_ARRIVAL, w.slot), 0) < 0) {}\n")], {"R0", "R1", "R4"}),
    ("a fence after the arrival", [(R, ARRIVAL_ALL, ARRIVAL_ALL + "\n            __threadfence();")], {"R0", "R4"}),
    ("a sleep in the entry", [(R, HOOK9, HOOK9 + "            __nanosleep(100);\n")], {"R0", "R4"}),
    ("the hand-over barrier removed", [(R, BARRIER_LINES, "        }")], {"R0", "R4", "R9"}),
    ("the fill read before the barrier", [(R, BARRIER_LINES, "        }\n        const int early = pos_smem;\n        __syncthreads();")], {"R0", "R9"}),
    ("the static_assert removed", [(C, 'static_assert(POS_MASK < (1 << POS_COUNT_SHIFT), "the fill of the open chunk must fit below the arrival counter");', "")], {"R5"}),
    # the constants
    ("the shift changed to 7", [(C, "constexpr int POS_COUNT_SHIFT = 8;", "constexpr int POS_COUNT_SHIFT = 7;")], {"R5"}),
    ("the mask changed to 127", [(C, "constexpr int POS_MASK = BT - 1;", "constexpr int POS_MASK = BT * 2 - 1;")], {"R5"}),
    ("the chunk length changed", [(P, "constexpr int BT = 64;", "constexpr int BT = 128;")], {"R5"}),
    # the geometry
    ("a grid with one head too many", [(L, "rows_kernel<true><<<dim3(a.N, a.H), NTHREADS", "rows_kernel<true><<<dim3(a.N, a.H + 1), NTHREADS")], {"R8"}),
    ("a grid with a third dimension", [(L, "rows_kernel<true><<<dim3(a.N, a.H), NTHREADS", "rows_kernel<true><<<dim3(a.N, a.H, 2), NTHREADS")], {"R8"}),
    ("a head count that is not the grid's", [(L, "    p.H = a.H;\n", "    p.H = a.H + 1;\n")], {"R8"}),
    ("token windows of another call", [(L, "const DecodeParams dp{p, a.cu_seqlens, a.fixed_len};", "const DecodeParams dp{p, nullptr, a.fixed_len};")], {"R8"}),
    ("the sequence and head index swapped", [(R, "    const int n = blockIdx.x, h = blockIdx.y, H = p.H;", "    const int n = blockIdx.y, h = blockIdx.x, H = p.H;")], {"R8"}),
    ("a second launch of the kernel", [(L, "const DecodeParams dp{p, a.cu_seqlens, a.fixed_len};", "const DecodeParams dp{p, a.cu_seqlens, a.fixed_len};\n            rows_kernel<true><<<dim3(a.N, a.H), NTHREADS, ROWS_DYNAMIC_SMEM, stream>>>(dp);")], {"R8"}),
    ("a heads limit beyond the counter capacity", [(B, 'H < 65536, "problem too large', 'H < 100000000, "problem too large')], {"R10"}),
    # the hooks of the default build
    ("the position index offset in the default build", [(R, "#else\n#define ROWS_POS_INDEX(access, slot) (slot)\n#endif", "#else\n#define ROWS_POS_INDEX(access, slot) ((slot) + 1)\n#endif")], {"R11"}),
    ("the arrival delays in the default build", [(R, "#else\n#define ROWS_DELAY(n) do {} while (0)\n#define ROWS_DECODE_ARRIVED(old, heads) do {} while (0)\n#endif",
                                                  "#else\n#define ROWS_DELAY(n) do {} while (0)\n#define ROWS_DECODE_ARRIVED(old, heads) rows_delay_arrived(old, heads)\n#endif")], {"R11"}),
    # macros: what the compiler sees while the tokens stay
    ("a macro that makes the elected store a plain assignment (the review's counterexample)", [(R, HELPER_END, HELPER_END + PLAIN_STORE_MACRO)], {"R12"}),
    ("the same macro in another header of the directory", [("cfbi_policy.h", "#pragma once", "#pragma once\n" + PLAIN_STORE_MACRO.rstrip("\n"))], {"R12"}),
    ("a second definition of a modeled hook", [(R, DEFAULT_HOOKS, DEFAULT_HOOKS + "\n#define ROWS_DELAY(n) rows_delay(n)")], {"R12", "R11"}),
    ("an #undef of a modeled hook", [(R, DEFAULT_HOOKS, DEFAULT_HOOKS + "\n#undef ROWS_POS_INDEX")], {"R12"}),
    ("a macro that rewrites the atomic", [(R, HELPER_END, HELPER_END + "#define atomicAdd(address, value) ((address)[0])\n")], {"R12"}),
    ("a macro that rewrites the value that the atomic returned", [(R, HELPER_END, HELPER_END + "#define old 0\n")], {"R12"}),
    ("a macro that rewrites the chunk length", [(P, "constexpr int BT = 64;", "constexpr int BT = 64;\n#define BT 128")], {"R12"}),
    ("a macro that rewrites a keyword of the entry", [(R, HELPER_END, HELPER_END + "#define return (void)\n")], {"R12"}),
    ("an #undef of the tunable that the kernel head reads", [(R, "#ifndef CFBI_ROWS_MINB\n#define CFBI_ROWS_MINB 2\n#endif", "#ifndef CFBI_ROWS_MINB\n#define CFBI_ROWS_MINB 2\n#endif\n#undef CFBI_ROWS_MINB")], {"R12"}),
    ("a build flag that renames the store helper", [("flags:_ext.py", FLAGS_ANCHOR, FLAGS_ANCHOR + '\n    "-Dst_relaxed_gpu(p,v)=(*(p)=(v))",')], {"R12"}),
    ("a build flag that removes a macro of a pinned name", [("flags:setup.py", "import os", "import os\n# -Uatomic\nFLAGS = ['-UBT']")], {"R12"}),
    ("an include of a file that the reader does not load", [(R, '#include "cfbi/cfbi_common.cuh"', '#include "cfbi/cfbi_common.cuh"\n#include "cfbi/hidden_macros.inc"')], {"R12"}),
    ("an include of a file next to the including file that does not exist", [("ref/ref_k1.cuh", '#include "cfbi/cfbi_common.cuh"', '#include "cfbi/cfbi_common.cuh"\n#include "hidden_macros.inc"')], {"R12"}),
    ("a forced include in the build flags", [("flags:_ext.py", FLAGS_ANCHOR, FLAGS_ANCHOR + '\n    "-include", "hidden_macros.h",')], {"R12"}),
    # the same macro in the other spellings of a directive (the preprocessor joins the lines first: logical lines)
    ("a continued #define of the store helper (the review's second counterexample)", [(R, HELPER_END, HELPER_END + "#define \\\nst_relaxed_gpu(ptr, v) (*(ptr) = (v))\n")], {"R12"}),
    ("a #define whose keyword is split by a continuation", [(R, HELPER_END, HELPER_END + "#def\\\nine st_relaxed_gpu(ptr, v) (*(ptr) = (v))\n")], {"R12"}),
    ("a #define whose hash is split from the keyword", [(R, HELPER_END, HELPER_END + "#\\\ndefine st_relaxed_gpu(ptr, v) (*(ptr) = (v))\n")], {"R12"}),
    ("a continuation with blanks before the line break (gcc joins it too)", [(R, HELPER_END, HELPER_END + "#define \\  \nst_relaxed_gpu(ptr, v) (*(ptr) = (v))\n")], {"R12"}),
    ("a #define over a block comment that spans lines", [(R, HELPER_END, HELPER_END + "#define /* the name\nfollows */ st_relaxed_gpu(ptr, v) (*(ptr) = (v))\n")], {"R12"}),
    ("a #define in the digraph spelling", [(R, HELPER_END, HELPER_END + "%:define st_relaxed_gpu(ptr, v) (*(ptr) = (v))\n")], {"R12"}),
    ("a #define with a form feed in the directive", [(R, HELPER_END, HELPER_END + "#\fdefine st_relaxed_gpu(ptr, v) (*(ptr) = (v))\n")], {"R12"}),
    ("a spliced #if 0 that hides the elected store from the compiler", [(R, STORE_LINE, "#i\\\nf 0\n" + STORE_LINE + "#endif\n")], {"R12"}),
    ("a conditional that hides the real chunk length from the compiler", [(P, "constexpr int BT = 64;", "#if 0\nconstexpr int BT = 64;\n#else\nconstexpr int BT = 128;\n#endif")], {"R12"}),
    ("an include of an unread file in the angle form", [(R, '#include "cfbi/cfbi_common.cuh"', '#include "cfbi/cfbi_common.cuh"\n#include <cfbi/hidden_macros.inc>')], {"R12"}),
    ("an include whose target is a macro", [(R, '#include "cfbi/cfbi_common.cuh"', '#include "cfbi/cfbi_common.cuh"\n#define HIDDEN_NAME "hidden_macros.inc"\n#include HIDDEN_NAME')], {"R12"}),
    ("a directive of a form that the reader does not know", [(R, HELPER_END, HELPER_END + "#assign st_relaxed_gpu 1\n")], {"R12"}),
    ("a trigraph that spells a line splice", [(R, HELPER_END, HELPER_END + "#define ??/\nst_relaxed_gpu(ptr, v) (*(ptr) = (v))\n")], {"R12"}),
    # text that the compiler does not see but the reader does
    ("the real protocol under #if 0", [(R, STORE_LINE, "#if 0\n" + STORE_LINE + "#endif\n")], {"R12"}),
    ("a conditional on a test macro with #elif", [(R, "#ifndef CFBI_DECODE_TEST_NO_ELECTED_STORE\n            if ((old >> POS_COUNT_SHIFT) == p.H - 1)", "#ifndef CFBI_DECODE_TEST_NO_ELECTED_STORE\n#elif 1\n            if ((old >> POS_COUNT_SHIFT) == p.H - 1)")], {"R12"}),
]


@pytest.mark.parametrize("name,edits,expect", MUTANTS, ids=[m[0] for m in MUTANTS])
def test_a_wrong_protocol_in_a_temporary_copy_of_the_sources_fails_its_rule(tmp_path, real, name, edits, expect):
    broken = ps.problems(copy_with(tmp_path, real, edits))
    rules = {r for r, _ in broken}
    assert expect <= rules, (name, sorted(rules), [m[:200] for r, m in broken][:4])


def test_every_rule_has_a_mutant_that_breaks_it():
    covered = set().union(*(m[2] for m in MUTANTS))
    assert {"R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11", "R12"} <= covered


# ---- harmless edits do not trip the pins ----------------------------------------------------------------------------------------------------------------------
HARMLESS = [
    ("spacing and line breaks of the entry", [(R, "        const SeqWindow w = seq_window(p.cu_seqlens, p.fixed_len, p.slot_ids, n);", "        const SeqWindow w =\n            seq_window( p.cu_seqlens,   p.fixed_len,\n                        p.slot_ids, n );")]),
    ("a comment that holds the protocol text", [(R, HOOK9, HOOK9 + "            // old = atomicAdd(p.pos + w.slot, 0);  if ((old >> POS_COUNT_SHIFT) == 0) p.pos[w.slot] = 1;\n")]),
    ("the numbers of the delay sites", [(R, HOOK9, "            ROWS_DELAY(13);   // test hook: before the arrival\n")]),
    ("the arguments of the arrival hook", [(R, ARRIVAL_ALL, "            ROWS_DECODE_ARRIVED(old, (p.H));")]),
    ("a variant branch (the first-arriver protocol of the tests)", [(R, "            if ((old >> POS_COUNT_SHIFT) == 0) st_relaxed_gpu(", "            if ((old >> POS_COUNT_SHIFT) == 1) st_relaxed_gpu(")]),
    ("the snapshot variant", [(R, "pos_smem = ld_relaxed_gpu(p.pos + w.slot) & POS_MASK;", "pos_smem = ld_relaxed_gpu(p.pos + w.slot + 1) & POS_MASK;")]),
    ("the tail store of the every-CTA variant", [(R, "#ifdef CFBI_DECODE_TEST_EVERY_CTA_WRITES\n        if (tid == 0) p.pos[sp.slot] = ", "#ifdef CFBI_DECODE_TEST_EVERY_CTA_WRITES\n        if (tid == 0 && true) p.pos[sp.slot] = ")]),
    ("a comment in the header constants", [(C, "constexpr int POS_MASK = BT - 1;", "constexpr int POS_MASK = BT - 1;   // low bits")]),
    ("the plain-store macro in a branch that no default build takes", [(R, HELPER_END, HELPER_END + "#ifdef CFBI_DECODE_TEST_SNAPSHOT_AFTER_ARRIVAL\n" + PLAIN_STORE_MACRO + "#endif\n")]),
    ("the plain-store macro in a line comment", [(R, HELPER_END, HELPER_END + "// " + PLAIN_STORE_MACRO)]),
    ("the plain-store macro in a block comment", [(R, HELPER_END, HELPER_END + "/*\n" + PLAIN_STORE_MACRO + "*/\n")]),
    ("a macro of a name that no pin reads", [(R, HELPER_END, HELPER_END + "#define SOME_UNRELATED_TUNABLE 3\n")]),
    ("a build flag of a name that no pin reads", [("flags:_ext.py", FLAGS_ANCHOR, FLAGS_ANCHOR + '\n    "-DSOME_UNRELATED_TUNABLE=3",')]),
    ("an include of another file of the directory, by the path of the build and relative to the including file", [(R, '#include "cfbi/cfbi_common.cuh"', '#include "cfbi/cfbi_common.cuh"\n#include "cfbi/ref/ref_k2.cuh"'),
                                                                                                                  ("ref/ref_k1.cuh", '#include "cfbi/cfbi_common.cuh"', '#include "cfbi/cfbi_common.cuh"\n#include "ref_k2.cuh"\n#include "../cfbi_primitives.cuh"')]),
    ("a macro hidden inside a continued line comment", [(R, HELPER_END, HELPER_END + "// a comment that goes on \\\n#define st_relaxed_gpu(ptr, v) (*(ptr) = (v))\n")]),
    ("a legitimate continued macro of an unrelated name", [(R, HELPER_END, HELPER_END + "#define SOME_UNRELATED_TUNABLE(a, b) \\\n    ((a) + (b))\n")]),
    ("a continued macro whose body calls the helper without defining it", [(R, HELPER_END, HELPER_END + "#define UNRELATED_STORE(p, v) \\\n    st_relaxed_gpu((p), (v))\n")]),
    ("a block comment over several lines before a directive of an unrelated name", [(R, HELPER_END, HELPER_END + "/* a comment\nover lines */ #define SOME_UNRELATED_TUNABLE 3\n")]),
    ("the continued plain-store macro in a branch that no default build takes", [(R, HELPER_END, HELPER_END + "#ifdef CFBI_DECODE_TEST_SNAPSHOT_AFTER_ARRIVAL\n#define \\\nst_relaxed_gpu(ptr, v) (*(ptr) = (v))\n#endif\n")]),
    ("an include in a comment and an angle-bracket include", [(R, '#include "cfbi/cfbi_common.cuh"', '#include "cfbi/cfbi_common.cuh"\n// #include "cfbi/hidden_macros.inc"\n#include <vector>')]),
]


@pytest.mark.parametrize("name,edits", HARMLESS, ids=[h[0] for h in HARMLESS])
def test_a_harmless_edit_does_not_trip_the_pins(tmp_path, real, name, edits):
    assert ps.problems(copy_with(tmp_path, real, edits)) == []


def test_a_renamed_entry_is_reported_not_a_crash(tmp_path, real):
    broken = ps.problems(copy_with(tmp_path, real, [(R, "SeqPlan rows_entry(const RowsParams<DECODE>& p, int n, int& pos_smem)", "SeqPlan rows_enter(const RowsParams<DECODE>& p, int n, int& pos_smem)")]))
    assert any(r == "R0" for r, _ in broken)


# ---- the model reads the sources ---------------------------------------------------------------------------------------------------------------------------
def test_an_election_that_the_model_cannot_read_is_refused(tmp_path, real):
    texts = copy_with(tmp_path, real, [(R, "if ((old >> POS_COUNT_SHIFT) == p.H - 1) st_relaxed_gpu(p.pos + ROWS_POS_INDEX(POS_ACCESS_ELECTED_STORE", "if (blockIdx.y == 0) st_relaxed_gpu(p.pos + ROWS_POS_INDEX(POS_ACCESS_ELECTED_STORE")])
    with pytest.raises(ValueError):
        ps.election(texts)


# ---- the macro rule's reader --------------------------------------------------------------------------------------------------------------------------------
def test_directives_are_read_per_directive_from_the_default_view_without_comments():
    text = "#define KEEP 1\n// #define IN_LINE_COMMENT 2\n/*\n#define IN_BLOCK 3\n*/\n  #  undef  GONE\n#define MULTI(a) \\\n    (a)\nint x;\n"
    assert ps.macro_directives({"a.cuh": ps.default_lines(text, set())}) == [("a.cuh", 1, "define", "KEEP"), ("a.cuh", 6, "undef", "GONE"), ("a.cuh", 7, "define", "MULTI")]
    text = "#ifdef CFBI_DECODE_TEST_X\n#define TEST_ONLY 1\n#else\n#define DEFAULT_ONLY 1\n#endif\n"
    lines = ps.default_lines(text, ps.macros_in([text]))
    assert lines == [(4, "#define DEFAULT_ONLY 1")]                       # the numbers are those of the file, not of the view
    assert ps.macro_directives({"b.cuh": lines}) == [("b.cuh", 4, "define", "DEFAULT_ONLY")]


def test_logical_lines_are_what_the_first_three_phases_of_translation_leave():
    ll = ps.logical_lines
    assert ll("a\nb\n") == ((1, "a"), (2, "b")) and ll("x") == ((1, "x"),) and ll("\n\nx\n") == ((1, ""), (2, ""), (3, "x")) and ll("a\r\nb\r\n") == ((1, "a"), (2, "b"))
    assert ll("a\\\nb\n") == ((1, "ab"),)                                                  # a backslash before a line break joins the lines
    assert ll("#define \\\n  X 1\nY\n") == ((1, "#define   X 1"), (3, "Y"))                  # the second line gets the number of its first physical line
    assert ll("a \\  \nb\n") == ((1, "a b"),)                                                # blanks between the backslash and the line break (gcc joins these too)
    assert ll("a /* x\ny */ b\nc\n") == ((1, "a   b"), (3, "c"))                              # a comment is one space and takes its line break with it
    assert ll("a // b \\\nc\nd\n") == ((1, "a  "), (3, "d"))                                  # a line comment goes on over a continuation
    assert ll("/* only\na comment */\nx\n") == ((1, " "), (3, "x"))
    assert ll('s = "//" + "/*"; // x\n') == ((1, 's = "//" + "/*";  '),)                       # comment markers inside a string are no comment
    assert ll("c = '\"'; // x\n") == ((1, "c = '\"';  "),)                                     # a quote inside a character literal starts no string
    assert ll("n = 1'000'000; // x\n") == ((1, "n = 1'000'000;  "),)                           # digit separators are no character literals
    text = "#ifndef T\n#define \\\n X 1\n#else\nA\n#endif\nB\n"
    assert ps.default_lines(text, {"T"}) == [(2, "#define  X 1"), (7, "B")]                    # the conditionals are resolved on the logical lines, numbers stay those of the file
    assert ps.default_lines(text.replace("ifndef", "ifdef"), {"T"}) == [(5, "A"), (7, "B")]    # a default build defines no test macro: the #else branch of #ifdef is the one that stays
    assert ps.tokens("#define X \\\n  y z\nint a; /* c\n */ int b;\n") == ["int", "a", ";", "int", "b", ";"]       # a directive that runs over lines is no code


# the same macro spelled in every way that matters: the reader and the real preprocessor must agree on which macros a text defines
SPELLINGS = [
    ("plain", "#define PIN_A 1\n", {"PIN_A"}),
    ("a continuation after the keyword", "#define \\\nPIN_A 1\n", {"PIN_A"}),
    ("the keyword split by a continuation", "#def\\\nine PIN_A 1\n", {"PIN_A"}),
    ("the hash split from the keyword", "#\\\ndefine PIN_A 1\n", {"PIN_A"}),
    ("blanks between the backslash and the line break", "#define \\  \nPIN_A 1\n", {"PIN_A"}),
    ("a block comment over lines inside the directive", "#define /* a\ncomment */ PIN_A 1\n", {"PIN_A"}),
    ("a comment between the hash and the keyword", "# /* c */ define PIN_A 1\n", {"PIN_A"}),
    ("the digraph for the hash", "%:define PIN_A 1\n", {"PIN_A"}),
    ("a form feed in the directive", "#\fdefine PIN_A 1\n", {"PIN_A"}),
    ("tabs and blanks", "\t#\tdefine\tPIN_A 1\n", {"PIN_A"}),
    ("a block comment before the directive", "/* a\ncomment */ #define PIN_A 1\n", {"PIN_A"}),
    ("a directive after a continued line comment is part of the comment", "// a comment \\\n#define PIN_A 1\nint x;\n", set()),
    ("a directive in a block comment", "/*\n#define PIN_A 1\n*/\n", set()),
    ("a directive in a string", 'const char* s = "#define PIN_A 1";\n', set()),
    ("an unrelated continued macro", "#define PIN_B(a) \\\n  ((a) + 1)\n", {"PIN_B"}),
    ("two directives around a comment that ends on the first", "#define PIN_A 1 /* x\n*/\n#define PIN_B 2\n", {"PIN_A", "PIN_B"}),
]


def host_cxx():
    import shutil
    return shutil.which("g++") or shutil.which("c++")


def gcc_defined(text):
    """The names of the macros named ``PIN_*`` that the real preprocessor (``g++ -E -dM``) has defined after reading ``text``."""
    import subprocess
    cxx = host_cxx()
    if cxx is None:
        pytest.skip("no host compiler on this machine")
    run = subprocess.run([cxx, "-E", "-dM", "-x", "c++", "-"], input=text, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
    return {m.group(1) for m in (re.match(r"#define (PIN_\w+)", line) for line in run.stdout.splitlines()) if m}


@pytest.mark.parametrize("name,text,expected", SPELLINGS, ids=[s[0] for s in SPELLINGS])
def test_the_reader_and_the_real_preprocessor_agree_on_which_macros_a_text_defines(name, text, expected):
    read = {d[3] for d in ps.macro_directives({"x.cuh": ps.default_lines(text, set())}) if d[2] == "define"}
    assert read == expected, (name, read)
    assert gcc_defined(text) == expected, name


@pytest.mark.parametrize("spelling", [s for s in SPELLINGS if s[2] == {"PIN_A"} and s[0] != "plain"][:6], ids=lambda s: s[0])
def test_the_continued_spellings_of_the_reviews_macro_really_change_the_call(spelling):
    """The review's counterexample under the real preprocessor: the elected call becomes a plain assignment in every one of these spellings."""
    import subprocess
    cxx = host_cxx()
    if cxx is None:
        pytest.skip("no host compiler on this machine")
    macro = spelling[1].replace("PIN_A 1", "st_relaxed_gpu(ptr, v) (*(ptr) = (v))")
    source = "static inline void st_relaxed_gpu(int* ptr, int v) { *ptr = v; }\n" + macro + "void f(int* pos, int old, int len) { if (old == 3) st_relaxed_gpu(pos + 1, old + len); }\n"
    out = subprocess.run([cxx, "-E", "-P", "-x", "c++", "-"], input=source, stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout
    call = out.strip().splitlines()[-1]
    assert "st_relaxed_gpu" not in call and "(*(pos + 1) = (old + len))" in call, call
    assert {d[3] for d in ps.macro_directives({"x.cuh": ps.default_lines(macro, set())})} == {"st_relaxed_gpu"}


def test_the_names_that_the_pins_read_include_keywords_constants_hooks_and_the_generic_names_of_the_entry(real):
    for name in ("return", "if", "old", "p", "n", "atomicAdd", "st_relaxed_gpu", "BT", "POS_MASK", "POS_COUNT_SHIFT", "ROWS_DELAY", "ROWS_DECODE_ARRIVED", "ROWS_POS_INDEX", "rows_kernel", "CFBI_ROWS_MINB"):
        assert name in ps.PINNED, name
    assert "SOME_UNRELATED_TUNABLE" not in ps.PINNED
    defined = [d[3] for d in ps.macro_directives(ps.default_numbered(real)) if d[3] in ps.PINNED]
    assert sorted(defined) == sorted(ps.ALLOWED_DEFINES), defined        # the real tree defines exactly the modeled hooks and the tunable, each once
