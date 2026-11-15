"""Source rules of the staged A fragments (F) of the row engine (plain Python, no GPU: ``pytest --noconftest tests/cfbi/test_rows_afrag_source.py``), each with mutants that break it.

F builds the TF32 hi / lo A fragments of phases B, C and D once per row (AB, AM, AD families, ``FragSmem`` of ``cfbi_rows.cuh``) and lets every consumer load them as plain ``uint4`` words. A defect
in a tile that no output row reads cannot change a result, so no comparison of outputs can show it; the rules are therefore pinned as text, and the fragment-level oracle of the device tests
(``CFBI_ROWS_TEST_AFRAG_CHECK``) covers what the text cannot:

* layout: the struct ``FragSmem`` (zero page first, pads, the three tiles as one arena), its asserts, and the default ``RowSmem`` (no ``gq`` / ``x1``: those belong to the oracle build);
* helpers: the bodies of the seven functions that are the only code to touch the tiles are pinned token by token (zero page written once, base pointers, the three store functions);
* access: no tile member is named outside those helpers, so a producer or consumer cannot bypass the protocol; the consumers load through the base pointers only, with the pinned offsets;
* windows: in the order of the code of a row, every store of a family stands in front of the barrier that publishes it and every load behind it (AB: barrier 13, AM: barrier 2, AD: barrier 4), and
  the zero page is written once in front of the row loop;
* oracle: each consumer family calls its check right behind the unpack of the words that it loaded (a check that is not called, or that looks at other words, is a blind oracle).

The default build is read with every test macro undefined (``protocol_source.default_text``), as in ``test_rows_hooks.py``.
"""
import re
from pathlib import Path

import pytest

import test_rows_profile as trp
from protocol_source import default_text, macros_in, match_close, norm, tokens
from test_rows_profile import mutate

CSRC = Path(__file__).resolve().parents[2] / "csrc" / "cfbi"


@pytest.fixture(scope="module")
def rows_text():
    return (CSRC / "cfbi_rows.cuh").read_text()


def view(text):
    return default_text(text, macros_in([text]))


def function_def(toks, name):
    """``(tokens before the name, parameter tokens, body tokens, index of the name, index behind the closing brace)`` of the first definition of ``name`` (``name ( .. ) {``), or None."""
    for i in range(len(toks) - 1):
        if toks[i] == name and toks[i + 1] == "(":
            k = match_close(toks, i + 1, "(", ")")
            if k + 1 < len(toks) and toks[k + 1] == "{":
                e = match_close(toks, k + 1, "{", "}")
                return toks[max(0, i - 6):i], toks[i + 2:k], toks[k + 2:e], i, e + 1
    return None


# ---- layout ---------------------------------------------------------------------------------------------------------------------------------------------------
FRAG_STRUCT = norm("""
struct alignas(128) FragSmem {
    uint4 pad0_[4];
    uint4 zero[16][2][4];
    uint4 pad1_[4];
    uint4 ab[16][2][4];
    uint4 am[6][2][2][4];
    uint4 ad[16][2][4];
};""")
ROWSMEM_DEFAULT = norm("""
struct RowSmem {
    float k[HEAD_K]; float q[HEAD_K]; float g2[HEAD_K]; float gref[HEAD_K]; float gend[HEAD_K]; float x0[HEAD_K];
    float m[3][HEAD_V];
    float p1[HEAD_V], p2[HEAD_V], u[HEAD_V];
    float akk[BT], aqk[BT];
    float beta;
};""")
LAYOUT_ASSERTS = [norm(c) for c in (
    "alignof(FragSmem) == 128",
    "offsetof(FragSmem, ab) % 128 == 0 && offsetof(FragSmem, am) % 128 == 0 && offsetof(FragSmem, ad) % 128 == 0",
    "offsetof(FragSmem, zero) % 128 == 64",
    "sizeof(FragSmem::zero) >= sizeof(FragSmem::ab) && sizeof(FragSmem::zero) >= sizeof(FragSmem::am) && sizeof(FragSmem::zero) >= sizeof(FragSmem::ad)",
    "offsetof(FragSmem, zero) + sizeof(FragSmem::zero) <= offsetof(FragSmem, ab) && offsetof(FragSmem, ab) + sizeof(FragSmem::ab) == offsetof(FragSmem, am) && "
    "offsetof(FragSmem, am) + sizeof(FragSmem::am) == offsetof(FragSmem, ad) && offsetof(FragSmem, ad) + sizeof(FragSmem::ad) == sizeof(FragSmem)",
    "ROWS_SMEM_LAYOUT_BYTES <= ROWS_BUILD_STATIC_SMEM_BYTES",
    "ROWS_BUILD_STATIC_SMEM_BYTES - ROWS_SMEM_LAYOUT_BYTES < 1024",
)]
SMEM_LAYOUT = norm("constexpr size_t ROWS_SMEM_LAYOUT_BYTES = rows_smem_up(sizeof(RowSmem), alignof(FragSmem)) + sizeof(FragSmem) + sizeof(int);")


def struct_tokens(toks, head):
    """The tokens of the struct that starts with the tokens ``head`` (``struct .. Name {``), from its first token through the ``;`` behind its closing brace; None if there is none."""
    h = tokens(head)
    for i in range(len(toks) - len(h)):
        if toks[i:i + len(h)] == h:
            e = match_close(toks, i + len(h) - 1, "{", "}")
            return toks[i:e + 2]
    return None


def static_asserts(toks):
    """The condition tokens (up to the top-level comma) of every ``static_assert (..)``, each as one single-spaced string."""
    out = []
    for i, t in enumerate(toks):
        if t == "static_assert" and toks[i + 1] == "(":
            k = match_close(toks, i + 1, "(", ")")
            depth, end = 0, k
            for j in range(i + 2, k):
                depth += toks[j] in "([{"
                depth -= toks[j] in ")]}"
                if toks[j] == "," and depth == 0:
                    end = j
                    break
            out.append(" ".join(toks[i + 2:end]))
    return out


def layout_problems(text):
    toks = tokens(view(text))
    probs = []
    frag = struct_tokens(toks, "struct alignas(128) FragSmem {")
    if frag is None or " ".join(frag) != FRAG_STRUCT:
        probs.append("FragSmem is not the pinned struct (zero page first, pads, ab / am / ad): %s" % (" ".join(frag) if frag else None))
    row = struct_tokens(toks, "struct RowSmem {")
    if row is None or " ".join(row) != ROWSMEM_DEFAULT:
        probs.append("the default RowSmem is not the pinned working set (the oracle's gq / x1 belong to the CFBI_ROWS_TEST_AFRAG_CHECK build): %s" % (" ".join(row) if row else None))
    have = static_asserts(toks)
    for want in LAYOUT_ASSERTS:
        if want not in have:
            probs.append("the layout assert %r is missing" % want)
    if SMEM_LAYOUT not in " ".join(toks):
        probs.append("ROWS_SMEM_LAYOUT_BYTES is not the working set + the fragment tiles + the fill word")
    return probs


def test_the_layout_of_the_staged_fragments_is_pinned(rows_text):
    assert layout_problems(rows_text) == []


@pytest.mark.parametrize("name,old,new,expect", [
    ("the AB tile one k-step short", "    uint4 ab[16][2][4];\n", "    uint4 ab[15][2][4];\n", "not the pinned struct"),
    ("a zero page smaller than a tile", "    uint4 zero[16][2][4];\n", "    uint4 zero[15][2][4];\n", "not the pinned struct"),
    ("a pad dropped", "    uint4 pad0_[4];\n", "", "not the pinned struct"),
    ("a smaller alignment", "struct alignas(128) FragSmem {", "struct alignas(64) FragSmem {", "not the pinned struct"),
    ("the AM tile behind the AD tile", "    uint4 am[6][2][2][4];\n    uint4 ad[16][2][4];\n", "    uint4 ad[16][2][4];\n    uint4 am[6][2][2][4];\n", "not the pinned struct"),
    ("the assert of the zero twin dropped", 'static_assert(offsetof(FragSmem, zero) % 128 == 64, "the zero twin of a 64-byte group lies in the other half of the banks");\n', "", "offsetof ( FragSmem , zero ) % 128 == 64"),
    ("the arena assert dropped", "static_assert(offsetof(FragSmem, zero) + sizeof(FragSmem::zero) <= offsetof(FragSmem, ab)", "static_assert(true || offsetof(FragSmem, zero) + sizeof(FragSmem::zero) <= offsetof(FragSmem, ab)", "the layout assert"),
    ("the need no longer covers the layout", "static_assert(ROWS_SMEM_LAYOUT_BYTES <= ROWS_BUILD_STATIC_SMEM_BYTES,", "static_assert(true,", "ROWS_SMEM_LAYOUT_BYTES <= ROWS_BUILD_STATIC_SMEM_BYTES"),
    ("the oracle operands in the default working set", "#ifdef CFBI_ROWS_TEST_AFRAG_CHECK\n    float gq[HEAD_K];", "#if 1\n    float gq[HEAD_K];", "the default RowSmem"),
    ("the fill word not counted", "+ sizeof(FragSmem) + sizeof(int);", "+ sizeof(FragSmem);", "ROWS_SMEM_LAYOUT_BYTES is not"),
])
def test_a_layout_defect_is_caught(rows_text, name, old, new, expect):
    probs = layout_problems(mutate(rows_text, old, new))
    assert any(expect in p for p in probs), (name, probs)


# ---- the helpers: the only code that touches the tiles --------------------------------------------------------------------------------------------------------
PTR_TAIL = "afrag_anchor((mine ? afrag_shared(&Fr.%s) : afrag_shared(&Fr.zero[0][0][0])) + 16u * static_cast<uint32_t>(t));"
HELPERS = {
    "afrag_zero_init": ("__device__ __forceinline__ void", "FragSmem& Fr, int tid", """
        if (tid < 128) reinterpret_cast<uint4*>(&Fr.zero[0][0][0])[tid] = make_uint4(0u, 0u, 0u, 0u);"""),
    "afrag_lds": ("__device__ __forceinline__ uint4", "uint32_t addr", """
        uint4 v;
        asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr) : "memory");
        return v;"""),
    "afrag_shared": ("__device__ __forceinline__ uint32_t", "const void* p", "return static_cast<uint32_t>(__cvta_generic_to_shared(p));"),
    "afrag_anchor": ("__device__ __forceinline__ uint32_t", "uint32_t addr", """
        uint32_t r;
        asm volatile("mov.u32 %0, %1;" : "=r"(r) : "r"(addr));
        return r;"""),
    "afrag_ptr_ab": ("__device__ __forceinline__ uint32_t", "const FragSmem& Fr, bool mine, int t", "return " + PTR_TAIL % "ab[0][0][0]"),
    "afrag_ptr_am": ("__device__ __forceinline__ uint32_t", "const FragSmem& Fr, bool mine, int t", "return " + PTR_TAIL % "am[0][0][0][0]"),
    "afrag_ptr_ad": ("__device__ __forceinline__ uint32_t", "const FragSmem& Fr, bool mine, int t", "return " + PTR_TAIL % "ad[0][0][0]"),
    "afrag_store_ab": ("__device__ __forceinline__ void", "FragSmem& Fr, int ch, int slot_hi, uint32_t kh, uint32_t kl, uint32_t qh, uint32_t ql", """
        const int ks = ch >> 3, half = (ch >> 2) & 1;
        const int tt = ch & 3;
        const int iik = slot_hi + 2 * half, iiq = (1 - slot_hi) + 2 * half;
        uint32_t* wh = reinterpret_cast<uint32_t*>(&Fr.ab[ks][0][tt]);
        uint32_t* wl = reinterpret_cast<uint32_t*>(&Fr.ab[ks][1][tt]);
        wh[iik] = kh;
        wl[iik] = kl;
        wh[iiq] = qh;
        wl[iiq] = ql;"""),
    "afrag_store_am": ("__device__ __forceinline__ void", "FragSmem& Fr, int w, int r, int slot_hi, float akk_v, float aqk_v", """
        const int tt = r & 3, half = r >> 2, iiv = slot_hi + 2 * half, iiz = (1 - slot_hi) + 2 * half;
        uint32_t h, l;
        split_tf32(akk_v, h, l);
        reinterpret_cast<uint32_t*>(&Fr.am[w][0][0][tt])[iiv] = h;
        reinterpret_cast<uint32_t*>(&Fr.am[w][0][1][tt])[iiv] = l;
        reinterpret_cast<uint32_t*>(&Fr.am[w][0][0][tt])[iiz] = 0u;
        reinterpret_cast<uint32_t*>(&Fr.am[w][0][1][tt])[iiz] = 0u;
        split_tf32(aqk_v, h, l);
        reinterpret_cast<uint32_t*>(&Fr.am[w][1][0][tt])[iiv] = h;
        reinterpret_cast<uint32_t*>(&Fr.am[w][1][1][tt])[iiv] = l;
        reinterpret_cast<uint32_t*>(&Fr.am[w][1][0][tt])[iiz] = 0u;
        reinterpret_cast<uint32_t*>(&Fr.am[w][1][1][tt])[iiz] = 0u;"""),
    "afrag_store_ad": ("__device__ __forceinline__ void", "FragSmem& Fr, int col, int which, int slot_hi, float val", """
        const int ks = col >> 3, tt = col & 3, half = (col >> 2) & 1, ii = (which ? slot_hi : 1 - slot_hi) + 2 * half;
        uint32_t h, l;
        split_tf32(val, h, l);
        reinterpret_cast<uint32_t*>(&Fr.ad[ks][0][tt])[ii] = h;
        reinterpret_cast<uint32_t*>(&Fr.ad[ks][1][tt])[ii] = l;"""),
}
# the code in which a tile member may be named: the helpers' own bodies
TILE_MEMBERS = ("zero", "ab", "am", "ad", "pad0_", "pad1_")


def helper_problems(text):
    toks = tokens(view(text))
    probs = []
    for name, (ret, params, body) in HELPERS.items():
        d = function_def(toks, name)
        if d is None:
            probs.append("the helper %s is missing" % name)
            continue
        before, got_params, got_body, _, _ = d
        if " ".join(before[-len(tokens(ret)):]) != norm(ret):
            probs.append("%s does not return %r: %s" % (name, ret, " ".join(before)))
        if " ".join(got_params) != norm(params):
            probs.append("%s takes (%s) instead of (%s)" % (name, " ".join(got_params), params))
        if " ".join(got_body) != norm(body):
            probs.append("the body of %s is not the pinned one: %s" % (name, " ".join(got_body)))
    return probs


def test_the_helpers_of_the_staged_fragments_are_the_pinned_ones(rows_text):
    assert helper_problems(rows_text) == []


@pytest.mark.parametrize("name,old,new,helper", [
    ("the zero page written by 64 threads only", "if (tid < 128) reinterpret_cast<uint4*>(&Fr.zero[0][0][0])[tid] = make_uint4(0u, 0u, 0u, 0u);", "if (tid < 64) reinterpret_cast<uint4*>(&Fr.zero[0][0][0])[tid] = make_uint4(0u, 0u, 0u, 0u);", "afrag_zero_init"),
    ("the zero page filled with ones", "[tid] = make_uint4(0u, 0u, 0u, 0u);\n#ifdef", "[tid] = make_uint4(1u, 0u, 0u, 0u);\n#ifdef", "afrag_zero_init"),
    ("the AB pointer of every lane is the tile", "afrag_anchor((mine ? afrag_shared(&Fr.ab[0][0][0]) : afrag_shared(&Fr.zero[0][0][0])) + 16u * static_cast<uint32_t>(t));", "afrag_anchor(afrag_shared(&Fr.ab[0][0][0]) + 16u * static_cast<uint32_t>(t));", "afrag_ptr_ab"),
    ("the AM pointer of the lanes that do not own the row is the AD tile", "afrag_anchor((mine ? afrag_shared(&Fr.am[0][0][0][0]) : afrag_shared(&Fr.zero[0][0][0])) + 16u * static_cast<uint32_t>(t));", "afrag_anchor((mine ? afrag_shared(&Fr.am[0][0][0][0]) : afrag_shared(&Fr.ad[0][0][0])) + 16u * static_cast<uint32_t>(t));", "afrag_ptr_am"),
    ("the AD pointer without the lane group", "afrag_anchor((mine ? afrag_shared(&Fr.ad[0][0][0]) : afrag_shared(&Fr.zero[0][0][0])) + 16u * static_cast<uint32_t>(t));", "afrag_anchor((mine ? afrag_shared(&Fr.ad[0][0][0]) : afrag_shared(&Fr.zero[0][0][0])));", "afrag_ptr_ad"),
    ("the AD pointer in words, not bytes", "afrag_anchor((mine ? afrag_shared(&Fr.ad[0][0][0]) : afrag_shared(&Fr.zero[0][0][0])) + 16u * static_cast<uint32_t>(t));", "afrag_anchor((mine ? afrag_shared(&Fr.ad[0][0][0]) : afrag_shared(&Fr.zero[0][0][0])) + 4u * static_cast<uint32_t>(t));", "afrag_ptr_ad"),
    ("the staged load without its ordering (not volatile)", "asm volatile(\"ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\"", "asm(\"ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\"", "afrag_lds"),
    ("the staged load without its memory clobber", ": \"r\"(addr) : \"memory\");", ": \"r\"(addr));", "afrag_lds"),
    ("the staged load as two 64-bit loads", "asm volatile(\"ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\"", "asm volatile(\"ld.shared.v2.u32 {%0, %1}, [%4];\"", "afrag_lds"),
    ("the anchor of a base not volatile", "asm volatile(\"mov.u32 %0, %1;\"", "asm(\"mov.u32 %0, %1;\"", "afrag_anchor"),
    ("the shared address cast dropped", "return static_cast<uint32_t>(__cvta_generic_to_shared(p));", "return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p));", "afrag_shared"),
    ("the AB stores shifted by one lane", "    const int tt = ch & 3;\n", "    const int tt = (ch + 1) & 3;\n", "afrag_store_ab"),
    ("the AB rows swapped", "const int iik = slot_hi + 2 * half, iiq = (1 - slot_hi) + 2 * half;", "const int iik = (1 - slot_hi) + 2 * half, iiq = slot_hi + 2 * half;", "afrag_store_ab"),
    ("the AB lo word of the qs row dropped", "    wl[iiq] = ql;\n", "", "afrag_store_ab"),
    ("the AB half of the element index dropped", "const int iik = slot_hi + 2 * half, iiq", "const int iik = slot_hi, iiq", "afrag_store_ab"),
    ("the AM zero words not stored", "    reinterpret_cast<uint32_t*>(&Fr.am[w][0][0][tt])[iiz] = 0u;\n", "", "afrag_store_am"),
    ("the AM aqk hi word into the akk row", "reinterpret_cast<uint32_t*>(&Fr.am[w][1][0][tt])[iiv] = h;", "reinterpret_cast<uint32_t*>(&Fr.am[w][0][0][tt])[iiv] = h;", "afrag_store_am"),
    ("the AM value stored for the wrong parity", "iiv = slot_hi + 2 * half, iiz = (1 - slot_hi) + 2 * half;", "iiv = (1 - slot_hi) + 2 * half, iiz = slot_hi + 2 * half;", "afrag_store_am"),
    ("the AD W and GQ parities swapped", "ii = (which ? slot_hi : 1 - slot_hi) + 2 * half;", "ii = (which ? 1 - slot_hi : slot_hi) + 2 * half;", "afrag_store_ad"),
    ("the AD lo word of the hi tile", "reinterpret_cast<uint32_t*>(&Fr.ad[ks][1][tt])[ii] = l;", "reinterpret_cast<uint32_t*>(&Fr.ad[ks][1][tt])[ii] = h;", "afrag_store_ad"),
    ("the zero page written by a producer", "#ifdef CFBI_ROWS_TEST_ZERO_TILE_WRITE\n    if (ch == 0)", "#if 1\n    if (ch == 0)", "afrag_store_ab"),
    ("the zero page of the AM tile rewritten once by default", "#ifdef CFBI_ROWS_TEST_AM_STALE_ZERO\n    if (tid < 96)", "#if 1\n    if (tid < 96)", "afrag_zero_init"),
    ("every lane reads the tile by default", "#ifdef CFBI_ROWS_TEST_NONMINE_TILE\n    (void)mine;   // control: the lanes that do not own the row read the tile\n    return afrag_anchor(afrag_shared(&Fr.ab[0][0][0]) + 16u * static_cast<uint32_t>(t));\n#else",
     "#if 1\n    (void)mine;   // control: the lanes that do not own the row read the tile\n    return afrag_anchor(afrag_shared(&Fr.ab[0][0][0]) + 16u * static_cast<uint32_t>(t));\n#else", "afrag_ptr_ab"),
])
def test_a_defect_in_a_helper_is_caught(rows_text, name, old, new, helper):
    probs = helper_problems(mutate(rows_text, old, new))
    assert any(helper in p for p in probs), (name, probs)


# ---- access: nothing but the helpers names a tile ----------------------------------------------------------------------------------------------------------------
def access_problems(text):
    toks = tokens(view(text))
    spans = [function_def(toks, name)[3:5] for name in HELPERS if function_def(toks, name) is not None]
    probs = []
    for i, t in enumerate(toks):
        inside = any(a <= i < b for a, b in spans)
        if t == "Fr" and i + 2 < len(toks) and toks[i + 1] == "." and toks[i + 2] in TILE_MEMBERS and not inside:
            probs.append("a tile member is named outside the helpers: %s" % " ".join(toks[max(0, i - 4):i + 8]))
        elif t == "Fr" and not inside:
            nxt, prev = toks[i + 1] if i + 1 < len(toks) else "", toks[i - 1] if i else ""
            declared = prev == "FragSmem" and nxt == ";"
            passed = nxt == "," and prev == "(" and toks[i - 2].startswith("afrag_")
            if not declared and not passed:
                probs.append("Fr is used other than by the helpers: %s" % " ".join(toks[max(0, i - 4):i + 8]))
        elif t == "FragSmem" and not inside and toks[i - 1] not in ("struct", ")", "sizeof", "alignof", "(", ",", "+", "::", "__shared__"):
            probs.append("FragSmem is used outside its definition: %s" % " ".join(toks[max(0, i - 4):i + 8]))
    declared = " ".join(toks).count("__shared__ FragSmem Fr ;")
    if declared != 1 or " ".join(toks).count("__shared__ FragSmem") != 1:
        probs.append("the fragment tiles are not declared exactly once, as __shared__ FragSmem Fr (%d declarations of Fr)" % declared)
    return probs


def test_nothing_but_the_helpers_names_a_tile(rows_text):
    assert access_problems(rows_text) == []


def test_no_other_source_names_the_fragment_tiles():
    for f in sorted(CSRC.rglob("*")):
        if f.suffix in (".cu", ".cuh", ".h", ".cpp") and f.name != "cfbi_rows.cuh":
            assert "FragSmem" not in f.read_text(), f.name


@pytest.mark.parametrize("name,old,new", [
    ("a consumer reads the tile itself", "uint4 nh = afrag_lds(abp), nl = afrag_lds(abp + 64);", "uint4 nh = Fr.ab[0][0][t], nl = Fr.ab[0][1][t];"),
    ("a store of the tile in a phase", "        ROWS_SYNC(3);\n        }\n        ROWS_TS(3);\n", "        ROWS_SYNC(3);\n        }\n        Fr.am[0][0][0][0].x = 0u;\n        ROWS_TS(3);\n"),
    ("the zero page written in the kernel", "    afrag_zero_init(Fr, tid);", "    Fr.zero[0][0][0].x = 0u;\n    afrag_zero_init(Fr, tid);"),
    ("the object passed to a function of another name", "    afrag_zero_init(Fr, tid);", "    afrag_zero_init(Fr, tid);\n    helper_copy(Fr, tid);"),
    ("a second copy of the tiles", "    __shared__ FragSmem Fr;\n", "    __shared__ FragSmem Fr;\n    __shared__ FragSmem Fr2;\n"),
])
def test_a_tile_access_outside_the_helpers_is_caught(rows_text, name, old, new):
    probs = access_problems(mutate(rows_text, old, new))
    assert probs, name


# ---- the consumers: volatile shared loads through a base that the phase computes itself, in the order of the source ---------------------------------------------------------
# line -> how often it is in the kernel (AB and AD have the same words per k-step, the AM loads are in rows_m_tiles)
CONSUMER_LINES = {
    "uint4 nh = afrag_lds(abp), nl = afrag_lds(abp + 64);": 1,                       # the words of k-step 0 of AB ...
    "uint4 nh = afrag_lds(adp), nl = afrag_lds(adp + 64);": 1,                       # ... and of AD
    "const uint4 vh = nh, vl = nl;": 2,
    "if (ks < 15) { nh = afrag_lds(abp + 128 * (ks + 1)); nl = afrag_lds(abp + 128 * (ks + 1) + 64); }": 1,      # one k-step ahead of the products
    "if (ks < 15) { nh = afrag_lds(adp + 128 * (ks + 1)); nl = afrag_lds(adp + 128 * (ks + 1) + 64); }": 1,
    "const uint4 vkh = afrag_lds(amp + ks * 256), vkl = afrag_lds(amp + ks * 256 + 64);": 1,                       # the akk words: the top of the k-step
    "const uint4 vqh = afrag_lds(amp + ks * 256 + 128), vql = afrag_lds(amp + ks * 256 + 192);": 1,                # the aqk words: where the first product that needs them starts
}
POINTER_LINES = (
    "const uint32_t abp = afrag_ptr_ab(Fr, mine, t);",
    "const uint32_t amp = afrag_ptr_am(Fr, mine, t);",
    "const uint32_t adp = afrag_ptr_ad(Fr, mine, t);",
    "const bool mine = g == (slot & 7);",
    "const int slot_hi = slot >> 3;",
)
# a staged word is read by afrag_lds only: no other load of a base, a tile or the zero page (the helpers' goldens say what afrag_lds is)
BASE_NAMES = {"abp": 5, "adp": 5}     # the base, the two words of k-step 0, the two words of the prefetch
# the A side of the products of phases B, C and D is loaded, never split again. The splits of the kernel body: the AB producer (2), the B sides of phases B and D (1 + 1) and the two of the chunk-end
# phase (its A fragment and its B side, not staged by F); rows_m_tiles splits the B side of the cross-sub-block products only
SPLIT_CALLS = {"rows_kernel": 6, "rows_m_tiles": 1}


def am_split_problems(text):
    """The aqk words of rows_m_tiles are loaded inside the ``q == 4`` block (the first tile of the aqk products), the akk words in front of the q loop: the two pairs are never live together."""
    d = function_def(tokens(view(text)), "rows_m_tiles")
    if d is None:
        return ["rows_m_tiles is missing"]
    body = d[2]
    probs = []
    names = [i for i, t in enumerate(body) if t in ("vkh", "vqh")]
    blocks = [i for i in range(len(body) - 5) if body[i:i + 5] == ["if", "(", "q", "==", "4"]]
    if len(blocks) != 1 or len(names) < 2:
        return probs + ["rows_m_tiles has %d blocks of q == 4 (expected one)" % len(blocks)]
    brace = body.index("{", blocks[0])
    end = match_close(body, brace, "{", "}")
    inside = [t for t in body[brace:end] if t in ("vkh", "vqh")]
    if set(inside) != {"vqh"}:
        probs.append("the q == 4 block of rows_m_tiles does not hold the aqk loads alone: %s" % inside)
    top = [t for t in body[:blocks[0]] if t in ("vkh", "vqh")]
    if "vqh" in top or "vkh" not in top:
        probs.append("the akk loads are not in front of the q loop, or the aqk loads are: %s" % top)
    return probs


def test_the_aqk_words_are_loaded_where_the_akk_words_are_dead(rows_text):
    assert am_split_problems(rows_text) == []


@pytest.mark.parametrize("name,old,new", [
    ("the aqk words loaded with the first tile", "if (q == 4) {   // the aqk words", "if (q == 0) {   // the aqk words"),
    ("the aqk words loaded behind their first product", "if (q == 4) {   // the aqk words", "if (q == 5) {   // the aqk words"),
    ("the aqk words loaded at the top again", "const uint4 vkh = afrag_lds(amp + ks * 256), vkl = afrag_lds(amp + ks * 256 + 64);",
     "const uint4 vqh = afrag_lds(amp + ks * 256 + 128), vkh = afrag_lds(amp + ks * 256), vkl = afrag_lds(amp + ks * 256 + 64);"),
])
def test_an_aqk_load_that_is_not_where_the_akk_words_are_dead_is_caught(rows_text, name, old, new):
    assert am_split_problems(mutate(rows_text, old, new)), name


def consumer_problems(text):
    toks = tokens(view(text))
    joined = " " + " ".join(toks) + " "
    probs = []
    for line, want in list(CONSUMER_LINES.items()) + [(l, 1) for l in POINTER_LINES]:
        n = joined.count(" " + norm(line) + " ")
        if n != want:
            probs.append("the pinned line %r occurs %d times (expected %d)" % (line, n, want))
    for base, want in BASE_NAMES.items():
        if toks.count(base) != want:
            probs.append("%s is named %d times (expected %d: the base, the words of k-step 0, the words of the prefetch)" % (base, toks.count(base), want))
    if toks.count("afrag_lds") != 1 + 2 + 2 + 2 + 2 + 4:      # the definition, k-step 0 and prefetch of AB and of AD (two words each), the four words of a k-step of AM
        probs.append("afrag_lds is named %d times: a staged word is loaded in another way or place" % toks.count("afrag_lds"))
    for name, want in SPLIT_CALLS.items():
        d = function_def(toks, name)
        got = None if d is None else d[2].count("split_tf32")
        if got != want:
            probs.append("%s calls split_tf32 %s times, expected %d (a fragment of the A side is split again)" % (name, got, want))
    return probs


def test_the_consumers_load_the_staged_words_only(rows_text):
    assert consumer_problems(rows_text) == []


@pytest.mark.parametrize("name,old,new,expect", [
    ("an AB prefetch of the lo word at the wrong offset", "nl = afrag_lds(abp + 128 * (ks + 1) + 64); }", "nl = afrag_lds(abp + 128 * (ks + 1) + 32); }", "abp + 128 * (ks + 1) + 64"),
    ("an AD prefetch of the wrong k-step stride", "nh = afrag_lds(adp + 128 * (ks + 1)); nl", "nh = afrag_lds(adp + 64 * (ks + 1)); nl", "adp + 128 * (ks + 1)"),
    ("an AB prefetch of the last k-step", "if (ks < 15) { nh = afrag_lds(abp + 128 * (ks + 1));", "if (ks < 16) { nh = afrag_lds(abp + 128 * (ks + 1));", "abp + 128 * (ks + 1)"),
    ("the words of k-step 0 of AD loaded from the AB base", "uint4 nh = afrag_lds(adp), nl = afrag_lds(adp + 64);", "uint4 nh = afrag_lds(abp), nl = afrag_lds(abp + 64);", "adp"),
    ("an AM load of the aqk lo word at the wrong offset", "vql = afrag_lds(amp + ks * 256 + 192)", "vql = afrag_lds(amp + ks * 256 + 191)", "amp + ks * 256 + 192"),
    ("an AM load through a scalar read", "vkh = afrag_lds(amp + ks * 256),", "vkh = *reinterpret_cast<const uint4*>(amp + ks * 256),", "afrag_lds"),
    ("the AB base from the AM family", "const uint32_t abp = afrag_ptr_ab(Fr, mine, t);", "const uint32_t abp = afrag_ptr_am(Fr, mine, t);", "afrag_ptr_ab"),
    ("the AB base carried from the top of the row", "            const uint32_t abp = afrag_ptr_ab(Fr, mine, t);   // the base of the loads, computed here: the tile of the lanes that own row slot, the zero page for the others\n",
     "            const uint32_t abp = afrag_shared(&Fr.zero[0][0][0]);\n", "afrag_ptr_ab"),
    ("the lanes that own the row chosen by another test", "const bool mine = g == (slot & 7);", "const bool mine = g == (slot & 3);", "const bool mine"),
    ("the AB words split again by the consumer", "                uint32_t bh[2], bl[2];\n#pragma unroll\n                for (int ii = 0; ii < 2; ++ii) {\n                    const int ch = 8 * ks + t + 4 * ii;",
     "                split_tf32(0.f, ah[0], al[0]);\n                uint32_t bh[2], bl[2];\n#pragma unroll\n                for (int ii = 0; ii < 2; ++ii) {\n                    const int ch = 8 * ks + t + 4 * ii;", "split_tf32"),
])
def test_a_consumer_defect_is_caught(rows_text, name, old, new, expect):
    probs = consumer_problems(mutate(rows_text, old, new))
    assert any(expect in p for p in probs), (name, probs)


# ---- windows: every store in front of the barrier that publishes the tile, every load behind it ---------------------------------------------------------------------
EVENTS = (
    ("ZINIT", re.compile(r"^afrag_zero_init\(Fr, tid\);")),
    ("LOOP", re.compile(r"^for \(int i = 0; i < nrows; \+\+i\) \{")),
    ("ANCH_AB", re.compile(r"^const uint32_t abp = ")),
    ("ANCH_AM", re.compile(r"^const uint32_t amp = ")),
    ("ANCH_AD", re.compile(r"^const uint32_t adp = ")),
    ("S13", re.compile(r"^ROWS_SYNC_AB\(\);")),
    ("S7", re.compile(r"^ROWS_SYNC_END\(\);")),
    ("S%s", re.compile(r"^ROWS_SYNC\((\d)\);")),
    ("ST_AB", re.compile(r"\bafrag_store_ab\(Fr, ")),
    ("ST_AM", re.compile(r"\bafrag_store_am\(Fr, ")),
    ("ST_AD", re.compile(r"\bafrag_store_ad\(Fr, ")),
    ("LD_B", re.compile(r"\bRk\[jc \* HEAD_K \+ ch\]")),             # the loads of the B operands of phase B: behind barrier 13, in front of their use
    ("LD_AB", re.compile(r"\bafrag_lds\(abp\b")),
    ("LD_AD", re.compile(r"\bafrag_lds\(adp\b")),
    ("LD_AM", re.compile(r"\brows_m_tiles<\d>\(S, amp, ")),
)
GOLDEN_EVENTS = ["ZINIT", "LOOP", "S1", "ST_AB", "S13", "LD_B", "ANCH_AB", "LD_AB", "LD_AB", "ST_AM", "S2", "ANCH_AM", "LD_AM", "LD_AM", "LD_AM", "S3", "ST_AD", "S4", "ANCH_AD", "LD_AD", "LD_AD", "S5", "S6", "S7"]


def events(text):
    code = trp.code_lines(trp.kernel_body(view(text)))
    out = []
    for k, line in code:
        for name, pat in EVENTS:
            m = pat.search(line)
            if m:
                out.append(name % m.group(1) if "%s" in name else name)
                break
    return out


def test_every_store_of_a_family_is_in_front_of_its_barrier_and_every_load_behind_it(rows_text):
    assert events(rows_text) == GOLDEN_EVENTS


def moved(text, line, anchor, after=True, keep=False):
    """``text`` with ``line`` (a whole source line) moved to just behind (or in front of) the line ``anchor``; ``keep`` leaves the original in place (a copy)."""
    out = text if keep else mutate(text, line, "")
    return mutate(out, anchor, anchor + line if after else line + anchor)


AB_STORE = "            afrag_store_ab(Fr, tid, slot_hi, kh, kl, qh, ql);\n"
AB_LOAD = "            uint4 nh = afrag_lds(abp), nl = afrag_lds(abp + 64);   // the words of k-step 0\n"
AM_STORE = "                    if (warp < 2 * s) afrag_store_am(Fr, warp, 2 * t + e, slot_hi, akk_v, aqk_v);   // the A fragments of the cross-sub-block products: the columns below 16 s\n"
AD_STORE = "            afrag_store_ad(Fr, col, which, slot_hi, gw);\n"
AD_LOAD = "            uint4 nh = afrag_lds(adp), nl = afrag_lds(adp + 64);   // the words of k-step 0\n"
ZERO_INIT = "    afrag_zero_init(Fr, tid);   // the +0 page of the staged A fragments: written once here, ordered by the first barrier of the first row\n"
AB_POINTER = "            const uint32_t abp = afrag_ptr_ab(Fr, mine, t);   // the base of the loads, computed here: the tile of the lanes that own row slot, the zero page for the others\n"
AM_POINTER = "            const uint32_t amp = afrag_ptr_am(Fr, mine, t);   // the base of the loads of the AM tile, computed here\n"
B_LOAD = "                    const float ak = Rk[jc * HEAD_K + ch], bg = Rg[jc * HEAD_K + ch];\n"


@pytest.mark.parametrize("name,make", [
    ("barrier 13 dropped", lambda t: mutate(t, "        ROWS_SYNC_AB();\n", "")),
    ("barrier 13 doubled", lambda t: mutate(t, "        ROWS_SYNC_AB();\n", "        ROWS_SYNC_AB();\n        ROWS_SYNC_AB();\n")),
    ("barrier 13 behind the AB loads", lambda t: moved(t, "        ROWS_SYNC_AB();\n", AB_LOAD)),
    ("the AB stores behind barrier 13", lambda t: moved(t, AB_STORE, "        ROWS_SYNC_AB();\n")),
    ("the AB stores behind barrier 2", lambda t: moved(t, AB_STORE, "        ROWS_SYNC(2);\n")),
    ("the AB stores twice", lambda t: moved(t, AB_STORE, AB_STORE, keep=True)),
    ("the AM stores missing", lambda t: mutate(t, AM_STORE, "")),
    ("the AM stores behind barrier 2", lambda t: moved(t, AM_STORE, "        ROWS_TS(2);\n")),
    ("the AD stores missing", lambda t: mutate(t, AD_STORE, "")),
    ("the AD stores behind barrier 4", lambda t: moved(t, AD_STORE, "        ROWS_TS(4);\n")),
    ("barrier 4 behind the AD loads", lambda t: moved(t, "        ROWS_SYNC(4);\n", AD_LOAD)),
    ("the zero page written in the row loop", lambda t: moved(t, ZERO_INIT, "        ROWS_SYNC(1);\n")),
    ("the zero page never written", lambda t: mutate(t, ZERO_INIT, "")),
    ("the zero page written as the first statement of the row loop", lambda t: moved(t, ZERO_INIT, "    for (int i = 0; i < nrows; ++i) {\n")),
    ("the AB base computed in front of barrier 13", lambda t: moved(t, AB_POINTER, "        ROWS_SYNC(1);\n")),
    ("the AM base computed in front of barrier 2", lambda t: moved(t, AM_POINTER, "        ROWS_SYNC(1);\n")),
    ("the AM base computed behind the first loads", lambda t: moved(t, AM_POINTER, "            if (s == 1) rows_m_tiles<2>(S, amp, Rut, Rw, Ru, warp, L, slot);\n")),
    ("the B operands loaded in front of barrier 13", lambda t: mutate(t, "        ROWS_SYNC_AB();\n", B_LOAD + "        ROWS_SYNC_AB();\n")),
    ("the B operands never loaded behind barrier 13", lambda t: mutate(t, B_LOAD, "")),
])
def test_a_store_or_load_on_the_wrong_side_of_its_barrier_is_caught(rows_text, name, make):
    assert events(make(rows_text)) != GOLDEN_EVENTS, name


# ---- the oracle: a consumer that is not checked, or a check that never counts, is blind ----------------------------------------------------------------------------
UNPACK_A = (r"ah\[0\] = vh\.x; ah\[1\] = vh\.y; ah\[2\] = vh\.z; ah\[3\] = vh\.w;\n\s*"
            r"al\[0\] = vl\.x; al\[1\] = vl\.y; al\[2\] = vl\.z; al\[3\] = vl\.w;\n")
UNPACK_AMK = (r"kh\[0\] = vkh\.x; kh\[1\] = vkh\.y; kh\[2\] = vkh\.z; kh\[3\] = vkh\.w;\n\s*"
              r"kl\[0\] = vkl\.x; kl\[1\] = vkl\.y; kl\[2\] = vkl\.z; kl\[3\] = vkl\.w;\n")
UNPACK_AMQ = (r"qh\[0\] = vqh\.x; qh\[1\] = vqh\.y; qh\[2\] = vqh\.z; qh\[3\] = vqh\.w;\n\s*"
              r"ql\[0\] = vql\.x; ql\[1\] = vql\.y; ql\[2\] = vql\.z; ql\[3\] = vql\.w;\n")
GUARDED = r"#ifdef CFBI_ROWS_TEST_AFRAG_CHECK\n\s*%s;\n#endif\n"
PREFETCH = r"const uint4 vh = nh, vl = nl;\n\s*if \(ks < 15\) \{ nh = afrag_lds\(%s \+ 128 \* \(ks \+ 1\)\); nl = afrag_lds\(%s \+ 128 \* \(ks \+ 1\) \+ 64\); \}[^\n]*\n\s*"
ORACLE_CALLS = {
    "afrag_check_ab": PREFETCH % ("abp", "abp") + UNPACK_A + GUARDED % r"afrag_check_ab\(S, mine, slot_hi, t, ks, ah, al\)",
    "afrag_check_ad": PREFETCH % ("adp", "adp") + UNPACK_A + GUARDED % r"afrag_check_ad\(S, mine, slot_hi, t, ks, ah, al\)",
    "afrag_check_am_k": r"vkl = afrag_lds\(amp \+ ks \* 256 \+ 64\);[^\n]*\n\s*" + UNPACK_AMK + GUARDED % r"afrag_check_am_k\(S, L, slot, ks, kh, kl\)",
    "afrag_check_am_q": r"vql = afrag_lds\(amp \+ ks \* 256 \+ 192\);\n\s*" + UNPACK_AMQ + GUARDED % r"afrag_check_am_q\(S, L, slot, ks, qh, ql\)",
}
ORACLE_COUNTS = {
    "afrag_check_ab": "afrag_count(h != ah[ii] || l != al[ii]);",
    "afrag_check_ad": "afrag_count(h != ah[ii] || l != al[ii]);",
    "afrag_check_am_k": "afrag_count(h != kh[ii] || l != kl[ii]);",
    "afrag_check_am_q": "afrag_count(h != qh[ii] || l != ql[ii]);",
}


def oracle_problems(text):
    probs = []
    for name, pat in ORACLE_CALLS.items():
        n = len(re.findall(pat, text))
        if n != 1:
            probs.append("%s is not called exactly once, inside the oracle guard and right behind the unpack of the words that the consumer loaded (%d matches)" % (name, n))
    toks = tokens(text)
    for name, count in ORACLE_COUNTS.items():
        d = function_def(toks, name)
        if d is None:
            probs.append("the check %s is missing" % name)
        elif (" " + norm(count) + " ") not in (" " + " ".join(d[2]) + " "):
            probs.append("the check %s does not count the words that differ from the unstaged fragment (%s)" % (name, count))
    return probs


def test_every_consumer_family_is_checked_by_the_oracle(rows_text):
    assert oracle_problems(rows_text) == []


@pytest.mark.parametrize("name,old,new,expect", [
    ("the AB check dropped", "#ifdef CFBI_ROWS_TEST_AFRAG_CHECK\n                afrag_check_ab(S, mine, slot_hi, t, ks, ah, al);\n#endif\n", "", "afrag_check_ab is not called"),
    ("the AD check dropped", "#ifdef CFBI_ROWS_TEST_AFRAG_CHECK\n                afrag_check_ad(S, mine, slot_hi, t, ks, ah, al);\n#endif\n", "", "afrag_check_ad is not called"),
    ("the AM akk check dropped", "#ifdef CFBI_ROWS_TEST_AFRAG_CHECK\n        afrag_check_am_k(S, L, slot, ks, kh, kl);\n#endif\n", "", "afrag_check_am_k is not called"),
    ("the AM aqk check dropped", "#ifdef CFBI_ROWS_TEST_AFRAG_CHECK\n                afrag_check_am_q(S, L, slot, ks, qh, ql);\n#endif\n", "", "afrag_check_am_q is not called"),
    ("the AB check outside its guard", "#ifdef CFBI_ROWS_TEST_AFRAG_CHECK\n                afrag_check_ab(S, mine, slot_hi, t, ks, ah, al);\n#endif\n", "                afrag_check_ab(S, mine, slot_hi, t, ks, ah, al);\n", "afrag_check_ab is not called"),
    ("the AB check behind the products", "#ifdef CFBI_ROWS_TEST_AFRAG_CHECK\n                afrag_check_ab(S, mine, slot_hi, t, ks, ah, al);\n#endif\n", "", "afrag_check_ab is not called"),
    ("the AD check looks at the AB words of the other row", "afrag_check_ad(S, mine, slot_hi, t, ks, ah, al);", "afrag_check_ad(S, mine, 1 - slot_hi, t, ks, ah, al);", "afrag_check_ad is not called"),
    ("the AB check never counts", "        const float v = mine ? (((ii & 1) == slot_hi) ? ksv : qsv) : 0.f;\n        uint32_t h, l;\n        split_tf32(v, h, l);\n        afrag_count(h != ah[ii] || l != al[ii]);",
     "        const float v = mine ? (((ii & 1) == slot_hi) ? ksv : qsv) : 0.f;\n        uint32_t h, l;\n        split_tf32(v, h, l);\n        afrag_count(false);", "the check afrag_check_ab does not count"),
    ("the AM akk check counts the hi words only", "afrag_count(h != kh[ii] || l != kl[ii]);", "afrag_count(h != kh[ii]);", "the check afrag_check_am_k does not count"),
    ("the AM aqk check counts the lo words only", "afrag_count(h != qh[ii] || l != ql[ii]);", "afrag_count(l != ql[ii]);", "the check afrag_check_am_q does not count"),
    ("the AD check compares the word with itself", "        const float v = mine ? (((ii & 1) == slot_hi) ? wv : gv) : 0.f;\n        uint32_t h, l;\n        split_tf32(v, h, l);\n        afrag_count(h != ah[ii] || l != al[ii]);",
     "        const float v = mine ? (((ii & 1) == slot_hi) ? wv : gv) : 0.f;\n        uint32_t h, l;\n        split_tf32(v, h, l);\n        afrag_count(ah[ii] != ah[ii] || al[ii] != al[ii]);", "the check afrag_check_ad does not count"),
])
def test_a_blind_oracle_is_caught(rows_text, name, old, new, expect):
    if name == "the AB check behind the products":
        text = mutate(mutate(rows_text, old, ""), "                mma_tf32_m16n8k8(big, ah, bh);\n            }\n            if (mine) {\n#pragma unroll\n                for (int e = 0; e < 2; ++e) {\n                    const int c = j0",
                      "                mma_tf32_m16n8k8(big, ah, bh);\n" + old + "            }\n            if (mine) {\n#pragma unroll\n                for (int e = 0; e < 2; ++e) {\n                    const int c = j0")
    else:
        text = mutate(rows_text, old, new)
    probs = oracle_problems(text)
    assert any(expect in p for p in probs), (name, probs)


# ---- the hit model of the delay tests for the two new sites (CPU: the device tests compare the hits that they read back with it) ------------------------------------
class _Step:
    def __init__(self, lens, H):
        self.lens, self.N, self.H = lens, len(lens), H

    def live(self, n):
        return self.lens[n] > 0

    def slot(self, n):
        return n


class _Warm:
    def __init__(self, pos):
        self.pos = pos


def delay_model():
    import test_rows_delay as trd

    return trd, _Step((3, 1, 0, 2, 4, 1), 4), _Warm([5, 63, 9, 20, 17, 0])


@pytest.mark.parametrize("site,passing", [(13, 8), (14, 4)])
def test_the_delay_model_counts_the_warps_that_pass_a_new_site(site, passing):
    trd, step, warm = delay_model()
    rows = sum(step.lens)                                              # every row of a live sequence passes both sites
    mask = 1 << site
    assert trd.expected_hits(step, warm, mask, (-1, 0, 0)) == passing * rows * step.H
    assert trd.expected_hits(step, warm, mask, (0, 0, 0)) == rows * step.H
    assert trd.expected_hits(step, warm, mask, (7, 0, 0)) == (rows * step.H if passing == 8 else 0)      # warp 7 never passes site 14: the AB producers are the warps 0 .. 3
    third = sum(step.lens[n] for n in range(step.N) for h in range(step.H) if (n * step.H + h) % 3 == 1)
    assert trd.expected_hits(step, warm, mask, (min(7, passing - 1), 3, 1)) == third


def test_a_configuration_of_the_delay_tests_is_never_vacuous_at_the_new_sites():
    trd, step, warm = delay_model()
    for name, site in trd.SITES.items():
        for tname, target in trd.SITE_TARGETS.get(site, trd.TARGETS).items():
            assert trd.expected_hits(step, warm, 1 << site, target) > 0, (name, tname)
    for tname, target in trd.TARGETS.items():
        assert trd.expected_hits(step, warm, trd.ALL_SITES, target) > 0, tname


def test_the_unadapted_target_would_be_vacuous_at_site_14():
    trd, step, warm = delay_model()
    assert trd.expected_hits(step, warm, 1 << 14, trd.TARGETS["warp7_of_every_third_cta"]) == 0       # why SITE_TARGETS exists
    assert trd.serial_sleeps(step, warm, trd.ALL_SITES) >= trd.serial_sleeps(step, warm, trd.ALL_SITES & ~(3 << 13))
