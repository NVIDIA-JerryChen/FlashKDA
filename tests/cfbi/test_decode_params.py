"""The argument block of the kernels (``Params``) keeps its layout; what only the single-launch decode kernel needs is a type of its own (plain Python, no GPU:
``pytest --noconftest tests/cfbi/test_decode_params.py``).

``Params`` is the first argument of every kernel. A kernel with a further argument behind it (``k2_dyn_kernel``: the schedule) finds that argument at the offset ``sizeof(Params)`` of the
constant bank, so a field appended to ``Params`` is another machine code for kernels that the change does not own (the code-generation gate shows it as a difference of
``k2_dyn_kernel``). The token windows of the call (``cu_seqlens``, ``fixed_len``), which the plan-less decode kernel reads, therefore sit in ``DecodeParams``, the argument of
``rows_kernel<true>`` only:

* the field list of ``Params`` is pinned (and its size, by a ``static_assert`` that this module ties to the field list);
* ``DecodeParams`` is ``Params`` plus exactly the two windows, and only ``rows_entry<true>`` reads them;
* the launcher builds it in the single-launch branch only;
* control: a field added to ``Params`` (and a second pointer added to ``DecodeParams``) fails the build with the ``static_assert`` text.
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))

import codegen_gate as cg  # noqa: E402

CSRC = ROOT / "csrc" / "cfbi"
COMMON, ROWS, LAUNCH = (CSRC / n for n in ("cfbi_common.cuh", "cfbi_rows.cuh", "cfbi_launch.cu"))

FROZEN_PARAMS = ("q", "k", "v", "g", "beta", "out", "out_f32", "A_log", "dt_bias", "scale", "scale2", "H", "N", "E", "rows", "pos", "slot_ids", "fuse_pos", "plan", "ws", "tile_x32",
                 "tile_bvb")
DECODE_FIELDS = ("cu_seqlens", "fixed_len")


def strip_comments(text):
    return re.sub(r"//[^\n]*", "", text)


def struct_body(text, header):
    start = text.index(header)
    body_start = text.index("{", start) + 1
    return strip_comments(text[body_start:text.index("\n};", body_start)])


def struct_fields(text, header):
    """``[(type, name)]`` of the data members of ``struct`` with this ``header`` line, in declaration order."""
    out = []
    for stmt in struct_body(text, header).split(";"):
        stmt = " ".join(stmt.split())
        if stmt:
            m = re.match(r"^(.*?[\s*])(\w+)$", stmt)
            assert m, "cannot read the member %r" % stmt
            out.append((m.group(1).strip(), m.group(2)))
    return out


def layout_size(fields):
    """``sizeof`` of a struct of pointers, ``int`` and ``float`` (natural alignment, the only types ``Params`` uses)."""
    offset, align = 0, 1
    for typ, name in fields:
        size = 8 if typ.endswith("*") else 4
        assert typ.endswith("*") or typ in ("int", "float"), (typ, name)
        offset = -(-offset // size) * size + size
        align = max(align, size)
    return -(-offset // align) * align


@pytest.fixture(scope="module")
def common():
    return COMMON.read_text()


@pytest.fixture(scope="module")
def rows():
    return ROWS.read_text()


@pytest.fixture(scope="module")
def launch():
    return LAUNCH.read_text()


def mutate(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


# ---- Params ------------------------------------------------------------------------------------------------------------------------------------------------
def params_problems(text):
    probs = []
    fields = struct_fields(text, "struct Params {")
    names = tuple(n for t, n in fields)
    if names != FROZEN_PARAMS:
        probs.append("the fields of Params are %s, expected %s: a change of the argument block moves the arguments behind it in the kernels that take one" % (names, FROZEN_PARAMS))
    pin = re.search(r"static_assert\(sizeof\(Params\) == (\d+),", text)
    if pin is None:
        probs.append("no static_assert on sizeof(Params)")
    elif int(pin.group(1)) != layout_size(fields):
        probs.append("static_assert(sizeof(Params) == %s) disagrees with the fields (%d)" % (pin.group(1), layout_size(fields)))
    return probs


def test_params_keeps_its_layout(common):
    assert params_problems(common) == []
    assert layout_size(struct_fields(common, "struct Params {")) == 152


@pytest.mark.parametrize("name,old,new,expect", [
    ("a field appended", "    int tile_bvb;\n};", "    int tile_bvb;\n    const int64_t* cu_seqlens;\n    int fixed_len;\n};", "fields of Params"),
    ("a field appended without the pin", "    int tile_bvb;\n};", "    int tile_bvb;\n    int extra;\n};", "fields of Params"),
    ("a field dropped", "    int tile_x32;\n", "", "fields of Params"),
    ("the pin dropped", 'static_assert(sizeof(Params) == 152, "Params is the argument block of every kernel: a change of its size changes the machine code of the kernels that take a further argument behind it");\n', "", "no static_assert"),
    ("the pin out of step", "static_assert(sizeof(Params) == 152,", "static_assert(sizeof(Params) == 160,", "disagrees"),
])
def test_a_change_of_the_argument_block_is_caught(common, name, old, new, expect):
    probs = params_problems(mutate(common, old, new))
    assert any(expect in p for p in probs), (name, probs)


# ---- DecodeParams ------------------------------------------------------------------------------------------------------------------------------------------
def decode_problems(common, rows, launch):
    probs = []
    if tuple(n for t, n in struct_fields(common, "struct DecodeParams : Params {")) != DECODE_FIELDS:
        probs.append("DecodeParams is not Params plus %s" % (DECODE_FIELDS,))
    if "static_assert(sizeof(DecodeParams) == sizeof(Params) + 16," not in common:
        probs.append("no static_assert that DecodeParams follows Params without a gap")
    if "using RowsParams = std::conditional_t<DECODE, DecodeParams, Params>;" not in rows:
        probs.append("RowsParams is not the decode arguments for rows_kernel<true> and Params for rows_kernel<false>")
    if "rows_kernel(const RowsParams<DECODE> p)" not in rows:
        probs.append("rows_kernel does not take RowsParams<DECODE>")
    entry = rows[rows.index("SeqPlan rows_entry("):rows.index("#ifndef CFBI_ROWS_MINB")]
    if "if constexpr (!DECODE) {\n        return p.plan[n];\n    } else {" not in entry:
        probs.append("rows_entry<false> is not the plan load behind an if constexpr (rows_entry<true> would not compile for Params)")
    sources = [f for f in CSRC.rglob("*") if f.suffix in (".cu", ".cuh", ".h", ".cpp")]
    where = {f.name for f in sources if re.search(r"\b(?:p|dp)\.(?:cu_seqlens|fixed_len)\b", strip_comments(f.read_text()))}
    if where - {"cfbi_rows.cuh", "cfbi_launch.cu"}:
        probs.append("the token windows of the decode arguments are read outside the row engine and the launcher: %s" % sorted(where))
    lines = strip_comments(launch).splitlines()
    if [l.strip() for l in lines if "DecodeParams" in l] != ["const DecodeParams dp{p, a.cu_seqlens, a.fixed_len};"]:
        probs.append("the launcher does not build the decode arguments exactly once from the call's Params and windows")
    if any(re.search(r"\bp\.(?:cu_seqlens|fixed_len)\b", l) for l in lines):
        probs.append("the launcher stores the windows in Params")
    return probs


def test_the_decode_arguments_are_a_type_of_their_own(common, rows, launch):
    assert decode_problems(common, rows, launch) == []


@pytest.mark.parametrize("where,name,old,new,expect", [
    ("common", "a third field", "    int fixed_len;\n};", "    int fixed_len;\n    int third;\n};", "not Params plus"),
    ("common", "the size pin dropped", "static_assert(sizeof(DecodeParams) == sizeof(Params) + 16,", "static_assert(sizeof(DecodeParams) >= sizeof(Params),", "without a gap"),
    ("rows", "Params for both", "using RowsParams = std::conditional_t<DECODE, DecodeParams, Params>;", "using RowsParams = Params;", "RowsParams is not"),
    ("rows", "a kernel on plain Params", "rows_kernel(const RowsParams<DECODE> p)", "rows_kernel(const Params p)", "does not take RowsParams"),
    ("rows", "no if constexpr", "    if constexpr (!DECODE) {\n        return p.plan[n];\n    } else {", "    if (!DECODE) {\n        return p.plan[n];\n    } else {", "if constexpr"),
    ("launch", "the windows in Params", "            const DecodeParams dp{p, a.cu_seqlens, a.fixed_len};", "            p.cu_seqlens = a.cu_seqlens;\n            const DecodeParams dp{p, a.cu_seqlens, a.fixed_len};", "stores the windows in Params"),
    ("launch", "built twice", "        } else rows_kernel<false>", "            const DecodeParams dq{p, a.cu_seqlens, a.fixed_len};\n        } else rows_kernel<false>", "exactly once"),
])
def test_a_decode_argument_outside_its_type_is_caught(common, rows, launch, where, name, old, new, expect):
    texts = {"common": common, "rows": rows, "launch": launch}
    texts[where] = mutate(texts[where], old, new)
    probs = decode_problems(texts["common"], texts["rows"], texts["launch"])
    assert any(expect in p for p in probs), (name, probs)


# ---- the build: a change of either block fails with the static_assert ---------------------------------------------------------------------------------------
def _toolchain_or_skip():
    try:
        return cg.find_toolchain()
    except cg.Unusable as e:
        pytest.skip("no toolchain for the compile-only control on this machine: %s" % e)


def _compile_header(tc, tree):
    """Compile a unit that only includes the common header (the ``static_assert``s of the layouts are evaluated, no kernel is instantiated): ``(exit status, text)``."""
    unit = tree / "check_layout.cu"
    unit.write_text('#include "cfbi/cfbi_common.cuh"\n')
    cmd = cg.compile_command(tc, "check_layout.cu", ("sm_90a",), (), str(tree / "check_layout.o"))
    r = subprocess.run(cmd, cwd=str(tree), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    return r.returncode, r.stdout


def _tree_with(tmp, old=None, new=None):
    tree = Path(tempfile.mkdtemp(dir=str(tmp)))
    shutil.copytree(CSRC, tree / "csrc" / "cfbi")
    if old is not None:
        target = tree / "csrc" / "cfbi" / "cfbi_common.cuh"
        target.write_text(mutate(target.read_text(), old, new))
    return tree


def test_a_field_added_to_the_argument_blocks_fails_the_build(tmp_path):
    tc = _toolchain_or_skip()
    assert _compile_header(tc, _tree_with(tmp_path))[0] == 0                                                    # the unchanged headers build: the control is not vacuous
    rc, text = _compile_header(tc, _tree_with(tmp_path, "    int tile_bvb;\n};", "    int tile_bvb;\n    const int64_t* cu_seqlens;\n    int fixed_len;\n};"))
    assert rc != 0 and "Params is the argument block of every kernel" in text, text[-1500:]
    rc, text = _compile_header(tc, _tree_with(tmp_path, "    int fixed_len;\n};", "    int fixed_len;\n    const int* more;\n};"))
    assert rc != 0 and "the decode arguments" in text, text[-1500:]
