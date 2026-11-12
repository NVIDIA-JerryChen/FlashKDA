"""The single-launch decode policy and the constants of the CUDA-free header (``csrc/cfbi/cfbi_policy.h``).

* the specification (``decode_policy.py``) uses the numbers of the header, and the header is plain C++ (no CUDA header, no CUDA qualifier);
* the C++ function, compiled with the host compiler (no CUDA, no GPU), equals the specification on the shape matrix: for SM counts 114 / 132 / 148, compute capabilities (9, 0) / (10, 0) /
  (10, 3) / (12, 0) and H in {6, 12, 33} (and 1, 8: a limit that the grid reaches exactly), the largest N whose grid holds at most the limit and the next larger N, plus mode 0,
  ``IMPL_REF_ROWS``, bit 64, the other bits and grids that do not exist;
* controls: a gate that is one CTA looser or tighter, another number of CTAs per SM, another routing set, a missing exclusion and mode 0 allowed are each a difference on the matrix;
* the route specification (``expected_route``: the launches of a decode call as the preset names of the decode benchmark) equals the C++ function on the decode rows of the matrix with the
  hidden calibration control off, and with it on equals the part of the rule that the control leaves (the C++ eligibility, which the same probe evaluates); the limit that it takes is the
  one that ``header_per_sm`` reads from the header of the tree that is measured, and a tree with another limit gives another route that the C++ function of that tree agrees with;
* the shared-memory need of the header covers the static figure that ptxas reports for the row engine on every architecture of the gate (compile control: a need below the layout, or more
  than an allocation unit above it, fails the build of the row engine).
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import decode_policy as dp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))

import codegen_gate as cg  # noqa: E402

CSRC = ROOT / "csrc" / "cfbi"
HEADER = CSRC / "cfbi_policy.h"
RESERVE = 1024        # shared memory that the driver reserves per CTA on sm_90 and later (cudaDevAttrReservedSharedMemoryPerBlock; the occupancy records of B200 and B300 say 1,024)
PROBE = """#include "cfbi/cfbi_policy.h"
#include <cstdio>
int main() {
    long long mode, impl, N, H, sms, cc;
    while (std::scanf("%lld %lld %lld %lld %lld %lld", &mode, &impl, &N, &H, &sms, &cc) == 6)
        std::printf("%d\\n", cfbi::decode_single_launch_policy((int)mode, (int)impl, N, H, (int)sms, (int)cc) ? 1 : 0);
    return 0;
}
"""


def header_text():
    return HEADER.read_text()


def header_values(text):
    """The integer literals of the ``constexpr`` constants of the header (a constant that is an expression of others is not listed): the reader of the specification module."""
    return dp.header_values(text)


# ---- the header and the specification ---------------------------------------------------------------------------------------------------------------------
def test_the_specification_uses_the_numbers_of_the_header():
    v = header_values(header_text())
    assert v["DECODE_SINGLE_MAX_CTAS_PER_SM"] == dp.MAX_CTAS_PER_SM
    assert v["DECODE_SINGLE_ROUTING_CC_MAJOR"] == dp.ROUTING_CC_MAJOR
    assert v["IMPL_REF_ROWS"] == dp.IMPL_REF_ROWS and v["IMPL_DECODE_PLAN"] == dp.IMPL_DECODE_PLAN and v["IMPL_MAX"] == dp.IMPL_MAX


PLAIN_CPP_BREAKERS = re.compile(r"#\s*include\s*[<\"](?:cuda|cub|cooperative_groups|nv|thrust)|__global__|__device__|__host__|__shared__|__CUDACC__|\bdim3\b|\bcuda[A-Z]\w*")


def cuda_constructs(text):
    """The CUDA headers, qualifiers and runtime names in ``text`` outside its comments."""
    return PLAIN_CPP_BREAKERS.findall(re.sub(r"//[^\n]*", "", text))


def test_the_header_is_plain_cpp():
    assert cuda_constructs(header_text()) == []


@pytest.mark.parametrize("planted", ["#include <cuda_runtime_api.h>\n", "__host__ __device__ inline int f() { return 0; }\n", "inline cudaError_t e() { return cudaSuccess; }\n"])
def test_a_cuda_construct_in_the_header_is_caught(planted):
    assert cuda_constructs(header_text() + planted)


def test_the_specification_on_hand_checked_points():
    b200 = dp.cc_number((10, 0))
    assert dp.policy(1, 0, 1, 12, 148, b200)
    assert dp.policy(1, 0, 98, 12, 148, b200) and not dp.policy(1, 0, 99, 12, 148, b200)          # 1176 CTAs fit the limit of 1184, 1188 do not
    assert dp.policy(1, 0, 148, 8, 148, b200) and not dp.policy(1, 0, 149, 8, 148, b200)           # the grid reaches the limit exactly
    assert dp.policy(1, 0, 1184, 1, 148, b200) and not dp.policy(1, 0, 1185, 1, 148, b200)
    assert dp.policy(1, 0, 4, 12, 148, dp.cc_number((10, 3))) and not dp.policy(1, 0, 4, 12, 132, dp.cc_number((9, 0))) and not dp.policy(1, 0, 4, 12, 148, dp.cc_number((12, 0)))
    assert not dp.policy(0, 0, 4, 12, 148, b200) and not dp.policy(1, 1, 4, 12, 148, b200) and not dp.policy(1, 64, 4, 12, 148, b200)
    assert dp.policy(1, 4, 4, 12, 148, b200) and dp.policy(1, 2 | 8, 4, 12, 148, b200) and dp.policy(1, 32, 4, 12, 148, b200)      # the other bits do not matter
    assert not dp.policy(1, 0, 0, 12, 148, b200) and not dp.policy(1, 0, 4, 0, 148, b200) and not dp.policy(1, 0, 4, 12, 0, b200) and not dp.policy(1, 0, 4, 12, -1, b200)


def test_the_matrix_straddles_every_condition():
    rows = dp.shape_matrix()
    for sms in dp.SM_COUNTS:
        for cap in dp.CAPABILITIES:
            cc = dp.cc_number(cap)
            for H in dp.HEADS + dp.EXTRA_HEADS:
                largest = dp.MAX_CTAS_PER_SM * sms // H
                assert largest * H <= dp.MAX_CTAS_PER_SM * sms < (largest + 1) * H
                assert (1, 0, largest, H, sms, cc) in rows and (1, 0, largest + 1, H, sms, cc) in rows
                want = dp.routed(cc)
                assert dp.policy(1, 0, largest, H, sms, cc) is want and not dp.policy(1, 0, largest + 1, H, sms, cc)
    results = [dp.policy(*r) for r in rows]
    assert 0 < sum(results) < len(results)
    assert {m for m, i, N, H, s, c in rows} == {0, 1} and {i for m, i, N, H, s, c in rows} >= {0, 1, 64, 65}
    assert {c for m, i, N, H, s, c in rows} == {dp.cc_number(cap) for cap in dp.CAPABILITIES}


# ---- the C++ function, compiled with the host compiler -------------------------------------------------------------------------------------------------
def host_compiler():
    """The system C++ compiler (the probe runs on this machine, so a conda compiler, whose binaries need another C library, is no fallback)."""
    for name in ("g++", "c++"):
        found = shutil.which(name)
        if found:
            return found
    return None


POLICY_CALL = "cfbi::decode_single_launch_policy((int)mode, (int)impl, N, H, (int)sms, (int)cc)"
ELIGIBLE_CALL = "cfbi::decode_single_launch_eligible((int)mode, (int)impl, N, H)"      # what the launcher answers with the hidden calibration control on (cfbi_launch.cu)


def run_probe(csrc, rows, tmp, eligible=False):
    """Compile the probe against the policy header in ``csrc`` with the host compiler alone (no CUDA include path) and run it on ``rows``: ``[0 | 1]`` for the policy function, or for the
    eligible part of the rule (the part that the calibration control leaves) when ``eligible`` is true."""
    cxx = host_compiler()
    if cxx is None:
        pytest.skip("no host compiler on this machine")
    src, exe = Path(tmp) / "probe.cpp", Path(tmp) / "probe"
    assert PROBE.count(POLICY_CALL) == 1
    src.write_text(PROBE.replace(POLICY_CALL, ELIGIBLE_CALL) if eligible else PROBE)
    built = subprocess.run([cxx, "-std=c++17", "-Wall", "-I", str(csrc), str(src), "-o", str(exe)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    assert built.returncode == 0, built.stdout[-2000:]
    run = subprocess.run([str(exe)], input="\n".join(" ".join(str(x) for x in r) for r in rows) + "\n", stdout=subprocess.PIPE, universal_newlines=True, check=True)
    out = [int(x) for x in run.stdout.split()]
    assert len(out) == len(rows)
    return out


@pytest.fixture(scope="module")
def cpp_on_the_matrix(tmp_path_factory):
    rows = dp.shape_matrix()
    return rows, run_probe(CSRC.parent, rows, tmp_path_factory.mktemp("policy_probe"))


def test_the_cpp_function_equals_the_specification_on_the_matrix(cpp_on_the_matrix):
    rows, got = cpp_on_the_matrix
    wrong = [(r, g) for r, g in zip(rows, got) if bool(g) != dp.policy(*r)]
    assert wrong == [], wrong[:5]
    assert 0 < sum(got) < len(got)                                                                       # both answers occur: the equality is not vacuous


CONTROLS = [
    ("a gate one CTA looser", dict(slack=1)),
    ("a gate one CTA tighter", dict(slack=-1)),
    ("nine CTAs per SM", dict(per_sm=9)),
    ("seven CTAs per SM", dict(per_sm=7)),
    ("Hopper in the routing set", dict(major=9)),
    ("sm_120 in the routing set", dict(major=12)),
    ("bit 64 not excluded", dict(blocked=dp.IMPL_REF_ROWS)),
    ("the reference row engine not excluded", dict(blocked=dp.IMPL_DECODE_PLAN)),
    ("mode 0 allowed", dict(modes=(0, 1))),
]


@pytest.mark.parametrize("name,knobs", CONTROLS, ids=[c[0] for c in CONTROLS])
def test_a_specification_with_another_rule_differs_from_the_cpp_function_on_the_matrix(cpp_on_the_matrix, name, knobs):
    rows, got = cpp_on_the_matrix
    assert [r for r, g in zip(rows, got) if bool(g) != dp.policy(*r, **knobs)], name


def test_the_probe_judges_the_header_it_is_given(tmp_path):
    """The probe is not a fixed answer: a header whose limit is 9 CTAs per SM gives other results than the real one."""
    csrc = tmp_path / "csrc"
    shutil.copytree(CSRC, csrc / "cfbi")
    target = csrc / "cfbi" / "cfbi_policy.h"
    text = target.read_text()
    assert text.count("DECODE_SINGLE_MAX_CTAS_PER_SM = 8;") == 1
    target.write_text(text.replace("DECODE_SINGLE_MAX_CTAS_PER_SM = 8;", "DECODE_SINGLE_MAX_CTAS_PER_SM = 9;"))
    rows = dp.shape_matrix()
    assert run_probe(csrc, rows, tmp_path) != [int(dp.policy(*r)) for r in rows]


# ---- the route specification and the limit of a tree -------------------------------------------------------------------------------------------------
def route_of(row, **knobs):
    """The route of a row of the matrix (mode, impl, N, H, sms, cc) as ``expected_route`` gives it."""
    mode, impl, N, H, sms, cc = row
    return dp.expected_route(impl, N, H, sms, cc, **knobs)


def route_name(answer):
    return dp.SINGLE_LAUNCH if answer else dp.TWO_LAUNCH


def decode_rows(rows, got):
    """The (row, answer) pairs of the decode calls: the route specification is for mode 1, a mode 0 call is no decode call."""
    return [(r, g) for r, g in zip(rows, got) if r[0] == 1]


@pytest.fixture(scope="module")
def cpp_eligible_on_the_matrix(tmp_path_factory):
    rows = dp.shape_matrix()
    return rows, run_probe(CSRC.parent, rows, tmp_path_factory.mktemp("eligible_probe"), eligible=True)


def test_the_route_specification_equals_the_cpp_function_on_the_matrix(cpp_on_the_matrix):
    """With the control off a decode call is one launch exactly where the C++ function says so."""
    rows, got = cpp_on_the_matrix
    decode = decode_rows(rows, got)
    assert len(rows) - len(decode) == 2 * len(dp.SM_COUNTS) * len(dp.CAPABILITIES)                  # only the two mode 0 rows of every (SM count, capability) are left out
    wrong = [(r, g, route_of(r)) for r, g in decode if route_of(r) != route_name(g)]
    assert wrong == [], wrong[:5]
    assert {route_of(r) for r, g in decode} == {dp.SINGLE_LAUNCH, dp.TWO_LAUNCH}                      # both answers occur: the equality is not vacuous


def test_the_route_specification_with_the_control_on_is_the_eligible_part_of_the_rule(cpp_eligible_on_the_matrix):
    """With the control on the call is one launch whatever the grid and the device: the eligibility, which the specification and the header's function both state."""
    rows, got = cpp_eligible_on_the_matrix
    decode = decode_rows(rows, got)
    assert [(r, g, route_of(r, force=True)) for r, g in decode if route_of(r, force=True) != route_name(g)] == []
    assert [(r, g) for r, g in decode if bool(g) != dp.eligible(*r[:4])] == []                        # the Python eligibility is the header's
    assert {route_of(r, force=True) for r, g in decode} == {dp.SINGLE_LAUNCH, dp.TWO_LAUNCH}


def test_the_control_overrides_the_limit_and_the_routing_set_and_nothing_else(cpp_on_the_matrix):
    rows, got = cpp_on_the_matrix
    decode = decode_rows(rows, got)
    forced_only = [r for r, g in decode if route_of(r, force=True) != route_of(r)]
    assert any(r[2] * r[3] > dp.MAX_CTAS_PER_SM * r[4] and dp.routed(r[5]) for r in forced_only)      # a grid beyond the limit on a device of the routing set
    assert any(not dp.routed(r[5]) and r[2] * r[3] <= dp.MAX_CTAS_PER_SM * r[4] for r in forced_only)   # a grid within the limit on a device outside it
    assert all(route_of(r) == dp.TWO_LAUNCH and route_of(r, force=True) == dp.SINGLE_LAUNCH for r in forced_only)
    kept = [r for r, g in decode if r[1] & (dp.IMPL_REF_ROWS | dp.IMPL_DECODE_PLAN) or r[2] < 1 or r[3] < 1]
    assert kept and all(route_of(r, force=True) == dp.TWO_LAUNCH for r in kept)                       # bit 64, the reference row engine and a grid that does not exist: never one launch


@pytest.mark.parametrize("name,knobs", [("nine CTAs per SM", dict(per_sm=9)), ("seven CTAs per SM", dict(per_sm=7)), ("the control on", dict(force=True))])
def test_a_route_specification_with_another_rule_differs_from_the_cpp_function_on_the_matrix(cpp_on_the_matrix, name, knobs):
    rows, got = cpp_on_the_matrix
    assert [r for r, g in decode_rows(rows, got) if route_of(r, **knobs) != route_name(g)], name


def policy_tree(tmp_path, replace=None, name="tree"):
    """A copy of ``csrc/cfbi`` under a root of its own; ``replace`` is ``(old, new)``: a text of the policy header to change."""
    root = tmp_path / name
    shutil.copytree(CSRC, root / "csrc" / "cfbi")
    if replace:
        target = root / "csrc" / "cfbi" / "cfbi_policy.h"
        text = target.read_text()
        assert text.count(replace[0]) == 1, replace[0]
        target.write_text(text.replace(*replace))
    return root


LIMIT_LINE = "constexpr int64_t DECODE_SINGLE_MAX_CTAS_PER_SM = 8;"


def test_the_limit_of_a_tree_is_what_its_header_defines():
    assert dp.header_per_sm(ROOT) == dp.MAX_CTAS_PER_SM == header_values(header_text())["DECODE_SINGLE_MAX_CTAS_PER_SM"]


def test_the_limit_of_another_tree_is_its_own_and_so_is_its_route(tmp_path):
    """A tree with 9 CTAs per SM: the reader gives 9, the specification that takes it agrees with the C++ function of that tree on the decode rows, and the limit of this tree does not."""
    root = policy_tree(tmp_path, (LIMIT_LINE, LIMIT_LINE.replace("= 8;", "= 9;")))
    per_sm = dp.header_per_sm(root)
    assert per_sm == 9
    rows = dp.shape_matrix()
    got = run_probe(root / "csrc", rows, tmp_path)
    decode = decode_rows(rows, got)
    assert [(r, g) for r, g in decode if route_of(r, per_sm=per_sm) != route_name(g)] == []
    assert [r for r, g in decode if route_of(r) != route_name(g)]                                    # the limit of 8 is another route, on some row


UNREADABLE_LIMITS = [
    ("no policy header", None, OSError),
    ("a limit that is an expression", (LIMIT_LINE, "constexpr int64_t DECODE_SINGLE_MAX_CTAS_PER_SM = 4 * 2;"), ValueError),
    ("a limit that is only in a comment", (LIMIT_LINE, "// " + LIMIT_LINE), ValueError),
    ("a limit that is only in a block comment", (LIMIT_LINE, "/* " + LIMIT_LINE + " */"), ValueError),
]


@pytest.mark.parametrize("what,replace,error", UNREADABLE_LIMITS, ids=[c[0] for c in UNREADABLE_LIMITS])
def test_a_tree_without_a_readable_limit_is_an_error_not_a_default(tmp_path, what, replace, error):
    root = tmp_path / "empty" if replace is None else policy_tree(tmp_path, replace)
    root.mkdir(exist_ok=True)
    with pytest.raises(error):
        dp.header_per_sm(root)


def test_a_commented_out_definition_does_not_change_the_limit(tmp_path):
    root = policy_tree(tmp_path, (LIMIT_LINE, "/* constexpr int64_t DECODE_SINGLE_MAX_CTAS_PER_SM = 5; */\n// constexpr int64_t DECODE_SINGLE_MAX_CTAS_PER_SM = 6;\n" + LIMIT_LINE))
    assert dp.header_per_sm(root) == 8


# ---- the shared-memory need of the header ----------------------------------------------------------------------------------------------------------------------
def need_problems(static_need, dynamic_need, figures):
    """``figures``: ``{arch: {kernel: SHARED of the machine-code dump}}`` (static shared memory plus the driver's reserve). The header's static plus dynamic need must be at least the
    compiler's static figure plus the dynamic constant on every architecture and for every instantiation; equality is not required."""
    out = []
    need = static_need + dynamic_need
    for arch, kernels in sorted(figures.items()):
        if not kernels:
            out.append("%s: no row-engine figure" % arch)
        for key, shared in sorted(kernels.items()):
            required = shared - RESERVE + dynamic_need
            if need < required:
                out.append("%s %s: the header's need of %d B is below the compiler's static figure of %d B" % (arch, key, need, required))
    return out


FIGURES = {"sm_90a": {"cfbi::rows_kernel<Lb0E>": 8832, "cfbi::rows_kernel<Lb1E>": 8832}, "sm_100a": {"cfbi::rows_kernel<Lb0E>": 9216, "cfbi::rows_kernel<Lb1E>": 9216},
           "sm_103a": {"cfbi::rows_kernel<Lb0E>": 9216, "cfbi::rows_kernel<Lb1E>": 9216}}      # the figures of the dumps (8,192 B on Blackwell, 7,808 B on sm_90a, plus the reserve)


@pytest.mark.parametrize("static,dynamic,expect", [(8192, 0, 0), (8192 + 512, 0, 0), (7808, 0, 4), (7680, 0, 6), (7684, 0, 6), (8191, 0, 4), (8192, 4096, 0)])
def test_the_need_is_judged_against_the_compilers_figure(static, dynamic, expect):
    assert len(need_problems(static, dynamic, FIGURES)) == expect, need_problems(static, dynamic, FIGURES)


def test_a_missing_figure_is_a_problem():
    assert need_problems(8192, 0, {"sm_90a": {}})


def test_the_header_states_the_need_that_the_figures_ask_for():
    v = header_values(header_text())
    assert need_problems(v["ROWS_STATIC_SMEM_BYTES"], v["ROWS_DYNAMIC_SMEM_BYTES"], FIGURES) == []


def _toolchain_or_skip():
    try:
        return cg.find_toolchain()
    except cg.Unusable as e:
        pytest.skip("no toolchain for the compile-only checks on this machine: %s" % e)


def test_the_static_need_of_the_header_covers_the_ptxas_figure_of_every_architecture(tmp_path):
    """The real figures: the launcher unit (both instantiations of the row engine) built for every architecture of the code-generation gate."""
    tc = _toolchain_or_skip()
    table, _, _ = cg.build_unit(str(ROOT), tc, "csrc/cfbi/cfbi_launch.cu", cg.GATE_ARCHS, (), str(tmp_path))
    figures = {arch: {k: rec["shared"] for k, rec in table[arch].items() if k in ("cfbi::rows_kernel<Lb0E>", "cfbi::rows_kernel<Lb1E>")} for arch in cg.GATE_ARCHS}
    assert all(len(v) == 2 for v in figures.values()), figures
    v = header_values(header_text())
    assert need_problems(v["ROWS_STATIC_SMEM_BYTES"], v["ROWS_DYNAMIC_SMEM_BYTES"], figures) == [], figures
    assert {arch: sorted(set(f.values())) for arch, f in figures.items()} == {"sm_90a": [7808 + RESERVE], "sm_100a": [8192 + RESERVE], "sm_103a": [8192 + RESERVE]}


def _compile_rows_header(tc, tree):
    """Compile a unit that only includes the row engine (the ``static_assert``s of its layout are evaluated, no kernel is instantiated): ``(exit status, text)``."""
    (tree / "check_rows.cu").write_text('#include "cfbi/cfbi_rows.cuh"\n')
    cmd = cg.compile_command(tc, "check_rows.cu", ("sm_90a",), (), str(tree / "check_rows.o"))
    r = subprocess.run(cmd, cwd=str(tree), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    return r.returncode, r.stdout


def _tree_with_need(tmp, need):
    tree = Path(tempfile.mkdtemp(dir=str(tmp)))
    shutil.copytree(CSRC, tree / "csrc" / "cfbi")
    if need is not None:
        target = tree / "csrc" / "cfbi" / "cfbi_policy.h"
        text = target.read_text()
        assert text.count("ROWS_STATIC_SMEM_BYTES = 8192;") == 1
        target.write_text(text.replace("ROWS_STATIC_SMEM_BYTES = 8192;", "ROWS_STATIC_SMEM_BYTES = %d;" % need))
    return tree


def test_a_need_that_does_not_match_the_layout_fails_the_build(tmp_path):
    tc = _toolchain_or_skip()
    assert _compile_rows_header(tc, _tree_with_need(tmp_path, None))[0] == 0                                  # the real header builds: the control is not vacuous
    rc, text = _compile_rows_header(tc, _tree_with_need(tmp_path, 7680))                                       # the size of the structure without its padding and the fill word
    assert rc != 0 and "must cover the static shared memory of the row engine" in text, text[-1500:]
    rc, text = _compile_rows_header(tc, _tree_with_need(tmp_path, 9000))                                       # more than an allocation unit above the layout
    assert rc != 0 and "more than an allocation unit above the layout" in text, text[-1500:]
