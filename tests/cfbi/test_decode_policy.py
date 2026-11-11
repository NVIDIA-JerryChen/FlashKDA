"""The single-launch decode policy and the constants of the CUDA-free header (``csrc/cfbi/cfbi_policy.h``).

* the specification (``decode_policy.py``) uses the numbers of the header, and the header is plain C++ (no CUDA header, no CUDA qualifier);
* the C++ function, compiled with the host compiler (no CUDA, no GPU), equals the specification on the shape matrix: for SM counts 114 / 132 / 148, compute capabilities (9, 0) / (10, 0) /
  (10, 3) / (12, 0) and H in {6, 12, 33} (and 1, 8: a limit that the grid reaches exactly), the largest N whose grid holds at most the limit and the next larger N, plus mode 0,
  ``IMPL_REF_ROWS``, bit 64, the other bits and grids that do not exist;
* controls: a gate that is one CTA looser or tighter, another number of CTAs per SM, another routing set, a missing exclusion and mode 0 allowed are each a difference on the matrix;
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
    """The integer literals of the ``constexpr`` constants of the header (a constant that is an expression of others is not listed)."""
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"constexpr\s+(?:int|int64_t)\s+(\w+)\s*=\s*(\d+)\s*;", text)}


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


def run_probe(csrc, rows, tmp):
    """Compile the probe against the policy header in ``csrc`` with the host compiler alone (no CUDA include path) and run it on ``rows``: ``[0 | 1]``."""
    cxx = host_compiler()
    if cxx is None:
        pytest.skip("no host compiler on this machine")
    src, exe = Path(tmp) / "probe.cpp", Path(tmp) / "probe"
    src.write_text(PROBE)
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
