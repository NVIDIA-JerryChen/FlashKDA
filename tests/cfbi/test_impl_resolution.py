"""The resolution of the implementation bits for a device: ``resolve_impl_for`` in the CUDA-free header ``csrc/cfbi/cfbi_policy.h`` and its wiring in the launcher (``resolve_impl``).

Plain Python and the host C++ compiler, no CUDA, no GPU (``pytest --noconftest tests/cfbi/test_impl_resolution.py``). The function takes everything it decides from as arguments (the compute
capability as major * 10 + minor, the per-block opt-in limit of shared memory, and the needs of the launches), so that it is called here with any device and any need:

* the C++ function, compiled with the host compiler against the real header, equals the specification on a matrix of every requested value 0 .. 127, the compute capabilities
  80 / 86 / 89 / 90 / 100 / 103 / 120 and opt-in limits that straddle every need (one byte below and exactly at it) as well as the limits of the real devices (227 KiB on sm_90 / 100 / 103,
  163 KiB on sm_80, 99 KiB on sm_86 / 89 / 120), for a row engine with static shared memory only and for one with a dynamic tile;
* the hand-checked points, among them the ones that matter for a row engine that needs more than the old 7 KiB: an EXPLICIT rows-only request (bit 4) skips the K1 check, so the row engine is
  checked on its own: kept at its need, the reference row engine one byte below it, and the 99 KiB of sm_120 keep the optimised row engine whose need is 75,776 B;
* controls: a specification with another rule (no check of the row engine for an explicit request, a strict comparison, the plain need instead of the dynamic one, the reserve of the driver
  added, an unforced reference below sm_90, K1 compared the wrong way) each differs from the C++ function on the matrix, and the probe judges the header it is given;
* the launcher passes the needs that the headers define (K1 of both families, the row engine's) in the order of the structure, and resolves with nothing else.
"""
import itertools
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CSRC = ROOT / "csrc" / "cfbi"
HEADER, LAUNCH = CSRC / "cfbi_policy.h", CSRC / "cfbi_launch.cu"
KIB = 1024
K1, K1_REF = 204 * KIB, 218 * KIB            # the shared memory that K1 of the optimised and of the reference chunk kernels needs (cfbi_launch.h: 204 KiB / 218 KiB)
ROWS_STATIC = 8192                           # a row engine with static shared memory only (what the engine needs before its state tile is staged)
ROWS_TILE = ROWS_STATIC + 128 * 132 * 4      # a row engine that also stages a 128 x 132 floats tile in dynamic shared memory: 8,192 + 67,584 = 75,776 B
LIMITS = {"sm_90 / 100 / 103": 227 * KIB, "sm_80": 163 * KIB, "sm_86 / 89 / 120": 99 * KIB}
CCS = (80, 86, 89, 90, 100, 103, 120)
PROBE = """#include "cfbi/cfbi_policy.h"
#include <cstdio>
int main() {
    long long req, cc, smem, k1, k1_ref, rows;
    while (std::scanf("%lld %lld %lld %lld %lld %lld", &req, &cc, &smem, &k1, &k1_ref, &rows) == 6)
        std::printf("%d\\n", cfbi::resolve_impl_for((int)req, (int)cc, smem, cfbi::ImplNeeds{k1, k1_ref, rows}));
    return 0;
}
"""


def header_bits(text):
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"constexpr\s+int\s+(IMPL_\w+)\s*=\s*(\d+)\s*;", text)}


BITS = header_bits(HEADER.read_text())
REF_ROWS, REF_CHUNK, ROWS_ONLY, UMMA, DYN, OLD, PLAN = (BITS[n] for n in ("IMPL_REF_ROWS", "IMPL_REF_CHUNK", "IMPL_ROWS_ONLY", "IMPL_K2_UMMA", "IMPL_K2_DYN", "IMPL_K2_OLD", "IMPL_DECODE_PLAN"))
MASK = REF_ROWS | REF_CHUNK | ROWS_ONLY | UMMA | DYN | PLAN          # the bits that a request can carry through (IMPL_K2_OLD is the routing rule's request, dropped here)


# ---- the specification -------------------------------------------------------------------------------------------------------------------------------------------------
def spec(requested, cc, smem, k1, k1_ref, rows, **knobs):
    """The bits that run for ``requested`` on a device, by the documented rules, one at a time (``knobs``: a rule of a control)."""
    major = cc // 10
    impl = requested & MASK
    if major != 10:
        impl &= ~UMMA                                                       # the tcgen05 K2 exists in the sm_100a / sm_103a code only
    if major < 9 and not knobs.get("unforced_below_sm90"):
        impl |= REF_ROWS | REF_CHUNK                                        # the optimised kernels use sm_90+ instructions
    chunk_need = k1_ref if impl & REF_CHUNK else k1
    if knobs.get("k1_wrong_way"):
        chunk_need = k1 if impl & REF_CHUNK else k1_ref
    if not impl & ROWS_ONLY and smem < chunk_need:
        impl |= ROWS_ONLY | REF_ROWS                                        # the chunk kernels do not fit: every row through the (reference) row engine
    need = rows
    if knobs.get("plain_need"):
        need = ROWS_STATIC                                                  # the dynamic part of the need left out
    if knobs.get("reserve"):
        need += 1024                                                        # the reserve of the driver is no part of the opt-in limit
    below = smem <= need if knobs.get("strict") else smem < need
    explicit_unchecked = knobs.get("explicit_unchecked") and requested & ROWS_ONLY
    if not impl & REF_ROWS and below and not explicit_unchecked:
        impl |= REF_ROWS                                                    # the optimised row engine does not fit: the reference row engine, whatever was requested
    if impl & (REF_CHUNK | ROWS_ONLY) or requested & UMMA:
        impl &= ~DYN                                                        # the dynamic-schedule K2 needs the optimised K1 and loses against the UMMA K2
    return impl


def matrix():
    """The rows ``(requested, cc, smem, k1, k1_ref, rows)``: every request, every compute capability, the limits of the devices and the limits around every need."""
    out = []
    for rows in (ROWS_STATIC, ROWS_TILE):
        smems = sorted(set(list(LIMITS.values()) + [0] + [n + d for n in (K1, K1_REF, rows, ROWS_STATIC) for d in (-1, 0, 1)]))
        out.extend((req, cc, smem, K1, K1_REF, rows) for req, cc, smem in itertools.product(range(128), CCS, smems))
    return out


def host_compiler():
    return shutil.which("g++") or shutil.which("c++")


def run_probe(csrc, rows, tmp):
    """Compile the probe against the header in ``csrc`` with the host compiler alone (no CUDA include path) and run it on ``rows``."""
    cxx = host_compiler()
    if cxx is None:
        pytest.skip("no host compiler on this machine")
    src, exe = Path(tmp) / "resolve_probe.cpp", Path(tmp) / "resolve_probe"
    src.write_text(PROBE)
    built = subprocess.run([cxx, "-std=c++17", "-Wall", "-I", str(csrc), str(src), "-o", str(exe)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    assert built.returncode == 0, built.stdout[-2000:]
    run = subprocess.run([str(exe)], input="\n".join(" ".join(str(x) for x in r) for r in rows) + "\n", stdout=subprocess.PIPE, universal_newlines=True, check=True)
    out = [int(x) for x in run.stdout.split()]
    assert len(out) == len(rows)
    return out


@pytest.fixture(scope="module")
def cpp_on_the_matrix(tmp_path_factory):
    rows = matrix()
    return rows, run_probe(CSRC.parent, rows, tmp_path_factory.mktemp("resolve_probe"))


# ---- the specification on hand-checked points -----------------------------------------------------------------------------------------------------------------
def test_the_specification_on_hand_checked_points():
    def point(requested, cc, smem, rows=ROWS_TILE):
        """The specification with the needs of K1 of both families and of the row engine (``ROWS_TILE`` unless given)."""
        return spec(requested, cc, smem, K1, K1_REF, rows)
    gate = LIMITS["sm_90 / 100 / 103"]
    for cc in (90, 100, 103):                                                                   # the devices that run the optimised kernels: nothing to fall back from
        assert point(0, cc, gate) == 0
        assert point(REF_ROWS, cc, gate) == REF_ROWS and point(PLAN, cc, gate) == PLAN
    assert point(UMMA, 100, gate) == UMMA and point(UMMA, 90, gate) == 0 and point(UMMA | DYN, 100, gate) == UMMA
    assert point(DYN, 100, gate) == DYN and point(DYN | REF_CHUNK, 100, gate) == REF_CHUNK and point(OLD, 100, gate) == 0
    for cc in (80, 86, 89):                                                                     # below sm_90: the reference kernels
        assert point(0, cc, LIMITS["sm_80"]) == REF_ROWS | REF_CHUNK | ROWS_ONLY                                       # 163 KiB are below the 218 KiB of the reference K1
    # sm_120 and the other 99 KiB devices: K1 does not fit, every row goes through the reference row engine, whatever the row engine needs
    assert point(0, 120, LIMITS["sm_86 / 89 / 120"]) == ROWS_ONLY | REF_ROWS
    # an explicit rows-only request skips the K1 check: the row engine is checked on its own need
    assert point(ROWS_ONLY, 120, LIMITS["sm_86 / 89 / 120"]) == ROWS_ONLY                                        # 75,776 B fit 99 KiB: the optimised row engine stays
    assert point(ROWS_ONLY, 100, ROWS_TILE) == ROWS_ONLY                                                            # kept at its need
    assert point(ROWS_ONLY, 100, ROWS_TILE - 1) == ROWS_ONLY | REF_ROWS                                            # the reference row engine one byte below it
    assert point(ROWS_ONLY | DYN, 100, ROWS_TILE - 1) == ROWS_ONLY | REF_ROWS                                     # the K2 bit is dropped with rows-only, as before
    assert point(ROWS_ONLY, 100, ROWS_STATIC, ROWS_STATIC) == ROWS_ONLY and point(ROWS_ONLY, 100, ROWS_STATIC - 1, ROWS_STATIC) == ROWS_ONLY | REF_ROWS
    assert point(ROWS_ONLY | REF_ROWS, 100, 0) == ROWS_ONLY | REF_ROWS                                            # an explicit reference row engine is never replaced
    assert point(0, 100, K1 - 1) == ROWS_ONLY | REF_ROWS and point(0, 100, K1) == 0                                                              # K1 at its need and one byte below it
    assert point(REF_CHUNK, 100, K1_REF - 1) == REF_CHUNK | ROWS_ONLY | REF_ROWS and point(REF_CHUNK, 100, K1_REF) == REF_CHUNK


def test_the_matrix_straddles_every_condition():
    rows = matrix()
    got = [spec(*r) for r in rows]
    assert set(got) >= {0, REF_ROWS, ROWS_ONLY, ROWS_ONLY | REF_ROWS, REF_ROWS | REF_CHUNK, UMMA, DYN, PLAN}
    for cc in (90, 100, 103, 120):                                                               # at a limit between the row engine's need and K1's the explicit request is kept, below it falls back
        at, below = (ROWS_ONLY, cc, ROWS_TILE, K1, K1_REF, ROWS_TILE), (ROWS_ONLY, cc, ROWS_TILE - 1, K1, K1_REF, ROWS_TILE)
        assert at in rows and below in rows and spec(*at) == ROWS_ONLY and spec(*below) == ROWS_ONLY | REF_ROWS
    assert {r[2] for r in rows} >= set(LIMITS.values()) | {ROWS_TILE - 1, ROWS_TILE, ROWS_STATIC - 1, ROWS_STATIC, K1 - 1, K1, K1_REF - 1, K1_REF}


# ---- the C++ function, compiled with the host compiler -----------------------------------------------------------------------------------------------------------
def test_the_cpp_function_equals_the_specification_on_the_matrix(cpp_on_the_matrix):
    rows, got = cpp_on_the_matrix
    wrong = [(r, g, spec(*r)) for r, g in zip(rows, got) if g != spec(*r)]
    assert wrong == [], wrong[:5]
    assert len(set(got)) > 8                                                                     # many answers occur: the equality is not vacuous


CONTROLS = [
    ("no check of the row engine for an explicit rows-only request", dict(explicit_unchecked=True)),
    ("a strict comparison: the reference row engine already at the need", dict(strict=True)),
    ("the plain need instead of the need with the dynamic tile", dict(plain_need=True)),
    ("the driver's reserve added to the need", dict(reserve=True)),
    ("no forced reference kernels below sm_90", dict(unforced_below_sm90=True)),
    ("K1 compared with the need of the other family", dict(k1_wrong_way=True)),
]


@pytest.mark.parametrize("name,knobs", CONTROLS, ids=[c[0] for c in CONTROLS])
def test_a_specification_with_another_rule_differs_from_the_cpp_function_on_the_matrix(cpp_on_the_matrix, name, knobs):
    rows, got = cpp_on_the_matrix
    assert [r for r, g in zip(rows, got) if g != spec(*r, **knobs)], name


def test_the_probe_judges_the_header_it_is_given(tmp_path):
    """The probe is not a fixed answer: a header whose row-engine check is strict gives other results than the real one."""
    csrc = tmp_path / "csrc"
    shutil.copytree(CSRC, csrc / "cfbi")
    target = csrc / "cfbi" / "cfbi_policy.h"
    text = target.read_text()
    assert text.count("smem_optin < needs.rows") == 1
    target.write_text(text.replace("smem_optin < needs.rows", "smem_optin <= needs.rows"))
    rows = matrix()
    assert run_probe(csrc, rows, tmp_path) != [spec(*r) for r in rows]


# ---- the launcher ----------------------------------------------------------------------------------------------------------------------------------------------------------
def launcher_problems(text):
    """What the launcher's ``resolve_impl`` must do and nothing more: the needs of K1 (both families) and of the row engine, in the order of ``ImplNeeds``, handed to the pure function."""
    problems = []
    needs = re.search(r"constexpr\s+ImplNeeds\s+kImplNeeds\s*\{([^}]*)\}\s*;", text)
    if not needs:
        return ["the launcher defines no kImplNeeds"]
    parts = [re.sub(r"\s+", "", p) for p in needs.group(1).split(",")]
    if parts != ["static_cast<int64_t>(K1_SMEM_BYTES)", "static_cast<int64_t>(ref::K1_SMEM_BYTES)", "ROWS_SMEM_NEED_BYTES"]:
        problems.append("kImplNeeds is %s, the needs are K1 (optimised), K1 (reference) and the row engine's in that order" % parts)
    body = re.search(r"int\s+resolve_impl\(int requested, int device\)\s*\{(.*?)\n\}", text, re.S)
    if not body:
        return problems + ["the launcher has no resolve_impl"]
    code = re.sub(r"//[^\n]*", "", body.group(1))
    if [re.sub(r"\s+", "", s) for s in code.split(";") if s.strip()] != ["constDeviceCapsc=device_caps(device)", "returnresolve_impl_for(requested,c.major*10+c.minor,c.smem_optin,kImplNeeds)"]:
        problems.append("resolve_impl is not the device's capabilities handed to resolve_impl_for with kImplNeeds: %s" % code.strip()[:200])
    return problems


def test_the_launcher_hands_the_needs_of_the_headers_to_the_pure_function():
    assert launcher_problems(LAUNCH.read_text()) == []


@pytest.mark.parametrize("old,new,expect", [
    ("static_cast<int64_t>(K1_SMEM_BYTES), static_cast<int64_t>(ref::K1_SMEM_BYTES)", "static_cast<int64_t>(ref::K1_SMEM_BYTES), static_cast<int64_t>(K1_SMEM_BYTES)", "kImplNeeds is"),
    ("ref::K1_SMEM_BYTES), ROWS_SMEM_NEED_BYTES}", "ref::K1_SMEM_BYTES), 0}", "kImplNeeds is"),
    ("c.major * 10 + c.minor, c.smem_optin, kImplNeeds", "c.major, c.smem_optin, kImplNeeds", "resolve_impl is not"),
    ("c.smem_optin, kImplNeeds)", "c.smem_optin, ImplNeeds{})", "resolve_impl is not"),
])
def test_a_launcher_that_resolves_with_other_needs_is_caught(old, new, expect):
    text = LAUNCH.read_text()
    assert text.count(old) == 1, old
    assert any(expect in p for p in launcher_problems(text.replace(old, new))), old
