"""The implementation bits and the accepted range of ``impl`` against every place that states them (plain Python, no GPU: ``pytest --noconftest tests/cfbi/test_impl_range_mirrors.py``).

``csrc/cfbi/cfbi_policy.h`` defines the bits once (``IMPL_*``), the largest accepted value (``IMPL_MAX``) and the next free bit, which is reserved (``IMPL_ROWS_SWAP_RESERVED``). The same
range is written out in the messages of the binding, the documentation of ``_impl``, the hardware report, the benchmark's usage text, the tests that sweep or reject values and the
performance document, and a mirror that falls behind the header (a value that is rejected where the header accepts it, a bit that the report cannot name) must be a failure here:

* the bits are distinct single bits without holes below ``IMPL_MAX``, the reserved bit is the next one, and no second definition of a bit exists anywhere in the sources;
* both messages of the binding, ``api.py``, the usage text of ``bench_breakdown.py`` and the performance document name the range, the first message names every bit, ``api.py`` names the
  reserved bit, ``verify.py`` names every bit that ``resolve_impl`` can return, and ``test_arch.py`` / ``test_k2_dyn.py`` sweep and reject the values that the header says.
"""
import math
import re
import sys
from pathlib import Path

import pytest

import decode_policy as dp

ROOT = Path(__file__).resolve().parents[2]
CSRC = ROOT / "csrc" / "cfbi"
HEADER, LAUNCH = CSRC / "cfbi_policy.h", CSRC / "cfbi_launch.cu"
MIRRORS = {"binding": CSRC / "cfbi_binding.cpp", "api": ROOT / "flash_kda" / "cfbi" / "api.py", "verify": ROOT / "flash_kda" / "cfbi" / "verify.py",
           "bench": ROOT / "benchmarks" / "bench_breakdown.py", "arch": ROOT / "tests" / "cfbi" / "test_arch.py", "invalid": ROOT / "tests" / "cfbi" / "test_k2_dyn.py",
           "perf": ROOT / "docs" / "20261103-cfbi-perf.md"}
NOT_BITS = ("IMPL_MAX", "IMPL_ROWS_SWAP_RESERVED")


def read_all():
    return {name: p.read_text() for name, p in MIRRORS.items()}


def header_ints(text):
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"constexpr\s+int\s+(IMPL_\w+)\s*=\s*(\d+)\s*;", text)}


def bit_problems(header_text, sources):
    """The bits of the header, and the sources (``{path: text}``) that must not define any of them again."""
    consts = header_ints(header_text)
    bits = {n: v for n, v in consts.items() if n not in NOT_BITS}
    probs = []
    for n, v in bits.items():
        if v < 1 or v & (v - 1):
            probs.append("%s = %d is not a single bit" % (n, v))
    if len(set(bits.values())) != len(bits):
        probs.append("two names claim the same bit: %s" % sorted(n for n in bits if list(bits.values()).count(bits[n]) > 1))
    top = max(bits.values())
    if sorted(bits.values()) != [1 << i for i in range(int(math.log2(top)) + 1)]:
        probs.append("the bits %s are not every bit up to the highest one" % sorted(bits.values()))
    if consts.get("IMPL_MAX") != 2 * top - 1:
        probs.append("IMPL_MAX = %s, the bits up to %d make %d" % (consts.get("IMPL_MAX"), top, 2 * top - 1))
    if consts.get("IMPL_ROWS_SWAP_RESERVED") != 2 * top:
        probs.append("the reserved bit is %s, the next free bit is %d" % (consts.get("IMPL_ROWS_SWAP_RESERVED"), 2 * top))
    for path, text in sources.items():
        for name in list(bits) + list(NOT_BITS):
            if re.search(r"(?:constexpr\s+int|#\s*define)\s+%s\b" % name, text):
                probs.append("%s defines %s again" % (path, name))
    return probs


def csrc_sources(skip=("cfbi_policy.h",)):
    return {str(f.relative_to(CSRC)): f.read_text() for f in sorted(CSRC.rglob("*")) if f.suffix in (".cu", ".cuh", ".h", ".cpp") and f.name not in skip}


def resolvable_bits(header_text, launch_text):
    """The bits that ``resolve_impl`` lets through (the mask of its first statement): the values that the hardware report can see."""
    consts = header_ints(header_text)
    mask = re.search(r"int impl = requested & \(([^)]*)\);", launch_text)
    assert mask, "the mask of resolve_impl is not found"
    return sorted(consts[n.strip()] for n in mask.group(1).split("|"))


def mirror_problems(header_text, launch_text, texts):
    consts = header_ints(header_text)
    top, reserved = consts["IMPL_MAX"], consts["IMPL_ROWS_SWAP_RESERVED"]
    rng = "0..%d" % top
    spaced = "0 .. %d" % top
    probs = []
    b = texts["binding"]
    if b.count("impl must be %s" % rng) != 2:
        probs.append("the binding does not state 'impl must be %s' in both of its messages" % rng)
    first = re.search(r'"impl must be [^"]*"(?:\s*"[^"]*")*', b)
    for i in range(int(math.log2(top + 1))):
        if first is None or "bit %d" % i not in first.group(0):
            probs.append("the first message of the binding does not name bit %d" % i)
    a = texts["api"]
    if spaced not in a or "value %d, is reserved" % reserved not in a:
        probs.append("api.py does not state the range %s and the reserved value %d" % (spaced, reserved))
    name_fn = re.search(r"def _impl_name\(impl\):(.*?)\ndef ", texts["verify"], re.S)
    for bit in resolvable_bits(header_text, launch_text):
        if name_fn is None or not re.search(r"impl & %d\b" % bit, name_fn.group(1)):
            probs.append("verify.py _impl_name does not name bit %d, which resolve_impl lets through" % bit)
    if spaced not in texts["bench"]:
        probs.append("the usage text of bench_breakdown.py does not state the range %s" % spaced)
    if "range(%d)" % (top + 1) not in texts["arch"]:
        probs.append("test_arch.py does not sweep range(%d)" % (top + 1))
    if not re.search(r"for bad in \(%d," % reserved, texts["invalid"]):
        probs.append("test_k2_dyn.py does not reject %d (the reserved value)" % reserved)
    if spaced not in texts["perf"]:
        probs.append("the performance document does not state the range %s" % spaced)
    if consts["IMPL_REF_ROWS"] != dp.IMPL_REF_ROWS or consts["IMPL_DECODE_PLAN"] != dp.IMPL_DECODE_PLAN:
        probs.append("decode_policy.py uses other bits than the header")
    return probs


@pytest.fixture(scope="module")
def header():
    return HEADER.read_text()


@pytest.fixture(scope="module")
def launch():
    return LAUNCH.read_text()


@pytest.fixture(scope="module")
def texts():
    return read_all()


def test_the_bits_of_the_header_are_assigned_once(header):
    assert bit_problems(header, csrc_sources()) == []
    assert header_ints(header)["IMPL_DECODE_PLAN"] == 64 and header_ints(header)["IMPL_MAX"] == 127 and header_ints(header)["IMPL_ROWS_SWAP_RESERVED"] == 128


def mutate(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


@pytest.mark.parametrize("old,new,expect", [
    ("IMPL_MAX = 127;", "IMPL_MAX = 63;", "IMPL_MAX = 63"),
    ("IMPL_ROWS_SWAP_RESERVED = 128;", "IMPL_ROWS_SWAP_RESERVED = 256;", "the reserved bit is 256"),
    ("IMPL_DECODE_PLAN = 64;", "IMPL_DECODE_PLAN = 32;", "claim the same bit"),
    ("IMPL_DECODE_PLAN = 64;", "IMPL_DECODE_PLAN = 96;", "not a single bit"),
    ("IMPL_DECODE_PLAN = 64;", "IMPL_DECODE_PLAN = 128;", "not every bit up to the highest"),
])
def test_a_bit_that_breaks_the_scheme_is_caught(header, old, new, expect):
    assert any(expect in p for p in bit_problems(mutate(header, old, new), csrc_sources()))


def test_a_second_claim_of_a_bit_in_another_source_is_caught(header):
    planted = dict(csrc_sources())
    planted["cfbi_common.cuh"] += "\nconstexpr int IMPL_DECODE_PLAN = 64;\n"
    assert any("defines IMPL_DECODE_PLAN again" in p for p in bit_problems(header, planted))
    planted = dict(csrc_sources())
    planted["cfbi_binding.cpp"] += "\n#define IMPL_NEW_BIT 128\n"
    planted["cfbi_k2_dyn.h"] = "constexpr int IMPL_ROWS_SWAP_RESERVED = 128;\n"
    assert any("IMPL_ROWS_SWAP_RESERVED again" in p for p in bit_problems(header, planted))


def test_the_mirrors_state_the_range_and_the_bits(header, launch, texts):
    assert mirror_problems(header, launch, texts) == []
    assert resolvable_bits(header, launch) == [1, 2, 4, 8, 16, 64]


@pytest.mark.parametrize("where,old,new,expect", [
    ("binding", "impl must be 0..127 (bit 0:", "impl must be 0..63 (bit 0:", "both of its messages"),
    ("binding", '"impl must be 0..127");', '"impl must be 0..63");', "both of its messages"),
    ("binding", "bit 6: decode keeps the separate plan kernel", "bit 7: decode keeps the separate plan kernel", "does not name bit 6"),
    ("api", "The accepted values are 0 .. 127 (bit 7, value 128, is reserved and rejected).", "The accepted values are 0 .. 63.", "api.py does not state"),
    ("verify", "if impl & 64 else name", "if impl & 32 else name", "does not name bit 64"),
    ("verify", "if impl & 8 and not impl & 2:", "if impl & 7 and not impl & 2:", "does not name bit 8"),
    ("bench", "every value 0 .. 127 is accepted", "every value 0 .. 63 is accepted", "bench_breakdown.py"),
    ("arch", "for impl in range(128):", "for impl in range(64):", "test_arch.py does not sweep"),
    ("invalid", "for bad in (128, 63,", "for bad in (64, 63,", "does not reject 128"),
    ("perf", "the accepted values are 0 .. 127", "the accepted values are 0 .. 63", "performance document"),
])
def test_a_mirror_that_falls_behind_the_header_is_caught(header, launch, texts, where, old, new, expect):
    mutated = dict(texts, **{where: mutate(texts[where], old, new)})
    assert any(expect in p for p in mirror_problems(header, launch, mutated)), where


def test_the_report_names_the_bits_that_resolve_impl_can_return(header, launch):
    """A bit that ``resolve_impl`` starts to let through is a bit that the report must name: the mask is read from the source."""
    widened = mutate(launch, "IMPL_K2_DYN | IMPL_DECODE_PLAN);", "IMPL_K2_DYN | IMPL_DECODE_PLAN | IMPL_K2_OLD);")
    assert resolvable_bits(header, widened) == [1, 2, 4, 8, 16, 32, 64]
    probs = mirror_problems(header, widened, read_all())
    assert any("does not name bit 32" in p for p in probs), probs
