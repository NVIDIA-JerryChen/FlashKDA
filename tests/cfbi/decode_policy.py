"""Specification of the single-launch decode policy (``decode_single_launch_policy`` of csrc/cfbi/cfbi_policy.h), in integer arithmetic. Plain Python (no torch), importable on a host without a GPU.

A call is ONE kernel launch of the optimised row engine (``rows_kernel<true>``, no plan kernel) iff

  * ``mode`` is 1 (decode; the caller passes the mode that ``resolve_mode`` returns, so mode 0 with ``IMPL_ROWS_ONLY`` is a 1 here),
  * the resolved ``impl`` has neither ``IMPL_REF_ROWS`` (the frozen oracle keeps the plan kernel; ``resolve_impl`` sets the bit below sm_90, where the optimised kernels cannot run) nor
    ``IMPL_DECODE_PLAN`` (bit 64, which asks for the two launches),
  * the grid exists (N >= 1 sequences, H >= 1 heads) and the SM count is positive (a failed device query keeps the plan kernel),
  * the device is in the routing set (compute capability major 10: B200 and B300), and
  * N * H, the CTAs of the grid, is at most ``MAX_CTAS_PER_SM`` times the SM count.

The product form below is deliberately not the header's ``N <= limit / H``: the two are equal for positive numbers, and the matrix (``shape_matrix``) puts a point on each side of the limit
of every (SM count, heads) pair, so a gate that is one CTA looser or tighter, a wrong routing set or a missing exclusion is a difference (the controls of test_decode_policy.py).

Two more things live here because the benchmarks and the tests read them from one place: ``expected_route`` (the launches of a decode call as the preset names of the decode benchmark,
with the hidden calibration control off or on) and ``header_per_sm`` (the limit constant of the policy header of a source tree, which may be another tree than the one that holds this file).
"""
import re
from pathlib import Path

IMPL_REF_ROWS = 1
IMPL_DECODE_PLAN = 64
IMPL_MAX = 127                                                # the largest accepted impl value: every bit up to IMPL_DECODE_PLAN
MAX_CTAS_PER_SM = 8
ROUTING_CC_MAJOR = 10

SM_COUNTS = (114, 132, 148)                                   # H100 PCIe, H100 SXM, B200 / B300
CAPABILITIES = ((9, 0), (10, 0), (10, 3), (12, 0))            # H100, B200, B300, sm_120
HEADS = (6, 12, 33)                                           # the heads of the shape matrix
EXTRA_HEADS = (1, 8)                                          # one head: the limit itself and one CTA above it; eight: a limit that N * H reaches exactly (for the SM counts above)

SINGLE_LAUNCH, TWO_LAUNCH = "single-launch", "two-launch"     # the launches of one decode call, named as the presets of benchmarks/bench_decode.py (CANONICAL_COMPUTE) name them
POLICY_HEADER = Path("csrc") / "cfbi" / "cfbi_policy.h"       # the header, relative to the root of a source tree
LIMIT_CONSTANT = "DECODE_SINGLE_MAX_CTAS_PER_SM"              # its constant for the limit of the grid, in CTAs per SM
HEADER_INTEGER = re.compile(r"constexpr\s+(?:int|int64_t)\s+(\w+)\s*=\s*(\d+)\s*;")
HEADER_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)                 # a block comment or a line comment, whichever starts first


def cc_number(capability):
    """The compute capability as the policy takes it: major * 10 + minor."""
    return 10 * capability[0] + capability[1]


def eligible(mode, impl, N, H):
    return mode == 1 and not impl & (IMPL_REF_ROWS | IMPL_DECODE_PLAN) and N >= 1 and H >= 1


def routed(cc, major=ROUTING_CC_MAJOR):
    return cc // 10 == major


def policy(mode, impl, N, H, sms, cc, per_sm=MAX_CTAS_PER_SM, slack=0, major=ROUTING_CC_MAJOR, blocked=IMPL_REF_ROWS | IMPL_DECODE_PLAN, modes=(1,)):
    """The specification. The keyword arguments are the knobs of the controls (a looser or tighter gate, another routing set, a missing exclusion, mode 0 allowed): the defaults are the rule."""
    return mode in modes and not impl & blocked and N >= 1 and H >= 1 and sms > 0 and routed(cc, major) and N * H <= per_sm * sms + slack


def shape_matrix():
    """``[(mode, impl, N, H, sms, cc)]``: for every SM count and capability and every H, the largest N whose grid holds at most the limit and the next larger N; the other conditions
    (mode 0, ``IMPL_REF_ROWS``, bit 64, the other bits, a grid or an SM count that does not exist) at one shape."""
    rows = []
    for sms in SM_COUNTS:
        for cap in CAPABILITIES:
            cc = cc_number(cap)
            for H in HEADS + EXTRA_HEADS:
                largest = MAX_CTAS_PER_SM * sms // H
                rows += [(1, 0, N, H, sms, cc) for N in (largest, largest + 1)]
            for mode, impl in ((0, 0), (0, 4), (1, 1), (1, 64), (1, 65), (1, 127), (1, 2), (1, 4), (1, 8), (1, 16), (1, 32), (1, 2 | 4 | 8 | 16 | 32), (1, 4 | 64)):
                rows.append((mode, impl, 4, 12, sms, cc))
            rows += [(1, 0, 0, 12, sms, cc), (1, 0, -1, 12, sms, cc), (1, 0, 4, 0, sms, cc), (1, 0, 4, -3, sms, cc)]
    rows += [(1, 0, 4, 12, sms, cc_number(cap)) for sms in (0, -1) for cap in CAPABILITIES]
    return rows


def expected_route(impl, N, H, sms, cc, per_sm=MAX_CTAS_PER_SM, force=False):
    """The launches of one decode call (mode 1) that the policy specifies, as ``SINGLE_LAUNCH`` (``rows_kernel`` alone) or ``TWO_LAUNCH`` (``plan_kernel`` and ``rows_kernel``): the preset
    names of the decode benchmark. ``impl`` is the implementation bits of the call (the request equals the resolved bits wherever the optimised row engine runs), ``sms`` and ``cc`` the
    device (``cc`` as ``cc_number`` makes it), ``per_sm`` the limit of the tree that is measured.

    ``force`` is the hidden calibration control: it answers the eligible part of the rule and nothing else, so with it on the call is one launch whatever the grid and the device
    (``eligible``: a grid that exists, no bit 64 and no reference row engine). Bit 64 asks for the two launches and is never overridden, with the control on or off. The reference row engine
    (bit 1: what ``resolve_impl`` sets below sm_90) is two launches too, but its kernels are the preset ``reference``, which this function does not name: do not ask for its route."""
    single = eligible(1, impl, N, H) if force else policy(1, impl, N, H, sms, cc, per_sm)
    return SINGLE_LAUNCH if single else TWO_LAUNCH


def header_values(text):
    """The integer literals of the ``constexpr`` constants of a policy header (a constant that is an expression of others is not listed; comments are not read)."""
    return {m.group(1): int(m.group(2)) for m in HEADER_INTEGER.finditer(HEADER_COMMENT.sub("", text))}


def header_per_sm(root):
    """The limit of the single launch, in CTAs per SM, that the policy header of the source tree ``root`` defines (``DECODE_SINGLE_MAX_CTAS_PER_SM`` of ``csrc/cfbi/cfbi_policy.h``): the limit of
    the C++ function of THAT tree, whichever tree this module comes from. ``OSError`` when the tree has no policy header, ``ValueError`` when the header does not define the constant as an
    integer literal."""
    path = Path(root) / POLICY_HEADER
    values = header_values(path.read_text())
    if LIMIT_CONSTANT not in values:
        raise ValueError("%s does not define %s as an integer literal" % (path, LIMIT_CONSTANT))
    return values[LIMIT_CONSTANT]
