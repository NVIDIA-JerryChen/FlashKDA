"""The environment of Suite Sd and the tests that must not depend on it (plain Python: no GPU; run it without the package conftest, ``pytest --noconftest tests/cfbi/test_decode_routing_env.py``).

Suite Sd is Suite S with ``FLASH_KDA_CFBI_IMPL=64``: every ``decode`` / ``prefill`` call that passes ``_impl=None`` (or no ``_impl``) is then a request for the two launches (bit 64), and the hidden
calibration control, which turns the single launch on for every eligible call, never overrides that request. A test that asserts the single launch therefore has to pass ``_impl=0``
itself; a test that leaves ``_impl`` out would capture the plan kernel under Sd and fail its own launch assertion (the graph test of the single-launch module did, once). Two controls:

* the environment routing: ``api._impl_arg`` turns ``None`` into 64 under the variable and leaves an explicit ``_impl=0`` alone, and the specification of the policy (``decode_policy.eligible``,
  which the binding is tested against) says that the request for two launches is not eligible for the single launch with or without the control while an explicit 0 is;
* the source rule: every ``decode`` / ``prefill`` / ``forward_train`` call of the single-launch module (the module whose tests assert launch sets) passes ``_impl`` explicitly and not as ``None``;
  the rule itself fails on a call without it and on ``_impl=None``.
"""
import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import decode_policy as dp  # noqa: E402
from flash_kda.cfbi import api  # noqa: E402

TESTS = ROOT / "tests" / "cfbi"
FORCED_PATH_MODULES = ("test_decode_single_launch.py",)       # the modules whose tests assert launch sets of the decode call itself (the ones that run under Suite Sd and must mean the same there)
CALLS = ("decode", "prefill", "forward_train")                # the public API calls that take ``_impl``
ENV = "FLASH_KDA_CFBI_IMPL"


def test_the_environment_pins_a_call_without_impl_and_leaves_an_explicit_impl_alone(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert [api._impl_arg(v) for v in (None, 0, 3, 64)] == [0, 0, 3, 64]
    monkeypatch.setenv(ENV, str(dp.IMPL_DECODE_PLAN))                       # Suite Sd
    assert [api._impl_arg(v) for v in (None, 0, 3, 64)] == [dp.IMPL_DECODE_PLAN, 0, 3, 64]


def test_under_suite_sd_a_call_without_impl_asks_for_two_launches_and_the_control_does_not_override_it(monkeypatch):
    monkeypatch.setenv(ENV, str(dp.IMPL_DECODE_PLAN))
    unpinned, explicit = api._impl_arg(None), api._impl_arg(0)
    for N, H in ((1, 1), (4, 6), (75, 12), (1000, 12)):
        assert not dp.eligible(1, unpinned, N, H), (N, H)                    # bit 64: the plan kernel, whatever the control says
        assert dp.eligible(1, explicit, N, H), (N, H)                        # an explicit 0: eligible, so the control makes it one launch
    monkeypatch.delenv(ENV)
    assert dp.eligible(1, api._impl_arg(None), 4, 6)                         # without the variable the same call is eligible: the meaning of a call without _impl is the environment's


def impl_problems(source):
    """The calls of ``source`` to ``decode`` / ``prefill`` / ``forward_train`` that do not pass ``_impl`` explicitly (missing, or the constant ``None``), as ``(line, call)``."""
    out = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", None))         # ``mod.f(...)`` or ``f(...)``
        if name not in CALLS:
            continue
        impl = [k for k in node.keywords if k.arg == "_impl"]
        if not impl or (isinstance(impl[0].value, ast.Constant) and impl[0].value.value is None):
            out.append((node.lineno, name))
    return out


@pytest.mark.parametrize("module", FORCED_PATH_MODULES)
def test_every_api_call_of_the_forced_path_tests_passes_impl_explicitly(module):
    source = (TESTS / module).read_text()
    assert "decode(" in source or "prefill(" in source
    assert impl_problems(source) == []


@pytest.mark.parametrize("snippet,expect", [
    ("decode(q, k, v, g, beta, scale, out, A, dt, lb, st)", 1),
    ("prefill(q, k, v, g, beta, scale, out, A, dt, lb, st, cu, out_fp32=o32)", 1),
    ("decode(q, k, v, g, beta, scale, out, A, dt, lb, st, _impl=None)", 1),
    ("api.decode(q, k, v, g, beta, scale, out, A, dt, lb, st)", 1),
    ("decode(q, k, v, g, beta, scale, out, A, dt, lb, st, _impl=0)", 0),
    ("decode(q, k, v, g, beta, scale, out, A, dt, lb, st, _impl=impl)", 0),
    ("prefill(q, k, v, g, beta, scale, out, A, dt, lb, st, cu, _mode=1, _impl=path.impl)", 0),
    ("something_else(q, k)", 0),
])
def test_the_source_rule_fails_on_a_call_without_an_explicit_impl(snippet, expect):
    assert len(impl_problems(snippet)) == expect, snippet
