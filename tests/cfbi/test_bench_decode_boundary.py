"""The campaign options of the decode benchmark (``benchmarks/bench_decode.py``) and the route specification they use (``tests/cfbi/decode_policy.py``). CPU only: no torch, no GPU, no
conftest (``--noconftest``). The measurement itself is NOT run here: a fake extension, fake inputs, a fake profiler and a fake ``torch`` stand in for the device, so these tests cover the cell
sets, the option rules, the records that the options write and the order in which a run switches the control on, asserts and restores it, not one timing and not one real launch.

* the default cells and the defaults of every older option are unchanged (retained records are bound to them);
* ``boundary_shapes`` / ``campaign_cell_specs``: the grids that straddle half the limit, the limit and twice the limit for 6, 12 and 33 heads, 18 shapes and 92 cells for 148 SMs and
  8 CTAs per SM, other shapes for another SM count or limit, each shape straddling its x exactly and none of them in the default matrix;
* ``expected_route``: literal cases (bit 64, the control, the routing set, the limit boundary at 12 heads, the limit of another tree) and the partition of the campaign cells it gives;
* the options: ``--boundary`` with and without an SM count, ``--single-force``, ``--expect-route spec`` (not together with the other two spellings of an expectation), ``--graph-nodes``;
* what a run records: ``single_force`` and ``single_per_sm`` in the meta, ``single_force`` and ``expect_route`` in every cell, the mapping that each cell was asserted against;
* the order of a run: the control is on before the first graph is captured and restored when the run ends, by a failing trial too; the extension is asked the specification's question
  before anything is measured; a cell on the other route fails its own assertion; one graph replay per trial is profiled and asserted when that is asked for, and the default run does not.
"""
import functools
import itertools
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "benchmarks", ROOT / "tests" / "cfbi"):                      # the modules under test, whatever the working directory and the PYTHONPATH of the run
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bench_decode as bd  # noqa: E402
import bench_records as br  # noqa: E402
import decode_policy as dp  # noqa: E402

START = 8                                                            # the limit (CTAs per SM) that the worked-out cells, routes and counts of this module are for
REAL_TREE_PER_SM, REAL_ROOT = bd.tree_per_sm, bd.ROOT               # the reader of a tree's limit and the measured tree, before the fixture below stands in for the header of the real tree


@pytest.fixture(autouse=True)
def the_tree_of_the_starting_limit(monkeypatch):
    """The driver reads the limit of the tree that it measures. The figures below are worked out for a tree whose header says START CTAs per SM, so the real tree stands in as that tree (a
    fake tree with another header is read as it is); the limit of the real tree is another number, and one test reads it from the header."""
    monkeypatch.setattr(bd, "tree_per_sm", lambda root: START if Path(root) == REAL_ROOT else REAL_TREE_PER_SM(root))


B200, B300, H100, SM120 = 100, 103, 90, 120                          # compute capabilities as the policy takes them (major * 10 + minor)
DEVICE_B200 = {"name": "NVIDIA B200", "sm_count": 148, "cc": "10.0", "l2_bytes": 126 * 2 ** 20, "eviction_bytes": 2 * 126 * 2 ** 20 + 2 ** 20}
DEVICE_H100 = dict(DEVICE_B200, name="NVIDIA H100", sm_count=132, cc="9.0")

BOUNDARY_148 = [(17, 33), (18, 33), (35, 33), (36, 33), (49, 12), (50, 12), (71, 33), (72, 33), (98, 6), (98, 12), (99, 6), (99, 12), (197, 6), (197, 12), (198, 6), (198, 12),
                (394, 6), (395, 6)]                                    # per_sm 8 on 148 SMs: x = 592, 1184, 2368, for H = 33, 12 and 6, the largest N and the next one
BOUNDARY_4_132 = [(8, 33), (9, 33), (16, 33), (17, 33), (22, 12), (23, 12), (32, 33), (33, 33), (44, 6), (44, 12), (45, 6), (45, 12), (88, 6), (88, 12), (89, 6), (89, 12),
                  (176, 6), (177, 6)]                                  # per_sm 4 on 132 SMs: limit 528, x = 264, 528, 1056: for the largest N of every pair N * H reaches x exactly
NOT_ROUTED_148 = [(36, 33), (71, 33), (72, 33), (99, 12), (128, 12), (197, 12), (198, 6), (198, 12), (256, 12), (394, 6), (395, 6)]     # N * H above 1184: the starting rule keeps the plan kernel

DEFAULT_OPTIONS = dict(impl=0, shapes=None, regimes=["lockstep", "staggered", "cold", "probe"], probe_shapes=None, positions=[0, 1, 15, 16, 31, 32, 47, 48, 62, 63], trials=5, calls=1536,
                       cold_calls=256, probe_calls=256, profile_calls=64, calls_scale=1, cells=None, expect_compute=None, expect_kernels=None, json=None, label="", list_cells=False)
NEW_OPTIONS = dict(boundary=None, single_force=False, expect_route=None, graph_nodes=False, route_order=False)


# ---- the cells ---------------------------------------------------------------------------------------------------------------------------------------------
def straddle_problems(shapes, per_sm, sms, heads=(6, 12, 33)):
    """Independent of ``boundary_shapes``: for every head count and every x = limit / 2, limit, 2 * limit (the half rounded down) the shapes of that head count are the largest N with
    N * H <= x and the next one, and nothing else (a shape without a sequence is no shape)."""
    limit, out = per_sm * sms, []
    for H in heads:
        mine = {N for N, h in shapes if h == H}
        want = set()
        for x in (limit // 2, limit, 2 * limit):
            low = max(n for n in range(0, x + 1) if n * H <= x)
            assert low * H <= x < (low + 1) * H
            want |= {n for n in (low, low + 1) if n >= 1}
        if mine != want:
            out.append((H, sorted(mine), sorted(want)))
    return out


def test_the_default_cells_are_still_the_38_cells_of_matrix_d_and_the_probe():
    specs = bd.cell_specs()
    keys = [s["key"] for s in specs]
    assert len(specs) == 38 and len(set(keys)) == 38
    assert bd.MATRIX_D == ((1, 12), (4, 12), (16, 12), (64, 12), (128, 12), (256, 12)) and bd.PROBE_SHAPES == ((1, 12), (64, 12))
    assert keys[:4] == ["decode;N=1;H=12;regime=lockstep;", "decode;N=1;H=12;regime=staggered;", "decode;N=1;H=12;regime=cold;", "decode;N=1;H=12;regime=probe;pos=0;"]
    assert keys[-1] == "decode;N=256;H=12;regime=cold;"
    assert [sum(s["regime"] == r for s in specs) for r in bd.REGIMES] == [6, 6, 6, 20]


def test_the_options_of_the_older_flags_keep_their_defaults():
    a = vars(bd.build_parser().parse_args([]))
    assert {k: a[k] for k in DEFAULT_OPTIONS} == DEFAULT_OPTIONS
    assert {k: a[k] for k in NEW_OPTIONS} == NEW_OPTIONS and set(a) == set(DEFAULT_OPTIONS) | set(NEW_OPTIONS)
    assert bd.DEFAULTS == {"trials": 5, "calls": 1536, "cold_calls": 256, "probe_calls": 256, "profile_calls": 64, "calls_scale": 1}


def test_the_boundary_shapes_of_148_sms_and_8_ctas_per_sm_are_the_18_that_straddle_the_limit():
    assert (bd.BOUNDARY_HEADS, bd.BOUNDARY_FRACTIONS) == ((6, 12, 33), ((1, 2), (1, 1), (2, 1)))
    shapes = bd.boundary_shapes(8, 148)
    assert shapes == BOUNDARY_148 and shapes == sorted(shapes) and len(shapes) == 18
    assert straddle_problems(shapes, 8, 148) == []
    assert set(shapes).isdisjoint(bd.MATRIX_D)
    assert sorted({n * h for n, h in shapes if n * h <= 1184}) == [561, 588, 594, 600, 1155, 1176, 1182]                     # at or below the limit of 8 * 148 = 1184
    assert sorted({n * h for n, h in shapes if n * h > 1184}) == [1188, 2343, 2364, 2370, 2376]                              # above it


def test_each_boundary_shape_straddles_its_limit_exactly_and_the_shapes_follow_the_arguments():
    for per_sm, sms in ((8, 148), (8, 132), (4, 148), (16, 114), (3, 133), (8, 147), (1, 148)):
        shapes = bd.boundary_shapes(per_sm, sms)
        assert straddle_problems(shapes, per_sm, sms) == [], (per_sm, sms)
        assert shapes == sorted(set(shapes)) and all(n >= 1 for n, h in shapes)
    assert bd.boundary_shapes(4, 132) == BOUNDARY_4_132                                       # N * H reaches x exactly (16 * 33 = 528): that N still belongs to the limit
    assert len({tuple(bd.boundary_shapes(*a)) for a in ((8, 148), (8, 132), (4, 148), (4, 132))}) == 4
    assert bd.boundary_shapes(8, 148, heads=(12,), fractions=((1, 1),)) == [(98, 12), (99, 12)]
    assert bd.boundary_shapes(8, 148, heads=(12, 6), fractions=((2, 1), (1, 2))) == [(49, 12), (50, 12), (98, 6), (99, 6), (197, 12), (198, 12), (394, 6), (395, 6)]


def test_half_of_an_odd_limit_is_rounded_down_and_a_shape_without_a_sequence_is_left_out():
    shapes = bd.boundary_shapes(3, 133)                                                       # limit 399, half 199.5: 199 holds the same integers N * H
    assert {(33, 6), (34, 6), (6, 33), (7, 33)} <= set(shapes)
    assert (32, 6) in bd.boundary_shapes(5, 79)                                               # limit 395, half 197.5: rounded down 197 holds 32 sequences of 6 heads, rounded up 198 would hold 33
    assert bd.boundary_shapes(1, 4) == [(1, 6), (1, 12), (1, 33), (2, 6)]                    # limit 4: x = 2, 4, 8 hold no sequence of 6 or more CTAs but the next larger N is one
    for bad in ((0, 148), (8, 0), (-1, 148)):
        with pytest.raises(ValueError):
            bd.boundary_shapes(*bad)


def test_the_campaign_cells_are_matrix_d_and_the_boundary_shapes_in_four_regimes():
    specs = bd.campaign_cell_specs(8, 148)
    keys = [s["key"] for s in specs]
    assert len(specs) == 92 and len(set(keys)) == 92
    shapes = sorted({(s["N"], s["H"]) for s in specs})
    assert shapes == sorted(set(bd.MATRIX_D) | set(BOUNDARY_148)) and len(shapes) == 24
    assert [sum(s["regime"] == r for s in specs) for r in bd.REGIMES] == [24, 24, 24, 20]    # the probe stays at N = 1 and 64, ten positions each
    assert {(s["N"], s["H"]) for s in specs if s["regime"] == "probe"} == set(bd.PROBE_SHAPES)
    assert [(s["N"], s["H"]) for s in specs] == sorted((s["N"], s["H"]) for s in specs)       # grouped by shape: one set of inputs and one graph per shape
    assert {s["key"] for s in bd.cell_specs()} <= set(keys)                                    # the default cells are inside the campaign cells, with the same keys
    assert [s["key"] for s in bd.campaign_cell_specs(8, 148)] == keys                          # a pure function of its arguments


def test_another_sm_count_or_limit_gives_other_cells():
    base = {s["key"] for s in bd.campaign_cell_specs(8, 148)}
    for per_sm, sms in ((8, 132), (4, 148), (16, 148)):
        other = bd.campaign_cell_specs(per_sm, sms)
        assert {s["key"] for s in other} != base and len(other) == 92, (per_sm, sms)
        assert {(s["N"], s["H"]) for s in other} == set(bd.MATRIX_D) | set(bd.boundary_shapes(per_sm, sms))
    assert {(s["N"], s["H"]) for s in bd.campaign_cell_specs(4, 132)} == set(bd.MATRIX_D) | set(BOUNDARY_4_132)


# ---- the route specification -------------------------------------------------------------------------------------------------------------------------------
def test_expected_route_on_literal_cases():
    route, one, two = functools.partial(dp.expected_route, per_sm=START), "single-launch", "two-launch"
    assert (dp.SINGLE_LAUNCH, dp.TWO_LAUNCH) == (one, two) and {one, two} < set(bd.CANONICAL_COMPUTE)
    assert bd.CANONICAL_COMPUTE[one] == {"rows_kernel": 1} and bd.CANONICAL_COMPUTE[two] == {"plan_kernel": 1, "rows_kernel": 1}
    # the limit boundary at 12 heads on 148 SMs (1184 CTAs): 98 sequences are 1176 CTAs, 99 are 1188
    assert route(0, 98, 12, 148, B200) == one and route(0, 99, 12, 148, B200) == two
    assert route(0, 98, 12, 148, B300) == one and route(0, 99, 12, 148, B300) == two
    assert route(0, 1, 12, 148, B200) == one and route(0, 197, 6, 148, B200) == one and route(0, 198, 6, 148, B200) == two
    # the routing set: compute capability 10 only
    assert route(0, 4, 12, 148, B200) == route(0, 4, 12, 148, B300) == one
    assert route(0, 4, 12, 132, H100) == route(0, 4, 12, 148, SM120) == two
    # bit 64 asks for two launches whatever the control, and so does every request that holds it
    for impl in (64, 65, 66, 68, 127):
        assert route(impl, 4, 12, 148, B200) == route(impl, 4, 12, 148, B200, force=True) == two, impl
    # the control skips the limit and the routing set, not the eligibility
    assert route(0, 99, 12, 148, B200, force=True) == one and route(0, 10 ** 6, 12, 148, B200, force=True) == one
    assert route(0, 4, 12, 132, H100, force=True) == route(0, 4, 12, 148, SM120, force=True) == one
    assert route(1, 4, 12, 148, B200, force=True) == two and route(3, 4, 12, 148, B200, force=True) == two          # the reference row engine keeps the plan kernel
    assert route(0, 0, 12, 148, B200, force=True) == route(0, 4, 0, 148, B200, force=True) == route(0, -1, 12, 148, B200, force=True) == two        # no grid: nothing to launch
    assert route(4, 4, 12, 132, H100, force=True) == one                                                           # another bit does not matter
    # the limit of the tree that is measured
    assert route(0, 99, 12, 148, B200, per_sm=9) == one and route(0, 98, 12, 148, B200, per_sm=7) == two
    assert route(0, 111, 12, 148, B200, per_sm=9) == one and route(0, 112, 12, 148, B200, per_sm=9) == two           # 9 * 148 = 1332 = 111 * 12


def test_the_starting_rule_partitions_the_campaign_cells_into_59_one_launch_cells_and_33_two_launch_cells():
    specs = bd.campaign_cell_specs(8, 148)
    for device in (DEVICE_B200, dict(DEVICE_B200, name="NVIDIA B300", cc="10.3")):
        routes = bd.route_expectations(specs, 0, device, 8, False)
        assert set(routes) == {s["key"] for s in specs} and set(routes.values()) == {"single-launch", "two-launch"}
        assert sum(v == "single-launch" for v in routes.values()) == 59 and sum(v == "two-launch" for v in routes.values()) == 33
        assert sorted({(s["N"], s["H"]) for s in specs if routes[s["key"]] == "two-launch"}) == NOT_ROUTED_148
        assert all(routes[s["key"]] == "single-launch" for s in specs if s["regime"] == "probe")                  # the probe shapes are far below the limit
    assert set(bd.route_expectations(specs, 0, DEVICE_B200, 8, True).values()) == {"single-launch"}                  # the control: every cell
    assert set(bd.route_expectations(specs, 64, DEVICE_B200, 8, False).values()) == set(bd.route_expectations(specs, 64, DEVICE_B200, 8, True).values()) == {"two-launch"}
    assert set(bd.route_expectations(specs, 0, DEVICE_H100, 8, False).values()) == {"two-launch"}                    # Hopper is outside the routing set
    assert set(bd.route_expectations(specs, 0, DEVICE_H100, 8, True).values()) == {"single-launch"}
    nine = bd.route_expectations(specs, 0, DEVICE_B200, 9, False)                                                    # the limit of another tree moves the partition
    assert sum(v == "single-launch" for v in nine.values()) > 59


def test_the_device_of_the_specification_is_its_sm_count_and_compute_capability():
    assert [bd.device_cc({"cc": c}) for c in ("9.0", "10.0", "10.3", "12.0")] == [90, 100, 103, 120]
    spec = {"N": 1, "H": 12, "regime": "lockstep", "pos": None, "key": "decode;N=1;H=12;regime=lockstep;"}
    assert bd.route_expectations([spec], 0, dict(DEVICE_B200, sm_count=1), 8, False) == {spec["key"]: "two-launch"}        # 12 CTAs against a limit of 8 CTAs on one SM
    assert bd.route_expectations([spec], 0, dict(DEVICE_B200, sm_count=2), 8, False) == {spec["key"]: "single-launch"}


# ---- the limit of the measured tree ------------------------------------------------------------------------------------------------------------------------
def tree_with_header(tmp_path, text, name="tree"):
    """A source tree whose policy header holds ``text`` (None: a tree with the directory and no header)."""
    header = tmp_path / name / "csrc" / "cfbi"
    header.mkdir(parents=True)
    if text is not None:
        (header / "cfbi_policy.h").write_text(text)
    return tmp_path / name


@pytest.fixture()
def fresh_spec():
    bd.policy_spec.cache_clear()
    yield
    bd.policy_spec.cache_clear()


def test_the_limit_of_the_measured_tree_comes_from_its_own_header(tmp_path):
    assert REAL_TREE_PER_SM(ROOT) == dp.MAX_CTAS_PER_SM
    assert bd.tree_per_sm(tree_with_header(tmp_path, "constexpr int64_t DECODE_SINGLE_MAX_CTAS_PER_SM = 12;\n", "a")) == 12
    assert bd.tree_per_sm(tree_with_header(tmp_path, None, "b")) is None                          # a tree from before the single launch: no such header
    assert bd.tree_per_sm(tree_with_header(tmp_path, "constexpr int64_t OTHER = 4;\n", "c")) is None
    assert bd.tree_per_sm(tmp_path / "nowhere") is None


def test_the_specification_is_loaded_from_the_tool_tree_by_its_path(tmp_path, monkeypatch, fresh_spec):
    spec = bd.policy_spec()
    assert Path(spec.__file__).resolve() == (ROOT / "tests" / "cfbi" / "decode_policy.py").resolve() and callable(spec.expected_route) and callable(spec.header_per_sm)
    other = tmp_path / "tests" / "cfbi"                                                         # another tool tree: its module is the specification
    other.mkdir(parents=True)
    (other / "decode_policy.py").write_text("MARK = 'the module of the other tree'\n")
    monkeypatch.setattr(bd, "TOOL_ROOT", tmp_path)
    bd.policy_spec.cache_clear()
    assert bd.policy_spec().MARK == "the module of the other tree" and not hasattr(bd.policy_spec(), "expected_route")


def test_a_measured_tree_with_an_older_copy_of_the_module_does_not_shadow_the_specification(tmp_path):
    """CFBI_BENCH_SRC names the tree that is measured and its tests/cfbi comes first on sys.path: an import by name finds ITS decode_policy.py, the specification must not."""
    older = tmp_path / "tests" / "cfbi"
    older.mkdir(parents=True)
    (older / "decode_policy.py").write_text("MAX_CTAS_PER_SM = 8\n")
    code = ("import pathlib, sys\nsys.path.insert(0, %r)\nimport bench_decode as bd\nimport decode_policy as by_name\n"
            "assert bd.ROOT == pathlib.Path(%r).resolve() and not hasattr(by_name, 'expected_route'), 'the hazard is not there'\n"
            "spec = bd.policy_spec()\nassert callable(spec.expected_route) and callable(spec.header_per_sm) and spec is not by_name\nprint('ok')\n" % (str(ROOT / "benchmarks"), str(tmp_path)))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["CFBI_BENCH_SRC"] = str(tmp_path)
    run = subprocess.run([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, env=env)
    assert run.returncode == 0 and run.stdout.strip() == "ok", run.stdout[-1500:]


# ---- the options -------------------------------------------------------------------------------------------------------------------------------------------
def parse(*argv):
    return bd.build_parser().parse_args(list(argv))


def exit_code_and_message(capsys, *argv):
    with pytest.raises(SystemExit) as e:
        bd.main(list(argv))
    return e.value.code, capsys.readouterr().err


def test_boundary_with_and_without_a_value_and_the_other_new_flags_parse():
    assert parse().boundary is None
    assert parse("--boundary").boundary == bd.FROM_DEVICE == 0                                   # no value: the device's SM count
    assert parse("--boundary", "148").boundary == 148 and parse("--boundary", "132").boundary == 132
    a = parse("--boundary", "--single-force", "--expect-route", "spec", "--graph-nodes")        # a bare --boundary does not take the next option as its value
    assert (a.boundary, a.single_force, a.expect_route, a.graph_nodes) == (0, True, "spec", True)
    a = parse("--boundary", "148", "--cells", "N=1;", "N=99;H=12;", "--impl", "64")
    assert a.boundary == 148 and a.cells == ["N=1;", "N=99;H=12;"] and a.impl == 64
    assert parse("--expect-route", "spec").expect_route == "spec" and parse("--single-force").single_force is True


@pytest.mark.parametrize("argv", [("--boundary", "0"), ("--boundary", "-148"), ("--boundary", "many"), ("--expect-route", "measured"), ("--expect-route",)])
def test_an_invalid_value_of_a_new_option_is_a_usage_error(argv, capsys):
    with pytest.raises(SystemExit) as e:
        bd.build_parser().parse_args(list(argv))
    assert e.value.code == 2 and capsys.readouterr().err


@pytest.mark.parametrize("extra", [("--expect-compute", "two-launch"), ("--expect-compute", "plan_kernel=1;rows_kernel=1"), ("--expect-kernels", "rows_kernel")])
def test_the_route_specification_is_not_combined_with_another_expectation(extra, capsys):
    code, err = exit_code_and_message(capsys, "--expect-route", "spec", *extra, "--list-cells")
    assert code == 2 and "two spellings of one assertion" in err
    code, err = exit_code_and_message(capsys, *extra, "--expect-route", "spec", "--list-cells")                # in either order
    assert code == 2 and "two spellings of one assertion" in err


@pytest.mark.parametrize("clash,argv", [("--shapes", ("--shapes", "4,12")), ("--probe-shapes", ("--probe-shapes", "1,12")), ("--regimes", ("--regimes", "lockstep")),
                                        ("--positions", ("--positions", "0", "1"))])
def test_boundary_fixes_the_cells_and_is_not_combined_with_the_options_that_choose_them(clash, argv, capsys):
    code, err = exit_code_and_message(capsys, "--boundary", "148", *argv, "--list-cells")
    assert code == 2 and "cannot be combined with %s;" % clash in err and "narrow the set with --cells" in err                # the rule's own message, not the usage text


def test_the_cells_are_listed_without_a_device_and_a_bare_boundary_needs_the_sm_count_to_list_them(capsys):
    assert bd.main(["--list-cells"]) == 0
    default = capsys.readouterr().out.split()
    assert default == [s["key"] for s in bd.cell_specs()] and len(default) == 38
    assert bd.main(["--list-cells", "--boundary", "148"]) == 0
    listed = capsys.readouterr().out.split()
    assert listed == [s["key"] for s in bd.campaign_cell_specs(8, 148)] and len(listed) == 92
    assert bd.main(["--list-cells", "--boundary", "132"]) == 0
    assert capsys.readouterr().out.split() == [s["key"] for s in bd.campaign_cell_specs(8, 132)]
    assert bd.main(["--list-cells", "--boundary", "148", "--cells", "N=99;H=12;", "regime=probe;pos=63;"]) == 0              # --cells narrows the campaign cells
    assert capsys.readouterr().out.split() == ["decode;N=1;H=12;regime=probe;pos=63;", "decode;N=64;H=12;regime=probe;pos=63;", "decode;N=99;H=12;regime=lockstep;",
                                               "decode;N=99;H=12;regime=staggered;", "decode;N=99;H=12;regime=cold;"]
    assert bd.main(["--list-cells", "--boundary", "148", "--cells", "N=77;"]) == 0 and capsys.readouterr().out == ""            # a listing of no cell is a listing
    assert bd.main(["--boundary", "148", "--cells", "N=77;"]) == 2 and "no cell selected" in capsys.readouterr().err         # a measurement of none is refused before any device is asked
    code, err = exit_code_and_message(capsys, "--list-cells", "--boundary")
    assert code == 2 and "--boundary 148" in err


def test_a_measured_tree_without_a_limit_cannot_be_asked_for_boundary_cells_or_the_specification(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bd, "ROOT", tree_with_header(tmp_path, None))
    for argv in (("--boundary", "148"), ("--expect-route", "spec")):
        code, err = exit_code_and_message(capsys, *argv, "--list-cells")
        assert code == 2 and "DECODE_SINGLE_MAX_CTAS_PER_SM" in err
    assert bd.main(["--list-cells"]) == 0                                                         # a default run of such a tree needs no limit
    assert len(capsys.readouterr().out.split()) == 38


def test_the_import_and_the_listing_need_no_torch():
    code = ("import sys; sys.path[:0] = %r\nimport bench_decode\nassert bench_decode.main(['--list-cells', '--boundary', '148']) == 0\n"
            "assert 'torch' not in sys.modules, 'torch was imported'\n" % [str(ROOT / "benchmarks"), str(ROOT / "tests" / "cfbi")])
    run = subprocess.run([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    assert run.returncode == 0, run.stderr[-1500:]
    assert len(run.stdout.split()) == 92


# ---- fakes: the extension, the inputs, the profiler and torch ----------------------------------------------------------------------------------------------
PLAN_K = "void cfbi::plan_kernel(cfbi::Params)"
ROWS_K = "void cfbi::rows_kernel<false>(cfbi::Params)"
ROWS_K1 = "void cfbi::rows_kernel<true>(cfbi::Params)"
FILL_K = "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>>(int, float)"
ONE, TWO = {"rows_kernel": 1}, {"plan_kernel": 1, "rows_kernel": 1}                      # the mappings of the two routes as a record holds them
COUNTS_ONE, COUNTS_TWO = {ROWS_K1: 1.0}, {PLAN_K: 1.0, ROWS_K: 1.0}                      # the same as the profiler reports them (launches per call)


class FakeExt(object):
    """The entries of the extension that the benchmark uses: the hidden control, the resolved bits and the launch decision of the binding, which answers as cfbi_launch.cu does (eligibility
    first, then the control, then the policy of a device of ``sms`` SMs and compute capability ``cc`` with ``per_sm`` CTAs per SM)."""

    def __init__(self, sms=148, cc=B200, per_sm=8, force=False, resolve=None):
        self.sms, self.cc, self.per_sm, self.force, self.resolve, self.log = sms, cc, per_sm, force, resolve, []

    def decode_single_force(self, on):
        previous, self.force = self.force, bool(on)
        self.log.append(("force", bool(on)))
        return previous

    def resolve_impl(self, impl, device):
        return impl if self.resolve is None else self.resolve(impl)

    def decode_single_launch(self, mode, impl, N, H, device=-1):
        if not dp.eligible(mode, impl, N, H):
            return False
        return self.force or dp.policy(mode, impl, N, H, self.sms, self.cc, self.per_sm)


class FakeBuffer(object):
    def __init__(self):
        self.fills = 0

    def fill_(self, value):
        self.fills += 1
        return self


class FakeEvent(object):
    clock = itertools.count()

    def __init__(self, enable_timing=False):
        self.t = 0

    def record(self):
        self.t = next(FakeEvent.clock)

    def elapsed_time(self, other):
        return 0.01 * (other.t - self.t)


class FakeTorch(types.ModuleType):
    """``torch`` as the benchmark uses it outside the real inputs: the device properties, events, synchronisation and the eviction buffer."""
    __version__ = "2.14.0"
    version = types.SimpleNamespace(cuda="13.4")
    float32 = "float32"

    def __init__(self, sms=148, major=10, minor=0, name="NVIDIA B200"):
        types.ModuleType.__init__(self, "torch")
        prop = types.SimpleNamespace(name=name, multi_processor_count=sms, major=major, minor=minor, L2_cache_size=126 * 2 ** 20)
        self.cuda = types.SimpleNamespace(Event=FakeEvent, synchronize=lambda: None, empty_cache=lambda: None, get_device_properties=lambda i: prop)
        self.buffers = []

    def empty(self, n, dtype=None, device=None):
        self.buffers.append(FakeBuffer())
        return self.buffers[-1]


class FakeShapeEnv(object):
    """The ShapeEnv of the benchmark. A graph bakes the route that was current when it was captured; an eager call asks the extension when it runs. ``World`` carries the extension and the log."""
    world = None

    def __init__(self, N, H, impl):
        w = FakeShapeEnv.world
        self.torch, self.N, self.H, self.impl, self.state = w.torch, N, H, impl, types.SimpleNamespace(pos=FakeBuffer())
        self.eager_calls = self.replay_calls = 0
        self.captured_single = None
        w.env = self

    def route_now(self):
        w = FakeShapeEnv.world
        return w.forced_routes.get((self.N, self.H), w.ext.decode_single_launch(1, self.impl, self.N, self.H))

    def capture(self):
        w = FakeShapeEnv.world
        self.captured_single = self.route_now()
        w.log.append(("capture", self.N, self.H, w.ext.force))

    def set_positions(self, regime):
        pass

    def eager(self, i):
        self.eager_calls += 1

    def replay(self, i):
        self.replay_calls += 1


class World(object):
    """What a fake run needs: the extension, the fake torch, the fake profiler and the log of the events that the order of a run is judged by."""

    def __init__(self, ext=None, torch=None):
        self.ext, self.torch, self.log, self.env, self.passes = ext or FakeExt(), torch or FakeTorch(), [], None, []
        self.forced_routes = {}                    # (N, H) -> True / False: the launches of the eager calls of that shape (the graph keeps what it captured)

    def kernel_times(self, fn, iters=5, warm=3):
        """The profiler: the launches per call of the pass, from what the pass ran (one launch or two by the route of the eager call or of the captured graph, the helper kernel when a helper ran)."""
        env = self.env
        eager0, replay0, fills0 = env.eager_calls, env.replay_calls, env.state.pos.fills + sum(b.fills for b in self.torch.buffers)
        for _ in range(warm + iters):
            fn()
        replay_only = env.replay_calls > replay0 and env.eager_calls == eager0
        fills = env.state.pos.fills + sum(b.fills for b in self.torch.buffers) - fills0
        counts = dict(COUNTS_ONE if (env.captured_single if replay_only else env.route_now()) else COUNTS_TWO)
        if fills:
            counts[FILL_K] = fills / float(iters + warm)
        self.passes.append(("replay" if replay_only else "eager", iters, warm))
        return {name: 1.0 + i for i, name in enumerate(counts)}, counts


def install(monkeypatch, w):
    """Put the fakes of ``w`` where the benchmark looks for torch, its profiler, the inputs of a shape and the extension."""
    FakeShapeEnv.world = w
    breakdown = types.ModuleType("bench_breakdown")
    breakdown.kernel_times = lambda fn, iters=5, warm=3: w.kernel_times(fn, iters, warm)
    monkeypatch.setitem(sys.modules, "bench_breakdown", breakdown)
    monkeypatch.setitem(sys.modules, "torch", w.torch)
    monkeypatch.setattr(bd, "ShapeEnv", FakeShapeEnv)
    monkeypatch.setattr(bd, "load_binding", lambda: w.ext)
    return w


@pytest.fixture()
def world(monkeypatch):
    yield install(monkeypatch, World())
    FakeShapeEnv.world = None


FAST = ["--trials", "5", "--calls", "64", "--cold-calls", "64", "--probe-calls", "64", "--profile-calls", "64"]


def run_main(tmp_path, *argv):
    """``bench_decode.main`` on the fakes with a record file: ``(exit code, record)``."""
    path = tmp_path / "record.json"
    path.unlink() if path.exists() else None
    code = bd.main(list(argv) + FAST + ["--json", str(path)])
    return code, json.loads(path.read_text())


# ---- the control: on before the first capture, restored whatever happens ----------------------------------------------------------------------------------
def test_the_control_is_switched_on_for_the_block_and_restored_after_it_and_after_an_error():
    ext = FakeExt()
    with bd.single_launch_control(ext, True):
        assert ext.force is True
    assert ext.force is False and ext.log == [("force", True), ("force", False)]
    with pytest.raises(SystemExit):
        with bd.single_launch_control(ext, True):
            assert ext.force is True
            raise SystemExit("a trial that launched the other path")
    assert ext.force is False
    on = FakeExt(force=True)                                                                      # a control that was on before comes back on
    with bd.single_launch_control(on, True):
        pass
    assert on.force is True
    off = FakeExt()
    with bd.single_launch_control(off, False):
        assert off.force is False
    with bd.single_launch_control(None, False):                                                  # a run without the option does not even need the extension
        pass
    assert off.log == []


def test_a_tree_without_the_control_cannot_be_forced():
    with pytest.raises(SystemExit) as e:
        with bd.single_launch_control(types.SimpleNamespace(), True):
            pass
    assert "--single-force" in str(e.value) and "decode_single_force" in str(e.value)


def test_a_forced_run_switches_the_control_on_before_every_capture_and_restores_it(world, tmp_path, capsys):
    code, rec = run_main(tmp_path, "--boundary", "148", "--single-force", "--cells", "N=4;H=12;regime=lockstep;", "N=99;H=12;regime=lockstep;", "N=395;H=6;regime=cold;")
    assert code == 0
    assert world.log == [("capture", 4, 12, True), ("capture", 99, 12, True), ("capture", 395, 6, True)]       # on at every capture, so every graph holds the single launch
    assert world.ext.force is False and world.ext.log == [("force", True), ("force", False)]
    assert rec["meta"]["single_force"] is True and all(c["args"]["single_force"] is True for c in rec["cells"])
    assert all(c["args"]["compute_nodes"] == ONE for c in rec["cells"])                                           # forced above the limit too: one launch in every cell
    assert capsys.readouterr().out.splitlines()[0].endswith("[boundary cells for 148 SMs; single-force]")


def test_a_run_that_fails_in_a_trial_restores_the_control(world, tmp_path):
    world.forced_routes[(99, 12)] = False                                                         # one shape launches the plan kernel although the run asserts one launch
    with pytest.raises(SystemExit) as e:
        bd.main(["--boundary", "148", "--single-force", "--expect-compute", "single-launch", "--cells", "N=4;H=12;regime=lockstep;", "N=99;H=12;regime=lockstep;"] + FAST
                + ["--json", str(tmp_path / "record.json")])
    assert "decode;N=99;H=12;regime=lockstep;" in str(e.value) and "trial 0" in str(e.value)
    assert world.ext.force is False and world.ext.log == [("force", True), ("force", False)]
    assert not (tmp_path / "record.json").exists()


def test_a_run_without_the_option_leaves_the_control_alone(world, tmp_path):
    code, rec = run_main(tmp_path, "--cells", "N=4;H=12;regime=lockstep;")
    assert code == 0 and world.ext.log == [] and world.log == [("capture", 4, 12, False)]
    assert rec["meta"]["single_force"] is False and rec["cells"][0]["args"]["single_force"] is False and rec["cells"][0]["args"]["expect_route"] is None


# ---- both sides of an A/B load the extension at the same point -------------------------------------------------------------------------------------------------
LOAD_ORDER_RUNS = [
    ("side A, the explicit two launches", ["--impl", "64", "--expect-compute", "two-launch", "--cells", "N=4;H=12;regime=lockstep;"]),
    ("side B, the default routing", ["--boundary", "148", "--expect-route", "spec", "--cells", "N=4;H=12;regime=lockstep;", "N=99;H=12;regime=lockstep;"]),
    ("side B, the control on", ["--boundary", "148", "--single-force", "--expect-route", "spec", "--cells", "N=99;H=12;regime=lockstep;"]),
    ("a run with none of the options of an A/B", ["--cells", "N=4;H=12;regime=lockstep;"])]


@pytest.mark.parametrize("what,argv", LOAD_ORDER_RUNS, ids=[r[0] for r in LOAD_ORDER_RUNS])
def test_every_run_loads_the_extension_once_before_its_first_shape_environment(world, tmp_path, monkeypatch, what, argv):
    """The two sides of an A/B have one history up to their first call: the extension, and its machine code with it, is loaded at the same point of both, before the first shape environment,
    whatever options a side runs with (side A used to load it at its first warm call, side B before the control was switched on). An ordering rule shows in no result, so the order is recorded."""
    events = []
    monkeypatch.setattr(bd, "load_binding", lambda: (events.append(("load",)), world.ext)[1])

    class Recording(FakeShapeEnv):
        def __init__(self, N, H, impl):
            events.append(("env", N, H))
            super(Recording, self).__init__(N, H, impl)

    monkeypatch.setattr(bd, "ShapeEnv", Recording)
    code, rec = run_main(tmp_path, *argv)
    assert code == 0
    assert events.count(("load",)) == 1 and events[0] == ("load",), (what, events)
    assert [e[0] for e in events[1:]] and set(e[0] for e in events[1:]) == {"env"}               # and the shapes were measured after it


# ---- the route specification, per cell -----------------------------------------------------------------------------------------------------------------------
def test_each_cell_is_asserted_against_the_route_that_the_specification_gives_for_it(world, tmp_path, capsys):
    cells = ["N=4;H=12;regime=lockstep;", "N=99;H=12;regime=lockstep;", "N=128;H=12;regime=staggered;", "N=395;H=6;regime=cold;", "N=1;H=12;regime=probe;pos=63;"]
    code, rec = run_main(tmp_path, "--boundary", "148", "--expect-route", "spec", "--cells", *cells)
    assert code == 0
    got = {c["key"]: c["args"] for c in rec["cells"]}
    assert {k: v["expect_compute"] for k, v in got.items()} == {
        "decode;N=1;H=12;regime=probe;pos=63;": ONE, "decode;N=4;H=12;regime=lockstep;": ONE, "decode;N=99;H=12;regime=lockstep;": TWO,
        "decode;N=128;H=12;regime=staggered;": TWO, "decode;N=395;H=6;regime=cold;": TWO}                       # the routed cells are one launch, the others two, in ONE process
    assert all(v["expect_route"] == "spec" and v["single_force"] is False for v in got.values())
    assert {k: v["compute_nodes"] for k, v in got.items()} == {k: v["expect_compute"] for k, v in got.items()}
    assert rec["meta"]["single_per_sm"] == 8 and rec["meta"]["boundary_sms"] == 148 and rec["meta"]["sm_count"] == 148 and rec["meta"]["cc"] == "10.0"
    assert world.ext.log == []                                                                    # no control: the extension was only asked
    assert "expect-route spec, 8 CTAs per SM" in capsys.readouterr().out


def test_a_forced_run_is_asserted_against_the_forced_specification(world, tmp_path):
    code, rec = run_main(tmp_path, "--boundary", "148", "--single-force", "--expect-route", "spec", "--cells", "N=99;H=12;regime=lockstep;", "N=395;H=6;regime=lockstep;", "N=4;H=12;regime=cold;")
    assert code == 0 and all(c["args"]["expect_compute"] == ONE and c["args"]["expect_route"] == "spec" and c["args"]["single_force"] is True for c in rec["cells"])
    code, rec = run_main(tmp_path, "--boundary", "148", "--impl", "64", "--single-force", "--expect-route", "spec", "--cells", "N=4;H=12;regime=lockstep;")
    assert code == 0 and rec["cells"][0]["args"]["expect_compute"] == TWO and rec["cells"][0]["args"]["impl"] == 64          # bit 64 stays two launches under the control


@pytest.mark.parametrize("flipped,launches_one", [((99, 12), True), ((4, 12), False)], ids=["a cell above the limit that launches one kernel", "a routed cell that launches two"])
def test_a_cell_that_runs_the_other_route_fails_its_own_assertion(world, tmp_path, flipped, launches_one):
    world.forced_routes[flipped] = launches_one       # the extension still answers as the specification does, so the check before the run passes and the per-trial assertion fires
    with pytest.raises(SystemExit) as e:
        run_main(tmp_path, "--boundary", "148", "--expect-route", "spec", "--cells", "N=4;H=12;regime=lockstep;", "N=99;H=12;regime=lockstep;")
    assert "decode;N=%d;H=%d;regime=lockstep;" % flipped in str(e.value) and "trial 0" in str(e.value)
    assert not (tmp_path / "record.json").exists()


def test_a_single_expectation_for_the_whole_run_dies_where_the_route_changes(world, tmp_path):
    """The reason for the per-cell specification: one mapping for a run that crosses the limit fails at the first cell on the other side."""
    with pytest.raises(SystemExit) as e:
        run_main(tmp_path, "--boundary", "148", "--expect-compute", "single-launch", "--cells", "N=4;H=12;regime=lockstep;", "N=99;H=12;regime=lockstep;")
    assert "decode;N=99;H=12;regime=lockstep;" in str(e.value)


def test_the_extension_is_asked_the_specifications_question_before_anything_is_measured(world, tmp_path):
    world.ext.per_sm = 9                                                                          # an extension whose limit is not the one of the tree's header
    with pytest.raises(SystemExit) as e:
        run_main(tmp_path, "--boundary", "148", "--expect-route", "spec", "--cells", "N=4;H=12;regime=lockstep;", "N=99;H=12;regime=lockstep;", "N=99;H=12;regime=cold;")
    message = str(e.value)
    assert "disagree" in message and "N=99 H=12: the extension says one launch, the specification two launches" in message and message.count("N=99") == 1      # one line per shape
    assert world.log == [] and world.ext.log == []                                                # nothing was captured, nothing was switched


def test_the_binding_check_covers_every_shape_and_names_the_ones_that_differ():
    specs = bd.campaign_cell_specs(8, 148)
    routes = bd.route_expectations(specs, 0, DEVICE_B200, 8, False)
    assert bd.route_binding_problems(FakeExt(), specs, 0, routes) == []
    for per_sm, shapes in ((9, {(36, 33), (99, 12), (198, 6)}), (7, {(35, 33), (98, 12), (197, 6)})):          # the shapes between the two limits
        problems = bd.route_binding_problems(FakeExt(per_sm=per_sm), specs, 0, routes)
        assert sorted(p.split(":")[0] for p in problems) == sorted("N=%d H=%d" % s for s in shapes), per_sm
    problems = bd.route_binding_problems(FakeExt(force=True), specs, 0, routes)                  # a control that is on in the extension although the run does not say so
    assert len(problems) == 11 and all("the extension says one launch" in p for p in problems)
    forced = bd.route_expectations(specs, 0, DEVICE_B200, 8, True)
    assert bd.route_binding_problems(FakeExt(force=True), specs, 0, forced) == []
    assert len(bd.route_binding_problems(FakeExt(), specs, 0, forced)) == 11 and all("the extension says two launches" in p for p in bd.route_binding_problems(FakeExt(), specs, 0, forced))
    assert bd.route_binding_problems(FakeExt(cc=H100), specs, 0, routes) != []                   # another device
    asked, probe = [], FakeExt()
    probe.decode_single_launch = lambda *a: asked.append(a) or True
    bd.route_binding_problems(probe, specs, 0, routes)
    assert len(asked) == 24 and {a[0] for a in asked} == {1} and {a[1] for a in asked} == {0}   # one question per shape, a decode call (mode 1), the resolved bits


def test_the_reference_row_engine_is_not_covered_by_the_specification():
    specs = bd.campaign_cell_specs(8, 148)[:3]
    routes = bd.route_expectations(specs, 0, DEVICE_B200, 8, False)
    problems = bd.route_binding_problems(FakeExt(resolve=lambda impl: impl | 1), specs, 0, routes)
    assert len(problems) == 1 and "reference" in problems[0]
    assert bd.route_binding_problems(FakeExt(), specs, 64, bd.route_expectations(specs, 64, DEVICE_B200, 8, False)) == []


# ---- the records ---------------------------------------------------------------------------------------------------------------------------------------
def test_a_bare_boundary_takes_the_sm_count_of_the_device_and_an_explicit_one_fixes_the_cells(world, tmp_path, monkeypatch):
    code, rec = run_main(tmp_path, "--boundary", "--cells", "N=99;H=12;regime=lockstep;")
    assert code == 0 and rec["meta"]["boundary_sms"] == 148 and rec["meta"]["sm_count"] == 148
    install(monkeypatch, World(ext=FakeExt(sms=132, cc=H100), torch=FakeTorch(sms=132, major=9, minor=0, name="NVIDIA H100")))
    code, rec = run_main(tmp_path, "--boundary", "--cells", "N=88;H=12;regime=lockstep;", "N=99;H=12;regime=lockstep;")                    # H100: the cells of its 132 SMs
    assert code == 0 and [c["key"] for c in rec["cells"]] == ["decode;N=88;H=12;regime=lockstep;"] and rec["meta"]["boundary_sms"] == 132 and rec["meta"]["sm_count"] == 132
    code, rec = run_main(tmp_path, "--boundary", "148", "--cells", "N=99;H=12;regime=lockstep;", "N=88;H=12;regime=lockstep;")               # the cells of 148 on a device of 132
    assert code == 0 and [c["key"] for c in rec["cells"]] == ["decode;N=99;H=12;regime=lockstep;"] and rec["meta"]["boundary_sms"] == 148 and rec["meta"]["sm_count"] == 132


def test_the_meta_carries_the_control_and_the_limit_of_the_measured_tree(tmp_path, monkeypatch):
    class FakeTorchVersion:
        __version__ = "2.14.0"
        version = types.SimpleNamespace(cuda="13.4")

    started, opts = "2026-11-10T10:00:00Z", dict(bd.DEFAULTS)
    old_style = types.SimpleNamespace(impl=0, regimes=list(bd.REGIMES), cells=None, label="x")          # the namespace that the older records were written with
    meta = bd.run_meta(old_style, started, opts, DEVICE_B200, FakeTorchVersion)
    assert (meta["single_force"], meta["single_per_sm"], meta["boundary_sms"], meta["graph_nodes"]) == (False, 8, None, False)
    a = parse("--single-force", "--boundary", "148", "--graph-nodes")
    meta = bd.run_meta(a, started, opts, DEVICE_B200, FakeTorchVersion)
    assert (meta["single_force"], meta["single_per_sm"], meta["boundary_sms"], meta["graph_nodes"]) == (True, 8, 148, True)
    assert bd.run_meta(parse("--boundary"), started, opts, DEVICE_H100, FakeTorchVersion)["boundary_sms"] == 132
    monkeypatch.delenv("CUDA_MODULE_LOADING", raising=False)
    assert meta["cuda_module_loading"] is None                                                                  # the module loading mode of the process, as the environment says it
    monkeypatch.setenv("CUDA_MODULE_LOADING", "EAGER")
    assert bd.run_meta(a, started, opts, DEVICE_B200, FakeTorchVersion)["cuda_module_loading"] == "EAGER"
    monkeypatch.delenv("CUDA_MODULE_LOADING")
    for field in br.REQUIRED_META + ("cc", "sm_count", "impl", "env_impl", "trials", "sample_stat"):
        assert field in meta, field
    monkeypatch.setattr(bd, "ROOT", tree_with_header(tmp_path, "constexpr int64_t DECODE_SINGLE_MAX_CTAS_PER_SM = 12;\n"))
    assert bd.run_meta(a, started, opts, DEVICE_B200, FakeTorchVersion)["single_per_sm"] == 12                 # the limit of the tree that is measured, not of the tool's
    monkeypatch.setattr(bd, "ROOT", tree_with_header(tmp_path, None, "old"))
    assert bd.run_meta(a, started, opts, DEVICE_B200, FakeTorchVersion)["single_per_sm"] is None


def test_a_campaign_run_writes_a_valid_record_of_every_cell_with_the_new_fields(world, tmp_path, capsys):
    code, rec = run_main(tmp_path, "--boundary", "148", "--expect-route", "spec")
    assert code == 0 and rec["meta"]["kind"] == "decode-wall" and len(rec["cells"]) == 92
    assert [c["key"] for c in rec["cells"]] == [s["key"] for s in bd.campaign_cell_specs(8, 148)]
    assert sum(c["args"]["expect_compute"] == ONE for c in rec["cells"]) == 59 and sum(c["args"]["expect_compute"] == TWO for c in rec["cells"]) == 33
    assert all(c["args"]["compute_nodes"] == c["args"]["expect_compute"] and c["args"]["expect_route"] == "spec" and c["args"]["single_force"] is False for c in rec["cells"])
    assert all(s["nodes"]["compute"] == c["args"]["compute_nodes"] and "graph_nodes" not in s for c in rec["cells"] for s in c["samples"])
    expected = {s["key"]: bd.CANONICAL_COMPUTE[dp.expected_route(0, s["N"], s["H"], 148, B200, 8)] for s in bd.campaign_cell_specs(8, 148)}      # the specification, not the record
    assert [p for c in rec["cells"] for p in br.cell_node_problems(c, expected[c["key"]])] == []
    broken = json.loads(json.dumps(rec["cells"][0]))                                                # a trial of a routed cell that was measured on the other path
    broken["samples"][3]["nodes"]["compute"] = dict(TWO)
    assert br.cell_node_problems(broken, expected[broken["key"]])
    rec["meta"].update(commit="abc1234", dirty=False)                                              # the identity of the tree that runs the test is not what is checked here
    assert br.validate_record(dict(rec, _path="record.json"), "candidate", require_clean=False) == []
    first = capsys.readouterr().out.splitlines()[0]
    assert first.startswith("device NVIDIA B200  impl=0  92 cell(s)  trials=5") and first.endswith("[boundary cells for 148 SMs; expect-route spec, 8 CTAs per SM]")


def test_a_default_run_prints_and_records_what_it_always_did_plus_the_two_new_cell_fields(world, tmp_path, capsys):
    code, rec = run_main(tmp_path, "--cells", "N=1;H=12;regime=lockstep;", "N=1;H=12;regime=probe;pos=0;")
    assert code == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "device NVIDIA B200  impl=0  2 cell(s)  trials=5  calls/trial: lockstep 64, cold 64, probe 64"             # no brackets: no campaign option was given
    cell = rec["cells"][0]
    assert sorted(cell["args"]) == sorted(["N", "H", "regime", "pos", "impl", "calls_per_trial", "profile_calls", "profile_cycles", "profile_start_phase", "calls_scale", "kernels",
                                           "compute_nodes", "compute_kernels", "aux_nodes", "expect_compute", "single_force", "expect_route"])
    assert cell["args"]["expect_compute"] is None and all(sorted(s["nodes"]) == ["aux", "compute"] for s in cell["samples"])
    assert world.passes == [("eager", 64, 1)] * 10                                                 # two cells, five trials, one eager profiled pass each: no replay is profiled


# ---- one graph replay per trial: the main flow --------------------------------------------------------------------------------------------------------------
def test_a_graph_replay_is_profiled_in_every_trial_when_asked_for_and_never_otherwise(world, tmp_path):
    cells = ["N=4;H=12;regime=lockstep;", "N=99;H=12;regime=staggered;"]
    code, plain = run_main(tmp_path, "--boundary", "148", "--expect-route", "spec", "--cells", *cells)
    assert code == 0 and world.passes == [("eager", 64, 1)] * 10
    world.passes.clear()
    code, rec = run_main(tmp_path, "--boundary", "148", "--expect-route", "spec", "--graph-nodes", "--cells", *cells)
    assert code == 0 and world.passes == [("eager", 64, 1), ("replay", 1, 0)] * 10                  # one replay of the graph after the eager pass of every trial, no warm-up, nothing timed
    for cell, mapping in zip(rec["cells"], (ONE, TWO)):
        assert cell["args"]["compute_nodes"] == mapping
        assert all(s["graph_nodes"] == {"compute": mapping, "aux": {}} and s["nodes"] == {"compute": mapping, "aux": {}} for s in cell["samples"])
        assert br.cell_node_problems(cell, mapping) == []                                              # the validator of the eager mapping is not disturbed by the second field
    assert rec["meta"]["graph_nodes"] is True and plain["meta"]["graph_nodes"] is False
    assert [c["args"] for c in rec["cells"]] == [c["args"] for c in plain["cells"]]                  # no new cell argument: a side with the option and a side without it compare
    assert [[s["gpu_graph_us"] > 0 for s in c["samples"]] for c in rec["cells"]] == [[True] * 5] * 2


def test_the_graph_replay_adds_one_replay_per_trial_and_nothing_else(world, tmp_path):
    run_main(tmp_path, "--cells", "N=4;H=12;regime=lockstep;")
    plain = (world.env.eager_calls, world.env.replay_calls)
    run_main(tmp_path, "--cells", "N=4;H=12;regime=lockstep;", "--graph-nodes")
    assert (world.env.eager_calls, world.env.replay_calls) == (plain[0], plain[1] + 5)


def test_a_graph_captured_before_the_control_was_on_is_found_only_by_profiling_a_replay(world):
    """The order that a run must not have: the shape is captured with the control off and measured with it on. The eager passes launch the single launch that the run asserts, the graph,
    which the gating metric replays, holds the two launches; only a profiled replay says so."""
    def late(N, H, impl):
        env = FakeShapeEnv(N, H, impl)
        capture = env.capture
        env.capture = lambda: (capture(), world.ext.decode_single_force(True))
        return env
    specs = [s for s in bd.campaign_cell_specs(8, 148) if (s["N"], s["H"], s["regime"]) == (99, 12, "lockstep")]
    opts = dict(bd.DEFAULTS, trials=5, calls=64, cold_calls=64, probe_calls=64, profile_calls=64, calls_scale=1)
    cells = bd.run_cells(specs, parse("--single-force"), opts, DEVICE_B200, world.torch, expect_compute=ONE, env_factory=late)            # the eager passes alone are satisfied
    assert cells[0]["args"]["compute_nodes"] == ONE
    world.ext.decode_single_force(False)
    with pytest.raises(SystemExit) as e:
        bd.run_cells(specs, parse("--single-force", "--graph-nodes"), opts, DEVICE_B200, world.torch, expect_compute=ONE, env_factory=late)
    assert "trial 0" in str(e.value) and "graph replay" in str(e.value) and "plan_kernel" in str(e.value)


# ---- one graph replay per trial: measure_cell with scripted profiler passes ------------------------------------------------------------------------------------
class ScriptEnv(object):
    def __init__(self):
        self.torch, self.impl, self.eager_calls, self.replay_calls, self.state = FakeTorch(), 0, 0, 0, types.SimpleNamespace(pos=FakeBuffer())

    def set_positions(self, regime):
        pass

    def eager(self, i):
        self.eager_calls += 1

    def replay(self, i):
        self.replay_calls += 1


def measure(monkeypatch, eager, replay, expect=None, graph=True, regime="lockstep", trials=5):
    """One cell of ``measure_cell`` whose n-th profiled eager pass shows the launch counts ``eager[n]`` and whose n-th profiled replay shows ``replay[n]`` (the last one repeats):
    ``(cell, env, log of the passes, eviction buffer)``."""
    env, seen, log = ScriptEnv(), {"eager": 0, "replay": 0}, []

    def kernel_times(fn, iters=5, warm=3):
        before = (env.eager_calls, env.replay_calls)
        for _ in range(warm + iters):
            fn()
        kind = "replay" if env.replay_calls > before[1] and env.eager_calls == before[0] else "eager"
        seq = replay if kind == "replay" else eager
        counts = dict(seq[min(seen[kind], len(seq) - 1)])
        seen[kind] += 1
        log.append((kind, iters, warm))
        return {name: 1.0 for name in counts}, counts
    breakdown = types.ModuleType("bench_breakdown")
    breakdown.kernel_times = kernel_times
    monkeypatch.setitem(sys.modules, "bench_breakdown", breakdown)
    opts = dict(bd.DEFAULTS, trials=trials, calls=64, cold_calls=64, probe_calls=64, profile_calls=64, calls_scale=1)
    spec = {"N": 4, "H": 12, "regime": regime, "pos": 5 if regime == "probe" else None, "key": "decode;N=4;H=12;regime=%s;" % regime}
    evict = FakeBuffer()
    return bd.measure_cell(spec, env, opts, {}, expect, evict, graph_nodes=graph), env, log, evict


@pytest.mark.parametrize("bad_trial", [0, 2, 4])
def test_a_replay_on_the_other_route_than_the_eager_pass_fails_in_its_own_trial(monkeypatch, bad_trial):
    replay = [COUNTS_TWO] * 5
    replay[bad_trial] = COUNTS_ONE
    for expect in (TWO, None):
        with pytest.raises(SystemExit) as e:
            measure(monkeypatch, [COUNTS_TWO], replay, expect=expect)
        assert "trial %d" % bad_trial in str(e.value) and "graph replay" in str(e.value), expect


GRAPH_DRIFTS = {"the plan kernel missing": COUNTS_ONE, "a duplicated launch": {PLAN_K: 1.0, ROWS_K: 2.0}, "a fractional count": {PLAN_K: 1.0, ROWS_K: 0.984},
                "another kernel as well": dict(COUNTS_TWO, **{"void cfbi::finalize_pos_kernel(int*, int)": 1.0}), "the row engine of the reference": {PLAN_K: 1.0, "void cfbi::ref::rows_kernel(cfbi::Params)": 1.0}}


@pytest.mark.parametrize("what", sorted(GRAPH_DRIFTS))
def test_a_replay_that_launches_anything_but_the_eager_mapping_is_rejected(monkeypatch, what):
    with pytest.raises(SystemExit) as e:
        measure(monkeypatch, [COUNTS_TWO], [GRAPH_DRIFTS[what]], expect=TWO)
    assert "trial 0" in str(e.value) and "graph replay" in str(e.value), what


def test_a_replay_is_not_asked_when_the_option_is_off_and_its_auxiliary_nodes_are_recorded_not_asserted(monkeypatch):
    cell, env, log, _ = measure(monkeypatch, [COUNTS_TWO], [COUNTS_ONE], expect=TWO, graph=False)             # the replays would be wrong, nobody looks at them
    assert log == [("eager", 64, 1)] * 5 and env.replay_calls == 50 + 5 * 64 and all("graph_nodes" not in s for s in cell["samples"])
    cell, env, log, _ = measure(monkeypatch, [COUNTS_TWO], [dict(COUNTS_TWO, **{"Memset (Device)": 1.0})], expect=TWO)
    assert log == [("eager", 64, 1), ("replay", 1, 0)] * 5 and env.replay_calls == 50 + 5 * 64 + 5
    assert all(s["graph_nodes"] == {"compute": TWO, "aux": {"Memset (Device)": 1}} and s["nodes"] == {"compute": TWO, "aux": {}} for s in cell["samples"])
    assert cell["args"]["aux_nodes"] == {} and br.cell_node_problems(cell, TWO) == []


def test_the_eager_pass_is_asserted_first_and_the_replay_follows_it(monkeypatch):
    with pytest.raises(SystemExit) as e:                                                          # the eager pass is on the other path, the replay is what is asked for
        measure(monkeypatch, [COUNTS_ONE], [COUNTS_TWO], expect=TWO)
    assert "trial 0" in str(e.value) and "graph replay" not in str(e.value)


@pytest.mark.parametrize("regime", ["cold", "probe"])
def test_the_helper_of_a_regime_runs_before_the_profiled_replay_too(monkeypatch, regime):
    """The eviction write (cold) or the position rewrite (probe) is enqueued before the calls of a profiled pass; the replay pass does the same, so the auxiliary node of the replay is the eager one."""
    def helper_fills(graph):
        cell, env, _, evict = measure(monkeypatch, [dict(COUNTS_TWO, **{FILL_K: 1.0})], [dict(COUNTS_TWO, **{FILL_K: 1.0})], graph=graph, regime=regime)
        return evict.fills if regime == "cold" else env.state.pos.fills
    assert helper_fills(True) == helper_fills(False) + 5                                           # one helper call per profiled replay, one replay per trial


def test_graph_problem_accepts_the_eager_mapping_and_names_every_other():
    eager = {"plan_kernel": 1.0, "rows_kernel": 1.0}
    assert bd.graph_problem(dict(eager), {}, dict(eager), None) is None
    assert bd.graph_problem(dict(eager), {}, dict(eager), TWO) is None
    assert bd.graph_problem(dict(eager), {"memset": 1.0}, dict(eager), TWO) is None                # the auxiliary nodes of the replay are recorded, not asserted
    assert "the graph replay launches" in bd.graph_problem({"rows_kernel": 1.0}, {}, dict(eager), None)
    assert "the graph replay launches" in bd.graph_problem(dict(eager, extra=1.0), {}, dict(eager), None)
    assert "expected" in bd.graph_problem({"plan_kernel": 1.0, "rows_kernel": 2.0}, {}, dict(eager), TWO)                               # a duplicated launch
    assert "whole number" in bd.graph_problem({"plan_kernel": 1.0, "rows_kernel": 0.984}, {}, dict(eager), None)                       # a fractional one
    assert "expected" in bd.graph_problem(dict(eager), {}, dict(eager), ONE)                                                              # not the asserted mapping


# ---- the route order: the cells that both sides run with the same kernels first ------------------------------------------------------------------------------
def shape_runs(specs):
    runs = []
    for s in specs:
        if not runs or runs[-1] != (s["N"], s["H"]):
            runs.append((s["N"], s["H"]))
    return runs


def test_the_route_order_puts_the_cells_of_the_two_launch_default_route_first_and_keeps_every_other_order():
    specs = bd.campaign_cell_specs(8, 148)
    plain = shape_runs(specs)
    ordered = bd.route_ordered(specs, DEVICE_B200, 8)
    assert sorted(s["key"] for s in ordered) == sorted(s["key"] for s in specs) and len(ordered) == 92                      # the same cells
    spec = bd.policy_spec()
    route = lambda s: spec.expected_route(0, s["N"], s["H"], 148, 100, 8, False)
    classes = [route(s) == spec.SINGLE_LAUNCH for s in ordered]
    assert classes == sorted(classes) and 0 < sum(classes) < len(classes)                                            # the two-launch cells, then the single-launch cells
    two, single = [s for s in specs if not route(s) == spec.SINGLE_LAUNCH], [s for s in specs if route(s) == spec.SINGLE_LAUNCH]
    assert [s["key"] for s in ordered] == [s["key"] for s in two + single]                                           # a stable sort: the order inside a class is the plain one
    assert plain.index((35, 33)) + 1 == plain.index((36, 33)) and shape_runs(ordered).index((36, 33)) < shape_runs(ordered).index((35, 33))      # N = 36 no longer follows N = 35
    assert [k for k in shape_runs(ordered) if k[0] * k[1] > 8 * 148] == [k for k in plain if k[0] * k[1] > 8 * 148] != []                       # the cells above the limit keep their order
    assert [s["key"] for s in bd.route_ordered(specs, DEVICE_H100, 8)] == [s["key"] for s in specs]                      # on a device that the default rule does not route nothing moves


def test_a_run_with_the_route_order_measures_in_that_order_and_records_it(world, tmp_path):
    cells = ["N=35;H=33;regime=lockstep;", "N=36;H=33;regime=lockstep;", "N=1;H=12;regime=lockstep;", "N=99;H=12;regime=lockstep;"]
    code, plain = run_main(tmp_path, "--boundary", "148", "--cells", *cells)
    assert code == 0 and plain["meta"]["route_order"] is False
    code, ordered = run_main(tmp_path, "--boundary", "148", "--route-order", "--cells", *cells)
    assert code == 0 and ordered["meta"]["route_order"] is True
    keys = [c["key"] for c in ordered["cells"]]
    assert sorted(keys) == sorted(c["key"] for c in plain["cells"])
    assert [k for k in keys if "N=36;" in k or "N=99;" in k] == keys[:2] and all("N=1;" in k or "N=35;" in k for k in keys[2:])      # the two-launch routes first, the single-launch ones after them
    side_a = [c["key"] for c in run_main(tmp_path, "--boundary", "148", "--route-order", "--impl", "64", "--cells", *cells)[1]["cells"]]
    assert side_a == keys                                                                                           # the same order on side A (bit 64) as on side B


def test_the_route_order_needs_the_limit_of_the_tree_and_is_listed_in_the_header(world, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bd, "ROOT", tree_with_header(tmp_path, None, "old"))
    with pytest.raises(SystemExit) as e:
        bd.main(["--route-order", "--boundary", "148", "--json", str(tmp_path / "x.json")] + FAST)
    assert e.value.code == 2
    assert "route-order" in "; ".join(bd.run_notes(parse("--boundary", "148", "--route-order"), 8, DEVICE_B200))
