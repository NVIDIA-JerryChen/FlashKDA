"""Self-tests of the benchmark tooling (benchmarks/bench_records.py, compare_bench.py, compare_binaries.py) and of the corpus V definitions.

Plain Python: no GPU and no torch are needed (run them without the package conftest, e.g. ``pytest --noconftest tests/cfbi/test_bench_tools.py``,
where torch is not installed). The negative cases show that the gates can fail: a wrong, dirty or unstable baseline is refused, a 1 % slower
candidate fails the 0.5 % protection gate, and a build with one more register or a spill is reported as different.
"""
import argparse
import ast
import concurrent.futures
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "benchmarks", _ROOT / "tests" / "cfbi"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bench_decode as bd  # noqa: E402
import bench_records as br  # noqa: E402
import codegen_gate as cg  # noqa: E402
import compare_bench  # noqa: E402
import compare_binaries as cb  # noqa: E402
import varlen_corpus as vc  # noqa: E402

BASE_COMMIT = "2dcb4f89714eb4dcc67a69c6e421ee2c60e1cb1a"


def rec(commit, cells, ts, gpu="NVIDIA B200", sm=148, host="node1", dirty=False, path="mem"):
    """A record whose cells are ``{key: [total-us repeats]}``; k2_stage_us = 0.7 * value, k1 = 0.3 * value."""
    out = []
    for key, vals in cells.items():
        out.append({"kind": "prefill", "key": key, "args": {},
                    "samples": [br.derive({"k1_kernel": 0.3 * v, "k2_kernel": 0.7 * v}) for v in vals]})
    meta = {"commit": commit, "dirty": dirty, "gpu": gpu, "sm_count": sm, "host": host, "timestamp": ts}
    return {"schema": br.SCHEMA, "meta": meta, "cells": out, "_path": path}


def interleaved(base_vals, cand_vals, base_commit=BASE_COMMIT, cand_commit="abc1234", **kw):
    """Three baseline and three candidate records with alternating timestamps; one cell ``c`` with the given in-process repeats."""
    base = [rec(base_commit, {"c": base_vals}, "2026-11-05T10:0%d:00Z" % (2 * i), **kw) for i in range(3)]
    cand = [rec(cand_commit, {"c": cand_vals}, "2026-11-05T10:0%d:00Z" % (2 * i + 1), **kw) for i in range(3)]
    return base, cand


def write_records(directory, prefix, recs):
    """Write ``recs`` as the record files ``<prefix>0.json``, ``<prefix>1.json``, ... in ``directory``; returns their paths."""
    paths = []
    for i, r in enumerate(recs):
        p = os.path.join(directory, "%s%d.json" % (prefix, i))
        br.write_record(p, r["meta"], r["cells"])
        paths.append(p)
    return paths


# ---- corpus V --------------------------------------------------------------------------------------------------------------------------------

def test_corpus_entries_match_the_recorded_definitions():
    c = vc.CORPUS_V
    assert c["V1"].lens == [8192] * 8
    assert c["V2"].lens == [8192] + [1024] * 7 and c["V2r"].lens == [1024] * 7 + [8192]
    assert c["V3"].lens == [8192] * 4 + [1024] * 4
    assert c["V4"].lens == [1536, 2368, 5312, 2432, 8192, 5632, 8192, 4928]
    assert c["V5"].lens == [7488, 2560, 3328, 6848, 2368, 4480, 2240, 4224]
    assert c["V6"].lens == [32768] + [1024] * 7
    assert len(c["V7a"].lens) == 1 and len(c["V7b"].lens) == 2 and len(c["V7c"].lens) == 64
    assert len(c["V7d"].lens) == 16 and c["V7d"].H == 32
    assert all(n > 0 and n % vc.CHUNK == 0 for e in c.values() for n in e.lens)
    assert min(c["V7c"].lens) >= 1024 and max(c["V7c"].lens) <= 8192


def test_corpus_spec_parsing():
    assert vc.parse_spec("V2")[1:] == ([8192] + [1024] * 7, 6)
    assert vc.parse_spec("V1@12")[2] == 12
    assert vc.parse_spec("8192+7*1024@6")[1:] == ([8192] + [1024] * 7, 6)
    assert vc.cu_seqlens_of([64, 128, 64]) == [0, 64, 192, 256]
    for bad in ("8192", "V9", "8192+0@6", "8192++64@6"):
        try:
            vc.parse_spec(bad)
        except ValueError:
            continue
        raise AssertionError("accepted %r" % bad)


# ---- kernel names and stages -----------------------------------------------------------------------------------------------------------------

def test_kernel_names_and_stages():
    assert br.kernel_base("void cfbi::k2_kernel<16>(cfbi::Params)") == "k2_kernel"
    assert br.kernel_base("void cfbi::ref::k2_kernel(cfbi::Params)") == "ref::k2_kernel"
    assert br.kernel_stage("k1_kernel") == "k1" and br.kernel_stage("ref::k1_kernel") == "k1"
    for k2 in ("k2_kernel", "k2_kernel_mixed", "k2u_kernel", "k2_dyn_schedule_kernel", "Memset (Device)"):
        assert br.kernel_stage(k2) == "k2", k2
    for other in ("plan_kernel", "rows_kernel", "finalize_pos_kernel"):
        assert br.kernel_stage(other) == "other", other
    d = br.derive({"k1_kernel": 10.0, "k2_dyn_kernel": 20.0, "k2_dyn_schedule_kernel": 2.0, "plan_kernel": 1.0})
    assert d["total_us"] == 33.0 and d["k1_us"] == 10.0 and d["k2_stage_us"] == 22.0 and d["k2_main_us"] == 20.0


def test_make_cell_folds_template_instances():
    cell = br.make_cell("prefill", "k", {}, [{"void cfbi::k2_kernel<8>(cfbi::Params)": 5.0, "void cfbi::k2_kernel<16>(cfbi::Params)": 7.0}])
    assert cell["samples"][0]["kernels"] == {"k2_kernel": 12.0} and cell["samples"][0]["k2_stage_us"] == 12.0


# ---- baseline protection: the comparison refuses a misused baseline --------------------------------------------------------------------------

def test_valid_comparison_passes_and_reports_the_delta():
    base, cand = interleaved([100.0, 100.1, 99.9], [100.2, 100.3, 100.1])
    res = br.compare(base, cand, metric="k2_stage_us", tol_pct=1.0, expect_base_commit="2dcb4f8")
    assert res["verdict"] == "PASS", res["problems"]
    assert abs(res["rows"][0]["delta_pct"] - 0.2) < 0.05


def test_wrong_baseline_commit_is_refused():
    base, cand = interleaved([100.0, 100.0, 100.0], [100.0, 100.0, 100.0], base_commit="deadbeef" + "0" * 32)
    res = br.compare(base, cand, expect_base_commit="2dcb4f8")
    assert res["verdict"] == "INVALID" and any("not the expected" in p for p in res["problems"])


def test_dirty_baseline_is_refused():
    base, cand = interleaved([100.0, 100.0, 100.0], [100.0, 100.0, 100.0])
    base[1]["meta"]["dirty"] = True
    res = br.compare(base, cand, expect_base_commit="2dcb4f8")
    assert res["verdict"] == "INVALID" and any("dirty" in p for p in res["problems"])


def test_missing_gpu_or_sm_metadata_is_refused():
    for field in ("gpu", "sm_count", "commit", "dirty"):
        base, cand = interleaved([100.0, 100.0, 100.0], [100.0, 100.0, 100.0])
        del base[0]["meta"][field]
        res = br.compare(base, cand, expect_base_commit="2dcb4f8")
        assert res["verdict"] == "INVALID" and any(field in p for p in res["problems"]), field


def test_wrong_schema_is_refused():
    base, cand = interleaved([100.0, 100.0, 100.0], [100.0, 100.0, 100.0])
    base[0]["schema"] = "something-else"
    assert br.compare(base, cand)["verdict"] == "INVALID"


def test_unstable_baseline_cell_is_not_accepted_as_a_gate_baseline():
    base, cand = interleaved([100.0, 100.5, 100.9], [100.0, 100.0, 100.0])  # spread 0.9 % > 0.3 %
    res = br.compare(base, cand, expect_base_commit="2dcb4f8")
    assert res["rows"][0]["status"] == "UNSTABLE_BASELINE" and res["verdict"] == "INVALID"


def test_unstable_baseline_worst_policy_references_the_fastest_baseline_repeat():
    base, cand = interleaved([100.0, 100.5, 100.9], [100.4, 100.4, 100.4])  # baseline spread 0.9 %, median 100.5
    assert br.compare(base, cand, baseline_unstable="invalid")["verdict"] == "INVALID"
    res = br.compare(base, cand, tol_pct=1.0, mode="regress", baseline_unstable="worst")
    row = res["rows"][0]
    assert res["verdict"] == "PASS" and row["base_reference"] < row["base_median"] and "fastest repeat" in row["note"]
    # the most adverse pairing can fail where the medians would pass: +0.90 % against the baseline median, +1.40 % against its fastest repeat
    base, cand = interleaved([100.0, 100.5, 100.9], [101.4, 101.4, 101.4])
    res = br.compare(base, cand, tol_pct=1.0, mode="regress", baseline_unstable="worst")
    assert res["verdict"] == "FAIL" and res["rows"][0]["status"] == "FAIL"
    # both sides unstable: baseline at its fastest, candidate at its slowest repeat
    base, cand = interleaved([100.0, 100.5, 100.9], [100.0, 100.5, 100.9])
    row = br.compare(base, cand, tol_pct=0.5, mode="regress", baseline_unstable="worst")["rows"][0]
    assert row["cand_judged"] > row["base_reference"] and row["status"] == "FAIL"


def test_unstable_candidate_cell_is_judged_on_its_worst_repeat():
    base, cand = interleaved([100.0, 100.0, 100.0], [100.0, 100.0, 100.8])  # candidate spread 0.8 %: worst repeat 100.8, median 100.0
    res = br.compare(base, cand, tol_pct=0.5, mode="within")
    row = res["rows"][0]
    assert row["cand_judged"] > row["cand_median"] and row["status"] == "FAIL" and "worst repeat" in row["note"]


def test_mixed_hardware_not_interleaved_and_too_few_repeats_are_invalid():
    base, cand = interleaved([100.0, 100.0, 100.0], [100.0, 100.0, 100.0])
    cand[0]["meta"]["sm_count"] = 132
    assert any("different GPU" in p for p in br.compare(base, cand)["problems"])
    base, cand = interleaved([100.0, 100.0, 100.0], [100.0, 100.0, 100.0])
    cand[1]["meta"]["host"] = "node2"
    assert any("different GPU" in p for p in br.compare(base, cand)["problems"])
    base, cand = interleaved([100.0, 100.0, 100.0], [100.0, 100.0, 100.0])
    for i, r in enumerate(base):
        r["meta"]["timestamp"] = "2026-11-05T09:0%d:00Z" % i  # all baseline records before all candidate records
    res = br.compare(base, cand)
    assert any("not interleaved" in p for p in res["problems"]) and res["verdict"] == "INVALID"
    assert br.compare(base, cand, require_interleaved=False)["verdict"] == "PASS"
    base, cand = interleaved([100.0, 100.0, 100.0], [100.0, 100.0, 100.0])
    assert any("at least 3" in p for p in br.compare(base[:2], cand)["problems"])


def test_missing_candidate_cell_is_invalid():
    base, cand = interleaved([100.0, 100.0, 100.0], [100.0, 100.0, 100.0])
    for r in cand:
        r["cells"][0]["key"] = "other"
    res = br.compare(base, cand)
    assert res["verdict"] == "INVALID" and any("missing in the candidate" in p for p in res["problems"])


# ---- the gates can fail -----------------------------------------------------------------------------------------------------------------------

def test_one_percent_slower_candidate_fails_the_half_percent_protection_gate():
    base, cand = interleaved([100.0, 100.1, 99.9], [101.0, 101.1, 100.9])
    res = br.compare(base, cand, tol_pct=0.5, mode="within", expect_base_commit="2dcb4f8")
    assert res["verdict"] == "FAIL" and res["rows"][0]["status"] == "FAIL"


def test_candidate_within_the_tolerance_passes_and_regress_mode_ignores_speedups():
    base, cand = interleaved([100.0, 100.1, 99.9], [100.4, 100.5, 100.3])
    assert br.compare(base, cand, tol_pct=0.5, mode="within")["verdict"] == "PASS"
    base, cand = interleaved([100.0, 100.1, 99.9], [90.0, 90.1, 89.9])  # 10 % faster
    assert br.compare(base, cand, tol_pct=1.0, mode="regress")["verdict"] == "PASS"
    assert br.compare(base, cand, tol_pct=1.0, mode="within")["verdict"] == "FAIL"


def test_no_regression_gate_one_percent_tolerance():
    base, cand = interleaved([100.0, 100.1, 99.9], [100.9, 101.0, 100.8])
    assert br.compare(base, cand, tol_pct=1.0, mode="regress")["verdict"] == "PASS"
    base, cand = interleaved([100.0, 100.1, 99.9], [101.5, 101.6, 101.4])
    assert br.compare(base, cand, tol_pct=1.0, mode="regress")["verdict"] == "FAIL"


def test_cli_exit_codes_and_json_output():
    d = tempfile.mkdtemp()
    base, cand = interleaved([100.0, 100.1, 99.9], [101.0, 101.1, 100.9])
    base_paths, cand_paths = write_records(d, "base", base), write_records(d, "cand", cand)
    common = ["--base"] + base_paths + ["--cand"] + cand_paths + ["--expect-base-commit", "2dcb4f8", "--metric", "k2_stage_us"]
    out = os.path.join(d, "res.json")
    assert compare_bench.main(common + ["--tol", "0.5", "--mode", "within", "--json", out]) == 1
    assert json.load(open(out))[0]["verdict"] == "FAIL"
    assert compare_bench.main(common + ["--tol", "2.0", "--mode", "within"]) == 0
    assert compare_bench.main(common + ["--expect-base-commit", "0000000"]) == 2  # the last --expect-base-commit wins
    assert compare_bench.main(["--base", os.path.join(d, "missing.json"), "--cand", cand_paths[0]]) == 2


def test_summary_reports_median_spread_and_flags_unstable_cells():
    recs = [rec(BASE_COMMIT, {"stable": [100.0, 100.1], "noisy": [100.0, 101.0]}, "2026-11-05T10:0%d:00Z" % i) for i in range(3)]
    rows = br.summarize(recs, metrics=("k2_stage_us",))
    by = {r["key"]: r for r in rows}
    assert by["stable"]["stable"] and by["stable"]["n"] == 6 and abs(by["stable"]["median"] - 70.0) < 0.1
    assert not by["noisy"]["stable"] and by["noisy"]["spread_pct"] > 0.3
    text = br.format_summary(rows)
    assert "UNSTABLE" in text and "stable" in text


def test_summary_cli_exit_codes():
    d = tempfile.mkdtemp()
    for name, cells, expect in (("ok", {"c": [100.0, 100.1]}, 0), ("noisy", {"c": [100.0, 101.0]}, 1)):
        paths = write_records(d, name, [rec(BASE_COMMIT, cells, "2026-11-05T10:0%d:00Z" % i) for i in range(3)])
        assert compare_bench.main(["--summary"] + paths + ["--metric", "k2_stage_us"]) == expect, name
        assert compare_bench.main(["--summary"] + paths + ["--expect-base-commit", "0000000"]) == 2
    assert compare_bench.main([]) == 2


def test_summary_covers_k1_and_total_not_only_the_k2_stage():
    """A cell whose K2 stage is steady but whose K1 time is noisy must be flagged by the default summary (a K2-only summary hides it)."""
    d = tempfile.mkdtemp()
    recs = []
    for i in range(3):
        samples = [br.derive({"k1_kernel": k1, "k2_kernel": k2}) for k1, k2 in ((30.0, 70.0), (30.3, 70.07))]  # K1 spread 1 %, K2 spread 0.1 %
        meta = {"commit": BASE_COMMIT, "dirty": False, "gpu": "NVIDIA B200", "sm_count": 148, "host": "node1", "timestamp": "2026-11-05T10:0%d:00Z" % i}
        recs.append({"schema": br.SCHEMA, "meta": meta, "cells": [{"kind": "prefill", "key": "c", "args": {}, "samples": samples}], "_path": "mem"})
    paths = write_records(d, "k1noisy", recs)
    assert compare_bench.main(["--summary"] + paths) == 1
    assert compare_bench.main(["--summary"] + paths + ["--metric", "k1_us"]) == 1
    assert compare_bench.main(["--summary"] + paths + ["--metric", "k2_stage_us"]) == 0
    flagged = {(r["key"], r["metric"]) for r in br.summarize(recs) if not r["stable"]}
    assert ("c", "k1_us") in flagged and ("c", "k2_stage_us") not in flagged


# ---- unusable input is INVALID (exit 2), never a performance FAIL (exit 1) --------------------------------------------------------------------

def test_malformed_records_are_invalid_and_never_raise():
    breakers = {
        "no metadata": lambda r: r.pop("meta"),
        "metadata is not an object": lambda r: r.update(meta=[1, 2]),
        "no cells": lambda r: r.pop("cells"),
        "empty cell list": lambda r: r.update(cells=[]),
        "cell without samples": lambda r: r["cells"][0].update(samples=[]),
        "cell without a key": lambda r: r["cells"][0].pop("key"),
        "sample without a derived metric": lambda r: r["cells"][0]["samples"][0].pop("total_us"),
        "sample without kernels": lambda r: r["cells"][0]["samples"][0].pop("kernels"),
    }
    for name, breaker in breakers.items():
        for side in (0, 1):
            base, cand = interleaved([100.0, 100.0, 100.0], [100.0, 100.0, 100.0])
            breaker((base if side == 0 else cand)[1])
            res = br.compare(base, cand)
            assert res["verdict"] == "INVALID" and res["problems"] and res["rows"] == [], (name, side)
    base, cand = interleaved([100.0, 100.0, 100.0], [100.0, 100.0, 100.0])
    assert br.compare([], cand)["verdict"] == "INVALID" and br.compare(base, [])["verdict"] == "INVALID"
    assert br.structure_problems([1]) and br.validate_record({"schema": "x"}) and br.validate_record({}) != []


def test_cli_exit_code_for_unusable_records_is_2():
    d = tempfile.mkdtemp()
    good_base, good_cand = interleaved([100.0, 100.0, 100.0], [100.0, 100.0, 100.0])
    base_paths, cand_paths = write_records(d, "b", good_base), write_records(d, "c", good_cand)
    not_an_object = os.path.join(d, "list.json")
    open(not_an_object, "w").write("[1, 2, 3]")
    no_meta = os.path.join(d, "nometa.json")
    json.dump({"schema": br.SCHEMA, "cells": good_base[0]["cells"]}, open(no_meta, "w"))
    not_json = os.path.join(d, "garbage.json")
    open(not_json, "w").write("{not json")
    assert compare_bench.main(["--base"] + base_paths + ["--cand"] + cand_paths) == 0
    for bad in (not_an_object, no_meta, not_json):
        assert compare_bench.main(["--base", bad] + base_paths[1:] + ["--cand"] + cand_paths) == 2, bad
        assert compare_bench.main(["--summary", bad]) == 2, bad


def test_unknown_metric_is_invalid_not_a_failure():
    base, cand = interleaved([100.0, 100.0, 100.0], [100.0, 100.0, 100.0])
    for typo in ("k2_stage_ms", "k2_kernell", ""):
        res = br.compare(base, cand, metric=typo)
        assert res["verdict"] == "INVALID" and any("metric" in p for p in res["problems"]), typo
    assert br.compare(base, cand, metric="k2_kernel")["verdict"] == "PASS"  # a kernel that appears in the records is a valid metric
    assert br.known_metric(base, "k1_us") and not br.known_metric(base, "nope")
    d = tempfile.mkdtemp()
    bp, cp = write_records(d, "b", base), write_records(d, "c", cand)
    assert compare_bench.main(["--base"] + bp + ["--cand"] + cp + ["--metric", "k2_kernell"]) == 2
    assert compare_bench.main(["--summary"] + bp + ["--metric", "k2_kernell"]) == 2
    assert compare_bench.main(["--summary"] + bp + ["--metric", "k2_kernel"]) == 0


def test_sass_delta_text_does_not_depend_on_set_order():
    from collections import Counter
    key = ("sm_100a", "f")
    base = {key: {"count": 3, "hist": Counter({"BBB": 1, "AAA": 1, "CCC": 1})}}
    cand = {key: {"count": 3, "hist": Counter({"BBB": 2, "AAA": 2, "CCC": 0})}}  # three deltas of equal magnitude
    for _ in range(5):
        diffs = cb.compare_sass(base, cand)
        assert diffs == ["sm_100a f: opcode histogram differs (AAA +1, BBB +1, CCC -1)"], diffs


def test_tree_identity_prefers_environment_then_snapshot_marker():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "SNAPSHOT_COMMIT"), "w") as f:
        f.write(BASE_COMMIT + "\n")
    saved = {k: os.environ.pop(k, None) for k in ("CFBI_BENCH_COMMIT", "CFBI_BENCH_DIRTY", "CFBI_BENCH_TREE")}
    try:
        assert br.tree_identity(d) == (BASE_COMMIT, False, "snapshot")
        os.environ.update({"CFBI_BENCH_COMMIT": "abc", "CFBI_BENCH_DIRTY": "1", "CFBI_BENCH_TREE": "worktree"})
        assert br.tree_identity(d) == ("abc", True, "worktree")
        os.environ["CFBI_BENCH_DIRTY"] = "0"
        assert br.tree_identity(d)[1] is False
    finally:
        for k in saved:
            os.environ.pop(k, None)
        os.environ.update({k: v for k, v in saved.items() if v is not None})


# ---- resource usage and SASS ------------------------------------------------------------------------------------------------------------------

REGS = """Fatbin elf code:
================
arch = sm_100a
code version = [1,8]
host = linux
compile_size = 64bit

\tFunction _ZN4cfbi9k1_kernelILi16EEEvNS_6ParamsE:
\t\tREG:128 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:1240 TEXTURE:0 SURFACE:0 SAMPLER:0
\tFunction _ZN4cfbi9k2_kernelILi16EEEvNS_6ParamsE:
\t\tREG:172 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:1240 TEXTURE:0 SURFACE:0 SAMPLER:0
\tFunction _ZN4cfbi15k2_kernel_mixedENS_6ParamsE:
\t\tREG:213 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:1240 TEXTURE:0 SURFACE:0 SAMPLER:0
"""


def test_resource_usage_parsing_and_equality():
    base = cb.parse_resource_usage(REGS)
    assert base[("sm_100a", "_ZN4cfbi9k1_kernelILi16EEEvNS_6ParamsE")] == {"REG": 128, "STACK": 0, "SHARED": 0, "LOCAL": 0}
    assert len(base) == 3
    assert cb.compare_resources(base, cb.parse_resource_usage(REGS)) == ([], [])


def test_one_more_register_or_any_spill_is_a_difference():
    base = cb.parse_resource_usage(REGS)
    one_more = cb.parse_resource_usage(REGS.replace("REG:172", "REG:173"))
    diffs, _ = cb.compare_resources(base, one_more)
    assert len(diffs) == 1 and "REG 172 -> 173" in diffs[0]
    spill = cb.parse_resource_usage(REGS.replace("REG:213 STACK:0", "REG:213 STACK:16"))
    assert any("STACK 0 -> 16" in d for d in cb.compare_resources(base, spill)[0])
    local = cb.parse_resource_usage(REGS.replace("REG:172 STACK:0 SHARED:0 LOCAL:0", "REG:172 STACK:0 SHARED:0 LOCAL:8"))
    assert any("LOCAL 0 -> 8" in d for d in cb.compare_resources(base, local)[0])


def test_new_kernels_are_not_a_difference_but_a_missing_one_is():
    base = cb.parse_resource_usage(REGS)
    cand_text = REGS + "\tFunction _ZN4cfbi15k2_dyn_kernelEv:\n\t\tREG:200 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:100\n"
    diffs, added = cb.compare_resources(base, cb.parse_resource_usage(cand_text))
    assert diffs == [] and len(added) == 1 and "k2_dyn_kernel" in added[0]
    removed = cb.parse_resource_usage(REGS.replace("\tFunction _ZN4cfbi15k2_kernel_mixedENS_6ParamsE:\n\t\tREG:213 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:1240 TEXTURE:0 SURFACE:0 SAMPLER:0\n", ""))
    assert any("missing in the candidate" in d for d in cb.compare_resources(base, removed)[0])
    assert cb.compare_resources(base, one_function_changed(base), functions=["k1_kernel"]) == ([], [])


def one_function_changed(parsed):
    out = {k: dict(v) for k, v in parsed.items()}
    for k in out:
        if "k2_kernel_mixed" in k[1]:
            out[k]["REG"] += 1
    return out


SASS = """Fatbin elf code:
================
arch = sm_100a

\tFunction : _ZN4cfbi9k1_kernelILi16EEEvNS_6ParamsE
\t.headerflags    @"EF_CUDA_SM100"
\t\t/*0000*/                   MOV R1, c[0x0][0x28] ;                    /* 0x00000a0000017a02 */
\t\t                                                                     /* 0x000fc40000000f00 */
\t\t/*0010*/                   S2R R0, SR_TID.X ;                        /* 0x0000000000007919 */
\t\t/*0020*/              @!P0 FFMA R2, R2, R3, R4 ;                     /* 0x0000000302028223 */
\t\t/*0030*/                   EXIT ;                                    /* 0x000000000000794d */
"""


def test_sass_instruction_count_and_opcode_histogram():
    base = cb.parse_sass(SASS)
    key = ("sm_100a", "_ZN4cfbi9k1_kernelILi16EEEvNS_6ParamsE")
    assert base[key]["count"] == 4 and base[key]["hist"]["FFMA"] == 1 and base[key]["hist"]["MOV"] == 1
    assert cb.compare_sass(base, cb.parse_sass(SASS)) == []
    more = cb.parse_sass(SASS.replace("EXIT ;", "NOP ;\n\t\t/*0040*/                   EXIT ;"))
    diffs = cb.compare_sass(base, more)
    assert any("instructions 4 -> 5" in d for d in diffs) and any("opcode histogram" in d for d in diffs)
    swapped = cb.parse_sass(SASS.replace("FFMA", "FADD"))
    d2 = cb.compare_sass(base, swapped)
    assert len(d2) == 1 and "opcode histogram" in d2[0]


def test_binary_comparison_cli_exit_codes():
    d = tempfile.mkdtemp()
    a, b, c = (os.path.join(d, n) for n in ("a.txt", "b.txt", "c.txt"))
    open(a, "w").write(REGS)
    open(b, "w").write(REGS)
    open(c, "w").write(REGS.replace("REG:128", "REG:129"))
    assert cb.main(["--base-regs", a, "--cand-regs", b]) == 0
    assert cb.main(["--base-regs", a, "--cand-regs", c]) == 1
    assert cb.main(["--base-regs", a, "--cand-regs", os.path.join(d, "missing.txt")]) == 2


# ---- compile-only code-generation gate (benchmarks/codegen_gate.py) ----------------------------------------------------------------------------

ROWS = "_ZN4cfbi11rows_kernelENS_6ParamsE"
K1 = "_ZN4cfbi9k1_kernelILi16EEEvNS_6ParamsE"
CODE = ["LDC R1, c[0x0][0x37c]", "S2R R5, SR_CTAID.X", "IMAD.WIDE.U32 R2, R5, 0x20, R2", "LDG.E R0, desc[UR16][R2.64+0xc]", "@!P0 EXIT",
        "FFMA R7, R2, R3, R4", "EXIT"]


def cg_sass(functions, arch="sm_100a", offset0=0, enc=0x1000, ctrl=0x2000):
    """``cuobjdump -sass`` text of ``{function: [instruction text]}``: an instruction line, then its control-word line, as cuobjdump prints them."""
    out = ["Fatbin elf code:", "================", "arch = %s" % arch, "", "\tcode for %s" % arch]
    for name, insns in functions.items():
        out += ["\t\tFunction : %s" % name, '\t.headerflags\t@"EF_CUDA_SM100"']
        for i, ins in enumerate(insns):
            out.append("        /*%04x*/                   %s ;      /* 0x%016x */" % (offset0 + 16 * i, ins, enc + i))
            out.append("                                         /* 0x%016x */" % (ctrl + i))
        out.append("\t\t..........")
    return "\n".join(out) + "\n"


def cg_regs(functions, arch="sm_100a"):
    out = ["Fatbin elf code:", "================", "arch = %s" % arch, "", "Resource usage:", " Common:", "  GLOBAL:0"]
    for name, (reg, stack) in functions.items():
        out += [" Function %s:" % name, "  REG:%d STACK:%d SHARED:0 LOCAL:0 CONSTANT[0]:16 TEXTURE:0 SURFACE:0 SAMPLER:0" % (reg, stack)]
    return "\n".join(out) + "\n"


def cg_table(code=CODE, **kw):
    return cg.kernel_table(cg_regs({ROWS: (128, 0), K1: (100, 0)}), cg_sass({ROWS: code, K1: ["EXIT"]}, **kw))


def cg_kernel(reg=128, stack=0, spill=(0, 0), sass="h0", ins=100, shared=0, local=0, maxthr=256, maxreg=128):
    return {"name": "n", "unit": None, "reg": reg, "stack": stack, "shared": shared, "local": local, "instructions": ins, "sass_hash": sass,
            "sass_hash_strict": sass, "spill_stores": spill[0], "spill_loads": spill[1], "max_threads": maxthr, "maxreg": maxreg}


def cg_dump(kernels, label="toolchain-A", defines=(), content="tree-1"):
    return {"schema": cg.SCHEMA,
            "meta": {"toolchain": {"source": "nvcc", "label": label, "nvcc": "nvcc 13.1.115", "host_compiler": "g++ 14.3.0", "flags": list(cg.PRODUCTION_FLAGS),
                                   "std": cg.STD, "archs": sorted(kernels), "src_hash": cg.FIXED_SRC_HASH},
                     "variant": {"defines": list(defines)}, "tree": {"path": None, "commit": None, "content_hash": content, "units": []}},
            "kernels": kernels}


def cg_base():
    """Two architectures with five kernels each: the row engine, its frozen reference, K1, K2 and the plan kernel."""
    def arch(rows_spill):
        return {"cfbi::rows_kernel": cg_kernel(reg=128, stack=112, spill=rows_spill, sass="rows"),
                "cfbi::ref::rows_kernel": cg_kernel(reg=48, sass="ref-rows", maxreg=None, maxthr=None),
                "cfbi::k1_kernel<Li16E>": cg_kernel(reg=128, sass="k1"),
                "cfbi::k2_kernel<Li16E>": cg_kernel(reg=172, sass="k2"),
                "cfbi::plan_kernel": cg_kernel(reg=40, sass="plan")}
    return cg_dump({"sm_100a": arch((4, 4)), "sm_90a": arch((428, 172))})


def cg_copy(dump):
    return json.loads(json.dumps(dump))


def cg_edit(dump, arch, key, **fields):
    out = cg_copy(dump)
    out["kernels"][arch][key].update(fields)
    return out


def test_codegen_kernel_keys_ignore_mangling_details():
    names = {
        "_ZN4cfbi3ref11rows_kernelENS_6ParamsE": "cfbi::ref::rows_kernel",
        "_ZN4cfbi11rows_kernelENS_6ParamsE": "cfbi::rows_kernel",
        "_ZN4cfbi11rows_kernelILb0EEEvNS_6ParamsE": "cfbi::rows_kernel<Lb0E>",
        "_ZN4cfbi9k2_kernelILi16EEEvNS_6ParamsE": "cfbi::k2_kernel<Li16E>",
        "_ZN4cfbi19finalize_pos_kernelEPKNS_7SeqPlanEPii": "cfbi::finalize_pos_kernel",
        "_ZN4cfbi15k2_sched_kernelILi1024EEEvPKNS_7SeqPlanEiiiii": "cfbi::k2_sched_kernel<Li1024E>",
        "_Z6simplePf": "simple",
    }
    for mangled, key in names.items():
        assert cg.kernel_key(mangled) == key, (mangled, cg.kernel_key(mangled))
    assert cg.kernel_key("not_a_mangled_name") == "not_a_mangled_name"
    assert cg.family("cfbi::rows_kernel<Lb1E>") == "cfbi::rows_kernel" and cg.family("cfbi::ref::rows_kernel") == "cfbi::ref::rows_kernel"
    assert cg.kernel_key("_ZN4cfbi3ref11rows_kernelENS_6ParamsE") != cg.kernel_key("_ZN4cfbi11rows_kernelENS_6ParamsE")   # frozen reference != row engine
    assert set(cg.assign_keys(["_ZN4cfbi1fEi", "_ZN4cfbi1fEf", "_ZN4cfbi1gEi"])) == {"cfbi::f(i)", "cfbi::f(f)", "cfbi::g"}    # overloads stay apart


def test_codegen_normalisation_drops_offsets_encodings_and_trailer_comments_only():
    a = cg.normalise_instruction("        /*0050*/                   IMAD.WIDE.U32 R2, R5, 0x20, R2 ;                 /* 0x0000002005027825 */")
    b = cg.normalise_instruction("   /*0a50*/   IMAD.WIDE.U32   R2,  R5, 0x20, R2 ;   /* 0x00000000deadbeef */   /* trailer */")
    assert a == b == "IMAD.WIDE.U32 R2, R5, 0x20, R2"
    assert cg.normalise_instruction("                                  /* 0x000fca00078e0002 */") is None      # the control-word line
    assert cg.normalise_instruction('\t.headerflags\t@"EF_CUDA_SM100"') is None
    assert cg.normalise_instruction("        /*0080*/              @!P0 EXIT ;   /* 0x0 */") == "@!P0 EXIT"
    assert cg.normalise_instruction("        /*10080*/ BRA 0x4c10 ; /* 0x1 */") == "BRA 0x4c10"                  # five-digit offsets, branch targets stay


def test_codegen_dumps_that_differ_only_in_offsets_and_encodings_compare_equal():
    base = cg_table()
    shifted = cg_table(offset0=0x400, enc=0x7000, ctrl=0x9000)
    a, b = base["sm_100a"]["cfbi::rows_kernel"], shifted["sm_100a"]["cfbi::rows_kernel"]
    assert a["sass_hash"] == b["sass_hash"] and a["instructions"] == b["instructions"] == len(CODE)
    assert a["sass_hash_strict"] != b["sass_hash_strict"]                 # the diagnostic hash keeps the encodings
    rep = cg.compare_dumps(cg_dump(base), cg_dump(shifted))
    assert rep["exit"] == 0 and rep["differences"] == [] and any("scheduling only" in n for n in rep["notes"])


def test_codegen_one_changed_opcode_operand_or_immediate_is_a_difference():
    base = cg_dump(cg_table())
    for idx, new in ((5, "FADD R7, R2, R3, R4"),                    # opcode
                     (5, "FFMA R8, R2, R3, R4"),                    # destination register only
                     (2, "IMAD.WIDE.U32 R2, R5, 0x24, R2"),         # immediate only
                     (3, "LDG.E R0, desc[UR16][R2.64+0x10]")):      # address offset only
        code = list(CODE)
        code[idx] = new
        rep = cg.compare_dumps(base, cg_dump(cg_table(code)))
        assert rep["exit"] == 1 and any("cfbi::rows_kernel" in d and "normalised SASS differs" in d for d in rep["differences"]), new
    longer = cg.compare_dumps(base, cg_dump(cg_table(CODE + ["NOP"])))
    assert longer["exit"] == 1 and any("INSTRUCTIONS 7 -> 8" in d for d in longer["differences"])


def test_codegen_one_more_register_or_any_spill_is_reported():
    base = cg_base()
    for fields, text in (({"reg": 173}, "REG 172 -> 173"), ({"stack": 16}, "STACK 0 -> 16"), ({"local": 8}, "LOCAL 0 -> 8"), ({"shared": 1024}, "SHARED 0 -> 1024"),
                         ({"spill_stores": 4}, "SPILL_STORES 0 -> 4"), ({"spill_loads": 4}, "SPILL_LOADS 0 -> 4")):
        rep = cg.compare_dumps(base, cg_edit(base, "sm_100a", "cfbi::k2_kernel<Li16E>", **fields))
        assert rep["exit"] == 1 and any(text in d for d in rep["differences"]), (fields, rep["differences"])


def test_codegen_allow_list_aliases_and_allowed_differences():
    base = cg_base()
    cand = cg_copy(base)                                                 # the row engine became an instantiation of a template
    for arch in cand["kernels"]:
        cand["kernels"][arch]["cfbi::rows_kernel<Lb0E>"] = cand["kernels"][arch].pop("cfbi::rows_kernel")
    rep = cg.compare_dumps(base, cand)
    assert rep["exit"] == 1 and any("cfbi::rows_kernel: missing in the candidate" in d for d in rep["differences"])
    allow = {"aliases": {"cfbi::rows_kernel<Lb0E>": "cfbi::rows_kernel"}, "may_differ": []}
    rep = cg.compare_dumps(base, cand, allow)
    assert rep["exit"] == 0 and rep["added"] == []
    changed = cg_edit(base, "sm_100a", "cfbi::k1_kernel<Li16E>", sass_hash="k1-new")
    assert cg.compare_dumps(base, changed)["exit"] == 1
    for entry in ("cfbi::k1_kernel<Li16E>", "cfbi::k1_kernel"):          # a changed kernel is tolerated only when it is on the allow-list (key or family)
        rep = cg.compare_dumps(base, changed, {"aliases": {}, "may_differ": [entry]})
        assert rep["exit"] == 0 and len(rep["allowed"]) == 1, entry
    rep = cg.compare_dumps(base, base, {"aliases": {"cfbi::gone": "cfbi::rows_kernel"}, "may_differ": []})
    assert any("matches no candidate kernel" in n for n in rep["notes"])  # an unused alias is reported, never silently ignored
    both = cg_copy(cand)
    for arch in both["kernels"]:
        both["kernels"][arch]["cfbi::rows_kernel"] = dict(both["kernels"][arch]["cfbi::rows_kernel<Lb0E>"])
    with pytest.raises(cg.Unusable):                                     # two candidate kernels cannot stand for one base kernel
        cg.compare_dumps(base, both, allow)


def test_codegen_missing_base_kernels_and_new_kernels():
    base = cg_base()
    missing = cg_copy(base)
    del missing["kernels"]["sm_90a"]["cfbi::plan_kernel"]
    rep = cg.compare_dumps(base, missing)
    assert rep["exit"] == 1 and any("sm_90a cfbi::plan_kernel: missing in the candidate" in d for d in rep["differences"])
    ok = cg_copy(base)                                                   # new kernels are listed, not a difference, and gated
    ok["kernels"]["sm_100a"]["cfbi::new_kernel"] = cg_kernel(reg=100, stack=0, spill=(0, 0), sass="new")
    rep = cg.compare_dumps(base, ok)
    assert rep["exit"] == 0 and rep["added"] == ["sm_100a cfbi::new_kernel"] and rep["gate_r"]["failures"] == []
    for fields, text in (({"reg": 129}, "129 registers"), ({"stack": 200}, "stack 200 B above the base 112 B"),
                         ({"spill_stores": 8}, "spill stores 8 B above the base 4 B"), ({"spill_loads": 9}, "spill loads 9 B above the base 4 B")):
        rep = cg.compare_dumps(base, cg_edit(ok, "sm_100a", "cfbi::new_kernel", **fields))
        assert rep["exit"] == 1 and any(text in f for f in rep["gate_r"]["failures"]), (fields, rep["gate_r"])


def test_codegen_change_mode_lets_only_the_row_engine_change_and_gates_it():
    base = cg_base()
    rows_ok = cg_copy(base)
    for arch in rows_ok["kernels"]:
        rows_ok["kernels"][arch]["cfbi::rows_kernel"].update(sass_hash="rows-new", instructions=90)
    assert cg.compare_dumps(base, rows_ok)["exit"] == 1                  # identity mode: any change is a difference
    rep = cg.compare_dumps(base, rows_ok, mode="change")
    assert rep["exit"] == 0 and len(rep["allowed"]) == 2 and len(rep["gate_r"]["checked"]) >= 2
    for fields, text in (({"reg": 129, "maxreg": 255}, "129 registers"), ({"stack": 113}, "stack 113 B above"), ({"spill_stores": 5}, "spill stores 5 B above"),
                         ({"maxreg": 255}, "launch-bound attribute maxreg 255")):
        rep = cg.compare_dumps(base, cg_edit(rows_ok, "sm_100a", "cfbi::rows_kernel", **fields), mode="change")
        assert rep["exit"] == 1 and any(text in f for f in rep["gate_r"]["failures"]), (fields, rep["gate_r"])
    for key in ("cfbi::k1_kernel<Li16E>", "cfbi::ref::rows_kernel", "cfbi::plan_kernel"):      # everything else, the frozen reference included, stays identical
        rep = cg.compare_dumps(base, cg_edit(rows_ok, "sm_90a", key, sass_hash="changed"), mode="change")
        assert rep["exit"] == 1 and any(key in d for d in rep["differences"]), key
    assert cg.compare_dumps(base, base, mode="change")["gate_r"]["checked"]                   # the row engine is gated even when it equals the base
    rep = cg.compare_dumps(base, rows_ok, mode="change", prev=rows_ok)
    assert any("unchanged against the previous accepted state" in n for n in rep["notes"])
    with pytest.raises(cg.Unusable):
        cg.compare_dumps(base, rows_ok, mode="bogus")


def test_codegen_refuses_dumps_of_different_toolchains_but_allows_different_variant_defines():
    base = cg_base()
    for field, value in (("nvcc", "nvcc 13.4"), ("host_compiler", "g++ 8.5"), ("flags", ["-O2"]), ("label", "another toolchain"), ("source", "files"), ("std", "c++17"),
                         ("src_hash", "0x1ull")):
        other = cg_copy(base)
        other["meta"]["toolchain"][field] = value
        with pytest.raises(cg.Unusable):
            cg.compare_dumps(base, other)
    fewer = cg_copy(base)
    fewer["meta"]["toolchain"]["archs"] = ["sm_100a"]
    del fewer["kernels"]["sm_90a"]
    with pytest.raises(cg.Unusable):
        cg.compare_dumps(base, fewer)
    assert cg.compare_dumps(base, cg_dump(base["kernels"], defines=["CFBI_NEG_CONTROL=1"]))["exit"] == 0      # variant defines are recorded separately


def test_codegen_empty_dumps_are_unusable_and_an_identical_tree_is_flagged():
    base = cg_base()
    empty = cg_copy(base)
    empty["kernels"]["sm_90a"] = {}
    with pytest.raises(cg.Unusable):
        cg.compare_dumps(base, empty)
    with pytest.raises(cg.Unusable):
        cg.compare_dumps(empty, base)
    with pytest.raises(cg.Unusable):
        cg.compare_dumps(cg_dump({}), cg_dump({}))
    with pytest.raises(cg.Unusable):
        cg.check_dump({"schema": "other"})
    assert any("A/A" in n for n in cg.compare_dumps(base, base)["notes"])
    assert not any("A/A" in n for n in cg.compare_dumps(base, cg_dump(base["kernels"], content="tree-2"))["notes"])


def test_codegen_cache_key_changes_with_every_input():
    tc = {"source": "nvcc", "label": "a", "flags": ["-O3"], "archs": ["sm_100a"]}
    key = cg.cache_key(tc, ["X=1"], ["u.cu"], "c1")
    assert key == cg.cache_key(dict(tc), ["X=1"], ["u.cu"], "c1")
    others = [cg.cache_key(dict(tc, label="b"), ["X=1"], ["u.cu"], "c1"), cg.cache_key(dict(tc, flags=["-O2"]), ["X=1"], ["u.cu"], "c1"),
              cg.cache_key(dict(tc, archs=["sm_90a"]), ["X=1"], ["u.cu"], "c1"), cg.cache_key(tc, [], ["u.cu"], "c1"), cg.cache_key(tc, ["X=1"], ["v.cu"], "c1"),
              cg.cache_key(tc, ["X=1"], ["u.cu"], "c2")]
    assert all(o != key for o in others) and len(set(others)) == len(others)


def test_codegen_tree_helpers():
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "csrc", "cfbi"))
    os.makedirs(os.path.join(d, "flash_kda", "cfbi"))
    open(os.path.join(d, "csrc", "cfbi", "x.cuh"), "w").write("a")
    open(os.path.join(d, "flash_kda", "cfbi", "_ext.py"), "w").write('_SOURCES = ("b.cpp", "u1.cu", "u2.cu")\nNVCC_FLAGS = ["-O3", "-lineinfo"]\n')
    assert cg.units_of(d) == ["csrc/cfbi/u1.cu", "csrc/cfbi/u2.cu"] and cg.ext_flags_of(d) == ["-O3", "-lineinfo"]
    h = cg.tree_content_hash(d)
    open(os.path.join(d, "README"), "w").write("outside the build inputs")
    assert cg.tree_content_hash(d) == h
    open(os.path.join(d, "csrc", "cfbi", "x.cuh"), "w").write("b")
    assert cg.tree_content_hash(d) != h
    empty = tempfile.mkdtemp()
    assert cg.units_of(empty) == list(cg.DEFAULT_UNITS) and cg.ext_flags_of(empty) is None
    open(os.path.join(empty, "SNAPSHOT_COMMIT"), "w").write("abc123\n")
    assert cg.tree_commit(empty) == "abc123"


def test_codegen_flag_unit_and_arch_lists_match_the_sources():
    assert cg.ext_flags_of(str(_ROOT)) == cg.PRODUCTION_FLAGS            # parsed from the text of flash_kda/cfbi/_ext.py, never imported
    assert cg.units_of(str(_ROOT)) == list(cg.DEFAULT_UNITS)
    values = {}
    for node in ast.parse((_ROOT / "setup.py").read_text()).body:       # setup.py is parsed, not imported (importing it updates the submodules)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("SUPPORTED_CUDA_ARCHS", "CFBI_CUDA_ARCHS") for t in node.targets):
            values[node.targets[0].id] = eval(compile(ast.Expression(node.value), "setup.py", "eval"), {}, dict(values))
    assert tuple("sm_" + a for a in values["CFBI_CUDA_ARCHS"]) == cg.MATRIX_ARCHS
    assert set(cg.GATE_ARCHS) <= set(cg.MATRIX_ARCHS)
    assert cg.load_allow(cg.DEFAULT_ALLOW)["aliases"] == {}              # the committed allow-list parses


PTXAS = """ptxas info    : 103 bytes gmem
ptxas info    : Compiling entry function '_ZN4cfbi11rows_kernelENS_6ParamsE' for 'sm_100a'
ptxas info    : Function properties for _ZN4cfbi11rows_kernelENS_6ParamsE
    112 bytes stack frame, 4 bytes spill stores, 4 bytes spill loads
ptxas info    : Used 128 registers, used 1 barriers, 112 bytes cumulative stack size, 8192 bytes smem
ptxas info    : Compile time = 3051.1 ms
ptxas info    : Compiling entry function '_ZN4cfbi11plan_kernelEPKliPKiS3_iiPNS_7SeqPlanE' for 'sm_100a'
ptxas info    : Function properties for _ZN4cfbi11plan_kernelEPKliPKiS3_iiPNS_7SeqPlanE
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers, used 1 barriers, 4096 bytes smem
ptxas info    : Compiling entry function '_ZN4cfbi11rows_kernelENS_6ParamsE' for 'sm_90a'
ptxas info    : Function properties for _ZN4cfbi11rows_kernelENS_6ParamsE
    168 bytes stack frame, 428 bytes spill stores, 172 bytes spill loads
ptxas info    : Used 128 registers, used 1 barriers, 8192 bytes smem
"""

ELF = """
Fatbin elf code:
================
arch = sm_100a
code version = [1,8]

.nv.info._ZN4cfbi11rows_kernelENS_6ParamsE
\t<0x1>
\tAttribute:\tEIATTR_CUDA_API_VERSION
\tFormat:\tEIFMT_SVAL
\tValue:\t0x83
\t<0x4>
\tAttribute:\tEIATTR_MAXREG_COUNT
\tFormat:\tEIFMT_HVAL
\tValue:\t0x80
\t<0x12>
\tAttribute:\tEIATTR_MAX_THREADS
\tFormat:\tEIFMT_SVAL
\tValue:\t0x100 0x1 0x1

.nv.info._ZN4cfbi11plan_kernelEPKliPKiS3_iiPNS_7SeqPlanE
\t<0x4>
\tAttribute:\tEIATTR_MAXREG_COUNT
\tFormat:\tEIFMT_HVAL
\tValue:\t0xff
"""


def test_codegen_ptxas_log_and_elf_attribute_parsers():
    p = cg.parse_ptxas_log(PTXAS)
    assert p[("sm_100a", ROWS)] == {"stack": 112, "spill_stores": 4, "spill_loads": 4, "registers": 128}
    assert p[("sm_90a", ROWS)]["spill_stores"] == 428 and p[("sm_90a", ROWS)]["spill_loads"] == 172
    assert p[("sm_100a", "_ZN4cfbi11plan_kernelEPKliPKiS3_iiPNS_7SeqPlanE")]["registers"] == 40
    e = cg.parse_elf_attributes(ELF)
    assert e[("sm_100a", ROWS)] == {"maxreg": 128, "max_threads": 256}
    assert e[("sm_100a", "_ZN4cfbi11plan_kernelEPKliPKiS3_iiPNS_7SeqPlanE")] == {"maxreg": 255}


def test_codegen_dump_from_toolchain_text_files_and_the_operand_only_control():
    d = tempfile.mkdtemp()

    def write(name, text):
        path = os.path.join(d, name)
        with open(path, "w") as f:
            f.write(text)
        return path

    regs, ptx = write("a.regs.txt", cg_regs({ROWS: (128, 112), K1: (100, 0)})), write("a.ptxas.txt", PTXAS)
    base = cg.dump_from_files(regs, write("a.sass.txt", cg_sass({ROWS: CODE, K1: ["EXIT"]})), "container toolchain", ptxas_log=ptx)
    cg.check_dump(base)
    rows = base["kernels"]["sm_100a"]["cfbi::rows_kernel"]
    assert rows["reg"] == 128 and rows["spill_stores"] == 4 and rows["instructions"] == len(CODE)     # the spill figures come from the ptxas lines
    assert cg.compare_dumps(base, base)["exit"] == 0
    for idx, new in ((5, "FFMA R8, R2, R3, R4"), (2, "IMAD.WIDE.U32 R2, R5, 0x24, R2")):          # operand only, immediate only: the gate must fail
        code = list(CODE)
        code[idx] = new
        cand = cg.dump_from_files(regs, write("b.sass.txt", cg_sass({ROWS: code, K1: ["EXIT"]})), "container toolchain", ptxas_log=ptx)
        assert cg.compare_dumps(base, cand)["exit"] == 1, new
    other = cg.dump_from_files(regs, write("a.sass.txt", cg_sass({ROWS: CODE, K1: ["EXIT"]})), "another toolchain", ptxas_log=ptx)
    with pytest.raises(cg.Unusable):                                                                   # another toolchain label is not comparable
        cg.compare_dumps(base, other)
    with pytest.raises(cg.Unusable):                                                                   # and neither is a dump built by nvcc here
        cg.compare_dumps(base, cg_base())


def test_codegen_reference_figures_are_a_warning_not_a_failure():
    base = cg_base()
    assert cg.literal_check(base) == []
    drift = cg_edit(base, "sm_100a", "cfbi::rows_kernel", spill_stores=8)
    assert any("toolchain drift" in w for w in cg.literal_check(drift))
    rep = cg.compare_dumps(drift, drift)
    assert rep["exit"] == 0 and "WARNING" in cg.format_report(rep, drift) and cg.format_report(rep).endswith("VERDICT: PASS")


def test_codegen_cli_exit_codes():
    d = tempfile.mkdtemp()
    base = cg_base()
    paths = {}
    for name, dump in (("base", base), ("same", base), ("diff", cg_edit(base, "sm_100a", "cfbi::k1_kernel<Li16E>", sass_hash="x")),
                       ("other", cg_dump(base["kernels"], label="other-toolchain"))):
        paths[name] = os.path.join(d, name + ".json")
        with open(paths[name], "w") as f:
            json.dump(dump, f)
    out = os.path.join(d, "report.json")
    assert cg.main(["compare", "--base", paths["base"], "--cand", paths["same"], "--json", out]) == 0
    with open(out) as f:
        assert json.load(f)["exit"] == 0
    assert cg.main(["compare", "--base", paths["base"], "--cand", paths["diff"]]) == 1
    assert cg.main(["compare", "--base", paths["base"], "--cand", paths["other"]]) == 2
    assert cg.main(["compare", "--base", paths["base"], "--cand", os.path.join(d, "missing.json")]) == 2
    with open(os.path.join(d, "bad.json"), "w") as f:
        f.write("{not json")
    assert cg.main(["compare", "--base", paths["base"], "--cand", os.path.join(d, "bad.json")]) == 2
    assert cg.main([]) == 2
    allow = os.path.join(d, "allow.json")
    with open(allow, "w") as f:
        json.dump({"schema": cg.ALLOW_SCHEMA, "aliases": {}, "may_differ": ["cfbi::k1_kernel"]}, f)
    assert cg.main(["compare", "--base", paths["base"], "--cand", paths["diff"], "--allow", allow]) == 0


def _toolchain_or_skip():
    try:
        return cg.find_toolchain()
    except cg.Unusable as e:
        pytest.skip("no toolchain for the compile-only gate on this machine: %s" % e)


def test_codegen_controls_with_nvcc():
    """The gate on the real sources (login node, conda nvcc): base against base passes, the existing row-engine negative control fails and names the
    row engine, and register-pressure variants of the row engine fail the resource gate."""
    tc, tree, cache = _toolchain_or_skip(), str(_ROOT), tempfile.mkdtemp()
    base = cg.dump_tree(tree, cg.GATE_ARCHS, (), tc, cache_dir=cache)
    assert set(base["kernels"]) == set(cg.GATE_ARCHS) and all(len(v) >= 10 for v in base["kernels"].values())
    assert cg.dump_tree(tree, cg.GATE_ARCHS, (), tc, cache_dir=cache)["meta"]["cache"] == "hit"
    assert cg.literal_check(base) == [], cg.literal_check(base)           # the recorded figures of the base row engine hold with this toolchain
    variants = {"same": (), "neg": ("CFBI_NEG_CONTROL=1",), "regs": ("CFBI_ROWS_MINB=1",), "spill": ("CFBI_ROWS_MINB=3",)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(variants)) as pool:
        futs = {k: pool.submit(cg.dump_tree, tree, cg.GATE_ARCHS, v, tc) for k, v in variants.items()}
        built = {k: f.result() for k, f in futs.items()}
    assert cg.compare_dumps(base, built["same"])["exit"] == 0
    neg = cg.compare_dumps(base, built["neg"])
    assert neg["exit"] == 1 and any("cfbi::rows_kernel:" in d for d in neg["differences"]), neg
    for name in ("regs", "spill"):
        rep = cg.compare_dumps(base, built[name], mode="change")
        assert rep["exit"] == 1 and rep["gate_r"]["failures"] and not rep["differences"], (name, rep)
        assert cg.compare_dumps(base, built[name])["exit"] == 1            # identity mode shows the same variant as a difference of the row engine


def test_codegen_compile_matrix_and_a_failing_build_with_nvcc():
    tc = _toolchain_or_skip()
    res = cg.matrix_build(str(_ROOT), cg.MATRIX_ARCHS, tc)
    assert len(res) == len(cg.units_of(str(_ROOT))) * len(cg.MATRIX_ARCHS)
    assert [(u, a, msg) for u, a, ok, msg in res if not ok] == []
    d = tempfile.mkdtemp()                                               # a deliberate compile error is a failed build, never a skip
    os.makedirs(os.path.join(d, "csrc", "cfbi"))
    os.makedirs(os.path.join(d, "flash_kda", "cfbi"))
    with open(os.path.join(d, "flash_kda", "cfbi", "_ext.py"), "w") as f:
        f.write('_SOURCES = ("broken.cu",)\n')
    with open(os.path.join(d, "csrc", "cfbi", "broken.cu"), "w") as f:
        f.write('#error deliberately broken\n')
    res = cg.matrix_build(d, ("sm_90a",), tc)
    assert len(res) == 1 and res[0][2] is False and "deliberately broken" in res[0][3]
    with pytest.raises(cg.Unusable):
        cg.dump_tree(d, ("sm_90a",), (), tc)
    assert cg.main(["matrix", "--tree", d, "--archs", "sm_90a"]) == 1
    assert cg.main(["gate", "--base-tree", d, "--cand-tree", d, "--archs", "sm_90a"]) == 2


# ---- wall metrics, per-process-median samples, the widening mode and the comparison tool --------------------------------------------------------

PLAN_K = "void cfbi::plan_kernel(long const*, int, int const*, int const*, int, int, cfbi::SeqPlan*)"
ROWS_K = "void cfbi::rows_kernel(cfbi::Params)"
FILL_K = "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>)"
WALL_KEY = "decode;N=1;H=12;regime=lockstep;"


def wall_cell(key, vals, kind="decode"):
    """A decode cell with one trial per value: ``gpu_graph_us`` = value, ``gpu_eager_us`` = value + 1, ``cpu_graph_us`` = value + 3; the kernels of a trial are
    plan 2 us + rows 10 us and one helper kernel of 40 us that must stay out of the kernel sum."""
    runs = [{PLAN_K: 2.0, ROWS_K: 10.0, FILL_K: 40.0} for _ in vals]
    wall = [{"gpu_graph_us": v, "gpu_eager_us": v + 1.0, "cpu_graph_us": v + 3.0} for v in vals]
    return br.make_cell(kind, key, {}, runs, drop_helpers=True, wall=wall)


def wall_rec(commit, cells, ts, stat=br.PROCESS_MEDIAN, gpu="NVIDIA B200", sm=148, host="node1", dirty=False):
    meta = {"commit": commit, "dirty": dirty, "gpu": gpu, "sm_count": sm, "host": host, "timestamp": ts, "sample_stat": stat, "trials": 5}
    return {"schema": br.SCHEMA, "meta": meta, "cells": [wall_cell(k, v) for k, v in cells.items()]}


def trials(median, spread=0.0):
    """Five trial values whose median is ``median`` and whose range is ``2 * spread`` (relative)."""
    return [median * (1 - spread), median * (1 - spread / 2), median, median * (1 + spread / 2), median * (1 + spread)]


def wall_sets(base_meds, cand_meds, keys=(WALL_KEY,), hour="10", cand_commit="abc1234", **kw):
    """Interleaved record lists (A B A B A B ...) with one per-process median per record file for each cell key."""
    base = [wall_rec(BASE_COMMIT, {k: trials(m) for k in keys}, "2026-11-10T%s:0%d:00Z" % (hour, 2 * i), **kw) for i, m in enumerate(base_meds)]
    cand = [wall_rec(cand_commit, {k: trials(m) for k in keys}, "2026-11-10T%s:0%d:00Z" % (hour, 2 * i + 1), **kw) for i, m in enumerate(cand_meds)]
    return base, cand


def test_wall_metrics_round_trip_and_helper_kernels_are_left_out_of_the_kernel_sum():
    cell = wall_cell("decode;N=1;H=12;regime=cold;", [14.0, 14.2, 14.1, 14.0, 14.3])
    s = cell["samples"][0]
    assert s["total_us"] == 12.0 and s["helper_us"] == 40.0 and set(s["kernels"]) == {"plan_kernel", "rows_kernel"}
    assert s["gpu_graph_us"] == 14.0 and s["gpu_eager_us"] == 15.0 and s["cpu_graph_us"] == 17.0 and "cpu_eager_us" not in s
    d = tempfile.mkdtemp()
    r = wall_rec(BASE_COMMIT, {"decode;N=1;H=12;regime=cold;": [14.0] * 5}, "2026-11-10T10:00:00Z")
    path = os.path.join(d, "r.json")
    br.write_record(path, r["meta"], r["cells"])
    back = br.load_record(path)
    assert br.structure_problems(back) == [] and br.metric_value(back["cells"][0]["samples"][0], "gpu_graph_us") == 14.0
    assert br.metric_value(back["cells"][0]["samples"][0], "cpu_eager_us") is None
    kept = br.make_cell("decode", "k", {}, [{PLAN_K: 2.0, ROWS_K: 10.0, FILL_K: 40.0}])             # without the exclusion every kernel is summed, as before
    assert kept["samples"][0]["total_us"] == 52.0 and "helper_us" not in kept["samples"][0]
    assert br.is_helper_kernel(FILL_K) and br.is_helper_kernel("Memset (Device)") and br.is_helper_kernel("void at::native::elementwise_kernel<128, 2>")
    assert not br.is_helper_kernel(ROWS_K) and not br.is_helper_kernel("void cfbi::ref::rows_kernel(cfbi::Params)")
    with pytest.raises(ValueError):
        br.make_cell("decode", "k", {}, [{ROWS_K: 1.0}], wall=[{"bogus_us": 1.0}])
    with pytest.raises(ValueError):
        br.make_cell("decode", "k", {}, [{ROWS_K: 1.0}], wall=[])
    broken = wall_rec(BASE_COMMIT, {WALL_KEY: [1.0] * 5}, "2026-11-10T10:00:00Z")
    broken["cells"][0]["samples"][0]["gpu_graph_us"] = "fast"
    assert br.structure_problems(broken)


def test_cell_keys_end_every_field_with_a_delimiter_and_the_cell_filter_uses_them():
    k1, k16 = br.cell_key("decode", N=1, H=12, regime="lockstep"), br.cell_key("decode", N=16, H=12, regime="lockstep")
    assert k1 == "decode;N=1;H=12;regime=lockstep;" and "N=1;" in k1 and "N=1;" not in k16
    base, cand = wall_sets([100.0, 100.05, 99.95], [100.0, 100.05, 99.95], keys=(k1, k16))
    res = br.compare(base, cand, metric="gpu_graph_us", cells=["N=1;"])
    assert [r["key"] for r in res["rows"]] == [k1]
    assert [r["key"] for r in br.compare(base, cand, metric="gpu_graph_us", cells=["N=16;"])["rows"]] == [k16]


def test_wall_metric_regress_gate_passes_aa_fails_a_3_percent_slower_and_passes_a_faster_candidate():
    d = tempfile.mkdtemp()
    for cand_meds, code in (([100.02, 100.07, 99.97], 0), ([103.0, 103.05, 102.95], 1), ([97.0, 97.05, 96.95], 0)):
        base, cand = wall_sets([100.0, 100.05, 99.95], cand_meds)
        args = (["--base"] + write_records(d, "a", base) + ["--cand"] + write_records(d, "b", cand) +
                ["--mode", "regress", "--tol", "1", "--metric", "gpu_graph_us", "--min-repeats", "3", "--expect-base-commit", BASE_COMMIT, "--require-clean-cand"])
        assert compare_bench.main(args) == code, cand_meds
    base, cand = wall_sets([100.0, 100.05, 99.95], [100.0, 100.05, 99.95])                       # the kernel sum of the same records gates as well
    assert br.compare(base, cand, metric="total_us")["verdict"] == "PASS" and br.compare(base, cand, metric="total_us")["rows"][0]["base_median"] == 12.0


def test_widening_mode_passes_an_unstable_aa_pair_that_the_worst_mode_fails_and_still_fails_a_slower_candidate():
    base, cand = wall_sets([100.0, 100.45, 100.9], [100.15, 100.6, 101.05])                      # 0.9 % spread on each side, offset by 0.15 %
    worst = br.compare(base, cand, metric="gpu_graph_us", baseline_unstable="worst")
    assert worst["verdict"] == "FAIL" and 1.0 < worst["rows"][0]["delta_pct"] < 1.1              # fastest baseline repeat vs slowest candidate repeat
    assert br.compare(base, cand, metric="gpu_graph_us")["verdict"] == "INVALID"                  # the default does not accept an unstable baseline
    wide = br.compare(base, cand, metric="gpu_graph_us", baseline_unstable="widen")
    row = wide["rows"][0]
    assert wide["verdict"] == "PASS" and row["noise_limited"] and abs(row["tol_pct_effective"] - 1.896) < 0.01 and row["median_delta_pct"] < 0.2
    assert "noise-limited" in br.format_table(wide)
    _, slower = wall_sets([100.0, 100.45, 100.9], [103.2, 103.65, 104.1])
    assert br.compare(base, slower, metric="gpu_graph_us", baseline_unstable="widen")["verdict"] == "FAIL"
    stable_b, stable_c = wall_sets([100.0, 100.05, 99.95], [100.0, 100.05, 99.95])               # stable cells keep the plain tolerance
    assert not br.compare(stable_b, stable_c, metric="gpu_graph_us", baseline_unstable="widen")["rows"][0]["noise_limited"]
    assert br.compare(base, cand, metric="gpu_graph_us", baseline_unstable="bogus")["verdict"] == "INVALID"


def test_process_median_samples_are_one_value_per_record_and_need_five_trials():
    vals = [100.0, 100.1, 99.9, 100.0, 130.0]                                                    # one outlier trial per process does not move its median
    base = [wall_rec(BASE_COMMIT, {WALL_KEY: vals}, "2026-11-10T10:0%d:00Z" % (2 * i)) for i in range(3)]
    cand = [wall_rec("abc1234", {WALL_KEY: vals}, "2026-11-10T10:0%d:00Z" % (2 * i + 1)) for i in range(3)]
    res = br.compare(base, cand, metric="gpu_graph_us")
    assert res["verdict"] == "PASS" and res["rows"][0]["n_base"] == 3 and res["rows"][0]["base_spread_pct"] == 0.0 and res["rows"][0]["base_median"] == 100.0
    few = [wall_rec(BASE_COMMIT, {WALL_KEY: [100.0] * 4}, "2026-11-10T10:0%d:00Z" % (2 * i)) for i in range(3)]
    res = br.compare(few, cand, metric="gpu_graph_us")
    assert res["verdict"] == "INVALID" and any("at least 5 are required" in p for p in res["problems"])
    pooled = [dict(r, meta=dict(r["meta"], sample_stat="pooled")) for r in base]
    res = br.compare(pooled, cand, metric="gpu_graph_us")
    assert res["verdict"] == "INVALID" and any("mix sample statistics" in p for p in res["problems"])
    assert br.compare(base[:2], cand[:2], metric="gpu_graph_us")["verdict"] == "INVALID"           # at least three record files per build
    odd = [dict(r, meta=dict(r["meta"], sample_stat="mean")) for r in base]
    assert br.compare(odd, odd, metric="gpu_graph_us")["verdict"] == "INVALID"


def test_a_metric_that_is_zero_or_absent_in_a_cell_is_invalid_not_a_failure():
    base, cand = wall_sets([100.0, 100.05, 99.95], [100.0, 100.05, 99.95])
    res = br.compare(base, cand)                                                                  # the default metric k2_stage_us: a decode call has no K2 stage
    assert res["verdict"] == "INVALID" and res["rows"] == [] and any("zero in cell" in p and "kind decode" in p for p in res["problems"])
    d = tempfile.mkdtemp()
    assert compare_bench.main(["--base"] + write_records(d, "a", base) + ["--cand"] + write_records(d, "b", cand)) == 2
    assert compare_bench.main(["--base"] + write_records(d, "a", base) + ["--cand"] + write_records(d, "b", cand) + ["--metric", "k2_main_us"]) == 2
    pre_b, pre_c = interleaved([100.0, 100.1, 99.9], [100.0, 100.1, 99.9])                       # prefill records carry no wall metric at all
    assert br.compare(pre_b, pre_c, metric="gpu_graph_us")["verdict"] == "INVALID"
    stripped = [json.loads(json.dumps(r)) for r in cand]                                          # the candidate lost the metric in this cell only
    for r in stripped:
        for cell in r["cells"]:
            for s in cell["samples"]:
                del s["gpu_graph_us"]
    res = br.compare(base, stripped, metric="gpu_graph_us")
    assert res["verdict"] == "INVALID" and any("absent in cell" in p and "candidate" in p for p in res["problems"])
    gone = [json.loads(json.dumps(r)) for r in cand]                                              # a derived metric that vanished in the candidate is not a speed-up
    for r in gone:
        for cell in r["cells"]:
            for s in cell["samples"]:
                s["total_us"] = 0.0
    assert br.compare(base, gone, metric="total_us")["verdict"] == "INVALID"


def test_rerun_records_override_the_cells_they_contain():
    k1, k2 = br.cell_key("decode", N=1, H=12, regime="lockstep"), br.cell_key("decode", N=4, H=12, regime="lockstep")
    base, cand = wall_sets([100.0, 100.45, 100.9], [100.15, 100.6, 101.05], keys=(k1, k2))       # both cells are noise-limited in the first campaign
    rbase, rcand = wall_sets([100.0, 100.05, 99.95], [100.1, 100.15, 100.05], keys=(k1,), hour="11")
    res = br.compare(base, cand, metric="gpu_graph_us", baseline_unstable="widen", base_rerun=rbase, cand_rerun=rcand)
    rows = {r["key"]: r for r in res["rows"]}
    assert res["verdict"] == "PASS" and rows[k1]["judged_on"] == "rerun" and not rows[k1]["noise_limited"] and "judged on the re-run" in rows[k1]["note"]
    assert rows[k2]["judged_on"] == "first" and rows[k2]["noise_limited"]
    assert br.compare(base, cand, metric="gpu_graph_us", baseline_unstable="widen", base_rerun=rbase)["verdict"] == "INVALID"      # one side only
    _, rcand_other = wall_sets([100.0, 100.05, 99.95], [100.1, 100.15, 100.05], keys=(k2,), hour="11")
    res = br.compare(base, cand, metric="gpu_graph_us", baseline_unstable="widen", base_rerun=rbase, cand_rerun=rcand_other)
    assert res["verdict"] == "INVALID" and any("one side only" in p for p in res["problems"])
    few = rbase[:2]
    assert br.compare(base, cand, metric="gpu_graph_us", baseline_unstable="widen", base_rerun=few, cand_rerun=rcand[:2])["verdict"] == "INVALID"


def test_classify_rows_reads_slower_faster_and_noise_limited_cells():
    rows = [{"key": "a", "median_delta_pct": 2.0, "noise_limited": False}, {"key": "b", "median_delta_pct": -2.0, "noise_limited": False},
            {"key": "c", "median_delta_pct": -2.0, "noise_limited": True}, {"key": "d", "median_delta_pct": 0.5, "noise_limited": False},
            {"key": "e", "median_delta_pct": -1.0, "noise_limited": False}, {"key": "f", "median_delta_pct": 1.0, "noise_limited": False},
            {"key": "g", "median_delta_pct": 3.0, "noise_limited": True}]
    out = br.classify_rows({"tol_pct": 1.0, "rows": rows})
    assert out["slower"] == ["a", "g"] and out["faster"] == ["b", "e"] and out["noise_limited"] == ["c", "g"] and out["neutral"] == ["d", "f"]


def test_compare_bench_cli_widen_rerun_and_the_json_classification():
    d = tempfile.mkdtemp()
    base, cand = wall_sets([100.0, 100.45, 100.9], [97.0, 97.4, 97.8])                            # unstable and about 3 % faster
    rbase, rcand = wall_sets([100.0, 100.05, 99.95], [97.0, 97.05, 96.95], hour="11")
    out = os.path.join(d, "out.json")
    common = ["--base"] + write_records(d, "a", base) + ["--cand"] + write_records(d, "b", cand) + ["--metric", "gpu_graph_us"]
    assert compare_bench.main(common) == 2                                                        # invalid: unstable baseline
    assert compare_bench.main(common + ["--unstable-baseline", "widen", "--json", out]) == 0
    with open(out) as f:
        res = json.load(f)[0]
    assert res["classification"]["noise_limited"] == [WALL_KEY] and res["classification"]["faster"] == []   # noise-limited: not the evidence of a gain
    assert compare_bench.main(common + ["--unstable-baseline", "widen", "--rerun-base"] + write_records(d, "ra", rbase) +
                              ["--rerun-cand"] + write_records(d, "rb", rcand) + ["--json", out]) == 0
    with open(out) as f:
        res = json.load(f)[0]
    assert res["classification"]["faster"] == [WALL_KEY] and res["rows"][0]["judged_on"] == "rerun" and res["classification"]["noise_limited"] == []
    assert compare_bench.main(common + ["--unstable-baseline", "widen", "--rerun-base"] + write_records(d, "ra", rbase)) == 2


# ---- the decode wall-time benchmark: cells, calls per trial, assertions (the measurement itself needs a GPU) ------------------------------------------

def test_decode_cells_cover_matrix_d_the_regimes_and_the_probe_positions():
    specs = bd.cell_specs()
    keys = [s["key"] for s in specs]
    assert len(keys) == len(set(keys)) and all(k.startswith("decode;") and k.endswith(";") for k in keys)
    for regime in ("lockstep", "staggered", "cold"):
        assert [(s["N"], s["H"]) for s in specs if s["regime"] == regime] == [(1, 12), (4, 12), (16, 12), (64, 12), (128, 12), (256, 12)]
    probe = [s for s in specs if s["regime"] == "probe"]
    assert sorted({s["N"] for s in probe}) == [1, 64] and sorted({s["pos"] for s in probe}) == [0, 1, 15, 16, 31, 32, 47, 48, 62, 63] and len(probe) == 20
    assert len(specs) == 3 * 6 + 20
    shapes = [(s["N"], s["H"]) for s in specs]
    assert shapes == sorted(shapes)                                       # grouped by shape: one input / state / graph set per shape
    assert [s["regime"] for s in specs if (s["N"], s["H"]) == (1, 12)][:3] == ["lockstep", "staggered", "cold"]
    only = bd.cell_specs(shapes=[(4, 3)], regimes=["lockstep", "probe"], probe_shapes=[(2, 3)], positions=[5])
    assert [s["key"] for s in only] == ["decode;N=2;H=3;regime=probe;pos=5;", "decode;N=4;H=3;regime=lockstep;"]
    with pytest.raises(ValueError):
        bd.cell_specs(regimes=["warm"])


def test_decode_cell_filter_uses_the_delimited_keys():
    specs = bd.cell_specs()
    n1 = bd.select_cells(specs, ["N=1;"])
    assert n1 and {s["N"] for s in n1} == {1} and not any("N=16;" in s["key"] for s in n1)
    assert {s["N"] for s in bd.select_cells(specs, ["N=16;"])} == {16}
    assert len(bd.select_cells(specs, ["regime=probe;pos=63;"])) == 2 and len(bd.select_cells(specs, None)) == len(specs) == len(bd.select_cells(specs, []))
    assert bd.select_cells(specs, ["N=2;"]) == []


def test_decode_calls_per_trial_cover_whole_position_cycles_and_scale_for_the_rerun():
    opts = dict(bd.DEFAULTS)
    assert [bd.calls_per_trial(r, opts) for r in bd.REGIMES] == [1536, 1536, 256, 256]
    assert bd.calls_per_trial("lockstep", dict(opts, calls=100)) == 128                       # rounded up to whole 64-position cycles
    assert bd.calls_per_trial("probe", dict(opts, probe_calls=100)) == 100                     # probe calls are independent of the position cycle
    scaled = dict(opts, calls_scale=3)
    assert [bd.calls_per_trial(r, scaled) for r in bd.REGIMES] == [3 * 1536, 3 * 1536, 3 * 256, 3 * 256]


def test_decode_expected_kernel_set_is_asserted_and_trials_carry_the_wall_metrics():
    assert bd.check_expected_kernels({"plan_kernel", "rows_kernel"}, ["rows_kernel", "plan_kernel"]) is None
    assert bd.check_expected_kernels({"anything"}, None) is None
    assert "launched ['rows_kernel']" in bd.check_expected_kernels({"rows_kernel"}, ["plan_kernel", "rows_kernel"])
    assert "expected ['rows_kernel']" in bd.check_expected_kernels({"plan_kernel", "rows_kernel"}, ["rows_kernel"])
    assert bd.trial_wall(14.0, 13.0, 18.0, 9.0) == {"gpu_eager_us": 14.0, "gpu_graph_us": 13.0, "cpu_eager_us": 18.0, "cpu_graph_us": 9.0}
    assert bd.trial_wall(14.0, 13.0) == {"gpu_eager_us": 14.0, "gpu_graph_us": 13.0}
    cell = wall_cell(WALL_KEY, [14.0, 14.2, 14.1, 14.0, 14.3])
    cell["args"]["calls_per_trial"] = 1536
    line = bd.cell_line({"key": WALL_KEY}, cell)
    assert WALL_KEY in line and "gpu_graph" in line and "total" in line and "5 trials, 1536 calls" in line


def test_decode_run_meta_carries_the_fields_of_a_usable_record():
    class FakeTorch:
        __version__ = "2.14.0"

        class version:
            cuda = "13.4"

    a = argparse.Namespace(impl=0, regimes=list(bd.REGIMES), cells=None, label="x")
    device = {"name": "NVIDIA B200", "sm_count": 148, "cc": "10.0", "l2_bytes": 126 * 2 ** 20, "eviction_bytes": 2 * 126 * 2 ** 20 + 2 ** 20}
    meta = bd.run_meta(a, "2026-11-10T10:00:00Z", dict(bd.DEFAULTS), device, FakeTorch)
    for field in br.REQUIRED_META + ("tree", "impl", "regimes", "trials", "calls_per_trial", "profile_calls", "sample_stat", "calls_scale", "host", "timestamp"):
        assert field in meta, field
    assert meta["sample_stat"] == br.PROCESS_MEDIAN and meta["eviction_bytes"] >= 2 * meta["l2_cache_bytes"] and meta["trials"] >= br.MIN_TRIALS
    assert meta["calls_per_trial"] == {"lockstep": 1536, "staggered": 1536, "cold": 256, "probe": 256}
    meta.update(commit=BASE_COMMIT, dirty=False)                          # the identity of the tree running the test is not what is checked here
    rec = {"schema": br.SCHEMA, "meta": meta, "cells": [wall_cell(WALL_KEY, [14.0] * 5)]}
    assert br.validate_record(rec, "baseline", expect_commit=BASE_COMMIT) == []
    rec["cells"] = [wall_cell(WALL_KEY, [14.0] * 4)]                      # fewer trials than a per-process median needs
    assert any("at least 5 are required" in p for p in br.validate_record(rec, "baseline"))
