"""Self-tests of the benchmark tooling (benchmarks/bench_records.py, compare_bench.py, compare_binaries.py) and of the corpus V definitions.

Plain Python: no GPU and no torch are needed (run them without the package conftest, e.g. ``pytest --noconftest tests/cfbi/test_bench_tools.py``,
where torch is not installed). The negative cases show that the gates can fail: a wrong, dirty or unstable baseline is refused, a 1 % slower
candidate fails the 0.5 % protection gate, and a build with one more register or a spill is reported as different.
"""
import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "benchmarks", _ROOT / "tests" / "cfbi"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bench_decode as bd  # noqa: E402
import bench_layer_chain as lc  # noqa: E402
import bench_records as br  # noqa: E402
import codegen_gate as cg  # noqa: E402
import compare_bench  # noqa: E402
import compare_binaries as cb  # noqa: E402
import suite_record as sr  # noqa: E402
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


def cg_ptxas(functions, arch="sm_100a"):
    """``ptxas -v`` lines of a build: ``{function: (stack, spill stores, spill loads, registers)}``."""
    out = []
    for name, (stack, stores, loads, regs) in functions.items():
        out += ["ptxas info    : Compiling entry function '%s' for '%s'" % (name, arch), "ptxas info    : Function properties for %s" % name,
                "    %d bytes stack frame, %d bytes spill stores, %d bytes spill loads" % (stack, stores, loads),
                "ptxas info    : Used %d registers, used 1 barriers, 4096 bytes smem" % regs]
    return "\n".join(out) + "\n"


def cg_elf(functions, arch="sm_100a"):
    """``cuobjdump -elf`` attribute blocks: ``{function: (maxreg, max threads or None)}``."""
    out = ["", "Fatbin elf code:", "================", "arch = %s" % arch, ""]
    for name, (maxreg, threads) in functions.items():
        out += [".nv.info.%s" % name, "\t<0x4>", "\tAttribute:\tEIATTR_MAXREG_COUNT", "\tFormat:\tEIFMT_HVAL", "\tValue:\t0x%x" % maxreg]
        if threads:
            out += ["\t<0x12>", "\tAttribute:\tEIATTR_MAX_THREADS", "\tFormat:\tEIFMT_SVAL", "\tValue:\t0x%x 0x1 0x1 " % threads]
        out.append("")
    return "\n".join(out) + "\n"


def cg_table(code=CODE, **kw):
    """A complete table (resources, SASS, spills, launch bounds) of two kernels."""
    return cg.kernel_table(cg_regs({ROWS: (128, 0), K1: (100, 0)}), cg_sass({ROWS: code, K1: ["EXIT"]}, **kw),
                           cg_ptxas({ROWS: (0, 0, 0, 128), K1: (0, 0, 0, 100)}), cg_elf({ROWS: (128, 256), K1: (128, 256)}))


def cg_hash(tag):
    """A SHA-256 hex digest of ``tag``: the kernel records hold digests, and an incomplete or malformed one is rejected by the gate."""
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def cg_kernel(reg=128, stack=0, spill=(0, 0), sass="h0", ins=100, shared=0, local=0, maxthr=256, maxreg=128):
    return {"name": "n", "unit": None, "reg": reg, "stack": stack, "shared": shared, "local": local, "instructions": ins, "sass_hash": cg_hash(sass),
            "sass_hash_strict": cg_hash(sass), "spill_stores": spill[0], "spill_loads": spill[1], "max_threads": maxthr, "maxreg": maxreg}


def cg_dump(kernels, label="toolchain-A", defines=(), content="tree-1"):
    return {"schema": cg.SCHEMA,
            "meta": {"toolchain": {"source": "nvcc", "label": label, "nvcc": "nvcc 13.1.115", "host_compiler": "g++ 14.3.0", "cuobjdump": "cuobjdump 13.1.115",
                                   "ptxas": "ptxas 13.1.115", "flags": list(cg.PRODUCTION_FLAGS), "std": cg.STD, "archs": sorted(kernels), "src_hash": cg.FIXED_SRC_HASH},
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
    changed = cg_edit(base, "sm_100a", "cfbi::k1_kernel<Li16E>", sass_hash=cg_hash("k1-new"))
    assert cg.compare_dumps(base, changed)["exit"] == 1
    for entry in ("cfbi::k1_kernel<Li16E>", "cfbi::k1_kernel"):          # a changed kernel is tolerated only when it is on the allow-list (key or family)
        rep = cg.compare_dumps(base, changed, {"aliases": {}, "may_differ": [entry]})
        assert rep["exit"] == 0 and len(rep["allowed"]) == 1, entry
    wide = cg_edit(base, "sm_100a", "cfbi::k2_kernel<Li16E>", reg=180, sass_hash=cg_hash("k2-new"))        # a tolerated K2 change is not held to the row engine's register cap
    rep = cg.compare_dumps(base, wide, {"aliases": {}, "may_differ": ["cfbi::k2_kernel"]})
    assert rep["exit"] == 0 and rep["gate_r"]["failures"] == [] and len(rep["allowed"]) == 1
    rows_bad = cg_edit(base, "sm_100a", "cfbi::rows_kernel", reg=129, sass_hash=cg_hash("rows-new"))      # a tolerated row-engine change still has to pass the resource gate
    rep = cg.compare_dumps(base, rows_bad, {"aliases": {}, "may_differ": ["cfbi::rows_kernel"]})
    assert rep["exit"] == 1 and any("129 registers" in f for f in rep["gate_r"]["failures"]) and len(rep["allowed"]) == 1
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
        rows_ok["kernels"][arch]["cfbi::rows_kernel"].update(sass_hash=cg_hash("rows-new"), instructions=90)
    assert cg.compare_dumps(base, rows_ok)["exit"] == 1                  # identity mode: any change is a difference
    rep = cg.compare_dumps(base, rows_ok, mode="change")
    assert rep["exit"] == 0 and len(rep["allowed"]) == 2 and len(rep["gate_r"]["checked"]) >= 2
    for fields, text in (({"reg": 129, "maxreg": 255}, "129 registers"), ({"stack": 113}, "stack 113 B above"), ({"spill_stores": 5}, "spill stores 5 B above"),
                         ({"maxreg": 255}, "launch-bound attribute maxreg 255")):
        rep = cg.compare_dumps(base, cg_edit(rows_ok, "sm_100a", "cfbi::rows_kernel", **fields), mode="change")
        assert rep["exit"] == 1 and any(text in f for f in rep["gate_r"]["failures"]), (fields, rep["gate_r"])
    for key in ("cfbi::k1_kernel<Li16E>", "cfbi::ref::rows_kernel", "cfbi::plan_kernel"):      # everything else, the frozen reference included, stays identical
        rep = cg.compare_dumps(base, cg_edit(rows_ok, "sm_90a", key, sass_hash=cg_hash("changed")), mode="change")
        assert rep["exit"] == 1 and any(key in d for d in rep["differences"]), key
    assert cg.compare_dumps(base, base, mode="change")["gate_r"]["checked"]                   # the row engine is gated even when it equals the base
    rep = cg.compare_dumps(base, rows_ok, mode="change", prev=rows_ok)
    assert any("unchanged against the previous accepted state" in n for n in rep["notes"])
    with pytest.raises(cg.Unusable):
        cg.compare_dumps(base, rows_ok, mode="bogus")


def test_codegen_refuses_dumps_of_different_toolchains_but_allows_different_variant_defines():
    base = cg_base()
    for field, value in (("nvcc", "nvcc 13.4"), ("host_compiler", "g++ 8.5"), ("cuobjdump", "cuobjdump 13.4"), ("ptxas", "ptxas 13.4"), ("flags", ["-O2"]),
                         ("label", "another toolchain"), ("source", "files"), ("std", "c++17"), ("src_hash", "0x1ull")):
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
    Path(d, "csrc", "cfbi", "x.cuh").write_text("a")
    Path(d, "flash_kda", "cfbi", "_ext.py").write_text('_SOURCES = ("b.cpp", "u1.cu", "u2.cu")\nNVCC_FLAGS = ["-O3", "-lineinfo"]\n')
    assert cg.units_of(d) == ["csrc/cfbi/u1.cu", "csrc/cfbi/u2.cu"] and cg.ext_flags_of(d) == ["-O3", "-lineinfo"]
    h = cg.tree_content_hash(d)
    Path(d, "README").write_text("outside the build inputs")
    assert cg.tree_content_hash(d) == h
    Path(d, "csrc", "cfbi", "x.cuh").write_text("b")
    assert cg.tree_content_hash(d) != h
    empty = tempfile.mkdtemp()
    assert cg.units_of(empty) == list(cg.DEFAULT_UNITS) and cg.ext_flags_of(empty) is None
    Path(empty, "SNAPSHOT_COMMIT").write_text("abc123\n")
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


def cg_manifest(**over):
    """A complete toolchain manifest of a production build (the values are made up; the real one is derived from the build log and the tool versions)."""
    m = {"schema": cg.MANIFEST_SCHEMA, "label": "container toolchain", "nvcc": "Cuda compilation tools, release 13.4, V13.4.59", "host_compiler": "c++ (Ubuntu 13.3.0) 13.3.0",
         "cuobjdump": "Cuda compilation tools, release 13.4, V13.4.49", "ptxas": "Cuda compilation tools, release 13.4, V13.4.59", "std": "c++20",
         "flags": list(cg.PRODUCTION_FLAGS) + ["-gencode=arch=compute_100a,code=sm_100a"], "archs": ["sm_100a"], "variant_defines": [], "src_hash": "0x15ca577bf4da4848ull",
         "build_log": {"sha256": cg_hash("build log"), "nvcc_compile_lines": 3, "units": ["a.cu", "b.cu", "c.cu"]}}
    m.update(over)
    return m


def cg_build_log(m, ptxas_text):
    """The verbose build log that the parameters of manifest ``m`` describe: one nvcc compile line per unit, then the ``ptxas -v`` lines."""
    flags = []
    for f in m["flags"]:
        flags += ["--compiler-options", "'%s'" % f[len("--compiler-options "):]] if f.startswith("--compiler-options ") else [f]
    tail = ["-DCFBI_SRC_HASH=%s" % m["src_hash"]] + ["-D%s" % d for d in m["variant_defines"]] + ["-std=%s" % m["std"]]
    units = m["build_log"]["units"]
    lines = ["ninja: build started"]
    for i, u in enumerate(units):
        lines.append("[%d/%d] /usr/local/cuda/bin/nvcc -MD -MF %s.o.d -DTORCH_EXTENSION_NAME=x -I/work/csrc %s %s -c /work/csrc/cfbi/%s -o %s.o" % (i + 1, len(units) + 1, u, " ".join(flags), " ".join(tail), u, u))
    return "\n".join(lines) + "\n" + ptxas_text


class CgFiles:
    """The text inputs of a dump of another toolchain, written to a temporary directory: resources, SASS, the verbose build log (with the ptxas lines), ELF attributes
    and the manifest. With ``bind`` (default) the build log is generated from the manifest's parameters and the manifest's ``build_log`` record is made to describe it,
    so the two belong together; ``bind=False`` writes the manifest as given and the plain ptxas text as the log (for manifests that are meant to be refused)."""

    def __init__(self, code=CODE, manifest=None, regs=None, ptxas=None, elf=None, sass=None, bind=True):
        self.dir = tempfile.mkdtemp()
        fns = {ROWS: (128, 112), K1: (100, 0)}
        ptxas_text = ptxas if ptxas is not None else cg_ptxas({ROWS: (112, 4, 4, 128), K1: (0, 0, 0, 100)})
        self.manifest = manifest if manifest is not None else cg_manifest()
        log = ptxas_text
        if bind:
            log = cg_build_log(self.manifest, ptxas_text)
            self.manifest = dict(self.manifest, build_log={"sha256": hashlib.sha256(log.encode("utf-8")).hexdigest(), "nvcc_compile_lines": len(self.manifest["build_log"]["units"]),
                                                          "units": sorted(self.manifest["build_log"]["units"])})
        self.paths = {"regs": self.write("regs.txt", regs if regs is not None else cg_regs(fns)),
                      "sass": self.write("sass.txt", sass if sass is not None else cg_sass({ROWS: code, K1: ["EXIT"]})),
                      "ptxas": self.write("ptxas.txt", log),
                      "elf": self.write("elf.txt", elf if elf is not None else cg_elf({ROWS: (128, 256), K1: (128, 256)}))}
        self.paths["manifest"] = self.write("manifest.json", json.dumps(self.manifest))

    def write(self, name, text):
        path = os.path.join(self.dir, name)
        with open(path, "w") as f:
            f.write(text)
        return path

    def dump(self):
        p = self.paths
        return cg.dump_from_files(p["regs"], p["sass"], p["manifest"], p["elf"], p["ptxas"])


def test_codegen_dump_from_toolchain_text_files_and_the_operand_only_control():
    base = CgFiles().dump()
    rows = base["kernels"]["sm_100a"]["cfbi::rows_kernel"]
    assert rows["reg"] == 128 and rows["spill_stores"] == 4 and rows["instructions"] == len(CODE)            # the spill figures come from the ptxas lines
    assert rows["max_threads"] == 256 and rows["maxreg"] == 128                                              # the launch bounds come from the ELF attributes
    assert base["meta"]["toolchain"]["host_compiler"] == "c++ (Ubuntu 13.3.0) 13.3.0" and base["meta"]["toolchain"]["flags"] == cg_manifest()["flags"]
    assert cg.compare_dumps(base, base)["exit"] == 0
    for idx, new in ((5, "FFMA R8, R2, R3, R4"), (2, "IMAD.WIDE.U32 R2, R5, 0x24, R2")):                  # operand only, immediate only: the gate must fail
        code = list(CODE)
        code[idx] = new
        assert cg.compare_dumps(base, CgFiles(code=code).dump())["exit"] == 1, new
    with pytest.raises(cg.Unusable):                                                                         # a dump built by nvcc here is another toolchain
        cg.compare_dumps(base, cg_base())


def test_codegen_every_missing_figure_of_a_kernel_is_unusable_never_an_equality():
    base = cg_dump(cg_table())
    cg.check_dump(base)
    for field, bad in (("instructions", None), ("instructions", 0), ("sass_hash", None), ("sass_hash", "h0"), ("sass_hash", "A" * 64), ("sass_hash_strict", None),
                       ("reg", None), ("stack", None), ("shared", None), ("local", None), ("spill_stores", None), ("spill_loads", None), ("spill_stores", -1), ("reg", True)):
        broken = cg_edit(base, "sm_100a", "cfbi::k1_kernel<Li16E>", **{field: bad})
        with pytest.raises(cg.Unusable):
            cg.check_dump(broken)
        for mode in ("identity", "change"):
            with pytest.raises(cg.Unusable):
                cg.compare_dumps(base, broken, mode=mode)
            with pytest.raises(cg.Unusable):
                cg.compare_dumps(broken, broken, mode=mode)                                                  # the same hole on both sides is not an equality either
    d = tempfile.mkdtemp()
    paths = {}
    for name, dump in (("ok", base), ("hole", cg_edit(base, "sm_100a", "cfbi::k1_kernel<Li16E>", spill_loads=None))):
        paths[name] = os.path.join(d, name + ".json")
        with open(paths[name], "w") as f:
            json.dump(dump, f)
    assert cg.main(["compare", "--base", paths["ok"], "--cand", paths["ok"]]) == 0
    assert cg.main(["compare", "--base", paths["ok"], "--cand", paths["hole"]]) == 2
    assert cg.main(["compare", "--base", paths["ok"], "--cand", paths["hole"], "--mode", "change"]) == 2


def test_codegen_review_counterexamples_are_exit_2():
    """The two reproductions of the round-0 review: every SASS / spill / launch-bound figure cleared on both sides, and one spill figure of the candidate row engine."""
    base = cg_dump(cg_table())
    cleared = cg_copy(base)
    for rec in cleared["kernels"]["sm_100a"].values():
        rec.update(instructions=None, sass_hash=None, sass_hash_strict=None, spill_stores=None, spill_loads=None, max_threads=None, maxreg=None)
    with pytest.raises(cg.Unusable):
        cg.compare_dumps(cleared, cleared, mode="change")
    with pytest.raises(cg.Unusable):
        cg.compare_dumps(base, cg_edit(base, "sm_100a", "cfbi::rows_kernel", spill_stores=None), mode="change")


def test_codegen_inconsistent_or_truncated_text_inputs_are_unusable():
    fns = {ROWS: (128, 0), K1: (100, 0)}
    ptx = {ROWS: (0, 0, 0, 128), K1: (0, 0, 0, 100)}
    sass = cg_sass({ROWS: CODE, K1: ["EXIT"]})
    assert sum(len(v) for v in cg.kernel_table(cg_regs(fns), sass, cg_ptxas(ptx), cg_elf({ROWS: (128, 256)})).values()) == 2
    with pytest.raises(cg.Unusable):                                                                         # a function that the SASS dump lacks
        cg.kernel_table(cg_regs(fns), cg_sass({ROWS: CODE}), cg_ptxas(ptx), cg_elf({ROWS: (128, 256)}))
    with pytest.raises(cg.Unusable):                                                                         # a SASS function without a resource entry
        cg.kernel_table(cg_regs({ROWS: (128, 0)}), sass, cg_ptxas(ptx), cg_elf({ROWS: (128, 256)}))
    with pytest.raises(cg.Unusable):                                                                         # a ptxas log of another build
        cg.kernel_table(cg_regs(fns), sass, cg_ptxas(dict(ptx, **{"_ZN4cfbi9other_kernelEv": (0, 0, 0, 8)})), cg_elf({ROWS: (128, 256)}))
    for text in ("", "Fatbin elf code:\narch = sm_100a\n"):                                                   # empty dumps
        with pytest.raises(cg.Unusable):
            cg.kernel_table(text, sass, cg_ptxas(ptx), None)
        with pytest.raises(cg.Unusable):
            cg.kernel_table(cg_regs(fns), text, cg_ptxas(ptx), None)
    truncated = cg.kernel_table(cg_regs(fns), sass, cg_ptxas({ROWS: (0, 0, 0, 128)}), cg_elf({ROWS: (128, 256)}))     # the log stops before K1: its spills are unknown
    with pytest.raises(cg.Unusable):
        cg.check_dump(cg_dump(truncated))
    empty_body = cg.kernel_table(cg_regs(fns), cg_sass({ROWS: CODE, K1: []}), cg_ptxas(ptx), cg_elf({ROWS: (128, 256)}))   # a function with no instruction
    with pytest.raises(cg.Unusable):
        cg.check_dump(cg_dump(empty_body))
    no_elf = cg.kernel_table(cg_regs(fns), sass, cg_ptxas(ptx), None)                                      # no launch-bound data for the row engine
    with pytest.raises(cg.Unusable):
        cg.check_dump(cg_dump(no_elf))


def test_codegen_row_engine_family_and_new_kernels_need_launch_bound_and_reference_data():
    base = cg_base()
    changed = cg_copy(base)
    for arch in changed["kernels"]:
        changed["kernels"][arch]["cfbi::rows_kernel"].update(sass_hash=cg_hash("rows-new"), instructions=90)
    assert cg.compare_dumps(base, changed, mode="change")["exit"] == 0
    for field in ("max_threads", "maxreg"):
        for side in ("cand", "base"):
            b, c = cg_copy(base), cg_copy(changed)
            target = c if side == "cand" else b
            target["kernels"]["sm_100a"]["cfbi::rows_kernel"][field] = None
            with pytest.raises(cg.Unusable):                                                                 # the row engine's launch bounds cannot be checked
                cg.check_dump(target)
    added = cg_copy(base)                                                                                    # a new kernel needs the base row engine as reference
    added["kernels"]["sm_100a"]["cfbi::new_kernel"] = cg_kernel(reg=100, sass="new", maxreg=None, maxthr=None)
    assert cg.compare_dumps(base, added)["exit"] == 0                                                        # attributes are only required for the row-engine family
    no_ref = cg_copy(base)
    no_ref["kernels"]["sm_100a"].pop("cfbi::rows_kernel")
    with pytest.raises(cg.Unusable):
        cg.compare_dumps(no_ref, cg_edit(added, "sm_100a", "cfbi::new_kernel", reg=100))


def test_codegen_files_dump_needs_every_input_and_a_complete_manifest():
    f = CgFiles()
    p = f.paths
    for missing in range(5):
        args = [p["regs"], p["sass"], p["manifest"], p["elf"], p["ptxas"]]
        args[missing] = None
        with pytest.raises(cg.Unusable):
            cg.dump_from_files(*args)
    for field in ("label", "nvcc", "host_compiler", "cuobjdump", "ptxas", "std", "flags", "archs"):
        for bad in (None, "", []):
            m = cg_manifest()
            m[field] = bad
            with pytest.raises(cg.Unusable):
                CgFiles(manifest=m, bind=False).dump()
        m = cg_manifest()
        del m[field]
        with pytest.raises(cg.Unusable):
            CgFiles(manifest=m, bind=False).dump()
    for bad in ({"schema": "other"}, dict(cg_manifest(), build_log={"sha256": "x", "nvcc_compile_lines": 3}), dict(cg_manifest(), build_log=None), dict(cg_manifest(), flags=[1, 2])):
        with pytest.raises(cg.Unusable):
            CgFiles(manifest=bad, bind=False).dump()
    other_arch = cg_manifest(flags=[x for x in cg_manifest()["flags"] if not x.startswith("-gencode")] + ["-gencode=arch=compute_90a,code=sm_90a"], archs=["sm_90a"])
    with pytest.raises(cg.Unusable, match="architectures"):                                                  # a manifest of another architecture does not describe these dumps
        CgFiles(manifest=other_arch).dump()


def test_codegen_files_dump_binds_the_build_log_to_the_manifest():
    f = CgFiles()
    p = f.paths
    log = open(p["ptxas"]).read()
    assert " -O3 " in log and cg.dump_from_files(p["regs"], p["sass"], p["manifest"], p["elf"], p["ptxas"])["meta"]["build_log"]["sha256"] == f.manifest["build_log"]["sha256"]

    def attempt(manifest=None, log_text=None):
        """dump_from_files with the files of ``f`` but another manifest and / or another build log."""
        manifest = p["manifest"] if manifest is None else f.write("m2.json", json.dumps(manifest))
        log_path = p["ptxas"] if log_text is None else f.write("log2.txt", log_text)
        return cg.dump_from_files(p["regs"], p["sass"], manifest, p["elf"], log_path)

    with pytest.raises(cg.Unusable, match="not the log the manifest was made from"):                          # review counterexample: the real -O3 manifest, a log copy that says -O2
        attempt(log_text=log.replace(" -O3 ", " -O2 "))
    other_build = CgFiles(manifest=cg_manifest(variant_defines=["CFBI_NEG_CONTROL=1"]))                       # the log of another build of the same tree
    with pytest.raises(cg.Unusable, match="not the log the manifest was made from"):
        attempt(log_text=open(other_build.paths["ptxas"]).read())
    with pytest.raises(cg.Unusable, match="not the log the manifest was made from"):                          # one character of the log changed
        attempt(log_text=log + " ")
    m = f.manifest
    contradictions = (("flags", [x.replace("-O3", "-O2") for x in m["flags"]]), ("std", "c++17"), ("archs", ["sm_90a"]), ("src_hash", "0x1ull"),
                      ("variant_defines", ["CFBI_NEG_CONTROL=1"]))
    for field, value in contradictions:                                                                      # the true digest, but the manifest says something else than the commands do
        with pytest.raises(cg.Unusable, match="contradict the manifest.*%s" % field):
            attempt(manifest=dict(m, **{field: value}))
    for bad in (dict(m["build_log"], nvcc_compile_lines=2), dict(m["build_log"], units=["a.cu", "b.cu"]), dict(m["build_log"], units=["a.cu", "b.cu", "other.cu"])):
        with pytest.raises(cg.Unusable, match="contradict the manifest.*units"):
            attempt(manifest=dict(m, build_log=bad))
    plain = open(p["ptxas"]).read().split("ptxas info", 1)[1]                                                # a log without any compile line cannot belong to a manifest
    plain = "ptxas info" + plain
    with pytest.raises(cg.Unusable, match="no nvcc compile"):
        attempt(manifest=dict(m, build_log=dict(m["build_log"], sha256=hashlib.sha256(plain.encode("utf-8")).hexdigest())), log_text=plain)
    out = os.path.join(f.dir, "o.json")
    args = ["dump-files", "--manifest", p["manifest"], "--regs", p["regs"], "--sass", p["sass"], "--elf", p["elf"]]
    assert cg.main(args + ["--ptxas-log", p["ptxas"], "--out", out]) == 0                                    # the command line: the matching log passes, the changed one is exit 2
    assert cg.main(args + ["--ptxas-log", f.write("o2.log", log.replace(" -O3 ", " -O2 ")), "--out", out]) == 2


def test_codegen_same_label_with_a_different_toolchain_is_refused_and_a_variant_or_source_change_is_not():
    base = CgFiles().dump()
    assert cg.compare_dumps(base, CgFiles().dump())["exit"] == 0
    for field, value in (("host_compiler", "c++ (Ubuntu 11.4.0) 11.4.0"), ("nvcc", "Cuda compilation tools, release 12.9, V12.9.86"), ("cuobjdump", "Cuda compilation tools, release 13.4, V13.4.1"),
                         ("ptxas", "Cuda compilation tools, release 13.4, V13.4.1"), ("std", "c++17"), ("flags", cg_manifest()["flags"] + ["-O0"]),
                         ("flags", [x for x in cg_manifest()["flags"] if x != "-fmad=false"] + ["-fmad=true"])):
        other = CgFiles(manifest=cg_manifest(**{field: value})).dump()
        assert other["meta"]["toolchain"]["label"] == base["meta"]["toolchain"]["label"]                    # the label alone proves nothing
        with pytest.raises(cg.Unusable):
            cg.compare_dumps(base, other)
    d = tempfile.mkdtemp()
    paths = []
    for i, m in enumerate((cg_manifest(), cg_manifest(host_compiler="c++ (Ubuntu 11.4.0) 11.4.0"))):
        paths.append(os.path.join(d, "d%d.json" % i))
        with open(paths[-1], "w") as fh:
            json.dump(CgFiles(manifest=m).dump(), fh)
    assert cg.main(["compare", "--base", paths[0], "--cand", paths[0]]) == 0
    assert cg.main(["compare", "--base", paths[0], "--cand", paths[1]]) == 2
    variant = CgFiles(manifest=cg_manifest(variant_defines=["CFBI_NEG_CONTROL=1"], src_hash="0xdeadbeefull")).dump()     # variants and the source hash are not the toolchain
    rep = cg.compare_dumps(base, variant)
    assert rep["exit"] == 0 and variant["meta"]["variant"]["defines"] == ["CFBI_NEG_CONTROL=1"] and variant["meta"]["tree"]["src_hash"] == "0xdeadbeefull"


JIT_NVCC = ("[%d/5] /usr/local/cuda/bin/nvcc -MD -MF %s.cuda.o.d -DTORCH_EXTENSION_NAME=flash_kda_cfbi_C_jit -DTORCH_API_INCLUDE_EXTENSION_H "
            "-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr "
            "-isystem /usr/lib/python3/torch/include -I/work/csrc -gencode=arch=compute_100a,code=sm_100a --compiler-options '-fPIC' -O3 -fmad=false -ftz=false "
            "-prec-div=true -prec-sqrt=true --expt-relaxed-constexpr -lineinfo --ptxas-options=-v %s -std=c++20 -c /work/csrc/cfbi/%s.cu -o %s.cuda.o")
TOOL_VERSIONS = """TOOLCHAIN python 3.12.3
TOOLCHAIN torch 2.14.0 cuda 13.4
TOOLCHAIN nvcc: nvcc: NVIDIA (R) Cuda compiler driver | Copyright (c) 2005-2026 NVIDIA Corporation | Built on Mon_Aug_03 | Cuda compilation tools, release 13.4, V13.4.59 | Build cuda_13.4
TOOLCHAIN ptxas: ptxas: NVIDIA (R) Ptx optimizing assembler | Built on Mon_Aug_03 | Cuda compilation tools, release 13.4, V13.4.59 | Build cuda_13.4
TOOLCHAIN cuobjdump: cuobjdump: NVIDIA (R) fat binary listing tool | Built on Thu_Jul_16 | Cuda compilation tools, release 13.4, V13.4.49 | Build cuda_13.4
TOOLCHAIN host: c++ (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 | Copyright (C) 2023 Free Software Foundation, Inc.
"""


def jit_log(units=("cfbi_probe", "cfbi_k2_dyn", "cfbi_launch"), extra=("-DCFBI_SRC_HASH=0x15ca577bf4da4848ull",), tweak=None):
    lines = ["ninja: build started"]
    for i, u in enumerate(units):
        line = JIT_NVCC % (i + 1, u, " ".join(extra), u, u)
        lines.append(tweak(i, line) if tweak else line)
    lines += ["[4/5] c++ -MMD -MF cfbi_binding.o.d -DTORCH_EXTENSION_NAME=flash_kda_cfbi_C_jit -O3 -std=c++20 -c /work/csrc/cfbi/cfbi_binding.cpp -o cfbi_binding.o",
              "ptxas info    : Compiling entry function '%s' for 'sm_100a'" % ROWS]
    return "\n".join(lines) + "\n"


def test_codegen_manifest_comes_from_the_build_log_and_the_probed_tool_versions():
    d = tempfile.mkdtemp()

    def write(name, text):
        path = os.path.join(d, name)
        with open(path, "w") as f:
            f.write(text)
        return path

    versions = write("versions.txt", TOOL_VERSIONS)
    m = cg.manifest_from_build(write("build.log", jit_log()), versions, "container toolchain")
    cg.validate_manifest(m)
    assert m["nvcc"] == "Cuda compilation tools, release 13.4, V13.4.59" and m["cuobjdump"].endswith("V13.4.49") and m["ptxas"].endswith("V13.4.59")
    assert m["host_compiler"] == "c++ (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0" and m["std"] == "c++20" and m["archs"] == ["sm_100a"]
    assert all(f in m["flags"] for f in cg.PRODUCTION_FLAGS) and "-D__CUDA_NO_HALF_OPERATORS__" in m["flags"] and "--compiler-options -fPIC" in m["flags"]
    assert not any(f.startswith(("-I", "-MF", "-MD", "-c", "-o", "-DTORCH_EXTENSION_NAME", "-DCFBI_SRC_HASH")) or f.endswith(".cu") for f in m["flags"]), m["flags"]
    assert m["src_hash"] == "0x15ca577bf4da4848ull" and m["variant_defines"] == [] and m["build_log"]["nvcc_compile_lines"] == 3
    assert m["build_log"]["units"] == ["cfbi_k2_dyn.cu", "cfbi_launch.cu", "cfbi_probe.cu"] and len(m["build_log"]["sha256"]) == 64
    v = cg.manifest_from_build(write("variant.log", jit_log(extra=("-DCFBI_SRC_HASH=0x1ull", "-DCFBI_NEG_CONTROL=1"))), versions, "container toolchain")
    assert v["variant_defines"] == ["CFBI_NEG_CONTROL=1"] and v["flags"] == m["flags"] and v["src_hash"] == "0x1ull"     # variant and source hash are kept out of the flags
    with pytest.raises(cg.Unusable):                                                                         # the units of one build must share their parameters
        cg.manifest_from_build(write("mixed.log", jit_log(tweak=lambda i, line: line.replace("-lineinfo", "") if i == 1 else line)), versions, "x")
    with pytest.raises(cg.Unusable):                                                                         # not built with the production flags
        cg.manifest_from_build(write("nonprod.log", jit_log(tweak=lambda i, line: line.replace("-fmad=false", "-fmad=true"))), versions, "x")
    assert cg.manifest_from_build(write("nonprod.log", jit_log(tweak=lambda i, line: line.replace("-fmad=false", "-fmad=true"))), versions, "x", allow_non_production_flags=True)
    with pytest.raises(cg.Unusable):                                                                         # the log of a cached build has no compile line
        cg.manifest_from_build(write("cached.log", "ninja: no work to do.\n"), versions, "x")
    with pytest.raises(cg.Unusable):                                                                         # no standard
        cg.manifest_from_build(write("nostd.log", jit_log(tweak=lambda i, line: line.replace("-std=c++20", ""))), versions, "x")
    for drop in ("host", "ptxas", "cuobjdump", "nvcc"):
        text = "".join(ln + "\n" for ln in TOOL_VERSIONS.splitlines() if not ln.startswith("TOOLCHAIN %s" % drop))
        with pytest.raises(cg.Unusable):
            cg.manifest_from_build(write("build.log", jit_log()), write("partial.txt", text), "x")
    so = write("ext.so", "binary stand-in")                                                                  # the built extension is named by its hash
    with_art = cg.manifest_from_build(write("build.log", jit_log()), versions, "container toolchain", artifact=so)
    assert with_art["artifact"]["sha256"] == hashlib.sha256(b"binary stand-in").hexdigest() and with_art["artifact"]["path"] == so
    cg.validate_manifest(with_art)
    with pytest.raises(cg.Unusable):
        cg.validate_manifest(dict(with_art, artifact={"path": so, "sha256": "short"}))
    assert CgFiles(manifest=with_art).dump()["meta"]["artifact"]["sha256"] == with_art["artifact"]["sha256"]
    out = os.path.join(d, "manifest.json")                                                                   # the command line
    assert cg.main(["manifest", "--build-log", os.path.join(d, "build.log"), "--tool-versions", versions, "--label", "container toolchain", "--out", out]) == 0
    with open(out) as fh:
        assert json.load(fh) == m
    assert cg.main(["manifest", "--build-log", os.path.join(d, "cached.log"), "--tool-versions", versions, "--label", "x", "--out", out]) == 2
    f = CgFiles(manifest=m)                                                                                   # and the manifest feeds a dump of the very log it was made from
    full = write("full.log", "\n".join(jit_log().splitlines()[:-1]) + "\n" + cg_ptxas({ROWS: (112, 4, 4, 128), K1: (0, 0, 0, 100)}))
    assert cg.main(["manifest", "--build-log", full, "--tool-versions", versions, "--label", "container toolchain", "--out", out]) == 0
    dump_args = ["--regs", f.paths["regs"], "--sass", f.paths["sass"], "--elf", f.paths["elf"], "--ptxas-log", full]
    assert cg.main(["dump-files", "--manifest", out] + dump_args + ["--out", os.path.join(d, "dump.json")]) == 0
    with open(os.path.join(d, "dump.json")) as fh:
        assert json.load(fh)["meta"]["tree"]["commit"] is None
    assert cg.main(["dump-files", "--manifest", out] + dump_args + ["--commit", BASE_COMMIT, "--out", os.path.join(d, "dump1.json")]) == 0
    with open(os.path.join(d, "dump1.json")) as fh:                                                           # the identity of the built source tree is recorded when it is given
        assert json.load(fh)["meta"]["tree"]["commit"] == BASE_COMMIT
    wrong = write("wrong.json", json.dumps(dict(json.load(open(out)), archs=["sm_90a"])))                       # a manifest that contradicts its own log does not describe this dump
    assert cg.main(["dump-files", "--manifest", wrong] + dump_args + ["--out", os.path.join(d, "dump2.json")]) == 2
    o2 = write("full-o2.log", open(full).read().replace(" -O3 ", " -O2 "))                                      # the real manifest with a copy of the log whose commands say -O2
    assert cg.main(["dump-files", "--manifest", out] + dump_args[:-1] + [o2, "--out", os.path.join(d, "dump3.json")]) == 2


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
    for name, dump in (("base", base), ("same", base), ("diff", cg_edit(base, "sm_100a", "cfbi::k1_kernel<Li16E>", sass_hash=cg_hash("x"))),
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


CALLS, PROFILE_CALLS = 1536, 256                 # calls per trial and profiled calls of a lock-step decode cell (the defaults of the benchmark)


def wall_cell(key, vals, kind="decode", scale=1):
    """A decode cell with one trial per value: ``gpu_graph_us`` = value, ``gpu_eager_us`` = value + 1, ``cpu_graph_us`` = value + 3; the kernels of a trial are
    plan 2 us + rows 10 us and one helper kernel of 40 us that must stay out of the kernel sum. ``scale``: the factor of a re-run on the default calls."""
    runs = [{PLAN_K: 2.0, ROWS_K: 10.0, FILL_K: 40.0} for _ in vals]
    wall = [{"gpu_graph_us": v, "gpu_eager_us": v + 1.0, "cpu_graph_us": v + 3.0} for v in vals]
    return br.make_cell(kind, key, {"calls_per_trial": scale * CALLS, "profile_calls": scale * PROFILE_CALLS}, runs, drop_helpers=True, wall=wall)


def wall_rec(commit, cells, ts, stat=br.PROCESS_MEDIAN, gpu="NVIDIA B200", sm=148, host="node1", dirty=False, scale=1):
    meta = {"commit": commit, "dirty": dirty, "gpu": gpu, "sm_count": sm, "host": host, "timestamp": ts, "sample_stat": stat, "trials": 5}
    return {"schema": br.SCHEMA, "meta": meta, "cells": [wall_cell(k, v, scale=scale) for k, v in cells.items()]}


def trials(median, spread=0.0):
    """Five trial values whose median is ``median`` and whose range is ``2 * spread`` (relative)."""
    return [median * (1 - spread), median * (1 - spread / 2), median, median * (1 + spread / 2), median * (1 + spread)]


def wall_sets(base_meds, cand_meds, keys=(WALL_KEY,), hour="10", cand_commit="abc1234", **kw):
    """Interleaved record lists (A B A B A B ...) with one per-process median per record file for each cell key (``scale=3`` for the records of a re-run)."""
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
    res = br.compare(base, cand, metric="total_us")
    assert res["verdict"] == "PASS" and res["rows"][0]["base_median"] == 12.0


def test_widening_mode_passes_an_unstable_aa_pair_that_the_worst_mode_fails_and_still_fails_a_slower_candidate():
    base, cand = wall_sets([100.0, 100.45, 100.9], [100.15, 100.6, 101.05])                      # 0.9 % spread on each side, offset by 0.15 %
    rbase, rcand = wall_sets([100.0, 100.45, 100.9], [100.15, 100.6, 101.05], hour="11", scale=3)   # still above 0.3 % with three times the calls
    worst = br.compare(base, cand, metric="gpu_graph_us", baseline_unstable="worst")
    assert worst["verdict"] == "FAIL" and 1.0 < worst["rows"][0]["delta_pct"] < 1.1              # fastest baseline repeat vs slowest candidate repeat
    assert br.compare(base, cand, metric="gpu_graph_us")["verdict"] == "INVALID"                  # the default does not accept an unstable baseline
    first = br.compare(base, cand, metric="gpu_graph_us", baseline_unstable="widen")              # no re-run yet: the widened tolerance is not granted
    row = first["rows"][0]
    assert first["verdict"] == "INVALID" and row["needs_rerun"] and row["status"] == "NEEDS_RERUN" and not row["noise_limited"]
    assert any("no valid re-run" in p for p in first["problems"]) and "needs a re-run" in br.format_table(first)
    wide = br.compare(base, cand, metric="gpu_graph_us", baseline_unstable="widen", base_rerun=rbase, cand_rerun=rcand)
    row = wide["rows"][0]
    assert wide["verdict"] == "PASS" and row["noise_limited"] and row["judged_on"] == "rerun" and not row["needs_rerun"]
    assert abs(row["tol_pct_effective"] - 1.896) < 0.01 and row["median_delta_pct"] < 0.2
    assert "noise-limited" in br.format_table(wide)
    _, slower = wall_sets([100.0, 100.45, 100.9], [103.2, 103.65, 104.1])
    _, rslower = wall_sets([100.0, 100.45, 100.9], [103.2, 103.65, 104.1], hour="11", scale=3)
    assert br.compare(base, slower, metric="gpu_graph_us", baseline_unstable="widen", base_rerun=rbase, cand_rerun=rslower)["verdict"] == "FAIL"
    stable_b, stable_c = wall_sets([100.0, 100.05, 99.95], [100.0, 100.05, 99.95])               # stable cells keep the plain tolerance and need no re-run
    stable = br.compare(stable_b, stable_c, metric="gpu_graph_us", baseline_unstable="widen")
    assert stable["verdict"] == "PASS" and not stable["rows"][0]["noise_limited"] and not stable["rows"][0]["needs_rerun"]
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
    assert res["verdict"] == "INVALID" and res["rows"] == [] and any("zero k2_stage_us" in p and "kind decode" in p for p in res["problems"])
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
    assert res["verdict"] == "INVALID" and any("carry no gpu_graph_us" in p and "candidate" in p for p in res["problems"])
    gone = [json.loads(json.dumps(r)) for r in cand]                                              # a derived metric that vanished in the candidate is not a speed-up
    for r in gone:
        for cell in r["cells"]:
            for s in cell["samples"]:
                s["total_us"] = 0.0
    assert br.compare(base, gone, metric="total_us")["verdict"] == "INVALID"


def test_rerun_records_override_the_cells_they_contain():
    k1, k2 = br.cell_key("decode", N=1, H=12, regime="lockstep"), br.cell_key("decode", N=4, H=12, regime="lockstep")
    base, cand = wall_sets([100.0, 100.45, 100.9], [100.15, 100.6, 101.05], keys=(k1, k2))       # both cells are above 0.3 % in the first campaign
    rbase, rcand = wall_sets([100.0, 100.05, 99.95], [100.1, 100.15, 100.05], keys=(k1,), hour="11", scale=3)
    res = br.compare(base, cand, metric="gpu_graph_us", baseline_unstable="widen", base_rerun=rbase, cand_rerun=rcand)
    rows = {r["key"]: r for r in res["rows"]}
    assert rows[k1]["judged_on"] == "rerun" and not rows[k1]["noise_limited"] and not rows[k1]["needs_rerun"] and "judged on the re-run" in rows[k1]["note"]
    assert rows[k2]["judged_on"] == "first" and rows[k2]["needs_rerun"] and rows[k2]["status"] == "NEEDS_RERUN"
    assert res["verdict"] == "INVALID" and any(k2 in p and "no valid re-run" in p for p in res["problems"])                 # the cell without a re-run holds the gate back
    rbase2, rcand2 = wall_sets([100.0, 100.05, 99.95], [100.1, 100.15, 100.05], keys=(k1, k2), hour="11", scale=3)
    res = br.compare(base, cand, metric="gpu_graph_us", baseline_unstable="widen", base_rerun=rbase2, cand_rerun=rcand2)
    assert res["verdict"] == "PASS" and [r["judged_on"] for r in res["rows"]] == ["rerun", "rerun"]
    assert br.compare(base, cand, metric="gpu_graph_us", baseline_unstable="widen", base_rerun=rbase2)["verdict"] == "INVALID"      # one side only
    _, rcand_other = wall_sets([100.0, 100.05, 99.95], [100.1, 100.15, 100.05], keys=(k2,), hour="11", scale=3)
    res = br.compare(base, cand, metric="gpu_graph_us", baseline_unstable="widen", base_rerun=rbase, cand_rerun=rcand_other)
    assert res["verdict"] == "INVALID" and any("one side only" in p for p in res["problems"])
    few = rbase2[:2]
    assert br.compare(base, cand, metric="gpu_graph_us", baseline_unstable="widen", base_rerun=few, cand_rerun=rcand2[:2])["verdict"] == "INVALID"
    other = ("decode;N=2;H=12;regime=lockstep;",)                                                                              # a re-run cell that was never measured first
    ob, oc = wall_sets([100.0, 100.05, 99.95], [100.1, 100.15, 100.05], keys=other, hour="11", scale=3)
    res = br.compare(base, cand, metric="gpu_graph_us", baseline_unstable="widen", base_rerun=ob, cand_rerun=oc)
    assert res["verdict"] == "INVALID" and any("not in the first campaign" in p for p in res["problems"])


def test_a_defective_sample_makes_the_comparison_invalid_and_names_the_cell():
    k1, k2 = br.cell_key("decode", N=1, H=12, regime="lockstep"), br.cell_key("decode", N=4, H=12, regime="lockstep")
    base, cand = wall_sets([100.0, 100.05, 99.95], [100.0, 100.05, 99.95], keys=(k1, k2))
    assert br.compare(base, cand, metric="gpu_graph_us")["verdict"] == "PASS"
    for bad in (float("nan"), float("inf"), float("-inf"), -3.0, 0.0, True):
        for side in ("baseline", "candidate"):
            recs = json.loads(json.dumps(base if side == "baseline" else cand, allow_nan=True))
            recs[1]["cells"][1]["samples"][2]["gpu_graph_us"] = bad                                # one trial of one process of the second cell
            res = br.compare(*((recs, cand) if side == "baseline" else (base, recs)), metric="gpu_graph_us")
            assert res["verdict"] == "INVALID" and any(k2 in p and side in p for p in res["problems"]) and not any(k1 in p for p in res["problems"]), (bad, side)
    rbase, rcand = wall_sets([100.0, 100.05, 99.95], [100.0, 100.05, 99.95], keys=(k1, k2), hour="11", scale=3)
    ok = br.compare(base, cand, metric="gpu_graph_us", base_rerun=rbase, cand_rerun=rcand)
    assert ok["verdict"] == "PASS" and [r["judged_on"] for r in ok["rows"]] == ["rerun", "rerun"]
    for bad in (float("nan"), 0.0):                                                                  # a defective sample of the FIRST campaign of a cell that was re-run counts too
        first = json.loads(json.dumps(base, allow_nan=True))
        first[2]["cells"][0]["samples"][1]["gpu_graph_us"] = bad
        res = br.compare(first, cand, metric="gpu_graph_us", base_rerun=rbase, cand_rerun=rcand)
        assert res["verdict"] == "INVALID" and any(k1 in p and "baseline record" in p for p in res["problems"]), bad
    few = json.loads(json.dumps(cand))                                                               # review counterexample: four of five trials carry no wall metric
    for s_ in few[1]["cells"][0]["samples"][:4]:
        del s_["gpu_graph_us"]
    res = br.compare(base, few, metric="gpu_graph_us")
    assert res["verdict"] == "INVALID" and any("4 of 5 sample(s) carry no gpu_graph_us" in p and k1 in p for p in res["problems"])
    all_nan = json.loads(json.dumps(cand, allow_nan=True))                                          # review counterexample: every compared value is NaN
    for r in all_nan:
        for s_ in r["cells"][0]["samples"]:
            s_["gpu_graph_us"] = float("nan")
    res = br.compare(base, all_nan, metric="gpu_graph_us")
    assert res["verdict"] == "INVALID" and any("non-finite" in p and k1 in p for p in res["problems"]) and not any(r["status"] == "PASS" and r["key"] == k1 for r in res["rows"])
    d = tempfile.mkdtemp()                                                                         # the file round trip keeps a NaN a NaN
    broken = json.loads(json.dumps(cand))
    broken[0]["cells"][0]["samples"][0]["gpu_graph_us"] = float("nan")
    assert compare_bench.main(["--base"] + write_records(d, "a", base) + ["--cand"] + write_records(d, "c", broken) + ["--metric", "gpu_graph_us"]) == 2
    kernel = json.loads(json.dumps(cand))                                                           # a kernel metric: zero is a legitimate value of a vanished kernel
    for r in kernel:
        for s in r["cells"][0]["samples"]:
            s["kernels"]["plan_kernel"] = 0.0
    assert br.compare(base, kernel, metric="plan_kernel", cells=[k1])["verdict"] == "PASS"
    assert br.compare(kernel, base, metric="plan_kernel", cells=[k1])["verdict"] == "INVALID"      # but a zero baseline has nothing to compare with


def test_every_record_of_a_side_holds_the_same_unique_cells():
    k1, k2 = br.cell_key("decode", N=1, H=12, regime="lockstep"), br.cell_key("decode", N=4, H=12, regime="lockstep")
    base, cand = wall_sets([100.0, 100.05, 99.95], [100.0, 100.05, 99.95], keys=(k1, k2))
    lost = json.loads(json.dumps(cand))
    lost[1]["cells"] = lost[1]["cells"][:1]                                                        # one candidate record lost a cell the others hold
    res = br.compare(base, lost, metric="gpu_graph_us")
    assert res["verdict"] == "INVALID" and any("lacks the cell" in p and k2 in p for p in res["problems"])
    first = json.loads(json.dumps(base))
    first[0]["cells"] = first[0]["cells"][:1]                                                      # the first record does not define the campaign
    assert any("lacks the cell" in p and k2 in p for p in br.compare(first, cand, metric="gpu_graph_us")["problems"])
    dup = json.loads(json.dumps(base))
    dup[0]["cells"].append(json.loads(json.dumps(dup[0]["cells"][0])))                            # one key twice in one record
    res = br.compare(dup, cand, metric="gpu_graph_us")
    assert res["verdict"] == "INVALID" and any("repeats the cell key" in p for p in res["problems"])
    extra = br.cell_key("decode", N=16, H=12, regime="lockstep")                                   # review counterexample: a cell that only the first file of EACH side holds
    both_b, both_c = json.loads(json.dumps(base)), json.loads(json.dumps(cand))
    for recs in (both_b, both_c):
        recs[0]["cells"].append(wall_cell(extra, trials(100.0)))
    res = br.compare(both_b, both_c, metric="gpu_graph_us")
    assert res["verdict"] == "INVALID" and any("lacks the cell" in p and extra in p for p in res["problems"]) and extra not in [r["key"] for r in res["rows"] if r["status"] == "PASS"]
    only = json.loads(json.dumps(cand))
    for r in only:
        r["cells"] = r["cells"][:1]                                                                # the candidate lost a cell everywhere
    res = br.compare(base, only, metric="gpu_graph_us")
    assert res["verdict"] == "INVALID" and any("missing in the candidate" in p and k2 in p for p in res["problems"])
    assert br.compare(base, only, metric="gpu_graph_us", cells=[k1])["verdict"] == "PASS"         # a cell filter compares what it names


def test_a_cell_needs_three_distinct_process_samples_per_side():
    base, cand = wall_sets([100.0, 100.05, 99.95], [100.0, 100.05, 99.95])
    same = json.loads(json.dumps(base))
    for r in same:
        r["meta"]["timestamp"] = "2026-11-10T10:00:00Z"                                            # three files of one process: one sample counted three times
    res = br.compare(same, cand, metric="gpu_graph_us", require_interleaved=False)
    assert res["verdict"] == "INVALID" and any("distinct process" in p and "baseline" in p for p in res["problems"])
    nostamp = json.loads(json.dumps(cand))
    for r in nostamp:
        del r["meta"]["timestamp"]
    assert any("distinct process" in p for p in br.compare(base, nostamp, metric="gpu_graph_us", require_interleaved=False)["problems"])
    assert br.compare(base, cand, metric="gpu_graph_us", min_repeats=4)["verdict"] == "INVALID"   # a stricter minimum is honoured


def test_a_rerun_proves_three_times_the_calls_of_the_same_configuration():
    base, cand = wall_sets([100.0, 100.45, 100.9], [100.15, 100.6, 101.05])
    good = wall_sets([100.0, 100.05, 99.95], [100.1, 100.15, 100.05], hour="11", scale=3)

    def run(pair):
        return br.compare(base, cand, metric="gpu_graph_us", baseline_unstable="widen", base_rerun=pair[0], cand_rerun=pair[1])
    assert run(good)["verdict"] == "PASS"
    for scale in (1, 2, 4):                                                                        # only exactly three times the calls is a re-run
        res = run(wall_sets([100.0, 100.05, 99.95], [100.1, 100.15, 100.05], hour="11", scale=scale))
        assert res["verdict"] == "INVALID" and any("not 3 times" in p for p in res["problems"]), scale

    def edited(side, edit):
        pair = json.loads(json.dumps(good))
        for r in pair[0 if side == "baseline" else 1]:
            edit(r)
        return pair
    half = edited("baseline", lambda r: r["cells"][0]["args"].update(profile_calls=PROFILE_CALLS))   # the profiled calls were not scaled
    res = run(half)
    assert res["verdict"] == "INVALID" and any("profile_calls" in p and "not 3 times" in p and "baseline" in p for p in res["problems"])
    for field in ("calls_per_trial", "profile_calls"):                                             # a count that is not recorded cannot be proven
        res = run(edited("candidate", lambda r: r["cells"][0]["args"].pop(field)))
        assert res["verdict"] == "INVALID" and any(field in p and "not recorded consistently" in p for p in res["problems"]), field
    for field, value in (("commit", "ffff000"), ("trials", 7), ("host", "node9"), ("impl", 1), ("sample_stat", "pooled"), ("env_impl", "32"), ("torch", "9.9"), ("cuda", "9.9"),
                         ("src", "/other/snapshot")):
        res = run(edited("candidate", lambda r: r["meta"].update({field: value})))                 # another snapshot, configuration or machine
        assert res["verdict"] == "INVALID" and res["problems"], field
    res = run(edited("candidate", lambda r: r["cells"][0]["args"].update(calls_per_trial=3 * CALLS + 64)))
    assert res["verdict"] == "INVALID"
    skewed_b, skewed_c = json.loads(json.dumps(base)), json.loads(json.dumps(cand))                # both sides re-run with 3x, but they started from different calls
    for r in skewed_c:
        r["cells"][0]["args"].update(calls_per_trial=1000, profile_calls=100)
    res = br.compare(skewed_b, skewed_c, metric="gpu_graph_us", baseline_unstable="widen", base_rerun=good[0], cand_rerun=good[1])
    assert res["verdict"] == "INVALID" and any("not 3 times" in p or "different calls" in p for p in res["problems"])


def test_classify_rows_reads_slower_faster_and_noise_limited_cells():
    rows = [{"key": "a", "median_delta_pct": 2.0, "noise_limited": False}, {"key": "b", "median_delta_pct": -2.0, "noise_limited": False},
            {"key": "c", "median_delta_pct": -2.0, "noise_limited": True}, {"key": "d", "median_delta_pct": 0.5, "noise_limited": False},
            {"key": "e", "median_delta_pct": -1.0, "noise_limited": False}, {"key": "f", "median_delta_pct": 1.0, "noise_limited": False},
            {"key": "g", "median_delta_pct": 3.0, "noise_limited": True}, {"key": "h", "median_delta_pct": -5.0, "noise_limited": False, "needs_rerun": True}]
    out = br.classify_rows({"tol_pct": 1.0, "rows": rows})
    assert out["slower"] == ["a", "g"] and out["faster"] == ["b", "e"] and out["noise_limited"] == ["c", "g"] and out["neutral"] == ["d", "f"]
    assert out["needs_rerun"] == ["h"]                                                              # nothing is read from an unstable first campaign


def test_compare_bench_cli_widen_rerun_and_the_json_classification():
    d = tempfile.mkdtemp()
    base, cand = wall_sets([100.0, 100.45, 100.9], [97.0, 97.4, 97.8])                            # unstable and about 3 % faster
    rbase, rcand = wall_sets([100.0, 100.05, 99.95], [97.0, 97.05, 96.95], hour="11", scale=3)
    nbase, ncand = wall_sets([100.0, 100.45, 100.9], [97.0, 97.4, 97.8], hour="12", scale=3)      # still above 0.3 % with three times the calls
    ubase, ucand = wall_sets([100.0, 100.05, 99.95], [97.0, 97.05, 96.95], hour="13")             # a re-run with the same calls as the first campaign
    out = os.path.join(d, "out.json")
    common = ["--base"] + write_records(d, "a", base) + ["--cand"] + write_records(d, "b", cand) + ["--metric", "gpu_graph_us"]

    def rerun(b, c, name):
        return ["--rerun-base"] + write_records(d, name + "a", b) + ["--rerun-cand"] + write_records(d, name + "b", c)

    def result():
        with open(out) as f:
            return json.load(f)[0]
    assert compare_bench.main(common) == 2                                                        # invalid: unstable baseline
    assert compare_bench.main(common + ["--unstable-baseline", "widen", "--json", out]) == 2      # invalid: the widening needs a valid re-run
    res = result()
    assert res["classification"]["needs_rerun"] == [WALL_KEY] and res["classification"]["faster"] == [] and res["classification"]["noise_limited"] == []
    assert compare_bench.main(common + ["--unstable-baseline", "widen", "--json", out] + rerun(nbase, ncand, "n")) == 0
    res = result()
    assert res["classification"]["noise_limited"] == [WALL_KEY] and res["classification"]["faster"] == []   # noise-limited: not the evidence of a gain
    assert compare_bench.main(common + ["--unstable-baseline", "widen", "--json", out] + rerun(rbase, rcand, "r")) == 0
    res = result()
    assert res["classification"]["faster"] == [WALL_KEY] and res["rows"][0]["judged_on"] == "rerun" and res["classification"]["noise_limited"] == []
    assert compare_bench.main(common + ["--unstable-baseline", "widen", "--json", out] + rerun(ubase, ucand, "u")) == 2    # not three times the calls
    assert any("not 3 times" in p for p in result()["problems"])
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


# ---- per-test-file counts of a pytest run (benchmarks/suite_record.py) ---------------------------------------------------------------------------

JUNIT = """<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests"><testsuite name="pytest" errors="1" failures="1" skipped="1" tests="7">
<testcase classname="tests.cfbi.test_arch" name="test_a" time="0.1" />
<testcase classname="tests.cfbi.test_arch" name="test_b[x]" time="0.1" />
<testcase classname="tests.cfbi.test_vs_ref" name="test_c" time="0.1"><failure message="boom">trace</failure></testcase>
<testcase classname="tests.cfbi.test_vs_ref" name="test_d" time="0.1" />
<testcase classname="tests.cfbi.test_guard_zones.TestGuards" name="test_e" time="0.1"><skipped type="pytest.skip" message="not here">skipped</skipped></testcase>
<testcase classname="tests.cfbi.test_guard_zones.TestGuards" name="test_f" time="0.1" />
<testcase classname="tests.cfbi.test_cp" name="test_g" time="0.1"><error message="setup failed">trace</error></testcase>
</testsuite></testsuites>
"""
PROBE = """NVIDIA B200, 183359 MiB
3.12.3 2.14.0a0+4fdf77b940.nv26.08 13.4 NVIDIA B200 148
/usr/local/cuda/bin/nvcc
"""


def suite_record_of(text, commit=BASE_COMMIT, suite="S", **meta):
    d = tempfile.mkdtemp()
    path = os.path.join(d, "r.xml")
    with open(path, "w") as f:
        f.write(text)
    m = {"suite": suite, "commit": commit, "dirty": False, "gpu": "NVIDIA B200", "sm_count": 148}
    m.update(meta)
    return sr.build_record(sr.parse_junit(path), m)


def test_suite_record_counts_tests_per_file_with_failures_errors_and_skips():
    rec = suite_record_of(JUNIT)
    assert sr.structure_problems(rec) == []
    assert rec["files"]["tests/cfbi/test_arch.py"] == {"passed": 2, "failed": 0, "skipped": 0, "errors": 0, "total": 2}
    assert rec["files"]["tests/cfbi/test_vs_ref.py"] == {"passed": 1, "failed": 1, "skipped": 0, "errors": 0, "total": 2}
    assert rec["files"]["tests/cfbi/test_guard_zones.py"] == {"passed": 1, "failed": 0, "skipped": 1, "errors": 0, "total": 2}     # a test class maps to its file
    assert rec["files"]["tests/cfbi/test_cp.py"]["errors"] == 1
    assert rec["totals"] == {"passed": 4, "failed": 1, "skipped": 1, "errors": 1, "total": 7}
    assert rec["failed"] == ["tests/cfbi/test_vs_ref.py::test_c", "tests/cfbi/test_cp.py::test_g"]
    assert rec["skipped"] == [{"id": "tests/cfbi/test_guard_zones.py::test_e", "reason": "not here"}]
    assert sr.classname_to_file("tests.cfbi.test_x.TestY") == "tests/cfbi/test_x.py" and sr.classname_to_file("tests.cfbi.test_x") == "tests/cfbi/test_x.py"
    assert sr.parse_probe(PROBE) == {"python": "3.12.3", "torch": "2.14.0a0+4fdf77b940.nv26.08", "cuda": "13.4", "gpu": "NVIDIA B200", "sm_count": 148}
    assert sr.parse_probe("no probe line") == {}


def test_suite_record_check_rejects_fewer_tests_missing_files_and_unlisted_failures():
    clean = JUNIT.replace('<failure message="boom">trace</failure>', "").replace('<error message="setup failed">trace</error>', "")
    base = suite_record_of(clean)
    assert sr.check(base, suite_record_of(clean)) == []
    assert sr.check(base, suite_record_of(clean.replace('<testcase classname="tests.cfbi.test_arch" name="test_a" time="0.1" />', ""))) != []       # one test fewer
    assert any("is missing in the candidate" in p for p in sr.check(base, suite_record_of(clean.replace("tests.cfbi.test_cp", "tests.cfbi.test_other"))))
    more = clean.replace("</testsuite>", '<testcase classname="tests.cfbi.test_new" name="test_n" time="0.1" /></testsuite>')
    assert sr.check(base, suite_record_of(more)) == []                                                                                           # new files are fine
    failing = suite_record_of(JUNIT)                                                                                                             # failures of the candidate
    assert any("failed or errored test" in p for p in sr.check(base, failing))
    assert not [p for p in sr.check(base, failing, ["tests/cfbi/test_vs_ref.py::test_c", "tests/cfbi/test_cp.py::test_g"]) if "failed or errored" in p]
    assert any("different suites" in p for p in sr.check(base, suite_record_of(clean, suite="gate")))
    assert sr.check(base, {"schema": "x"}) != [] and sr.check({"schema": sr.SCHEMA, "meta": {}, "files": {}}, base) != []


def test_suite_record_cli_builds_checks_and_refuses_bad_input():
    d = tempfile.mkdtemp()
    xml_path, probe, out = (os.path.join(d, n) for n in ("r.xml", "probe.txt", "rec.json"))
    with open(xml_path, "w") as f:
        f.write(JUNIT)
    with open(probe, "w") as f:
        f.write(PROBE)
    args = ["--junit", xml_path, "--suite", "S", "--out", out, "--commit", BASE_COMMIT, "--probe", probe, "--env", "CFBI_BIG=1", "--ignored", "tests/cfbi/x.py"]
    assert sr.main(args) == 0
    with open(out) as f:
        rec = json.load(f)
    assert rec["meta"]["gpu"] == "NVIDIA B200" and rec["meta"]["sm_count"] == 148 and rec["meta"]["env"] == {"CFBI_BIG": "1"} and rec["meta"]["cuda"] == "13.4"
    assert rec["meta"]["ignored"] == ["tests/cfbi/x.py"] and rec["meta"]["dirty"] is False
    assert sr.main(["--check", out, out]) == 1                                                       # the record has failures that nobody allowed
    allow = os.path.join(d, "allow.txt")
    with open(allow, "w") as f:
        f.write("# by design on this device\ntests/cfbi/test_vs_ref.py::test_c\ntests/cfbi/test_cp.py::test_g\n")
    assert sr.main(["--check", out, out, "--allow-failures", allow]) == 0
    assert sr.main(["--junit", os.path.join(d, "missing.xml"), "--suite", "S", "--out", out, "--commit", "x"]) == 2
    assert sr.main(["--suite", "S"]) == 2
    with open(xml_path, "w") as f:
        f.write("<not xml")
    assert sr.main(args) == 2


SAN_FILE = "tests/cfbi/test_k2_dyn_sanitizer.py"
SESSION = ('<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests"><testsuite name="pytest" errors="0" failures="0" skipped="0" tests="%d" time="3.2" '
           'timestamp="2026-11-10T19:37:30.65" hostname="node7">%s</testsuite></testsuites>')


def junit_of(names, cls="tests.cfbi.test_x"):
    return SESSION % (len(names), "".join('<testcase classname="%s" name="%s" time="0.1" />' % (cls, n) for n in names))


def merge_part(names, cls="tests.cfbi.test_x", ignored=(), **meta):
    m = {"suite": "S", "commit": BASE_COMMIT, "dirty": False, "gpu": "NVIDIA B200", "sm_count": 148, "tree": "at-7ebb589", "env": {"CFBI_BIG": "1"}, "python": "3.12.3",
         "torch": "2.14.0", "cuda": "13.4", "ignored": list(ignored), "label": "part", "junit": "p.xml", "subset": True, "host": "node7"}
    m.update(meta)
    return suite_record_of(junit_of(names, cls), **m)


def test_suite_record_keeps_per_test_outcomes_the_session_and_refuses_duplicate_ids():
    rec = suite_record_of(JUNIT)
    assert rec["tests"] == {"tests/cfbi/test_arch.py::test_a": "passed", "tests/cfbi/test_arch.py::test_b[x]": "passed", "tests/cfbi/test_vs_ref.py::test_c": "failed",
                            "tests/cfbi/test_vs_ref.py::test_d": "passed", "tests/cfbi/test_guard_zones.py::test_e": "skipped",
                            "tests/cfbi/test_guard_zones.py::test_f": "passed", "tests/cfbi/test_cp.py::test_g": "errors"}
    assert sr.structure_problems(rec) == []
    wrong = json.loads(json.dumps(rec))
    wrong["tests"]["tests/cfbi/test_vs_ref.py::test_c"] = "passed"                                  # the outcomes must say what the counts say
    assert any("do not match the per-test outcomes" in p for p in sr.structure_problems(wrong))
    assert any("malformed per-test outcomes" in p for p in sr.structure_problems(dict(rec, tests={"a::b": "ok"})))
    d = tempfile.mkdtemp()
    xml = os.path.join(d, "s.xml")
    with open(xml, "w") as f:
        f.write(junit_of(["test_a", "test_a"]))
    parsed = sr.parse_junit(xml)
    assert parsed["duplicates"] == ["tests/cfbi/test_x.py::test_a"] and parsed["session"]["hostname"] == "node7" and parsed["session"]["timestamp"] == "2026-11-10T19:37:30.65"
    assert sr.main(["--junit", xml, "--suite", "S", "--out", os.path.join(d, "o.json"), "--commit", BASE_COMMIT, "--gpu", "g", "--sm-count", "1"]) == 2
    with open(xml, "w") as f:
        f.write(junit_of(["test_a", "test_b"]))
    out = os.path.join(d, "o.json")
    assert sr.main(["--junit", xml, "--suite", "S", "--out", out, "--commit", BASE_COMMIT, "--gpu", "g", "--sm-count", "1", "--ignored", SAN_FILE]) == 0
    with open(out) as f:
        back = json.load(f)
    assert back["meta"]["host"] == "node7" and back["meta"]["kind"] == "single" and back["meta"]["subset"] is True and len(back["tests"]) == 2


def test_suite_record_merge_joins_separate_runs_and_validates_every_rule():
    main_part = merge_part(["test_a", "test_b[1]"], ignored=[SAN_FILE])
    san = merge_part(["test_san[memcheck]", "test_san[initcheck]"], cls="tests.cfbi.test_k2_dyn_sanitizer", host="node8")
    infos = [{"file": "a.json", "sha256": "1" * 64}, {"file": "b.json", "sha256": "2" * 64}]
    files = ["tests/cfbi/test_x.py", SAN_FILE]
    ids = ["tests/cfbi/test_x.py::test_a", "tests/cfbi/test_x.py::test_b[1]", SAN_FILE + "::test_san[memcheck]", SAN_FILE + "::test_san[initcheck]"]
    merged = sr.merge_records([main_part, san], infos, "full suite", files, ids, "tree + collection")
    assert sr.structure_problems(merged) == [] and merged["meta"]["kind"] == "merge" and merged["meta"]["subset"] is False and merged["meta"]["ignored"] == []
    assert merged["totals"] == {"passed": 4, "failed": 0, "skipped": 0, "errors": 0, "total": 4} and sorted(merged["files"]) == sorted(files)
    assert [p["sha256"] for p in merged["meta"]["parts"]] == ["1" * 64, "2" * 64] and merged["meta"]["parts"][1]["tests"] == 2 and merged["meta"]["hosts"] == ["node7", "node8"]
    assert merged["meta"]["expected"] == {"source": "tree + collection", "files": 2, "tests": 4}
    assert sr.check(merged, merged) == []

    def problems(*parts, **kw):
        return sr.merge_problems(list(parts), kw.get("files"), kw.get("ids"))
    assert problems(main_part, san, files=files, ids=ids) == []
    assert any("at least two" in p for p in problems(main_part))
    for field, value in (("commit", "f" * 40), ("tree", "at-other"), ("gpu", "NVIDIA B300"), ("sm_count", 160), ("env", {}), ("torch", "9.9"), ("cuda", "12.9"), ("suite", "gate")):
        other = merge_part(["test_san[memcheck]"], cls="tests.cfbi.test_k2_dyn_sanitizer", **{field: value})
        assert any("differ in %s" % field in p for p in problems(main_part, other)), field          # one session of one tree and toolchain
    assert any("dirty" in p for p in problems(main_part, merge_part(["test_s"], dirty=True)))
    assert any("is in part 0 and in part 1" in p for p in problems(main_part, merge_part(["test_a"])))      # a test is counted once
    assert any("ignored by part 0 and no part covers it" in p for p in problems(main_part, merge_part(["test_c"])))
    assert any("expected file" in p and SAN_FILE in p for p in problems(main_part, merge_part(["test_c"], ignored=[]), files=files))
    assert any("expected test" in p for p in problems(main_part, san, ids=ids + [SAN_FILE + "::test_san[racecheck]"]))
    assert any("not expected" in p for p in problems(main_part, san, ids=ids[:3]))
    assert any("not expected" in p and "file" in p for p in problems(main_part, san, files=files[:1]))
    old = json.loads(json.dumps(main_part))
    del old["tests"]                                                                                  # a record of the first round has no per-test outcomes
    assert any("no per-test outcomes" in p for p in problems(old, san))
    assert any("part 1" in p for p in problems(main_part, {"schema": "x"}))
    with pytest.raises(ValueError):
        sr.merge_records([main_part, merge_part(["test_a"])], infos, "x")
    assert sr.completeness_problems(merged, files, ids) == []                                         # only the merge certifies the whole suite
    assert any("subset record" in p for p in sr.completeness_problems(main_part, files, ids)) and any("ignores" in p for p in sr.completeness_problems(main_part))
    assert any("expected file" in p for p in sr.completeness_problems(merged, files + ["tests/cfbi/test_y.py"]))
    assert any("expected test" in p for p in sr.completeness_problems(merged, None, ids + ["tests/cfbi/test_x.py::test_z"]))
    assert any("not expected" in p for p in sr.completeness_problems(merged, None, ids[:3]))
    old = json.loads(json.dumps(merged))
    del old["tests"]
    assert any("no per-test outcomes" in p for p in sr.completeness_problems(old, None, ids))


def test_suite_record_merge_command_line():
    d = tempfile.mkdtemp()
    parts, tree = [], os.path.join(d, "tree", "tests", "cfbi")
    os.makedirs(tree)
    for name in ("test_x.py", "test_k2_dyn_sanitizer.py", "helpers.py"):
        with open(os.path.join(tree, name), "w") as f:
            f.write("")
    for name, rec_ in (("a.json", merge_part(["test_a"], ignored=[SAN_FILE])), ("b.json", merge_part(["test_san[memcheck]"], cls="tests.cfbi.test_k2_dyn_sanitizer"))):
        parts.append(os.path.join(d, name))
        with open(parts[-1], "w") as f:
            json.dump(rec_, f)
    ids = os.path.join(d, "ids.txt")
    with open(ids, "w") as f:
        f.write("# collection\ntests/cfbi/test_x.py::test_a\n" + SAN_FILE + "::test_san[memcheck]\n")
    out = os.path.join(d, "m.json")
    assert sr.main(["--merge"] + parts + ["--out", out, "--label", "full", "--expect-files-from", os.path.join(d, "tree"), "--expect-tests", ids]) == 0
    with open(out) as f:
        merged = json.load(f)
    assert merged["meta"]["kind"] == "merge" and merged["totals"]["passed"] == 2 and len(merged["meta"]["parts"][0]["sha256"]) == 64
    assert merged["meta"]["parts"][0]["file"] == "a.json" and merged["meta"]["expected"]["files"] == 2          # helpers.py is no test file
    with open(ids, "a") as f:
        f.write(SAN_FILE + "::test_san[racecheck]\n")
    assert sr.main(["--merge"] + parts + ["--out", out, "--expect-tests", ids]) == 2                                # an expected test that no part ran
    assert sr.main(["--merge", parts[0], "--out", out]) == 2 and sr.main(["--merge"] + parts) == 2
    with open(ids, "w") as f:
        f.write("tests/cfbi/test_x.py::test_a\n" + SAN_FILE + "::test_san[memcheck]\n")
    common = ["--expect-files-from", os.path.join(d, "tree"), "--expect-tests", ids]
    assert sr.main(["--merge"] + parts + ["--out", out] + common) == 0
    assert sr.main(["--complete", out] + common) == 0 and sr.main(["--complete", parts[0]] + common) == 1      # the part without the sanitizer file is not the suite


# ---- the layer-chain graph benchmark: cells, node lists and the zero-free step (the measurement needs a GPU) -------------------------------------

CHAIN_COUNTS = {"void cfbi::lin::linear_kernel<128, 2>(cfbi::lin::Args)": 6.0, "void cfbi::conv::conv_kernel<4>(cfbi::conv::Args)": 3.0,
                "void cfbi::plan_kernel(long const*, int)": 1.0, "void cfbi::rows_kernel(cfbi::Params)": 1.0}


TWO_LAUNCH = lc.CANONICAL_COMPUTE["two-launch"]
LIN_K = "void cfbi::lin::linear_kernel<128, 2>(cfbi::lin::Args)"
MEMSET = "Memset (Device)"
ATEN_FILL = "void at::native::vectorized_elementwise_kernel<4, FillFunctor>"


def test_layer_chain_cells_and_the_exact_kernel_node_mappings():
    specs = lc.cell_specs()
    assert [s["key"] for s in specs] == ["chain;N=1;H=12;Dm=2048;regime=lockstep;", "chain;N=64;H=12;Dm=2048;regime=lockstep;"]
    assert [s["N"] for s in lc.select_cells(specs, ["N=1;"])] == [1] and [s["N"] for s in lc.select_cells(specs, ["N=64;"])] == [64] and lc.select_cells(specs, ["N=6;"]) == []
    assert lc.calls_per_trial(dict(lc.DEFAULTS)) == 1536 and lc.calls_per_trial(dict(lc.DEFAULTS, calls_scale=3)) == 4608 and lc.calls_per_trial(dict(lc.DEFAULTS, calls=100)) == 128
    assert lc.calls_per_trial(dict(lc.DEFAULTS, calls=100, calls_scale=3)) == 384                    # a re-run has exactly three times the rounded replays
    assert lc.profile_calls_of(dict(lc.DEFAULTS)) == 64 and lc.profile_calls_of(dict(lc.DEFAULTS, calls_scale=3)) == 192            # one and three whole 64-position cycles
    assert lc.profile_calls_of(dict(lc.DEFAULTS, profile_calls=16)) == 64 and lc.profile_calls_of(dict(lc.DEFAULTS, profile_calls=100, calls_scale=3)) == 384
    compute, aux = lc.split_nodes(CHAIN_COUNTS)
    assert sum(compute.values()) == 11 and aux == {} and compute["rows_kernel"] == 1 and compute["plan_kernel"] == 1 and compute["lin::linear_kernel"] == 6
    assert compute == TWO_LAUNCH
    assert lc.check_nodes(compute, aux, TWO_LAUNCH, {}) is None and lc.check_nodes(compute, aux, None, None) is None
    swapped = dict(CHAIN_COUNTS)                                                                     # still 11 compute kernels, but not the canonical ones
    swapped[LIN_K], swapped["void cfbi::conv::conv_kernel<4>(cfbi::conv::Args)"] = 5.0, 4.0
    c3, a3 = lc.split_nodes(swapped)
    assert sum(c3.values()) == 11 and "expected" in lc.check_nodes(c3, a3, TWO_LAUNCH, {})
    single = {k: v for k, v in CHAIN_COUNTS.items() if "plan_kernel" not in k}                       # the single-launch decode has one kernel less
    c1, a1 = lc.split_nodes(single)
    assert lc.check_nodes(c1, a1, {"conv::conv_kernel": 3, "lin::linear_kernel": 6, "rows_kernel": 1}, {}) is None and "expected" in lc.check_nodes(c1, a1, TWO_LAUNCH, {})
    noisy = dict(CHAIN_COUNTS)                                                                       # a graph with zero-fill and copy nodes: they are auxiliary nodes
    noisy[MEMSET], noisy[ATEN_FILL] = 10.0, 2.0
    c2, a2 = lc.split_nodes(noisy)
    assert sum(c2.values()) == 11 and sorted(a2) == [MEMSET, ATEN_FILL]
    msg = lc.check_nodes(c2, a2, TWO_LAUNCH, {})
    assert "auxiliary nodes" in msg and MEMSET in msg
    assert lc.check_nodes(c2, a2, TWO_LAUNCH, {MEMSET: 10, ATEN_FILL: 2}) is None                  # the exact mapping of the auxiliary nodes that may remain
    assert lc.check_nodes(c2, a2, TWO_LAUNCH, {MEMSET: 9, ATEN_FILL: 2}) is not None and lc.check_nodes(c2, a2, TWO_LAUNCH, {MEMSET: 10}) is not None
    frac = dict(CHAIN_COUNTS)
    frac[LIN_K] = 5.984                                                                              # one launch missing from one replay of 64
    c4, a4 = lc.split_nodes(frac)
    assert "whole number" in lc.check_nodes(c4, a4, None, None)


def test_layer_chain_wrong_paths_with_the_right_number_of_kernels_are_refused():
    single = lc.CANONICAL_COMPUTE["single-launch"]
    assert single == {k: v for k, v in TWO_LAUNCH.items() if k != "plan_kernel"} and sum(single.values()) == 10 and sum(TWO_LAUNCH.values()) == 11
    ref_rows = "void cfbi::ref::rows_kernel(cfbi::Params)"                                            # the frozen reference kernel, a compute kernel of its own name
    wrong = {"review counterexample: no plan, two reference rows kernels": dict(CHAIN_COUNTS, **{"void cfbi::plan_kernel(long const*, int)": 0.0,
                                                                                               "void cfbi::rows_kernel(cfbi::Params)": 0.0, ref_rows: 2.0}),
             "missing plan, rows twice": dict(CHAIN_COUNTS, **{"void cfbi::plan_kernel(long const*, int)": 0.0, "void cfbi::rows_kernel(cfbi::Params)": 2.0}),
             "reference rows instead of the optimised one": dict(CHAIN_COUNTS, **{"void cfbi::rows_kernel(cfbi::Params)": 0.0, ref_rows: 1.0}),
             "plan twice, one linear less": dict(CHAIN_COUNTS, **{"void cfbi::plan_kernel(long const*, int)": 2.0, LIN_K: 5.0})}
    for what, counts in sorted(wrong.items()):
        counts = dict((k, v) for k, v in counts.items() if v)                                         # the profiler lists only the kernels that ran
        compute, aux = lc.split_nodes(counts)
        assert sum(compute.values()) == 11, what                                                       # the number of compute kernels alone cannot tell
        assert "expected" in lc.check_nodes(compute, aux, TWO_LAUNCH, {}), what
    compute, aux = lc.split_nodes({k: v for k, v in CHAIN_COUNTS.items() if "plan_kernel" not in k})
    assert lc.check_nodes(compute, aux, single, {}) is None and lc.check_nodes(compute, aux, TWO_LAUNCH, {}) is not None
    compute, aux = lc.split_nodes(CHAIN_COUNTS)
    assert lc.check_nodes(compute, aux, single, {}) is not None                                       # the two-launch step is not the single-launch one


def test_layer_chain_node_mappings_are_parsed_strictly():
    assert lc.parse_mapping("two-launch", lc.CANONICAL_COMPUTE) == TWO_LAUNCH and lc.parse_mapping("none") == {} and lc.parse_mapping("") == {}
    assert lc.parse_mapping("single-launch", lc.CANONICAL_COMPUTE) == lc.CANONICAL_COMPUTE["single-launch"]
    assert lc.parse_mapping("conv::conv_kernel=3;lin::linear_kernel=6") == {"conv::conv_kernel": 3, "lin::linear_kernel": 6}
    assert lc.parse_mapping("void at::native::vectorized_elementwise_kernel<4, FillFunctor>=2") == {"void at::native::vectorized_elementwise_kernel<4, FillFunctor>": 2}
    for bad in ("11", "a", "=3", "a=0", "a=x", "a=1;a=2", "a=1;;b=2"):
        with pytest.raises(ValueError):
            lc.parse_mapping(bad)
    for tool in (bd, lc):                                                                             # a bad option stops the tool before it touches the GPU
        for argv in (["--calls-scale", "0"], ["--trials", "0"], ["--profile-calls", "-1"], ["--calls", "0"]):
            with pytest.raises(SystemExit):
                tool.main(argv)
    for argv in (["--expect-compute", "11"], ["--expect-aux", "a=0"]):
        with pytest.raises(SystemExit):
            lc.main(argv)


# ---- the measurement loops of the benchmarks driven by a fake device: how many calls and profiled calls really run -------------------------------

class FakeEvent(object):
    """A CUDA event of a fake device: ``elapsed_time`` is 0.01 ms per ``record`` that happened in between."""
    clock = [0]

    def __init__(self, enable_timing=False):
        self.t = 0

    def record(self):
        FakeEvent.clock[0] += 1
        self.t = FakeEvent.clock[0]

    def elapsed_time(self, other):
        return 0.01 * (other.t - self.t)


class FakeTorch(object):
    class cuda(object):
        Event = FakeEvent

        @staticmethod
        def synchronize():
            pass


class FakeBuffer(object):
    def fill_(self, value):
        return self


class FakeDecodeEnv(object):
    """The ShapeEnv of bench_decode: counts the eager calls and graph replays and logs the position resets."""

    def __init__(self, log):
        self.torch, self.impl, self.log, self.calls = FakeTorch, 0, log, {"eager": 0, "replay": 0}
        self.state = types.SimpleNamespace(pos=FakeBuffer())

    def set_positions(self, regime):
        self.log.append(("positions", regime))

    def eager(self, i):
        self.calls["eager"] += 1

    def replay(self, i):
        self.calls["replay"] += 1


class FakeChainEnv(object):
    def __init__(self, log):
        self.torch, self.impl, self.log, self.replays = FakeTorch, 0, log, 0

    def set_positions(self):
        self.log.append(("positions",))

    def replay(self, i=0):
        self.replays += 1


def fake_breakdown(log, kernels, counts_seq=None):
    """A stand-in for ``bench_breakdown`` whose ``kernel_times`` calls ``fn`` warm + iters times, logs its arguments and returns the kernel times and the launch
    counts per call of the n-th invocation (``counts_seq[n]``, the last one when there are fewer)."""
    mod = types.ModuleType("bench_breakdown")
    state = {"n": 0}

    def kernel_times(fn, iters=5, warm=3):
        log.append(("kernel_times", iters, warm))
        for _ in range(warm + iters):
            fn()
        counts = counts_seq[min(state["n"], len(counts_seq) - 1)] if counts_seq else {k: 1.0 for k in kernels}
        state["n"] += 1
        return dict(kernels), dict(counts)
    mod.kernel_times = kernel_times
    return mod


def run_decode_cell(monkeypatch, regime, scale, profile_calls=100):
    log = []
    monkeypatch.setitem(sys.modules, "bench_breakdown", fake_breakdown(log, {PLAN_K: 2.0, ROWS_K: 10.0}))
    env = FakeDecodeEnv(log)
    opts = dict(bd.DEFAULTS, trials=5, calls=100, cold_calls=100, probe_calls=70, profile_calls=profile_calls, calls_scale=scale)
    spec = {"N": 4, "H": 12, "regime": regime, "pos": 5 if regime == "probe" else None, "key": "decode;N=4;H=12;regime=%s;" % regime}
    return bd.measure_cell(spec, env, opts, {}, None, FakeBuffer()), env, log, opts


def test_decode_profiled_pass_scales_with_the_rerun_and_covers_whole_position_cycles(monkeypatch):
    for regime in bd.REGIMES:
        loop = regime != "probe"
        for scale in (1, 3):
            cell, env, log, _ = run_decode_cell(monkeypatch, regime, scale)
            want_calls = (128 if loop else 70) * scale                                           # 100 calls round up to two whole cycles, the probe calls stay
            want_profiled = (128 if loop else 100) * scale
            passes = [e for e in log if e[0] == "kernel_times"]
            assert passes == [("kernel_times", want_profiled, 1)] * 5, (regime, scale)           # the profiler really ran that many calls after one warm-up call, in every trial
            args = cell["args"]
            assert args["profile_calls"] == want_profiled and args["calls_per_trial"] == want_calls and args["calls_scale"] == scale, (regime, scale)
            if loop:
                assert want_profiled % 64 == 0 and args["profile_cycles"] == want_profiled // 64 and args["profile_start_phase"] == 1
            else:
                assert args["profile_cycles"] is None and args["profile_start_phase"] is None
            assert all(log[i - 1] == ("positions", regime) for i, e in enumerate(log) if e[0] == "kernel_times")   # every profiled pass starts from freshly set positions
            assert env.calls["eager"] == 200 + 5 * (want_calls + 1 + want_profiled) and env.calls["replay"] == 50 + 5 * want_calls, (regime, scale)
            assert len(cell["samples"]) == 5 and all(sample["total_us"] == 12.0 for sample in cell["samples"])


def test_decode_reruns_prove_three_times_the_calls_and_the_profiled_calls_to_the_comparator(monkeypatch):
    def as_records(cell):
        return [{"meta": {"trials": 5, "sample_stat": br.PROCESS_MEDIAN, "commit": "c"}, "cells": [cell]}]
    for regime in bd.REGIMES:
        first, _, _, _ = run_decode_cell(monkeypatch, regime, 1)
        rerun, _, _, _ = run_decode_cell(monkeypatch, regime, 3)
        key = first["key"]
        assert br.rerun_problems(as_records(first), as_records(rerun), "baseline", key) == [], regime
        unscaled = json.loads(json.dumps(rerun))                                                  # a re-run whose profiled calls were not scaled is not a re-run
        unscaled["args"]["profile_calls"] = first["args"]["profile_calls"]
        assert any("profile_calls" in p and "not 3 times" in p for p in br.rerun_problems(as_records(first), as_records(unscaled), "baseline", key)), regime
    odd, _, _, _ = run_decode_cell(monkeypatch, "lockstep", 1, profile_calls=65)                  # 65 calls are not a whole cycle: two cycles are profiled, and recorded
    assert odd["args"]["profile_calls"] == 128 and odd["args"]["profile_cycles"] == 2


def run_chain_cell(monkeypatch, counts_seq, expect_compute=TWO_LAUNCH, expect_aux={}, scale=1, trials=5):          # noqa: B006  (never modified)
    log = []
    monkeypatch.setitem(sys.modules, "bench_breakdown", fake_breakdown(log, {"conv::conv_kernel": 6.0, "lin::linear_kernel": 30.0, "plan_kernel": 2.0, "rows_kernel": 12.0},
                                                                      counts_seq))
    env = FakeChainEnv(log)
    spec = lc.cell_specs((1,))[0]
    opts = dict(lc.DEFAULTS, trials=trials, calls=100, calls_scale=scale)
    return lc.measure_cell(spec, env, opts, expect_compute, expect_aux), env, log


def test_layer_chain_cell_records_effective_calls_and_the_nodes_of_every_trial(monkeypatch):
    for scale in (1, 3):
        cell, env, log = run_chain_cell(monkeypatch, [CHAIN_COUNTS], scale=scale)
        args = cell["args"]
        profiled, replays = 64 * scale, 128 * scale
        assert args["profile_calls"] == profiled and args["calls_per_trial"] == replays and args["profile_cycles"] == scale and args["profile_start_phase"] == 1
        assert [e for e in log if e[0] == "kernel_times"] == [("kernel_times", profiled, 1)] * 5
        assert all(log[i - 1] == ("positions",) for i, e in enumerate(log) if e[0] == "kernel_times")
        assert env.replays == 100 + 5 * (replays + 1 + profiled)
        assert args["compute_nodes"] == TWO_LAUNCH and args["aux_nodes"] == {} and args["compute_kernels"] == 11
        assert [s["nodes"] for s in cell["samples"]] == [{"compute": TWO_LAUNCH, "aux": {}}] * 5
        assert lc.cell_node_problems(cell, TWO_LAUNCH, {}) == [] and lc.cell_node_problems(cell) == []
        tampered = json.loads(json.dumps(cell))                                                  # the record check finds a trial that carries another mapping
        tampered["samples"][3]["nodes"]["compute"]["rows_kernel"] = 2
        assert any("trial 3" in p for p in lc.cell_node_problems(tampered))
        stripped = json.loads(json.dumps(cell))
        del stripped["samples"][1]["nodes"]
        assert any("trial 1 carries no node mapping" in p for p in lc.cell_node_problems(stripped))
        assert lc.cell_node_problems(cell, {"rows_kernel": 1}, {}) != [] and lc.cell_node_problems(cell, TWO_LAUNCH, {MEMSET: 1}) != []
        assert any("records no" in p for p in lc.cell_node_problems({"key": "k", "args": {}, "samples": cell["samples"]}))


def test_layer_chain_asserts_the_node_mapping_in_every_trial_not_only_the_last(monkeypatch):
    good = dict(CHAIN_COUNTS)
    drifts = {"a launch missing": dict(good, **{LIN_K: 5.0}), "a launch missing from some replays": dict(good, **{LIN_K: 5.984}),
              "a memset": dict(good, **{MEMSET: 1.0}), "another kernel": dict(good, **{"void cfbi::conv::conv_kernel<4>(cfbi::conv::Args)": 4.0})}
    for what, bad in sorted(drifts.items()):
        for bad_trial in (0, 2, 4):
            seq = [good] * 5
            seq[bad_trial] = bad
            with pytest.raises(SystemExit) as e:
                run_chain_cell(monkeypatch, seq)
            assert "trial %d" % bad_trial in str(e.value), (what, bad_trial)
    for bad_trial in (1, 2, 4):                                                                    # without an expectation the trials must still agree with each other
        seq = [good] * 5
        seq[bad_trial] = drifts["a memset"]
        with pytest.raises(SystemExit) as e:
            run_chain_cell(monkeypatch, seq, expect_compute=None, expect_aux=None)
        assert "trial %d" % bad_trial in str(e.value) and "differ" in str(e.value)
    cell, _, _ = run_chain_cell(monkeypatch, [dict(good, **{MEMSET: 10.0})], expect_aux={MEMSET: 10})   # an exactly asserted auxiliary mapping is recorded
    assert cell["args"]["aux_nodes"] == {MEMSET: 10} and lc.cell_node_problems(cell, TWO_LAUNCH, {MEMSET: 10}) == []


def test_layer_chain_step_has_no_zero_fill_or_fp32_side_output_call():
    """The step that is captured must not hold auxiliary zeroing (the layer of the chain test zero-fills its fp32 outputs inside the calls)."""
    tree = ast.parse((_ROOT / "benchmarks" / "bench_layer_chain.py").read_text())
    step = [n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "ChainEnv" for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "step"]
    assert len(step) == 1
    names = set()
    for node in ast.walk(step[0]):
        if isinstance(node, ast.Call):
            f = node.func
            names.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
            names.update(k.arg for k in node.keywords)
    assert not names & {"zeros", "zeros_like", "zero_", "fill_", "full", "full_like", "empty", "empty_like", "out_fp32"}, names
    assert {"out"} <= names                                                                           # the outputs are pre-allocated buffers passed as out=
