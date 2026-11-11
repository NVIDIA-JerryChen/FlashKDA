"""Self-tests of the benchmark tooling (benchmarks/bench_records.py, compare_bench.py, compare_binaries.py) and of the corpus V definitions.

Plain Python: no GPU and no torch are needed (run them without the package conftest, e.g. ``pytest --noconftest tests/cfbi/test_bench_tools.py``,
where torch is not installed). The negative cases show that the gates can fail: a wrong, dirty or unstable baseline is refused, a 1 % slower
candidate fails the 0.5 % protection gate, and a build with one more register or a spill is reported as different.
"""
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
