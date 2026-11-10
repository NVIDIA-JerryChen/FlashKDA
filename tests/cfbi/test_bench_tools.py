"""Self-tests of the benchmark tooling (benchmarks/bench_records.py, compare_bench.py, compare_binaries.py) and of the corpus V definitions.

Plain Python: no GPU and no torch are needed (run them without the package conftest, e.g. ``pytest --noconftest tests/cfbi/test_bench_tools.py``,
where torch is not installed). The negative cases show that the gates can fail: a wrong, dirty or unstable baseline is refused, a 1 % slower
candidate fails the 0.5 % protection gate, and a build with one more register or a spill is reported as different.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "benchmarks", _ROOT / "tests" / "cfbi"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bench_records as br  # noqa: E402
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
