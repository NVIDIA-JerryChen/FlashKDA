"""Self-tests of the record checker (benchmarks/record_checker.py): every rule accepts a valid record and rejects the records that break it.

Plain Python, no GPU and no torch (``pytest --noconftest tests/cfbi/test_record_checker.py``). The valid records are built here from synthetic stamps / samples / git histories; each
mutant changes one thing of a valid record and the test says which problem must be reported, so a rule that cannot fail does not pass.
"""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "benchmarks", _ROOT / "tests" / "cfbi"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ncu_record as nr  # noqa: E402
import record_checker as rc  # noqa: E402
import rows_profile_record as rpr  # noqa: E402
from test_rows_profile import make_slots  # noqa: E402

HEX40 = "0123456789abcdef0123456789abcdef01234567"


def mutated(rec, edit):
    out = copy.deepcopy(rec)
    edit(out)
    return out


def has(problems, needle):
    return any(needle in p for p in problems)


# ---- profile records -------------------------------------------------------------------------------------------------------------------------
SMALL_SHAPES, SMALL_HEADS = (1, 2, 4), 3


def phases_of(pos, cta):
    return [10 + (pos + cta) % 5, 20 + (pos * 3 + cta) % 7, 30 + (pos + 2 * cta) % 9, 5 + cta % 3, 50 + (pos * 5 + cta) % 11, 8 + (pos + cta) % 4, 3 + (pos == 63) * 400]


def profile_record(shapes=SMALL_SHAPES, heads=SMALL_HEADS, gpu="NVIDIA B200", commit="abc1234def56", sm_count=148, dirty=False):
    cells = []
    for n in shapes:
        slots = make_slots(n, heads, phases_of, sm_of=lambda pos, cta: (pos * 7 + cta) % sm_count)
        cells.append(rpr.cell_from_slots(slots, n, heads))
    meta = {"commit": commit, "dirty": dirty, "tree": "snapshot", "src": "/x", "gpu": gpu, "sm_count": sm_count, "cc": "10.0", "host": "node1", "timestamp": "2026-11-11T06:19:05.997132Z",
            "torch": "2.14.0", "cuda": "13.4", "variant": ["CFBI_ROWS_PROFILE=1"], "heads": heads, "shapes": list(shapes), "positions": 64, "warmup_passes": 1, "label": "", "tag": "b200",
            "build_dir": None, "argv": []}
    return {"schema": rpr.SCHEMA, "meta": meta, "cells": cells, "_path": "mem"}


@pytest.fixture(scope="module")
def prof():
    return profile_record()


def pcheck(rec, **kw):
    return rc.profile_problems(rec, kw.pop("shapes", SMALL_SHAPES), kw.pop("heads", SMALL_HEADS), **kw)


def test_a_valid_profile_record_passes(prof):
    assert pcheck(prof) == []
    assert pcheck(prof, expect_commit="abc1234") == []


def test_the_default_shapes_of_the_decode_profile_pass():
    """N = 1, 16 and 128 at 12 heads: the cells of the real records."""
    rec = profile_record(shapes=rc.PROFILE_SHAPES, heads=rc.PROFILE_HEADS)
    assert rc.profile_problems(rec) == []


@pytest.mark.parametrize("name,edit,expected", [
    ("a missing cell", lambda r: r["cells"].pop(1), "is missing"),
    ("a cell twice", lambda r: r["cells"].append(copy.deepcopy(r["cells"][0])), "appears 2 times"),
    ("a cell of another shape", lambda r: r["cells"][2].update(key="rows-profile;N=8;H=3;"), "not one of the expected cells"),
    ("no cells", lambda r: r.update(cells=[]), "is missing"),
    ("a dirty tree", lambda r: r["meta"].update(dirty=True), "dirty"),
    ("a dirty flag that is a string", lambda r: r["meta"].update(dirty="False"), "must be a boolean"),
    ("an empty commit", lambda r: r["meta"].update(commit=""), "is not a hexadecimal commit hash"),
    ("a short commit", lambda r: r["meta"].update(commit="abc12"), "is not a hexadecimal commit hash"),
    ("the default build", lambda r: r["meta"].update(variant=[]), "not the profile build"),
    ("another number of heads", lambda r: r["meta"].update(heads=12), "heads"),
    ("other shapes", lambda r: r["meta"].update(shapes=[1, 2]), "shapes"),
    ("no GPU", lambda r: r["meta"].pop("gpu"), "gpu"),
    ("an SM count that is a string", lambda r: r["meta"].update(sm_count="148"), "sm_count"),
    ("no timestamp", lambda r: r["meta"].update(timestamp=""), "timestamp"),
    ("rows that were not recorded", lambda r: r["cells"][0].update(rows_recorded=r["cells"][0]["rows_recorded"] - 1), "were recorded"),
    ("a wrong CTA count", lambda r: r["cells"][1].update(ctas=5), "ctas"),
    ("an SM number beyond the device", lambda r: r["cells"][0]["sms"].append(148), "SM numbers up to"),
    ("a repeated SM number", lambda r: r["cells"][0].update(sms=[0, 0, 1]), "distinct SM numbers"),
    ("a group with the wrong number of rows", lambda r: r["cells"][0]["rows"]["chunk_end"].update(rows=7), "rows, expected"),
    ("one position fewer", lambda r: r["cells"][0]["per_position"].pop(), "per_position must hold 64 blocks"),
    ("a position with the wrong index", lambda r: r["cells"][0]["per_position"][5].update(pos=6), "pos 6"),
    ("a share that is not the phase over the row", lambda r: r["cells"][0]["rows"]["regular"]["shares"].update(prep=0.9), "the share of prep"),
    ("a kernel total that is not prologue + row + epilogue", lambda r: r["cells"][0]["rows"]["all"].update(kernel_total=r["cells"][0]["rows"]["all"]["kernel_total"] * 1.1), "kernel total"),
    ("a group that is not the mean of its positions", lambda r: r["cells"][1]["rows"]["regular"].update(
        prologue=r["cells"][1]["rows"]["regular"]["prologue"] + 5, kernel_total=r["cells"][1]["rows"]["regular"]["kernel_total"] + 5), "the means of its positions give"),
    ("a position that does not fit its group", lambda r: r["cells"][0]["per_position"][10]["phases"].update(prep=500.0), "position 10"),
    ("percentiles out of order", lambda r: r["cells"][0]["rows"]["all"].update(row_total_p50=r["cells"][0]["rows"]["all"]["row_total_p95"] + 1), "not ordered"),
    ("a maximum that is not the largest position", lambda r: r["cells"][0]["rows"]["regular"].update(row_total_max=r["cells"][0]["rows"]["regular"]["row_total_max"] + 1.0),
     "maximum row duration"),
    ("a number that is NaN", lambda r: r["cells"][0]["rows"]["all"].update(prologue=float("nan")), "finite number"),
    ("a number that is a boolean", lambda r: r["cells"][0]["rows"]["all"].update(epilogue=True), "finite number"),
    ("a negative phase", lambda r: r["cells"][0]["rows"]["all"]["phases"].update(prep=-1.0), "negative"),
    ("a prologue of zero cycles", lambda r: r["cells"][0]["rows"]["all"].update(prologue=0), "entry stamp must precede"),
])
def test_a_broken_profile_record_is_rejected(prof, name, edit, expected):
    assert has(pcheck(mutated(prof, edit)), expected), (name, pcheck(mutated(prof, edit))[:3])


def scale_phase(block, name, factor):
    """The block with one phase scaled and its row duration, share and kernel total re-derived as a consistent but wrong producer would."""
    block["phases"][name] *= factor


def test_phases_that_do_not_add_up_to_the_row_duration_are_rejected_beyond_the_tolerance():
    block = rc.combined([{"rows": 1, "row_total_max": 100.0, "prologue": 10.0, "epilogue": 5.0, "row_total": 100.0, "kernel_total": 115.0,
                          "phases": dict(zip(rpr.PHASES, (10.0, 20.0, 30.0, 10.0, 15.0, 10.0, 5.0)))}])
    block.update(shares=dict((k, v / 100.0) for k, v in block["phases"].items()), row_total_p50=100.0, row_total_p95=100.0)
    assert rc.block_problems(block, "b") == []
    for factor in (1.6, 3.0):                                                                  # 6 and 20 cycles of 100: more than 5 % apart
        assert has(rc.block_problems(mutated(block, lambda b: scale_phase(b, "prep", factor)), "b"), "more than 5.0 % apart")
    near = mutated(block, lambda b: scale_phase(b, "prep", 1.3))                               # 3 cycles of 100: inside 5 % (the shares are then stale)
    assert not has(rc.block_problems(near, "b"), "apart") and has(rc.block_problems(near, "b"), "the share of prep")
    assert not has(rc.block_problems(near, "b", tolerance=0.2), "apart") and has(rc.block_problems(near, "b", tolerance=0.02), "apart")


def test_the_phases_of_a_whole_record_are_checked_against_the_row_duration(prof):
    bad = mutated(prof, lambda r: r["cells"][0]["rows"]["regular"]["phases"].update(prep=r["cells"][0]["rows"]["regular"]["phases"]["prep"] * 1.6))
    problems = pcheck(bad)
    assert has(problems, "the phases add up to") and has(problems, "regular rows")


def test_a_profile_set_needs_a_valid_record_for_every_device_and_one_commit():
    b200, b300 = profile_record(gpu="NVIDIA B200"), profile_record(gpu="NVIDIA B300 SXM6 AC", sm_count=148)
    devices = ("B200", "B300")
    check = lambda recs, **kw: rc.profile_set_problems(recs, devices, SMALL_SHAPES, SMALL_HEADS, **kw)
    assert check([b200, b300]) == [] and check([b200, b300, profile_record()]) == []                 # repeated runs of a device are fine
    assert has(check([b200]), "no profile record for B300") and has(check([]), "no profile record for B200")
    assert has(check([b200, profile_record(gpu="NVIDIA H100")]), "is none of B200, B300")
    assert has(check([b200, b300, profile_record(gpu="NVIDIA B300", commit="fff0000aaaa")]), "different commits")
    assert has(check([b200, mutated(b300, lambda r: r["meta"].update(dirty=True))]), "dirty")
    assert has(check([b200, b300], expect_commit="9999999"), "is not the expected")


def test_profile_command_line(tmp_path, capsys):
    good = tmp_path / "good.json"
    bad = tmp_path / "bad.json"
    rec = profile_record()
    rec.pop("_path")
    good.write_text(json.dumps(rec))
    bad.write_text(json.dumps(mutated(rec, lambda r: r["meta"].update(dirty=True))))
    args = ["profile", "--shapes", "1", "2", "4", "--heads", "3"]
    assert rc.main(args + ["--record", str(good)]) == 0
    assert rc.main(args + ["--record", str(good), "--device", "B200"]) == 0
    assert rc.main(args + ["--record", str(good), "--device", "B200", "B300"]) == 1
    assert rc.main(args + ["--record", str(bad)]) == 1
    assert "PROBLEM:" in capsys.readouterr().out
    assert rc.main(args + ["--record", str(tmp_path / "missing.json")]) == 2
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert rc.main(args + ["--record", str(broken)]) == 2
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"schema": "x"}))
    assert rc.main(args + ["--record", str(other)]) == 1
    assert rc.main([]) == 2


# ---- ncu segment records ---------------------------------------------------------------------------------------------------------------------
def csv_rows(n_segments=4, per_segment=5, samples=400, stall_names=("stall_long_sb", "stall_barrier", "stall_wait")):
    rows = []
    for g in range(n_segments):
        for i in range(per_segment):
            last = i == per_segment - 1
            src = "BAR.SYNC 0x0" if last else ("HMMA.1688.F32.TF32 R16, R16, R22, R80" if i == 0 else ("LDS R4, [R2]" if i == 1 else "@P0 LDG.E R3, [R2.64]"))
            r = {"Address": "0x7f00000%04x" % (g * 100 + i), "Source": src, rc.nr.SAMPLES: str(samples // per_segment + g), rc.nr.EXECUTED: str(100 * (g + 1) + i)}
            for k, c in enumerate(stall_names):
                r[c] = str((samples // per_segment + g) // (k + 1) // 2)
            rows.append(r)
    return rows, list(stall_names)


def ncu_record(n=1, position=49, cache="all", gpu="NVIDIA B200", commit="abc1234def56", samples=4000, heads=12):
    rows, cols = csv_rows(samples=samples)
    summary = nr.summarise(rows, cols)
    meta = {"commit": commit, "dirty": False, "gpu": gpu, "sm_count": 148, "host": "node1", "tree": "snapshot", "ncu_version": "2026.3.0.0", "timestamp": "2026-11-11T06:51:48Z",
            "case": {"kernel": "rows_kernel", "N": n, "H": heads, "position": position, "launch_skip": position, "cache_control": cache},
            "collection": {"set": "full", "clock_control": "boost", "warp_sampling_interval": "0", "command": "ncu --set full ..."},
            "report": {"name": "full_pos49_cold.ncu-rep", "sha256": "a" * 64, "bytes": 123456},
            "source_csv": {"name": "source_pos49_cold.csv", "sha256": "b" * 64, "bytes": 654321},
            "launch": {"grid_size": n * heads, "block_size": 256, "registers_per_thread": 128}}
    return dict(nr.build_record(meta, summary), _path="mem")


def test_a_valid_ncu_record_passes():
    assert rc.ncu_problems(ncu_record()) == []
    assert rc.ncu_problems(ncu_record(samples=4000), expect_commit="abc1234") == []


# the output of `ncu --version` inside the container, as logged by the node runner: the PyTorch banner (with its own "Version" line) comes first
NCU_VERSION_LOG = """
=============
== PyTorch ==
=============

NVIDIA Release 26.08 (build 406036884)
PyTorch Version 2.14.0a0+4fdf77b
Container image Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NOTE: Mellanox network driver detected, but NVIDIA peer memory driver not
      detected.  Multi-node communication performance may be reduced.

NVIDIA (R) Nsight Compute Command Line Profiler
Copyright (c) 2018-2026 NVIDIA Corporation
Version 2026.3.0.0 (build 38525999) (public-release)
[node_run] 2026-11-11T07:18:02Z host=umbriel-b200-091 tag=b200 mode=pyrun exit=0
"""


def test_the_ncu_version_is_the_nsight_compute_one_and_never_the_pytorch_banner():
    assert nr.parse_ncu_version(NCU_VERSION_LOG) == "2026.3.0.0"
    assert nr.parse_ncu_version(NCU_VERSION_LOG.replace("2026.3.0.0", "2027.1.2")) == "2027.1.2"
    assert nr.parse_ncu_version(NCU_VERSION_LOG + NCU_VERSION_LOG) == "2026.3.0.0"                       # the same version twice is not ambiguous
    for name, text in (("no Nsight Compute banner", NCU_VERSION_LOG.split("NVIDIA (R) Nsight")[0]),
                       ("only a bare version line", "Version 2026.3.0.0 (build 38525999) (public-release)\n"),
                       ("a banner without a version line", "NVIDIA (R) Nsight Compute Command Line Profiler\nCopyright (c) 2018-2026 NVIDIA Corporation\nnothing\n"),
                       ("a version of another shape", NCU_VERSION_LOG.replace("2026.3.0.0 (build 38525999)", "2.14.0a0+4fdf77b (build 1)")),
                       ("two different versions", NCU_VERSION_LOG + NCU_VERSION_LOG.replace("2026.3.0.0", "2025.4.1.0")),
                       ("an empty log", "")):
        with pytest.raises(ValueError):
            nr.parse_ncu_version(text)
        assert name


@pytest.mark.parametrize("value", ["2.14.0a0+4fdf77b", "", "unknown", None, 2026.3, "2026.3", "v2026.3.0.0", "2026.3.0.0 (build 38525999)"])
def test_the_ncu_record_names_a_real_nsight_compute_version(value):
    rec = ncu_record()
    assert rc.ncu_problems(rec) == []
    assert has(rc.ncu_problems(mutated(rec, lambda r: r["meta"].update(ncu_version=value))), "ncu_version")
    for good in ("2026.3.0.0", "2026.1.1", "2027.2.0.1"):
        assert rc.ncu_problems(mutated(rec, lambda r: r["meta"].update(ncu_version=good))) == []


@pytest.mark.parametrize("name,edit,expected", [
    ("too few stall samples", lambda r: r["summary"].update(samples=71.0), "at least 1000 are needed"),
    ("a dirty tree", lambda r: r["meta"].update(dirty=True), "dirty"),
    ("a wrong kernel", lambda r: r["meta"]["case"].update(kernel="plan_kernel"), "case.kernel"),
    ("another head count", lambda r: r["meta"]["case"].update(H=6), "case.H"),
    ("a position out of range", lambda r: r["meta"]["case"].update(position=64, launch_skip=64), "case.position"),
    ("a launch that is not the row at its position", lambda r: r["meta"]["case"].update(launch_skip=3), "launch_skip"),
    ("an unknown cache control", lambda r: r["meta"]["case"].update(cache_control="some"), "cache_control"),
    ("the default sampling set", lambda r: r["meta"]["collection"].update(set="basic"), "--set full"),
    ("a base clock", lambda r: r["meta"]["collection"].update(clock_control="base"), "--set full"),
    ("no sampling interval", lambda r: r["meta"]["collection"].pop("warp_sampling_interval"), "warp sampling interval"),
    ("an automatic sampling interval", lambda r: r["meta"]["collection"].update(warp_sampling_interval="auto"), "warp sampling interval"),
    ("a report without a hash", lambda r: r["meta"]["report"].pop("sha256"), "report needs"),
    ("a page with a bad hash", lambda r: r["meta"]["source_csv"].update(sha256="xyz"), "source_csv needs"),
    ("a launch with the wrong grid", lambda r: r["meta"]["launch"].update(grid_size=13), "CTAs"),
    ("segments that do not hold the executed instructions", lambda r: r["summary"].update(executed=r["summary"]["executed"] * 2), "executed"),
    ("segments that do not hold the samples", lambda r: r["summary"].update(samples=r["summary"]["samples"] + 500), "samples"),
    ("segments that do not hold the static instructions", lambda r: r["summary"].update(static=1), "static instructions"),
    ("shares that do not add up", lambda r: r["summary"]["segments"][0].update(samples_pct=r["summary"]["segments"][0]["samples_pct"] + 5), "samples_pct"),
    ("segments out of order", lambda r: r["summary"]["segments"][1].update(seg=7), "not numbered"),
    ("a malformed segment", lambda r: r["summary"]["segments"][0].update(ops={"HMMA": 1}), "malformed"),
    ("stalls that are not largest first", lambda r: r["summary"]["segments"][0].update(top_stalls=[["a", 10.0], ["b", 20.0]]), "largest first"),
    ("no stalls in a segment with samples", lambda r: r["summary"]["segments"][0].update(top_stalls=[]), "largest first"),
    ("no segments", lambda r: r["summary"].update(segments=[]), "needs totals"),
])
def test_a_broken_ncu_record_is_rejected(name, edit, expected):
    rec = ncu_record()
    assert rc.ncu_problems(rec) == []
    assert has(rc.ncu_problems(mutated(rec, edit)), expected), (name, rc.ncu_problems(mutated(rec, edit))[:3])


def test_the_ncu_set_needs_every_device_and_shape_at_the_required_position():
    recs = [ncu_record(n=n, gpu=g) for n in (1, 128) for g in ("NVIDIA B200", "NVIDIA B300 SXM6 AC")]
    assert rc.ncu_set_problems(recs) == []
    assert rc.ncu_set_problems(recs + [ncu_record(n=1, position=63)]) == []                                        # more launches are welcome
    assert has(rc.ncu_set_problems(recs[:-1]), "no ncu record of B300 for N=128")
    assert has(rc.ncu_set_problems([r for r in recs if r["meta"]["case"]["N"] != 1]), "no ncu record of B200 for N=1")
    assert has(rc.ncu_set_problems([ncu_record(n=n, gpu=g, position=63) for n in (1, 128) for g in ("NVIDIA B200", "NVIDIA B300")]), "at position 49")
    assert has(rc.ncu_set_problems([ncu_record(n=n, gpu=g, cache="none") for n in (1, 128) for g in ("NVIDIA B200", "NVIDIA B300")]), "cache control all")
    assert has(rc.ncu_set_problems(recs + [ncu_record(n=1, commit="fff0000aaaa")]), "different commits")
    assert has(rc.ncu_set_problems(recs + [ncu_record(n=1, samples=100)]), "at least 1000")


# the text of benchmarks/micro/ncu_segments.py before the summary became data: the same lines for the same page
def reference_text(rows, stall_cols, topn=12):
    from collections import Counter, defaultdict
    num = nr.num
    out = []
    tot_exec = sum(num(r["Instructions Executed"]) for r in rows)
    tot_samp = sum(num(r["Warp Stall Sampling (All Samples)"]) for r in rows)
    out.append(f"{len(rows)} static instructions, {tot_exec:.0f} warp-instructions executed, {tot_samp:.0f} stall samples")
    segs, cur = [], []
    for r in rows:
        cur.append(r)
        if "BAR.SYNC" in r["Source"] or "BAR.ARV" in r["Source"]:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    out.append(f"{'seg':>3} {'static':>6} {'exec(M)':>8} {'%exec':>6} {'samples':>8} {'%samp':>6}  mma  lds  ldg  stg  sts  top stalls")
    for i, s in enumerate(segs):
        ex = sum(num(r["Instructions Executed"]) for r in s)
        sm = sum(num(r["Warp Stall Sampling (All Samples)"]) for r in s)
        cnt = Counter()
        for r in s:
            op = r["Source"].split()[0] if r["Source"].split() else ""
            cnt[op.split(".")[0]] += 1
        st = defaultdict(float)
        for r in s:
            for c in stall_cols:
                st[c] += num(r[c])
        top = sorted(st.items(), key=lambda kv: -kv[1])[:3]
        tops = " ".join(f"{k[6:]}={v/sm*100:.0f}%" for k, v in top if sm)
        out.append(f"{i:>3} {len(s):>6} {ex/1e6:>8.2f} {ex/tot_exec*100:>5.1f}% {sm:>8.0f} {sm/tot_samp*100:>5.1f}%  {cnt['HMMA']:>3}  {cnt['LDS']:>3}  {cnt['LDG']:>3}  {cnt['STG']:>3}  {cnt['STS']:>3}  {tops}")
    out.append("\nhottest instructions by stall samples:")
    for r in sorted(rows, key=lambda r: -num(r["Warp Stall Sampling (All Samples)"]))[:topn]:
        st = sorted(((c, num(r[c])) for c in stall_cols), key=lambda kv: -kv[1])[:2]
        out.append(f"  {r['Address'][-5:]} samples={num(r['Warp Stall Sampling (All Samples)']):>6.0f} exec={num(r['Instructions Executed']):>9.0f} {r['Source'].strip()[:70]:<70} "
                   + " ".join(f"{k[6:]}={v:.0f}" for k, v in st))
    out.append("\nexecuted warp-instructions by opcode (all segments):")
    byop, static = Counter(), Counter()
    for r in rows:
        toks = r["Source"].split()
        if not toks:
            continue
        op = toks[0]
        if op.startswith("@"):
            op = toks[1] if len(toks) > 1 else op
        byop[op.split(".")[0]] += num(r["Instructions Executed"])
        static[op.split(".")[0]] += 1
    for op, v in byop.most_common(28):
        out.append(f"  {op:<10} {v/1e6:>8.2f} M  {v/tot_exec*100:>5.1f}%   static {static[op]}")
    return "\n".join(out)


def test_the_segment_text_is_what_the_tool_printed_before_it_became_data():
    rows, cols = csv_rows(n_segments=6, per_segment=7, samples=3000)
    rows.append({"Address": "0x7f0009999", "Source": "", rc.nr.SAMPLES: "0", rc.nr.EXECUTED: "0", **dict((c, "0") for c in cols)})     # a trailing segment without a barrier, an empty source
    assert nr.format_summary(nr.summarise(rows, cols)) == reference_text(rows, cols)
    assert nr.format_summary(nr.summarise(rows, cols, 3)) == reference_text(rows, cols, 3)


def test_the_segment_tool_reads_a_page_prints_the_text_and_writes_the_record(tmp_path, capsys):
    sys.path.insert(0, str(_ROOT / "benchmarks" / "micro"))
    try:
        import ncu_segments
    finally:
        sys.path.pop(0)
    rows, cols = csv_rows(samples=4000)
    page = tmp_path / "source_x.csv"
    header = ["Address", "Source", rc.nr.SAMPLES, "Warp Stall Sampling (Not-issued Samples)", rc.nr.EXECUTED] + cols
    lines = ['"Kernel Name","cfbi::rows_kernel(cfbi::Params)",', ",".join('"%s"' % h for h in header)]
    for r in rows:
        lines.append(",".join('"%s"' % {**r, "Warp Stall Sampling (Not-issued Samples)": "0"}[h] for h in header))
    page.write_text("\n".join(lines) + "\n")
    meta = tmp_path / "meta.json"
    out = tmp_path / "rec.json"
    meta.write_text(json.dumps(ncu_record()["meta"]))
    assert ncu_segments.main([str(page), "--json", str(out), "--meta", str(meta)]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("20 static instructions") and printed == nr.format_summary(nr.summarise(*nr.load_rows(str(page)))) + "\n"
    rec = json.loads(out.read_text())
    assert rec["schema"] == nr.SCHEMA and rec["meta"]["source_csv"]["name"] == "source_x.csv" and len(rec["meta"]["source_csv"]["sha256"]) == 64
    rec["_path"] = str(out)
    assert not has(rc.ncu_problems(rec), "segments") and not has(rc.ncu_problems(rec), "source_csv")      # the written record has the shape that the checker reads


# ---- the record commit rule, on real git histories -------------------------------------------------------------------------------------------
def git(repo, *args):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com")
    return subprocess.check_output(["git", "-C", str(repo)] + list(args), env=env, stderr=subprocess.STDOUT).decode().strip()


def commit_file(repo, path, body, message):
    full = Path(repo) / path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(body)
    git(repo, "add", path)
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture()
def history(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    h = {"repo": repo}
    h["a"] = commit_file(repo, "csrc/k.cu", "kernel v1\n", "kernel")
    h["b"] = commit_file(repo, "docs/perf.md", "table\n", "document the records")           # only docs/ changes
    h["c"] = commit_file(repo, "docs/more.md", "more\n", "document more")
    h["d"] = commit_file(repo, "benchmarks/tool.py", "tool\n", "tool change")              # a path outside docs/
    return h


def test_a_record_commit_may_differ_from_head_only_under_docs(history):
    repo = history["repo"]
    assert rc.record_commit_problems(history["a"], repo, history["c"]) == []           # a .. c: docs only
    assert rc.record_commit_problems(history["b"], repo, history["c"]) == []
    assert rc.record_commit_problems(history["c"], repo, history["c"]) == []           # the commit itself
    assert rc.record_commit_problems(history["a"][:10], repo, history["c"]) == []      # an abbreviation of 7+ characters
    assert has(rc.record_commit_problems(history["a"], repo, history["d"]), "outside docs/")
    assert has(rc.record_commit_problems(history["a"], repo), "benchmarks/tool.py")    # HEAD is d
    assert has(rc.record_commit_problems("0" * 40, repo), "does not have")
    assert has(rc.record_commit_problems("abc", repo), "not a hexadecimal commit hash")
    assert has(rc.record_commit_problems("", repo), "not a hexadecimal commit hash")


def test_a_record_commit_must_be_an_ancestor_of_head(history):
    repo = history["repo"]
    git(repo, "checkout", "-q", "-b", "side", history["a"])
    side = commit_file(repo, "docs/side.md", "side\n", "a docs-only commit on another line")
    git(repo, "checkout", "-q", "main")
    assert has(rc.record_commit_problems(side, repo, "main"), "not an ancestor")


def test_the_record_commit_command_line(history, capsys):
    repo = str(history["repo"])
    assert rc.main(["commit", "--record-commit", history["a"], "--repo", repo, "--head", history["c"]]) == 0
    assert rc.main(["commit", "--record-commit", history["a"], "--repo", repo]) == 1
    assert rc.main(["commit", "--record-commit", history["a"], "--repo", repo, "--head", history["c"], "--require-clean"]) == 0
    (history["repo"] / "csrc" / "k.cu").write_text("kernel v2 uncommitted\n")
    assert rc.main(["commit", "--record-commit", history["d"], "--repo", repo, "--require-clean"]) == 1
    assert "working tree is not clean" in capsys.readouterr().out
    (history["repo"] / "csrc" / "k.cu").write_text("kernel v1\n")
    evidence = history["repo"] / rc.UNTRACKED_OK[3:]
    evidence.mkdir()
    (evidence / "x.txt").write_text("evidence\n")                                      # untracked evidence in the evidence directory does not make a tree dirty
    assert rc.main(["commit", "--record-commit", history["d"], "--repo", repo, "--require-clean"]) == 0
