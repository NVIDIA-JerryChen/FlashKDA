"""Per-segment summary of an Nsight Compute SASS source page as data, text and a record (``cfbi-ncu-segments-v1``; plain Python, no torch, no GPU).

The page comes from ``ncu --import REPORT --page source --csv --print-source sass`` of a report collected with ``--set full``. The SASS of a kernel is cut into segments at every
barrier instruction (``BAR.SYNC`` or ``BAR.ARV``), so a segment is what a warp executes between two barriers; for every segment the summary gives the static instructions, the executed
warp instructions, the stall samples, a few instruction counts and the three stall reasons with the most samples. ``benchmarks/micro/ncu_segments.py`` prints it as text (the same text
as before) and writes it, with the metadata of the collection, as a record; ``benchmarks/record_checker.py`` validates such records.

Stall sampling needs enough samples: ncu's automatic warp sampling interval gave 71 samples for a whole launch of a 12-CTA grid, which says nothing about a segment. The records of this
repository are collected with ``--warp-sampling-interval 0`` (a sample every 32 cycles) and a launch with fewer than ``MIN_SAMPLES`` samples is not accepted as evidence.
"""
import csv
from collections import Counter, defaultdict

SCHEMA = "cfbi-ncu-segments-v1"
SAMPLES = "Warp Stall Sampling (All Samples)"
EXECUTED = "Instructions Executed"
MIN_SAMPLES = 1000
OPS = ("HMMA", "LDS", "LDG", "STG", "STS")      # the instruction counts that a segment line shows
TOP_OPCODES = 28
BARRIERS = ("BAR.SYNC", "BAR.ARV")


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def load_rows(path):
    """``(rows, stall columns)`` of a source page exported as CSV: the rows after the header line that starts with ``"Address"``; the stall columns are the ``stall_*`` columns."""
    with open(path, newline="") as f:
        lines = f.read().splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith('"Address"'))
    reader = csv.DictReader(lines[header:])
    stall_cols = [c for c in reader.fieldnames if c.startswith("stall_") and "Not Issued" not in c]
    return list(reader), stall_cols


def summarise(rows, stall_cols, topn=12):
    """The summary of ``rows``: totals, one entry per barrier-delimited segment, the instructions with the most stall samples and the executed instructions by opcode."""
    tot_exec = sum(num(r[EXECUTED]) for r in rows)
    tot_samp = sum(num(r[SAMPLES]) for r in rows)
    segs, cur = [], []
    for r in rows:
        cur.append(r)
        if any(b in r["Source"] for b in BARRIERS):
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    segments = []
    for i, s in enumerate(segs):
        ex = sum(num(r[EXECUTED]) for r in s)
        sm = sum(num(r[SAMPLES]) for r in s)
        cnt = Counter()
        for r in s:
            op = r["Source"].split()[0] if r["Source"].split() else ""
            cnt[op.split(".")[0]] += 1
        st = defaultdict(float)
        for r in s:
            for c in stall_cols:
                st[c] += num(r[c])
        top = sorted(st.items(), key=lambda kv: -kv[1])[:3]
        segments.append({"seg": i, "static": len(s), "executed": ex, "exec_pct": ex / tot_exec * 100 if tot_exec else 0.0, "samples": sm,
                         "samples_pct": sm / tot_samp * 100 if tot_samp else 0.0, "ops": dict((k, cnt[k]) for k in OPS),
                         "top_stalls": [[k[6:], v / sm * 100] for k, v in top] if sm else []})
    hottest = []
    for r in sorted(rows, key=lambda r: -num(r[SAMPLES]))[:topn]:
        st = sorted(((c, num(r[c])) for c in stall_cols), key=lambda kv: -kv[1])[:2]
        hottest.append({"address": r["Address"][-5:], "samples": num(r[SAMPLES]), "executed": num(r[EXECUTED]), "source": r["Source"].strip()[:70],
                        "stalls": [[k[6:], v] for k, v in st]})
    byop, static = Counter(), Counter()
    for r in rows:
        toks = r["Source"].split()
        if not toks:
            continue
        op = toks[0]
        if op.startswith("@"):
            op = toks[1] if len(toks) > 1 else op
        byop[op.split(".")[0]] += num(r[EXECUTED])
        static[op.split(".")[0]] += 1
    opcodes = [{"op": op, "executed": v, "exec_pct": v / tot_exec * 100 if tot_exec else 0.0, "static": static[op]} for op, v in byop.most_common(TOP_OPCODES)]
    return {"static": len(rows), "executed": tot_exec, "samples": tot_samp, "segments": segments, "hottest": hottest, "opcodes": opcodes}


def format_summary(s):
    """The text of ``ncu_segments.py`` for a summary (one string, lines joined by newlines, no final newline)."""
    lines = ["%d static instructions, %.0f warp-instructions executed, %.0f stall samples" % (s["static"], s["executed"], s["samples"]),
             "%3s %6s %8s %6s %8s %6s  mma  lds  ldg  stg  sts  top stalls" % ("seg", "static", "exec(M)", "%exec", "samples", "%samp")]
    for g in s["segments"]:
        tops = " ".join("%s=%.0f%%" % (name, pct) for name, pct in g["top_stalls"])
        lines.append("%3d %6d %8.2f %5.1f%% %8.0f %5.1f%%  %3d  %3d  %3d  %3d  %3d  %s" % (g["seg"], g["static"], g["executed"] / 1e6, g["exec_pct"], g["samples"], g["samples_pct"],
                                                                                         g["ops"]["HMMA"], g["ops"]["LDS"], g["ops"]["LDG"], g["ops"]["STG"], g["ops"]["STS"], tops))
    lines.append("\nhottest instructions by stall samples:")
    for h in s["hottest"]:
        lines.append("  %s samples=%6.0f exec=%9.0f %-70s " % (h["address"], h["samples"], h["executed"], h["source"]) + " ".join("%s=%.0f" % (name, v) for name, v in h["stalls"]))
    lines.append("\nexecuted warp-instructions by opcode (all segments):")
    for o in s["opcodes"]:
        lines.append("  %-10s %8.2f M  %5.1f%%   static %d" % (o["op"], o["executed"] / 1e6, o["exec_pct"], o["static"]))
    return "\n".join(lines)


def build_record(meta, summary):
    return {"schema": SCHEMA, "meta": meta, "summary": summary}
