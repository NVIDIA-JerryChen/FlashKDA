"""Per-phase clock profile of the optimised row engine, from kernel entry, written as a JSON record (``cfbi-rows-profile-v1``).

usage: python benchmarks/micro/rows_profile.py [--shapes N ...] [--heads H] [--warmup PASSES] [--json PATH] [--label TEXT]

The tool builds the profile variant of the extension (``-DCFBI_ROWS_PROFILE=1``: ``csrc/cfbi/cfbi_rows.cuh`` stamps ``clock64`` at kernel entry and at eight points of every
row and keeps them in a device buffer that ``rows_profile_read`` returns), runs mode-1 decode of N sequences with H heads for 64 calls (every sequence at the same position,
advancing by one per call, so that every position 0 .. 63 is measured once per sequence and head) after ``--warmup`` passes of 64 calls, and aggregates the stamps of all CTAs
(``benchmarks/rows_profile_record.py``): the prologue, the seven row phases and the epilogue in cycles, as means over the same rows for the regular rows (positions 0 .. 62),
the chunk-end row (position 63) and all rows, with the phase shares of the row duration and the means of every position. The profile build adds clock reads and a few
stores: the cycle figures describe the instrumented build, the default build is unchanged.

CFBI_BENCH_SRC=<dir> profiles the kernels of another source tree (a ``git archive`` snapshot of a commit); the record then carries that tree's commit.
"""
import argparse
import datetime
import os
import socket
import sys
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parents[2]  # the tree that holds this script and the record modules
ROOT = Path(os.environ.get("CFBI_BENCH_SRC") or TOOL_ROOT).resolve()  # the tree whose kernels are profiled
for _p in (TOOL_ROOT / "tests" / "cfbi", TOOL_ROOT / "benchmarks", ROOT, ROOT / "tests" / "cfbi"):
    sys.path.insert(0, str(_p))

import bench_records as records  # noqa: E402
import rows_profile_record as rpr  # noqa: E402

DEFAULT_SHAPES = (1, 16, 128)
DEFAULT_HEADS = 12
VARIANT = ("CFBI_ROWS_PROFILE=1",)


def profile_shape(torch, mod, alloc_state, make_case, N, H, warmup):
    """The cell of one shape: ``warmup`` passes of 64 decode calls, a reset of the buffer, one recorded pass of 64 calls, the stamps of every row."""
    steps = rpr.POSITIONS
    case = make_case(N * steps, H, "authors", seed=0)
    st = alloc_state(N, H)

    def step(s):
        d = {x: torch.stack([case[x][0, n * steps + s] for n in range(N)], 0).unsqueeze(1).contiguous() for x in ("q", "k", "v", "g", "beta")}
        out = torch.zeros_like(d["q"])
        mod.forward(d["q"], d["k"], d["v"], d["g"], d["beta"], case["scale"], out, case["A_log"], case["dt_bias"], -5.0, st.E, st.rows, st.pos, None, 1, None, 0)

    for _ in range(warmup):
        for s in range(steps):
            step(s)
    torch.cuda.synchronize()
    mod.rows_profile_clear()
    for s in range(steps):
        step(s)
    torch.cuda.synchronize()
    buf = mod.rows_profile_read()
    if buf.numel() == 0:
        raise SystemExit("the extension is not a profile build (rows_profile_read returned an empty buffer)")
    if N * H > buf.shape[1]:
        raise SystemExit("N * H = %d CTAs exceed the %d slots of the profile buffer" % (N * H, buf.shape[1]))
    return rpr.cell_from_slots(buf[:, :N * H].tolist(), N, H)


def format_cell(cell):
    lines = ["N=%-3d H=%-2d  %d of %d rows recorded on %d SMs" % (cell["N"], cell["H"], cell["rows_recorded"], cell["rows_expected"], len(cell["sms"]))]
    for g in rpr.GROUPS:
        b = cell["rows"][g]
        if not b.get("rows"):
            continue
        lines.append("  %-9s %6d rows: prologue %7.0f | %s | epilogue %6.0f | row %7.0f cycles (kernel %7.0f)" % (
            g, b["rows"], b["prologue"], " ".join("%s %.0f (%.0f%%)" % (k, b["phases"][k], 100 * b["shares"][k]) for k in rpr.PHASES), b["epilogue"], b["row_total"], b["kernel_total"]))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", nargs="+", type=int, default=list(DEFAULT_SHAPES), help="numbers of sequences N (default 1 16 128)")
    ap.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    ap.add_argument("--warmup", type=int, default=1, help="passes of 64 calls before the recorded pass (default 1)")
    ap.add_argument("--json", default=None, help="write the record here")
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    if a.heads < 1 or a.warmup < 0 or any(n < 1 for n in a.shapes):
        ap.error("the shapes and the heads must be positive and the warm-up passes not negative")

    import torch
    from flash_kda.cfbi import alloc_state
    from flash_kda.cfbi._ext import load_variant
    from helpers import make_case

    mod = load_variant("rowsprof", VARIANT)
    prop = torch.cuda.get_device_properties(0)
    commit, dirty, tree = records.tree_identity(ROOT)
    meta = {"commit": commit, "dirty": dirty, "tree": tree, "src": str(ROOT), "gpu": prop.name, "sm_count": prop.multi_processor_count, "cc": "%d.%d" % (prop.major, prop.minor),
            "host": socket.gethostname(), "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"), "torch": torch.__version__,
            "cuda": torch.version.cuda, "variant": list(VARIANT), "heads": a.heads, "shapes": list(a.shapes), "positions": rpr.POSITIONS, "warmup_passes": a.warmup,
            "label": a.label, "tag": os.environ.get("CFBI_BENCH_TAG"), "build_dir": os.environ.get("FLASH_KDA_CFBI_BUILD_DIR"), "argv": sys.argv[1:]}
    cells = []
    for N in a.shapes:
        cell = profile_shape(torch, mod, alloc_state, make_case, N, a.heads, a.warmup)
        cells.append(cell)
        print(format_cell(cell), flush=True)
    if a.json:
        rpr.write_record(a.json, meta, cells)
        print("record written to %s" % a.json, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
