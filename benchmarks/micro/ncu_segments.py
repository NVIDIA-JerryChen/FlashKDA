"""Summarise an Nsight Compute SASS source CSV (ncu --page source --csv --print-source sass) per barrier-delimited segment.
usage: python ncu_segments.py <csv> [topN] [--json OUT [--meta META.json]]

The text is the table of the barrier-delimited segments (static and executed instructions, stall samples, instruction counts and the top stall reasons of each), the instructions with
the most stall samples and the executed instructions by opcode. ``--json`` also writes the summary as a ``cfbi-ncu-segments-v1`` record (see ``benchmarks/ncu_record.py``): the
metadata of the collection (device, commit, case, ncu settings, hash of the report) come from the JSON file given with ``--meta``, the hash of the CSV is added."""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ncu_record as nr  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("topn", nargs="?", type=int, default=12)
    ap.add_argument("--json", default=None, help="write the summary as a cfbi-ncu-segments-v1 record")
    ap.add_argument("--meta", default=None, help="JSON file with the metadata of the collection (for --json)")
    a = ap.parse_args(argv)
    rows, stall_cols = nr.load_rows(a.csv)
    summary = nr.summarise(rows, stall_cols, a.topn)
    print(nr.format_summary(summary))
    if a.json:
        meta = {}
        if a.meta:
            with open(a.meta) as f:
                meta = json.load(f)
        meta["source_csv"] = {"name": os.path.basename(a.csv), "sha256": sha256(a.csv), "bytes": os.path.getsize(a.csv)}
        with open(a.json, "w") as f:
            json.dump(nr.build_record(meta, summary), f, indent=1, sort_keys=True)
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
