"""Dump the SASS around the UMMA instructions with stall samples / execution counts: python umma_k2_ncu_sass.py <csv> [before after]"""
import csv, io, sys
lines = open(sys.argv[1]).read().splitlines()
before = int(sys.argv[2]) if len(sys.argv) > 2 else 60
after = int(sys.argv[3]) if len(sys.argv) > 3 else 80
hi = next(i for i, l in enumerate(lines) if l.startswith('"Address"'))
rows = list(csv.DictReader(io.StringIO("\n".join(lines[hi:]))))
key = [c for c in rows[0] if "Warp Stall Sampling (All" in c][0]
idx = [i for i, r in enumerate(rows) if "UTC" in r["Source"] or "UTMA" in r["Source"] or "UTMMA" in r["Source"]]
print("rows", len(rows), "UTC rows", len(idx), [rows[i]["Source"][:60] for i in idx[:6]])
pick = int(sys.argv[4]) if len(sys.argv) > 4 else 0
first = idx[pick] if idx else 0
for r in rows[max(0, first - before): first + after]:
    print("%7s %9s  %s" % (r[key], r["Instructions Executed"], r["Source"][:110]))
