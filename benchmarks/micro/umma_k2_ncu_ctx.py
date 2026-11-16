"""Show the SASS context of the instructions with the most stall samples: python umma_k2_ncu_ctx.py <csv> [ntop] [ctx]"""
import csv, io, sys
lines = open(sys.argv[1]).read().splitlines()
ntop = int(sys.argv[2]) if len(sys.argv) > 2 else 6
ctx = int(sys.argv[3]) if len(sys.argv) > 3 else 14
hi = next(i for i, l in enumerate(lines) if l.startswith('"Address"'))
rows = list(csv.DictReader(io.StringIO("\n".join(lines[hi:]))))
key = [c for c in rows[0] if "Warp Stall Sampling (All" in c][0]
def num(x):
    try: return float(x.replace(",", ""))
    except: return 0.0
order = sorted(range(len(rows)), key=lambda i: -num(rows[i][key]))[:ntop]
for i in order:
    print("=" * 30, "stall samples", rows[i][key], "at row", i)
    for r in rows[max(0, i - ctx): i + 3]:
        print("%7s %9s  %s" % (r[key], r["Instructions Executed"], r["Source"][:110]))
