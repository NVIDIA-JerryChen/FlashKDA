"""Top SASS instructions by warp-stall samples from an ncu report: python umma_k2_ncu_top.py <report.ncu-rep> [N]"""
import csv, io, subprocess, sys
rep = sys.argv[1]; topn = int(sys.argv[2]) if len(sys.argv) > 2 else 40
out = subprocess.run(["ncu", "-i", rep, "--page", "source", "--print-source", "sass", "--csv", "--print-units", "base"], capture_output=True, text=True).stdout
lines = [l for l in out.splitlines() if l.strip()]
# find the header
hi = next(i for i, l in enumerate(lines) if l.startswith('"Address"') or l.startswith('"Source"') or "Warp Stall Sampling (All Samples)" in l)
rd = csv.DictReader(io.StringIO("\n".join(lines[hi:])))
rows = list(rd)
key = [c for c in rows[0].keys() if "Warp Stall Sampling (All" in c][0]
src = [c for c in rows[0].keys() if c.strip('"') in ("Source", "SASS")][0]
def num(x):
    try: return float(x.replace(",", ""))
    except: return 0.0
tot = sum(num(r[key]) for r in rows)
print(f"total stall samples {tot:.0f}; columns: {list(rows[0].keys())[:12]}")
for r in sorted(rows, key=lambda r: -num(r[key]))[:topn]:
    print(f"{num(r[key]):9.0f} {100*num(r[key])/max(tot,1):5.1f}%  {r.get(src,'')[:110]}")
