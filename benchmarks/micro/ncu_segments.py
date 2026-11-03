"""Summarise an Nsight Compute SASS source CSV (ncu --page source --csv --print-source sass) per barrier-delimited segment.
usage: python ncu_segments.py <csv> [topN]"""
import csv, sys
from collections import Counter, defaultdict

path = sys.argv[1]
topn = int(sys.argv[2]) if len(sys.argv) > 2 else 12
rows = []
with open(path, newline="") as f:
    lines = f.read().splitlines()
hdr_i = next(i for i, l in enumerate(lines) if l.startswith('"Address"'))
rd = csv.DictReader(lines[hdr_i:])
stall_cols = [c for c in rd.fieldnames if c.startswith("stall_") and "Not Issued" not in c]
for r in rd:
    rows.append(r)
def num(x):
    try: return float(x)
    except: return 0.0
tot_exec = sum(num(r["Instructions Executed"]) for r in rows)
tot_samp = sum(num(r["Warp Stall Sampling (All Samples)"]) for r in rows)
print(f"{len(rows)} static instructions, {tot_exec:.0f} warp-instructions executed, {tot_samp:.0f} stall samples")
segs, cur = [], []
for r in rows:
    cur.append(r)
    if "BAR.SYNC" in r["Source"] or "BAR.ARV" in r["Source"]:
        segs.append(cur); cur = []
if cur: segs.append(cur)
print(f"{'seg':>3} {'static':>6} {'exec(M)':>8} {'%exec':>6} {'samples':>8} {'%samp':>6}  mma  lds  ldg  stg  sts  top stalls")
for i, s in enumerate(segs):
    ex = sum(num(r["Instructions Executed"]) for r in s)
    sm = sum(num(r["Warp Stall Sampling (All Samples)"]) for r in s)
    mma = sum(1 for r in s if "HMMA" in r["Source"] or "MMA" in r["Source"].split()[0:2].__str__())
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
    print(f"{i:>3} {len(s):>6} {ex/1e6:>8.2f} {ex/tot_exec*100:>5.1f}% {sm:>8.0f} {sm/tot_samp*100:>5.1f}%  {cnt['HMMA']:>3}  {cnt['LDS']:>3}  {cnt['LDG']:>3}  {cnt['STG']:>3}  {cnt['STS']:>3}  {tops}")
print("\nhottest instructions by stall samples:")
for r in sorted(rows, key=lambda r: -num(r["Warp Stall Sampling (All Samples)"]))[:topn]:
    st = sorted(((c, num(r[c])) for c in stall_cols), key=lambda kv: -kv[1])[:2]
    print(f"  {r['Address'][-5:]} samples={num(r['Warp Stall Sampling (All Samples)']):>6.0f} exec={num(r['Instructions Executed']):>9.0f} {r['Source'].strip()[:70]:<70} " + " ".join(f"{k[6:]}={v:.0f}" for k, v in st))

print("\nexecuted warp-instructions by opcode (all segments):")
byop = Counter(); static = Counter()
for r in rows:
    toks = r["Source"].split()
    if not toks: continue
    op = toks[0]
    if op.startswith("@"): op = toks[1] if len(toks) > 1 else op
    byop[op.split(".")[0]] += num(r["Instructions Executed"]); static[op.split(".")[0]] += 1
for op, v in byop.most_common(28):
    print(f"  {op:<10} {v/1e6:>8.2f} M  {v/tot_exec*100:>5.1f}%   static {static[op]}")
