"""Compare ``arch_dump.py`` files bit for bit (runs on any CPU; torch is only used to read the files).

    python tests/cfbi/arch_compare.py A.pt B.pt [C.pt ...]      # every file against the first one

Per tensor group: elements, number of elements whose bits differ, share, and the largest distance in ULPs (bf16 / fp32
ordered-integer distance). Input hashes are compared first: if the inputs of the two machines are not bit-identical the
comparison of the workload results says nothing (the primitive probes use machine-independent inputs).
"""
import sys
from collections import OrderedDict

import torch


def ordered(t: torch.Tensor) -> torch.Tensor:
    """Monotone integer image of a float tensor (so that |a - b| counts ULPs)."""
    if t.dtype == torch.bfloat16:
        i = t.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
        return torch.where(i >= 0x8000, 0x8000 - i, i)
    i = t.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return torch.where(i >= 0x80000000, 0x80000000 - i, i)


def diff_stats(a: torch.Tensor, b: torch.Tensor):
    assert a.shape == b.shape and a.dtype == b.dtype, (a.shape, b.shape, a.dtype, b.dtype)
    if a.dtype in (torch.bfloat16, torch.float32):
        d = (ordered(a) - ordered(b)).abs()
        return a.numel(), int((d != 0).sum()), int(d.max()) if d.numel() else 0
    ne = (a != b)
    return a.numel(), int(ne.sum()), int(ne.any())


def group_of(key: str) -> str:
    parts = key.split("/")
    if parts[0] == "w":
        return f"workload {parts[2]}"   # out (bf16), out32, E, rows, pos, train (bf16), train32
    return f"primitive {parts[1]}"


def main(paths):
    dumps = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    base = dumps[0]
    print(f"reference: {paths[0]}\n  {base['meta']['device']} sm_{base['meta']['capability'].replace('.', '')}, "
          f"{base['meta']['kernels']} (impl {base['meta']['impl_requested']} -> {base['meta']['impl_effective']})")
    for path, d in zip(paths[1:], dumps[1:]):
        m = d["meta"]
        print(f"\n=== {path}\n  {m['device']} sm_{m['capability'].replace('.', '')}, {m['kernels']} "
              f"(impl {m['impl_requested']} -> {m['impl_effective']})")
        bad_inputs = [k for k, v in base["meta"]["input_hashes"].items() if m["input_hashes"].get(k) != v]
        print(f"  input hashes: {'identical' if not bad_inputs else 'DIFFERENT for ' + ', '.join(bad_inputs)}")
        groups = OrderedDict()
        for key, ta in base["tensors"].items():
            if key.endswith("_x") or key.endswith("_scale"):
                continue
            n, nd, mx = diff_stats(ta, d["tensors"][key])
            g = groups.setdefault(group_of(key), [0, 0, 0])
            g[0] += n
            g[1] += nd
            g[2] = max(g[2], mx)
        total_diff = 0
        for g, (n, nd, mx) in groups.items():
            total_diff += nd
            print(f"  {g:28s} elements {n:>10d}  differing {nd:>9d} ({nd / max(n, 1):.4%})  max ULP distance {mx}")
        for name in ("dot3_normal", "dot3_wide"):   # size of the product differences relative to sum|a*b|
            a, b, sc = base["tensors"][f"p/{name}"], d["tensors"][f"p/{name}"], base["tensors"][f"p/{name}_scale"]
            rel = ((a.double() - b.double()).abs() / sc.double().clamp_min(1e-300))
            print(f"  {name}: max |difference| / sum|a*b| = {float(rel.max()):.3e}, mean = {float(rel.mean()):.3e}")
        print(f"  => {'BIT-IDENTICAL on every dumped tensor' if total_diff == 0 else 'DIFFERENT: ' + str(total_diff) + ' differing elements in total'}")
        if total_diff:   # per regime for the workload outputs
            for regime in sorted({k.split('/')[1] for k in base["tensors"] if k.startswith("w/")}):
                row = []
                for name in ("out", "out32", "E", "rows", "train", "train32"):
                    key = f"w/{regime}/{name}"
                    n, nd, _ = diff_stats(base["tensors"][key], d["tensors"][key])
                    row.append(f"{name} {nd}/{n}")
                print(f"     {regime:11s}: " + ", ".join(row))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1:])
