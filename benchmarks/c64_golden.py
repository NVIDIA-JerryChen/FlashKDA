"""Save / compare a golden fwd64 + replay output (bitwise regression check across rebuilds that must not change bits).
usage: python benchmarks/c64_golden.py save|check [path]"""
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import c64_helpers as Hh
from flash_kda import replay64 as R
mode = sys.argv[1]
path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/c64_golden.pt"
outs = {}
for name, T, H, lens in (("a", 1000, 3, None), ("b", 700, 2, [100, 64, 300, 236]), ("c", 333, 4, [1, 64, 268])):
    case = Hh.make_case(T, H, "heavy_tail", seed=11)
    q, k, v, g, beta = (case[n] for n in ("q", "k", "v", "g", "beta"))
    cu = None if lens is None else torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int64, device="cuda")
    N = 1 if lens is None else len(lens)
    out = torch.zeros_like(q); st = torch.zeros(N, H, 128, 128, dtype=torch.bfloat16, device="cuda")
    R.fwd64(q, k, v, g, beta, case["scale"], out, case["A_log"], case["dt_bias"], case["lower_bound"], final_state=st, cu_seqlens=cu)
    s = R.alloc_state(1, H); o2 = torch.zeros(1, 1, H, 128, dtype=torch.bfloat16, device="cuda"); dec = []
    for t in range(min(T, 150)):
        R.decode(q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], case["scale"], o2[:, 0], case["A_log"], case["dt_bias"], case["lower_bound"], s)
        dec.append(o2.clone())
    outs[name] = (out, st, torch.cat(dec, 1))
torch.cuda.synchronize()
if mode == "save":
    torch.save({k: tuple(t.cpu() for t in v) for k, v in outs.items()}, path)
    print("[golden] saved", path)
else:
    ref = torch.load(path)
    for k, v in outs.items():
        print(f"[golden] {k}: differing elements vs saved = {[Hh.n_diff(a.cpu(), b) for a, b in zip(v, ref[k])]}")
