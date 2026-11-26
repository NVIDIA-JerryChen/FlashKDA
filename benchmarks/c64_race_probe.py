"""Minimal calls for compute-sanitizer racecheck: one fwd64 call per K2 shape, then one replay decode."""
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import c64_helpers as Hh
from flash_kda import replay64 as R
which = sys.argv[1]
case = Hh.make_case(100, 2, "authors", seed=1)
q, k, v, g, beta = (case[n] for n in ("q", "k", "v", "g", "beta"))
out = torch.zeros_like(q)
if which.startswith("fwd"):
    R.fwd64(q, k, v, g, beta, case["scale"], out, case["A_log"], case["dt_bias"], case["lower_bound"], _k2_cfg=int(which[3:]))
else:
    st = R.alloc_state(1, 2)
    R.prefill(q[:, :70], k[:, :70], v[:, :70], g[:, :70], beta[:, :70], case["scale"], out[:, :70], case["A_log"], case["dt_bias"], case["lower_bound"], st, _path="replay")
torch.cuda.synchronize()
print("[probe] done", which)
