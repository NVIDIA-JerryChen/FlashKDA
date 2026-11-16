"""Dump the results of a fixed CF-BI workload and of the raw hardware primitives of THIS device to a .pt file, so that
different GPU architectures can be compared bit for bit with ``arch_compare.py`` (run on each machine, compare on any
CPU; the files only need a shared directory).

    python tests/cfbi/arch_dump.py OUT.pt [--impl N]       # N: 0 default kernels, 3 reference, 7 reference rows-only

What is dumped (all CPU tensors, nothing is compared here):
  * ``w/<regime>/{out,out32,E,rows,pos,train,train32}``: three prefill calls (cuts at 37 % / 71 % / 100 % of every
    sequence, varlen, 6 sequences x 4 heads, one open chunk left) and the training forward of the whole workload, for
    three data regimes (the bf16 outputs, the fp32 values before the bf16 rounding, the state, the training forward);
  * ``p/dot3_*``: the three-term TF32 product ``probe_dot3`` on fixed random tiles (K = 128; normal and wide exponents);
  * ``p/ex2``, ``p/rcp``, ``p/rsqrt``: the approximate instructions of the contract on fixed grids (a strided sweep of
    mantissas / of the exponent range);
  * ``meta``: device, compute capability, versions, the effective implementation, and SHA-256 hashes of every INPUT
    (the comparison is only meaningful if the inputs are bit-identical on both machines: the hashes say so).
Primitive inputs come from numpy's PCG64 integer stream and exact float arithmetic (identical on every machine).
The workload inputs are NOT regenerated per machine: ``torch.randn`` on the GPU is not bit-identical across GPU
models (the L40S differs from the H100 in q/k/v/g), so the inputs are generated once on the CPU
(``helpers.make_case(device="cpu")``) into a shared ``--inputs`` file that every machine loads; ``arch_compare.py`` still
checks the input hashes.
"""
import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from flash_kda.cfbi import alloc_state, forward_train, load_ext, prefill, verify_hardware
from flash_kda.cfbi.verify import _impl_name
from helpers import make_case

REGIMES = ("authors", "heavy_tail", "long_memory")
LENS = [150, 1, 64, 129, 33, 300]
H = 4
CUTS = (0.37, 0.71, 1.0)


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def uniform(n, seed):
    """n floats in [-1, 1) from the PCG64 integer stream and exact arithmetic (identical on every machine)."""
    ints = np.random.default_rng(seed).integers(0, 2**32, size=n, dtype=np.uint32)
    return ((ints >> 8).astype(np.float32) * np.float32(2.0 ** -23)) - np.float32(1.0)


def pow2(n, seed, lo, hi):
    ints = np.random.default_rng(seed).integers(lo, hi + 1, size=n)
    return np.exp2(ints.astype(np.float32)).astype(np.float32)


def load_inputs(path):
    """The shared workload inputs (CPU tensors per regime); generated on first use if the file does not exist (do
    that once, on one machine, before the dumps start)."""
    if not os.path.exists(path):
        cases = {r: make_case(sum(LENS), H, r, seed=99, device="cpu") for r in REGIMES}
        tmp = f"{path}.tmp{os.getpid()}"
        torch.save(cases, tmp)
        os.replace(tmp, path)
    return torch.load(path, map_location="cpu", weights_only=False)


def workload(regime, impl, inputs, tensors, hashes):
    case = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inputs[regime].items()}
    for key in ("q", "k", "v", "g", "beta", "A_log", "dt_bias"):
        hashes[f"w/{regime}/in_{key}"] = sha(case[key])
    cu = [0]
    for L in LENS:
        cu.append(cu[-1] + L)
    N = len(LENS)
    st = alloc_state(N, H)
    done, outs, outs32 = [0] * N, [], []
    for cut in CUTS:
        take = [int(L * cut) - d for L, d in zip(LENS, done)]
        pieces = [{k: case[k][:, cu[n] + done[n]:cu[n] + done[n] + take[n]] for k in ("q", "k", "v", "g", "beta")} for n in range(N)]
        sub = {k: torch.cat([p[k] for p in pieces], dim=1).contiguous() for k in ("q", "k", "v", "g", "beta")}
        cu_call = [0]
        for t in take:
            cu_call.append(cu_call[-1] + t)
        out = torch.zeros_like(sub["q"])
        out32 = torch.zeros(sub["q"].shape, dtype=torch.float32, device="cuda")
        prefill(sub["q"], sub["k"], sub["v"], sub["g"], sub["beta"], case["scale"], out, case["A_log"], case["dt_bias"],
                case["lower_bound"], st, torch.tensor(cu_call, dtype=torch.int64, device="cuda"), out_fp32=out32, _impl=impl)
        outs.append(out)
        outs32.append(out32)
        done = [d + t for d, t in zip(done, take)]
    valid = (torch.arange(64, device="cuda")[None, :] < st.pos[:, None])[:, None, None, :, None]
    t32 = torch.zeros(case["q"].shape, dtype=torch.float32, device="cuda")
    ot = forward_train(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], case["A_log"],
                       case["dt_bias"], case["lower_bound"], torch.tensor(cu, dtype=torch.int64, device="cuda"),
                       out_fp32=t32, _impl=impl)
    for name, t in (("out", torch.cat(outs, 1)), ("out32", torch.cat(outs32, 1)), ("E", st.E), ("rows", st.rows * valid),
                    ("pos", st.pos), ("train", ot), ("train32", t32)):
        tensors[f"w/{regime}/{name}"] = t.detach().cpu()


def primitives(ext, tensors, hashes):
    K = 128
    for name, ntiles, wide in (("dot3_normal", 2048, False), ("dot3_wide", 1024, True)):
        a = uniform(ntiles * 16 * K, 1 + wide) * 3.0
        b = uniform(ntiles * K * 8, 11 + wide) * 3.0
        if wide:
            a = a * pow2(a.size, 21, -30, 30)
            b = b * pow2(b.size, 22, -30, 30)
        A = torch.from_numpy(a).view(ntiles, 16, K).cuda()
        B = torch.from_numpy(b).view(ntiles, K, 8).cuda()
        hashes[f"p/{name}/in"] = sha(A) + sha(B)
        tensors[f"p/{name}"] = ext.probe_dot3(A, B, False).cpu()
        # sum |a*b| of every output element (fp64): the natural scale of the product's rounding error
        tensors[f"p/{name}_scale"] = torch.einsum("tik,tkj->tij", A.double().abs(), B.double().abs()).float().cpu()
    n = 1 << 21
    i = np.arange(n, dtype=np.int64)
    grids = {
        # ex2: a strided sweep of [-126, 0) (the contract's range, underflow below 2^-126 is flushed) and of [0, 1)
        "ex2": np.concatenate([-126.0 + i[: n // 2].astype(np.float32) * np.float32(126.0 / (n // 2)),
                               i[: n // 2].astype(np.float32) * np.float32(1.0 / (n // 2))]).astype(np.float32),
        # rcp / rsqrt: every 4th mantissa of the octaves [1, 2) and [2, 4) (the function repeats on the other octaves)
        "rcp": (np.uint32(0x3F800000) + (i.astype(np.uint32) * np.uint32(4))).view(np.float32),
        "rsqrt": np.concatenate([(np.uint32(0x3F800000) + (i[: n // 2].astype(np.uint32) * np.uint32(8))).view(np.float32),
                                 (np.uint32(0x40000000) + (i[: n // 2].astype(np.uint32) * np.uint32(8))).view(np.float32)]),
    }
    for which, (name, x) in enumerate((("ex2", grids["ex2"]), ("rsqrt", grids["rsqrt"]), ("rcp", grids["rcp"]))):
        X = torch.from_numpy(np.ascontiguousarray(x)).cuda()
        hashes[f"p/{name}/in"] = sha(X)
        # probe_transc: 0 = ex2.approx.ftz, 2 = rsqrt.approx.ftz, 3 = rcp.approx.ftz
        tensors[f"p/{name}"] = ext.probe_transc(X, {"ex2": 0, "rsqrt": 2, "rcp": 3}[name]).cpu()
        tensors[f"p/{name}_x"] = X.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--impl", type=int, default=int(os.environ.get("FLASH_KDA_CFBI_IMPL", "0")))
    ap.add_argument("--inputs", default=None, help="shared workload inputs (default: arch_inputs.pt next to OUT)")
    args = ap.parse_args()
    inputs = load_inputs(args.inputs or os.path.join(os.path.dirname(os.path.abspath(args.out)), "arch_inputs.pt"))
    ext = load_ext()
    verdict = verify_hardware()
    prop = torch.cuda.get_device_properties(0)
    hashes, tensors = {}, {}
    for regime in REGIMES:
        workload(regime, args.impl, inputs, tensors, hashes)
    primitives(ext, tensors, hashes)
    eff = int(ext.resolve_impl(args.impl, -1))
    meta = dict(device=prop.name, capability=f"{prop.major}.{prop.minor}", torch=torch.__version__, cuda=torch.version.cuda,
                impl_requested=args.impl, impl_effective=eff, kernels=_impl_name(eff), premise_ok=verdict["ok"],
                input_hashes=hashes)
    torch.save(dict(meta=meta, tensors=tensors), args.out)
    h = hashlib.sha256()
    for k in sorted(tensors):
        if not (k.endswith("_x") or k.endswith("_scale")):
            h.update(k.encode())
            h.update(tensors[k].contiguous().view(torch.uint8).numpy().tobytes())
    print(f"[arch_dump] {prop.name} sm_{prop.major}{prop.minor} impl {args.impl} -> effective {meta['impl_effective']} "
          f"({meta['kernels']}), {len(tensors)} tensors, SHA-256 of all results {h.hexdigest()}")


if __name__ == "__main__":
    main()
