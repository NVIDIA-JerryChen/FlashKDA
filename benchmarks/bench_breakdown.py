"""Per-kernel GPU time of the CF-BI pipeline (plan / row engine / K1 / K2) for one implementation.

usage: python benchmarks/bench_breakdown.py [--impl N] [--prefill B,L,H ...] [--varlen SPEC ...] [--decode N,H ...] [--train B,L,H ...]
           [--warm POS] [--k2bv 0|8|16|32] [--repeat R] [--iters I] [--json PATH] [--label TEXT] [--list-corpus]
  --impl     0 = optimised kernels (default; a --varlen call that the routing rule predicts to gain, on a Blackwell device, also gets the device-scheduled K2, see
             csrc/cfbi/cfbi_launch.h), 3 = the frozen v1 reference kernels,
             8 = optimised row engine + K1 with the tcgen05 (UMMA) K2 (kernel k2u_kernel instead of k2_kernel),
             16 = K1 + the K2 driven by a schedule built on the device (kernels k2_sched_kernel, k2_dyn_kernel), 32 = the old K2 kernels
             also where the default would route to the device-scheduled one (the baseline of a comparison with 16)
  --prefill  one or more "B,L,H" shapes (default: the CF-BI report's H6/L8K shapes)
  --varlen   one or more variable-length prefill calls (one cu_seqlens call, B = 1): a corpus entry (V1 .. V7d, see --list-corpus), a corpus
             entry with another head count ("V2@12") or an explicit list ("8192+7*1024@6": terms LEN or COUNT*LEN joined by +, head count after @)
  --decode   one or more "N,H" shapes (decode averaged over 64 consecutive steps = every chunk phase)
  --warm     chunk phase (pos) the sequences start the prefill call at (exercises the row engine's head rows)
  --repeat   in-process repeats of every cell (each repeat profiles the call --iters times); the printed line shows the mean
  --iters    profiled calls per repeat of a prefill / varlen / train cell (default 3; more calls average out run-to-run noise)
  --json     write the measurements of this run as a record file (schema in benchmarks/bench_records.py): one record per process run, to be
             compared with benchmarks/compare_bench.py
  --label    free text stored in the record (for example the variant name)
Times are the sum of kernel durations from the CUDA profiler (no launch overhead), microseconds per call.

The tool measures the flash_kda of its own source tree. CFBI_BENCH_SRC=<dir> measures the flash_kda of another tree (for example a git archive
snapshot of a baseline commit) with this tool; the record then carries that tree's commit.
"""
import argparse
import datetime
import os
import socket
import sys
from collections import defaultdict
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parents[1]  # the tree that holds this script, bench_records.py and varlen_corpus.py
ROOT = Path(os.environ.get("CFBI_BENCH_SRC") or TOOL_ROOT).resolve()  # the tree whose flash_kda is measured
sys.path.insert(0, str(TOOL_ROOT / "tests" / "cfbi"))
sys.path.insert(0, str(TOOL_ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "cfbi"))

import torch
from torch.profiler import ProfilerActivity, profile

import bench_records as records
from flash_kda.cfbi import alloc_state, decode, forward_train, prefill
from helpers import make_case
from varlen_corpus import CORPUS_V, cu_seqlens_of, parse_spec

LB = -5.0
REPEAT = 1
ITERS = 3
CELLS = []


def kernel_times(fn, iters=5, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    acc = defaultdict(float)
    cnt = defaultdict(int)
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            name = ev.name.split("(")[0].split("<")[0]
            acc[name] += getattr(ev, "device_time", None) or getattr(ev, "cuda_time", 0.0)
            cnt[name] += 1
    return {k: v / iters for k, v in acc.items()}, {k: c / iters for k, c in cnt.items()}


def short(name):
    for key in ("plan_kernel", "finalize_pos_kernel", "rows_kernel", "k1_kernel", "k2_kernel", "k1_", "k2_", "rows_"):
        if key in name:
            return ("ref::" if "ref" in name else "") + name.split("cfbi")[-1].strip(":") if False else name
    return name


def report(label, times, counts):
    total = sum(times.values())
    parts = "  ".join(f"{k.replace('cfbi::', '')}={v:9.1f}" + (f"(x{counts[k]:.0f})" if counts[k] > 1.01 else "")
                      for k, v in sorted(times.items(), key=lambda kv: -kv[1]))
    print(f"{label:34s} total {total:10.1f} us   {parts}", flush=True)


def emit(kind, label, args, fn, iters, warm=3):
    """Measure ``fn`` REPEAT times, print the mean line and keep every repeat for the JSON record."""
    runs = [kernel_times(fn, iters=iters, warm=warm if i == 0 else 1) for i in range(REPEAT)]
    mean_t, mean_c = defaultdict(float), defaultdict(float)
    for t, c in runs:
        for k, v in t.items():
            mean_t[k] += v / len(runs)
        for k, v in c.items():
            mean_c[k] += v / len(runs)
    report(label, mean_t, mean_c)
    CELLS.append(records.make_cell(kind, label, args, [t for t, _ in runs]))


def bench_prefill(B, L, H, impl, warm_pos, k2_bv=0):
    case = make_case(B * L, H, "authors", seed=0)
    q = case["q"].reshape(B, L, H, 128).contiguous()
    k = case["k"].reshape(B, L, H, 128).contiguous()
    v = case["v"].reshape(B, L, H, 128).contiguous()
    g = case["g"].reshape(B, L, H, 128).contiguous()
    beta = case["beta"].reshape(B, L, H).contiguous()
    out = torch.zeros_like(q)
    st = alloc_state(B, H)
    if warm_pos:
        st.pos.fill_(warm_pos)

    def f():
        if warm_pos:
            st.pos.fill_(warm_pos)
        prefill(q, k, v, g, beta, case["scale"], out, case["A_log"], case["dt_bias"], LB, st, _impl=impl, _k2_bv=k2_bv)

    emit("prefill", f"prefill B={B} L={L} H={H} pos0={warm_pos}", {"B": B, "L": L, "H": H, "pos0": warm_pos, "impl": impl, "k2_bv": k2_bv}, f, iters=ITERS)
    del q, k, v, g, beta, out, st
    torch.cuda.empty_cache()


def bench_varlen(spec, impl, k2_bv=0):
    """One cu_seqlens prefill call (B = 1): sequence lengths and head count from a corpus entry or an explicit list."""
    name, lens, H = parse_spec(spec)
    T = sum(lens)
    case = make_case(T, H, "authors", seed=0)
    cu = torch.tensor(cu_seqlens_of(lens), dtype=torch.int64, device="cuda")
    out = torch.zeros_like(case["q"])
    st = alloc_state(len(lens), H)

    def f():
        prefill(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], out, case["A_log"], case["dt_bias"], LB, st,
                cu_seqlens=cu, _impl=impl, _k2_bv=k2_bv)

    emit("varlen", f"varlen {name} N={len(lens)} H={H} T={T}",
         {"spec": spec, "name": name, "lens": lens, "N": len(lens), "H": H, "T": T, "impl": impl, "k2_bv": k2_bv}, f, iters=ITERS)
    del case, cu, out, st
    torch.cuda.empty_cache()


def bench_train(B, L, H, impl):
    case = make_case(B * L, H, "authors", seed=0)
    q = case["q"].reshape(B, L, H, 128).contiguous()
    k = case["k"].reshape(B, L, H, 128).contiguous()
    v = case["v"].reshape(B, L, H, 128).contiguous()
    g = case["g"].reshape(B, L, H, 128).contiguous()
    beta = case["beta"].reshape(B, L, H).contiguous()
    emit("train", f"train-fwd B={B} L={L} H={H}", {"B": B, "L": L, "H": H, "impl": impl},
         lambda: forward_train(q, k, v, g, beta, case["scale"], case["A_log"], case["dt_bias"], LB, _impl=impl), iters=ITERS)


def bench_decode(N, H, impl):
    steps = 64
    case = make_case(N * steps, H, "authors", seed=0)
    st = alloc_state(N, H)
    # per-step contiguous [N,1,H,*] inputs; sequences advance one token per call (all 64 chunk phases are visited)
    per_step = []
    for s in range(steps):
        d = {key: torch.stack([case[key][0, n * steps + s] for n in range(N)], dim=0).unsqueeze(1).contiguous()
             for key in ("q", "k", "v", "g", "beta")}
        per_step.append(d)
    out = torch.zeros_like(per_step[0]["q"])
    i = [0]

    def f():
        d = per_step[i[0] % steps]
        i[0] += 1
        decode(d["q"], d["k"], d["v"], d["g"], d["beta"], case["scale"], out, case["A_log"], case["dt_bias"], LB, st, _impl=impl)

    emit("decode", f"decode N={N} H={H} (64-step avg)", {"N": N, "H": H, "impl": impl}, f, iters=64, warm=2)


def run_meta(a, started):
    commit, dirty, tree = records.tree_identity(ROOT)
    prop = torch.cuda.get_device_properties(0)
    return {"commit": commit, "dirty": dirty, "tree": tree, "gpu": prop.name, "sm_count": prop.multi_processor_count,
            "cc": "%d.%d" % (prop.major, prop.minor), "host": socket.gethostname(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "timestamp": started, "impl": a.impl, "k2_bv": a.k2bv, "warm_pos": a.warm, "repeat": a.repeat, "iters": a.iters, "label": a.label,
            "env_impl": os.environ.get("FLASH_KDA_CFBI_IMPL"), "tag": os.environ.get("CFBI_BENCH_TAG"),
            "build_dir": os.environ.get("FLASH_KDA_CFBI_BUILD_DIR"), "src": str(ROOT), "argv": sys.argv[1:]}


def main():
    global REPEAT, ITERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", type=int, default=0)
    ap.add_argument("--prefill", nargs="*", default=None)
    ap.add_argument("--varlen", nargs="*", default=None)
    ap.add_argument("--decode", nargs="*", default=None)
    ap.add_argument("--train", nargs="*", default=None)
    ap.add_argument("--warm", type=int, default=0)
    ap.add_argument("--k2bv", type=int, default=int(os.environ.get("CFBI_BENCH_K2BV", "0")), help="force the K2 value-tile width (8, 16 or 32); 0 = automatic")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--json", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--list-corpus", action="store_true")
    a = ap.parse_args()
    if a.list_corpus:
        for e in CORPUS_V.values():
            print(f"{e.name:5s} N={len(e.lens):3d} H={e.H:2d} T={sum(e.lens):7d}  {e.note}")
        return
    REPEAT = max(1, a.repeat)
    ITERS = max(1, a.iters)
    started = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    print(f"device {torch.cuda.get_device_name()}  impl={a.impl}")
    pre = a.prefill or []
    dec = a.decode or []
    if a.prefill is None and a.varlen is None and a.train is None and a.decode is None:  # no shape option: the default shapes
        pre = ["1,8192,6", "8,8192,6", "64,8192,6", "1,32768,32"]
        dec = ["1,12", "16,12", "64,12", "128,12"]
    for s in pre:
        B, L, H = (int(x) for x in s.split(","))
        bench_prefill(B, L, H, a.impl, a.warm, a.k2bv)
    for s in a.varlen or []:
        bench_varlen(s, a.impl, a.k2bv)
    for s in a.train or []:
        B, L, H = (int(x) for x in s.split(","))
        bench_train(B, L, H, a.impl)
    for s in dec:
        N, H = (int(x) for x in s.split(","))
        bench_decode(N, H, a.impl)
    if a.json:
        records.write_record(a.json, run_meta(a, started), CELLS)
        print(f"record written to {a.json}", flush=True)


if __name__ == "__main__":
    main()
