"""Bring-up checks of the chunk-64 kernels: accuracy of fwd64 against the fp64 recurrence, replay == fwd64 bits.

usage: python benchmarks/c64_bringup.py [quick]
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import c64_helpers as Hh  # noqa: E402
import flash_kda  # noqa: E402
from flash_kda import replay64 as R  # noqa: E402

torch.manual_seed(0)
dev = "cuda"


def fwd64_out(case):
    q, k, v, g, beta = (case[n] for n in ("q", "k", "v", "g", "beta"))
    out = torch.zeros_like(q)
    R.fwd64(q, k, v, g, beta, case["scale"], out, case["A_log"], case["dt_bias"], case["lower_bound"])
    torch.cuda.synchronize()
    return out


def acc(T, H, regime):
    case = Hh.make_case(T, H, regime, seed=1)
    out = fwd64_out(case)
    ref = Hh.exact_recurrence(case)
    refbf = ref.to(torch.bfloat16)
    floor = Hh.rel_rms(refbf, ref)
    rms = Hh.rel_rms(out, ref)
    fin = bool(torch.isfinite(out.float()).all())
    print(f"[acc] T={T:5d} H={H} {regime:12s} rms/floor = {rms / floor:6.2f}x  flip = {float((out != refbf).float().mean()):.1%}"
          f"  finite={fin}", flush=True)
    return out, case


def replay_vs_fwd(T, H, regime="authors", pieces=None):
    """tokens through the replay kernel only (in pieces) vs one fwd64 call over the whole sequence"""
    case = Hh.make_case(T, H, regime, seed=3)
    ref = fwd64_out(case)
    st = R.alloc_state(1, H)
    out = torch.zeros_like(ref)
    bounds = pieces or [T]
    a = 0
    for b in bounds + ([T] if bounds[-1] != T else []):
        sl = slice(a, b)
        x = {n: case[n][:, sl].contiguous() for n in ("q", "k", "v", "g", "beta")}
        o = torch.zeros(1, b - a, H, 128, dtype=torch.bfloat16, device=dev)
        R.prefill(x["q"], x["k"], x["v"], x["g"], x["beta"], case["scale"], o, case["A_log"], case["dt_bias"],
                  case["lower_bound"], st, _path="replay")
        out[:, sl] = o
        a = b
    torch.cuda.synchronize()
    nd = Hh.n_diff(out, ref)
    print(f"[replay==fwd64] T={T} H={H} pieces={pieces}: differing bf16 elements = {nd} of {out.numel()}", flush=True)
    return nd


def cfg_equal(T, H, lens=None):
    """every K2 CTA shape must produce the same bits (they only change which warp owns an output element)"""
    case = Hh.make_case(T, H, "authors", seed=5)
    q, k, v, g, beta = (case[n] for n in ("q", "k", "v", "g", "beta"))
    cu = None
    if lens:
        cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int64, device=dev)
    outs = []
    for cfg in range(4):
        out = torch.zeros_like(q)
        st = torch.zeros(len(lens) if lens else 1, H, 128, 128, dtype=torch.bfloat16, device=dev)
        R.fwd64(q, k, v, g, beta, case["scale"], out, case["A_log"], case["dt_bias"], case["lower_bound"],
                final_state=st, cu_seqlens=cu, _k2_cfg=cfg)
        torch.cuda.synchronize()
        outs.append((out, st))
    nd = [Hh.n_diff(outs[0][0], o) + Hh.n_diff(outs[0][1], s2) for o, s2 in outs[1:]]
    print(f"[k2 cfgs equal] T={T} H={H} lens={lens}: differing elements cfg1/2/3 vs cfg0 = {nd}", flush=True)


def host_plan_equal(H=3):
    """two prefill calls per sequence (prompt, then a continuation) through the host plan and the device plan == fwd64"""
    lens1 = [64, 100, 1, 0, 130, 63]
    lens2 = [70, 28, 63, 64, 1, 129]
    N = len(lens1)
    T = [a + b for a, b in zip(lens1, lens2)]
    case = Hh.make_case(sum(T), H, "authors", seed=9)
    whole = {n: case[n][0] for n in ("q", "k", "v", "g", "beta")}
    # oracle: one fwd64 call per sequence (varlen)
    cu = torch.tensor([0] + list(torch.tensor(T).cumsum(0).tolist()), dtype=torch.int64, device=dev)
    ref = torch.zeros(1, sum(T), H, 128, dtype=torch.bfloat16, device=dev)
    R.fwd64(*(whole[n][None] for n in ("q", "k", "v", "g", "beta")), case["scale"], ref, case["A_log"], case["dt_bias"],
            case["lower_bound"], cu_seqlens=cu)
    starts = [0] + list(torch.tensor(T).cumsum(0).tolist())[:-1]
    res = {}
    for plan in ("device", "host"):
        st = R.alloc_state(N, H)
        out = torch.zeros_like(ref)
        ctx = [0] * N
        for part, lens in ((0, lens1), (1, lens2)):
            rows = [list(range(starts[n] + (0 if part == 0 else lens1[n]), starts[n] + (0 if part == 0 else lens1[n]) + lens[n]))
                    for n in range(N)]
            idx = torch.tensor([r for rr in rows for r in rr], dtype=torch.int64, device=dev)
            x = {n: whole[n][idx][None].contiguous() for n in whole}
            cu_p = [0] + list(torch.tensor(lens).cumsum(0).tolist())
            cu_d = torch.tensor(cu_p, dtype=torch.int64, device=dev)
            o = torch.zeros(1, idx.numel(), H, 128, dtype=torch.bfloat16, device=dev)
            kw = dict(context_lens_cpu=list(ctx), cu_seqlens_cpu=cu_p) if plan == "host" else {}
            R.prefill(x["q"], x["k"], x["v"], x["g"], x["beta"], case["scale"], o, case["A_log"], case["dt_bias"],
                      case["lower_bound"], st, cu_seqlens=cu_d, _path="fwd", **kw)
            out[0, idx] = o[0]
            ctx = [c + l for c, l in zip(ctx, lens)]
        torch.cuda.synchronize()
        res[plan] = Hh.n_diff(out, ref)
    print(f"[host/device plan == fwd64] differing elements: {res}", flush=True)


def main():
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    print("smem bytes (K1, K2, replay):", flash_kda.replay64._C.smem_bytes(), flush=True)
    acc(64, 1, "authors")
    acc(100, 2, "authors")
    if quick:
        replay_vs_fwd(100, 2, pieces=[37, 64])
        host_plan_equal()
        return
    for reg in ("authors", "unit", "heavy_tail", "long_memory", "no_decay", "eps_dominated", "saturated", "one_hot"):
        acc(1024, 4, reg)
    replay_vs_fwd(64, 1)
    replay_vs_fwd(100, 2)
    replay_vs_fwd(300, 2, pieces=[1, 2, 3, 63, 64, 65, 130, 200])
    replay_vs_fwd(300, 3, pieces=list(range(1, 300, 7)))
    cfg_equal(1000, 3)
    cfg_equal(700, 2, lens=[100, 64, 300, 236])
    host_plan_equal()


if __name__ == "__main__":
    main()
