"""Context parallelism: K1 on every shard at once, K2 chained through E, bit-identical to one prefill call."""
import os
import tempfile

import pytest
import torch
import torch.multiprocessing as mp

from flash_kda.cfbi import alloc_state, cp_prefill, prefill, prefill_k1, prefill_k2, state_equal, state_mismatch

from helpers import make_case, n_diff


def _slice(case, lo, hi):
    out = dict(case)
    for key in ("q", "k", "v", "g", "beta"):
        out[key] = case[key][:, lo:hi].contiguous()
    return out


def _whole(case, H):
    st = alloc_state(1, H)
    out = torch.zeros_like(case["q"])
    out32 = torch.zeros(case["q"].shape, dtype=torch.float32, device="cuda")
    prefill(case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], out, case["A_log"], case["dt_bias"],
            case["lower_bound"], st, out_fp32=out32)
    return out, out32, st


@pytest.mark.parametrize("regime", ["authors", "heavy_tail", "long_memory"])
@pytest.mark.parametrize("cuts", [[192, 448], [64], [640, 704, 768]])
def test_split_phases_equal_whole_prefill(ext, regime, cuts):
    H, T = 3, 64 * 12 + 21
    case = make_case(T, H, regime, seed=7)
    ref_out, ref_out32, ref_state = _whole(case, H)
    bounds = [0] + cuts + [T]
    shards = [_slice(case, bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    states = [alloc_state(1, H) for _ in shards]
    # K1 of every shard first (the parallel part), then K2 chained through E
    h1 = [prefill_k1(s["q"], s["k"], s["v"], s["g"], s["beta"], case["scale"], case["A_log"], case["dt_bias"],
                     case["lower_bound"], st) for s, st in zip(shards, states)]
    outs, out32s = [], []
    for i, (s, st) in enumerate(zip(shards, states)):
        if i > 0:
            st.E.copy_(states[i - 1].E)
        o = torch.zeros_like(s["q"])
        o32 = torch.zeros(s["q"].shape, dtype=torch.float32, device="cuda")
        prefill_k2(h1[i], o, st, out_fp32=o32)
        outs.append(o)
        out32s.append(o32)
    assert n_diff(torch.cat(outs, dim=1), ref_out) == 0, "bf16 outputs differ"
    assert n_diff(torch.cat(out32s, dim=1), ref_out32) == 0, "fp32 outputs differ"
    assert state_equal(states[-1], ref_state), state_mismatch(states[-1], ref_state)


def test_split_phases_varlen_batch(ext):
    """Several sequences per shard (varlen call), each cut at the same grid lines."""
    H = 2
    lens = [64 * 7 + 5, 64 * 5, 64 * 9 + 63]
    cut = 64 * 3
    tot = sum(lens)
    case = make_case(tot, H, "authors", seed=8)
    cu = [0]
    for L in lens:
        cu.append(cu[-1] + L)
    N = len(lens)

    def part(lo_of, hi_of):
        pieces = [_slice(case, cu[n] + lo_of(n), cu[n] + hi_of(n)) for n in range(N)]
        sub = {k: torch.cat([p[k] for p in pieces], dim=1).contiguous() for k in ("q", "k", "v", "g", "beta")}
        c = [0]
        for n in range(N):
            c.append(c[-1] + hi_of(n) - lo_of(n))
        return sub, torch.tensor(c, dtype=torch.int64, device="cuda")

    ref_state = alloc_state(N, H)
    ref_sub, ref_cu = part(lambda n: 0, lambda n: lens[n])
    ref_out = torch.zeros_like(ref_sub["q"])
    prefill(ref_sub["q"], ref_sub["k"], ref_sub["v"], ref_sub["g"], ref_sub["beta"], case["scale"], ref_out, case["A_log"],
            case["dt_bias"], case["lower_bound"], ref_state, ref_cu)

    s0, c0 = part(lambda n: 0, lambda n: cut)
    s1, c1 = part(lambda n: cut, lambda n: lens[n])
    st0, st1 = alloc_state(N, H), alloc_state(N, H)
    h0 = prefill_k1(s0["q"], s0["k"], s0["v"], s0["g"], s0["beta"], case["scale"], case["A_log"], case["dt_bias"], case["lower_bound"], st0, c0)
    h1 = prefill_k1(s1["q"], s1["k"], s1["v"], s1["g"], s1["beta"], case["scale"], case["A_log"], case["dt_bias"], case["lower_bound"], st1, c1)
    o0, o1 = torch.zeros_like(s0["q"]), torch.zeros_like(s1["q"])
    prefill_k2(h0, o0, st0)
    st1.E.copy_(st0.E)
    prefill_k2(h1, o1, st1)
    # reassemble in sequence order
    pieces = []
    for n in range(N):
        pieces.append(o0[:, n * cut:(n + 1) * cut])
        lo = sum(lens[m] - cut for m in range(n))
        pieces.append(o1[:, lo:lo + lens[n] - cut])
    assert n_diff(torch.cat(pieces, dim=1), ref_out) == 0
    assert state_equal(st1, ref_state), state_mismatch(st1, ref_state)


def _worker(rank, world, init_file, case_file, result_dir):
    import torch.distributed as dist
    from flash_kda.cfbi import alloc_state as alloc, cp_prefill as cp

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    d = torch.load(case_file)
    H = d["H"]
    lo, hi = d["bounds"][rank], d["bounds"][rank + 1]
    shard = {k: d[k][:, lo:hi].contiguous().cuda() for k in ("q", "k", "v", "g", "beta")}
    st = alloc(1, H)
    out = torch.zeros_like(shard["q"])
    cp(shard["q"], shard["k"], shard["v"], shard["g"], shard["beta"], d["scale"], out, d["A_log"].cuda(), d["dt_bias"].cuda(),
       d["lower_bound"], st, transport="cpu")
    torch.cuda.synchronize()
    torch.save({"out": out.cpu(), "E": st.E.cpu(), "rows": st.rows.cpu(), "pos": st.pos.cpu()}, os.path.join(result_dir, f"r{rank}.pt"))
    dist.barrier()
    dist.destroy_process_group()


def test_cp_prefill_two_processes_gloo(ext):
    """Two OS processes (rank 0 / 1, gloo, E through host memory) reproduce the single-call result bit for bit."""
    H, T = 2, 64 * 6 + 33
    case = make_case(T, H, "authors", seed=9)
    ref_out, _, ref_state = _whole(case, H)
    bounds = [0, 64 * 3, T]
    with tempfile.TemporaryDirectory() as tmp:
        case_file = os.path.join(tmp, "case.pt")
        torch.save({**{k: case[k].cpu() for k in ("q", "k", "v", "g", "beta", "A_log", "dt_bias")}, "scale": case["scale"],
                    "lower_bound": case["lower_bound"], "H": H, "bounds": bounds}, case_file)
        init_file = os.path.join(tmp, "init")
        mp.spawn(_worker, args=(2, init_file, case_file, tmp), nprocs=2, join=True)
        r0, r1 = torch.load(os.path.join(tmp, "r0.pt")), torch.load(os.path.join(tmp, "r1.pt"))
    out = torch.cat([r0["out"], r1["out"]], dim=1).cuda()
    assert n_diff(out, ref_out) == 0, "context-parallel outputs differ from the single-rank prefill"
    assert torch.equal(r1["E"].view(torch.int32), ref_state.E.cpu().view(torch.int32)), "final E differs"
    assert torch.equal(r1["pos"], ref_state.pos.cpu())
    valid = int(ref_state.pos[0])
    assert torch.equal(r1["rows"][:, :, :, :valid].view(torch.int32), ref_state.rows.cpu()[:, :, :, :valid].view(torch.int32))
