"""Does cutting a varlen flash_kda.fwd call into extra (empty) pseudo-sequences slow it down? Same tokens and bodies
every time; only the cu_seqlens layout changes (plain N | 2N+1 with empty gaps | 3N with empty heads and tails).

    PYTHONPATH=. python benchmarks/probe_fwd_pseudo_sequences.py
"""
import torch, flash_kda
torch.manual_seed(0)
H, D = 12, 128
def gt(fn, iters=10):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay(); torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(iters): g.replay()
        b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3 / iters)
    return sorted(ts)[2]
for lens in ([8192] * 8, [512] * 64, [2048] * 16):
    T = sum(lens); N = len(lens)
    r = lambda *s: torch.randn(*s, device="cuda").to(torch.bfloat16)
    q, k, v, g = (r(1, T, H, D) for _ in range(4)); beta = r(1, T, H)
    A = torch.rand(H, device="cuda"); dt = torch.rand(H, D, device="cuda"); out = torch.empty_like(q)
    bounds = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist())
    layouts = {
        "plain N": bounds,
        "2N+1 (empty gaps)": torch.cat([bounds[:1], torch.stack([bounds[:-1], bounds[1:]], 1).reshape(-1), bounds[-1:]]),
        "3N (empty head/tail)": torch.cat([bounds[:1], torch.stack([bounds[:-1], bounds[1:], bounds[1:]], 1).reshape(-1)]),
    }
    for name, cu in layouts.items():
        cu = cu.to(torch.int64).cuda(); Np = cu.numel() - 1
        st = torch.zeros(Np, H, D, D, dtype=torch.bfloat16, device="cuda")
        t = gt(lambda: flash_kda.fwd(q, k, v, g, beta, 0.088, out, A, dt, -5.0, initial_state=st, final_state=st, cu_seqlens=cu))
        print(f"{str(lens[:2])+'x'+str(N):>16} {name:>22}: N'={Np:4d}  {t:9.1f} us", flush=True)
