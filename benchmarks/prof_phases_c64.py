"""Per-phase clock64 timestamps of one K1 CTA, the K2 wait/compute split and the replay kernel phases (profiling only)."""
import os, sys
import torch
import flash_kda
from flash_kda import replay64 as R
C = R._C
dev = "cuda"; D = 128


def tokens(T, H):
    r = lambda *s: torch.randn(*s, device=dev).to(torch.bfloat16)
    return r(T, H, D), r(T, H, D), r(T, H, D), r(T, H, D), r(T, H)


def fwd_prof(name, lens, H, cfg):
    T = sum(lens); N = len(lens)
    q, k, v, g, beta = tokens(T, H)
    out = torch.empty(T, H, D, dtype=torch.bfloat16, device=dev)
    A_log = torch.rand(H, device=dev); dt = torch.rand(H, D, device=dev)
    starts = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist())[:-1], dtype=torch.int64, device=dev)
    lens_t = torch.tensor(lens, dtype=torch.int32, device=dev)
    slots = torch.arange(N, dtype=torch.int32, device=dev)
    tiles = (T + 63) // 64 + N
    ws = torch.empty(C.fwd64_ws_size(tiles, H, N), dtype=torch.uint8, device=dev)
    st = torch.zeros(N, H, D, D, dtype=torch.bfloat16, device=dev)
    prof = torch.zeros(32, dtype=torch.int64, device=dev)
    for _ in range(3):
        C.fwd64(q, k, v, g, beta, D ** -0.5, out, A_log, dt, -5.0, starts, lens_t, slots, tiles, ws, state_out=st, k2_cfg=cfg, prof=prof)
    torch.cuda.synchronize()
    t = prof.tolist()
    k1 = [t[0] - t[8], t[1] - t[0], t[2] - t[1], t[3] - t[2], t[4] - t[3], t[5] - t[4], t[6] - t[5]]
    print(f"[K1 {name} H{H}] cycles: load {k1[0]}  prep {k1[1]}  pairs {k1[2]}  inv_diag {k1[3]}  inv_levels {k1[4]}  operands {k1[5]}  store {k1[6]}  total {t[6] - t[8]}", flush=True)
    print(f"[K2 {name} H{H} cfg{cfg}] per step: wait {t[16] / max(t[18], 1):.0f} cycles  compute {t[17] / max(t[18], 1):.0f} cycles  (steps {t[18]})", flush=True)


def replay_prof(N, H, pos):
    A_log = torch.rand(H, device=dev); dt = torch.rand(H, D, device=dev)
    s = R.alloc_state(N, H)
    q, k, v, g, b = tokens(N, H)
    out = torch.empty(N, H, D, dtype=torch.bfloat16, device=dev)
    prof = torch.zeros(32, dtype=torch.int64, device=dev)
    for _ in range(3):
        s.pos.fill_(pos)
        C.replay64(q, k, v, g, b, D ** -0.5, out, A_log, dt, -5.0, s.S, s.rows_k, s.rows_v, s.rows_g, s.rows_beta, s.pos, N, 1, prof=prof)
    torch.cuda.synchronize()
    t = prof.tolist()
    print(f"[replay N={N} H{H} pos={pos}] cycles: stage {t[9] - t[8]}  prep {t[1] - t[0]}  pairs {t[2] - t[1]}  inv_diag {t[3] - t[2]}  inv_levels {t[4] - t[3]}  "
          f"operands {t[5] - t[4]}  S-wait {t[10] - t[5]}  K2 {t[11] - t[10]}  tail {t[12] - t[11]}  total {t[12] - t[8]}", flush=True)


fwd_prof("8x8192", [8192] * 8, 12, 1)
fwd_prof("64x512", [512] * 64, 12, 1)
fwd_prof("1x32768", [32768], 12, 3)
fwd_prof("1x32768", [32768], 12, 2)
fwd_prof("1x32768", [32768], 12, 1)
for N, pos in ((1, 0), (1, 20), (1, 40), (1, 63), (128, 40), (128, 63)):
    replay_prof(N, 12, pos)
