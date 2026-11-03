"""Time every tile configuration of bi_linear for a range of M (K = N = 4096 by default); all configurations give the same bits
(tests/cfbi/test_linear.py::test_all_tile_configurations_agree), so the table decides the shape -> config choice.
usage: python benchmarks/micro/linear_configs.py [K] [N]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

import torch

from bench_conv_linear import graph_us_per_call, randbf
from flash_kda.cfbi import bi_linear
from flash_kda.cfbi.linear import pick_config

NCFG = 5


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    wbytes = N * K * 2
    copies = max(2, int(512e6 // wbytes) + 1)
    Ws = [randbf(N, K, scale=0.05) for _ in range(copies)]
    print(f"K={K} N={N}: us per call with the weights streaming from HBM (rotating {copies} matrices); auto = current choice")
    print(f"{'M':>6} {'auto':>4} | " + " ".join(f"{'cfg' + str(c):>7}" for c in range(NCFG)) + " | best")
    for M in (1, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 2048, 8192):
        x = randbf(M, K)
        out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        row = []
        ref = bi_linear(x, Ws[0])
        for c in range(NCFG):
            try:
                same = bool((bi_linear(x, Ws[0], _cfg=c).view(torch.int16) == ref.view(torch.int16)).all())
                t = graph_us_per_call([lambda W=W: bi_linear(x, W, out=out, _cfg=c) for W in Ws], replays=10)
                row.append(t if same else float("nan"))
            except Exception:
                row.append(float("inf"))
        best = min(range(NCFG), key=lambda c: row[c] if row[c] == row[c] else 1e9)
        print(f"{M:>6} {pick_config(M, N):>4} | " + " ".join(f"{t:>7.1f}" for t in row) + f" | cfg{best}", flush=True)
        del x, out
    print("(nan = bits differ from the auto result: must not happen)")


if __name__ == "__main__":
    main()
