"""Prints a SHA-256 digest of a fixed conv + linear workload (outputs, fp32 values before the rounding, states).

Run in separate processes (tests/cfbi/test_convlin_runtime.py::test_cross_process_digest does) and, by hand, once with the JIT-built
extension and once with the setup.py-built one (FLASH_KDA_BUILD_EXTS=cfbi_conv, different nvcc command line): the digests
must be identical (no dependence on process state, allocator addresses, build route or stream timing).
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from flash_kda.cfbi import alloc_conv_state, bi_linear, short_conv
from convlin_helpers import cu_tensor, make_weights, make_x


def update(h, t):
    h.update(t.contiguous().view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32).cpu().numpy().tobytes())


def main():
    h = hashlib.sha256()
    # ---- conv: a varlen batch served in three calls with different cuts (state in / out), then the stateless forward
    D, W = 1000, 4
    lens = [150, 1, 64, 0, 129, 33, 2]
    cu = [0]
    for L in lens:
        cu.append(cu[-1] + L)
    N = len(lens)
    x = make_x(cu[-1], D, seed=99)
    w, b = make_weights(D, W, seed=98, wdtype=torch.bfloat16, with_bias=torch.float32)
    st = alloc_conv_state(N, D, W)
    done = [0] * N
    for cut in (0.37, 0.71, 1.0):
        take = [int(L * cut) - d for L, d in zip(lens, done)]
        sub = torch.cat([x[:, cu[n] + done[n]:cu[n] + done[n] + take[n]] for n in range(N)], dim=1).contiguous()
        cu_call = [0]
        for t in take:
            cu_call.append(cu_call[-1] + t)
        out32 = torch.zeros(sub.shape, dtype=torch.float32, device="cuda")
        out = short_conv(sub, w, b, st, "silu", cu_tensor(cu_call), out_fp32=out32)
        update(h, out)
        update(h, out32)
        done = [d + t for d, t in zip(done, take)]
    update(h, st)
    out32 = torch.zeros(x.shape, dtype=torch.float32, device="cuda")
    update(h, short_conv(x, w, b, None, "silu", cu_tensor(cu), out_fp32=out32))
    update(h, out32)
    # ---- linear: the same weights with M = 1, 17, 300 rows, with and without bias
    K, Nn = 1040, 130
    g = torch.Generator(device="cuda")
    g.manual_seed(97)
    X = (torch.randn(300, K, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    Wl = (torch.randn(Nn, K, device="cuda", generator=g) * 0.05).to(torch.bfloat16)
    bl = torch.randn(Nn, device="cuda", generator=g)
    for M in (1, 17, 300):
        for bias in (None, bl):
            o32 = torch.zeros(M, Nn, dtype=torch.float32, device="cuda")
            update(h, bi_linear(X[:M], Wl, bias, out_fp32=o32))
            update(h, o32)
    print("DIGEST", h.hexdigest())


if __name__ == "__main__":
    main()
