"""Speed of the CF-BI implementations that can run on THIS device (event-timed wall clock, median of 5 after 2 warm-ups).

    python benchmarks/bench_cfbi_arch.py

``impl`` is the hidden ``_impl`` switch: 0 = default for the device, 3 = reference kernels, 7 = reference row engine for
every row (what sm_80/86/89 run: K1 does not fit in their shared memory). Different implementations give identical
bits (tests/cfbi/test_arch.py); this only measures what the fallbacks cost. Prefill: B = 1, T = 4096, H = 6 (the
CF-BI report's per-GPU head count after TP8); decode: N = 16 sequences, H = 12, one token.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "cfbi"))

import torch

from flash_kda.cfbi import alloc_state, decode, load_ext, prefill
from helpers import make_case


def timed(fn, reps=5, warm=2):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


def main():
    ext = load_ext()
    prop = torch.cuda.get_device_properties(0)
    print(f"{prop.name} sm_{prop.major}{prop.minor}")
    seen = set()
    for impl in (0, 3, 7):
        eff = ext.resolve_impl(impl, -1)
        if eff in seen:
            continue
        seen.add(eff)
        T, H = 4096, 6
        c = make_case(T, H, "authors", seed=1)
        out = torch.zeros_like(c["q"])

        def pre():
            st = alloc_state(1, H)
            prefill(c["q"], c["k"], c["v"], c["g"], c["beta"], c["scale"], out, c["A_log"], c["dt_bias"], c["lower_bound"], st,
                    _impl=impl)

        N, Hd = 16, 12
        d = make_case(N, Hd, "authors", seed=2)
        q1, k1, v1, g1 = (d[k].reshape(N, 1, Hd, 128).contiguous() for k in ("q", "k", "v", "g"))
        b1 = d["beta"].reshape(N, 1, Hd).contiguous()
        o1 = torch.zeros_like(q1)
        stp = alloc_state(N, Hd)

        def dec():
            decode(q1, k1, v1, g1, b1, d["scale"], o1, d["A_log"], d["dt_bias"], d["lower_bound"], stp, _impl=impl)

        print(f"  impl {impl} -> effective {eff}: prefill B=1 T={T} H={H}: {timed(pre):8.2f} ms;  decode N={N} H={Hd}: "
              f"{timed(dec) * 1000:8.1f} us")


if __name__ == "__main__":
    main()
