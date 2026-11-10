"""Runtime-level gates of the conv / linear kernels: no dependence on repetition, on the process, or on timing.

* repeated launches of the same stateful varlen call (many tiles per sequence, so blocks of one sequence run concurrently
  on different SMs) give the same bits every time: a race in the in-place state update would show up here;
* two independent processes produce the same SHA-256 digest of a fixed workload.
"""
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from flash_kda.cfbi import alloc_conv_state, bi_linear, short_conv

from convlin_helpers import cu_tensor, cx, make_weights, make_x, n_diff  # noqa: F401


def test_repeated_stateful_varlen_calls_are_identical(cx):
    D, W = 520, 4
    lens = [64 * 40 + 3, 1, 64 * 17, 2, 64 * 9 + 63, 0, 300]
    cu = [0]
    for L in lens:
        cu.append(cu[-1] + L)
    N = len(lens)
    x = make_x(cu[-1], D, seed=41)
    w, b = make_weights(D, W, seed=42, with_bias=torch.float32)
    head = make_x(N * 5, D, seed=43)
    st0 = alloc_conv_state(N, D, W)
    short_conv(head, w, b, st0, "silu", cu_tensor([5 * i for i in range(N + 1)]))
    cu_t = cu_tensor(cu)

    def once():
        st = st0.clone()
        o32 = torch.zeros(x.shape, dtype=torch.float32, device="cuda")
        o = short_conv(x, w, b, st, "silu", cu_t, out_fp32=o32)
        return o, o32, st

    ref = once()
    for i in range(150):
        got = once()
        assert n_diff(ref[0], got[0]) == 0 and n_diff(ref[1], got[1]) == 0 and n_diff(ref[2], got[2]) == 0, f"repeat {i}"


def test_repeated_linear_calls_are_identical(cx):
    X = (torch.randn(777, 2048, device="cuda") * 0.5).to(torch.bfloat16)
    Wt = (torch.randn(1024, 2048, device="cuda") * 0.05).to(torch.bfloat16)
    ref = bi_linear(X, Wt)
    for i in range(50):
        assert n_diff(ref, bi_linear(X, Wt)) == 0, f"repeat {i}"


def test_cross_process_digest(cx):
    script = Path(__file__).resolve().parent / "digest_convlin.py"
    digests = []
    for _ in range(2):
        r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=True)
        digests.append([ln for ln in r.stdout.splitlines() if ln.startswith("DIGEST")][-1])
    assert digests[0] == digests[1], digests


def test_too_small_max_seqlen_traps_instead_of_dropping_tokens(cx):
    """max_seqlen only sizes the grid; a sequence longer than it must fail loudly (the trap poisons the CUDA context, so
    the check runs in a subprocess), while a true bound works."""
    root = Path(__file__).resolve().parents[2]
    code = (
        "import sys, torch\n"
        "from flash_kda.cfbi import short_conv\n"
        "x = torch.randn(1, 300, 64, device='cuda').to(torch.bfloat16)\n"
        "w = torch.randn(64, 4, device='cuda')\n"
        "cu = torch.tensor([0, 300], dtype=torch.int64, device='cuda')\n"
        "y = short_conv(x, w, None, None, 'silu', cu, max_seqlen=int(sys.argv[1]))\n"
        "torch.cuda.synchronize()\n"
        "print('OK')\n"
    )
    ok = subprocess.run([sys.executable, "-c", code, "300"], capture_output=True, text=True, cwd=root)
    assert ok.returncode == 0 and "OK" in ok.stdout, ok.stderr[-500:]
    bad = subprocess.run([sys.executable, "-c", code, "64"], capture_output=True, text=True, cwd=root)
    assert bad.returncode != 0 and "OK" not in bad.stdout, "a too-small max_seqlen did not fail"


def test_launches_on_the_current_stream(cx):
    """The kernels run on torch's current stream (a side stream here), not on the default one."""
    D, W = 96, 4
    x = make_x(100, D, seed=1)
    w, b = make_weights(D, W, seed=2, with_bias=torch.float32)
    X = (torch.randn(70, 256, device="cuda") * 0.5).to(torch.bfloat16)
    Wt = (torch.randn(40, 256, device="cuda") * 0.1).to(torch.bfloat16)
    st_ref = alloc_conv_state(1, D, W)
    ref_c, ref_l = short_conv(x, w, b, st_ref, "silu"), bi_linear(X, Wt)
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        st = alloc_conv_state(1, D, W)
        out_c, out_l = short_conv(x, w, b, st, "silu"), bi_linear(X, Wt)
    side.synchronize()
    assert n_diff(out_c, ref_c) == 0 and n_diff(out_l, ref_l) == 0 and n_diff(st, st_ref) == 0
