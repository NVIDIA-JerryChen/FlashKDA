"""compute-sanitizer over a whole prefill with the device-scheduled K2 (impl bit 16): memcheck, synccheck, initcheck (and racecheck where the tool supports the kernel).

The runs are UNFILTERED (no ``--kernel-name``): the filter was not verified to select every kernel (for example the mixed K2 kernel), and the new path
launches two kernels of its own (``k2_sched_kernel``, ``k2_dyn_kernel``) next to K1 and the plan kernel. The workload is a corpus-V shape whose device schedule mixes
tile widths (checked through the schedule probe before the run), a forced-width call, and a split K1 / K2 call. Every sanitizer run is bounded by ``TIMEOUT_S``
(3600 s, the bound recorded for these runs); a run that exceeds it fails the test. The default workload is V2 at one eighth of its length (a long sequence
of 1024 tokens and seven of 128: the same 8:1 skew and the same mixed 8 / 32-column schedule on 132 .. 148 SMs, 1920 tokens): initcheck is orders of magnitude
slower than the kernels (about 10 minutes per call here); ``CFBI_BIG=1`` runs the full corpus entry V2.

What initcheck can say here. The K2 kernels copy the Aqk operand of a chunk from the workspace with one bulk copy of 2624 floats; its diagonal 16x16 blocks have
a row stride of 17 floats and K1 never writes the 17th float of a row. Those padding floats are read by the copy and never used, but initcheck reports them
for EVERY K2 kernel, the old ones included (the unmodified kernels report the same lines), whenever the workspace is memory that nothing has written yet.
The driver therefore hands the calls a zero-filled workspace (and a fresh, never written plan buffer, so a read of an unwritten schedule entry or plan field
would still be reported): any report left is a finding of the new path.
"""
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from varlen_corpus import CORPUS_V

TIMEOUT_S = 3600
BIG = os.environ.get("CFBI_BIG", "0") == "1"
SCALED_V2 = [1024] + [128] * 7
TOOLS = ["memcheck", "synccheck", "initcheck"]
REPO = Path(__file__).resolve().parents[2]

DRIVER = textwrap.dedent('''
    import itertools, sys
    sys.path.insert(0, "{tests}")
    sys.path.insert(0, "{repo}")
    import torch
    from flash_kda.cfbi import alloc_state, load_ext
    from helpers import make_case

    ext = load_ext()
    lens, H = {lens}, {heads}
    N = len(lens)
    T = sum(lens)
    ws_n, plan_n = ext.workspace_sizes(T, N, H)
    # plan buffers come first and stay alive: every call gets plan memory that nothing has written (initcheck sees a read before the first write)
    plans = [torch.empty(plan_n, dtype=torch.int32, device="cuda") for _ in range(6)]
    cu = torch.tensor([0] + list(itertools.accumulate(lens)), dtype=torch.int64, device="cuda")
    ext.probe_k2_schedule(plans[0], H, N, 0, cu, None, None, 0, 0)
    torch.cuda.synchronize()
    items = plans[0][(N + 1) * 8 + 16: (N + 1) * 8 + 16 + int(plans[0][(N + 1) * 8])].cpu().tolist()
    widths = {{8 << ((it >> 29) & 3) for it in items}}
    assert len(widths) >= 2, "the workload must mix tile widths, got %s" % sorted(widths)
    case = make_case(T, H, "authors", seed=7)
    out32 = torch.zeros(case["q"].shape, dtype=torch.float32, device="cuda")

    def forward(st, out, plan, ws, phase, bv):
        ext.forward(case["q"], case["k"], case["v"], case["g"], case["beta"], float(case["scale"]), out, case["A_log"], case["dt_bias"], float(case["lower_bound"]),
                    st.E, st.rows, st.pos, cu, 0, out32 if phase != 1 else None, 16, None, ws, plan, phase, bv)

    for i, bv in enumerate((0, 8, 32)):                 # the device policy, then forced widths
        forward(alloc_state(N, H), torch.zeros_like(case["q"]), plans[1 + i], torch.zeros(ws_n, dtype=torch.float32, device="cuda"), 3, bv)
    st = alloc_state(N, H)                              # a split K1 / K2 call: the schedule is built from the plan of the first phase
    ws = torch.zeros(ws_n, dtype=torch.float32, device="cuda")
    forward(st, torch.zeros_like(case["q"]), plans[4], ws, 1, 0)
    forward(st, torch.zeros_like(case["q"]), plans[4], ws, 2, 0)
    torch.cuda.synchronize()
    print("SANITIZER_WORKLOAD_DONE widths", sorted(widths))
''')


def _sanitizer():
    exe = shutil.which("compute-sanitizer")
    if exe is None:
        pytest.skip("compute-sanitizer is not installed")
    return exe


@pytest.fixture(autouse=True)
def _dyn_runs_here(ext):
    if not ext.resolve_impl(16, -1) & 16:
        pytest.skip("the device cannot run the device-scheduled K2 (impl bit 16 is dropped here)")


def _run(tool, tmp_path, extra=()):
    exe = _sanitizer()
    script = tmp_path / "driver.py"
    lens = CORPUS_V["V2"].lens if BIG else SCALED_V2
    script.write_text(DRIVER.format(tests=str(REPO / "tests" / "cfbi"), repo=str(REPO), lens=repr(list(lens)), heads=CORPUS_V["V2"].H))
    cmd = [exe, "--tool", tool, "--error-exitcode", "99", *extra, sys.executable, str(script)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, env=dict(os.environ))
    except subprocess.TimeoutExpired:
        pytest.fail(f"compute-sanitizer {tool} exceeded the recorded bound of {TIMEOUT_S} s")
    return r


@pytest.mark.parametrize("tool", TOOLS)
def test_sanitizer_reports_no_error_over_a_whole_device_scheduled_prefill(tool, tmp_path):
    r = _run(tool, tmp_path)
    text = r.stdout + r.stderr
    assert "SANITIZER_WORKLOAD_DONE" in text, text[-3000:]
    assert r.returncode == 0 and "ERROR SUMMARY: 0 errors" in text, text[-3000:]


def test_racecheck_where_the_tool_supports_the_kernels(tmp_path):
    """racecheck covers shared-memory hazards; if it cannot analyse the bulk-copy / mbarrier kernels it says so and the run is recorded as unsupported."""
    r = _run("racecheck", tmp_path, extra=("--racecheck-report", "hazard"))
    text = r.stdout + r.stderr
    if "SANITIZER_WORKLOAD_DONE" not in text or "not supported" in text.lower() or "unsupported" in text.lower():
        pytest.skip("racecheck does not support these kernels: " + text[-500:])
    assert r.returncode == 0 and "RACECHECK SUMMARY: 0 hazards" in text, text[-3000:]
