"""CPU-only: the replay kernel must be the current copy of the prefill kernels' arithmetic (tools/gen_replay_kernel.py)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_replay_kernel_is_generated_from_current_prefill():
    r = subprocess.run([sys.executable, str(ROOT / "tools/gen_replay_kernel.py"), str(ROOT), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_hand_copied_expressions_still_match_the_prefill():
    """Two values are computed outside the generated blocks; they must stay identical to the prefill's own lines."""
    k1 = (ROOT / "csrc/smxx/fwd_kernel1.cuh").read_text()
    cpp = (ROOT / "csrc/flash_kda.cpp").read_text()
    tmpl = (ROOT / "tools/replay_kernel.cuh.in").read_text()
    assert "float a_log_exp = expf(A_log_ptr[head_idx]);" in k1
    assert "float a_log_exp = expf(p.A_log[head_idx]);" in tmpl
    assert cpp.count("float(lower_bound * 1.4426950408889634)") == 2, "fwd() and replay() must compute gate_scale alike"
