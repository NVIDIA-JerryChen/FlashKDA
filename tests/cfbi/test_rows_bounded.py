"""Controls of the bounded group runner of the GPU stress tests (``rows_bounded.py``), without a GPU: ``pytest --noconftest tests/cfbi/test_rows_bounded.py``.

The runner puts a group of cases in a child process of its own under a hard limit, so that a stress test that never completes fails instead of waiting for the end of the allocation. What is
shown here, with fake drivers (``fake_groups.py`` written into a temporary directory):

* a good group returns its cases, with its size, repetitions, duration and limit recorded (printed, and appended to the file that ``CFBI_GROUP_LOG`` names);
* a failing *case* is a value of the group, not a failure of it; stray output of the child does not disturb the result;
* a group that hangs fails within its limit, says so, and leaves nothing running: the child is gone, and so is a grandchild that it started (the whole process group is killed);
* a group that crashes fails with its own error text and exit status; one that exits without its result fails and says so; one that returns no cases fails;
* source rules over the GPU stress modules (the row-engine delay, decode families and graph tests and the arrival-protocol variants of the single launch): each states an explicit ``GROUP_TIMEOUT_S`` and passes it to every ``run_group`` call, and no test function of them launches kernels itself
  (every stress workload runs inside a group), with mutants that break each rule.
"""
import ast
import json
import os
import textwrap
import time
from pathlib import Path

import pytest

import rows_bounded as rb

TESTS = Path(__file__).resolve().parent
STRESS_MODULES = ("test_rows_delay.py", "test_rows_decode_families.py", "test_rows_graph.py", "test_decode_single_launch_variants.py")
LAUNCHING = {"run_family", "run_arm", "run_step", "time_step", "warm_pool", "make_workload", "Replay", "synchronize"}      # helpers that launch kernels or wait on the device

FAKE = textwrap.dedent('''
    import os
    import subprocess
    import sys
    import time


    def good(ext, n=3):
        return {"cases": {"case%d" % i: None for i in range(n)}, "reps": 5}


    def failing_case(ext):
        return {"cases": {"a": None, "b": "AssertionError: b differs"}, "reps": 1}


    def noisy(ext):
        print("stray output before the result")
        return {"cases": {"x": None}}


    def hang(ext, marker=""):
        open(marker, "w").write(str(os.getpid()))
        time.sleep(3600)


    def hang_with_grandchild(ext, marker=""):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])
        open(marker, "w").write("%d %d" % (os.getpid(), child.pid))
        time.sleep(3600)


    def crash(ext):
        raise RuntimeError("boom in the group")


    def silent_exit(ext):
        os._exit(3)


    def no_cases(ext):
        return {"cases": {}}
''')


@pytest.fixture()
def fake(tmp_path):
    (tmp_path / "fake_groups.py").write_text(FAKE)

    def run(function, args=None, timeout_s=30):
        return rb.run_group("fake_groups", function, args, timeout_s=timeout_s, tests_dir=str(tmp_path), needs_ext=False)

    return run


def alive(pid):
    """Is there a process (a zombie that nobody has reaped yet is none)?"""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        with open("/proc/%d/stat" % pid) as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def gone(pid, wait_s=5.0):
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.05)
    return not alive(pid)


# ---- a group that works -----------------------------------------------------------------------------------------------------------------------------------
def test_a_good_group_returns_its_cases_with_its_size_duration_and_limit(fake, capsys):
    res = fake("good", {"n": 4}, timeout_s=30)
    assert sorted(res["cases"]) == ["case0", "case1", "case2", "case3"] and all(v is None for v in res["cases"].values())
    assert res["case_count"] == 4 and res["reps"] == 5 and res["limit_s"] == 30 and 0 <= res["seconds"] < 30
    assert res["name"] == "fake_groups.good(n=4)"
    line = capsys.readouterr().out
    assert "[group] fake_groups.good(n=4): 4 cases, 5 repetition(s) each" in line and "of 30 s" in line


def test_the_group_is_recorded_in_the_log_that_the_environment_names(fake, tmp_path, monkeypatch):
    log = tmp_path / "groups.jsonl"
    monkeypatch.setenv(rb.LOG_ENV, str(log))
    fake("failing_case")
    fake("good", {"n": 2})
    rows = [json.loads(ln) for ln in log.read_text().splitlines()]
    assert [r["name"] for r in rows] == ["fake_groups.failing_case()", "fake_groups.good(n=2)"]
    assert rows[0]["case_count"] == 2 and rows[0]["failed_cases"] == ["b"] and rows[0]["limit_s"] == 30 and rows[0]["reps"] == 1
    assert rows[1]["case_count"] == 2 and rows[1]["failed_cases"] == [] and rows[1]["reps"] == 5 and isinstance(rows[1]["seconds"], float)


def test_a_failing_case_is_a_value_of_the_group_not_a_failure_of_it(fake):
    res = fake("failing_case")
    assert res["cases"]["a"] is None and res["cases"]["b"] == "AssertionError: b differs"


def test_stray_output_of_the_child_does_not_disturb_the_result(fake):
    assert fake("noisy")["cases"] == {"x": None}


# ---- a group that hangs, crashes or says nothing ----------------------------------------------------------------------------------------------------------
def test_a_hanging_group_fails_within_its_limit_and_leaves_no_process(fake, tmp_path):
    marker = tmp_path / "pid"
    t0 = time.time()
    with pytest.raises(pytest.fail.Exception) as e:
        fake("hang", {"marker": str(marker)}, timeout_s=2)
    assert time.time() - t0 < 12, "the limit of 2 s was not enforced"
    assert "exceeded its limit of 2 s" in str(e.value) and "process group was killed" in str(e.value) and "fake_groups.hang" in str(e.value)
    assert gone(int(marker.read_text())), "the hanging child is still running"


def test_a_hanging_grandchild_is_killed_with_the_group(fake, tmp_path):
    marker = tmp_path / "pids"
    with pytest.raises(pytest.fail.Exception, match="exceeded its limit"):
        fake("hang_with_grandchild", {"marker": str(marker)}, timeout_s=2)
    child, grandchild = (int(x) for x in marker.read_text().split())
    assert gone(child) and gone(grandchild), "a process of the group survived the timeout (child %s, grandchild %s)" % (child, grandchild)


def test_a_crashing_group_fails_with_its_own_error(fake):
    with pytest.raises(pytest.fail.Exception) as e:
        fake("crash")
    assert "boom in the group" in str(e.value) and "exit status 1" in str(e.value) and "without a result" in str(e.value)


def test_a_group_that_exits_without_its_result_fails(fake):
    with pytest.raises(pytest.fail.Exception, match="exit status 3 .* without a result"):
        fake("silent_exit")


def test_a_group_without_cases_fails(fake):
    with pytest.raises(pytest.fail.Exception, match="returned no cases"):
        fake("no_cases")


def test_an_interrupted_wait_kills_the_group_too(tmp_path, monkeypatch):
    """Whatever ends the wait (here a stand-in for KeyboardInterrupt) the child's process group is killed."""
    (tmp_path / "fake_groups.py").write_text(FAKE)
    marker = tmp_path / "pid"
    real = rb.subprocess.Popen.communicate

    def interrupted(self, *args, **kwargs):
        deadline = time.time() + 10
        while not marker.exists() and time.time() < deadline:       # the child is up and has written its pid
            time.sleep(0.05)
        raise KeyboardInterrupt

    monkeypatch.setattr(rb.subprocess.Popen, "communicate", interrupted)
    with pytest.raises(KeyboardInterrupt):
        rb.run_group("fake_groups", "hang", {"marker": str(marker)}, timeout_s=60, tests_dir=str(tmp_path), needs_ext=False)
    monkeypatch.setattr(rb.subprocess.Popen, "communicate", real)
    assert gone(int(marker.read_text()))


# ---- capture ----------------------------------------------------------------------------------------------------------------------------------------------
def test_capture_returns_none_or_the_text_of_what_failed_the_case():
    def fails():
        raise AssertionError("one is not two")         # explicit raises: pytest's assertion rewriting would add its own introspection to an assert statement of this module

    def bare():
        raise AssertionError()

    def errors():
        raise RuntimeError("CUDA error: an illegal memory access")

    assert rb.capture(lambda: None) is None
    assert rb.capture(fails) == "AssertionError: one is not two"
    assert rb.capture(bare) == "AssertionError: assertion failed"
    assert rb.capture(errors) == "RuntimeError: CUDA error: an illegal memory access"
    assert rb.capture(lambda a, b=0: None, 1, b=2) is None


# ---- the stress modules: an explicit limit, and no stress workload outside a group ------------------------------------------------------------------------------
def module_problems(name, text):
    tree = ast.parse(text)
    probs = []
    limits = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "GROUP_TIMEOUT_S" for t in n.targets)]
    if len(limits) != 1 or not isinstance(limits[0].value, ast.Constant) or not isinstance(limits[0].value.value, int) or limits[0].value.value <= 0:
        probs.append("%s: GROUP_TIMEOUT_S must be stated once as a positive integer" % name)
    calls = [c for c in ast.walk(tree) if isinstance(c, ast.Call) and getattr(c.func, "attr", getattr(c.func, "id", None)) == "run_group"]
    if not calls:
        probs.append("%s: no group is run through rows_bounded.run_group" % name)
    for c in calls:
        if not any(isinstance(a, ast.Name) and a.id == "GROUP_TIMEOUT_S" for a in list(c.args) + [k.value for k in c.keywords]):
            probs.append("%s:%d: a group is run without the explicit GROUP_TIMEOUT_S" % (name, c.lineno))
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and fn.name.startswith("test_"):
            used = {getattr(c.func, "attr", getattr(c.func, "id", None)) for c in ast.walk(fn) if isinstance(c, ast.Call)}
            if used & LAUNCHING:
                probs.append("%s:%d: the test %s launches kernels itself (%s) outside a bounded group" % (name, fn.lineno, fn.name, ", ".join(sorted(used & LAUNCHING))))
    return probs


def stress_texts():
    return {n: (TESTS / n).read_text() for n in STRESS_MODULES}


def test_the_stress_modules_state_an_explicit_limit_and_run_every_workload_in_a_group():
    for name, text in stress_texts().items():
        assert module_problems(name, text) == [], name


def test_a_stress_module_without_a_limit_or_with_a_workload_outside_a_group_is_caught():
    texts = stress_texts()
    delay = texts["test_rows_delay.py"]
    assert any("GROUP_TIMEOUT_S must be stated" in p for p in module_problems("m", delay.replace("GROUP_TIMEOUT_S = 600", "GROUP_TIMEOUT_S = None", 1)))
    assert any("GROUP_TIMEOUT_S must be stated" in p for p in module_problems("m", delay.replace("GROUP_TIMEOUT_S = 600", "", 1)))
    assert any("without the explicit GROUP_TIMEOUT_S" in p for p in module_problems("m", delay.replace('"group_misc", {}, GROUP_TIMEOUT_S)', '"group_misc", {}, 99999)', 1)))
    inline = delay + "\n\ndef test_a_stress_workload_in_the_test_itself(ext):\n    rw.run_arm(ext, 0, None, None)\n"
    assert any("launches kernels itself (run_arm)" in p for p in module_problems("m", inline))
    sync = texts["test_rows_graph.py"] + "\n\ndef test_x(ext):\n    torch.cuda.synchronize()\n"
    assert any("synchronize" in p for p in module_problems("m", sync))
    assert any("no group is run" in p for p in module_problems("m", "GROUP_TIMEOUT_S = 5\n"))
