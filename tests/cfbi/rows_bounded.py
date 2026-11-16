"""Bounded child-process groups for the GPU stress tests of the row engine (delay variants, decode families, graph replay).

A GPU stress test that waits on the device cannot be interrupted from inside its own process: a wrong barrier or a future protocol that never completes ends only when the allocation does.
``run_group`` therefore runs a *group* of cases (a plain function of a test module, ``function(ext, **args)``, that returns ``{"cases": {case id: None | failure text}}``) in a child process
of its own, in a session of its own, under an explicit hard time limit: on a timeout the whole process group of the child is killed (the child, a sanitizer or a build it started), the test
FAILS, and nothing of it is left running. A group that crashes, or exits without its result, fails with the child's own output. The tests of a module then assert the case outcome that they
are named after, so every case keeps its id, its assertions and its repetitions; only where it runs changed.

What is recorded, per group: its name, the number of cases, the repetitions that the group reports, the duration and the limit (printed, and appended as one JSON line to the file named by
``CFBI_GROUP_LOG`` when that is set; the node runner passes it into the container). Nothing here needs pytest-timeout, signals or a GPU: ``tests/cfbi/test_rows_hooks.py`` runs it against fake
drivers (a good one, a hanging one with a hanging grandchild, a crashing one).

The child entry point is this file: ``python rows_bounded.py <module> <function> <json args> <needs_ext>``.
"""
import importlib
import json
import os
import signal
import subprocess
import sys
import time

RESULT_LABEL = "ROWS_GROUP_RESULT"
LOG_ENV = "CFBI_GROUP_LOG"
TAIL = 1500


def _tail(text):
    text = (text or "").strip()
    return text[-TAIL:] if len(text) > TAIL else text


def _kill_group(proc):
    """Kill the child's whole process group (it is the leader of a session of its own) and reap the child."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.kill()
    except OSError:
        pass
    proc.wait()


def _fail(message):
    import pytest

    pytest.fail(message, pytrace=False)


def group_name(module, function, args):
    return "%s.%s(%s)" % (module, function, ", ".join("%s=%r" % kv for kv in sorted((args or {}).items())))


def run_group(module, function, args=None, timeout_s=600, tests_dir=None, needs_ext=True, env=None, python=None):
    """Run ``module.function(ext, **args)`` in a child process group under a hard limit of ``timeout_s`` seconds.

    Returns the group's result: ``{"cases": {...}, "name", "case_count", "reps", "seconds", "limit_s"}``. Fails the test (``pytest.fail``) on a timeout, on a crash and on a child that exits
    without a result; a failing *case* is not a failure of the group (it is a value in ``cases`` for the test of that case to assert)."""
    args = dict(args or {})
    name = group_name(module, function, args)
    tests_dir = tests_dir or os.path.dirname(os.path.abspath(__file__))
    cmd = [python or sys.executable, os.path.abspath(__file__), module, function, json.dumps(args, sort_keys=True), "1" if needs_ext else "0"]
    child_env = dict(os.environ, **(env or {}))
    child_env["PYTHONPATH"] = os.pathsep.join(p for p in (tests_dir, child_env.get("PYTHONPATH", "")) if p)
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, env=child_env, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        out, err = proc.communicate()
        _fail("the group %s exceeded its limit of %d s: the child's process group was killed (output so far: %s)" % (name, timeout_s, _tail(out + "\n" + err)))
    except BaseException:
        _kill_group(proc)
        raise
    seconds = time.time() - t0
    lines = [ln for ln in out.splitlines() if ln.startswith(RESULT_LABEL + " ")]
    if proc.returncode != 0 or not lines:
        _fail("the group %s ended with exit status %s after %.1f s and %s: %s" % (name, proc.returncode, seconds, "without a result" if not lines else "with a result", _tail(err or out)))
    result = json.loads(lines[-1][len(RESULT_LABEL) + 1:])
    cases = result.get("cases")
    if not isinstance(cases, dict) or not cases:
        _fail("the group %s returned no cases: %s" % (name, _tail(lines[-1])))
    result.update({"name": name, "case_count": len(cases), "seconds": round(seconds, 2), "limit_s": timeout_s})
    record_group(result)
    return result


def record_group(result):
    """Print the size, duration and limit of a group, and append them to the file that ``CFBI_GROUP_LOG`` names."""
    line = {k: result.get(k) for k in ("name", "case_count", "reps", "seconds", "limit_s")}
    line["failed_cases"] = sorted(c for c, v in result["cases"].items() if v is not None)
    print("\n[group] %(name)s: %(case_count)s cases, %(reps)s repetition(s) each, %(seconds)s s of %(limit_s)s s" % dict(line, reps=line["reps"] if line["reps"] is not None else 1))
    path = os.environ.get(LOG_ENV)
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(line, sort_keys=True) + "\n")


def capture(fn, *args, **kwargs):
    """Run one case of a group: ``None`` when it passed, else the text of what failed it (an assertion or any other error of the case)."""
    try:
        fn(*args, **kwargs)
    except AssertionError as e:
        return "AssertionError: %s" % ((str(e) or "assertion failed")[:3000])
    except Exception as e:   # a CUDA error, a missing attribute: a failure of the case, with its type
        return "%s: %s" % (type(e).__name__, str(e)[:3000])
    return None


def _child_main(argv):
    module, function, args_json, needs_ext = argv
    importlib.invalidate_caches()
    fn = getattr(importlib.import_module(module), function)
    ext = None
    if needs_ext == "1":
        from flash_kda.cfbi import load_ext

        ext = load_ext()
    result = fn(ext, **json.loads(args_json))
    sys.stdout.flush()
    print(RESULT_LABEL + " " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    _child_main(sys.argv[1:])
