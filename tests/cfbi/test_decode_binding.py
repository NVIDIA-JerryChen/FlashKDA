"""The binding of the single-launch decode policy, checked on a machine without a GPU (``pytest --noconftest tests/cfbi/test_decode_binding.py``).

``csrc/cfbi/cfbi_binding.cpp`` is only compiled by the JIT build on a GPU node, so a mistake in it would show up late; the first test compiles it here (syntax only, against the torch
headers) and shows that the check can fail. The other tests read the sources:

* the routed-K2 conditions of the binding ignore ``IMPL_DECODE_PLAN`` for ``impl`` and for the resolved ``eff_impl`` (bit 64 chooses the schedule of a decode call and changes nothing for
  any other call: a prefill call with and without it launches the same kernels; the device test is in ``test_decode_single_launch.py``);
* the binding exposes the pure policy (any SM count and capability, no device) next to the wrapper that reads the device, and the hidden calibration control; neither is part of the
  documented API, and the control is no build option (no test macro);
* the wrapper applies the control to the eligible part of the rule and nothing else.
"""
import re
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))

import codegen_gate as cg  # noqa: E402

CSRC = ROOT / "csrc" / "cfbi"
BINDING, LAUNCH, HEADER = CSRC / "cfbi_binding.cpp", CSRC / "cfbi_launch.cu", CSRC / "cfbi_launch.h"


# ---- the binding compiles ---------------------------------------------------------------------------------------------------------------------------------
def cuda_include(nvcc):
    """The directory with ``cuda_runtime_api.h`` of the toolkit that ``nvcc`` belongs to (``include`` or the conda layout ``targets/x86_64-linux/include``)."""
    root = Path(nvcc).resolve().parent.parent
    for candidate in (root / "include", root / "targets" / "x86_64-linux" / "include"):
        if (candidate / "cuda_runtime_api.h").exists():
            return candidate
    return None


def syntax_check(source, cwd, fatal=False):
    """``(exit status, output)`` of a syntax-only compile of ``source`` as the JIT build compiles the binding (the torch headers, the extension name), with the host compiler of the toolchain."""
    torch_utils = pytest.importorskip("torch.utils.cpp_extension")
    tc = cg.find_toolchain()
    include = cuda_include(tc["nvcc"])
    if include is None:
        pytest.skip("no CUDA include directory next to nvcc")
    cmd = [tc["ccbin"], "-std=c++17", "-fsyntax-only", "-DTORCH_EXTENSION_NAME=cfbi_check", "-DTORCH_API_INCLUDE_EXTENSION_H", "-D_GLIBCXX_USE_CXX11_ABI=1", "-I", str(CSRC.parent)]
    cmd += ["-I" + p for p in torch_utils.include_paths()] + ["-I" + sysconfig.get_paths()["include"], "-I" + str(include)]
    if fatal:
        cmd.append("-Wfatal-errors")
    r = subprocess.run(cmd + [str(source)], cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    return r.returncode, r.stdout


def _toolchain_or_skip():
    try:
        return cg.find_toolchain()
    except cg.Unusable as e:
        pytest.skip("no toolchain on this machine: %s" % e)


def test_the_binding_compiles_without_a_gpu(tmp_path):
    _toolchain_or_skip()
    rc, text = syntax_check(BINDING, tmp_path)
    assert rc == 0, text[-3000:]


def test_the_syntax_check_can_fail(tmp_path):
    """The control: the same command line rejects a binding that carries an error (a ``#error`` first, so the check stops at once)."""
    _toolchain_or_skip()
    broken = tmp_path / "broken.cpp"
    broken.write_text('#error planted\n' + BINDING.read_text())
    rc, text = syntax_check(broken, tmp_path, fatal=True)
    assert rc != 0 and "planted" in text, text[-1500:]


# ---- the sources ------------------------------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def binding():
    return BINDING.read_text()


@pytest.fixture(scope="module")
def launch():
    return LAUNCH.read_text()


def mutate(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def routing_problems(text):
    """The routed-K2 condition (the one that calls ``k2_dyn_default_route``) takes ``impl`` and ``eff_impl`` without bit 64 and no other form of 'asks for nothing'."""
    conds = [l for l in text.splitlines() if "k2_dyn_default_route" in l or re.search(r"\bimpl\b.*==\s*0\b.*\beff_impl\b", l)]
    probs = []
    cond = next((l for l in conds if "k2_bv == 0" in l), None)
    if cond is None:
        return ["the routed-K2 condition is not found"]
    if "(impl & ~cfbi::IMPL_DECODE_PLAN) == 0" not in cond:
        probs.append("the routed-K2 condition does not ignore bit 64 of impl")
    if "(eff_impl & ~cfbi::IMPL_DECODE_PLAN) == 0" not in cond:
        probs.append("the routed-K2 condition does not ignore bit 64 of the resolved eff_impl")
    if re.search(r"\bimpl == 0\b|\beff_impl == 0\b", cond):
        probs.append("the routed-K2 condition still compares an implementation word with zero")
    return probs


def test_the_routed_k2_condition_ignores_bit_64(binding):
    assert routing_problems(binding) == []


@pytest.mark.parametrize("name,old,new,expect", [
    ("impl compared with zero", "(impl & ~cfbi::IMPL_DECODE_PLAN) == 0 && k2_bv == 0", "impl == 0 && k2_bv == 0", "bit 64 of impl"),
    ("eff_impl compared with zero", "(eff_impl & ~cfbi::IMPL_DECODE_PLAN) == 0 && eff_mode == 0", "eff_impl == 0 && eff_mode == 0", "resolved eff_impl"),
    ("another bit masked", "(impl & ~cfbi::IMPL_DECODE_PLAN) == 0", "(impl & ~cfbi::IMPL_K2_OLD) == 0", "bit 64 of impl"),
])
def test_a_routed_k2_condition_that_does_not_mask_bit_64_is_caught(binding, name, old, new, expect):
    assert any(expect in p for p in routing_problems(mutate(binding, old, new))), name


def exposure_problems(binding, launch, header):
    probs = []
    for name in ("decode_single_launch", "decode_single_launch_policy", "decode_single_force"):
        if 'm.def("%s"' % name not in binding:
            probs.append("the binding does not expose %s" % name)
    fn = re.search(r"bool decode_single_launch_policy_py\([^)]*\) \{(.*?)\n\}", binding, re.S)
    if fn is None or "cfbi::decode_single_launch_policy(" not in fn.group(1) or "cudaGetDevice" in fn.group(1) or "cudaDeviceGetAttribute" in fn.group(1):
        probs.append("the exposed policy is not the pure function (it must not read the device)")
    if fn is not None and not all(a in fn.group(1) for a in ("sms", "cc")):
        probs.append("the exposed policy does not take the SM count and the compute capability")
    wrapper = re.search(r"bool decode_single_launch\(int mode, int impl, int64_t N, int64_t H, int device\) \{(.*?)\n\}", launch, re.S)
    if wrapper is None:
        probs.append("the wrapper decode_single_launch is not found")
    else:
        body = wrapper.group(1)
        eligible, force, query, policy = (body.find(s) for s in ("decode_single_launch_eligible(", "g_decode_single_force.load()", "cudaDeviceGetAttribute", "return decode_single_launch_policy("))
        if not 0 <= eligible < force < query < policy:
            probs.append("the wrapper does not apply the eligible part first, then the control, then the device and the policy")
        if "return true" not in body[force:query]:
            probs.append("the control does not return true for an eligible call")
    for text, name in ((launch, "cfbi_launch.cu"), (header, "cfbi_launch.h")):
        for m in re.finditer(r"#\s*(?:ifdef|ifndef|if)\b[^\n]*\bCFBI_\w*(?:FORCE|CALIBRAT)\w*", text):
            probs.append("%s: the calibration control is a build option (%s)" % (name, m.group(0).strip()))
    return probs


def test_the_binding_exposes_the_policy_and_the_hidden_control(binding, launch):
    assert exposure_problems(binding, launch, HEADER.read_text()) == []


@pytest.mark.parametrize("which,name,old,new,expect", [
    ("binding", "the pure policy not exposed", 'm.def("decode_single_launch_policy"', 'm.def("decode_policy_x"', "does not expose decode_single_launch_policy"),
    ("binding", "the control not exposed", 'm.def("decode_single_force"', 'm.def("decode_force_x"', "does not expose decode_single_force"),
    ("binding", "the policy reads the device", "return cfbi::decode_single_launch_policy(checked_int(mode", "int sms_dev = 0; cudaDeviceGetAttribute(&sms_dev, cudaDevAttrMultiProcessorCount, 0);\n    return cfbi::decode_single_launch_policy(checked_int(mode", "must not read the device"),
    ("launch", "the control before the eligible part", "    if (!decode_single_launch_eligible(mode, impl, N, H)) return false;\n    if (g_decode_single_force.load() != 0) return true;",
     "    if (g_decode_single_force.load() != 0) return true;\n    if (!decode_single_launch_eligible(mode, impl, N, H)) return false;", "eligible part first"),
    ("launch", "the control after the device query", "    if (g_decode_single_force.load() != 0) return true;\n    if (device < 0", "    if (device < 0", "eligible part first"),
])
def test_a_binding_without_the_pure_policy_or_the_control_in_its_place_is_caught(binding, launch, which, name, old, new, expect):
    b, l = (mutate(binding, old, new), launch) if which == "binding" else (binding, mutate(launch, old, new))
    assert any(expect in p for p in exposure_problems(b, l, HEADER.read_text())), name


def test_the_control_is_not_part_of_the_documented_api():
    """None of the hidden functions is exported by the package, and the documentation does not offer the control as an interface (the prose of ``_impl`` names the rule, the policy)."""
    import ast

    init = (ROOT / "flash_kda" / "cfbi" / "__init__.py").read_text()
    exported = set()
    for node in ast.parse(init).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__all__" for t in node.targets):
            exported |= {e.value for e in node.value.elts if isinstance(e, ast.Constant)} if sys.version_info >= (3, 8) else {e.s for e in node.value.elts}
    assert exported, "no __all__ found"
    hidden = ("decode_single_force", "decode_single_launch_policy", "decode_single_launch", "rows_occupancy")
    assert not exported & set(hidden)
    api = (ROOT / "flash_kda" / "cfbi" / "api.py").read_text() + init
    assert "decode_single_force" not in api                          # the control is for tests and the benchmark and is not described to a user (the prose may name the rule, the policy)


def test_a_control_compiled_in_by_a_macro_is_caught(binding, launch):
    planted = launch.replace("static std::atomic<int> g_decode_single_force{0};", "#ifdef CFBI_DECODE_FORCE\nstatic std::atomic<int> g_decode_single_force{0};\n#endif")
    assert any("build option" in p for p in exposure_problems(binding, planted, HEADER.read_text()))
