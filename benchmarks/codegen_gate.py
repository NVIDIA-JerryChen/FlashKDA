#!/usr/bin/env python3
"""Compile-only code-generation gate for the CF-BI CUDA sources (plain Python, no torch, no GPU).

The tool compiles every CUDA translation unit of the CF-BI extension of a source tree with one nvcc, one host compiler and the production flags of
``flash_kda/cfbi/_ext.py``, reads the machine code and the resource figures of every kernel (``cuobjdump``, ``ptxas -v``) into a *dump*, and
compares the dumps of two trees (a base and a candidate) built in the same invocation or taken from a cache under an identical key. A dump can also be
made from cuobjdump text that another toolchain produced (the node modes ``regs`` and ``sass`` of the container), so the same gate runs on the production toolchain.

usage:
  codegen_gate.py dump       --tree DIR --out FILE [--archs sm_90a sm_100a sm_103a] [--define NAME[=VALUE] ...] [--cache-dir DIR] [--keep-dir DIR]
  codegen_gate.py manifest   --build-log FILE --tool-versions FILE --label TEXT --out FILE [--artifact SO] [--allow-non-production-flags]
  codegen_gate.py dump-files --manifest FILE --regs FILE --sass FILE --elf FILE --ptxas-log FILE --out FILE
  codegen_gate.py compare    --base FILE --cand FILE [--prev FILE] [--mode identity|change] [--allow FILE] [--json FILE]
  codegen_gate.py gate       --base-tree DIR --cand-tree DIR [--prev-tree DIR] [--mode identity|change] [--cand-define NAME[=VALUE] ...] [--cache-dir DIR]
  codegen_gate.py matrix     --tree DIR [--archs ...]

Identity mode: every kernel of the base exists in the candidate with the same resources and the same normalised SASS hash unless it is on the allow-list;
kernels and instantiations that only the candidate has are listed and checked against the resource gate (gate R). Change mode: the kernels of the owned family
(``cfbi::rows_kernel`` by default) may differ, must satisfy gate R against the base, and every other kernel stays identical to the base.

The normalised SASS drops instruction offsets, encodings (the control words included) and trailer comments; operands and immediates stay, so an
operand-only or an immediate-only change is a difference. Name mangling of template arguments is handled by the kernel key (``cfbi::k2_kernel<Li16E>``)
and by the ``aliases`` of the allow-list file (candidate key -> base key).

exit status: 0 pass, 1 differences or a resource-gate failure, 2 unusable input (unreadable or incompatible dumps, toolchain mismatch, a failed build, empty kernel sets)
"""
import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_binaries as cb  # noqa: E402  (resource-usage parser and the arch / function line patterns)

SCHEMA = "cfbi-codegen-v1"
ALLOW_SCHEMA = "cfbi-codegen-allow-v1"

# Copied from NVCC_FLAGS of flash_kda/cfbi/_ext.py (importing setup.py or the package would run a submodule update / need torch);
# tests/cfbi/test_bench_tools.py compares this list with the source text of _ext.py.
PRODUCTION_FLAGS = [
    "-O3",
    "-fmad=false",
    "-ftz=false",
    "-prec-div=true",
    "-prec-sqrt=true",
    "--expt-relaxed-constexpr",
    "-lineinfo",
    "--ptxas-options=-v",
]
STD = "c++20"                                    # torch's cpp_extension passes the standard; the system g++ 8.5 cannot compile it
FIXED_SRC_HASH = "0x0000000000000000ull"         # _ext.py passes a hash of the sources; a constant keeps two trees comparable
GATE_ARCHS = ("sm_90a", "sm_100a", "sm_103a")
MATRIX_ARCHS = ("sm_80", "sm_86", "sm_89", "sm_90a", "sm_100a", "sm_103a", "sm_120a")   # CFBI_CUDA_ARCHS of setup.py
DEFAULT_UNITS = ("csrc/cfbi/cfbi_launch.cu", "csrc/cfbi/cfbi_k2_dyn.cu", "csrc/cfbi/cfbi_probe.cu")
DEFAULT_CONDA = "/home/scratch.jennym_gpu/miniforge3/envs/tlang"
DEFAULT_OWNED = ("cfbi::rows_kernel",)           # the family a row-engine step may change (change mode) and gate R applies to
GATE_R_REF = "cfbi::rows_kernel"                 # the base kernel whose stack and spill figures bound a gate-R target that has no kernel of its own in the base: the row engine ...
GATE_R_REF_INSTANCE = "cfbi::rows_kernel<Lb0E>"  # ... or, in a base that has it as instantiations, the one behind the plan kernel (rows_kernel<false>; <Lb1E> is the single-launch decode)
REG_CAP = 128                                    # __launch_bounds__(256, 2): 64K registers / (2 * 256 threads)
# Reference figures of the base rows_kernel (same-compiler values of the recorded round); a deviation is toolchain drift (a warning), not a failure.
LITERALS = {
    "sm_100a": {"reg": 128, "spill_stores": 4, "spill_loads": 4, "stack": 112},
    "sm_103a": {"reg": 128, "spill_stores": 12, "spill_loads": 8, "stack": 112},
    "sm_90a": {"reg": 128, "spill_stores": 428, "spill_loads": 172},
}
DEFAULT_ALLOW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "codegen_allow.json")

_RESOURCE_FIELDS = ("reg", "stack", "shared", "local")
_SPILL_FIELDS = ("spill_stores", "spill_loads")
_ATTR_FIELDS = ("max_threads", "maxreg")
_TOOLCHAIN_FIELDS = ("source", "label", "nvcc", "host_compiler", "cuobjdump", "ptxas", "flags", "std", "archs", "src_hash")
_TOOLCHAIN_REQUIRED = ("source", "label", "nvcc", "host_compiler", "cuobjdump", "ptxas", "flags", "std", "archs")   # src_hash: a fixed constant (nvcc driver) or None (files)
MANIFEST_SCHEMA = "cfbi-toolchain-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_EXT_PY = os.path.join("flash_kda", "cfbi", "_ext.py")      # relative to a tree


class Unusable(Exception):
    """Input that cannot be compared (exit status 2)."""


# ---- kernel names -----------------------------------------------------------------------------------------------------------------------------

def split_mangled(name):
    """``(namespace path, template arguments, parameters)`` of a simple Itanium-mangled name, or None.

    ``_ZN4cfbi9k2_kernelILi16EEEvNS_6ParamsE`` -> ``(("cfbi", "k2_kernel"), "Li16E", "vNS_6ParamsE")``. Only plain (nested) names with an optional template
    argument list are understood; the caller falls back to the mangled text for anything else.
    """
    if not name.startswith("_Z") or len(name) < 4:
        return None
    s, n = name, len(name)

    def source_name(i):
        j = i
        while j < n and s[j].isdigit():
            j += 1
        if j == i:
            raise ValueError
        k = int(s[i:j])
        if j + k > n:
            raise ValueError
        return s[j:j + k], j + k

    def skip_group(i):
        """Index after the group that opens at ``s[i]`` (one of I N X J) and closes with E."""
        depth = 0
        while i < n:
            c = s[i]
            if c in "INXJ":
                depth += 1
                i += 1
            elif c == "E":
                depth -= 1
                i += 1
                if depth == 0:
                    return i
            elif c.isdigit():
                i = source_name(i)[1]
            elif c == "L":                       # literal: type letters, value, closing E
                i += 1
                while s[i] != "E":
                    i += 1
                i += 1
            elif c == "S":                       # substitution: St Sa ... or S<seq-id>_
                i += 1
                if s[i].islower():
                    i += 1
                else:
                    while s[i] != "_":
                        i += 1
                    i += 1
            else:
                i += 1
        raise ValueError

    def template_args(i):
        """The text inside the template-argument group that opens at ``s[i]`` and the index after the group."""
        j = skip_group(i)
        return s[i + 1:j - 1], j

    try:
        i, path, targs = 2, [], ""
        if s[i] == "N":
            i += 1
            while s[i] != "E":
                if s[i].isdigit():
                    ident, i = source_name(i)
                    path.append(ident)
                elif s[i] == "I":
                    targs, i = template_args(i)
                else:
                    return None
            i += 1
        elif s[i].isdigit():
            ident, i = source_name(i)
            path.append(ident)
            if i < n and s[i] == "I":
                targs, i = template_args(i)
        else:
            return None
        return tuple(path), targs, s[i:]
    except (ValueError, IndexError):
        return None


def kernel_key(name):
    """Readable, mangling-independent key: ``cfbi::rows_kernel``, ``cfbi::k2_kernel<Li16E>``; the mangled text when the name is not understood."""
    sp = split_mangled(name)
    if sp is None or not sp[0]:
        return name
    return "::".join(sp[0]) + ("<%s>" % sp[1] if sp[1] else "")


def family(key):
    """The key without its template arguments."""
    return key.split("<", 1)[0]


def assign_keys(names):
    """``{key: mangled name}``; names that would share a key get their parameter text appended (deterministic)."""
    groups = {}
    for name in sorted(names):
        groups.setdefault(kernel_key(name), []).append(name)
    out = {}
    for key, same in groups.items():
        if len(same) == 1:
            out[key] = same[0]
        else:
            for name in same:
                sp = split_mangled(name)
                out["%s(%s)" % (key, sp[2] if sp else name)] = name
    return out


# ---- parsers of the tool outputs --------------------------------------------------------------------------------------------------------------

_OFFSET = re.compile(r"^\s*/\*[0-9a-f]{4,6}\*/\s*")
_COMMENT = re.compile(r"/\*.*?\*/")
_PTXAS_ENTRY = re.compile(r"ptxas info\s*:\s*Compiling entry function '([^']+)' for '(sm_\w+)'")
_PTXAS_STACK = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
_PTXAS_USED = re.compile(r"ptxas info\s*:\s*Used (\d+) registers")
_ELF_INFO = re.compile(r"^\.nv\.info\.(_Z\S+)\s*$")
_ELF_ATTR = re.compile(r"^\s*Attribute:\s*(EIATTR_\w+)")
_ELF_VALUE = re.compile(r"^\s*Value:\s*(.*)$")


def normalise_instruction(line):
    """The instruction text of one ``cuobjdump -sass`` line without offset, encoding comment and terminating ``;``; None for every other line.

    The control word of an instruction sits on a second, comment-only line: it is not an instruction and is dropped with the other encodings."""
    m = _OFFSET.match(line)
    if not m:
        return None
    body = _COMMENT.sub("", line[m.end():]).strip()
    if body.endswith(";"):
        body = body[:-1].rstrip()
    return " ".join(body.split()) or None


def strict_line(line):
    """Like ``normalise_instruction`` but keeping the encoding comments (and the comment-only control-word lines): diagnostic hash only."""
    m = _OFFSET.match(line)
    if m:
        return " ".join(line[m.end():].split()) or None
    text = line.strip()
    if text.startswith("/*") and text.endswith("*/"):
        return " ".join(text.split())
    return None


def parse_sass_functions(text):
    """``{(arch, mangled name): {"lines": [...], "strict": [...]}}`` of a ``cuobjdump -sass`` dump."""
    out, arch, func = {}, "?", None
    for line in text.splitlines():
        m = cb._ARCH.match(line)
        if m:
            arch, func = m.group(1), None
            continue
        m = cb._FUNC.match(line)
        if m:
            func = (arch, m.group(1))
            out[func] = {"lines": [], "strict": []}
            continue
        if func is None:
            continue
        norm = normalise_instruction(line)
        if norm is not None:
            out[func]["lines"].append(norm)
        st = strict_line(line)
        if st is not None:
            out[func]["strict"].append(st)
    return out


def parse_ptxas_log(text):
    """``{(arch, mangled name): {"stack", "spill_stores", "spill_loads", "registers"}}`` of the ``ptxas -v`` lines of a build log."""
    out, cur = {}, None
    for line in text.splitlines():
        m = _PTXAS_ENTRY.search(line)
        if m:
            cur = (m.group(2), m.group(1))
            out[cur] = {}
            continue
        if cur is None:
            continue
        m = _PTXAS_STACK.search(line)
        if m:
            out[cur].update(stack=int(m.group(1)), spill_stores=int(m.group(2)), spill_loads=int(m.group(3)))
            continue
        m = _PTXAS_USED.search(line)
        if m:
            out[cur]["registers"] = int(m.group(1))
    return out


def parse_elf_attributes(text):
    """``{(arch, mangled name): {"max_threads", "maxreg"}}`` from the ``.nv.info.<function>`` blocks of ``cuobjdump -elf``
    (``EIATTR_MAX_THREADS`` = the product of the block bounds, ``EIATTR_MAXREG_COUNT`` = the register bound that minBlocks of ``__launch_bounds__`` implies)."""
    out, arch, func, attr = {}, "?", None, None
    for line in text.splitlines():
        m = cb._ARCH.match(line)
        if m:
            arch, func, attr = m.group(1), None, None
            continue
        m = _ELF_INFO.match(line)
        if m:
            func, attr = (arch, m.group(1)), None
            out.setdefault(func, {})
            continue
        if func is None:
            continue
        if line and not line[0].isspace():       # next section header
            func = attr = None
            continue
        m = _ELF_ATTR.match(line)
        if m:
            attr = m.group(1)
            continue
        m = _ELF_VALUE.match(line)
        if m and attr in ("EIATTR_MAX_THREADS", "EIATTR_MAXREG_COUNT"):
            nums = [int(x, 16) for x in re.findall(r"0x[0-9a-fA-F]+", m.group(1))]
            if nums:
                if attr == "EIATTR_MAXREG_COUNT":
                    out[func]["maxreg"] = nums[0]
                else:
                    prod = 1
                    for v in nums:
                        prod *= v
                    out[func]["max_threads"] = prod
            attr = None
    return out


def digest(lines):
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def kernel_table(regs_text, sass_text, ptxas_text=None, elf_text=None, unit=None):
    """``{arch: {key: record}}`` from the text outputs of one build."""
    res = cb.parse_resource_usage(regs_text)
    sass = parse_sass_functions(sass_text)
    ptxas = parse_ptxas_log(ptxas_text) if ptxas_text else {}
    attrs = parse_elf_attributes(elf_text) if elf_text else {}
    if not res or not sass:
        raise Unusable("the resource dump or the SASS dump holds no function")
    only_res, only_sass = sorted(set(res) - set(sass)), sorted(set(sass) - set(res))
    if only_res or only_sass:
        raise Unusable("the resource dump and the SASS dump list different functions (resource dump only: %s; SASS dump only: %s)"
                       % (["%s %s" % k for k in only_res[:5]], ["%s %s" % k for k in only_sass[:5]]))
    stray = sorted(set(ptxas) - set(res))
    if stray:
        raise Unusable("the ptxas log names functions that the resource dump does not have (a log of another build?): %s" % ["%s %s" % k for k in stray[:5]])
    by_arch = {}
    for (arch, name), r in res.items():
        s = sass.get((arch, name))
        rec = {"name": name, "unit": unit, "reg": r["REG"], "stack": r["STACK"], "shared": r["SHARED"], "local": r["LOCAL"],
               "instructions": len(s["lines"]) if s else None,
               "sass_hash": digest(s["lines"]) if s else None,
               "sass_hash_strict": digest(s["strict"]) if s else None}
        p = ptxas.get((arch, name), {})
        rec["spill_stores"], rec["spill_loads"] = p.get("spill_stores"), p.get("spill_loads")
        a = attrs.get((arch, name), {})
        rec["max_threads"], rec["maxreg"] = a.get("max_threads"), a.get("maxreg")
        by_arch.setdefault(arch, {})[name] = rec
    out = {}
    for arch, recs in by_arch.items():
        keys = assign_keys(recs)
        out[arch] = {key: recs[name] for key, name in keys.items()}
    return out


def merge_tables(tables):
    """Merge the tables of several translation units; a function that two units define must have the same code."""
    out, warnings = {}, []
    for tab in tables:
        for arch, kernels in tab.items():
            dst = out.setdefault(arch, {})
            for key, rec in kernels.items():
                if key in dst:
                    if dst[key]["sass_hash"] != rec["sass_hash"]:
                        warnings.append("%s %s is defined by %s and %s with different code" % (arch, key, dst[key]["unit"], rec["unit"]))
                    continue
                dst[key] = rec
    return out, warnings


# ---- trees, toolchain, builds -----------------------------------------------------------------------------------------------------------------

def read_text(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _ext_assignment(tree, name):
    """The value expression of the first module-level assignment to ``name`` in the tree's ``flash_kda/cfbi/_ext.py`` (parsed, not imported), or None."""
    mod = ast.parse(read_text(os.path.join(tree, _EXT_PY)))
    for node in mod.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    return None


def units_of(tree):
    """The CUDA translation units of the extension: the ``.cu`` entries of ``_SOURCES`` in ``flash_kda/cfbi/_ext.py`` (parsed, not imported)."""
    try:
        sources = _ext_assignment(tree, "_SOURCES")
        if sources is not None:
            return ["csrc/cfbi/" + s for s in ast.literal_eval(sources) if s.endswith(".cu")]
    except (OSError, SyntaxError, ValueError):
        pass
    return list(DEFAULT_UNITS)


def ext_flags_of(tree):
    """``NVCC_FLAGS`` of the tree's ``flash_kda/cfbi/_ext.py`` (parsed, not imported), or None."""
    try:
        flags = _ext_assignment(tree, "NVCC_FLAGS")
    except (OSError, SyntaxError):
        return None
    return None if flags is None else list(ast.literal_eval(flags))


def tree_content_hash(tree):
    """Hash of every file under ``csrc`` plus ``flash_kda/cfbi/_ext.py``: what a build reads."""
    h = hashlib.sha256()
    paths = []
    for base, _, files in os.walk(os.path.join(tree, "csrc")):
        paths.extend(os.path.join(base, f) for f in files)
    ext = os.path.join(tree, _EXT_PY)
    if os.path.exists(ext):
        paths.append(ext)
    for p in sorted(paths):
        h.update(os.path.relpath(p, tree).encode("utf-8"))
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def _git(tree, *args):
    try:
        r = subprocess.run(["git", "-C", tree] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    except OSError:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def tree_commit(tree):
    marker = os.path.join(tree, "SNAPSHOT_COMMIT")
    if os.path.exists(marker):
        return read_text(marker).strip()
    return _git(tree, "rev-parse", "HEAD")


def find_toolchain(nvcc=None, ccbin=None):
    """``{"nvcc", "ccbin", "cuobjdump", "label", ...}`` of the toolchain (defaults: the conda environment ``tlang``)."""
    nvcc = nvcc or os.environ.get("CFBI_GATE_NVCC") or os.path.join(DEFAULT_CONDA, "bin", "nvcc")
    if not os.path.exists(nvcc):
        raise Unusable("nvcc not found at %s (use --nvcc or CFBI_GATE_NVCC)" % nvcc)
    bindir = os.path.dirname(os.path.abspath(nvcc))
    ccbin = ccbin or os.environ.get("CFBI_GATE_CCBIN") or os.path.join(bindir, "x86_64-conda-linux-gnu-g++")
    if not os.path.exists(ccbin):
        raise Unusable("host compiler not found at %s (use --ccbin or CFBI_GATE_CCBIN)" % ccbin)
    cuobjdump = os.path.join(bindir, "cuobjdump")
    if not os.path.exists(cuobjdump):
        raise Unusable("cuobjdump not found next to nvcc (%s)" % cuobjdump)

    def version_line(binary, pick):
        """The first non-empty line of ``binary --version`` that contains ``pick``, else its first non-empty line."""
        r = subprocess.run([binary, "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
        for ln in lines:
            if pick in ln:
                return ln.strip()
        return lines[0].strip() if lines else "?"

    ptxas = os.path.join(bindir, "ptxas")
    if not os.path.exists(ptxas):
        raise Unusable("ptxas not found next to nvcc (%s)" % ptxas)
    nvcc_version, host_version = version_line(nvcc, "release"), version_line(ccbin, "")
    return {"nvcc": nvcc, "ccbin": ccbin, "cuobjdump": cuobjdump, "nvcc_version": nvcc_version, "host_version": host_version,
            "cuobjdump_version": version_line(cuobjdump, "release"), "ptxas_version": version_line(ptxas, "release"),
            "label": "%s | %s" % (nvcc_version, host_version)}


def _run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return r.returncode, r.stdout, r.stderr


def gencode_flags(archs):
    out = []
    for arch in archs:
        out += ["-gencode", "arch=compute_%s,code=%s" % (arch[3:], arch)]
    return out


def compile_command(tc, unit, archs, defines, obj):
    return ([tc["nvcc"], "-ccbin", tc["ccbin"], "-std=" + STD] + PRODUCTION_FLAGS + ["-DCFBI_SRC_HASH=" + FIXED_SRC_HASH]
            + ["-D" + d for d in defines] + ["-I", "csrc"] + gencode_flags(archs) + ["--threads", str(max(1, len(archs))), "-c", unit, "-o", obj])


def build_unit(tree, tc, unit, archs, defines, work):
    """Compile one translation unit (from the tree root, so the identifier inside the object does not depend on the tree location) and read it."""
    obj = os.path.join(work, os.path.basename(unit) + ".o")
    rc, out, err = _run(compile_command(tc, unit, archs, defines, obj), cwd=tree)
    if rc != 0:
        raise Unusable("build of %s failed (exit %d):\n%s" % (unit, rc, "\n".join((err or out).splitlines()[-25:])))
    ptxas_text = out + "\n" + err
    texts = {}
    for name, flags in (("regs", ["--dump-resource-usage"]), ("sass", ["-sass"]), ("elf", ["-elf"])):
        rc, out, err = _run([tc["cuobjdump"]] + flags + [obj])
        if rc != 0:
            raise Unusable("cuobjdump %s of %s failed: %s" % (" ".join(flags), obj, err.strip()[:300]))
        texts[name] = out
    return kernel_table(texts["regs"], texts["sass"], ptxas_text, texts["elf"], unit=unit), texts, ptxas_text


def cache_key(toolchain, defines, units, content):
    """Key of a cached dump: the toolchain (compiler versions, flags, architectures), the source-variant defines, the units and the content hash of
    the sources. A change of any of them gives another key, so a cached base can never stand in for a different tree or toolchain."""
    blob = json.dumps({"toolchain": toolchain, "defines": sorted(defines), "units": list(units), "content": content, "schema": SCHEMA}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


def dump_tree(tree, archs=GATE_ARCHS, defines=(), tc=None, jobs=None, cache_dir=None, keep_dir=None):
    """The dump of one tree (a dict, schema ``cfbi-codegen-v1``); read from ``cache_dir`` when a dump with an identical key exists."""
    tree = os.path.abspath(tree)
    tc = tc or find_toolchain()
    archs, defines = list(archs), sorted(defines)
    units = units_of(tree)
    content = tree_content_hash(tree)
    commit = tree_commit(tree)
    toolchain = {"source": "nvcc", "label": tc["label"], "nvcc": tc["nvcc_version"], "host_compiler": tc["host_version"], "cuobjdump": tc["cuobjdump_version"],
                 "ptxas": tc["ptxas_version"], "flags": list(PRODUCTION_FLAGS), "std": STD, "archs": archs, "src_hash": FIXED_SRC_HASH}
    key = cache_key(toolchain, defines, units, content)
    cache_file = os.path.join(cache_dir, "codegen-%s.json" % key) if cache_dir else None
    if cache_file and os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            dump = json.load(f)
        dump["meta"]["cache"] = "hit"
        return dump
    tree_flags = ext_flags_of(tree)
    warnings = []
    if tree_flags is not None and tree_flags != PRODUCTION_FLAGS:
        warnings.append("NVCC_FLAGS of the tree differ from the flags of this tool: %s" % tree_flags)
    work = keep_dir or tempfile.mkdtemp(prefix="codegen-")
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs or min(8, max(1, len(units)))) as pool:
            futs = [pool.submit(build_unit, tree, tc, u, archs, defines, work) for u in units]
            results = [fut.result() for fut in futs]
    finally:
        if not keep_dir:
            shutil.rmtree(work, ignore_errors=True)
    tables = [r[0] for r in results]
    kernels, merge_warnings = merge_tables(tables)
    if keep_dir:
        for u, (_, texts, ptxas_text) in zip(units, results):
            stem = os.path.join(keep_dir, os.path.basename(u))
            for name, text in list(texts.items()) + [("ptxas", ptxas_text)]:
                with open("%s.%s.txt" % (stem, name), "w", encoding="utf-8") as f:
                    f.write(text)
    dump = {"schema": SCHEMA,
            "meta": {"toolchain": toolchain, "variant": {"defines": defines},
                     "tree": {"path": tree, "commit": commit, "content_hash": content, "units": units},
                     "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "build_seconds": round(time.time() - t0, 1),
                     "cache": "miss" if cache_file else "off", "warnings": warnings + merge_warnings},
            "kernels": kernels}
    check_dump(dump, tree)                           # an incomplete dump is never cached and never compared
    if cache_file:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = cache_file + ".tmp%d" % os.getpid()
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(dump, f, indent=1, sort_keys=True)
        os.replace(tmp, cache_file)
    return dump


# ---- dumps of another toolchain (the container): the manifest ---------------------------------------------------------------------------------

def parse_nvcc_command(line):
    """The compile parameters of one nvcc command line of a torch JIT build log, or None when the line is not an nvcc compile of a source file.

    Returns ``{"unit", "flags", "std", "archs", "src_hash", "variant_defines"}``. Include paths, dependency files, the output, the extension name and the per-tree
    source hash are no toolchain parameters; the source hash and the ``CFBI_*`` source-variant defines are returned apart so that a change of the sources or of a
    variant is never read as a change of the toolchain."""
    try:
        tokens = shlex.split(line)
    except ValueError:
        return None
    start = next((i for i, t in enumerate(tokens) if os.path.basename(t) == "nvcc"), None)
    if start is None:
        return None
    toks = tokens[start + 1:]
    unit, flags, std, archs, src_hash, variants = None, [], None, [], None, []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in ("-MF", "-o", "-I", "-isystem"):
            i += 2
            continue
        if t in ("-MD", "-c") or t.startswith("-I"):
            if t == "-c" and i + 1 < len(toks) and not toks[i + 1].startswith("-"):
                unit = toks[i + 1]
                i += 1
        elif t == "--compiler-options" and i + 1 < len(toks):
            flags.append("--compiler-options " + toks[i + 1])
            i += 1
        elif t.startswith("-DTORCH_EXTENSION_NAME="):
            pass
        elif t.startswith("-DCFBI_SRC_HASH="):
            src_hash = t.split("=", 1)[1]
        elif t.startswith("-DCFBI_"):
            variants.append(t[2:])
        elif t.startswith("-std="):
            std = t[5:]
        elif t.startswith("-"):
            flags.append(t)
            m = re.match(r"^-gencode=arch=compute_\w+,code=(\S+)$", t)
            if m:
                archs += [c for c in m.group(1).split(",") if c.startswith("sm_")]
        elif t.endswith((".cu", ".cpp", ".cc")):
            unit = t
        i += 1
    if unit is None:
        return None
    return {"unit": unit, "flags": flags, "std": std, "archs": sorted(set(archs)), "src_hash": src_hash, "variant_defines": sorted(variants)}


def parse_tool_versions(text):
    """``{"nvcc", "ptxas", "cuobjdump", "host_compiler"}`` from the ``TOOLCHAIN <tool>: ...`` lines that ``toolchain_probe.py`` prints inside the container."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"^TOOLCHAIN (nvcc|ptxas|cuobjdump): (.*)$", line)
        if m:
            v = re.search(r"Cuda compilation tools, release (\d+\.\d+), V(\d+\.\d+\.\d+)", m.group(2))
            if v:
                out[m.group(1)] = "Cuda compilation tools, release %s, V%s" % v.groups()
            continue
        m = re.match(r"^TOOLCHAIN host: (.*)$", line)
        if m and m.group(1).strip() and not m.group(1).startswith("unavailable"):
            out["host_compiler"] = m.group(1).split(" | ")[0].strip()
    return out


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_log_parameters(text, name="the build log"):
    """``(units, first)`` of a verbose torch JIT build log: the parsed nvcc compile of every ``.cu`` unit and the common parameter set (``flags``, ``std``, ``archs``,
    ``src_hash``, ``variant_defines``). Raises ``Unusable`` for a log without a compile of a ``.cu`` file (the log of a cached build has none), for units compiled
    with different parameters and for a command line without a standard or an architecture."""
    units = [c for c in (parse_nvcc_command(ln) for ln in text.splitlines()) if c and c["unit"].endswith(".cu")]
    if not units:
        raise Unusable("the build log %s holds no nvcc compile of a .cu file: it cannot describe a toolchain (the log of a cached build has none, rebuild with verbose output)" % name)
    first = units[0]
    for u in units[1:]:
        for f in ("flags", "std", "archs", "src_hash", "variant_defines"):
            if u[f] != first[f]:
                raise Unusable("the units of the build were compiled with different %s: %s vs %s" % (f, first[f], u[f]))
    if not first["std"] or not first["archs"]:
        raise Unusable("the nvcc command line of the build log names no standard (%r) or no architecture (%r)" % (first["std"], first["archs"]))
    return units, first


def bind_manifest_to_log(manifest, text, name="the build log"):
    """Raise ``Unusable`` unless ``text`` is the build log that the manifest was made from: the digest of the log equals the recorded one and the compile commands in
    it give the parameters the manifest states (flags, standard, architectures, variant defines, source hash, units and number of compile lines). A manifest and a
    log that do not belong together would let one build's flags describe another build's dump."""
    log = manifest["build_log"]
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != log["sha256"]:
        raise Unusable("the build log %s (sha256 %s...) is not the log the manifest was made from (sha256 %s...)" % (name, digest[:12], log["sha256"][:12]))
    units, first = build_log_parameters(text, name)
    conflicts = ["%s: log %s, manifest %s" % (f, first[f], manifest.get(f)) for f in ("flags", "std", "archs", "src_hash", "variant_defines") if first[f] != manifest.get(f)]
    names = sorted(os.path.basename(u["unit"]) for u in units)
    if names != sorted(log.get("units") or []) or len(units) != log["nvcc_compile_lines"]:
        conflicts.append("units: log %s (%d compile lines), manifest %s (%s compile lines)" % (names, len(units), sorted(log.get("units") or []), log["nvcc_compile_lines"]))
    if conflicts:
        raise Unusable("the compile commands of the build log %s contradict the manifest: %s" % (name, "; ".join(conflicts)))


def manifest_from_build(build_log, tool_versions, label, allow_non_production_flags=False, artifact=None):
    """The toolchain manifest of a production build: the nvcc command lines of its build log (verbose torch JIT build) plus the probed tool versions.

    The build log must contain the compile of at least one ``.cu`` file (the log of a cached build does not and is refused); every unit must have been compiled with
    the same parameters; the production flags of ``_ext.py`` must all be present. The sha256 of the log is recorded so that the manifest names the build it describes;
    ``artifact`` (the built extension that the resource / SASS / ELF dumps were taken from) is recorded with its sha256 as well."""
    text, versions_text = read_text(build_log), read_text(tool_versions)
    units, first = build_log_parameters(text, build_log)
    missing = [f for f in PRODUCTION_FLAGS if f not in first["flags"]]
    if missing and not allow_non_production_flags:
        raise Unusable("the build did not use the production flags (missing %s)" % missing)
    versions = parse_tool_versions(versions_text)
    absent = [f for f in ("nvcc", "ptxas", "cuobjdump", "host_compiler") if not versions.get(f)]
    if absent:
        raise Unusable("the tool-version file %s has no %s" % (tool_versions, ", ".join(absent)))
    manifest = {"schema": MANIFEST_SCHEMA, "label": label, "nvcc": versions["nvcc"], "host_compiler": versions["host_compiler"], "cuobjdump": versions["cuobjdump"],
                "ptxas": versions["ptxas"], "std": first["std"], "flags": first["flags"], "archs": first["archs"], "variant_defines": first["variant_defines"],
                "src_hash": first["src_hash"],
                "build_log": {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "nvcc_compile_lines": len(units),
                              "units": sorted(os.path.basename(u["unit"]) for u in units)},
                "tool_versions_sha256": hashlib.sha256(versions_text.encode("utf-8")).hexdigest()}
    if artifact:
        manifest["artifact"] = {"path": os.path.abspath(artifact), "sha256": file_sha256(artifact), "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(artifact)))}
    return manifest


def validate_manifest(m, where="toolchain manifest"):
    if not isinstance(m, dict) or m.get("schema") != MANIFEST_SCHEMA:
        raise Unusable("%s: schema is not %s" % (where, MANIFEST_SCHEMA))
    problems = toolchain_problems(dict(m, source="files"))
    log = m.get("build_log")
    if not isinstance(log, dict) or not (isinstance(log.get("sha256"), str) and _HEX64.match(log["sha256"])) or not log.get("nvcc_compile_lines"):
        problems.append("build_log must carry the sha256 of the build log and the number of nvcc compile lines")
    art = m.get("artifact")
    if art is not None and not (isinstance(art, dict) and isinstance(art.get("sha256"), str) and _HEX64.match(art["sha256"])):
        problems.append("artifact must carry the sha256 of the built extension")
    if problems:
        raise Unusable("%s: %s" % (where, "; ".join(problems)))
    return m


def dump_from_files(regs, sass, manifest, elf, ptxas_log, extra_meta=None):
    """A dump from the cuobjdump text of another toolchain (the container), with the structured toolchain manifest of the build that produced it.

    Every input is required: the resource dump, the SASS dump, the ELF attributes (launch bounds), the verbose build log (it holds the ``ptxas -v`` lines) and the manifest;
    a dump that cannot show spills and launch bounds, or whose toolchain is only a label, is not accepted as a gate input. The build log must be the very log the
    manifest was made from: its digest and the compile commands in it are checked against the manifest (``bind_manifest_to_log``)."""
    if not (regs and sass and manifest and elf and ptxas_log):
        raise Unusable("a dump of another toolchain needs the resource dump, the SASS dump, the ELF attributes, the ptxas log and the toolchain manifest")
    m = validate_manifest(manifest if isinstance(manifest, dict) else _read_json(manifest, "toolchain manifest"))
    log_text = read_text(ptxas_log)
    bind_manifest_to_log(m, log_text, ptxas_log)
    tables = kernel_table(read_text(regs), read_text(sass), log_text, read_text(elf), unit=None)
    archs = sorted(tables)
    if archs != sorted(m["archs"]):
        raise Unusable("the manifest names the architectures %s, the dumps hold %s" % (sorted(m["archs"]), archs))
    meta = {"toolchain": {"source": "files", "label": m["label"], "nvcc": m["nvcc"], "host_compiler": m["host_compiler"], "cuobjdump": m["cuobjdump"], "ptxas": m["ptxas"],
                          "flags": list(m["flags"]), "std": m["std"], "archs": archs, "src_hash": None},
            "variant": {"defines": list(m.get("variant_defines", []))},
            "tree": {"path": None, "commit": None, "content_hash": None, "src_hash": m.get("src_hash"), "units": list(m["build_log"].get("units", []))},
            "build_log": dict(m["build_log"]), "artifact": m.get("artifact"), "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cache": "off", "warnings": []}
    if extra_meta:
        meta["tree"].update(extra_meta)
    return check_dump({"schema": SCHEMA, "meta": meta, "kernels": tables}, "dump of files")


def matrix_build(tree, archs=MATRIX_ARCHS, tc=None, jobs=None):
    """Compile every unit for every architecture separately: ``[(unit, arch, ok, message)]``."""
    tree = os.path.abspath(tree)
    tc = tc or find_toolchain()
    units = units_of(tree)
    work = tempfile.mkdtemp(prefix="codegen-matrix-")

    def one(unit, arch):
        obj = os.path.join(work, "%s.%s.o" % (os.path.basename(unit), arch))
        rc, out, err = _run(compile_command(tc, unit, [arch], [], obj), cwd=tree)
        if rc == 0:
            return unit, arch, True, ""
        return unit, arch, False, "\n".join((err or out).splitlines()[-8:])

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs or 8) as pool:
            futs = [pool.submit(one, u, a) for u in units for a in archs]
            return [f.result() for f in futs]
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---- comparison -------------------------------------------------------------------------------------------------------------------------------

def _read_json(path, what):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise Unusable("cannot read the %s %s: %s" % (what, path, e))


def load_dump(path):
    return check_dump(_read_json(path, "dump"), path)


def _is_count(value):
    return type(value) is int and value >= 0


def kernel_record_problems(rec):
    """Why one kernel record cannot be compared (an empty list means complete): every figure that a comparison reads must be present, an unknown is never an equality."""
    if not isinstance(rec, dict):
        return ["the record is not an object"]
    out = ["%s is %r, not a non-negative integer" % (f, rec.get(f)) for f in _RESOURCE_FIELDS + _SPILL_FIELDS if not _is_count(rec.get(f))]
    if not _is_count(rec.get("instructions")) or rec["instructions"] == 0:
        out.append("no SASS instructions (%r)" % (rec.get("instructions"),))
    out += ["%s is not a SHA-256 hex digest" % f for f in ("sass_hash", "sass_hash_strict") if not (isinstance(rec.get(f), str) and _HEX64.match(rec[f]))]
    out += ["%s is %r, not a positive integer" % (f, rec[f]) for f in _ATTR_FIELDS if rec.get(f) is not None and (type(rec[f]) is not int or rec[f] <= 0)]
    return out


def toolchain_problems(toolchain):
    """Why a toolchain description cannot prove that two dumps share a toolchain (missing or empty fields)."""
    out = []
    for f in _TOOLCHAIN_REQUIRED:
        v = toolchain.get(f)
        if v is None or v == "" or v == [] or (f in ("flags", "archs") and not (isinstance(v, list) and all(isinstance(x, str) and x for x in v))):
            out.append("toolchain field %s is missing or empty (%r)" % (f, v))
    return out


def check_dump(dump, label="dump"):
    """The dump itself, or ``Unusable``: a complete toolchain description and complete evidence for every kernel of every architecture."""
    if not isinstance(dump, dict) or dump.get("schema") != SCHEMA:
        raise Unusable("%s: schema is not %s" % (label, SCHEMA))
    if not isinstance(dump.get("kernels"), dict) or not isinstance(dump.get("meta"), dict) or not isinstance(dump["meta"].get("toolchain"), dict):
        raise Unusable("%s: malformed dump" % label)
    problems = toolchain_problems(dump["meta"]["toolchain"])
    archs = dump["meta"]["toolchain"].get("archs") or []
    for arch in archs:
        if not dump["kernels"].get(arch):
            raise Unusable("%s: no kernels for %s (an empty kernel set would make every comparison vacuous)" % (label, arch))
    if not dump["kernels"]:
        raise Unusable("%s: no kernels at all" % label)
    if sorted(dump["kernels"]) != sorted(archs):
        problems.append("the kernels are listed for %s, the toolchain names %s" % (sorted(dump["kernels"]), sorted(archs)))
    for arch in sorted(dump["kernels"]):
        for key, rec in sorted(dump["kernels"][arch].items()):
            problems += ["%s %s: %s" % (arch, key, p) for p in kernel_record_problems(rec)]
            if isinstance(rec, dict) and family(key) in DEFAULT_OWNED:           # the row engine's launch bounds are what the resource gate compares
                problems += ["%s %s: the launch-bound attribute %s of the row-engine family is missing (no ELF attributes in the dump?)" % (arch, key, f)
                             for f in _ATTR_FIELDS if rec.get(f) is None]
    if problems:
        shown = problems[:12] + (["... and %d more" % (len(problems) - 12)] if len(problems) > 12 else [])
        raise Unusable("%s: incomplete evidence, a missing figure is never taken as an equality:\n  %s" % (label, "\n  ".join(shown)))
    return dump


def load_allow(path):
    if path is None:
        return {"aliases": {}, "may_differ": []}
    allow = _read_json(path, "allow-list")
    if allow.get("schema") != ALLOW_SCHEMA:
        raise Unusable("%s: schema is not %s" % (path, ALLOW_SCHEMA))
    allow.setdefault("aliases", {})
    allow.setdefault("may_differ", [])
    return allow


def require_same_toolchain(a, b, what):
    ta, tb = a["meta"]["toolchain"], b["meta"]["toolchain"]
    bad = ["%s: %r vs %r" % (f, ta.get(f), tb.get(f)) for f in _TOOLCHAIN_FIELDS if ta.get(f) != tb.get(f)]
    if bad:
        raise Unusable("%s were built with different toolchains (%s); only same-toolchain dumps can be compared. Source-variant defines are recorded "
                       "separately and may differ." % (what, "; ".join(bad)))


def _field_diffs(b, c):
    out = []
    for f in _RESOURCE_FIELDS + ("instructions",):
        if b.get(f) != c.get(f):
            out.append("%s %s -> %s" % (f.upper(), b.get(f), c.get(f)))
    for f in _SPILL_FIELDS + _ATTR_FIELDS:
        if b.get(f) != c.get(f):
            out.append("%s %s -> %s" % (f.upper(), b.get(f), c.get(f)))
    if b.get("sass_hash") != c.get("sass_hash"):
        out.append("normalised SASS differs")
    return out


def gate_r_reference(base_kernels, bkey=None, owned=DEFAULT_OWNED):
    """The base kernel whose resource figures bound a resource-gate target whose key in the base is ``bkey`` (``base_kernels``: ``{key: record}`` of one architecture): the base kernel of
    that very key when it belongs to the owned family, so that every instantiation of the row engine is held to its own figures; otherwise the base's row engine: the plain
    ``rows_kernel`` of a base from before the instantiations, else the instantiation behind the plan kernel. ``None`` when the base has none of them."""
    if bkey is not None and family(bkey) in owned and bkey in base_kernels:
        return base_kernels[bkey]
    for key in (GATE_R_REF, GATE_R_REF_INSTANCE):
        if key in base_kernels:
            return base_kernels[key]
    return None


def gate_r_role(ckey, aliases, base_keys, owned):
    """The role of a candidate kernel for the resource gate: ``owned`` (a kernel of the owned family, or an alias of one: a rename does not take it out of the gate),
    ``new`` (only the candidate has it) or None (an existing kernel that is not owned, which the gate does not bind: K2 needs more than 128 registers)."""
    bkey = aliases.get(ckey, ckey)
    if family(ckey) in owned or family(bkey) in owned:
        return "owned"
    return "new" if bkey not in base_keys else None


def _gate_r(arch, key, rec, ref, role):
    """Failures of the resource gate for one kernel (``rec``) against the base reference kernel ``ref``; ``role`` is ``owned`` or ``new`` (``gate_r_role``). The owned
    family and every new kernel are bound alike: at most 128 registers, stack and spill not above the base's, and the launch-bound attributes present and equal to the
    base's (``__launch_bounds__(256, 2)``): a missing attribute is unusable evidence (exit 2), a different value a failure (exit 1)."""
    if role not in ("owned", "new"):
        raise ValueError("the role of a resource-gate target is owned or new, got %r" % (role,))
    if ref is None:
        raise Unusable("%s %s: the base has no %s (nor %s) to bound the stack and spill figures, the resource gate cannot be evaluated" % (arch, key, GATE_R_REF, GATE_R_REF_INSTANCE))
    fails = []
    if rec["reg"] > REG_CAP:
        fails.append("%d registers (limit %d)" % (rec["reg"], REG_CAP))
    if rec["stack"] > ref["stack"]:
        fails.append("stack %d B above the base %d B" % (rec["stack"], ref["stack"]))
    for f in _SPILL_FIELDS:
        if rec[f] > ref[f]:
            fails.append("%s %d B above the base %d B" % (f.replace("_", " "), rec[f], ref[f]))
    for f in _ATTR_FIELDS:
        if rec.get(f) is None or ref.get(f) is None:
            raise Unusable("%s %s: the launch-bound attribute %s is missing (%s kernel %r, base reference %r), the resource gate cannot be evaluated"
                           % (arch, key, f, role, rec.get(f), ref.get(f)))
        if rec[f] != ref[f]:
            fails.append("launch-bound attribute %s %d differs from the base %d" % (f, rec[f], ref[f]))
    return fails


def compare_dumps(base, cand, allow=None, mode="identity", prev=None, owned=DEFAULT_OWNED):
    """Compare two dumps. Returns a dict: ``exit`` (0 / 1), ``differences``, ``allowed``, ``added``, ``gate_r``, ``notes``, ``compared``.

    Raises ``Unusable`` when the dumps cannot be compared (toolchain mismatch, empty kernel sets, ambiguous aliases)."""
    if mode not in ("identity", "change"):
        raise Unusable("mode must be identity or change")
    allow = allow or {"aliases": {}, "may_differ": []}
    check_dump(base, "base")
    check_dump(cand, "candidate")
    require_same_toolchain(base, cand, "the base and the candidate")
    if prev is not None:
        check_dump(prev, "previous")
        require_same_toolchain(base, prev, "the base and the previous accepted state")
    aliases, may_differ, owned = dict(allow.get("aliases", {})), set(allow.get("may_differ", [])), set(owned)
    diffs, allowed, added, gate, notes = [], [], [], {"checked": [], "failures": []}, []
    compared = 0

    def check_gate_r(arch, ckey, rec, ref, role):
        gate["checked"].append((arch, ckey))
        gate["failures"].extend("%s %s: %s" % (arch, ckey, why) for why in _gate_r(arch, ckey, rec, ref, role))

    base_content = base["meta"]["tree"].get("content_hash")
    if base_content and base_content == cand["meta"]["tree"].get("content_hash") and base["meta"].get("variant") == cand["meta"].get("variant"):
        notes.append("the candidate tree and the base tree have the same content and variant: an A/A comparison")
    used_alias = set()
    for arch in sorted(base["kernels"]):
        bk, ck = base["kernels"][arch], cand["kernels"].get(arch)
        if not ck:
            diffs.append("%s: no kernels in the candidate" % arch)
            continue
        by_base = {}
        for ckey in ck:
            bkey = aliases.get(ckey, ckey)
            if ckey in aliases:
                used_alias.add(ckey)
            if bkey in by_base:
                raise Unusable("%s: the candidate kernels %s and %s both map to the base kernel %s" % (arch, by_base[bkey], ckey, bkey))
            by_base[bkey] = ckey
        for bkey in sorted(bk):
            ckey = by_base.get(bkey)
            if ckey is None:
                diffs.append("%s %s: missing in the candidate" % (arch, bkey))
                continue
            compared += 1
            changes = _field_diffs(bk[bkey], ck[ckey])
            if not changes:
                if bk[bkey].get("sass_hash_strict") != ck[ckey].get("sass_hash_strict"):
                    notes.append("%s %s: instructions equal, encodings or control words differ (scheduling only)" % (arch, bkey))
                continue
            entry = "%s %s: %s" % (arch, bkey, ", ".join(changes))
            tolerated = bkey in may_differ or family(bkey) in may_differ or (mode == "change" and family(bkey) in owned)
            if tolerated:
                allowed.append(entry)
                role = gate_r_role(ckey, aliases, bk, owned)
                if role == "owned":                          # the resource gate binds the row-engine family and new kernels, not every tolerated change (K2 needs more than 128 registers)
                    check_gate_r(arch, ckey, ck[ckey], gate_r_reference(bk, bkey, owned), role)
            else:
                diffs.append(entry)
        for ckey in sorted(ck):
            if aliases.get(ckey, ckey) in bk:
                continue
            added.append("%s %s" % (arch, ckey))
            check_gate_r(arch, ckey, ck[ckey], gate_r_reference(bk, aliases.get(ckey, ckey), owned), gate_r_role(ckey, aliases, bk, owned))
        if mode == "change":                                          # the owned family is gated even when it did not change against the base, an alias of it included
            for ckey in sorted(ck):
                if gate_r_role(ckey, aliases, bk, owned) == "owned" and (arch, ckey) not in gate["checked"]:
                    check_gate_r(arch, ckey, ck[ckey], gate_r_reference(bk, aliases.get(ckey, ckey), owned), "owned")
    for ckey in sorted(set(aliases) - used_alias):
        notes.append("allow-list alias %s matches no candidate kernel" % ckey)
    if prev is not None and mode == "change":
        for arch in sorted(cand["kernels"]):
            for ckey, rec in sorted(cand["kernels"][arch].items()):
                if family(ckey) not in owned:
                    continue
                prev_rec = prev["kernels"].get(arch, {}).get(aliases.get(ckey, ckey))
                if prev_rec is None:
                    state = "new"
                elif _field_diffs(prev_rec, rec):
                    state = "changed"
                else:
                    state = "unchanged"
                notes.append("%s %s: %s against the previous accepted state" % (arch, ckey, state))
    if compared == 0:
        raise Unusable("no kernel of the base was compared")
    exit_code = 1 if (diffs or gate["failures"]) else 0
    return {"mode": mode, "exit": exit_code, "compared": compared, "differences": diffs, "allowed": allowed, "added": added, "gate_r": gate,
            "notes": notes}


def literal_check(dump):
    """Warnings for deviations of the base ``rows_kernel`` figures from the recorded literals (toolchain drift)."""
    out = []
    for arch, lit in LITERALS.items():
        rec = gate_r_reference(dump["kernels"].get(arch, {}))
        if rec is None:
            continue
        for f, want in sorted(lit.items()):
            if rec.get(f) is not None and rec[f] != want:
                out.append("%s %s of %s is %s, the recorded figure is %s (toolchain drift)" % (arch, f, GATE_R_REF, rec[f], want))
    return out


def format_report(rep, base=None):
    def figures(pairs):
        return ", ".join("%s %s" % (f.replace("_", " "), v) for f, v in pairs)

    lines = []
    for d in rep["differences"]:
        lines.append("DIFF: %s" % d)
    for d in rep["allowed"]:
        lines.append("ALLOWED DIFFERENCE: %s" % d)
    for a in rep["added"]:
        lines.append("ADDED (candidate only): %s" % a)
    for f in rep["gate_r"]["failures"]:
        lines.append("GATE R: %s" % f)
    for n in rep["notes"]:
        lines.append("NOTE: %s" % n)
    if base is not None:
        for arch in sorted(LITERALS):
            rec = gate_r_reference(base["kernels"].get(arch, {}))
            if rec is not None:
                recorded = sorted(LITERALS[arch].items())
                measured = [(f, rec.get(f)) for f, _ in recorded]
                lines.append("CROSS-CHECK %s %s: %s (recorded: %s)" % (arch, GATE_R_REF, figures(measured), figures(recorded)))
        for w in literal_check(base):
            lines.append("WARNING: %s" % w)
    lines.append("kernels compared: %d, differences: %d, allowed: %d, added: %d, gate R checked: %d, gate R failures: %d"
                 % (rep["compared"], len(rep["differences"]), len(rep["allowed"]), len(rep["added"]), len(rep["gate_r"]["checked"]),
                    len(rep["gate_r"]["failures"])))
    lines.append("VERDICT: %s" % ("PASS" if rep["exit"] == 0 else "FAIL"))
    return "\n".join(lines)


# ---- command line -----------------------------------------------------------------------------------------------------------------------------

def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")


def _kernel_summary(dump):
    return "%d kernels on %s" % (sum(len(v) for v in dump["kernels"].values()), ", ".join(sorted(dump["kernels"])))


def _compare_and_print(base, cand, a, prev=None):
    if a.allow:
        allow_path = a.allow
    elif os.path.exists(DEFAULT_ALLOW):
        allow_path = DEFAULT_ALLOW
    else:
        allow_path = None
    rep = compare_dumps(base, cand, load_allow(allow_path), a.mode, prev=prev)
    print(format_report(rep, base))
    if a.json:
        _write_json(a.json, rep)
    return rep["exit"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    tool = argparse.ArgumentParser(add_help=False)
    tool.add_argument("--nvcc", default=None)
    tool.add_argument("--ccbin", default=None)
    tool.add_argument("--jobs", type=int, default=None)
    tool.add_argument("--archs", nargs="+", default=None)

    p = sub.add_parser("dump", parents=[tool])
    p.add_argument("--tree", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--define", action="append", default=[])
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--keep-dir", default=None)

    p = sub.add_parser("manifest")
    p.add_argument("--build-log", required=True)
    p.add_argument("--tool-versions", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-non-production-flags", action="store_true")
    p.add_argument("--artifact", default=None, help="the built extension (.so) that the resource, SASS and ELF dumps were taken from; its sha256 is recorded")

    p = sub.add_parser("dump-files")
    p.add_argument("--manifest", required=True)
    p.add_argument("--regs", required=True)
    p.add_argument("--sass", required=True)
    p.add_argument("--elf", required=True)
    p.add_argument("--ptxas-log", required=True, help="the verbose build log the manifest was made from (its digest and compile commands are checked against the manifest)")
    p.add_argument("--commit", default=None, help="the commit of the source tree that was built (recorded as the tree identity of the dump)")
    p.add_argument("--out", required=True)

    p = sub.add_parser("compare")
    p.add_argument("--base", required=True)
    p.add_argument("--cand", required=True)
    p.add_argument("--prev", default=None)
    p.add_argument("--mode", choices=("identity", "change"), default="identity")
    p.add_argument("--allow", default=None)
    p.add_argument("--json", default=None)

    p = sub.add_parser("gate", parents=[tool])
    p.add_argument("--base-tree", required=True)
    p.add_argument("--cand-tree", required=True)
    p.add_argument("--prev-tree", default=None)
    p.add_argument("--mode", choices=("identity", "change"), default="identity")
    p.add_argument("--allow", default=None)
    p.add_argument("--json", default=None)
    p.add_argument("--base-define", action="append", default=[])
    p.add_argument("--cand-define", action="append", default=[])
    p.add_argument("--cache-dir", default=None)

    p = sub.add_parser("matrix", parents=[tool])
    p.add_argument("--tree", required=True)

    a = ap.parse_args(argv)
    if not a.cmd:
        ap.print_usage(sys.stderr)
        return 2
    try:
        if a.cmd == "dump":
            dump = dump_tree(a.tree, a.archs or GATE_ARCHS, a.define, find_toolchain(a.nvcc, a.ccbin), a.jobs, a.cache_dir, a.keep_dir)
            _write_json(a.out, dump)
            print("dump of %s: %s (cache: %s)" % (a.tree, _kernel_summary(dump), dump["meta"].get("cache")))
            return 0
        if a.cmd == "manifest":
            m = validate_manifest(manifest_from_build(a.build_log, a.tool_versions, a.label, a.allow_non_production_flags, a.artifact))
            _write_json(a.out, m)
            print("manifest of %s: nvcc %s, host %s, %s, %d flags, %s" % (a.build_log, m["nvcc"], m["host_compiler"], m["std"], len(m["flags"]), ",".join(m["archs"])))
            return 0
        if a.cmd == "dump-files":
            dump = dump_from_files(a.regs, a.sass, a.manifest, a.elf, a.ptxas_log, {"commit": a.commit} if a.commit else None)
            _write_json(a.out, dump)
            print("dump of %s: %s" % (a.sass, _kernel_summary(dump)))
            return 0
        if a.cmd == "compare":
            return _compare_and_print(load_dump(a.base), load_dump(a.cand), a, load_dump(a.prev) if a.prev else None)
        if a.cmd == "gate":
            tc, archs = find_toolchain(a.nvcc, a.ccbin), a.archs or GATE_ARCHS
            base = dump_tree(a.base_tree, archs, a.base_define, tc, a.jobs, a.cache_dir)
            cand = dump_tree(a.cand_tree, archs, a.cand_define, tc, a.jobs, None)
            prev = dump_tree(a.prev_tree, archs, a.base_define, tc, a.jobs, a.cache_dir) if a.prev_tree else None
            print("toolchain: %s" % tc["label"])
            print("base: %s (cache %s), candidate: %s" % (base["meta"]["tree"].get("commit"), base["meta"].get("cache"), cand["meta"]["tree"].get("commit")))
            return _compare_and_print(base, cand, a, prev)
        if a.cmd == "matrix":
            res = matrix_build(a.tree, a.archs or MATRIX_ARCHS, find_toolchain(a.nvcc, a.ccbin), a.jobs)
            bad = [r for r in res if not r[2]]
            for unit, arch, ok, msg in res:
                print("%-4s %s %s" % ("OK" if ok else "FAIL", arch, unit))
                if not ok:
                    print(msg)
            print("compile matrix: %d of %d builds failed" % (len(bad), len(res)))
            return 1 if bad else 0
    except Unusable as e:
        print("unusable input: %s" % e, file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
