#!/usr/bin/env python3
"""Compare the resource usage and the SASS of the kernels of two built CF-BI extensions (cuobjdump text; plain Python, no torch).

usage: python benchmarks/compare_binaries.py --base-regs A.regs.txt --cand-regs B.regs.txt [--base-sass A.sass.txt --cand-sass B.sass.txt]
           [--functions SUBSTR ...]

The inputs are the outputs of ``cuobjdump --dump-resource-usage <extension.so>`` and ``cuobjdump -sass <extension.so>`` (the node driver's modes
``regs`` and ``sass``). For every function (per architecture) of the baseline the tool requires equal REG, STACK (spill frame), SHARED and LOCAL
figures, an equal SASS instruction count and an equal opcode histogram in the candidate. Functions that only exist in the candidate (new kernels)
are listed and are not a difference; a baseline function missing in the candidate is. Equality is an equivalence heuristic, not proof of identical
machine code: a difference is recorded and the timing gate decides.

exit status: 0 equal, 1 differences, 2 unusable input
"""
import argparse
import re
import sys
from collections import Counter

_ARCH = re.compile(r"^\s*arch\s*=\s*(\S+)")
_FUNC = re.compile(r"^\s*Function\s*:?\s*(\S+?)\s*:?\s*$")
_RES = re.compile(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)")
_INSN = re.compile(r"^\s*/\*([0-9a-f]{4,6})\*/\s+(.*?)\s*;\s*(/\*.*)?$")
_RESOURCE_FIELDS = ("REG", "STACK", "SHARED", "LOCAL")


def parse_resource_usage(text):
    """``{(arch, function): {"REG": n, "STACK": n, "SHARED": n, "LOCAL": n}}`` of a ``cuobjdump --dump-resource-usage`` dump."""
    out, arch, func = {}, "?", None
    for line in text.splitlines():
        m = _ARCH.match(line)
        if m:
            arch = m.group(1)
            continue
        m = _FUNC.match(line)
        if m:
            func = m.group(1)
            continue
        m = _RES.search(line)
        if m and func:
            out[(arch, func)] = dict(zip(_RESOURCE_FIELDS, (int(x) for x in m.groups())))
            func = None
    return out


def parse_sass(text):
    """``{(arch, function): {"count": n, "hist": Counter(opcode)}}`` of a ``cuobjdump -sass`` dump (opcode = first token after the predicate)."""
    out, arch, func = {}, "?", None
    for line in text.splitlines():
        m = _ARCH.match(line)
        if m:
            arch = m.group(1)
            continue
        m = _FUNC.match(line)
        if m:
            func = m.group(1)
            out[(arch, func)] = {"count": 0, "hist": Counter()}
            continue
        m = _INSN.match(line)
        if m and func:
            body = m.group(2)
            op = re.sub(r"^@!?U?P\w+\s+", "", body).split()[0] if body else ""
            entry = out[(arch, func)]
            entry["count"] += 1
            entry["hist"][op] += 1
    return out


def _selected(key, functions):
    return not functions or any(s in key[1] for s in functions)


def compare_resources(base, cand, functions=None):
    """``(differences, added)``: lines describing every difference / the candidate-only functions."""
    diffs, added = [], []
    for key in sorted(base):
        if not _selected(key, functions):
            continue
        if key not in cand:
            diffs.append("%s %s: missing in the candidate" % key)
            continue
        for field in _RESOURCE_FIELDS:
            if base[key][field] != cand[key][field]:
                diffs.append("%s %s: %s %d -> %d" % (key[0], key[1], field, base[key][field], cand[key][field]))
    for key in sorted(cand):
        if key not in base and _selected(key, functions):
            added.append("%s %s" % key)
    return diffs, added


def compare_sass(base, cand, functions=None):
    diffs = []
    for key in sorted(base):
        if not _selected(key, functions):
            continue
        if key not in cand:
            diffs.append("%s %s: SASS missing in the candidate" % key)
            continue
        if base[key]["count"] != cand[key]["count"]:
            diffs.append("%s %s: SASS instructions %d -> %d" % (key[0], key[1], base[key]["count"], cand[key]["count"]))
        hist_b, hist_c = base[key]["hist"], cand[key]["hist"]
        if hist_b != hist_c:
            delta = {op: hist_c[op] - hist_b[op] for op in set(hist_b) | set(hist_c) if hist_c[op] != hist_b[op]}
            top = sorted(delta.items(), key=lambda kv: (-abs(kv[1]), kv[0]))[:5]  # ties by opcode name: the same input always gives the same text
            diffs.append("%s %s: opcode histogram differs (%s)" % (key[0], key[1], ", ".join("%s %+d" % kv for kv in top)))
    return diffs


def _read_text(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-regs", required=True)
    ap.add_argument("--cand-regs", required=True)
    ap.add_argument("--base-sass", default=None)
    ap.add_argument("--cand-sass", default=None)
    ap.add_argument("--functions", nargs="*", default=None, help="only functions whose (mangled) name contains one of these substrings")
    a = ap.parse_args(argv)
    try:
        base_r, cand_r = parse_resource_usage(_read_text(a.base_regs)), parse_resource_usage(_read_text(a.cand_regs))
        base_s = parse_sass(_read_text(a.base_sass)) if a.base_sass else None
        cand_s = parse_sass(_read_text(a.cand_sass)) if a.cand_sass else None
    except OSError as e:
        print("cannot read the input: %s" % e, file=sys.stderr)
        return 2
    if not base_r or not cand_r:
        print("no resource-usage entries found in the input", file=sys.stderr)
        return 2
    diffs, added = compare_resources(base_r, cand_r, a.functions)
    if base_s is not None and cand_s is not None:
        diffs.extend(compare_sass(base_s, cand_s, a.functions))
    for d in diffs:
        print("DIFF: %s" % d)
    for ad in added:
        print("ADDED (candidate only): %s" % ad)
    print("functions compared: %d, differences: %d, added: %d" % (len([k for k in base_r if _selected(k, a.functions)]), len(diffs), len(added)))
    print("VERDICT: %s" % ("EQUAL" if not diffs else "DIFFERENT"))
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
