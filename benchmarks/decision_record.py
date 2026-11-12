#!/usr/bin/env python3
"""The decision record of the row engine: what the clock profile, the occupancy API, the Nsight Compute launch metrics and the compile dumps say before the first row-engine step is taken
(plain Python, no torch: runs on the login node).

A record is the JSON object that ``build_record`` derives from the raw evidence and nothing else: the per-phase shares of a row (the clock profile of both devices), the order of the
row-engine steps (the default one, a step may only be left out), the inputs of the value-split decision (the share of the value-independent prefix, the N * H ranges, the capacity) and the
budget table per architecture (registers, stack, spill and static shared memory of the base and of every prototype as compiled, the shared-memory tiles with their sizes, the CTAs per SM
and the L1 that remains). Every figure of the budget table has a provenance: ``compiler`` (a ptxas figure), ``source`` (a constant read from the source at a commit), ``api`` (the occupancy
API of a device), ``measured`` (an Nsight Compute launch metric), ``plan`` (a limit of the plan), ``plan_estimate`` (a size that no source defines yet) or ``arithmetic`` / ``model``
(derived). The checker (``record_problems``) reads the evidence again, derives the record again and rejects every difference: a record cannot say more than its evidence, and a number or a
label that was edited by hand is a difference. The section of the performance document is generated from the record (``render_section``) and must equal the generated text.

  decision_record.py build  --evidence MANIFEST --repo DIR --out RECORD [--recorded-on REV]
  decision_record.py render --record RECORD --document FILE [--number N]
  decision_record.py check  --record RECORD --evidence MANIFEST --repo DIR [--document FILE] [--head REV]

The evidence manifest (a JSON file next to the evidence, paths relative to it) names ``profile`` (the profile records of both devices), ``ncu`` (their Nsight Compute records),
``launch_config`` (the launch metrics of the retained reports, ``cfbi-launch-config-v1``), ``compile`` (``base`` and one compile dump per prototype, by the prototype's topic name),
``container`` (the production-toolchain dump of the profiled commit per device) and ``occupancy`` (the occupancy-API record per device). Exit status 0 pass, 1 a check fails, 2 unusable input.
"""
import argparse
import ast
import hashlib
import json
import math
import operator
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_decode as bd  # noqa: E402
import bench_records as br  # noqa: E402
import codegen_gate as cg  # noqa: E402
import record_checker as rc  # noqa: E402
import rows_profile_record as rpr  # noqa: E402

SCHEMA = "cfbi-decision-v1"
LAUNCH_SCHEMA = "cfbi-launch-config-v1"
SECTION_TITLE = "Row engine: decision record"
ARCHS = ("sm_90a", "sm_100a", "sm_103a")
DEVICE_ARCH = {"b200": "sm_100a", "b300": "sm_103a"}      # the architecture of each device that has hardware evidence
GROUPS = ("regular", "chunk_end", "all")
PROCESSES = 3                                              # profile records (processes) per device
ROWS_KERNEL, REFERENCE_KERNEL = "cfbi::rows_kernel", "cfbi::ref::rows_kernel"
THREADS = 256                                              # threads of a row-engine CTA (eight warps)
GATE_REGISTERS = 128                                       # the register ceiling of the resource gate
GATE_CTAS = 2                                              # CTAs per SM that the resource gate requires
UNIFIED_BYTES = 262144                                     # L1 + shared memory of an SM in the plan's model: 228 KiB of shared memory and 28 KiB of L1
STATIC_CAP_BYTES = 49152                                   # static shared memory per CTA: 48 KiB
A_FRAGMENTS_PLAN_BYTES = 6144                              # "about 6 KiB" of the plan: a size that no source defines yet
DRAFT = {"nh_up_to": 74, "gain_percent": [10, 25]}         # the draft's estimate for the value split: "10 to 25 % of the row-engine time for N * H up to about 74", never profiled
PLAN_DEVICE = {"registers_per_sm": 65536, "shared_per_sm": 233472, "threads_per_sm": 2048, "reserve_per_cta": 1024}      # the plan's limits for the architecture without hardware evidence
PREFIX_PHASES, CERTAIN_PHASES = ("prep", "akk_aqk"), ("akk_aqk",)

# the candidates of the budget table: id, the compile dump that holds it, the instantiation, the step (or the future work) that it belongs to, what it is
CANDIDATES = (
    ("base", "base", ROWS_KERNEL, None, "the optimised row engine of the base tree"),
    ("trims", "row-critical-path-re-pipelining", ROWS_KERNEL, "trims", "the trims prototype as committed"),
    ("e_tile_with_paired_m_loads", "coalesced-state-access-path", ROWS_KERNEL, "e_tile", "the staged E tile prototype as committed: it also carries the paired-column loads of the M tiles, which are not part of the tile step"),
    ("kd_tile", "chunk-end-row-offload", ROWS_KERNEL, "kd_tile", "the chunk-end KD tile prototype as committed"),
    ("single_launch_off", "launch-chain-collapse", ROWS_KERNEL + "<Lb0E>", "single_launch", "the single-launch prototype with the decode path compiled out"),
    ("single_launch_on", "launch-chain-collapse", ROWS_KERNEL + "<Lb1E>", "single_launch", "the single-launch prototype with the decode path compiled in"),
    ("operand_swap_off", "operand-swapped-mma-tiles", ROWS_KERNEL + "<Lb0E>", "operand_swap", "the operand-swap prototype with the swap compiled out"),
    ("operand_swap_on", "operand-swapped-mma-tiles", ROWS_KERNEL + "<Lb1E>", "operand_swap", "the operand-swap prototype with the swap compiled in"),
    ("value_split_128", "value-split-decode-ctas", ROWS_KERNEL + "<Li128E>", "value_split", "the value-split prototype with one slice of 128 value columns"),
    ("value_split_64", "value-split-decode-ctas", ROWS_KERNEL + "<Li64E>", "value_split", "the value-split prototype with slices of 64 value columns"),
    ("value_split_32", "value-split-decode-ctas", ROWS_KERNEL + "<Li32E>", "value_split", "the value-split prototype with slices of 32 value columns"),
    ("value_split_16", "value-split-decode-ctas", ROWS_KERNEL + "<Li16E>", "value_split", "the value-split prototype with slices of 16 value columns"),
)
PROTOTYPE_DUMPS = tuple(sorted(set(c[1] for c in CANDIDATES if c[1] != "base")))
# the phases that each row-engine step changes (the change sets of the steps, by the names of the clock profile; the prologue is the kernel's, not a row phase)
STEP_TARGETS = {"trims": ("prep", "akk_aqk", "p1_p2", "chunk_end_sync"), "e_tile": ("p1_p2", "chunk_end_sync"), "a_fragments": ("akk_aqk", "m_tiles", "p1_p2")}
STEP_PROTOTYPES = {"trims": "row-critical-path-re-pipelining", "e_tile": "coalesced-state-access-path", "a_fragments": None}
# the phases in which each tile holds data that a later phase still reads, by the plan's lifetimes: RowSmem in every phase, the staged E tile from its staging in the first row to the
# chunk-end block that rewrites it, the KD tile in the chunk-end block only, the A fragments in the phases that consume them
TILE_LIVE_PHASES = {"row_smem": tuple(rpr.PHASES), "e_tile": tuple(rpr.PHASES), "kd_tile": ("chunk_end_sync",), "a_fragments": ("akk_aqk", "m_tiles", "p1_p2")}


def tile_lifetimes():
    """The pairs of tiles with disjoint live phases (they may alias) and with overlapping ones (they must not): a design inference from the phase structure, not a tested layout."""
    names = sorted(TILE_LIVE_PHASES)
    pairs = [[a, b] for i, a in enumerate(names) for b in names[i + 1:]]

    def overlap(pair):
        return bool(set(TILE_LIVE_PHASES[pair[0]]) & set(TILE_LIVE_PHASES[pair[1]]))
    return {"label": "plan", "tested": False, "live_phases": dict((t, list(TILE_LIVE_PHASES[t])) for t in names),
            "may_alias": [p for p in pairs if not overlap(p)], "must_not_alias": [p for p in pairs if overlap(p)]}


STATEMENTS = {
    "order": (
        "The default order is kept: trims, then the staged E tile, then the once-per-row A fragments; no step is left out.",
        "Every step changes phases that have a measured share of the row, and no measurement shows a step to be futile.",
        "The A fragments rely on the registers that the staged E tile frees: a step may be left out but never moved before an earlier one.",
        "The prototypes compiled as committed break the resource gate on Blackwell: this is an input to the repair of each step and not a reason to skip it, and no prototype has been run on a device.",
        "Nothing in this record is a timing result of a step.",
    ),
    "value_split": (
        "The share of the value-independent prefix cannot be separated exactly with the stamps of the profile: the first phase includes loads that depend on the value dimension and later phases mix independent and dependent work.",
        "Two figures are given instead: the first two phases together as a proxy of the prefix, and the key-side product phase alone as the part that is certainly independent of the value dimension.",
        "The draft's figure is an estimate of a gain that was never profiled: this record neither confirms nor excludes it.",
        "The per-phase profile does not cover every matrix shape that lies inside the draft's range.",
        "The capacity bound depends on the model: one CTA per SM (the policy of the prototype) or every resident slot.",
    ),
    "budget": (
        "Registers, stack, spill and static shared memory of the prototypes are compile-only figures of the prototypes as committed, not of the steps as they will be implemented.",
        "The dynamic shared memory of the staged tile and the size of the KD tile are read from the source constants; the size of the A-fragment tile is the plan's estimate and no compiled figure.",
        "CTAs per SM are API figures for the base kernel on the two devices and arithmetic everywhere else; the arithmetic reproduces the API figures of the base and of the reference kernel.",
        "The shared-memory carve-out is measured for the base kernel only and is never applied to another configuration: the L1 of the other configurations is an upper bound of a continuous model.",
        "The static shared memory of the row engine is the compiler's figure, which is larger than the struct of the source by the padding of the allocation; the budgets use the compiler's figure.",
        "The prototypes were compiled with the development toolchain only; the production toolchain has compiled the base and governs where the two differ.",
        "The lifetimes of the tiles are the plan's, read from the phase structure of the sources: disjoint lifetimes are a design inference, no alias has been built or checked with the sanitizers, and an alias belongs to the step that needs it.",
        "The Hopper architecture has no hardware evidence: its figures are arithmetic on the limits of the plan.",
    ),
}
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])")
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv}
INSTANCE_FIGURES = ("registers", "static_smem", "dynamic_smem", "threads", "ctas_per_sm")      # what an occupancy-API instance reports
LOOP_DIR = rc.UNTRACKED_OK.split()[-1].rstrip("/")                                        # the directory of the loop (the checker's untracked evidence directory)
SECTION_NUMBER = 8                                                                           # the number of the section in the performance document


# ---- reading the evidence ---------------------------------------------------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_evidence(manifest_path):
    """The evidence that a manifest names: the loaded records and, in ``inputs``, the name and digest of every file (what the record says about its inputs)."""
    manifest = rc.load_json(manifest_path)
    root = os.path.dirname(os.path.abspath(manifest_path))

    def path(x):
        return x if os.path.isabs(x) else os.path.join(root, x)

    def entry(p):
        return {"name": os.path.basename(p), "sha256": sha256_file(p)}

    ev = {"inputs": {}}
    ev["profiles"] = [br.load_record(path(p)) for p in manifest["profile"]]
    ev["ncu"] = [rc.load_json(path(p)) for p in manifest["ncu"]]
    ev["launch"] = rc.load_json(path(manifest["launch_config"]))
    ev["compile"] = dict((k, rc.load_json(path(p))) for k, p in sorted(manifest["compile"].items()))
    ev["container"] = dict((k, rc.load_json(path(p))) for k, p in sorted(manifest["container"].items()))
    ev["occupancy"] = dict((k, rc.load_json(path(p))) for k, p in sorted(manifest["occupancy"].items()))
    ev["inputs"] = {"profile": sorted((entry(path(p)) for p in manifest["profile"]), key=lambda e: e["name"]),
                    "ncu": sorted((entry(path(p)) for p in manifest["ncu"]), key=lambda e: e["name"]),
                    "launch_config": entry(path(manifest["launch_config"])),
                    "compile": dict((k, entry(path(p))) for k, p in sorted(manifest["compile"].items())),
                    "container": dict((k, entry(path(p))) for k, p in sorted(manifest["container"].items())),
                    "occupancy": dict((k, entry(path(p))) for k, p in sorted(manifest["occupancy"].items()))}
    return ev


def device_of(rec):
    gpu = str(((rec.get("meta") or {}).get("gpu") if isinstance(rec.get("meta"), dict) else rec.get("gpu")) or "")
    found = [d for d in rc.DEVICES if d.upper() in gpu.upper()]
    return found[0] if len(found) == 1 else None


def kernel(dump, arch, name):
    return ((dump.get("kernels") or {}).get(arch) or {}).get(name)


def figures(k):
    return {"registers": k["reg"], "stack": k["stack"], "spill_loads": k["spill_loads"], "spill_stores": k["spill_stores"]}


# ---- the arithmetic of the occupancy and of the shared memory -----------------------------------------------------------------------------------
def ctas_per_sm(registers, threads, static, dynamic, dev):
    """CTAs per SM that the register file, the shared memory and the thread slots allow (the registers are allocated per warp in units of 256)."""
    warps = -(-threads // 32)
    per_warp = -(-registers * 32 // 256) * 256
    by_registers = dev["registers_per_sm"] // (per_warp * warps)
    by_shared = dev["shared_per_sm"] // (static + dynamic + dev["reserve_per_cta"])
    return min(by_registers, by_shared, dev["threads_per_sm"] // threads)


def occupancy_device(rec):
    d = (rec.get("occupancy") or {}).get("device") or {}
    return {"registers_per_sm": d.get("regs_per_sm"), "shared_per_sm": d.get("smem_per_sm"), "threads_per_sm": d.get("max_threads_per_sm"), "reserve_per_cta": d.get("smem_reserved_per_block"),
            "sm_count": d.get("sm_count")}


def label_commit(rec, dev):
    """The commit that an occupancy record names in its label (``<device>-<commit>``), or None."""
    m = re.match(r"^%s-([0-9a-f]{7,40})$" % dev, str(rec.get("label"))) if isinstance(rec, dict) else None
    return m.group(1) if m else None


occupancy_commit = label_commit                                    # for a record that has passed ``occupancy_problems``


def instance(rec, name):
    found = [i for i in (rec.get("occupancy") or {}).get("instances") or [] if isinstance(i, dict) and i.get("name") == name]
    return found[0] if len(found) == 1 else None


# ---- the names of the optimised row engine: read from the record and from the dump, not a constant ------------------------------------------------------------
# A commit from before the template has one row engine, ``cfbi::rows_kernel``. After it the occupancy API names the instantiations ``cfbi::rows_kernel<false>`` (behind the plan kernel: the base's
# row engine) and ``cfbi::rows_kernel<true>`` (the single-launch decode), and the machine-code dumps spell the template argument ``<Lb0E>`` / ``<Lb1E>``.
ROW_ENGINE_NAME = re.compile(r"^cfbi::rows_kernel(<false>|<true>)?$")
DUMP_SPELLING = {"<false>": "<Lb0E>", "<true>": "<Lb1E>"}
ROW_ENGINE_SHAPES = ([ROWS_KERNEL], [ROWS_KERNEL + "<false>", ROWS_KERNEL + "<true>"])      # what a record may name: the plain engine, or both instantiations (each once)


def row_names(rec):
    """The names of the optimised row engine in an occupancy record, in the record's order (the frozen reference is not one of them)."""
    instances = ((rec.get("occupancy") or {}).get("instances") or []) if isinstance(rec, dict) else []
    return [i["name"] for i in instances if isinstance(i, dict) and isinstance(i.get("name"), str) and ROW_ENGINE_NAME.match(i["name"])]


def dump_name(name):
    """The key of the machine-code dump for the name that the occupancy API gives an instance (``cfbi::rows_kernel<true>`` is ``cfbi::rows_kernel<Lb1E>``; a plain name stays)."""
    head, sep, arg = name.partition("<")
    return head + DUMP_SPELLING.get("<" + arg, "<" + arg) if sep else name


def base_row_name(rec):
    """The instance of an occupancy record that is the base's optimised row engine: the plain name, else the instantiation behind the plan kernel (``<false>``); None when it has neither."""
    names = row_names(rec)
    return next((n for n in (ROWS_KERNEL, ROWS_KERNEL + "<false>") if n in names), None)


def dump_row_keys(dump, arch):
    """The keys of the optimised row engine in a dump for one architecture: the plain key or its instantiations."""
    kernels = ((dump.get("kernels") or {}).get(arch) or {}) if isinstance(dump, dict) else {}
    return sorted(k for k in kernels if cg.family(k) == ROWS_KERNEL)


# ---- the source facts: constants read from the sources at a commit -----------------------------------------------------------------------------
def source_text(repo, commit, path):
    code, out, err = rc.git(repo, "show", "%s:%s" % (commit, path))
    if code != 0:
        raise ValueError("cannot read %s at %s: %s" % (path, commit, err.strip()[:120]))
    return out


def eval_constant(expr, resolve):
    """An integer constant expression of the sources (+ - * //, parentheses, names that ``resolve`` knows, ``sizeof`` of float and float4, ``static_cast``)."""
    expr = re.sub(r"static_cast<[^>]+>\(", "(", expr).replace("sizeof(float4)", "16").replace("sizeof(float)", "4").strip()

    def literal(node):
        if sys.version_info >= (3, 8):
            return node.value if isinstance(node, ast.Constant) and type(node.value) is int else None
        return node.n if isinstance(node, ast.Num) and type(node.n) is int else None

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if literal(node) is not None:
            return literal(node)
        if isinstance(node, ast.Name):
            return resolve(node.id)
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            a, b = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.FloorDiv) and b == 0:
                raise ValueError("division by zero in the constant expression %r" % expr)
            return OPERATORS[type(node.op)](a, b)
        raise ValueError("not an integer constant expression: %r" % expr)
    try:
        return walk(ast.parse(expr, mode="eval"))
    except SyntaxError:
        raise ValueError("not an integer constant expression: %r" % expr)


def constants_of(*texts):
    """``resolve(name)``: the value of a ``constexpr int`` of the given source texts."""
    defs, env = {}, {}
    for t in texts:
        defs.update(dict(re.findall(r"constexpr\s+(?:int|size_t)\s+(\w+)\s*=\s*([^;]+);", t)))

    def resolve(name, stack=()):
        if name not in env:
            if name in stack:
                raise ValueError("constant %s is defined in terms of itself" % name)
            if name not in defs:
                raise ValueError("constant %s is not defined in the sources" % name)
            env[name] = eval_constant(defs[name], lambda n: resolve(n, stack + (name,)))
        return env[name]
    return resolve


def row_smem_floats(rows_text, resolve):
    """The floats of ``struct RowSmem``: the sum over its fields of the product of their dimensions (every field must be a float)."""
    m = re.search(r"struct\s+RowSmem\s*\{(.*?)\n\};", rows_text, re.S)
    if not m:
        raise ValueError("struct RowSmem is not in the source")
    total = 0
    for line in m.group(1).splitlines():
        line = line.split("//")[0].strip()
        if not line:
            continue
        if not line.startswith("float "):
            raise ValueError("RowSmem has a field that is not a float: %r" % line)
        for decl in line[len("float "):].rstrip(";").split(","):
            d = re.match(r"\s*(\w+)((?:\[[^\]]+\])*)\s*$", decl)
            if not d:
                raise ValueError("cannot read the field %r of RowSmem" % decl)
            n = 1
            for dim in re.findall(r"\[([^\]]+)\]", d.group(2)):
                n *= eval_constant(dim, resolve)
            total += n
    return total


def source_facts(repo, base_commit, e_commit, kd_commit):
    """The sizes of the tiles as the sources define them: RowSmem at the base commit, the dynamic bytes of the staged E tile and the bytes of the KD tile at the prototypes' commits."""
    primitives, rows = "csrc/cfbi/cfbi_primitives.cuh", "csrc/cfbi/cfbi_rows.cuh"
    base_resolve = constants_of(source_text(repo, base_commit, primitives))
    floats = row_smem_floats(source_text(repo, base_commit, rows), base_resolve)
    e_resolve = constants_of(source_text(repo, e_commit, primitives), source_text(repo, e_commit, rows))
    kd_resolve = constants_of(source_text(repo, kd_commit, primitives), source_text(repo, kd_commit, rows))
    return [
        {"name": "row_smem_bytes", "value": 4 * floats, "floats": floats, "commit": base_commit, "path": rows, "defines": ["RowSmem"]},
        {"name": "e_tile_dynamic_bytes", "value": e_resolve("ROWS_DYN_SMEM_BYTES"), "commit": e_commit, "path": rows, "defines": ["ROWS_E_STRIDE", "ROWS_DYN_SMEM_BYTES"]},
        {"name": "kd_tile_bytes", "value": 16 * kd_resolve("ROWS_KD_FRAGS"), "commit": kd_commit, "path": rows, "defines": ["ROWS_KD_FRAGS"]},
    ]


# ---- the checks of the evidence ---------------------------------------------------------------------------------------------------------------
LAUNCH_SOURCE, PRIMITIVES_SOURCE = "csrc/cfbi/cfbi_launch.cu", "csrc/cfbi/cfbi_primitives.cuh"
POLICY_SOURCE = "csrc/cfbi/cfbi_policy.h"     # the CUDA-free header: where the launcher's shared-memory constant is defined once the header exists


def launch_constants(repo, commit):
    """The threads and the dynamic shared memory of the row-engine launch at a commit, as the sources define them: the thread count of the kernels and the launcher's constant that its
    launch sites and its occupancy hook both read (a number, or the constant of the CUDA-free header that the launcher takes it from; the header is absent at the earlier commits)."""
    texts = [source_text(repo, commit, PRIMITIVES_SOURCE), source_text(repo, commit, LAUNCH_SOURCE)]
    try:
        texts.append(source_text(repo, commit, POLICY_SOURCE))
    except ValueError:
        pass                                                                    # no header at this commit: a constant of the launcher that names one is then not defined (an error below)
    resolve = constants_of(*texts)
    return {"threads": resolve("NTHREADS"), "dynamic_smem": resolve("ROWS_DYNAMIC_SMEM")}


def launch_problems(launch, ncu):
    """The launch metrics of the retained reports: every report of the Nsight Compute records exactly once (by the digest of the report), of its device, with the launch that the record
    holds, and one shared-memory carve-out per device."""
    where = "launch-config record"
    if not isinstance(launch, dict) or launch.get("schema") != LAUNCH_SCHEMA or not isinstance(launch.get("reports"), list):
        return ["%s: not a %s record" % (where, LAUNCH_SCHEMA)]
    out = []
    by_digest = {}
    for rec in ncu:
        digest = ((rec.get("meta") or {}).get("report") or {}).get("sha256")
        by_digest.setdefault(digest, []).append(rec)
    out.extend("%s: %d Nsight Compute records name the report %s: every report is named once" % (where, len(recs), digest) for digest, recs in sorted(by_digest.items(), key=str) if len(recs) > 1 or digest is None)
    seen = {}
    for r in launch["reports"]:
        if not isinstance(r, dict):
            out.append("%s: an entry of the reports is not an object" % where)
            continue
        digest = r.get("report_sha256")
        match = by_digest.get(digest) or []
        if not match:
            out.append("%s: the report %r is not one of the reports of the Nsight Compute records" % (where, digest))
            continue
        if len(match) > 1 or digest is None:
            continue                                                   # named above
        rec = match[0]
        lc = r.get("launch") if isinstance(r.get("launch"), dict) else {}
        seen[digest] = seen.get(digest, 0) + 1
        if device_of(rec) != r.get("device"):
            out.append("%s: the report %s is of %s, the entry says %r" % (where, digest[:12], device_of(rec), r.get("device")))
        ml = (rec.get("meta") or {}).get("launch") or {}
        for field, mfield in (("block_size", "block_size"), ("grid_size", "grid_size"), ("registers_per_thread", "registers_per_thread")):
            if lc.get(field) != ml.get(mfield):
                out.append("%s: the report %s says %s = %r, the Nsight Compute record %r" % (where, digest[:12], field, lc.get(field), ml.get(mfield)))
        if not (rc.count(lc.get("shared_mem_per_block_static"), 1) and rc.count(lc.get("shared_mem_per_block_driver"), 1)
                and lc.get("shared_mem_per_block_static") + lc.get("shared_mem_per_block_driver") == ml.get("shared_mem_per_block")):
            out.append("%s: the report %s: static + driver shared memory is not the record's shared memory per block" % (where, digest[:12]))
        if not rc.count(lc.get("shared_mem_config_size"), 1):
            out.append("%s: the report %s has no shared-memory carve-out (shared_mem_config_size)" % (where, digest[:12]))
    for digest in by_digest:
        if seen.get(digest) != 1:
            out.append("%s: the report %s of the Nsight Compute records appears %d time(s) in the launch metrics, once is required" % (where, str(digest)[:12], seen.get(digest, 0)))
    for dev in rc.DEVICES:
        values = sorted(set(r["launch"].get("shared_mem_config_size") for r in launch["reports"] if isinstance(r, dict) and r.get("device") == dev and isinstance(r.get("launch"), dict)), key=str)
        if len(values) != 1:
            out.append("%s: %s has %d carve-out values %s, one is required" % (where, dev, len(values), values))
    return out


def occupancy_problems(rec, dev, container, launch=None, commit=None):
    """The occupancy-API record of a device, bound to what it claims: the device, the commit (the label; ``commit``: the commit that the evidence is of), the instantiations (the names are
    read from the record, the keys of the dump from them, and every row-engine kernel of the dump needs an instance), the registers and the static shared memory against the
    production-toolchain dump, the threads and the dynamic shared memory against the launch in the sources at the record's commit (``launch``: ``launch_constants``), at least two CTAs per
    SM and at most 128 registers for every instantiation of the row engine, and the arithmetic of ``ctas_per_sm`` reproducing the API for every instance (the CTA count alone does not say
    which configuration it is the count of)."""
    where = "occupancy record %s" % (rec.get("_path", dev) if isinstance(rec, dict) else dev)
    if not isinstance(rec, dict):
        return ["%s: not a record" % where]
    out = []
    if device_of(rec) != dev:
        out.append("%s: the GPU %r is not %s" % (where, rec.get("gpu"), dev))
    dv = occupancy_device(rec)
    if not all(rc.count(dv[k], 1) for k in ("registers_per_sm", "shared_per_sm", "threads_per_sm", "reserve_per_cta", "sm_count")) or rec.get("sm_count") != dv["sm_count"]:
        out.append("%s: the device limits (registers, shared memory, threads, reserve, SM count) must be positive integers and the SM count the record's own" % where)
        return out
    if label_commit(rec, dev) is None:
        out.append("%s: the label %r does not name the device and the commit (%s-<commit>)" % (where, rec.get("label"), dev))
    elif commit is not None and not rc.commit_matches(label_commit(rec, dev), commit):
        out.append("%s: the record is of commit %s, the evidence it belongs to is of commit %s" % (where, label_commit(rec, dev), commit))
    names = row_names(rec)
    if sorted(names) not in ROW_ENGINE_SHAPES:
        out.append("%s: the optimised row engine is named %s: either the plain %s or both of its instantiations <false> and <true>, each once (an instantiation without an occupancy figure is unchecked)"
                   % (where, names or "nowhere", ROWS_KERNEL))
    covered = set(dump_name(n) for n in names)
    out.extend("%s: the production dump has the row-engine kernel %s, the record has no instance for it" % (where, key) for key in dump_row_keys(container, DEVICE_ARCH[dev]) if key not in covered)
    for name in (names or [ROWS_KERNEL]) + [REFERENCE_KERNEL]:
        inst = instance(rec, name)
        want = kernel(container, DEVICE_ARCH[dev], dump_name(name))
        if inst is None or want is None:
            out.append("%s: the instantiation %s is not in the record or not in the production dump" % (where, name))
            continue
        if not (all(rc.count(inst.get(f)) for f in INSTANCE_FIGURES) and inst["registers"] > 0 and inst["threads"] > 0):
            out.append("%s: %s: every figure must be a non-negative integer, the registers and the threads positive" % (where, name))
            continue
        if inst["registers"] != want["reg"] or inst["static_smem"] + dv["reserve_per_cta"] != want["shared"]:
            out.append("%s: %s: the API says %d registers and %d B static shared memory, the production dump %d registers and %d B including the reserve"
                       % (where, name, inst["registers"], inst["static_smem"], want["reg"], want["shared"]))
        if inst["threads"] != THREADS:
            out.append("%s: %s: %d threads per CTA, %d are required" % (where, name, inst["threads"], THREADS))
        if launch is not None and (inst["threads"], inst["dynamic_smem"]) != (launch["threads"], launch["dynamic_smem"]):
            out.append("%s: %s: the API says %d threads and %d B dynamic shared memory, the launch in the sources at the record's commit uses %d threads and %d B"
                       % (where, name, inst["threads"], inst["dynamic_smem"], launch["threads"], launch["dynamic_smem"]))
        arithmetic = ctas_per_sm(inst["registers"], inst["threads"], inst["static_smem"], inst["dynamic_smem"], dv)
        if inst["ctas_per_sm"] != arithmetic:
            out.append("%s: %s: the API says %d CTAs per SM, the arithmetic of the registers, the shared memory and the threads %d" % (where, name, inst["ctas_per_sm"], arithmetic))
        if name != REFERENCE_KERNEL and (inst["ctas_per_sm"] < GATE_CTAS or inst["registers"] > GATE_REGISTERS):
            out.append("%s: the row engine %s has %d CTAs per SM with %d registers: at least %d CTAs and at most %d registers are required" % (where, name, inst["ctas_per_sm"], inst["registers"], GATE_CTAS, GATE_REGISTERS))
    return out


def retained_launch_problems(rec, dev, launch):
    """The retained Nsight Compute launches of a device against the occupancy instance of the optimised row engine, figure by figure (static, driver and dynamic shared memory, threads,
    registers: a sum alone would let a dynamic figure hide behind a static one)."""
    inst = instance(rec, base_row_name(rec)) if isinstance(rec, dict) else None            # the profiled launch is the base's row engine
    dv = occupancy_device(rec) if isinstance(rec, dict) else {}
    if inst is None or not isinstance(launch, dict) or not all(rc.count(inst.get(f)) for f in INSTANCE_FIGURES) or not rc.count(dv.get("reserve_per_cta"), 1):
        return []                                                  # named by the other checks
    want = {"shared_mem_per_block_static": inst["static_smem"], "shared_mem_per_block_driver": dv["reserve_per_cta"], "shared_mem_per_block_dynamic": inst["dynamic_smem"],
            "block_size": inst["threads"], "registers_per_thread": inst["registers"]}
    out = []
    for r in launch.get("reports", []):
        if not isinstance(r, dict) or r.get("device") != dev or not isinstance(r.get("launch"), dict):
            continue
        out.extend("launch-config record: the report %s says %s = %r, the occupancy record of %s %r" % (str(r.get("report_sha256"))[:12], f, r["launch"].get(f), dev, v)
                   for f, v in sorted(want.items()) if r["launch"].get(f) != v)
    return out


def dump_problems(dump, name, archs, where):
    """A dump is evidence only if the codegen gate's own validator accepts it (a complete toolchain description, complete figures for every kernel of every architecture, the launch bounds
    of the row-engine family); then the architectures it covers and the commit it names."""
    try:
        cg.check_dump(dump, name)
    except cg.Unusable as e:
        return ["%s: %s" % (where, " ".join(str(e).split()))]
    out = []
    if sorted(dump["kernels"]) != sorted(archs):
        out.append("%s: the dump of %s covers %s, %s are required" % (where, name, sorted(dump["kernels"]), sorted(archs)))
    commit = ((dump.get("meta") or {}).get("tree") or {}).get("commit")
    if not rc.hexcommit(commit):
        out.append("%s: the dump of %s names no commit" % (where, name))
    return out


def compile_problems(compiled, container, profile_commit):
    where = "compile evidence"
    out = []
    if sorted(compiled) != sorted(("base",) + PROTOTYPE_DUMPS):
        out.append("%s: the dumps are %s, the base and the prototypes %s are required" % (where, sorted(compiled), list(PROTOTYPE_DUMPS)))
        return out
    for name, dump in sorted(compiled.items()):
        out.extend(dump_problems(dump, name, ARCHS, where))
    if out:
        return out
    for name in PROTOTYPE_DUMPS:                                   # a resource figure is only comparable with the base's under the same compiler: every field, the label is for display
        try:
            cg.require_same_toolchain(compiled["base"], compiled[name], "the dumps of the base and of %s" % name)
        except cg.Unusable as e:
            out.append("%s: %s" % (where, " ".join(str(e).split())))
    if out:
        return out
    for cid, dump, inst, step, what in CANDIDATES:
        for arch in ARCHS:
            if kernel(compiled[dump], arch, inst) is None:
                out.append("%s: %s has no instantiation %s for %s" % (where, dump, inst, arch))
    for arch in ARCHS:
        if kernel(compiled["base"], arch, REFERENCE_KERNEL) is None:
            out.append("%s: the base dump has no reference row engine for %s" % (where, arch))
    for dev in rc.DEVICES:
        dump = container.get(dev)
        out.extend(dump_problems(dump, "the production toolchain of %s" % dev, (DEVICE_ARCH[dev],), where))
        if isinstance(dump, dict) and isinstance(dump.get("meta"), dict):
            if "container" not in str(((dump["meta"].get("toolchain") or {}).get("label"))):
                out.append("%s: the production dump of %s was not made with the container toolchain" % (where, dev))
            if not rc.commit_matches(str(((dump["meta"].get("tree") or {}).get("commit"))), str(profile_commit)):
                out.append("%s: the production dump of %s is of commit %s, the profile is of %s" % (where, dev, ((dump["meta"].get("tree") or {}).get("commit")), profile_commit))
            if not dump_row_keys(dump, DEVICE_ARCH[dev]):
                out.append("%s: the production dump of %s has no %s" % (where, dev, ROWS_KERNEL))
            if kernel(dump, DEVICE_ARCH[dev], REFERENCE_KERNEL) is None:
                out.append("%s: the production dump of %s has no %s" % (where, dev, REFERENCE_KERNEL))
    return out


def evidence_problems(ev, repo=None):
    """Why the evidence cannot be the basis of a record (the profile set of both devices with three processes each and one commit, the Nsight Compute set of that commit, the launch metrics,
    the compile dumps of the base and of the prototypes with one toolchain, the production dumps and the occupancy records bound to their devices and, with ``repo``, to the launch in
    the sources at their commits)."""
    out = []
    profiles = ev.get("profiles") or []
    out.extend(rc.profile_set_problems(profiles))
    for dev in rc.DEVICES:
        n = len([r for r in profiles if device_of(r) == dev])
        if n != PROCESSES:
            out.append("%d profile record(s) of %s, %d (one per process) are required" % (n, dev, PROCESSES))
    commits = sorted(set(str((r.get("meta") or {}).get("commit")) for r in profiles))
    if len(commits) != 1 or not rc.hexcommit(commits[0]):
        return out + ["the profile records must name one commit, they name %s" % commits]
    out.extend(rc.ncu_set_problems(ev.get("ncu") or [], expect_commit=commits[0]))
    out.extend(launch_problems(ev.get("launch"), ev.get("ncu") or []))
    out.extend(compile_problems(ev.get("compile") or {}, ev.get("container") or {}, commits[0]))
    for dev in rc.DEVICES:
        rec = (ev.get("occupancy") or {}).get(dev)
        launch, commit = None, label_commit(rec, dev)
        if repo is not None and commit is not None:
            try:
                launch = launch_constants(repo, commit)
            except ValueError as e:
                out.append("occupancy record of %s: the launch cannot be read from the sources at commit %s: %s" % (dev, commit, e))
        out.extend(occupancy_problems(rec, dev, (ev.get("container") or {}).get(dev) or {}, launch))
        out.extend(retained_launch_problems(rec, dev, ev.get("launch")))
    configs = set((inst["threads"], inst["dynamic_smem"]) for r in (ev.get("occupancy") or {}).values() for inst in (instance(r, n) for n in row_names(r)) if inst and inst.get("threads") is not None)
    if len(configs) > 1:
        out.append("the occupancy records of the two devices describe different launches of the row engine (threads, dynamic shared memory): %s" % sorted(configs))
    return out


# ---- deriving the record ------------------------------------------------------------------------------------------------------------------------
def _mean(xs):
    return sum(xs) / float(len(xs))


def _r(x, n=3):
    return round(float(x), n)


def phase_profile(profiles):
    """Per device, shape and group of rows the pooled cycles of the three processes (the mean of their per-process means), the shares of the phases of the pooled row, the range of the
    shares over the processes and the spread of the row duration."""
    cells = []
    for dev in rc.DEVICES:
        recs = [r for r in profiles if device_of(r) == dev]
        for n in rc.PROFILE_SHAPES:
            for group in GROUPS:
                blocks = [next(c for c in r["cells"] if c["N"] == n)["rows"][group] for r in recs]
                pooled = dict((f, _mean([b[f] for b in blocks])) for f in ("row_total", "prologue", "epilogue", "kernel_total"))
                phase = dict((p, _mean([b["phases"][p] for b in blocks])) for p in rpr.PHASES)
                totals = sorted(b["row_total"] for b in blocks)
                per_process = dict((p, [100.0 * b["phases"][p] / b["row_total"] for b in blocks]) for p in rpr.PHASES)
                cells.append({
                    "device": dev, "n": n, "group": group, "rows": blocks[0]["rows"],
                    "row_cycles": _r(pooled["row_total"]), "row_spread_percent": _r(100.0 * (totals[-1] - totals[0]) / totals[len(totals) // 2]),
                    "kernel_cycles": _r(pooled["kernel_total"]),
                    "phase_cycles": dict((p, _r(phase[p])) for p in rpr.PHASES),
                    "share_percent": dict((p, _r(100.0 * phase[p] / pooled["row_total"])) for p in rpr.PHASES),
                    "share_min_percent": dict((p, _r(min(v))) for p, v in per_process.items()),
                    "share_max_percent": dict((p, _r(max(v))) for p, v in per_process.items()),
                    "prologue_percent_of_kernel": _r(100.0 * pooled["prologue"] / pooled["kernel_total"]),
                    "epilogue_percent_of_kernel": _r(100.0 * pooled["epilogue"] / pooled["kernel_total"]),
                    "prefix_proxy_percent": _r(100.0 * sum(phase[p] for p in PREFIX_PHASES) / pooled["row_total"]),
                    "certain_part_percent": _r(100.0 * sum(phase[p] for p in CERTAIN_PHASES) / pooled["row_total"]),
                })
    return cells


def value_split_inputs(occupancy, heads):
    matrix_nh = sorted(set(N * H for N, H in bd.MATRIX_D))
    profile_nh = sorted(n * heads for n in rc.PROFILE_SHAPES)
    within = [x for x in matrix_nh if x <= DRAFT["nh_up_to"]]
    capacity = []
    for dev in rc.DEVICES:
        d = occupancy_device(occupancy[dev])
        ctas = instance(occupancy[dev], base_row_name(occupancy[dev]))["ctas_per_sm"]
        slots = d["sm_count"] * ctas
        capacity.append({"device": dev, "sm_count": d["sm_count"], "ctas_per_sm": ctas, "resident_slots": slots,
                         "max_nh_by_slices": [{"slices": s, "one_cta_per_sm": d["sm_count"] // s, "resident_slots": slots // s} for s in (2, 4, 8)]})
    return {
        "prefix_proxy_phases": list(PREFIX_PHASES), "certain_part_phases": list(CERTAIN_PHASES), "strict_v_independent_share": None,
        "matrix_nh": matrix_nh, "profile_nh": profile_nh, "draft": dict(DRAFT, label="draft_estimate_unprofiled"),
        "matrix_nh_within_draft": within, "profile_nh_within_draft": [x for x in profile_nh if x <= DRAFT["nh_up_to"]],
        "profile_covers_matrix_within_draft": all(x in profile_nh for x in within),
        "capacity": capacity,
        "statements": list(STATEMENTS["value_split"]),
    }


def excess(base, cand):
    """The figures of a candidate that are above the base's of the same architecture (the resource gate), or above the register ceiling."""
    out = [f for f in ("stack", "spill_loads", "spill_stores") if cand[f] > base[f]]
    return (["registers"] if cand["registers"] > GATE_REGISTERS else []) + out


def budget_table(ev, facts, launches):
    compiled, container, occupancy = ev["compile"], ev["container"], ev["occupancy"]
    base_dynamic = launches[rc.DEVICES[0]]["dynamic_smem"]          # the launch of the row engine is one configuration (the same on both devices, see evidence_problems)
    reserve = {}
    devices = {}
    for arch in ARCHS:
        dev = next((d for d in rc.DEVICES if DEVICE_ARCH[d] == arch), None)
        if dev is None:
            devices[arch] = dict(PLAN_DEVICE, label="plan", device=None, sm_count=None, commit=None)
        else:
            d = occupancy_device(occupancy[dev])
            devices[arch] = {"registers_per_sm": d["registers_per_sm"], "shared_per_sm": d["shared_per_sm"], "threads_per_sm": d["threads_per_sm"], "reserve_per_cta": d["reserve_per_cta"],
                             "label": "api", "device": dev, "sm_count": d["sm_count"], "commit": occupancy_commit(occupancy[dev], dev)}
        reserve[arch] = devices[arch]["reserve_per_cta"]
    base_fig = dict((arch, figures(kernel(compiled["base"], arch, ROWS_KERNEL))) for arch in ARCHS)
    out = {
        "constants": {"unified_l1_and_shared_bytes": {"value": UNIFIED_BYTES, "label": "model"}, "static_cap_bytes": {"value": STATIC_CAP_BYTES, "label": "plan"},
                      "gate_registers": GATE_REGISTERS, "gate_ctas_per_sm": GATE_CTAS, "threads_per_cta": THREADS},
        "devices": devices,
        "api_instances": {}, "carveout": {}, "source_facts": facts, "launch": launches,
        "compiled": {}, "production": {}, "tiles": {}, "configurations": {},
        "statements": list(STATEMENTS["budget"]),
    }
    for dev in rc.DEVICES:
        rec = occupancy[dev]
        base_name = base_row_name(rec)
        out["api_instances"][dev] = [{"name": n, "registers": instance(rec, n)["registers"], "static_smem": instance(rec, n)["static_smem"], "dynamic_smem": instance(rec, n)["dynamic_smem"],
                                      "threads": instance(rec, n)["threads"], "ctas_per_sm": instance(rec, n)["ctas_per_sm"], "label": "api"} for n in row_names(rec) + [REFERENCE_KERNEL]]
        values = [r["launch"]["shared_mem_config_size"] for r in ev["launch"]["reports"] if r["device"] == dev]
        out["carveout"][dev] = {"shared_mem_config_bytes": values[0], "reports": len(values), "label": "measured", "kernel": base_name}
        k = kernel(container[dev], DEVICE_ARCH[dev], dump_name(base_name))
        ref = kernel(container[dev], DEVICE_ARCH[dev], REFERENCE_KERNEL)
        conda_ref = kernel(compiled["base"], DEVICE_ARCH[dev], REFERENCE_KERNEL)
        out["production"][dev] = {"arch": DEVICE_ARCH[dev],
                                  "base": dict(figures(k), static_smem=k["shared"] - reserve[DEVICE_ARCH[dev]], label="compiler", same_as_conda=figures(k) == base_fig[DEVICE_ARCH[dev]]),
                                  "reference": dict(figures(ref), static_smem=ref["shared"] - reserve[DEVICE_ARCH[dev]], label="compiler", same_as_conda=figures(ref) == figures(conda_ref))}
    for arch in ARCHS:
        rows = []
        for cid, dump, inst, step, what in CANDIDATES:
            k = kernel(compiled[dump], arch, inst)
            fig = figures(k)
            rows.append(dict(fig, id=cid, prototype=dump if dump != "base" else None, commit=compiled[dump]["meta"]["tree"]["commit"], instantiation=inst, step=step, description=what,
                             static_smem=k["shared"] - reserve[arch], gate_excess=excess(base_fig[arch], fig) if cid != "base" else [], label="compiler"))
        ref = kernel(compiled["base"], arch, REFERENCE_KERNEL)
        rows.append(dict(figures(ref), id="reference", prototype=None, commit=compiled["base"]["meta"]["tree"]["commit"], instantiation=REFERENCE_KERNEL, step=None,
                         description="the frozen reference row engine", static_smem=ref["shared"] - reserve[arch], gate_excess=[], label="compiler"))
        out["compiled"][arch] = rows
    # the tiles (static bytes of the base come from the compiler, the others from the sources, the A fragments from the plan)
    fact = dict((f["name"], f["value"]) for f in facts)
    out["tiles"] = {
        "row_smem": {"source_bytes": fact["row_smem_bytes"], "compiler_static_bytes": dict((arch, kernel(compiled["base"], arch, ROWS_KERNEL)["shared"] - reserve[arch]) for arch in ARCHS),
                     "label": "source"},
        "e_tile": {"dynamic_bytes": fact["e_tile_dynamic_bytes"], "label": "source"},
        "kd_tile": {"static_bytes": fact["kd_tile_bytes"], "label": "source",
                    "agrees_with_compiler": dict((arch, next(r for r in out["compiled"][arch] if r["id"] == "kd_tile")["static_smem"] - next(r for r in out["compiled"][arch] if r["id"] == "base")["static_smem"]
                                                  == fact["kd_tile_bytes"]) for arch in ARCHS)},
        "a_fragments": {"static_bytes": A_FRAGMENTS_PLAN_BYTES, "label": "plan_estimate", "compiled": False},
    }
    out["lifetimes"] = tile_lifetimes()
    for arch in ARCHS:
        dv = devices[arch]
        static0 = kernel(compiled["base"], arch, ROWS_KERNEL)["shared"] - reserve[arch]
        kd, a = fact["kd_tile_bytes"], A_FRAGMENTS_PLAN_BYTES
        definitions = (
            ("base", "row_smem", static0, base_dynamic, "compiler"),
            ("e_tile", "row_smem + e_tile", static0, fact["e_tile_dynamic_bytes"], "arithmetic"),
            ("kd_tile", "row_smem + kd_tile", static0 + kd, 0, "arithmetic"),
            ("a_fragments", "row_smem + a_fragments", static0 + a, 0, "arithmetic_with_plan_estimate"),
            ("e_tile_a_fragments", "row_smem + e_tile + a_fragments", static0 + a, fact["e_tile_dynamic_bytes"], "arithmetic_with_plan_estimate"),
            ("e_tile_kd_tile_aliased_with_a_fragments", "row_smem + e_tile + max(kd_tile, a_fragments)", static0 + max(kd, a), fact["e_tile_dynamic_bytes"], "arithmetic_with_plan_estimate"),
            ("e_tile_kd_tile_a_fragments_without_alias", "row_smem + e_tile + kd_tile + a_fragments", static0 + kd + a, fact["e_tile_dynamic_bytes"], "arithmetic_with_plan_estimate"),
        )
        rows = []
        for cid, formula, static, dynamic, label in definitions:
            per_cta = static + dynamic + dv["reserve_per_cta"]
            row = {"id": cid, "formula": formula, "static_bytes": static, "dynamic_bytes": dynamic, "per_cta_bytes": per_cta, "two_ctas_bytes": 2 * per_cta,
                   "slack_bytes": dv["shared_per_sm"] - 2 * per_cta, "static_within_cap": static <= STATIC_CAP_BYTES, "two_ctas_fit": 2 * per_cta <= dv["shared_per_sm"],
                   "ctas_per_sm": ctas_per_sm(GATE_REGISTERS, THREADS, static, dynamic, dv), "ctas_per_sm_label": "arithmetic_at_the_register_ceiling",
                   "l1_upper_bound_bytes": UNIFIED_BYTES - 2 * per_cta, "l1_label": "arithmetic", "label": label}
            if cid == "base" and dv["device"] is not None:
                api = instance(occupancy[dv["device"]], base_row_name(occupancy[dv["device"]]))
                carve = out["carveout"][dv["device"]]["shared_mem_config_bytes"]
                row.update(ctas_per_sm=api["ctas_per_sm"], ctas_per_sm_label="api", l1_after_measured_carveout_bytes=UNIFIED_BYTES - carve, l1_after_measured_carveout_label="model")
            rows.append(row)
        out["configurations"][arch] = rows
    return out


def build_record(ev, recorded_on, repo):
    """The record that the evidence gives (the evidence must have passed ``evidence_problems``); ValueError when a source cannot be read."""
    profiles = ev["profiles"]
    commit = profiles[0]["meta"]["commit"]
    heads = profiles[0]["meta"]["heads"]
    base_commit = ev["compile"]["base"]["meta"]["tree"]["commit"]
    facts = source_facts(repo, base_commit, ev["compile"]["coalesced-state-access-path"]["meta"]["tree"]["commit"], ev["compile"]["chunk-end-row-offload"]["meta"]["tree"]["commit"])
    launches = dict((dev, dict(launch_constants(repo, occupancy_commit(ev["occupancy"][dev], dev)), commit=occupancy_commit(ev["occupancy"][dev], dev), path=LAUNCH_SOURCE, label="source"))
                    for dev in rc.DEVICES)
    cells = phase_profile(profiles)
    rec = {
        "schema": SCHEMA,
        "meta": {"recorded_on": recorded_on, "base_commit": base_commit, "profile_commit": commit,
                 "occupancy_commits": dict((dev, occupancy_commit(ev["occupancy"][dev], dev)) for dev in rc.DEVICES),
                 "profile": {"heads": heads, "shapes": list(rc.PROFILE_SHAPES), "positions": rpr.POSITIONS, "processes_per_device": PROCESSES, "phases": list(rpr.PHASES)},
                 "compile_toolchain": ev["compile"]["base"]["meta"]["toolchain"]["label"],
                 "container_toolchain": dict((dev, ev["container"][dev]["meta"]["toolchain"]["label"]) for dev in rc.DEVICES)},
        "inputs": ev["inputs"],
        "row_engine_order": list(rc.ROW_ENGINE_ORDER),
        "skipped": {},
        "order_basis": {"targets": dict((s, list(STEP_TARGETS[s])) for s in rc.ROW_ENGINE_ORDER), "prototypes": dict(STEP_PROTOTYPES), "statements": list(STATEMENTS["order"])},
        "phase_profile": {"phases": list(rpr.PHASES), "cells": cells},
        "value_split_inputs": value_split_inputs(ev["occupancy"], heads),
        "budget_table": budget_table(ev, facts, launches),
    }
    return json.loads(json.dumps(rec))


# ---- the section of the performance document ---------------------------------------------------------------------------------------------------
def _fmt(v):
    if v is None:
        return "none"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return "%d" % v
    if isinstance(v, float):
        return "%.3f" % v
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt(x) for x in v) if v else "none"
    return str(v)


def _table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] + ["| " + " | ".join(_fmt(c) for c in r) + " |" for r in rows]


def render_section(rec, number=SECTION_NUMBER):
    """The text of the section: a pure function of the record (the document must equal it, so that every number of the document is a number of the record)."""
    ph = rpr.PHASES
    cells = rec["phase_profile"]["cells"]
    vs, bt = rec["value_split_inputs"], rec["budget_table"]
    out = ["## %d. %s" % (number, SECTION_TITLE), "",
           "Generated from the decision record by `benchmarks/decision_record.py`; every number below is a number of the record, and the record is derived from the raw evidence (it is checked by deriving it again).", "",
           "### Order of the row-engine steps", "", "Order: %s. Left out: %s." % (_fmt(rec["row_engine_order"]), _fmt(sorted(rec["skipped"]))), ""]
    out += ["- " + s for s in rec["order_basis"]["statements"]] + [""]
    out += _table(["step", "phases it changes", "prototype (compile-only)"],
                  [[s, rec["order_basis"]["targets"][s], rec["order_basis"]["prototypes"][s]] for s in rec["row_engine_order"]]) + [""]
    out += ["### Where the time of a row goes", "",
            "Clock profile of the optimised row engine on both devices (heads %d, positions 0 to %d, three processes per device). Cycles and shares are the means of the processes; regular rows are the positions "
            "before the chunk end, the chunk-end row is the last position. A share is the phase's cycles over the row's cycles, in percent; the range over the processes is in the record." % (
                rec["meta"]["profile"]["heads"], rec["meta"]["profile"]["positions"] - 1), ""]
    for group, title in (("regular", "Regular rows"), ("chunk_end", "Chunk-end row")):
        out += ["#### %s: share of the row in percent" % title, ""]
        out += _table(["device", "N", "row cycles", "spread %"] + list(ph), [[c["device"], c["n"], c["row_cycles"], c["row_spread_percent"]] + [c["share_percent"][p] for p in ph] for c in cells if c["group"] == group]) + [""]
    out += ["#### Prologue and epilogue: share of the kernel in percent", ""]
    out += _table(["device", "N", "group of rows", "kernel cycles", "prologue", "epilogue"], [[c["device"], c["n"], c["group"], c["kernel_cycles"], c["prologue_percent_of_kernel"], c["epilogue_percent_of_kernel"]]
                                                                                for c in cells]) + [""]
    out += ["### Inputs of the value-split decision", ""] + ["- " + s for s in vs["statements"]] + [""]
    out += _table(["device", "N", "group of rows", "prefix proxy %", "certain part %"], [[c["device"], c["n"], c["group"], c["prefix_proxy_percent"], c["certain_part_percent"]] for c in cells]) + [""]
    out += ["Prefix proxy: %s; certain part: %s. Strict value-independent share: %s." % (_fmt(vs["prefix_proxy_phases"]), _fmt(vs["certain_part_phases"]), _fmt(vs["strict_v_independent_share"])), "",
            "N * H of the matrix: %s. N * H of the profile: %s. The draft: N * H up to %s, gain %s percent (%s)." % (_fmt(vs["matrix_nh"]), _fmt(vs["profile_nh"]), _fmt(vs["draft"]["nh_up_to"]),
                                                                           _fmt(vs["draft"]["gain_percent"]), vs["draft"]["label"]),
            "Inside the draft's range: matrix %s, profile %s; the profile covers them: %s." % (_fmt(vs["matrix_nh_within_draft"]), _fmt(vs["profile_nh_within_draft"]), _fmt(vs["profile_covers_matrix_within_draft"])), ""]
    out += _table(["device", "SMs", "CTAs per SM", "resident slots", "slices", "N * H, one CTA per SM", "N * H, resident slots"],
                  [[c["device"], c["sm_count"], c["ctas_per_sm"], c["resident_slots"], m["slices"], m["one_cta_per_sm"], m["resident_slots"]] for c in vs["capacity"] for m in c["max_nh_by_slices"]]) + [""]
    out += ["### Budget per architecture", ""] + ["- " + s for s in bt["statements"]] + [""]
    out += ["#### Occupancy API and measured carve-out of the base kernel", ""]
    out += _table(["device", "instantiation", "registers", "static B", "dynamic B", "threads", "CTAs per SM", "provenance"],
                  [[d, i["name"], i["registers"], i["static_smem"], i["dynamic_smem"], i["threads"], i["ctas_per_sm"], i["label"]] for d in rc.DEVICES for i in bt["api_instances"][d]]) + [""]
    out += _table(["device", "shared-memory carve-out B", "launches", "provenance"], [[d, bt["carveout"][d]["shared_mem_config_bytes"], bt["carveout"][d]["reports"], bt["carveout"][d]["label"]] for d in rc.DEVICES]) + [""]
    out += _table(["device", "architecture", "kernel", "registers", "stack", "spill loads", "spill stores", "static B", "same figures as the conda toolchain"],
                  [[d, bt["production"][d]["arch"], k, bt["production"][d][k]["registers"], bt["production"][d][k]["stack"], bt["production"][d][k]["spill_loads"], bt["production"][d][k]["spill_stores"],
                    bt["production"][d][k]["static_smem"], bt["production"][d][k]["same_as_conda"]] for d in rc.DEVICES for k in ("base", "reference")]) + [""]
    out += ["#### Source constants", ""]
    out += _table(["name", "bytes", "commit", "path", "defines"], [[f["name"], f["value"], f["commit"][:12], f["path"], f["defines"]] for f in bt["source_facts"]]) + [""]
    out += ["#### Launch of the row engine (source constants at the commits of the occupancy records)", ""]
    out += _table(["device", "commit", "path", "threads", "dynamic B", "provenance"], [[d, bt["launch"][d]["commit"], bt["launch"][d]["path"], bt["launch"][d]["threads"], bt["launch"][d]["dynamic_smem"], bt["launch"][d]["label"]]
                                                                                      for d in rc.DEVICES]) + [""]
    lt = bt["lifetimes"]
    out += ["#### Tile lifetimes (provenance: %s, tested: %s)" % (lt["label"], _fmt(lt["tested"])), ""]
    out += _table(["tile", "live phases"], [[t, lt["live_phases"][t]] for t in sorted(lt["live_phases"])]) + [""]
    out += ["May alias: %s. Must not alias: %s." % (_fmt(["%s with %s" % tuple(p) for p in lt["may_alias"]]), _fmt(["%s with %s" % tuple(p) for p in lt["must_not_alias"]])), ""]
    for arch in ARCHS:
        out += ["#### Compiled resources of the prototypes, %s (compile-only, as committed)" % arch, ""]
        out += _table(["candidate", "registers", "stack", "spill loads", "spill stores", "static B", "above the base"],
                      [[r["id"], r["registers"], r["stack"], r["spill_loads"], r["spill_stores"], r["static_smem"], r["gate_excess"]] for r in bt["compiled"][arch]]) + [""]
        d = bt["devices"][arch]
        out += ["#### Shared-memory configurations, %s (shared memory per SM %s B, reserve %s B per CTA, provenance of the limits: %s)" % (arch, _fmt(d["shared_per_sm"]), _fmt(d["reserve_per_cta"]), d["label"]), ""]
        out += _table(["configuration", "static B", "dynamic B", "per CTA B", "two CTAs B", "slack B", "CTAs per SM", "L1 upper bound B", "measured carve-out: L1 B", "provenance"],
                      [[r["id"], r["static_bytes"], r["dynamic_bytes"], r["per_cta_bytes"], r["two_ctas_bytes"], r["slack_bytes"], "%s (%s)" % (_fmt(r["ctas_per_sm"]), r["ctas_per_sm_label"]),
                        r["l1_upper_bound_bytes"], r.get("l1_after_measured_carveout_bytes"), r["label"]] for r in bt["configurations"][arch]]) + [""]
    return "\n".join(out).rstrip("\n") + "\n"


def section_bounds(document_text):
    """``(first line, last line + 1)`` of the section headed ``Row engine: decision record`` (None when there is none): up to the next heading of the same or a higher level."""
    lines = document_text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^(#+)\s*(.*)$", line)
        if m and SECTION_TITLE.lower() in m.group(2).lower():
            for j in range(i + 1, len(lines)):
                n = re.match(r"^(#+)\s", lines[j])
                if n and len(n.group(1)) <= len(m.group(1)):
                    return i, j
            return i, len(lines)
    return None


def replace_section(document_text, section):
    """The document with the section replaced by ``section`` (appended when the document has none)."""
    b = section_bounds(document_text)
    lines = document_text.splitlines()
    new = section.rstrip("\n").splitlines()
    if b is None:
        return "\n".join(lines + [""] + new) + "\n"
    return "\n".join(lines[:b[0]] + new + [""] + lines[b[1]:]).rstrip("\n") + "\n"


# ---- the check ----------------------------------------------------------------------------------------------------------------------------------
def diff(want, got, path="record", limit=6):
    """The paths at which two JSON values differ (at most ``limit``)."""
    out = []

    def walk(a, b, p):
        if len(out) >= limit:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if k not in a:
                    out.append("%s.%s is not in the derived record" % (p, k))
                elif k not in b:
                    out.append("%s.%s is missing" % (p, k))
                else:
                    walk(a[k], b[k], "%s.%s" % (p, k))
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                out.append("%s has %d entries, the derived record %d" % (p, len(b), len(a)))
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, "%s[%d]" % (p, i))
        elif a != b or type(a) is not type(b):
            out.append("%s is %r, the evidence gives %r" % (p, b, a))
    walk(want, got, path)
    return out[:limit]


def record_problems(rec, ev, repo, document_text=None, head="HEAD"):
    """Everything that is wrong with a decision record: its structure (``rc.decision_problems``), the evidence it rests on, a difference from the record that the evidence gives, prose that
    carries a number or a plan marker or a path of the loop, the commits against the repository, and a document section that is not the generated one."""
    where = "decision record %s" % (rec.get("_path", "?") if isinstance(rec, dict) else "?")
    if not isinstance(rec, dict) or rec.get("schema") != SCHEMA:
        return ["%s: not a %s record" % (where, SCHEMA)]
    problems = rc.decision_problems(rec)
    try:
        problems.extend("%s: %s" % (where, p) for p in evidence_problems(ev, repo))
    except (AttributeError, KeyError, TypeError, ValueError, ArithmeticError) as e:
        problems.append("%s: the evidence cannot be read: %s" % (where, e))
    if repo is None:
        problems.append("%s: the sources that define the tile sizes are read from the repository: repo is required" % where)
    if problems:
        return problems
    meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
    recorded_on = meta.get("recorded_on")
    if not rc.hexcommit(recorded_on):
        return ["%s: meta.recorded_on must be the commit that the record was written on" % where]
    try:
        want = build_record(ev, recorded_on, repo)
    except (ValueError, KeyError, StopIteration, TypeError, ArithmeticError) as e:
        return ["%s: the record cannot be derived from the evidence: %s" % (where, e)]
    given = json.loads(json.dumps(dict((k, v) for k, v in rec.items() if k != "_path")))
    problems.extend("%s: %s" % (where, d) for d in diff(want, given))
    text = json.dumps(given, sort_keys=True)
    if rc.PLAN_VOCABULARY.search(text) or LOOP_DIR in text:
        problems.append("%s: the record carries a plan marker or a path of the loop" % where)
    for group, statements in sorted(STATEMENTS.items()):
        if any(NUMBER.search(s) for s in statements):
            problems.append("%s: the statements of %s must carry no number: figures live in the tables" % (where, group))
    for dev, commit in sorted(want["meta"]["occupancy_commits"].items()):
        problems.extend("%s: %s" % (where, p) for p in commit_problems("occupancy record of %s" % dev, commit, repo, recorded_on, ancestor=True))
    problems.extend("%s: %s" % (where, p) for p in commit_problems("the profile", want["meta"]["profile_commit"], repo, recorded_on, ancestor=True))
    problems.extend("%s: %s" % (where, p) for p in commit_problems("the base", want["meta"]["base_commit"], repo, recorded_on, ancestor=True))
    problems.extend("%s: %s" % (where, p) for p in commit_problems("the record's own commit", recorded_on, repo, head, ancestor=True))
    for tree in sorted(PROTOTYPE_DUMPS):
        commit = ev["compile"][tree]["meta"]["tree"]["commit"]
        problems.extend("%s: %s" % (where, p) for p in commit_problems("the prototype %s" % tree, commit, repo, None, ancestor=False))
    if document_text is not None:
        problems.extend(document_problems(want, document_text, where))                  # the derived record: a record that differs from it is a problem already, and must not crash the rendering
    return problems


def commit_problems(what, commit, repo, descendant, ancestor):
    full = rc.resolve(repo, commit)
    if full is None:
        return ["%s names commit %s, which the repository does not have" % (what, commit)]
    if ancestor:
        top = rc.resolve(repo, descendant)
        if top is None:
            return ["cannot resolve %s in the repository" % descendant]
        if full != top and not rc.is_ancestor(repo, full, top):
            return ["%s names commit %s, which is not an ancestor of %s" % (what, commit, descendant)]
    return []


def document_problems(rec, document_text, where):
    b = section_bounds(document_text)
    if b is None:
        return ["%s: the performance document has no section headed %r" % (where, SECTION_TITLE)]
    lines = document_text.splitlines()
    m = re.match(r"^#+\s*(\d+)\.", lines[b[0]])
    number = int(m.group(1)) if m else SECTION_NUMBER
    have = "\n".join(lines[b[0]:b[1]]).rstrip("\n") + "\n"
    want = render_section(rec, number)
    if have == want:
        return []
    h, w = have.splitlines(), want.splitlines()
    first = next((i for i, (x, y) in enumerate(zip(h, w)) if x != y), min(len(h), len(w)))
    return ["%s: the section of the performance document is not the generated one (line %d of the section: %r, generated %r)"
            % (where, first + 1, h[first] if first < len(h) else None, w[first] if first < len(w) else None)]


# ---- command line -------------------------------------------------------------------------------------------------------------------------------
def _report(problems):
    for p in problems:
        print("PROBLEM: %s" % p)
    print("VERDICT: %s" % ("FAIL" if problems else "PASS"))
    return 1 if problems else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("build")
    p.add_argument("--evidence", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--recorded-on", default="HEAD")
    p = sub.add_parser("render")
    p.add_argument("--record", required=True)
    p.add_argument("--document", required=True)
    p.add_argument("--number", type=int, default=SECTION_NUMBER)
    p = sub.add_parser("check")
    p.add_argument("--record", required=True)
    p.add_argument("--evidence", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--document", default=None)
    p.add_argument("--head", default="HEAD")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "build":
            ev = load_evidence(a.evidence)
            problems = evidence_problems(ev, a.repo)
            if problems:
                return _report(problems)
            on = rc.resolve(a.repo, a.recorded_on)
            if on is None:
                print("cannot resolve %s in %s" % (a.recorded_on, a.repo), file=sys.stderr)
                return 2
            rec = build_record(ev, on, a.repo)
            with open(a.out, "w") as f:
                json.dump(rec, f, indent=1, sort_keys=True)
                f.write("\n")
            print("wrote %s (%d bytes)" % (a.out, os.path.getsize(a.out)))
            return 0
        if a.cmd == "render":
            rec = rc.load_json(a.record)
            with open(a.document, encoding="utf-8") as f:
                doc = f.read()
            new = replace_section(doc, render_section(rec, a.number))
            with open(a.document, "w", encoding="utf-8") as f:
                f.write(new)
            return 0
        if a.cmd == "check":
            rec = rc.load_json(a.record)
            ev = load_evidence(a.evidence)
            doc = None
            if a.document:
                with open(a.document, encoding="utf-8") as f:
                    doc = f.read()
            return _report(record_problems(rec, ev, a.repo, doc, a.head))
    except (OSError, ValueError, KeyError, ArithmeticError) as e:
        print("unusable input: %s" % e, file=sys.stderr)
        return 2
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
