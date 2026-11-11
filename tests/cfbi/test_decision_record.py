"""Self-tests of the decision record of the row engine (benchmarks/decision_record.py): the record is derived from the raw evidence, the checker derives it again and rejects every difference.

Plain Python, no GPU and no torch (``pytest --noconftest tests/cfbi/test_decision_record.py``). The evidence is synthetic and has numbers that can be computed by hand: a clock profile whose
phases are constants (so that every share is a ratio of small integers), compile dumps with chosen register / stack / spill figures, occupancy-API records, Nsight Compute launch metrics and a
git history whose sources define the tile sizes. Every rule has a mutant of the record and a mutant of the evidence.
"""
import contextlib
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "benchmarks", _ROOT / "tests" / "cfbi"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import decision_record as dr  # noqa: E402
import record_checker as rc  # noqa: E402
import rows_profile_record as rpr  # noqa: E402
import test_bench_tools as tbt  # noqa: E402
from test_record_checker import commit_file, git, has, mutated, ncu_record, profile_record  # noqa: E402
from test_rows_profile import make_slots  # noqa: E402

GPU = {"b200": "NVIDIA B200", "b300": "NVIDIA B300 SXM6 AC"}
PRIMITIVES = "constexpr int HEAD_K = 128;\nconstexpr int HEAD_V = 128;\nconstexpr int BT = 64;\nconstexpr int NTHREADS = 256;   // all kernels use 8 warps\n"
ROWS = """namespace cfbi {
struct RowSmem {
    float k[HEAD_K];      // khat of the current row
    float q[HEAD_K];      // qhat of the current row
    float g2[HEAD_K];     // G2 of the current row
    float gref[HEAD_K];   // G2 of the reference row r(pos)
    float gq[HEAD_K];     // GQ = Gamma * qhat
    float gend[HEAD_K];   // GEND (chunk end only)
    float x[2][HEAD_K];   // UT_t, W_t of the current row
    float m[3][HEAD_V];   // cross-sub-block products: UT, W, and U (for O)
    float p1[HEAD_V], p2[HEAD_V], u[HEAD_V];
    float akk[BT], aqk[BT];
    float beta;
};
}
"""
ROWS_FLOATS = 6 * 128 + 2 * 128 + 3 * 128 + 3 * 128 + 2 * 64 + 1                      # 1921, counted by hand from the struct above
E_LINES = "constexpr int ROWS_E_STRIDE = HEAD_K + 4;\nconstexpr int ROWS_DYN_SMEM_BYTES = HEAD_V * ROWS_E_STRIDE * static_cast<int>(sizeof(float));   // staged copy of E\n"
KD_LINES = "constexpr int ROWS_KD_FRAGS = 8 * 8 * 32;\n"
LAUNCH = "constexpr size_t ROWS_DYNAMIC_SMEM = 0;   // the dynamic shared memory of the row-engine launch: the launch sites and the occupancy hook read it\n"
E_BYTES, KD_BYTES = 128 * 132 * 4, 8 * 8 * 32 * 16                                    # 67584 and 32768, by hand
RESERVE = 1024
# the base row engine per architecture: (registers, stack, spill stores, spill loads, static shared memory)
BASE = {"sm_90a": (128, 168, 428, 172, 7808), "sm_100a": (128, 112, 4, 4, 8192), "sm_103a": (128, 112, 12, 8, 8192)}
REFERENCE = {"sm_90a": (48, 0, 0, 0, 7296), "sm_100a": (48, 0, 0, 0, 8192), "sm_103a": (48, 0, 0, 0, 8192)}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


# ---- the evidence -------------------------------------------------------------------------------------------------------------------------------
def phases_of(k):
    """The phases of the process ``k``: constants, so that every share is a ratio of integers (prep 10 + k cycles, the chunk-end row adds 400 to the last phase)."""
    return lambda pos, cta: [10 + k, 20, 30, 5, 50, 8, 3 + (pos == 63) * 400]


def process_record(k, commit):
    rec = profile_record(shapes=(1,), heads=12, commit=commit)
    rec["meta"]["shapes"] = list(rc.PROFILE_SHAPES)
    rec["cells"] = [rpr.cell_from_slots(make_slots(n, 12, phases_of(k), sm_of=lambda pos, cta: cta % 148), n, 12) for n in rc.PROFILE_SHAPES]
    return rec


def kernel_entry(figures, **extra):
    reg, stack, stores, loads, static = figures
    return tbt.cg_kernel(reg=reg, stack=stack, spill=(stores, loads), shared=static + RESERVE, **extra)


def compile_dump(archs_kernels, commit, label="conda 13.1"):
    dump = tbt.cg_dump(archs_kernels, label=label)
    dump["meta"]["tree"]["commit"] = commit
    return dump


def prototype_figures(arch, cid):
    """A distinct figure set for every candidate: the base's plus a step for the ones that break the resource gate (above the base in the stack or the spills)."""
    reg, stack, stores, loads, static = BASE[arch]
    return {"base": BASE[arch], "trims": (reg, stack - 80, stores + 28, loads + 20, static), "e_tile_with_paired_m_loads": (reg, stack + 224, stores + 400, loads + 224, static),
            "kd_tile": (reg, stack + 48, stores + 148, loads + 56, static + KD_BYTES), "single_launch_off": BASE[arch], "single_launch_on": (reg, stack - 16, 0, 0, static),
            "operand_swap_off": BASE[arch], "operand_swap_on": (reg, stack - 64, stores + 16, loads + 12, static), "value_split_128": (reg, stack, stores + 8, loads + 4, static),
            "value_split_64": (reg, stack - 32, stores, loads + 4, static), "value_split_32": (reg, stack - 48, 0, 0, static), "value_split_16": (reg, stack - 16, stores + 16, loads + 16, static)}[cid]


def occupancy_record(dev, commit, rows_ctas=2, rows_threads=256, rows_registers=128):
    return {"capability": [10, 0], "cuda": "13.4", "gpu": GPU[dev], "label": "%s-%s" % (dev, commit), "sm_count": 148, "torch": "2.14.0",
            "occupancy": {"device": {"max_threads_per_sm": 2048, "regs_per_sm": 65536, "sm_count": 148, "smem_per_sm": 233472, "smem_reserved_per_block": RESERVE},
                          "instances": [{"name": "cfbi::rows_kernel", "registers": rows_registers, "static_smem": 8192, "dynamic_smem": 0, "threads": rows_threads, "ctas_per_sm": rows_ctas},
                                        {"name": "cfbi::ref::rows_kernel", "registers": 40, "static_smem": 8192, "dynamic_smem": 0, "threads": 256, "ctas_per_sm": 6}]}}


@pytest.fixture(scope="module")
def processes():
    """The three process records (the slow part), built once; a device gets copies of them with its own GPU name."""
    recs = [process_record(k, "c0ffee0123456789") for k in range(3)]
    out = {}
    for dev in rc.DEVICES:
        out[dev] = []
        for rec in recs:
            rec = copy.deepcopy(rec)
            rec["meta"]["gpu"] = GPU[dev]
            out[dev].append(rec)
    return out


def write(path, obj):
    path.write_text(json.dumps(obj, sort_keys=True))
    return path


def make_evidence(root, processes):
    """A repository (the base, four loop commits, the prototypes' commits) and the evidence files with a manifest; ``(manifest path, repo, commits)``."""
    root = Path(root)
    repo = root / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    commit_file(repo, "csrc/cfbi/cfbi_primitives.cuh", PRIMITIVES, "the sizes of a head")
    commits = {"base": commit_file(repo, "csrc/cfbi/cfbi_rows.cuh", ROWS, "the row engine")}
    git(repo, "checkout", "-q", "-b", "proto-e")
    commits["e"] = commit_file(repo, "csrc/cfbi/cfbi_rows.cuh", ROWS + E_LINES, "the staged tile")
    commits["e2"] = commit_file(repo, "csrc/cfbi/cfbi_rows.cuh", ROWS + E_LINES.replace("HEAD_K + 4", "HEAD_K + 12"), "the staged tile with another stride")
    git(repo, "checkout", "-q", "main")
    git(repo, "checkout", "-q", "-b", "proto-kd")
    commits["kd"] = commit_file(repo, "csrc/cfbi/cfbi_rows.cuh", ROWS + KD_LINES, "the KD tile")
    git(repo, "checkout", "-q", "main")
    commits["profile"] = commit_file(repo, "docs/profile.md", "profile\n", "the profile build")
    commits["occ_b200"] = commit_file(repo, "csrc/cfbi/cfbi_launch.cu", LAUNCH, "the occupancy binding")
    commits["occ_b300"] = commit_file(repo, "csrc/cfbi/cfbi_launch.cu", LAUNCH + "// second device\n", "the occupancy binding, second device")
    commits["cfg"] = commit_file(repo, "csrc/cfbi/cfbi_launch.cu", LAUNCH.replace("= 0;", "= 4096;"), "a launch with dynamic shared memory")
    commits["head"] = commit_file(repo, "docs/last.md", "last\n", "the commit that the record is written on")
    profile_commit = commits["profile"]
    ev = root / "evidence"
    ev.mkdir()
    manifest = {"profile": [], "ncu": [], "compile": {}, "container": {}, "occupancy": {}}
    for dev in rc.DEVICES:
        for k, rec in enumerate(processes[dev]):
            rec = copy.deepcopy(rec)
            rec["meta"]["commit"] = profile_commit
            rec.pop("_path", None)
            manifest["profile"].append(write(ev / ("profile-%s-r%d.json" % (dev, k)), rec).name)
    ncu_paths, launches = [], []
    for dev in rc.DEVICES:
        for n in rc.NCU_SHAPES:
            rec = ncu_record(n=n, position=49, cache="all", gpu=GPU[dev], commit=profile_commit)
            rec["meta"]["report"]["sha256"] = sha("%s-%d" % (dev, n))
            rec["meta"]["launch"]["shared_mem_per_block"] = 8192 + RESERVE
            rec.pop("_path", None)
            manifest["ncu"].append(write(ev / ("ncu-%s-n%d.json" % (dev, n)), rec).name)
            launches.append({"device": dev, "case": "n%d-pos49_cold" % n, "report_sha256": sha("%s-%d" % (dev, n)), "cc": "10.0",
                             "launch": {"shared_mem_config_size": 65536, "shared_mem_per_block_static": 8192, "shared_mem_per_block_driver": RESERVE, "shared_mem_per_block_dynamic": 0,
                                        "occupancy_limit_registers": 2, "occupancy_limit_shared_mem": 7, "registers_per_thread": 128, "block_size": 256, "grid_size": n * 12}})
    manifest["launch_config"] = write(ev / "launch.json", {"schema": dr.LAUNCH_SCHEMA, "tool": "ncu", "reports": launches}).name
    for tree in ("base",) + dr.PROTOTYPE_DUMPS:
        tree_commit = {"base": commits["base"], "coalesced-state-access-path": commits["e"], "chunk-end-row-offload": commits["kd"]}.get(tree, commits["base"])
        kernels = dict((arch, {}) for arch in dr.ARCHS)
        for cid, dump, inst, step, what in dr.CANDIDATES:
            if dump == tree:
                for arch in dr.ARCHS:
                    kernels[arch][inst] = kernel_entry(prototype_figures(arch, cid))
        if tree == "base":
            for arch in dr.ARCHS:
                kernels[arch][dr.REFERENCE_KERNEL] = kernel_entry(REFERENCE[arch], maxreg=None, maxthr=None)
        manifest["compile"][tree] = write(ev / ("compile-%s.json" % tree), compile_dump(kernels, tree_commit)).name
    for dev, arch in dr.DEVICE_ARCH.items():
        kernels = {arch: {dr.ROWS_KERNEL: kernel_entry(BASE[arch]), dr.REFERENCE_KERNEL: kernel_entry((40, 0, 0, 0, 8192), maxreg=None, maxthr=None)}}
        manifest["container"][dev] = write(ev / ("container-%s.json" % dev), compile_dump(kernels, profile_commit, label="container NGC pytorch 26.08: nvcc 13.4.59")).name
        manifest["occupancy"][dev] = write(ev / ("occupancy-%s.json" % dev), occupancy_record(dev, commits["occ_" + dev][:9])).name
    path = write(ev / "manifest.json", manifest)
    return path, repo, commits


@pytest.fixture(scope="module")
def world(tmp_path_factory, processes):
    path, repo, commits = make_evidence(tmp_path_factory.mktemp("decision"), processes)
    ev = dr.load_evidence(path)
    rec = dr.build_record(ev, commits["head"], repo)
    return {"manifest": path, "repo": repo, "commits": commits, "ev": ev, "rec": rec}


def check(world, rec=None, document=None, ev=None, repo="repo", head=None):
    rec = copy.deepcopy(world["rec"]) if rec is None else rec
    ev = world["ev"] if ev is None else ev
    return dr.record_problems(rec, ev, world["repo"] if repo == "repo" else repo, document, head or world["commits"]["head"])


@contextlib.contextmanager
def edited(world, files):
    """The evidence files of ``files`` rewritten on disk for the duration of the block: ``files`` maps a manifest path ('profile', 0) / ('compile', 'base') / ... to a function that edits
    the loaded JSON."""
    manifest = json.loads(Path(world["manifest"]).read_text())
    root = Path(world["manifest"]).parent
    saved = {}
    try:
        for (group, key), edit in files.items():
            name = manifest[group][key] if isinstance(manifest[group], (list, dict)) else manifest[group]
            path = root / name
            saved[path] = path.read_text()
            content = json.loads(saved[path])
            edit(content)
            path.write_text(json.dumps(content, sort_keys=True))
        yield
    finally:
        for path, text in saved.items():
            path.write_text(text)


def reloaded(world, files):
    """The evidence with some of its files replaced (see ``edited``)."""
    with edited(world, files):
        return dr.load_evidence(world["manifest"])


# ---- the derivation, by hand ---------------------------------------------------------------------------------------------------------------------
def test_the_record_passes_its_own_check_and_the_numbers_are_the_hand_computed_ones(world):
    rec = world["rec"]
    assert check(world) == []
    cell = next(c for c in rec["phase_profile"]["cells"] if (c["device"], c["n"], c["group"]) == ("b200", 16, "regular"))
    # prep is 10, 11, 12 cycles in the three processes: the pooled row is 126 + 1 cycles, the pooled prep 11
    assert cell["row_cycles"] == 127.0 and cell["phase_cycles"]["prep"] == 11.0
    assert cell["share_percent"]["prep"] == round(100 * 11 / 127.0, 3)
    assert cell["share_min_percent"]["prep"] == round(100 * 10 / 126.0, 3) and cell["share_max_percent"]["prep"] == round(100 * 12 / 128.0, 3)
    assert cell["row_spread_percent"] == round(100 * 2 / 127.0, 3)
    assert cell["prefix_proxy_percent"] == round(100 * (11 + 20) / 127.0, 3) and cell["certain_part_percent"] == round(100 * 20 / 127.0, 3)
    chunk = next(c for c in rec["phase_profile"]["cells"] if (c["device"], c["n"], c["group"]) == ("b300", 1, "chunk_end"))
    assert chunk["share_percent"]["chunk_end_sync"] == round(100 * 403 / 527.0, 3)
    assert rec["row_engine_order"] == ["trims", "e_tile", "a_fragments"] and rec["skipped"] == {}


def test_the_sizes_of_the_tiles_come_from_the_sources_at_their_commits(world):
    facts = dict((f["name"], f) for f in world["rec"]["budget_table"]["source_facts"])
    assert facts["row_smem_bytes"]["value"] == 4 * ROWS_FLOATS == 7684 and facts["row_smem_bytes"]["commit"] == world["commits"]["base"]
    assert facts["e_tile_dynamic_bytes"]["value"] == E_BYTES == 67584 and facts["e_tile_dynamic_bytes"]["commit"] == world["commits"]["e"]
    assert facts["kd_tile_bytes"]["value"] == KD_BYTES == 32768 and facts["kd_tile_bytes"]["commit"] == world["commits"]["kd"]
    tiles = world["rec"]["budget_table"]["tiles"]
    assert tiles["kd_tile"]["agrees_with_compiler"] == {"sm_90a": True, "sm_100a": True, "sm_103a": True}           # the compiler's static growth is the source's size
    assert tiles["a_fragments"] == {"static_bytes": 6144, "label": "plan_estimate", "compiled": False}


def test_the_budget_arithmetic_and_the_provenance_of_every_figure(world):
    bt = world["rec"]["budget_table"]
    rows = dict((r["id"], r) for r in bt["configurations"]["sm_100a"])
    assert rows["base"]["per_cta_bytes"] == 8192 + 1024 and rows["base"]["ctas_per_sm"] == 2 and rows["base"]["ctas_per_sm_label"] == "api"
    assert rows["base"]["l1_after_measured_carveout_bytes"] == 262144 - 65536                                       # only the base kernel gets the measured carve-out
    for cid, row in rows.items():
        if cid != "base":
            assert "l1_after_measured_carveout_bytes" not in row and row["ctas_per_sm_label"] == "arithmetic_at_the_register_ceiling" and row["l1_label"] == "arithmetic"
    assert rows["e_tile"]["per_cta_bytes"] == 8192 + E_BYTES + 1024 and rows["e_tile"]["l1_upper_bound_bytes"] == 262144 - 2 * (8192 + E_BYTES + 1024)
    full = rows["e_tile_kd_tile_a_fragments_without_alias"]
    assert full["static_bytes"] == 8192 + KD_BYTES + 6144 and full["two_ctas_bytes"] == 2 * (8192 + KD_BYTES + 6144 + E_BYTES + 1024) and full["label"] == "arithmetic_with_plan_estimate"
    assert full["slack_bytes"] == 233472 - full["two_ctas_bytes"] and full["two_ctas_fit"] is True
    hopper = dict((r["id"], r) for r in bt["configurations"]["sm_90a"])
    assert bt["devices"]["sm_90a"]["label"] == "plan" and bt["devices"]["sm_90a"]["device"] is None and hopper["base"]["ctas_per_sm_label"] == "arithmetic_at_the_register_ceiling"
    assert bt["devices"]["sm_100a"]["label"] == "api" and bt["devices"]["sm_100a"]["commit"] == world["commits"]["occ_b200"][:9]
    assert bt["carveout"]["b300"] == {"shared_mem_config_bytes": 65536, "reports": 2, "label": "measured", "kernel": "cfbi::rows_kernel"}
    prototypes = dict((r["id"], r) for r in bt["compiled"]["sm_100a"])
    assert prototypes["base"]["gate_excess"] == [] and prototypes["single_launch_on"]["gate_excess"] == [] and prototypes["value_split_32"]["gate_excess"] == []
    assert prototypes["trims"]["gate_excess"] == ["spill_loads", "spill_stores"] and prototypes["e_tile_with_paired_m_loads"]["gate_excess"] == ["stack", "spill_loads", "spill_stores"]
    assert prototypes["value_split_64"]["gate_excess"] == ["spill_loads"]                                           # a spill load above the base fails even when the stores are not above it
    assert bt["production"]["b200"]["reference"]["same_as_conda"] is False and bt["production"]["b200"]["base"]["same_as_conda"] is True      # 40 registers against 48: the toolchains differ


def test_the_value_split_inputs_and_the_capacity(world):
    vs = world["rec"]["value_split_inputs"]
    assert vs["matrix_nh"] == [12, 48, 192, 768, 1536, 3072] and vs["profile_nh"] == [12, 192, 1536]
    assert vs["matrix_nh_within_draft"] == [12, 48] and vs["profile_nh_within_draft"] == [12] and vs["profile_covers_matrix_within_draft"] is False
    assert vs["strict_v_independent_share"] is None and vs["draft"] == {"nh_up_to": 74, "gain_percent": [10, 25], "label": "draft_estimate_unprofiled"}
    cap = vs["capacity"][0]
    assert cap["resident_slots"] == 296 and [(m["slices"], m["one_cta_per_sm"], m["resident_slots"]) for m in cap["max_nh_by_slices"]] == [(2, 74, 148), (4, 37, 74), (8, 18, 37)]


def test_the_tile_lifetimes_give_the_alias_pairs_from_the_live_phases(world):
    lt = world["rec"]["budget_table"]["lifetimes"]
    assert lt["label"] == "plan" and lt["tested"] is False                                                          # a design inference: no alias has been built or race-checked
    assert lt["live_phases"]["kd_tile"] == ["chunk_end_sync"] and lt["live_phases"]["a_fragments"] == ["akk_aqk", "m_tiles", "p1_p2"]
    assert lt["live_phases"]["e_tile"] == list(rpr.PHASES) == lt["live_phases"]["row_smem"]
    assert lt["may_alias"] == [["a_fragments", "kd_tile"]]                                                          # the only pair whose live phases are disjoint
    assert lt["must_not_alias"] == [["a_fragments", "e_tile"], ["a_fragments", "row_smem"], ["e_tile", "kd_tile"], ["e_tile", "row_smem"], ["kd_tile", "row_smem"]]
    # the derivation follows the live phases: a tile that is live in the chunk-end block as well can no longer share with the KD tile
    saved = dr.TILE_LIVE_PHASES["a_fragments"]
    try:
        dr.TILE_LIVE_PHASES["a_fragments"] = saved + ("chunk_end_sync",)
        assert dr.tile_lifetimes()["may_alias"] == []
    finally:
        dr.TILE_LIVE_PHASES["a_fragments"] = saved


def test_the_occupancy_arithmetic_reproduces_the_api_and_reads_the_limits():
    dev = {"registers_per_sm": 65536, "shared_per_sm": 233472, "threads_per_sm": 2048, "reserve_per_cta": 1024}
    assert dr.ctas_per_sm(128, 256, 8192, 0, dev) == 2 and dr.ctas_per_sm(40, 256, 8192, 0, dev) == 6           # the API figures of the row engine and of the reference
    assert dr.ctas_per_sm(128, 256, 120000, 0, dev) == 1                                                        # 121024 B per CTA: one fits, two do not
    assert dr.ctas_per_sm(128, 256, 8192, 67584, dev) == 2 and dr.ctas_per_sm(128, 256, 100000, 40000, dev) == 1
    assert dr.ctas_per_sm(255, 256, 8192, 0, dev) == 1                                                          # the register file allows one CTA of 255 registers per thread


def test_constants_and_the_struct_are_read_from_the_source_text():
    resolve = dr.constants_of(PRIMITIVES, E_LINES)
    assert resolve("HEAD_K") == 128 and resolve("ROWS_E_STRIDE") == 132 and resolve("ROWS_DYN_SMEM_BYTES") == E_BYTES
    assert dr.row_smem_floats(ROWS, resolve) == ROWS_FLOATS
    with pytest.raises(ValueError):
        dr.row_smem_floats(ROWS.replace("float beta;", "int beta;"), resolve)                                   # a field that is not a float is not counted silently
    with pytest.raises(ValueError):
        dr.row_smem_floats("no struct here", resolve)
    for expr in ("__import__('os')", "2 ** 3", "3 / 2", "HEAD_K(2)", "UNKNOWN + 1"):
        with pytest.raises(ValueError):
            dr.eval_constant(expr, resolve)
    assert dr.eval_constant("static_cast<int>(sizeof(float4)) * (3 + 1) // 2", resolve) == 32


def test_no_statement_of_the_record_carries_a_number_or_a_plan_marker():
    for group, statements in dr.STATEMENTS.items():
        for s in statements:
            assert not dr.NUMBER.search(s) and not rc.PLAN_VOCABULARY.search(s) and dr.LOOP_DIR not in s, (group, s)


# ---- the document -----------------------------------------------------------------------------------------------------------------------------
def document_with(rec, number=8, before="# Performance\n\n## 7. Something\ntext\n\n", after="\n## 9. Later\nmore\n"):
    return before + dr.render_section(rec, number) + after


def test_the_section_is_generated_and_the_document_must_equal_it(world):
    rec = world["rec"]
    doc = document_with(rec)
    assert check(world, document=doc) == []
    assert dr.section_bounds(doc) is not None and "## 9. Later" in doc and doc.index("## 8.") < doc.index("## 9.")
    assert dr.replace_section(doc, dr.render_section(rec, 8)) == doc                                              # replacing by the same text changes nothing
    appended = dr.replace_section("# Performance\n\ntext\n", dr.render_section(rec, 8))
    assert appended.count(dr.SECTION_TITLE) == 1 and check(world, document=appended) == []
    rendered = dr.render_section(rec)
    assert "| b200 | 16 | 127.000 |" in rendered                                                                  # the pooled row of the hand-computed example, as the document shows it


@pytest.mark.parametrize("name,edit,expected", [
    ("a number edited in a table", lambda t: t.replace("| b200 | 16 | 127.000 |", "| b200 | 16 | 128.000 |", 1), "not the generated one"),
    ("a sentence added", lambda t: t.replace("### Order of the row-engine steps", "### Order of the row-engine steps\n\nThis was faster by 3 percent.", 1), "not the generated one"),
    ("a table row removed", lambda t: t.replace("| trims |", "| trimsX |", 1), "not the generated one"),
    ("the section dropped", lambda t: t[:t.index("## 8.")] + t[t.index("## 9."):], "no section headed"),
])
def test_a_document_that_is_not_the_generated_section_is_rejected(world, name, edit, expected):
    doc = edit(document_with(world["rec"]))
    assert has(check(world, document=doc), expected), (name, check(world, document=doc)[:2])


def test_a_record_that_lacks_a_section_of_the_derivation_is_a_problem_and_not_a_crash_when_the_document_is_given(world):
    """An older record (without the launch section) next to its document: the document is compared with the derived record, so the missing section is reported and nothing raises."""
    old = mutated(world["rec"], lambda r: r["budget_table"].pop("launch"))
    probs = check(world, rec=old, document=document_with(world["rec"]))
    assert has(probs, "budget_table.launch is missing") and not has(probs, "not the generated one")


# ---- mutants of the record ---------------------------------------------------------------------------------------------------------------------
def set_path(path, value):
    def edit(rec):
        node = rec
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
    return edit


RECORD_MUTANTS = [
    ("a share edited by hand", set_path(["phase_profile", "cells", 0, "share_percent", "prep"], 99.0), "share_percent.prep"),
    ("a spread edited by hand", set_path(["phase_profile", "cells", 3, "row_spread_percent"], 0.0), "row_spread_percent"),
    ("the prefix proxy edited", set_path(["phase_profile", "cells", 1, "prefix_proxy_percent"], 10.0), "prefix_proxy_percent"),
    ("a provenance label edited", set_path(["budget_table", "configurations", "sm_100a", 1, "ctas_per_sm_label"], "api"), "ctas_per_sm_label"),
    ("the A-fragment tile marked as compiled", set_path(["budget_table", "tiles", "a_fragments", "compiled"], True), "compiled"),
    ("the measured carve-out applied to a candidate", set_path(["budget_table", "configurations", "sm_100a", 1, "l1_after_measured_carveout_bytes"], 196608), "l1_after_measured_carveout_bytes"),
    ("a resource figure edited", set_path(["budget_table", "compiled", "sm_100a", 1, "spill_loads"], 0), "spill_loads"),
    ("a gate excess hidden", set_path(["budget_table", "compiled", "sm_100a", 2, "gate_excess"], []), "gate_excess"),
    ("the E tile dynamic bytes edited", set_path(["budget_table", "source_facts", 1, "value"], 60000), "source_facts"),
    ("the carve-out edited", set_path(["budget_table", "carveout", "b200", "shared_mem_config_bytes"], 32768), "shared_mem_config_bytes"),
    ("an alias pair invented", set_path(["budget_table", "lifetimes", "may_alias"], [["a_fragments", "kd_tile"], ["e_tile", "kd_tile"]]), "may_alias"),
    ("the lifetimes marked as tested", set_path(["budget_table", "lifetimes", "tested"], True), "tested"),
    ("the draft's limit edited", set_path(["value_split_inputs", "draft", "nh_up_to"], 200), "nh_up_to"),
    ("a strict share invented", set_path(["value_split_inputs", "strict_v_independent_share"], 20.0), "strict_v_independent_share"),
    ("a statement edited", set_path(["order_basis", "statements", 0], "The order was changed."), "statements"),
    ("an input digest edited", set_path(["inputs", "launch_config", "sha256"], "0" * 64), "sha256"),
    ("the profile commit edited", set_path(["meta", "profile_commit"], "1" * 40), "profile_commit"),
    ("a key added", lambda rec: rec.update(extra={"a": 1}), "extra"),
]


@pytest.mark.parametrize("name,edit,expected", RECORD_MUTANTS, ids=[m[0] for m in RECORD_MUTANTS])
def test_a_record_that_differs_from_the_derivation_is_rejected(world, name, edit, expected):
    probs = check(world, mutated(world["rec"], edit))
    assert has(probs, expected), (name, probs[:3])


STRUCTURE_MUTANTS = [
    ("a reordered step list", lambda rec: rec.update(row_engine_order=["e_tile", "trims", "a_fragments"]), "reorders the steps"),
    ("a step left out without a reason", lambda rec: rec.update(row_engine_order=["trims", "a_fragments"]), "left out of row_engine_order without a reason"),
    ("a missing key", lambda rec: rec.pop("budget_table"), "budget_table"),
    ("another schema", lambda rec: rec.update(schema="cfbi-decision-v0"), "not a cfbi-decision-v1 record"),
    ("a plan marker in the record", lambda rec: rec["order_basis"]["targets"].update(trims=["prep", "dir-0" + "2"]), "plan marker"),
    ("a path of the loop in the record", lambda rec: rec["inputs"]["launch_config"].update(name=dr.LOOP_DIR + "/records/x.json"), "path of the loop"),
]


@pytest.mark.parametrize("name,edit,expected", STRUCTURE_MUTANTS, ids=[m[0] for m in STRUCTURE_MUTANTS])
def test_a_malformed_record_is_rejected(world, name, edit, expected):
    probs = check(world, mutated(world["rec"], edit))
    assert has(probs, expected), (name, probs[:3])


def test_the_commits_of_the_record_are_bound_to_the_repository(world):
    c = world["commits"]
    assert has(check(world, mutated(world["rec"], set_path(["meta", "recorded_on"], "f" * 40))), "does not have"), "a commit that the repository does not have"
    assert has(check(world, head=c["profile"]), "is not an ancestor of"), "a record written after HEAD"
    other = mutated(world["rec"], set_path(["meta", "recorded_on"], c["e"]))                                       # the prototype's commit is not an ancestor of the loop's
    assert has(check(world, other), "is not an ancestor of")
    assert has(dr.record_problems(copy.deepcopy(world["rec"]), world["ev"], None), "repo is required")


# ---- mutants of the evidence -------------------------------------------------------------------------------------------------------------------
def test_a_profile_record_that_was_edited_changes_the_derived_record_or_fails_its_own_checks(world):
    ev = reloaded(world, {("profile", 0): lambda rec: [c["rows"]["regular"]["phases"].update(prep=c["rows"]["regular"]["phases"]["prep"] * 2) for c in rec["cells"]]})
    probs = check(world, ev=ev)
    assert has(probs, "profile record") and has(probs, "prep"), probs[:3]                                              # the edited record fails the profile checker's own rules


def test_the_profile_set_must_have_three_processes_per_device_of_one_commit(world):
    ev = copy.copy(world["ev"])
    ev["profiles"] = world["ev"]["profiles"][1:]
    assert has(check(world, ev=ev), "3 (one per process) are required")
    ev = copy.copy(world["ev"])
    ev["profiles"] = [r for r in world["ev"]["profiles"] if dr.device_of(r) == "b200"]
    probs = check(world, ev=ev)
    assert has(probs, "no profile record for B300") or has(probs, "profile record(s) of b300")
    other = reloaded(world, {("profile", 2): lambda rec: rec["meta"].update(commit="0badc0de0000")})
    assert has(check(world, ev=other), "different commits")


@pytest.mark.parametrize("name,files,expected", [
    ("a report that is not one of the Nsight Compute reports", {("launch_config", None): lambda rec: rec["reports"][0].update(report_sha256="0" * 64)}, "is not one of the reports"),
    ("a launch report that is missing", {("launch_config", None): lambda rec: rec["reports"].pop()}, "appears 0 time(s)"),
    ("a launch of another device", {("launch_config", None): lambda rec: rec["reports"][0].update(device="b300")}, "the entry says"),
    ("two carve-outs on one device", {("launch_config", None): lambda rec: rec["reports"][0]["launch"].update(shared_mem_config_size=32768)}, "carve-out values"),
    ("a launch that is not the record's", {("launch_config", None): lambda rec: rec["reports"][0]["launch"].update(grid_size=7)}, "grid_size"),
    ("no carve-out", {("launch_config", None): lambda rec: rec["reports"][1]["launch"].pop("shared_mem_config_size")}, "no shared-memory carve-out"),
    ("another schema", {("launch_config", None): lambda rec: rec.update(schema="x")}, "not a cfbi-launch-config-v1 record"),
])
def test_the_launch_metrics_are_bound_to_the_nsight_compute_records(world, name, files, expected):
    probs = check(world, ev=reloaded(world, files))
    assert has(probs, expected), (name, probs[:3])


@pytest.mark.parametrize("name,edit,expected", [
    ("one CTA per SM", lambda rec: rec["occupancy"]["instances"][0].update(ctas_per_sm=1), "at least 2 CTAs"),
    ("a CTA of other threads", lambda rec: rec["occupancy"]["instances"][0].update(threads=128, ctas_per_sm=4), "256 are required"),
    ("registers that differ from the dump", lambda rec: rec["occupancy"]["instances"][0].update(registers=96), "the production dump"),
    ("more registers than the gate allows", lambda rec: rec["occupancy"]["instances"][0].update(registers=136, ctas_per_sm=1), "at most 128 registers"),
    ("a figure that the arithmetic does not reproduce", lambda rec: rec["occupancy"]["instances"][1].update(ctas_per_sm=3), "the arithmetic"),
    ("the record of another GPU", lambda rec: rec.update(gpu="NVIDIA H100"), "is not b200"),
    ("an SM count that is not the device's own", lambda rec: rec.update(sm_count=132), "SM count"),
    ("a label without the commit", lambda rec: rec.update(label="b200"), "does not name the device and the commit"),
    ("static shared memory that differs from the dump", lambda rec: rec["occupancy"]["instances"][0].update(static_smem=7680), "static shared memory"),
    ("an instantiation that is missing", lambda rec: rec["occupancy"]["instances"].pop(1), "is not in the record"),
])
def test_the_occupancy_record_is_bound_to_its_device_commit_instantiation_and_shared_memory(world, name, edit, expected):
    probs = check(world, ev=reloaded(world, {("occupancy", "b200"): edit}))
    assert has(probs, expected), (name, probs[:3])


def toolchain_edit(**fields):
    return lambda d: d["meta"]["toolchain"].update(fields)


@pytest.mark.parametrize("name,files,expected", [
    ("a prototype dump with another toolchain label", {("compile", "chunk-end-row-offload"): toolchain_edit(label="another")}, "different toolchains"),
    ("a prototype dump of the same label with other flags (review control)", {("compile", "chunk-end-row-offload"): toolchain_edit(flags=["-O0"])}, "different toolchains"),
    ("a prototype dump of the same label with another nvcc (review control)", {("compile", "chunk-end-row-offload"): toolchain_edit(nvcc="CUDA 12.9")}, "different toolchains"),
    ("another host compiler", {("compile", "launch-chain-collapse"): toolchain_edit(host_compiler="g++ 11")}, "different toolchains"),
    ("another ptxas", {("compile", "operand-swapped-mma-tiles"): toolchain_edit(ptxas="ptxas 12.9")}, "different toolchains"),
    ("another cuobjdump", {("compile", "value-split-decode-ctas"): toolchain_edit(cuobjdump="cuobjdump 12.9")}, "different toolchains"),
    ("another language standard", {("compile", "coalesced-state-access-path"): toolchain_edit(std="c++17")}, "different toolchains"),
    ("a base dump of another toolchain than every prototype", {("compile", "base"): toolchain_edit(flags=["-O2"])}, "different toolchains"),
    ("a dump that names an architecture that it has no kernels for", {("compile", "base"): lambda d: d["meta"]["toolchain"]["archs"].append("sm_120a")}, "no kernels for sm_120a"),
    ("a dump that lacks an architecture", {("compile", "base"): lambda d: d["kernels"].pop("sm_103a")}, "no kernels for sm_103a"),
    ("a dump without a spill figure (review control)", {("compile", "value-split-decode-ctas"): lambda d: d["kernels"]["sm_100a"]["cfbi::rows_kernel<Li32E>"].pop("spill_loads")}, "incomplete evidence"),
    ("a base dump without the launch bounds of the row engine", {("compile", "base"): lambda d: d["kernels"]["sm_100a"]["cfbi::rows_kernel"].update(max_threads=None)}, "launch-bound"),
    ("a production dump without a figure", {("container", "b200"): lambda d: d["kernels"]["sm_100a"]["cfbi::rows_kernel"].pop("stack")}, "incomplete evidence"),
    ("a dump that lacks an instantiation", {("compile", "value-split-decode-ctas"): lambda d: d["kernels"]["sm_100a"].pop("cfbi::rows_kernel<Li32E>")}, "has no instantiation"),
    ("a dump that names no commit", {("compile", "launch-chain-collapse"): lambda d: d["meta"]["tree"].update(commit=None)}, "names no commit"),
    ("a production dump of another commit", {("container", "b200"): lambda d: d["meta"]["tree"].update(commit="1234567abcdef")}, "the profile is of"),
    ("a production dump made with the conda toolchain", {("container", "b300"): toolchain_edit(label="conda 13.1")}, "container toolchain"),
    ("a production dump whose figures differ from the API's", {("container", "b300"): lambda d: d["kernels"]["sm_103a"]["cfbi::rows_kernel"].update(reg=112)}, "the production dump"),
])
def test_the_compile_dumps_are_complete_one_toolchain_and_the_production_dumps_the_profiled_commit(world, name, files, expected):
    probs = check(world, ev=reloaded(world, files))
    assert has(probs, expected), (name, probs[:3])


def test_a_prototype_that_breaks_the_resource_gate_is_data_and_the_container_toolchain_may_differ_from_the_conda_one(world):
    """The checks of the dumps do not reject the analysis because a prototype is over the base's resources, and the production dumps are not compared with the development toolchain."""
    gates = [r["gate_excess"] for r in world["rec"]["budget_table"]["compiled"]["sm_100a"] if r["id"] != "base"]
    assert any(gates) and check(world) == []
    assert world["ev"]["container"]["b200"]["meta"]["toolchain"]["label"] != world["ev"]["compile"]["base"]["meta"]["toolchain"]["label"]


# ---- B14: the occupancy figures are the figures of the launch --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dynamic", [1, 32768])
def test_an_occupancy_dynamic_shared_memory_that_the_launch_does_not_use_is_rejected(world, dynamic):
    """The review's control: the CTAs per SM stay 2 (the register file allows two whatever the dynamic memory), so the arithmetic alone cannot see which configuration they are the count of."""
    probs = check(world, ev=reloaded(world, {("occupancy", "b200"): lambda rec: rec["occupancy"]["instances"][0].update(dynamic_smem=dynamic)}))
    assert has(probs, "the launch in the sources at the record's commit uses 256 threads and 0 B"), probs[:3]
    assert has(probs, "says shared_mem_per_block_dynamic = 0"), probs[:3]                                              # the retained launches say 0 as well


def test_the_launch_of_the_sources_at_the_records_commit_is_the_configuration_that_counts(world):
    c = world["commits"]
    other = reloaded(world, {("occupancy", "b200"): lambda rec: rec.update(label="b200-" + c["cfg"][:9])})            # the launch of that commit has dynamic shared memory
    assert has(check(world, ev=other), "uses 256 threads and 4096 B"), "a record of another launch configuration"
    nolaunch = reloaded(world, {("occupancy", "b200"): lambda rec: rec.update(label="b200-" + c["e"][:9])})           # a commit that has no launcher
    assert has(check(world, ev=nolaunch), "the launch cannot be read from the sources")


@pytest.mark.parametrize("field,value", [("shared_mem_per_block_static", 4096), ("shared_mem_per_block_driver", 512), ("shared_mem_per_block_dynamic", 1), ("block_size", 128),
                                         ("registers_per_thread", 96)])
def test_the_retained_launch_is_checked_figure_by_figure(world, field, value):
    probs = check(world, ev=reloaded(world, {("launch_config", None): lambda rec: rec["reports"][0]["launch"].update({field: value})}))
    assert has(probs, "says %s = %r" % (field, value)), (field, probs[:3])


def test_the_two_devices_describe_one_launch_of_the_row_engine(world):
    probs = check(world, ev=reloaded(world, {("occupancy", "b300"): lambda rec: rec["occupancy"]["instances"][0].update(dynamic_smem=1)}))
    assert has(probs, "describe different launches")


def test_the_base_configuration_is_derived_from_the_validated_launch(world):
    bt = world["rec"]["budget_table"]
    assert bt["launch"]["b200"] == {"commit": world["commits"]["occ_b200"][:9], "path": dr.LAUNCH_SOURCE, "threads": 256, "dynamic_smem": 0, "label": "source"}
    base = bt["configurations"]["sm_100a"][0]
    assert base["id"] == "base" and base["dynamic_bytes"] == 0 and base["per_cta_bytes"] == 8192 + 1024

    def occupancy(rec):                                                         # a legitimate other launch: dynamic shared memory 4096 in the sources, in the API and in the retained launches
        rec["label"] = rec["label"].split("-")[0] + "-" + world["commits"]["cfg"][:9]
        for inst in rec["occupancy"]["instances"]:
            inst["dynamic_smem"] = 4096

    def retained(rec):
        for r in rec["reports"]:
            r["launch"]["shared_mem_per_block_dynamic"] = 4096
    ev = reloaded(world, {("occupancy", "b200"): occupancy, ("occupancy", "b300"): occupancy, ("launch_config", None): retained})
    rec = dr.build_record(ev, world["commits"]["head"], world["repo"])
    assert dr.record_problems(copy.deepcopy(rec), ev, world["repo"], None, world["commits"]["head"]) == []
    moved = dict((r["id"], r) for r in rec["budget_table"]["configurations"]["sm_100a"])["base"]
    assert moved["dynamic_bytes"] == 4096 and moved["per_cta_bytes"] == 8192 + 4096 + 1024                              # nothing is hard-coded to 0


def test_the_decision_command_rejects_the_review_controls_of_the_evidence(world, tmp_path, capsys):
    """B13 and B14 at the command line: the same label with other flags, and the base's dynamic shared memory written as 32 KiB: exit 1 and VERDICT: FAIL (both were exit 0 and PASS)."""
    rec_path, doc_path = tmp_path / "record.json", tmp_path / "perf.md"
    rec_path.write_text(json.dumps(world["rec"]))
    doc_path.write_text(dr.render_section(world["rec"]))
    args = ["decision", "--record", str(rec_path), "--evidence", str(world["manifest"]), "--repo", str(world["repo"]), "--document", str(doc_path), "--head", world["commits"]["head"]]
    assert rc.main(args) == 0
    capsys.readouterr()
    for name, files, expected in (("flags", {("compile", "chunk-end-row-offload"): toolchain_edit(flags=["-O0"])}, "different toolchains"),
                                  ("dynamic", {("occupancy", "b200"): lambda rec: rec["occupancy"]["instances"][0].update(dynamic_smem=32768)}, "the launch in the sources")):
        with edited(world, files):
            assert rc.main(args) == 1
        out = capsys.readouterr().out
        assert "VERDICT: FAIL" in out and expected in out, (name, out[:300])


def test_a_figure_of_a_dump_moves_the_budget_table(world):
    ev = reloaded(world, {("compile", "row-critical-path-re-pipelining"): lambda d: d["kernels"]["sm_100a"]["cfbi::rows_kernel"].update(spill_loads=0, spill_stores=0)})
    probs = check(world, ev=ev)
    assert has(probs, "gate_excess") and has(probs, "spill_loads")                                                # the record says what it said, the evidence now says something else


@pytest.mark.parametrize("figure", ["threads", "registers"])
def test_an_occupancy_figure_of_zero_is_a_problem_and_not_a_crash(world, figure):
    probs = check(world, ev=reloaded(world, {("occupancy", "b200"): lambda rec: rec["occupancy"]["instances"][0].update({figure: 0})}))
    assert has(probs, "the registers and the threads positive"), probs[:3]


def test_a_constant_that_divides_by_zero_or_is_circular_is_not_a_constant():
    with pytest.raises(ValueError, match="division by zero"):
        dr.eval_constant("4 // (2 - 2)", dr.constants_of(PRIMITIVES))
    with pytest.raises(ValueError, match="in terms of itself"):
        dr.constants_of("constexpr int A = B + 1;\nconstexpr int B = A + 1;\n")("A")
    with pytest.raises(ValueError, match="not defined"):
        dr.constants_of(PRIMITIVES)("NOT_THERE")


def test_the_diff_of_a_record_stops_at_its_limit():
    assert len(dr.diff(dict(("k%d" % i, i) for i in range(20)), {}, limit=6)) == 6
    assert dr.diff({"a": [1, 2]}, {"a": [1, 3]}) == ["record.a[1] is 3, the evidence gives 2"]


def test_a_launch_entry_that_is_not_an_object_or_a_report_named_twice_is_a_problem(world):
    probs = check(world, ev=reloaded(world, {("launch_config", None): lambda rec: rec["reports"].append("not an object")}))
    assert has(probs, "is not an object")
    ev = copy.deepcopy(world["ev"])
    ev["ncu"][1]["meta"]["report"]["sha256"] = ev["ncu"][0]["meta"]["report"]["sha256"]                       # two Nsight Compute records name the same report
    twice = ev
    assert has(check(world, ev=twice), "every report is named once")


def test_a_source_that_changes_moves_the_tile_sizes_and_an_unreadable_source_is_a_problem(world, tmp_path):
    ev = reloaded(world, {("compile", "coalesced-state-access-path"): lambda d: d["meta"]["tree"].update(commit=world["commits"]["e2"])})
    probs = check(world, ev=ev)
    assert has(probs, "the evidence gives 80896"), probs[:6]                                                         # 8192 + 128 * 140 * 4 + 1024 B per CTA: the record still says 76800
    elsewhere = tmp_path / "other"
    elsewhere.mkdir()
    git(elsewhere, "init", "-q", "-b", "main")
    commit_file(elsewhere, "csrc/cfbi/cfbi_primitives.cuh", PRIMITIVES, "an unrelated history")
    assert has(check(world, repo=elsewhere), "the launch cannot be read from the sources")                          # the commits of the evidence are not in this repository
    with pytest.raises(ValueError):
        dr.source_facts(elsewhere, world["commits"]["base"], world["commits"]["e"], world["commits"]["kd"])


HEADER_NEED = "constexpr int ROWS_STATIC_SMEM_BYTES = 8192;\nconstexpr int ROWS_DYNAMIC_SMEM_BYTES = %d;\nconstexpr int ROWS_SMEM_NEED_BYTES = ROWS_STATIC_SMEM_BYTES + ROWS_DYNAMIC_SMEM_BYTES;\n"
LAUNCH_FROM_HEADER = "constexpr size_t ROWS_DYNAMIC_SMEM = ROWS_DYNAMIC_SMEM_BYTES;   // read from the CUDA-free header\n"


def test_the_launch_constants_follow_the_launcher_into_the_header(tmp_path):
    """The launcher may name the header's constant instead of a number: the dynamic shared memory is then the header's at that commit, and a launcher that names a constant that the
    commit does not define cannot be read (never read as zero)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    commit_file(repo, "csrc/cfbi/cfbi_primitives.cuh", PRIMITIVES, "the sizes of a head")
    literal = commit_file(repo, "csrc/cfbi/cfbi_launch.cu", LAUNCH, "a launcher with a number")
    aliased = commit_file(repo, "csrc/cfbi/cfbi_launch.cu", LAUNCH_FROM_HEADER, "a launcher that names the header's constant, no header yet")
    zero = commit_file(repo, "csrc/cfbi/cfbi_policy.h", HEADER_NEED % 0, "the header")
    four = commit_file(repo, "csrc/cfbi/cfbi_policy.h", HEADER_NEED % 4096, "the header with dynamic shared memory")
    assert dr.launch_constants(repo, literal) == {"threads": 256, "dynamic_smem": 0}                                    # the earlier commits read as before
    with pytest.raises(ValueError, match="ROWS_DYNAMIC_SMEM_BYTES is not defined"):
        dr.launch_constants(repo, aliased)
    assert dr.launch_constants(repo, zero) == {"threads": 256, "dynamic_smem": 0}
    assert dr.launch_constants(repo, four) == {"threads": 256, "dynamic_smem": 4096}                                    # nothing is hard-coded to 0


# ---- the command lines -----------------------------------------------------------------------------------------------------------------------------
def test_the_command_lines_build_render_and_check(world, tmp_path, capsys):
    out, doc = tmp_path / "record.json", tmp_path / "perf.md"
    doc.write_text("# Performance\n\n## 7. Something\ntext\n")
    manifest, repo = str(world["manifest"]), str(world["repo"])
    assert dr.main(["build", "--evidence", manifest, "--repo", repo, "--out", str(out), "--recorded-on", world["commits"]["head"]]) == 0
    assert json.loads(out.read_text()) == json.loads(json.dumps(world["rec"]))                                     # the command writes the record that the module derives
    assert dr.main(["render", "--record", str(out), "--document", str(doc)]) == 0
    capsys.readouterr()
    assert dr.main(["check", "--record", str(out), "--evidence", manifest, "--repo", repo, "--document", str(doc)]) == 0
    assert "VERDICT: PASS" in capsys.readouterr().out
    # the same check through the record checker's command
    assert rc.main(["decision", "--record", str(out), "--evidence", manifest, "--repo", repo, "--document", str(doc)]) == 0
    capsys.readouterr()
    doc.write_text(doc.read_text().replace("| 127.000 |", "| 128.000 |", 1))
    assert rc.main(["decision", "--record", str(out), "--evidence", manifest, "--repo", repo, "--document", str(doc)]) == 1
    assert "not the generated one" in capsys.readouterr().out
    bad = json.loads(out.read_text())
    bad["phase_profile"]["cells"][0]["share_percent"]["prep"] = 1.0
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    assert dr.main(["check", "--record", str(tmp_path / "bad.json"), "--evidence", manifest, "--repo", repo]) == 1
    assert "share_percent.prep" in capsys.readouterr().out
    assert dr.main(["check", "--record", str(tmp_path / "missing.json"), "--evidence", manifest, "--repo", repo]) == 2
    assert rc.main(["decision", "--record", str(out)]) == 0                                                         # without evidence only the structure is checked, as before
