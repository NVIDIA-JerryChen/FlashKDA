"""Self-tests of the start order of the record files of an A/B comparison (strict alternation or ABBA blocks), of the launch mapping that a campaign can require of its records, and of the meta
fields and cell arguments that bind a re-run to its first campaign (benchmarks/bench_records.py, benchmarks/compare_bench.py).

Plain Python, no GPU and no torch (``pytest --noconftest tests/cfbi/test_bench_order.py``). The record sets are built from a label string: the n-th letter is the build (A baseline, B candidate)
of the n-th record file in start order, so "ABBAABBA" is a campaign of four files per build that ran in two ABBA blocks.
"""
import itertools
import json
import os
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "benchmarks", _ROOT / "tests" / "cfbi"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bench_records as br  # noqa: E402
import compare_bench  # noqa: E402
import test_bench_tools as tbt  # noqa: E402

KEY = tbt.WALL_KEY
FIRST = "first campaign"
MEDIANS = [100.0, 100.05, 99.95, 100.02, 99.98, 100.03, 99.97, 100.01, 99.99, 100.04]            # per-process medians of one build: a spread of 0.1 %, stable
UNSTABLE_A = [100.0, 100.45, 100.9, 100.45, 100.2, 100.7]                                          # a spread of 0.9 %: the cell needs a re-run
UNSTABLE_B = [100.15, 100.6, 101.05, 100.6, 100.35, 100.85]
RERUN_A = [100.0, 100.05, 99.95, 100.02, 99.98, 100.03]
RERUN_B = [100.1, 100.15, 100.05, 100.12, 100.08, 100.13]
TWO, ONE = {"plan_kernel": 1, "rows_kernel": 1}, {"rows_kernel": 1}                                # the two-launch and the single-launch decode, as the benchmark records them


def stamp(slot, hour="10"):
    """The start stamp of the ``slot``-th run of a campaign, one minute apart."""
    return "2026-11-10T%s:%02d:00Z" % (hour, slot)


def ordered_sets(labels, hour="10", scale=1, keys=(KEY,), meds_a=None, meds_b=None, cand_commit="abc1234", **kw):
    """Record lists ``(A, B)`` whose files started in the order of the label string ``labels``; the n-th file of a build gets the n-th median of ``meds_a`` / ``meds_b``."""
    meds, out = {"A": list(meds_a or MEDIANS), "B": list(meds_b or MEDIANS)}, {"A": [], "B": []}
    for slot, side in enumerate(labels):
        commit = tbt.BASE_COMMIT if side == "A" else cand_commit
        out[side].append(tbt.wall_rec(commit, {k: tbt.trials(meds[side][len(out[side])]) for k in keys}, stamp(slot, hour), scale=scale, **kw))
    return out["A"], out["B"]


def compare(base, cand, order="alternate", **kw):
    """``br.compare`` of the gpu_graph_us metric; the default order is not passed, so that the strict-alternation cases call the comparator exactly as its older callers do."""
    if order != "alternate":
        kw["order"] = order
    return br.compare(base, cand, metric="gpu_graph_us", **kw)


def switches(labels):
    return sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])


def has(problems, *words):
    return any(all(w in p for w in words) for p in problems)


def with_nodes(records, compute, aux=None, **args):
    """Copies of ``records`` in which every cell carries the launch mapping ``compute`` (and ``aux``) in its arguments and in every trial, as the benchmark writes them."""
    out = json.loads(json.dumps(records))
    for r in out:
        for cell in r["cells"]:
            cell["args"].update(compute_nodes=dict(compute), aux_nodes=dict(aux or {}), compute_kernels=sum(compute.values()), expect_compute=dict(compute), **args)
            for s in cell["samples"]:
                s["nodes"] = {"compute": dict(compute), "aux": dict(aux or {})}
    return out


def set_meta(records, **fields):
    for r in records:
        r["meta"].update(fields)
    return records


def set_args(records, **fields):
    for r in records:
        for cell in r["cells"]:
            cell["args"].update(fields)
    return records


GONE = object()                                                                                     # edit_fields: remove the field instead of setting it


def edit_fields(records, meta=None, args=None):
    """Set (or, for ``GONE``, remove) fields of the meta and of the cell arguments of every record."""
    for r in records:
        for field, value in (meta or {}).items():
            if value is GONE:
                r["meta"].pop(field, None)
            else:
                r["meta"][field] = value
        for cell in r["cells"]:
            for field, value in (args or {}).items():
                if value is GONE:
                    cell["args"].pop(field, None)
                else:
                    cell["args"][field] = value


# ---- the order contract: ABBA blocks ---------------------------------------------------------------------------------------------------------

def test_the_review_counterexample_abba_blocks_pass_under_the_abba_order_and_stay_invalid_under_alternate():
    base, cand = ordered_sets("ABBAABBA")
    res = compare(base, cand)                                                                       # the default order is strict alternation
    assert res["verdict"] == "INVALID" and has(res["problems"], "not interleaved", "4 variant switches, 7 needed") and res["order"] == "alternate"
    res = compare(base, cand, "abba")
    assert res["verdict"] == "PASS" and res["problems"] == [] and res["order"] == "abba" and res["rows"][0]["status"] == "PASS"
    strict_b, strict_c = ordered_sets("ABABABAB")                                                  # strict alternation is not an ABBA campaign
    res = compare(strict_b, strict_c, "abba")
    assert res["verdict"] == "INVALID" and has(res["problems"], "ABBA or BAAB", "ABABABAB") and compare(strict_b, strict_c)["verdict"] == "PASS"


ABBA_4 = ("ABBAABBA", "ABBABAAB", "BAABABBA", "BAABBAAB")                                           # the four campaigns of two blocks


def test_abba_is_a_block_grammar_not_a_switch_count():
    for bad in ("AAAABBBB", "BBBBAAAA", "ABABABAB", "ABBAABAB", "AABBABBA", "ABBABBAA", "ABBAABB", "ABBAABBAA", "ABBAAAAB", "BAABBBBA"):   # the last four: seven and nine files, 5 A and 3 B, 3 A and 5 B
        base, cand = ordered_sets(bad)
        res = compare(base, cand, "abba")
        assert res["verdict"] == "INVALID" and has(res["problems"], "ABBA or BAAB", bad, "%d variant switches" % switches(bad)), bad
    base, cand = ordered_sets("ABBAABBA")
    assert br.order_problems(base, cand, "abba", FIRST) == []
    for blocks in (2, 3, 4):                                                                        # every mix of ABBA and BAAB blocks passes
        for mix in itertools.product(br.ABBA_BLOCKS, repeat=blocks):
            base, cand = ordered_sets("".join(mix))
            res = compare(base, cand, "abba")
            assert res["verdict"] == "PASS" and br.order_problems(base, cand, "abba", FIRST) == [], mix
    one = ordered_sets("ABBA")                                                                      # one block is a sentence of the grammar; how many files a build needs is min_repeats' business
    assert br.order_problems(*one, "abba", FIRST) == [] and has(compare(*one, "abba")["problems"], "at least 3")
    assert compare(*one, "abba", min_repeats=2)["verdict"] == "PASS"
    uneven = ordered_sets("ABBAABBAA")                                                              # four files of one build and five of the other
    assert has(br.order_problems(*uneven, "abba", FIRST), "5 baseline and 4 candidate")
    strings = sorted(set("".join(p) for p in itertools.permutations("AAAABBBB")))
    assert len(strings) == 70
    valid = [s for s in strings if not br.order_problems(*ordered_sets(s), "abba", FIRST)]
    assert valid == sorted(ABBA_4)
    many = [s for s in strings if switches(s) >= 4]
    assert len(many) == 44 and len([s for s in many if s not in valid]) == 40                       # a lowered switch threshold would accept 40 strings that are no ABBA blocks
    assert [s for s in strings if switches(s) >= 7] == ["ABABABAB", "BABABABA"]                    # what the alternate order accepts of them


def test_the_abba_message_names_the_first_bad_block_the_label_string_and_its_switches():
    msg = br.order_problems(*ordered_sets("ABBAABAB"), "abba", FIRST)
    assert msg == ["baseline and candidate records of the first campaign are not in blocks of ABBA or BAAB in time (start order ABBAABAB, 5 variant switches): "
                   "the block of record files 5 to 8 is ABAB"]
    assert has(br.order_problems(*ordered_sets("ABBABBAA"), "abba", "re-run"), "of the re-run", "the block of record files 5 to 8 is BBAA")
    assert not has(br.order_problems(*ordered_sets("ABBAABBAA"), "abba", FIRST), "9 record files")  # unequal counts are named before the length
    assert has(br.order_problems(*ordered_sets("ABBAAB"), "abba", FIRST), "6 record files are not a whole number of blocks of four")
    assert br.order_problems(*ordered_sets("ABBAABBA"), "bogus", FIRST) == ["order must be one of alternate, abba, got 'bogus'"]


def test_the_alternate_order_keeps_the_switch_rule_for_every_label_sequence():
    """The strict-alternation rule and its message: a campaign is interleaved iff its label string has at least 2 * min(files) - 1 switches (a guard: this rule must not move)."""
    checked = 0
    for n_a, n_b in ((3, 3), (4, 4), (4, 3), (3, 4)):
        for places in itertools.combinations(range(n_a + n_b), n_a):
            labels = "".join("A" if i in places else "B" for i in range(n_a + n_b))
            need, got = 2 * min(n_a, n_b) - 1, switches(labels)
            res = compare(*ordered_sets(labels))
            if got >= need:
                assert res["verdict"] == "PASS", labels
            else:
                assert res["verdict"] == "INVALID" and has(res["problems"], "baseline and candidate records of the first campaign are not interleaved in time (%d variant switches, %d needed)" % (got, need)), labels
            checked += 1
    assert checked == 20 + 70 + 35 + 35


def test_the_rerun_is_validated_against_the_same_order_as_the_first_campaign():
    first = ordered_sets("ABBAABBA", meds_a=UNSTABLE_A, meds_b=UNSTABLE_B)

    def run(labels, order, first=first):
        rerun_b, rerun_c = ordered_sets(labels, hour="11", scale=3, meds_a=RERUN_A, meds_b=RERUN_B)
        return compare(*first, order, baseline_unstable="widen", base_rerun=rerun_b, cand_rerun=rerun_c)
    ok = run("ABBABAAB", "abba")
    assert ok["verdict"] == "PASS" and ok["rows"][0]["judged_on"] == "rerun" and ok["order"] == "abba"
    for labels in ("ABABABAB", "AAAABBBB", "ABBAABAB", "BAABBAAA"):                                    # a re-run that is not in ABBA blocks, whatever it is otherwise
        res = run(labels, "abba")
        assert res["verdict"] == "INVALID" and has(res["problems"], "of the re-run", "ABBA or BAAB") and not has(res["problems"], "of the first campaign"), labels
    strict = ordered_sets("ABABABAB", meds_a=UNSTABLE_A, meds_b=UNSTABLE_B)                          # under the alternate order the same holds the other way round
    assert run("ABABABAB", "alternate", first=strict)["verdict"] == "PASS"
    res = run("ABBAABBA", "alternate", first=strict)
    assert res["verdict"] == "INVALID" and has(res["problems"], "of the re-run", "not interleaved") and not has(res["problems"], "of the first campaign")
    res = run("ABABABAB", "alternate")                                                              # the first campaign is the grouped one here
    assert res["verdict"] == "INVALID" and has(res["problems"], "of the first campaign", "not interleaved") and not has(res["problems"], "of the re-run")
    res = run("ABBABAAB", "bogus")
    assert res["verdict"] == "INVALID" and has(res["problems"], "order must be one of") and res["order"] == "bogus"


def test_run_plan_is_accepted_by_its_own_validator_and_balances_the_sides():
    assert br.run_plan(4, "abba") == [("A", 1), ("B", 1), ("B", 2), ("A", 2), ("B", 3), ("A", 3), ("A", 4), ("B", 4)]
    assert br.run_plan(3, "alternate") == [("A", 1), ("B", 1), ("A", 2), ("B", 2), ("A", 3), ("B", 3)] and br.run_plan(3) == br.run_plan(3, "alternate")
    for n in (4, 6, 8, 10, 12, 16):
        plan = br.run_plan(n, "abba")
        labels = "".join(side for side, _ in plan)
        assert labels.count("A") == labels.count("B") == n
        assert [i for side, i in plan if side == "A"] == list(range(1, n + 1)) == [i for side, i in plan if side == "B"]      # indices count from 1 in run order within a build
        base, cand = ordered_sets(labels, meds_a=MEDIANS * 2, meds_b=MEDIANS * 2)
        assert br.run_labels(base, cand) == labels and br.order_problems(base, cand, "abba", FIRST) == []
        assert compare(base, cand, "abba")["verdict"] == "PASS" and has(br.order_problems(base, cand, "alternate", FIRST), "not interleaved")
        slots = {side: [slot for slot, (s, _) in enumerate(plan) if s == side] for side in "AB"}
        assert sum(slots["A"]) == sum(slots["B"])                                                  # the same mean position in the campaign
        squares = sum(slot * slot for slot in slots["A"]) - sum(slot * slot for slot in slots["B"])
        assert squares == (0 if n % 4 == 0 else 4), n                                              # and the same mean squared position for whole pairs of blocks: one unpaired block leaves exactly 4
        strict = br.run_plan(n, "alternate")
        assert [s for s, _ in strict] == ["A", "B"] * n and not br.order_problems(*ordered_sets("AB" * n, meds_a=MEDIANS * 2, meds_b=MEDIANS * 2), "alternate", FIRST)
        assert sum(i for i, (s, _) in enumerate(strict) if s == "B") - sum(i for i, (s, _) in enumerate(strict) if s == "A") == n   # strict alternation: the baseline always runs first
    for bad in (0, 1, 2, 3, 5, 7, -4, 4.0, True, "4", None):
        with pytest.raises(ValueError):
            br.run_plan(bad, "abba")
    for bad in (0, -1, 2.0, True, None):
        with pytest.raises(ValueError):
            br.run_plan(bad, "alternate")
    for order in ("bogus", "", None, "ABBA"):
        with pytest.raises(ValueError):
            br.run_plan(4, order)


def test_compare_bench_order_option_exit_codes(tmp_path):
    d = str(tmp_path)
    abba_b, abba_c = ordered_sets("ABBABAAB")
    strict_b, strict_c = ordered_sets("ABABABAB")
    grouped_b, grouped_c = ordered_sets("AAAABBBB")

    def files(name, pair):
        return ["--base"] + tbt.write_records(d, name + "a", pair[0]) + ["--cand"] + tbt.write_records(d, name + "b", pair[1]) + ["--metric", "gpu_graph_us"]
    abba, strict, grouped = files("abba", (abba_b, abba_c)), files("strict", (strict_b, strict_c)), files("grouped", (grouped_b, grouped_c))
    out = os.path.join(d, "out.json")
    assert compare_bench.main(abba + ["--order", "abba", "--json", out]) == 0
    assert json.load(open(out))[0]["order"] == "abba" and json.load(open(out))[0]["verdict"] == "PASS"
    assert compare_bench.main(abba) == 2 and compare_bench.main(abba + ["--order", "alternate"]) == 2
    assert compare_bench.main(strict + ["--json", out]) == 0 and json.load(open(out))[0]["order"] == "alternate"
    assert compare_bench.main(strict + ["--order", "abba"]) == 2
    assert compare_bench.main(grouped + ["--order", "abba"]) == 2 and compare_bench.main(grouped) == 2
    assert compare_bench.main(grouped + ["--no-interleave-check"]) == 0 and compare_bench.main(grouped + ["--no-interleave-check", "--order", "abba"]) == 0      # the diagnostic skips the order
    for bogus in ("bogus", "ABBA", ""):
        with pytest.raises(SystemExit) as e:
            compare_bench.main(abba + ["--order", bogus])
        assert e.value.code == 2
    assert "--order" in compare_bench.__doc__ and "--require-nodes" in compare_bench.__doc__


def test_no_tool_but_the_comparator_and_its_command_line_reaches_the_interleaving_diagnostic():
    """``require_interleaved=False`` (``--no-interleave-check``) skips the start order of a comparison: a diagnostic of the person at the keyboard, never something that a checker or a driver passes."""
    for path in sorted((_ROOT / "benchmarks").glob("*.py")):
        if path.name in ("bench_records.py", "compare_bench.py"):
            continue
        text = path.read_text()
        assert "require_interleaved" not in text and "no-interleave-check" not in text, path.name


# ---- the same record file on both sides, ties, missing and unparsable start stamps -----------------------------------------------------------

def written(tmp_path, n=4, name="run"):
    """``n`` record files of one build in a directory, one minute apart; returns their paths."""
    base, _ = ordered_sets("A" * n)
    return tbt.write_records(str(tmp_path), name, base)


def test_the_same_record_files_listed_for_both_sides_are_not_an_interleaving(tmp_path):
    paths = written(tmp_path)
    for n in (3, 4):                                                                                # three files: the review's listing; four files: enough for the abba order
        base, cand = [br.load_record(p) for p in paths[:n]], [br.load_record(p) for p in paths[:n]]
        res = compare(base, cand)                                                                   # sorting equal stamps A before B made this a perfect alternation: five of five switches
        assert res["verdict"] == "INVALID" and has(res["problems"], "are the same file"), n
        assert has(res["problems"], "the start order of the first campaign records is not defined", "start at the same instant")
        if n == 4:
            res = compare(base, cand, "abba")
            assert res["verdict"] == "INVALID" and has(res["problems"], "are the same file") and has(res["problems"], "the start order of the first campaign records is not defined")
        assert br.order_problems(base, cand, "alternate", FIRST) != [] and br.order_problems(base, cand, "abba", FIRST) != []
    cli = ["--base"] + paths + ["--cand"] + paths + ["--metric", "gpu_graph_us"]
    for extra in ([], ["--order", "abba"], ["--no-interleave-check"], ["--no-interleave-check", "--order", "abba"]):
        assert compare_bench.main(cli + extra) == 2, extra                                          # the diagnostic does not make one file two measurements
    assert br.run_labels([br.load_record(p) for p in paths], [br.load_record(p) for p in paths]) is None


def test_a_record_file_is_the_same_file_by_its_path_by_its_content_and_by_its_object(tmp_path):
    paths = written(tmp_path)
    base = [br.load_record(p) for p in paths]
    copies = tbt.write_records(str(tmp_path), "copy", [br.load_record(p) for p in paths])         # the same content in other files
    res = compare(base, [br.load_record(p) for p in copies])
    assert res["verdict"] == "INVALID" and has(res["problems"], "have the same content") and not has(res["problems"], "are the same file")
    link = os.path.join(str(tmp_path), "link0.json")
    os.symlink(paths[0], link)                                                                      # another spelling of the same file, and a candidate whose loaded content was edited
    edited = br.load_record(link)
    edited["meta"]["timestamp"] = stamp(30)
    res = compare(base, [edited, br.load_record(copies[1]), br.load_record(copies[2])])
    assert res["verdict"] == "INVALID" and has(res["problems"], "are the same file") and has(res["problems"], "have the same content")
    hard = os.path.join(str(tmp_path), "hard1.json")
    os.link(paths[1], hard)
    assert has(br.shared_record_problems(base, [br.load_record(hard)], FIRST), "are the same file")
    assert br.shared_record_problems(base, [br.load_record(copies[0])], FIRST) != [] and br.shared_record_problems(base, [], FIRST) == []
    memory_b, memory_c = ordered_sets("ABABAB")                                                     # records that were built in memory: no path, different content
    assert br.shared_record_problems(memory_b, memory_c, FIRST) == [] and br.shared_record_problems(memory_b, [json.loads(json.dumps(memory_b[1]))], FIRST) != []
    holder_b, holder_c = tbt.interleaved([100.0, 100.0, 100.0], [100.0, 100.0, 100.0])             # records that all carry one placeholder path: a path that names no file is no identity
    assert {r["_path"] for r in holder_b + holder_c} == {"mem"} and br.shared_record_problems(holder_b, holder_c, FIRST) == [] and br.compare(holder_b, holder_c)["verdict"] == "PASS"
    same_objects = compare(memory_b, memory_b)
    assert same_objects["verdict"] == "INVALID" and has(same_objects["problems"], "have the same content")
    assert br.shared_record_problems([], [], FIRST) == [] and br.shared_record_problems(base, base[:1] * 6, FIRST)[-1].startswith("... and 2 more record(s) of the first campaign")


def test_equal_start_stamps_leave_the_order_undefined_in_both_orders():
    for labels, order in (("ABABAB", "alternate"), ("ABBAABBA", "abba")):
        base, cand = ordered_sets(labels)
        assert compare(base, cand, order)["verdict"] == "PASS"
        tie = json.loads(json.dumps(cand))
        tie[1]["meta"]["timestamp"] = base[1]["meta"]["timestamp"]                                  # one candidate record started at the instant of a baseline record
        res = compare(base, tie, order)
        assert res["verdict"] == "INVALID" and has(res["problems"], "the start order of the first campaign records is not defined", "start at the same instant"), labels
        assert br.run_labels(base, tie) is None
        every = json.loads(json.dumps(cand))
        for b, c in zip(base, every):
            c["meta"]["timestamp"] = b["meta"]["timestamp"]                                         # all stamps of B are the stamps of A (sorting ties A before B made this alternate)
        res = compare(base, every, order)
        assert res["verdict"] == "INVALID" and has(res["problems"], "start at the same instant"), labels
        twice = json.loads(json.dumps(base))
        twice[2]["meta"]["timestamp"] = twice[0]["meta"]["timestamp"]                               # two records of ONE build at one instant
        res = compare(twice, cand, order)
        assert res["verdict"] == "INVALID" and has(res["problems"], "the start order of the first campaign records is not defined", "baseline record"), labels
        assert br.order_problems(twice, cand, order, "re-run")[0].startswith("the start order of the re-run records is not defined")


def test_a_missing_or_unparsable_start_stamp_is_invalid_in_both_orders_and_never_raises():
    bad_stamps = {"missing": None, "none": "NONE", "empty": "", "text": "yesterday", "no zone": "2026-11-10T10:05:00", "space": "2026-11-10 10:05:00Z", "number": 1794400000,
                  "seven digits": "2026-11-10T10:05:00.1234567Z", "offset": "2026-11-10T10:05:00+00:00", "day 32": "2026-11-32T10:05:00Z", "trailing": "2026-11-10T10:05:00Z "}
    for labels, order in (("ABABAB", "alternate"), ("ABBAABBA", "abba")):
        for what, value in bad_stamps.items():
            base, cand = ordered_sets(labels)
            broken = json.loads(json.dumps(cand))
            if what == "missing":
                del broken[1]["meta"]["timestamp"]
            else:
                broken[1]["meta"]["timestamp"] = None if value == "NONE" else value
            res = compare(base, broken, order)
            assert res["verdict"] == "INVALID" and has(res["problems"], "the start order of the first campaign records is not defined", "candidate record"), (labels, what)
            assert has(res["problems"], "has no timestamp" if what in ("missing", "none") else "is not a UTC start stamp"), (labels, what)
            assert br.run_labels(base, broken) is None
            res = compare(broken, cand, order)                                                      # on the baseline side as well
            assert res["verdict"] == "INVALID" and has(res["problems"], "baseline record", "the start order of the first campaign records is not defined"), (labels, what)


def test_start_stamps_are_parsed_as_instants_not_compared_as_text():
    a_stamps = ["2026-11-10T10:00:00Z", "2026-11-10T10:00:01.0Z", "2026-11-10T10:00:02Z"]
    b_stamps = ["2026-11-10T10:00:00.5Z", "2026-11-10T10:00:01.5Z", "2026-11-10T10:00:02.5Z"]       # in time A B A B A B; as text B A A B B A
    base = [tbt.wall_rec(tbt.BASE_COMMIT, {KEY: tbt.trials(m)}, t) for m, t in zip(MEDIANS, a_stamps)]
    cand = [tbt.wall_rec("abc1234", {KEY: tbt.trials(m)}, t) for m, t in zip(MEDIANS, b_stamps)]
    res = compare(base, cand)
    assert res["verdict"] == "PASS", res["problems"]                                                # sorted as text this campaign is B A A B B A, which is not interleaved
    assert "2026-11-10T10:00:00.5Z" < "2026-11-10T10:00:00Z"                                        # as text the later instant sorts first
    assert br.parse_stamp("2026-11-10T10:00:00Z") == br.parse_stamp("2026-11-10T10:00:00.000000Z") and br.parse_stamp("2026-11-10T10:00:00.5Z") > br.parse_stamp("2026-11-10T10:00:00Z")
    for bad in (None, 5, "", "2026-11-10", "2026-11-10T10:00:00", "2026-11-10T10:00:00.Z", "2026-11-10T10:00:00.1234567Z", " 2026-11-10T10:00:00Z", ["2026-11-10T10:00:00Z"]):
        assert br.parse_stamp(bad) is None, bad
    assert br.run_labels(base, cand) == "ABABAB" and br.run_labels(cand, base) == "BABABA"
    assert br.run_labels(list(reversed(base)), list(reversed(cand))) == "ABABAB"                    # the order of the lists does not matter, only the instants
    base2 = [tbt.wall_rec(tbt.BASE_COMMIT, {KEY: tbt.trials(m)}, t) for m, t in zip(MEDIANS, ["2026-11-10T10:00:00.1Z", "2026-11-10T10:00:00Z"])]
    cand2 = [tbt.wall_rec("abc1234", {KEY: tbt.trials(m)}, t) for m, t in zip(MEDIANS, ["2026-11-10T10:00:00.9Z", "2026-11-10T10:00:01Z"])]   # as text A B A B; in time A A B B
    assert br.run_labels(base2, cand2) == "AABB"
    assert br.order_problems(base2, cand2, "alternate", FIRST) == ["baseline and candidate records of the first campaign are not interleaved in time (1 variant switches, 3 needed)"]
    mixed = ordered_sets("ABBAABBA")                                                                # stamps with and without microseconds in one campaign
    for r in mixed[0][::2] + mixed[1][::2]:
        r["meta"]["timestamp"] = r["meta"]["timestamp"][:-1] + ".250000Z"
    assert br.run_labels(*mixed) == "ABBAABBA" and compare(*mixed, "abba")["verdict"] == "PASS"


def test_the_interleaving_diagnostic_skips_the_order_but_not_the_check_of_shared_files():
    base, cand = ordered_sets("AAABBB")
    assert compare(base, cand)["verdict"] == "INVALID" and compare(base, cand, require_interleaved=False)["verdict"] == "PASS"
    assert compare(base, cand, "abba", require_interleaved=False)["verdict"] == "PASS"
    res = compare(base, base, require_interleaved=False)
    assert res["verdict"] == "INVALID" and has(res["problems"], "have the same content")
    res = compare(base, cand, "bogus", require_interleaved=False)                                    # an unknown order is a usage error even where it is not used
    assert res["verdict"] == "INVALID" and has(res["problems"], "order must be one of")


def test_the_result_carries_the_order_and_an_unknown_order_is_invalid():
    base, cand = ordered_sets("ABABAB")
    assert compare(base, cand)["order"] == "alternate" and compare(base, cand, "alternate")["order"] == "alternate"
    for bad in ("", "Abba", "ABBA", None, "alternating", ("abba",), 4):
        res = compare(base, cand, bad)
        assert res["verdict"] == "INVALID" and has(res["problems"], "order must be one of alternate, abba") and res["order"] == bad, bad
    assert compare([], cand, "abba")["order"] == "abba" and compare([], cand, "abba")["verdict"] == "INVALID"      # an early INVALID names its order too
    uneven = ordered_sets("ABBAABBAA")                                                              # unequal numbers of files: five baseline and four candidate
    res = compare(*uneven, "abba")
    assert res["verdict"] == "INVALID" and has(res["problems"], "5 baseline and 4 candidate")
    assert br.ORDERS == ("alternate", "abba") and br.ABBA_BLOCKS == ("ABBA", "BAAB")
    assert br.format_table(compare(base, cand)).splitlines()[0] == "metric gpu_graph_us, tolerance 1.00 % (regress mode)"          # the header of every older result is unchanged
    abba_b, abba_c = ordered_sets("ABBABAAB")
    assert br.format_table(compare(abba_b, abba_c, "abba")).splitlines()[0] == "metric gpu_graph_us, tolerance 1.00 % (regress mode, abba start order)"
    older = {k: v for k, v in compare(base, cand).items() if k != "order"}                                                        # a report of before the order was recorded
    assert br.format_table(older).splitlines()[0] == "metric gpu_graph_us, tolerance 1.00 % (regress mode)"


# ---- the launch mapping: require_nodes (every record of both builds and of the re-runs) and route_problems (an independent specification) ----

def node_sets(labels="ABABAB", **kw):
    """Two-launch records for build A, single-launch records for build B (every trial carries its mapping), as lists."""
    base, cand = ordered_sets(labels, **kw)
    return with_nodes(base, TWO), with_nodes(cand, ONE)


def named(records, prefix):
    """Give the records of a list distinct paths (``<prefix>0``, ...: no such files exist), so that a message says which list a record belongs to."""
    for i, r in enumerate(records):
        r["_path"] = "%s%d" % (prefix, i)
    return records


def test_require_nodes_makes_every_record_of_both_builds_and_of_the_rerun_carry_its_launch_mapping():
    base, cand = node_sets()
    res = compare(base, cand, require_nodes=True)
    assert res["verdict"] == "PASS" and res["require_nodes"] is True and compare(base, cand)["require_nodes"] is False
    legacy_b, legacy_c = ordered_sets("ABABAB")                                                     # records of before the mappings: valid unless the caller asks for them
    assert compare(legacy_b, legacy_c)["verdict"] == "PASS"
    res = compare(legacy_b, legacy_c, require_nodes=True)
    assert res["verdict"] == "INVALID" and has(res["problems"], "baseline", "records no compute_nodes") and has(res["problems"], "candidate", "records no compute_nodes")
    first = ordered_sets("ABBAABBA", meds_a=UNSTABLE_A, meds_b=UNSTABLE_B)
    rerun = ordered_sets("ABBABAAB", hour="11", scale=3, meds_a=RERUN_A, meds_b=RERUN_B)
    names = ("baseline", "candidate", "baseline re-run", "candidate re-run")
    lists = [named(with_nodes(x, TWO if i % 2 == 0 else ONE), name.replace(" ", "_")) for i, (x, name) in enumerate(zip((first[0], first[1], rerun[0], rerun[1]), names))]

    def judge(sets, **kw):
        return compare(sets[0], sets[1], "abba", baseline_unstable="widen", base_rerun=sets[2], cand_rerun=sets[3], **kw)
    ok = judge(lists, require_nodes=True)
    assert ok["verdict"] == "PASS" and ok["rows"][0]["judged_on"] == "rerun", ok["problems"]
    for which, name in enumerate(names):                                                            # one record of any of the four lists that lost the mappings of its trials
        sets = json.loads(json.dumps(lists))
        for s in sets[which][1]["cells"][0]["samples"]:
            del s["nodes"]
        named(sets[which], name.replace(" ", "_"))
        assert judge(sets)["verdict"] == "PASS", name                                               # without the flag the record is a valid one (the arguments still name the mapping)
        res = judge(sets, require_nodes=True)
        assert res["verdict"] == "INVALID" and has(res["problems"], "%s1" % name.replace(" ", "_"), "trial 0 carries no node mapping"), name
    plain = [json.loads(json.dumps(x)) for x in (first[0], first[1], rerun[0], rerun[1])]            # all records of a campaign, re-run included, without any mapping
    assert judge(plain)["verdict"] == "PASS"
    res = judge(plain, require_nodes=True)
    assert res["verdict"] == "INVALID" and has(res["problems"], "baseline", "records no compute_nodes") and has(res["problems"], "candidate", "records no compute_nodes")
    for side, name in ((0, "baseline"), (1, "candidate")):                                          # one build whose records never carried the mappings (its first campaign has no re-run)
        pair = [json.loads(json.dumps(x)) for x in node_sets("ABBAABBA")]
        pair[side] = json.loads(json.dumps(ordered_sets("ABBAABBA")[side]))
        assert compare(*pair, "abba")["verdict"] == "PASS", name                                     # the builds may differ in what they record about their launches
        res = compare(*pair, "abba", require_nodes=True)
        assert res["verdict"] == "INVALID" and has(res["problems"], name, "records no compute_nodes") and not has(res["problems"], ("candidate", "baseline")[side]), name
    tampered = json.loads(json.dumps(lists))
    tampered[1][2]["cells"][0]["samples"][3]["nodes"]["compute"] = dict(TWO)                         # a middle trial on the other path: invalid with and without the flag
    for flag in (False, True):
        res = judge(tampered, require_nodes=flag)
        assert res["verdict"] == "INVALID" and has(res["problems"], "trial 3"), flag


def test_the_command_line_requires_the_launch_mappings_only_when_asked(tmp_path):
    d = str(tmp_path)
    nb, nc = node_sets()
    lb, lc = ordered_sets("ABABAB")
    with_nodes_args = ["--base"] + tbt.write_records(d, "na", nb) + ["--cand"] + tbt.write_records(d, "nc", nc) + ["--metric", "gpu_graph_us"]
    legacy_args = ["--base"] + tbt.write_records(d, "la", lb) + ["--cand"] + tbt.write_records(d, "lc", lc) + ["--metric", "gpu_graph_us"]
    out = os.path.join(d, "out.json")
    assert compare_bench.main(with_nodes_args + ["--require-nodes", "--json", out]) == 0 and json.load(open(out))[0]["require_nodes"] is True
    assert compare_bench.main(legacy_args) == 0 and compare_bench.main(legacy_args + ["--require-nodes", "--json", out]) == 2
    assert json.load(open(out))[0]["verdict"] == "INVALID" and has(json.load(open(out))[0]["problems"], "records no compute_nodes")
    assert compare_bench.main(with_nodes_args + ["--require-nodes", "--order", "abba"]) == 2          # strict files: the order is judged as before


def route_records(compute_by_cell, n=3, path="rec"):
    """``n`` record files of one build whose cells (key -> mapping) carry that mapping everywhere; the record paths are ``<path><i>``."""
    recs = []
    for i in range(n):
        rec = tbt.wall_rec(tbt.BASE_COMMIT, {k: tbt.trials(100.0) for k in compute_by_cell}, stamp(2 * i))
        for cell in rec["cells"]:
            cell["args"].update(compute_nodes=dict(compute_by_cell[cell["key"]]), aux_nodes={}, compute_kernels=sum(compute_by_cell[cell["key"]].values()),
                                expect_compute=dict(compute_by_cell[cell["key"]]))
            for s in cell["samples"]:
                s["nodes"] = {"compute": dict(compute_by_cell[cell["key"]]), "aux": {}}
        rec["_path"] = "%s%d" % (path, i)
        recs.append(rec)
    return recs


K1, K2, K3 = br.cell_key("decode", N=1, H=12, regime="lockstep"), br.cell_key("decode", N=99, H=12, regime="staggered"), br.cell_key("decode", N=394, H=6, regime="cold")


def test_route_problems_accept_the_specified_launches_and_reject_a_flipped_route_a_missing_key_and_a_legacy_record():
    spec = {K1: ONE, K2: ONE, K3: TWO}
    recs = route_records(spec)
    assert br.route_problems(recs, spec, "baseline") == [] and br.route_problems([], spec) == []
    flipped = dict(spec, **{K2: TWO})                                                               # the specification says two launches where the records show one
    res = br.route_problems(recs, flipped, "candidate")
    assert len(res) == 3 and all(p.startswith("candidate record rec%d: " % i) and K2 in p and "expected" in p for i, p in enumerate(res)), res
    assert has(res, "the recorded compute nodes {'rows_kernel': 1} are not the expected {'plan_kernel': 1, 'rows_kernel': 1}")
    assert not any(K1 in p or K3 in p for p in res)
    for missing in (K1, K3):
        res = br.route_problems(recs, {k: v for k, v in spec.items() if k != missing})                # a cell without a specification is a problem, not a skipped cell
        assert len(res) == 3 and all(missing in p and "has no expected launch mapping" in p for p in res), missing
    assert br.route_problems(recs, {}) != [] and br.route_problems(recs, dict(spec, **{K1: None})) != [] and br.route_problems(recs, dict(spec, **{K1: "single-launch"})) != []
    assert br.route_problems(recs, None) != [] and br.route_problems(recs, [spec]) != []                 # no specification at all is not a specification that every cell meets
    assert br.route_problems(recs, types.MappingProxyType({k: types.MappingProxyType(v) for k, v in spec.items()})) == []      # any mapping type will do
    assert br.route_problems(recs, dict(spec, **{"decode;N=7;": TWO})) == []                         # a specification for a cell that no record holds is not examined here
    one_trial = json.loads(json.dumps(recs))
    one_trial[1]["cells"][2]["samples"][3]["nodes"]["compute"] = dict(ONE)                           # one trial of one record on the other path
    res = br.route_problems(one_trial, spec, "candidate")
    assert len(res) == 1 and res[0].startswith("candidate record rec1: ") and K3 in res[0] and "trial 3" in res[0]
    fraction = json.loads(json.dumps(recs))
    fraction[0]["cells"][0]["samples"][0]["nodes"]["compute"]["rows_kernel"] = 0.984                 # a launch that is missing from some calls
    assert has(br.route_problems(fraction, spec), "whole number of times")
    vouching = route_records({K1: TWO, K2: TWO, K3: TWO})                                           # a record whose cells and trials all agree with each other, on the wrong route
    assert br.route_problems(vouching, {K1: TWO, K2: TWO, K3: TWO}) == [] and len(br.route_problems(vouching, spec)) == 6    # internal consistency is not the specification (K1 and K2 per record, 3 records)
    legacy = json.loads(json.dumps(recs))
    for cell in legacy[2]["cells"]:
        for name in ("compute_nodes", "aux_nodes", "compute_kernels", "expect_compute"):
            del cell["args"][name]
        for s in cell["samples"]:
            del s["nodes"]
    res = br.route_problems(legacy, spec, "baseline")
    assert len(res) == 3 and all(p.startswith("baseline record rec2: ") and "records no compute_nodes" in p for p in res)
    stripped = json.loads(json.dumps(recs))
    del stripped[0]["cells"][1]["samples"][2]["nodes"]
    assert has(br.route_problems(stripped, spec), K2, "trial 2 carries no node mapping")
    assert br.route_problems([{"cells": []}, {"_path": "x"}, {"cells": [{"samples": []}, 5]}], spec)[0].endswith("no cells")
    garbled = json.loads(json.dumps(recs))                                                          # records that were not validated first: problems, never an exception
    garbled[0]["cells"][0]["samples"][1]["nodes"]["compute"]["rows_kernel"] = "one"
    garbled[1]["cells"][1]["samples"][0] = 5
    res = br.route_problems(garbled, spec, "candidate")
    assert len(res) == 2 and all("cannot be read" in p for p in res) and K1 in res[0] and K2 in res[1], res
    many = {"decode;N=%d;" % i: ONE for i in range(10)}                                              # ten flipped cells: six are listed and the rest is counted
    res = br.route_problems(route_records(many, n=1), {k: TWO for k in many})
    assert len(res) == 7 and res[-1] == "side record rec0: ... and 4 more cell(s) whose launches are not the expected ones", res


# ---- the control state and the route expectation: what a side shares, what a re-run shares with its first campaign, what A and B may differ in ----

def test_the_single_launch_fields_are_side_meta_a_rerun_shares_and_build_arguments_that_a_and_b_may_differ_in():
    assert br.SIDE_META[-2:] == ("single_force", "single_per_sm") and set(br.SIDE_META) <= set(br.RERUN_SAME_CONFIG)
    assert {"single_force", "single_per_sm"} <= set(br.RERUN_SAME_CONFIG) and not {"single_force", "single_per_sm"} & set(br.AB_META)
    assert {"single_force", "expect_route", "expect_compute"} <= set(br.BUILD_ARGS) and not {"single_force", "expect_route"} & set(br.SCALED_ARGS)


def forced_pair(labels="ABABAB", **kw):
    """A: the two-launch route, the control off; B: the default route with the control on and the mapping of every cell taken from the specification."""
    base, cand = ordered_sets(labels, **kw)
    set_meta(base, single_force=False, single_per_sm=8)
    set_args(base, single_force=False, expect_route=None, expect_compute="two-launch")
    set_meta(cand, single_force=True, single_per_sm=8)
    set_args(cand, single_force=True, expect_route="spec", expect_compute="single-launch")
    return base, cand


def test_a_and_b_may_differ_in_the_control_state_and_the_route_expectation_but_in_nothing_else_of_the_workload():
    base, cand = forced_pair()
    res = compare(base, cand)
    assert res["verdict"] == "PASS", res["problems"]
    for field, value in (("calls_per_trial", 7), ("profile_calls", 7), ("N", 2), ("regime", "cold")):   # any other argument of the workload still has to agree
        other = json.loads(json.dumps(cand))
        set_args(other, **{field: value})
        res = compare(base, other)
        assert res["verdict"] == "INVALID" and has(res["problems"], "differ in %s" % field), field
    other = json.loads(json.dumps(cand))
    set_meta(other, trials=7)                                                                       # and so does the contract of the measurement
    assert has(compare(base, other)["problems"], "differ in meta trials")
    for field in ("single_force", "single_per_sm"):                                                 # a build without the fields (the base tree has no such control) is a build like any other
        legacy_side = json.loads(json.dumps(base))
        edit_fields(legacy_side, {field: GONE})
        assert compare(legacy_side, cand)["verdict"] == "PASS", field
    legacy_b, legacy_c = ordered_sets("ABABAB")                                                     # records of before the fields: no difference anywhere
    assert compare(legacy_b, legacy_c)["verdict"] == "PASS"


def test_the_files_of_one_side_share_the_control_state_and_the_limit_constant():
    for field, value in (("single_force", False), ("single_per_sm", 16), ("single_force", None)):
        base, cand = forced_pair()
        if value is None:
            del cand[1]["meta"][field]
        else:
            cand[1]["meta"][field] = value
        res = compare(base, cand)
        assert res["verdict"] == "INVALID" and has(res["problems"], "candidate records disagree on meta %s" % field), (field, value)
    base, cand = forced_pair()
    base[2]["cells"][0]["args"]["single_force"] = True                                              # one process of the baseline measured with the control on, under the same key
    res = compare(base, cand)
    assert res["verdict"] == "INVALID" and has(res["problems"], "baseline records disagree on the arguments of cell")
    base, cand = forced_pair()
    cand[0]["cells"][0]["args"]["expect_route"] = None
    res = compare(base, cand)
    assert res["verdict"] == "INVALID" and has(res["problems"], "candidate records disagree on the arguments of cell")


def rerun_pair(rerun_meta=None, rerun_args=None, first_meta=None, first_args=None):
    """A campaign whose cell needs a re-run (a spread of 0.9 %), measured with the control on in B, and its re-run of the same state; ``*_meta`` / ``*_args`` edit one of the two campaigns of B."""
    first = forced_pair(meds_a=UNSTABLE_A, meds_b=UNSTABLE_B)
    rerun = forced_pair(hour="11", scale=3, meds_a=RERUN_A, meds_b=RERUN_B)
    edit_fields(first[1], first_meta, first_args)
    edit_fields(rerun[1], rerun_meta, rerun_args)
    return compare(*first, baseline_unstable="widen", base_rerun=rerun[0], cand_rerun=rerun[1])


def test_a_rerun_shares_the_control_state_and_the_limit_constant_of_its_first_campaign_in_the_meta():
    """Only the meta carries the fields here (both builds alike, no cell argument): a re-run that lost or changed one of them is not a re-run of its first campaign."""
    def judge(**rerun_meta):
        first = ordered_sets("ABABAB", meds_a=UNSTABLE_A, meds_b=UNSTABLE_B)
        rerun = ordered_sets("ABABAB", hour="11", scale=3, meds_a=RERUN_A, meds_b=RERUN_B)
        for pair, meta in ((first, {"single_force": True, "single_per_sm": 8}), (rerun, dict({"single_force": True, "single_per_sm": 8}, **rerun_meta))):
            for side in pair:
                edit_fields(side, meta)
        return compare(*first, baseline_unstable="widen", base_rerun=rerun[0], cand_rerun=rerun[1])
    ok = judge()
    assert ok["verdict"] == "PASS" and ok["rows"][0]["judged_on"] == "rerun", ok["problems"]
    for what, change in (("the control off in the re-run", dict(single_force=False)), ("no control state in the re-run", dict(single_force=GONE)),
                         ("another limit constant in the re-run", dict(single_per_sm=16)), ("no limit constant in the re-run", dict(single_per_sm=GONE))):
        res = judge(**change)
        field = "single_force" if "single_force" in change else "single_per_sm"
        assert res["verdict"] == "INVALID" and has(res["problems"], "re-run does not come from the same snapshot / configuration as its first campaign", "meta %s" % field), what


def test_a_rerun_without_the_force_state_of_its_first_campaign_is_rejected():
    ok = rerun_pair()
    assert ok["verdict"] == "PASS" and ok["rows"][0]["judged_on"] == "rerun", ok["problems"]
    for what, kw in (("the control off in the re-run", dict(rerun_meta={"single_force": False}, rerun_args={"single_force": False})),
                     ("the control off in the re-run, in the meta only", dict(rerun_meta={"single_force": False})),
                     ("the control off in the re-run, in the cell arguments only", dict(rerun_args={"single_force": False})),
                     ("no control state in the re-run meta", dict(rerun_meta={"single_force": GONE})),
                     ("the control on in the re-run only", dict(first_meta={"single_force": False}, first_args={"single_force": False})),
                     ("a legacy first campaign and a re-run with the state", dict(first_meta={"single_force": GONE}, first_args={"single_force": GONE})),
                     ("another limit constant in the re-run", dict(rerun_meta={"single_per_sm": 16})),
                     ("no limit constant in the re-run", dict(rerun_meta={"single_per_sm": GONE})),
                     ("another route expectation in the re-run", dict(rerun_args={"expect_route": None}))):
        res = rerun_pair(**kw)
        assert res["verdict"] == "INVALID" and res["problems"], what
    res = rerun_pair(rerun_meta={"single_force": False}, rerun_args={"single_force": False})
    assert has(res["problems"], "re-run does not come from the same snapshot / configuration as its first campaign", "meta single_force") and has(res["problems"], "differ in single_force")
    assert has(rerun_pair(rerun_meta={"single_per_sm": 16})["problems"], "meta single_per_sm")
    assert has(rerun_pair(rerun_args={"expect_route": None})["problems"], "differ in expect_route")
    legacy_first = ordered_sets("ABABAB", meds_a=UNSTABLE_A, meds_b=UNSTABLE_B)                      # no field anywhere: a legacy campaign and its legacy re-run are consistent
    legacy_rerun = ordered_sets("ABABAB", hour="11", scale=3, meds_a=RERUN_A, meds_b=RERUN_B)
    assert compare(*legacy_first, baseline_unstable="widen", base_rerun=legacy_rerun[0], cand_rerun=legacy_rerun[1])["verdict"] == "PASS"
    first = forced_pair(meds_a=UNSTABLE_A, meds_b=UNSTABLE_B)                                       # the control state of the baseline's re-run is bound as well (both builds, both campaigns)
    rerun = forced_pair(hour="11", scale=3, meds_a=RERUN_A, meds_b=RERUN_B)
    set_meta(rerun[0], single_force=True)
    res = compare(*first, baseline_unstable="widen", base_rerun=rerun[0], cand_rerun=rerun[1])
    assert res["verdict"] == "INVALID" and has(res["problems"], "baseline re-run does not come from the same snapshot", "meta single_force")


def test_the_comparison_cannot_tell_which_launch_path_ran_so_the_route_specification_has_to():
    """The comparison and the route check are two halves of one gate: records that agree with themselves on the wrong path pass the first and fail the second."""
    spec = {KEY: ONE}
    base, cand = forced_pair()
    base, cand = with_nodes(base, TWO), with_nodes(cand, ONE)
    assert compare(base, cand, require_nodes=True)["verdict"] == "PASS" and br.route_problems(cand, spec, "candidate") == [] and br.route_problems(base, {KEY: TWO}, "baseline") == []
    swapped_c = with_nodes(cand, TWO)                                                               # the candidate silently ran the two-launch path in every trial and says so everywhere
    assert compare(base, swapped_c, require_nodes=True)["verdict"] == "PASS"                       # the comparator cannot tell: nothing in the records contradicts itself ...
    assert has(br.route_problems(swapped_c, spec, "candidate"), "candidate record", "not the expected")  # ... the specification does
