"""The preconditions of the single-launch decode are written down where a user and a maintainer read them (plain Python, no GPU: ``PYTHONPATH=. pytest --noconftest tests/cfbi/test_decode_preconditions_docs.py``).

The kernel counts the arrivals of the head CTAs of a sequence in the high bits of the ``pos`` word while a call runs. Four conditions on the state and the call are therefore part of the
contract, documented and not validated on the device (a validation would need a device synchronisation; it stays a future item):

1. the non-negative ``slot_ids`` of one call are distinct, a padding entry ``-1`` may repeat;
2. every ``pos`` is at rest, in 0 .. 63, without residue of an earlier call: an interrupted call may leave arrival bits that no later call heals, and a ``pos`` in range does not prove that a
   call finished;
3. the state is owned by one stream while a call runs;
4. the launch gives every sequence exactly ``H`` CTAs.

Each statement must be in the docstring of ``prefill`` (``flash_kda/cfbi/api.py``) and in the performance document, ``decode`` must point at the list, and the earlier wording ("slots of one call
must be distinct", which wrongly includes the padding entries) must not come back. Every statement has a mutant that deletes it.
"""
import ast
import re
from pathlib import Path

import pytest

import protocol_source as ps

ROOT = Path(__file__).resolve().parents[2]
API = ROOT / "flash_kda" / "cfbi" / "api.py"
PERF = ROOT / "docs" / "20261103-cfbi-perf.md"
HEADING = "**Preconditions of the single launch"

# statement -> the phrases that must all be there (after the backticks are dropped and the white space is collapsed)
API_PHRASES = {
    "distinct slots": [r"non-negative entries of slot_ids are distinct within one call", r"-1 padding entries skip the call and may repeat"],
    "pos at rest without residue": [r"every pos word is at rest, in 0 \.\. 63, with no residue of an earlier call", r"a call that was interrupted", r"no later call heals",
                                    r"does not prove that a call finished", r"its state has to be rebuilt"],
    "one stream": [r"the state is owned by one stream while a call runs", r"after an event or a synchronisation"],
    "exactly H CTAs": [r"exactly H CTAs, one per head"],
}
PERF_PHRASES = {
    "distinct slots": [r"the non-negative slot_ids of one call are distinct", r"a padding entry -1 may repeat"],
    "pos at rest without residue": [r"every pos is at rest in 0 \.\. 63 without residue", r"an interrupted call may leave arrival bits in the word", r"not healed",
                                    r"does not prove that a call finished", r"clearing the bits does not repair them"],
    "one stream": [r"one stream owns the state during a call"],
    "exactly H CTAs": [r"exactly H CTAs per sequence"],
}
REJECTED = r"Slots of one call must be distinct\."


def plain(text):
    return re.sub(r"\s+", " ", text.replace("`", ""))


def docstring(function):
    tree = ast.parse(API.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            return ast.get_docstring(node)
    raise KeyError(function)


def perf_paragraph():
    text = PERF.read_text()
    start = text.index(HEADING)
    end = text.find("\n\n", start)
    return text[start:end if end > 0 else len(text)]


def missing(text, phrases):
    """The statements whose phrases are not all in ``text``, as ``{statement: [missing phrase]}``."""
    t = plain(text)
    out = {}
    for statement, pats in phrases.items():
        gone = [p for p in pats if not re.search(p, t)]
        if gone:
            out[statement] = gone
    return out


def test_the_api_documentation_states_the_four_preconditions():
    assert missing(docstring("prefill"), API_PHRASES) == {}
    assert not re.search(REJECTED, plain(docstring("prefill"))), "the unqualified claim that every slot of a call is distinct (padding entries may repeat) is back"


def test_decode_points_at_the_list_of_preconditions():
    doc = plain(docstring("decode"))
    assert "preconditions of the state" in doc and all(w in doc for w in ("distinct slots", "residue", "one stream", "exactly H CTAs"))


def test_the_performance_document_states_the_four_preconditions():
    assert missing(perf_paragraph(), PERF_PHRASES) == {}
    assert "documented, not validated on the device" in perf_paragraph()


@pytest.mark.parametrize("statement", list(API_PHRASES))
def test_a_deleted_statement_is_caught_in_the_api_documentation(statement):
    text = re.sub(API_PHRASES[statement][0], "", plain(docstring("prefill")))
    assert statement in missing(text, API_PHRASES)


@pytest.mark.parametrize("statement", list(PERF_PHRASES))
def test_a_deleted_statement_is_caught_in_the_performance_document(statement):
    text = re.sub(PERF_PHRASES[statement][0], "", plain(perf_paragraph()))
    assert statement in missing(text, PERF_PHRASES)


def test_the_old_unqualified_wording_is_caught():
    doc = docstring("prefill") + "\nSlots of one call must be distinct. The rest.\n"
    assert re.search(REJECTED, plain(doc))


def test_the_counter_capacity_the_document_quotes_is_the_one_of_the_sources():
    sources = ps.load()
    text = plain(perf_paragraph())
    assert "23 bits above the shift" in text and "2^23 heads" in text and "2^16 heads" in text
    assert 31 - ps.constants(sources)["POS_COUNT_SHIFT"] == 23 and ps.counter_limit_problems(sources) == []
