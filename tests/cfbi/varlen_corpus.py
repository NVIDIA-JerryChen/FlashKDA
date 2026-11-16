"""Variable-length corpus V of the CF-BI K2 schedule benchmarks and tests (plain Python, no torch).

One entry is the list of sequence lengths (tokens) of a single ``cu_seqlens`` prefill call (B = 1) together with the head count H. The entries
are defined once, here, and shared by ``benchmarks/bench_breakdown.py`` (``--varlen``), the scheduler tests and ``docs/20261103-cfbi-perf.md``.

    V1  equal lengths                      8 x 8192                       H = 6
    V2  one long sequence                  8192 + 7 x 1024                H = 6   (V2r: the long sequence last)
    V3  half long, half short              4 x 8192 + 4 x 1024            H = 6
    V4  log-normal-like                    8 lengths, random.Random(3)    H = 6
    V5  uniform 2K .. 8K                   8 lengths, the same stream     H = 6
    V6  extreme skew                       32768 + 7 x 1024               H = 6
    V7a one sequence (below one SM wave)   8192                           H = 6
    V7b two sequences, skewed              8192 + 1024                    H = 6
    V7c 64 sequences, 1K .. 8K             random.Random(64)              H = 6   (many waves of CTAs)
    V7d 16 sequences, log-normal-like      random.Random(16)              H = 32  (N * H = 512 pairs)

All lengths are multiples of the chunk size (64 tokens).
"""
import random
from collections import OrderedDict, namedtuple

CHUNK = 64

Entry = namedtuple("Entry", "name lens H note")


def lognormal_like(rng, n, sigma=1.0):
    """n lengths of ``max(1, min(128, int(lognormvariate(4.2, sigma)))) * 64`` tokens."""
    return [max(1, min(128, int(rng.lognormvariate(4.2, sigma)))) * CHUNK for _ in range(n)]


def uniform_chunks(rng, n, lo=32, hi=128):
    """n lengths of ``randint(lo, hi) * 64`` tokens."""
    return [rng.randint(lo, hi) * CHUNK for _ in range(n)]


def _build():
    rng = random.Random(3)
    v4 = lognormal_like(rng, 8)
    v5 = uniform_chunks(rng, 8)  # the same random stream, after the log-normal draws
    v7c = uniform_chunks(random.Random(64), 64, 16, 128)
    v7d = lognormal_like(random.Random(16), 16)
    entries = [
        Entry("V1", [8192] * 8, 6, "equal lengths"),
        Entry("V2", [8192] + [1024] * 7, 6, "one long sequence first"),
        Entry("V2r", [1024] * 7 + [8192], 6, "one long sequence last"),
        Entry("V3", [8192] * 4 + [1024] * 4, 6, "half long, half short"),
        Entry("V4", v4, 6, "log-normal-like"),
        Entry("V5", v5, 6, "uniform 2K..8K"),
        Entry("V6", [32768] + [1024] * 7, 6, "extreme skew"),
        Entry("V7a", [8192], 6, "single sequence, below one wave"),
        Entry("V7b", [8192, 1024], 6, "two sequences, skewed"),
        Entry("V7c", v7c, 6, "64 sequences of 1K..8K tokens"),
        Entry("V7d", v7d, 32, "16 log-normal-like sequences, H = 32"),
    ]
    return OrderedDict((e.name, e) for e in entries)


CORPUS_V = _build()


def corpus_names():
    return list(CORPUS_V.keys())


def cu_seqlens_of(lens):
    """Cumulative offsets ``[0, l0, l0 + l1, ...]`` of a length list."""
    out = [0]
    for n in lens:
        out.append(out[-1] + int(n))
    return out


def _parse_terms(text):
    lens = []
    for term in text.split("+"):
        term = term.strip()
        if not term:
            raise ValueError("empty term in length list %r" % text)
        if "*" in term:
            count, length = term.split("*", 1)
            lens.extend([int(length)] * int(count))
        else:
            lens.append(int(term))
    return lens


def parse_spec(spec):
    """``(name, lens, H)`` of a benchmark spec.

    ``V2`` (a corpus entry), ``V2@12`` (the entry with another head count) or an explicit list ``8192+7*1024@6``
    (terms ``LEN`` or ``COUNT*LEN`` joined by ``+``; the head count after ``@`` is required).
    """
    spec = spec.strip()
    head, at, h_text = spec.partition("@")
    if head in CORPUS_V:
        e = CORPUS_V[head]
        return e.name, list(e.lens), (int(h_text) if at else e.H)
    if not at:
        raise ValueError("unknown corpus entry %r (known: %s); an explicit list needs '@H', e.g. 8192+7*1024@6" % (head, ", ".join(corpus_names())))
    lens = _parse_terms(head)
    if any(n <= 0 for n in lens):
        raise ValueError("lengths must be positive: %r" % spec)
    return spec, lens, int(h_text)
