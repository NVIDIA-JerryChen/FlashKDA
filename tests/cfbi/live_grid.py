"""The live grid of a workload that claims to be beyond 2.5 resident waves, planned from what its launches execute (plain Python: no torch, runs on the login node).

A launch puts a CTA on the device for every head of a sequence that runs rows. A padding entry (slot -1), an empty sequence and an entry that takes no token in this launch exit before the
row engine and are not live, so the NOMINAL grid (entries x heads) is no measure of the waves. The planners below reserve the ACTIVE quota first (the live sequences per launch that 2.5 waves
of the instantiation under test need, ``active_quota``) and add the padding, empty and zero-take entries ON TOP of it; the tests then assert, before any GPU work, that every planned launch
has at least ``live_floor`` live CTAs (``rows_workload.beyond_waves_problems`` over ``rows_workload.launch_coverage`` of the planned lens and slots).

The round-7 planner of the mode-1 test sized N x H = 741 CTAs and then made every eleventh entry empty and split the lengths 1 and 2 into ``[0, 0, L]``: its launches had 495, 495 and 675
live CTAs (1.67, 1.67 and 2.28 waves at 296 resident CTAs) and the guard that existed (``coverage_problems``, which only asks for one live CTA) passed.
"""
import math
import random

WAVES = 2.5                        # equal to helpers.WAVES (a CPU test says so)
MODE1_LENGTHS = (5, 33, 64, 65, 97, 130)    # the lengths of the ACTIVE entries of the mode-1 workload: every one takes tokens in each of its three calls (length // 3 >= 1)
CALLS = 3
EMPTY_EVERY, ONE_EVERY, TWO_EVERY = (11, 5), (13, 7), (17, 3)   # (modulus, residue) of the entries that are extras: an empty sequence, a sequence of one token, a sequence of two tokens


def live_floor(capacity, waves=WAVES):
    """The fewest live CTAs per launch that make it ``waves`` waves of ``capacity`` resident CTAs."""
    return int(math.ceil(waves * capacity))


def active_quota(capacity, heads, waves=WAVES):
    """The number of ACTIVE sequences (each live in every launch, ``heads`` CTAs each) that reach ``live_floor``; equal to ``helpers.sequences_beyond_waves`` (a CPU test says so)."""
    return max(1, -(-live_floor(capacity, waves) // heads))


def padded_batch(active, every, residue):
    """The smallest batch whose entries leave ``active`` live ones when the entries ``i % every == residue`` are padding (they come on top of the quota): ``(N, padding indices)``."""
    n = live = 0
    while live < active:
        if n % every != residue:
            live += 1
        n += 1
    return n, [i for i in range(n) if i % every == residue]


def fraction_batch(active, divisor):
    """The smallest batch ``N`` with ``N - N // divisor >= active``: the batch of a test that makes ``N // divisor`` entries padding (which ones is its own choice): ``(N, padding count)``."""
    n = active
    while n - n // divisor < active:
        n += 1
    return n, n // divisor


def mode1_plan(active, seed=7):
    """Lengths and per-call takes of a three-call mode-1 workload with ``active`` ACTIVE sequences (lengths from ``MODE1_LENGTHS``, tokens in every call) and the extras on top of them: empty
    entries, entries of one and of two tokens (zero take in the first two calls). Returns ``(lens, takes)`` with ``takes[call][n]``; every call takes ``lens`` apart in three shares."""
    rng = random.Random(seed)
    lens = []
    n_active = n = 0
    while n_active < active:
        if n % EMPTY_EVERY[0] == EMPTY_EVERY[1]:
            lens.append(0)
        elif n % ONE_EVERY[0] == ONE_EVERY[1]:
            lens.append(1)
        elif n % TWO_EVERY[0] == TWO_EVERY[1]:
            lens.append(2)
        else:
            lens.append(rng.choice(MODE1_LENGTHS))
            n_active += 1
        n += 1
    takes = [[L // CALLS if c < CALLS - 1 else L - (CALLS - 1) * (L // CALLS) for L in lens] for c in range(CALLS)]
    return lens, takes


def launches(takes, slots=None):
    """``[(lens, slots)]`` of the planned launches for ``rows_workload.launch_coverage`` (``slots`` default: entry n on slot n)."""
    n = len(takes[0])
    return [(list(take), list(range(n)) if slots is None else list(slots)) for take in takes]
