"""Workloads, comparers and variant builds of the row-engine tests (the decode families, the delay variant, the sanitizer driver and the graph replay share them).

A *step* is one mode-1 launch of the row engine: ``lens[n]`` rows of sequence n (without ``cu_seqlens`` every sequence has the same number of rows ``T`` and the inputs are
``[N, T, H, 128]``; with it any lengths, empty ones included), and the pool slot of every sequence (``slots``; ``None``: a compact state, sequence n uses entry n; ``-1``: a padding entry
that the launch skips). Both arms of a comparison start from clones of one *warm* state, made by the reference kernels, and run the same step: the optimised row engine (``impl 0``) and
the frozen reference (``impl 3``). What is compared afterwards is the *outcome*: the bf16 output, the fp32 output, ``E``, the valid rows of the cache and ``pos`` of every entry of the state.

The grid is derived from the device (``helpers.decode_sequences``: SMs times the CTAs per SM of the occupancy API) so that the sequences that really run (not the padding or empty
entries, which exit at once) fill at least ``WAVES`` waves of resident CTAs; extra padding and empty entries come on top.
"""
import itertools
import random

import torch

from flash_kda.cfbi import alloc_state, state_mismatch

from helpers import WAVES, bits, make_case, n_diff, resident_ctas, sequences_beyond_waves

BT = 64
NAN = float("nan")
OPTIMISED, REFERENCE = 0, 3       # the _impl of the arm under test and of the oracle
POOL_SPARE = 19                   # slots of a pool that no sequence of the call uses


# ---- variant builds: a build that fails is an error of the test, never a skip ---------------------------------------------------------------------------
def build_variant(name, defines, loader=None):
    """The JIT build of a differently flagged copy of the extension (``load_variant``). A build that fails fails the test: a skipped comparison proves nothing."""
    import pytest

    if loader is None:
        from flash_kda.cfbi._ext import load_variant as loader
    try:
        return loader(name, tuple(defines))
    except Exception as e:   # the compiler's message is in the text of the exception (after the long command lines of the failed compilations)
        text = str(e)
        errors = [ln.strip() for ln in text.splitlines() if "error" in ln.lower()]
        pytest.fail("the variant %s %s does not build: %s" % (name, tuple(defines), "\n".join(errors[-12:]) if errors else text[-2000:]), pytrace=False)


# ---- the inputs of a step -------------------------------------------------------------------------------------------------------------------------------
def make_params(H, seed=0):
    """The layer parameters (A_log, dt_bias, scale, lower_bound) that every call of one workload shares."""
    case = make_case(1, H, "authors", seed=seed)
    return {k: case[k] for k in ("A_log", "dt_bias", "scale", "lower_bound")}


class Step:
    """One mode-1 launch: the tensors, ``cu`` (int64 [N + 1] or None), ``slot_ids`` (int32 [N] or None) and the python lists they were made from."""

    def __init__(self, par, H, lens, slots, fixed, seed):
        self.par, self.H, self.lens, self.slots, self.fixed = par, H, list(lens), None if slots is None else list(slots), fixed
        N = len(lens)
        total = sum(lens)
        case = make_case(max(total, 1), H, "authors", seed=seed)
        if fixed:
            T = lens[0]
            assert T >= 1 and all(l == T for l in lens), "a fixed-length step has the same number of rows for every sequence"
            self.q, self.k, self.v, self.g = (case[x].view(N, T, H, 128) for x in ("q", "k", "v", "g"))
            self.beta = case["beta"].view(N, T, H)
            self.cu = None
        else:
            self.q, self.k, self.v, self.g, self.beta = (case[x][:, :total].contiguous() if total else case[x][:, :0].contiguous() for x in ("q", "k", "v", "g", "beta"))
            self.cu = torch.tensor([0] + list(itertools.accumulate(lens)), dtype=torch.int64, device="cuda")
        self.slot_ids = None if slots is None else torch.tensor(slots, dtype=torch.int32, device="cuda")

    @property
    def N(self):
        return len(self.lens)

    def slot(self, n):
        return n if self.slots is None else self.slots[n]

    def live(self, n):
        """Does sequence n process rows (not a padding entry, not an empty sequence)?"""
        return self.slot(n) >= 0 and self.lens[n] > 0

    def live_ctas(self):
        return sum(1 for n in range(self.N) if self.live(n)) * self.H

    def rows_of(self, n):
        """The index range of the output rows of sequence n: ``(first, count)`` along the token axis of ``out[n]`` (fixed layout) or of ``out[0]`` (``cu_seqlens``)."""
        if self.fixed:
            return 0, self.lens[n]
        cu = self.cu_list()
        return cu[n], self.lens[n]

    def cu_list(self):
        return [0] + list(itertools.accumulate(self.lens))


def outputs_like(step):
    """Output buffers that hold a NaN sentinel: a row that the launch does not write stays NaN in both arms."""
    out = torch.full(step.q.shape, NAN, dtype=torch.bfloat16, device="cuda")
    out32 = torch.full(step.q.shape, NAN, dtype=torch.float32, device="cuda")
    return out, out32


def run_step(mod, impl, state, step):
    """Run one step through ``mod.forward`` (mode 1) against ``state`` (updated in place); returns ``(out, out32)``."""
    out, out32 = outputs_like(step)
    par = step.par
    mod.forward(step.q, step.k, step.v, step.g, step.beta, float(par["scale"]), out, par["A_log"], par["dt_bias"], float(par["lower_bound"]),
                state.E, state.rows, state.pos, step.cu, 1, out32, impl, step.slot_ids)
    return out, out32


# ---- warm states ----------------------------------------------------------------------------------------------------------------------------------------
def warm_pool(mod, S, H, fills, par, seed, chunks=None):
    """A pool of S slots in which slot s ends with ``fills[s]`` rows (0 .. 63) in its open chunk after ``chunks[s]`` complete chunks. Made by the REFERENCE kernels (prefill of random
    tokens through the pool path, in a random slot order), so both arms of a comparison start from the oracle's own state. Slots with no token stay fresh."""
    assert len(fills) == S and all(0 <= f < BT for f in fills)
    rng = random.Random(seed)
    chunks = chunks if chunks is not None else [rng.choice([0, 1]) for _ in range(S)]
    lens = [fills[s] + BT * chunks[s] for s in range(S)]
    pool = alloc_state(S, H)
    active = [s for s in range(S) if lens[s] > 0]
    rng.shuffle(active)
    if active:
        a_lens = [lens[s] for s in active]
        case = make_case(sum(a_lens), H, "authors", seed=seed)
        cu = torch.tensor([0] + list(itertools.accumulate(a_lens)), dtype=torch.int64, device="cuda")
        ids = torch.tensor(active, dtype=torch.int32, device="cuda")
        out = torch.zeros_like(case["q"])
        mod.forward(case["q"], case["k"], case["v"], case["g"], case["beta"], float(par["scale"]), out, par["A_log"], par["dt_bias"], float(par["lower_bound"]),
                    pool.E, pool.rows, pool.pos, cu, 0, None, REFERENCE, ids)
        torch.cuda.synchronize()
    assert pool.pos.cpu().tolist() == list(fills), "the warm-up left the open chunks at other positions than asked for"
    return pool


# ---- the families: how a step is drawn ------------------------------------------------------------------------------------------------------------------
class Family:
    """``fixed``: rows per sequence of a fixed-length step (None: ``cu_seqlens`` with ``lens``, drawn from the choices); ``pool``: a pool with permuted slots (False: a compact state);
    ``pad``: padding entries (slot -1) per live sequence; ``empty``: empty sequences per live sequence (``cu_seqlens`` only); ``fills``: the positions of the open chunks of a warm slot."""

    def __init__(self, name, fixed=None, lens=None, pool=False, pad=0.0, empty=0.0, fills="mixed", steps=3):
        self.name, self.fixed, self.lens, self.pool, self.pad, self.empty, self.fills, self.steps = name, fixed, lens, pool, pad, empty, fills, steps

    def __repr__(self):
        return self.name


FILLS = {
    "mixed": lambda rng: rng.choice([0, 1, rng.randint(2, 61), rng.randint(2, 61), 62, 63]),
    "open": lambda rng: rng.choice([1, 2, 31, 58, 59, 60, 61, 62, 63]),               # every chunk is open (no empty one), the last positions are well covered
    "crossing": lambda rng: rng.choice([57, 58, 59, 60, 61, 62, 63]),                  # a multi-row step runs over the chunk end
    "boundary": lambda rng: rng.choice([0, 1, 62, 63]),
    "sixty": lambda rng: 60,                                                           # every sequence of a five-row launch finishes its chunk at its fourth row and goes on
}
MUST_HAVE = {"mixed": [0, 1, 62, 63], "boundary": [0, 1, 62, 63], "open": [63, 62, 1], "crossing": [63]}    # positions that every draw contains at least once (when the pool is big enough)


def draw_fills(rng, S, kind):
    fills = [FILLS[kind](rng) for _ in range(S)]
    for i, must in enumerate(MUST_HAVE.get(kind, [])):
        if S > i:
            fills[i] = must
    rng.shuffle(fills)
    return fills


def draw_step(ext, fam, H, par, rng, seed):
    """A step of the family on a grid whose live CTAs fill at least ``WAVES`` resident waves of the optimised row engine (the padding and the empty entries come on top of them);
    returns ``(step, S)`` with the pool size (N for a compact state)."""
    live = sequences_beyond_waves(resident_ctas(ext), H)
    n_pad, n_empty = int(live * fam.pad), int(live * fam.empty)
    assert not (n_pad and not fam.pool) and not (n_empty and fam.fixed is not None), "padding entries need a pool, empty sequences need cu_seqlens"
    N = live + n_pad + n_empty
    special = rng.sample(range(N), n_pad + n_empty)
    pads, empties = set(special[:n_pad]), set(special[n_pad:])
    if fam.fixed is not None:
        lens = [fam.fixed] * N
    else:
        lens = [0 if n in empties else rng.choice([x for x in fam.lens if x > 0]) for n in range(N)]
    S = N + POOL_SPARE if fam.pool else N
    if fam.pool:
        slots = rng.sample(range(S), N)
        slots = [-1 if n in pads else slots[n] for n in range(N)]
    else:
        slots = None
    return Step(par, H, lens, slots, fam.fixed is not None, seed), S


# ---- the outcome of an arm and its checks -----------------------------------------------------------------------------------------------------------------
class Outcome:
    def __init__(self, out, out32, state):
        self.out, self.out32, self.state = out, out32, state


def run_arm(mod, impl, warm, step):
    state = warm.clone()
    out, out32 = run_step(mod, impl, state, step)
    torch.cuda.synchronize()
    return Outcome(out, out32, state)


def mismatch(a, b):
    """The counts of differing words between two outcomes: output, fp32 output, E, valid rows, pos (zero everywhere = bitwise equal)."""
    m = {"out": n_diff(a.out, b.out), "out32": n_diff(a.out32, b.out32)}
    m.update(state_mismatch(a.state, b.state))
    return m


def at_rest_problems(step, warm, outcome):
    """What must hold after a launch whatever the kernel did, independently of the oracle: every position stays in [0, 63]; a sequence that ran advanced its position by its rows (mod 64) and
    every other entry of the state is untouched, bit for bit (E and the whole cache included); a row of a padding or empty sequence stays at its sentinel and every row that ran is finite."""
    probs = []
    st = outcome.state
    pos, pos0 = st.pos.cpu().tolist(), warm.pos.cpu().tolist()
    if not all(0 <= p < BT for p in pos):
        probs.append("a position left [0, 63]: %s" % [p for p in pos if not 0 <= p < BT][:5])
    ran = {}
    for n in range(step.N):
        if step.live(n):
            ran[step.slot(n)] = n
    e_changed = (bits(st.E) != bits(warm.E)).flatten(1).any(1).cpu().tolist()          # per slot: any bit of E / of the whole cache (stale rows too) changed
    r_changed = (bits(st.rows) != bits(warm.rows)).flatten(1).any(1).cpu().tolist()
    for s in range(st.N):
        if s in ran:
            want = (pos0[s] + step.lens[ran[s]]) % BT
            if pos[s] != want:
                probs.append("slot %d: position %d after %d rows from %d, expected %d" % (s, pos[s], step.lens[ran[s]], pos0[s], want))
        elif pos[s] != pos0[s] or e_changed[s] or r_changed[s]:
            probs.append("slot %d was not used by the launch and changed" % s)
    for n in range(step.N):
        first, count = step.rows_of(n)
        sel = (slice(n, n + 1), slice(0, count)) if step.fixed else (slice(0, 1), slice(first, first + count))
        o, o32 = outcome.out[sel], outcome.out32[sel]
        if step.live(n):
            if not (torch.isfinite(o.float()).all() and torch.isfinite(o32).all()):
                probs.append("sequence %d (slot %d) ran and left a non-finite output" % (n, step.slot(n)))
        elif count and not (torch.isnan(o.float()).all() and torch.isnan(o32).all()):
            probs.append("sequence %d (slot %d) did not run and its output rows were written" % (n, step.slot(n)))
    return probs


def check_step(ext, mod, step, warm):
    """Both arms from the same warm state: ``(optimised outcome, reference outcome)``; asserts the at-rest invariants on the optimised arm and bitwise equality with the oracle."""
    opt = run_arm(mod, OPTIMISED, warm, step)
    ref = run_arm(ext, REFERENCE, warm, step)
    probs = at_rest_problems(step, warm, opt)
    assert not probs, "; ".join(probs[:6])
    m = mismatch(opt, ref)
    assert all(v == 0 for v in m.values()), "the optimised row engine differs from the reference: %s" % m
    return opt, ref


def run_family(ext, fam, H, seed, mod=None):
    """Warm a state, then run ``fam.steps`` successive steps on it through both arms; every step is compared before the next one (the state evolves, positions wrap).
    Returns the list of ``(step, opt, ref, warm)``."""
    mod = ext if mod is None else mod
    rng = random.Random(seed * 1000 + H)
    par = make_params(H, seed)
    results = []
    step, S = draw_step(ext, fam, H, par, rng, seed * 10)
    warm = warm_pool(ext, S, H, draw_fills(rng, S, fam.fills), par, seed)
    for i in range(fam.steps):
        if i:
            step, S2 = draw_step(ext, fam, H, par, rng, seed * 10 + i)
            assert S2 == S
        assert step.live_ctas() >= WAVES * resident_ctas(ext), "the live grid is not beyond %.1f resident waves" % WAVES
        opt, ref = check_step(ext, mod, step, warm)
        results.append((step, opt, ref, warm))
        warm = ref.state.clone()     # the oracle's state continues (both arms agree, asserted above)
    return results


# ---- controls: one-bit corruptions of an outcome --------------------------------------------------------------------------------------------------------------
def corrupt(outcome, what, slot=None):
    """A copy of ``outcome`` with one bit flipped: 'out', 'out32', 'E', 'row' (a valid row of the cache), 'stale_row' (a row at or beyond ``pos``: not part of the state), 'pos'."""
    st = outcome.state.clone()
    out, out32 = outcome.out.clone(), outcome.out32.clone()
    if slot is None:
        slot = int((st.pos > 0).nonzero()[0])
    if what == "out":
        bits(out).view(-1)[7] ^= 1
    elif what == "out32":
        bits(out32).view(-1)[7] ^= 1
    elif what == "E":
        st.E.view(torch.int32)[slot, 0, 5, 9] ^= 1
    elif what == "row":
        assert int(st.pos[slot]) > 0
        st.rows.view(torch.int32)[slot, 0, 2, 0, 3] ^= 1
    elif what == "stale_row":
        st.rows.view(torch.int32)[slot, 0, 2, 63, 3] ^= 1   # row 63 is stale unless pos is 64, which it never is
    elif what == "pos":
        st.pos[slot] = (st.pos[slot] + 1) % BT
    else:
        raise ValueError(what)
    return Outcome(out, out32, st)


# ---- what a family reached (the guard against a vacuous family) --------------------------------------------------------------------------------------------------
def coverage(results):
    """The counts of what the steps of a family reached, for the assertions that keep a family from being vacuous: the largest grid and live grid (CTAs), padding and empty entries,
    rows that finished a chunk (position 63), rows that follow such a row in the same launch, slots of the pool that no sequence used, whether the slots came in a permuted order,
    sequences that continued an open chunk (position != 0) and the most rows of one sequence."""
    c = {"steps": len(results), "grid_ctas": 0, "live_ctas": 10 ** 9, "padding": 0, "empty": 0, "chunk_end_rows": 0, "rows_after_chunk_end": 0, "pool_unused": 0, "permuted": False,
         "open_chunk_sequences": 0, "live_sequences": 0, "max_rows": 0}
    for step, opt, ref, warm in results:
        pos0 = warm.pos.cpu().tolist()
        c["grid_ctas"] = max(c["grid_ctas"], step.N * step.H)
        c["live_ctas"] = min(c["live_ctas"], step.live_ctas())
        c["padding"] += sum(1 for n in range(step.N) if step.slot(n) < 0)
        c["empty"] += sum(1 for n in range(step.N) if step.slot(n) >= 0 and step.lens[n] == 0)
        used = [step.slot(n) for n in range(step.N) if step.slot(n) >= 0]
        c["pool_unused"] = max(c["pool_unused"], warm.N - len(used)) if step.slots is not None else 0
        c["permuted"] = c["permuted"] or (step.slots is not None and used != sorted(used))
        for n in range(step.N):
            if not step.live(n):
                continue
            c["live_sequences"] += 1
            c["open_chunk_sequences"] += 1 if pos0[step.slot(n)] != 0 else 0
            c["max_rows"] = max(c["max_rows"], step.lens[n])
            for r in range(step.lens[n]):
                if (pos0[step.slot(n)] + r) % BT == BT - 1:
                    c["chunk_end_rows"] += 1
                    c["rows_after_chunk_end"] += 1 if r < step.lens[n] - 1 else 0
    return c


# ---- timing of a step ---------------------------------------------------------------------------------------------------------------------------------------
def time_step(mod, impl, warm, step, reps):
    """Milliseconds of ``reps`` launches of the step (CUDA events around ``mod.forward`` only; every launch starts from its own clone of the warm state), after one launch that is not timed."""
    par = step.par
    times = []
    for i in range(reps + 1):
        state = warm.clone()
        out, out32 = outputs_like(step)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        mod.forward(step.q, step.k, step.v, step.g, step.beta, float(par["scale"]), out, par["A_log"], par["dt_bias"], float(par["lower_bound"]),
                    state.E, state.rows, state.pos, step.cu, 1, out32, impl, step.slot_ids)
        e1.record()
        torch.cuda.synchronize()
        if i:
            times.append(e0.elapsed_time(e1))
    return times
