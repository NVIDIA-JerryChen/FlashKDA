"""Host reference of the K2 schedule policy of the dynamic-schedule K2 (impl bit 16), plus the deterministic corpus of length lists.

K2 is a serial scan per (sequence, head, value slice); one CTA fills an SM, so the kernel time is the time of the busiest SM and depends on the ORDER
of the CTAs and on the tile WIDTH of every (sequence, head) pair. The device builds the schedule from the chunk counts of the plan; this module is the
specification it has to match exactly (same chosen candidate, same widths, same item order), in integer arithmetic, and it holds the exact
list-scheduling makespan that validates the policy. Plain Python (no torch), importable on a host without a GPU.

Model constants (``T_CHUNK``): the time one CTA needs per 64-token chunk at each tile width, in units of 0.1 K cycles. They are the measurement of the
earlier K2 work on B200 and B300 (docs/20261103-cfbi-perf.md; 3.4 : 4.7 : 7.8 K cycles, the constants of ``k2_grid_cost`` in csrc/cfbi/cfbi_launch.cu; the closed form
equals a brute-force list-scheduling simulation, and the predicted mixed-grid gain agrees with the measurement within about one point there). The
constants are NOT calibrated for Hopper (the model over-predicts the mixed-grid gain by about 3 points there); a per-architecture calibration is
future work. The SM count is an input everywhere, never a constant.

Definitions (the device implements the same, see csrc/cfbi/cfbi_k2_dyn.cu):
  * pair p = h * N + n for head h, sequence n; chunk count c[n] = nfull + (tail > 0); a pair is active iff c[n] > 0 (skipped / empty sequences have none).
  * a pair at width w owns 128 / w items (value slices); an item takes c * T(w) time units.
  * sorted pair list: active pairs by (c descending, p ascending); a candidate (a, b, x) gives the first x pairs of that list width a, the others width b.
  * total item order of a sorted schedule: key = c * T(width) descending, then p ascending, then value slice ascending.
  * old schedule: the pairs in natural order (p ascending, slice ascending) with the widths the old launcher used (call type dependent): a variable-length call
    8 columns if N * H * 16 CTAs fit one wave of SMs, else 16; a fixed-length call the closed form of ``choose_k2_tiles(N * H, SMs)`` laid out over ALL N * H
    pair positions, 32 columns for the pairs at absolute positions p < x32_old and bvb_old columns for the others (the pairs of skipped sequences are absent).
  * a fixed-length call has equal sequence lengths: its chunk counts differ only by the head rows of a warm sequence (at most one chunk) and by skipped slots (no
    chunks), so the closed form is exact up to one chunk per wave. With every active count equal the schedule is the closed form of the active pairs; with
    unequal counts it is the exact old schedule (no candidate search: the old grid is the no-regression reference of the call type).
  * unequal lengths of a variable-length call: candidates (a, b, x) over a fixed grid of split positions are compared with the reference schedule R = the sorted
    order at the old width.
    The bounds are those of list scheduling with S SMs: for a schedule in non-increasing duration order max(dmax, W / S + (1 - 1 / S) * d_(S+1)) is an
    upper bound (``bound_ub``), max(dmax, W / S, d_S + d_(S+1)) a lower bound of every schedule (``bound_pig``), W / S + (1 - 1 / S) * dmax an upper bound
    of every list order (``bound_graham``). With many waves of items the upper bounds are compared (a candidate must be 3 % better); with few waves a
    candidate must be certified better. R replaces the old order if its upper bound is 3 % below the old order's Graham bound.
"""
import heapq
import itertools
import random

HEAD_V = 128
T_CHUNK = {8: 34, 16: 47, 32: 78}
GRID_INTERVALS = 16        # split positions of the candidates: x = (i * pairs) // GRID_INTERVALS, i = 0 .. GRID_INTERVALS
ACCEPT_PCT = 97            # a candidate replaces the reference schedule only if its bound is at most this percentage of the reference's
REORDER_PCT_FEW = 90       # with few waves the sorted order at the old width replaces the old order only if its upper bound is this percentage of the old one's
MANY_WAVES = 3             # the reference schedule has more than MANY_WAVES * S items: the LPT bound is accurate, bounds are compared directly; with fewer
                           # items (the last wave decides) a candidate must be certified: its upper bound is at most the reference's lower bound
SCHEDULE_MAX_SEQUENCES = 1024
SCHEDULE_MAX_PAIRS = 16896  # 512 sequences x 33 heads
ITEMS_PER_PAIR_MAX = HEAD_V // 8
ROUTE_COMPUTE_CAPABILITY = 10      # default routing (``route_default``): on a Blackwell device (the cost model is measured on B200 / B300) ...
ROUTE_MIN_TOKENS_PER_SEQ = 256     # ... on average at least four chunks per sequence (a CTA with one chunk of work does not repay the table read) ...
ROUTE_OVERHEAD_PCT = 8             # ... and the device policy, assumed to meet equal-length sequences (the worst case: skew only helps), must be predicted to save
ROUTE_UNIT_X10NS = 526             # more than the schedule builder and this much extra time of the table-driven kernel (percent of the old K2 stage). One unit of
ROUTE_CTA_X10NS = 1064             # ``grid_cost`` per chunk is 52.6 ns at the K2 clock; one CTA round also costs about 5 us (per unit of cost: 106.4 ns); times in 0.1 ns.
ROUTE_MAX_TOKENS_PER_SEQ = 1 << 20  # (the estimate saturates here: more tokens never switch the routing off, and the C++ arithmetic stays inside 64 bits)
ROUTE_MAX_SEQUENCES = 512          # the builder takes up to 1024 sequences, but its time (68 us at 1024 against 29 us at 512) and the table-driven grid lose the saving there


def schedule_capacity(H):
    """Largest number of sequences the device schedule builder handles for ``H`` heads (more: the launcher runs the old kernels)."""
    return min(SCHEDULE_MAX_SEQUENCES, SCHEDULE_MAX_PAIRS // H) if H > 0 else 0


def route_builder_x10ns(N, pairs):
    """Duration of the schedule builder (units of 0.1 ns) for ``N`` sequences and ``pairs`` = N * H pairs: an upper envelope of the medians measured on B200 / B300. It
    grows with both (the scan over the sequences and the item table of the pairs): the larger of the two estimates."""
    by_sequences = 55000 if N <= 4 else 70000 if N <= 16 else 95000 if N <= 32 else 120000 if N <= 128 else 160000 if N <= 256 else 300000
    by_pairs = 55000 if pairs <= 24 else 75000 if pairs <= 128 else 95000 if pairs <= 192 else 120000 if pairs <= 768 else 160000 if pairs <= 1536 else 300000
    return max(by_sequences, by_pairs)


def route_default(N, H, T_total, sms, cc_major=ROUTE_COMPUTE_CAPABILITY):
    """Whether a variable-length prefill call of the default implementation (impl 0, no forced tile width, mode 0) runs the device-scheduled K2
    (``k2_dyn_default_route`` in csrc/cfbi/cfbi_launch.cu). Only host-known quantities: ``N`` sequences, ``H`` heads, ``T_total`` tokens, the SM count.

    The host does not see the lengths. For equal lengths the device policy is the closed form of the fixed-length launcher (``choose_k2_tiles``), and equal
    lengths are the worst case for it, so a call is routed when that grid is predicted to beat the old rule's by more than the cost of the builder and
    ROUTE_OVERHEAD_PCT of the old K2 stage:  (co - cd) / co * stage - overhead * stage >= builder, with the old K2 stage estimated from the old grid's cost co
    (``grid_cost``, per chunk) and the tokens per sequence; the allowance covers the table-driven kernel (2 - 5 % slower on the same grid) and the error of the model
    (about 3 points). Up to one wave of SMs at 8 columns (nothing to order or choose, no builder runs), with fewer than
    ROUTE_MIN_TOKENS_PER_SEQ tokens per sequence, with more than ROUTE_MAX_SEQUENCES sequences (or beyond the capacity of the builder), or when the closed form picks the
    old grid, the old kernels run."""
    if cc_major != ROUTE_COMPUTE_CAPABILITY:
        return False
    nh = N * H
    if N < 1 or H < 1 or N > min(schedule_capacity(H), ROUTE_MAX_SEQUENCES) or nh * ITEMS_PER_PAIR_MAX <= sms or T_total < ROUTE_MIN_TOKENS_PER_SEQ * N:
        return False
    x_old, bvb_old = old_rule(N, H, sms, False)
    x_dyn, bvb_dyn = choose_k2_tiles(nh, sms)
    if (x_dyn, bvb_dyn) == (x_old, bvb_old):
        return False
    co, cd = grid_cost(nh, x_old, bvb_old, sms), grid_cost(nh, x_dyn, bvb_dyn, sms)
    tokens_per_seq = min(T_total // N, ROUTE_MAX_TOKENS_PER_SEQ)
    stage = co * (ROUTE_UNIT_X10NS * tokens_per_seq // 64 + ROUTE_CTA_X10NS)
    return (100 * (co - cd) - ROUTE_OVERHEAD_PCT * co) * stage >= 100 * co * route_builder_x10ns(N, nh)


def chunk_count(length, pos0=0):
    """Chunk count of the plan kernel for ``length`` tokens that start at open-chunk fill ``pos0`` (csrc/cfbi/cfbi_plan.cuh)."""
    hrows = 0 if pos0 == 0 else min(length, 64 - pos0)
    rest = length - hrows
    return rest // 64 + (1 if rest % 64 > 0 else 0)


# ---- the closed forms of the fixed-length launcher (csrc/cfbi/cfbi_launch.cu) ---------------------------------------------------------------------
def grid_cost(nh, x32, bvb, sms):
    """Busiest SM's time per chunk (units of 0.1 K cycles) for ``nh`` equal pairs, the first ``x32`` with 32-column tiles, the rest ``bvb``-column tiles,
    CTAs dispatched in grid order (32-column tiles first) to the SM that frees first. Exact for equal lengths (``k2_grid_cost``)."""
    tl, ts = T_CHUNK[32], T_CHUNK[bvb]
    nl, ns = 4 * x32, (nh - x32) * (HEAD_V // bvb)
    a, r = divmod(nl, sms)
    f0, f1 = a * tl, (a + 1) * tl
    busy_long = f1 if r > 0 else f0
    if ns == 0:
        return busy_long

    def slots(t):
        return (sms - r) * (((t - f0) // ts + 1) if t >= f0 else 0) + r * (((t - f1) // ts + 1) if t >= f1 else 0)

    lo, hi = f0, f1 + ns * ts
    while lo < hi:
        mid = (lo + hi) // 2
        if slots(mid) >= ns:
            hi = mid
        else:
            lo = mid + 1
    return max(lo + ts, busy_long)


def choose_k2_tiles(nh, sms):
    """Tile widths of a call of ``nh`` equal-length pairs: ``(x32, bvb)`` = the first x32 pairs 32 columns, the others bvb columns (``choose_k2_tiles``)."""
    best, bx, bw = grid_cost(nh, 0, 8, sms), 0, 8
    c16, c32 = grid_cost(nh, 0, 16, sms), grid_cost(nh, nh, 16, sms)
    if c16 < best:
        best, bx, bw = c16, 0, 16
    if c32 < best:
        best, bx, bw = c32, nh, 16
    mixed, mx = best, -1
    k = 1
    while 4 * nh > k * sms:
        x = min(nh - 1, k * sms // 4)
        if x >= 1:
            c = grid_cost(nh, x, 16, sms)
            if c < mixed:
                mixed, mx = c, x
        k += 1
    if mx > 0 and mixed * 100 <= best * 97:
        bx, bw = mx, 16
    return bx, bw


def old_rule(N, H, sms, call_fixed):
    """Widths of the old launcher: ``(x32, bvb)`` over the N * H pair positions. A fixed-length call (no cu_seqlens) used the equal-length closed form; a
    variable-length call 8 columns if N * H * 16 CTAs fit one wave of SMs, else 16."""
    if call_fixed:
        return choose_k2_tiles(N * H, sms)
    return 0, (8 if N * H * 16 <= sms else 16)


# ---- pairs, estimates, candidates ----------------------------------------------------------------------------------------------------------------
def active_pairs_sorted(c, H):
    """The active pairs as ``(chunks, p)``, c descending then p ascending."""
    N = len(c)
    pairs = [(c[p % N], p) for p in range(N * H) if c[p % N] > 0]
    pairs.sort(key=lambda cp: (-cp[0], cp[1]))
    return pairs


def _kth_largest_pair(durations_runs, ks):
    """The k-th largest item duration for every k in ``ks`` (ascending), from runs ``(duration, count)`` in non-increasing duration order (0: too few items)."""
    out = []
    seen = 0
    it = iter(durations_runs)
    cur = next(it, None)
    for k in ks:
        while cur is not None and seen + cur[1] < k:
            seen += cur[1]
            cur = next(it, None)
        out.append(cur[0] if cur is not None else 0)
    return out


def _merged_runs(pairs, a, b, x):
    """Runs (duration, count) of the candidate (a, b, x) in non-increasing duration order (a lazy merge of the two groups, which are each sorted)."""
    ma, mb = HEAD_V // a, HEAD_V // b
    ta, tb = T_CHUNK[a], T_CHUNK[b]
    A = ((pairs[i][0] * ta, ma) for i in range(x))
    B = ((pairs[j][0] * tb, mb) for j in range(x, len(pairs)))
    return heapq.merge(A, B, key=lambda r: -r[0])


def candidate_stats(pairs, a, b, x, sms, prefix=None):
    """``(W, dmax, dS, dS1)``: total work, longest item, S-th and (S+1)-th longest item (0 when the candidate has fewer items). ``prefix`` = running sums
    of the chunk counts of ``pairs`` (optional, avoids re-summing)."""
    ma, mb = HEAD_V // a, HEAD_V // b
    ta, tb = T_CHUNK[a], T_CHUNK[b]
    P = len(pairs)
    if prefix is None:
        prefix = [0] + list(itertools.accumulate(cp[0] for cp in pairs))
    cA = prefix[x]
    cB = prefix[P] - prefix[x]
    W = ma * ta * cA + mb * tb * cB
    dmax = max(pairs[0][0] * ta if x > 0 else 0, pairs[x][0] * tb if x < P else 0)
    dS, dS1 = _kth_largest_pair(_merged_runs(pairs, a, b, x), (sms, sms + 1))
    return W, dmax, dS, dS1


def _ceil_div(a, b):
    return -(-a // b)


def bound_ub(S, W, dmax, dS, dS1):
    """Upper bound of the list-scheduling makespan of a schedule in non-increasing duration order (LPT): max(dmax, W / S + (1 - 1 / S) * d_(S+1))."""
    return max(dmax, _ceil_div(W, S) + ((S - 1) * dS1) // S)


def bound_pig(S, W, dmax, dS, dS1):
    """Lower bound of any schedule: max(dmax, W / S, d_S + d_(S+1)) (some SM runs two of the S + 1 longest items)."""
    return max(dmax, _ceil_div(W, S), dS + dS1)


def bound_graham(S, W, dmax, dS, dS1):
    """Upper bound of the makespan of a list schedule in ANY order (Graham): W / S + (1 - 1 / S) * dmax."""
    return _ceil_div(W, S) + ((S - 1) * dmax) // S


def candidate_list(pairs_count):
    """Candidates (a, b, x) in the fixed enumeration order of the device: the homogeneous widths 8, 16, 32 (x = 0), then every mixed width pair
    (a != b, a-major) with x = (i * pairs) // GRID_INTERVALS for i = 0 .. GRID_INTERVALS."""
    cands = [(w, w, 0) for w in (8, 16, 32)]
    for a, b in itertools.product((8, 16, 32), (8, 16, 32)):
        if a == b:
            continue
        for i in range(GRID_INTERVALS + 1):
            cands.append((a, b, (i * pairs_count) // GRID_INTERVALS))
    return cands


# ---- the schedule ----------------------------------------------------------------------------------------------------------------------------------
class Schedule(object):
    """Result of ``build_schedule``: ``kind`` 'old' | 'sorted', ``choice`` (a, b, x) in sorted-pair space (None for old widths of an unequal list),
    ``items`` [(p, slice, width)] in execution order, ``widths`` {p: width}, ``trace`` the numbers the decision used."""

    def __init__(self, kind, choice, items, widths, trace):
        self.kind, self.choice, self.items, self.widths, self.trace = kind, choice, items, widths, trace

    def width_counts(self):
        out = {8: 0, 16: 0, 32: 0}
        for w in self.widths.values():
            out[w] += 1
        return out


def _natural_items(c, H, x32, bvb):
    N = len(c)
    items, widths = [], {}
    for p in range(N * H):
        if c[p % N] == 0:
            continue
        w = 32 if p < x32 else bvb
        widths[p] = w
        items.extend((p, s, w) for s in range(HEAD_V // w))
    return items, widths


def _sorted_items(c, pairs, widths_by_position):
    """Items in the total order key (chunks * T(width)) descending, p ascending, slice ascending."""
    ent = []
    widths = {}
    for i, (cc, p) in enumerate(pairs):
        w = widths_by_position[i]
        widths[p] = w
        ent.append((-cc * T_CHUNK[w], p, w))
    ent.sort()
    items = []
    for _, p, w in ent:
        items.extend((p, s, w) for s in range(HEAD_V // w))
    return items, widths


def build_schedule(c, H, sms, call_fixed=False, forced_bv=0):
    """The schedule the device builds for chunk counts ``c`` (one per sequence; 0 = no work), ``H`` heads on ``sms`` SMs.
    ``call_fixed``: the call had no cu_seqlens (the old schedule is then the equal-length closed form). ``forced_bv`` 8 / 16 / 32: every pair that width."""
    N = len(c)
    pairs = active_pairs_sorted(c, H)
    Pa = len(pairs)
    x32_old, bvb_old = old_rule(N, H, sms, call_fixed)
    trace = {"pairs": Pa, "old": (x32_old, bvb_old)}
    if Pa == 0:
        return Schedule("old", None, [], {}, trace)
    if forced_bv:
        items, widths = _sorted_items(c, pairs, [forced_bv] * Pa)
        trace["choice"] = (forced_bv, forced_bv, 0)
        return Schedule("sorted", (forced_bv, forced_bv, 0), items, widths, trace)
    if pairs[0][0] == pairs[-1][0]:                       # every active list entry has the same chunk count: the closed form is exact
        x32, bvb = choose_k2_tiles(Pa, sms)
        x32_old_active = sum(1 for _, p in pairs if p < x32_old)
        kind = "old" if (x32, bvb) == (x32_old_active, bvb_old) else "sorted"
        items, widths = _sorted_items(c, pairs, [32 if i < x32 else bvb for i in range(Pa)])
        trace["choice"] = (32, bvb, x32)
        return Schedule(kind, (32, bvb, x32), items, widths, trace)
    if call_fixed:
        # a fixed-length call with unequal active chunk counts (warm open-chunk positions of the sequences; skipped slots are absent): the exact old schedule,
        # the closed form of the launcher over all N * H pair positions in natural order. The candidate search below is for variable-length calls.
        items, widths = _natural_items(c, H, x32_old, bvb_old)
        trace["items_old"] = len(items)
        return Schedule("old", None, items, widths, trace)
    S = sms
    w0 = bvb_old
    cands = candidate_list(Pa)
    prefix = [0] + list(itertools.accumulate(cp[0] for cp in pairs))
    stats = {}
    for cand in cands:
        a, b, x = cand
        stats[cand] = candidate_stats(pairs, a, b, x, S, prefix)
    ref = (w0, w0, 0)
    ub_ref = bound_ub(S, *stats[ref])
    pig_ref = bound_pig(S, *stats[ref])
    many_waves = Pa * (HEAD_V // w0) > MANY_WAVES * S
    best = None
    for idx, cand in enumerate(cands):
        st = stats[cand]
        ub = bound_ub(S, *st)
        accepted = 100 * ub <= ACCEPT_PCT * ub_ref if many_waves else ub <= pig_ref
        if accepted:
            key = (bound_pig(S, *st), ub, idx)
            if best is None or key < best[0]:
                best = (key, cand)
    # old widths, natural order: all pairs 8 or 16 columns (only variable-length calls reach this branch, x32_old is 0)
    old_items, old_widths = _natural_items(c, H, x32_old, bvb_old)
    n_old = len(old_items)
    W_old = sum(c[p % N] * T_CHUNK[w] for p, _, w in old_items)
    dmax_old = max(c[p % N] * T_CHUNK[w] for p, _, w in old_items)
    ub_old = bound_graham(S, W_old, dmax_old, 0, 0)
    trace.update({"ub_ref": ub_ref, "ub_old": ub_old, "items_old": n_old})
    if best is not None:
        a, b, x = best[1]
        items, widths = _sorted_items(c, pairs, [a if i < x else b for i in range(Pa)])
        trace["choice"] = best[1]
        trace["rank"] = best[0]
        return Schedule("sorted", best[1], items, widths, trace)
    if 100 * ub_ref <= (ACCEPT_PCT if many_waves else REORDER_PCT_FEW) * ub_old:
        items, widths = _sorted_items(c, pairs, [w0] * Pa)
        trace["choice"] = ref
        return Schedule("sorted", ref, items, widths, trace)
    return Schedule("old", None, old_items, old_widths, trace)


# ---- exact validation ------------------------------------------------------------------------------------------------------------------------------
def list_makespan(items, c, sms, N):
    """Exact makespan of greedy list scheduling of ``items`` (in the given order) on ``sms`` SMs: each item starts on the SM that frees first."""
    free = [0] * sms
    for p, _, w in items:
        heapq.heapreplace(free, free[0] + c[p % N] * T_CHUNK[w])
    return max(free) if items else 0


def check_invariants(items, c, H, kind):
    """Structural invariants of a schedule (returns a list of problems, empty when fine): every active (pair, slice) exactly once, nothing for inactive
    pairs, one width per pair, valid slice range, and the total order of the kind (sorted: key descending, p ascending, slice ascending; old: natural)."""
    N = len(c)
    problems = []
    seen = set()
    width = {}
    for p, s, w in items:
        if not (0 <= p < N * H) or c[p % N] == 0:
            problems.append("entry for an inactive or out-of-range pair %d" % p)
            continue
        if w not in (8, 16, 32) or not (0 <= s < HEAD_V // w):
            problems.append("bad slice %d of width %d" % (s, w))
        if width.setdefault(p, w) != w:
            problems.append("pair %d has two widths" % p)
        if (p, s) in seen:
            problems.append("duplicate entry (%d, %d)" % (p, s))
        seen.add((p, s))
    for p in range(N * H):
        if c[p % N] > 0:
            w = width.get(p)
            if w is None:
                problems.append("missing pair %d" % p)
            else:
                for s in range(HEAD_V // w):
                    if (p, s) not in seen:
                        problems.append("missing entry (%d, %d)" % (p, s))
    if not problems:
        if kind == "sorted":
            keys = [(-c[p % N] * T_CHUNK[w], p, s) for p, s, w in items]
        else:
            keys = [(p, s) for p, s, w in items]
        if keys != sorted(keys):
            problems.append("items are not in the total order of the %s schedule" % kind)
    return problems


# ---- the deterministic corpus of length lists ------------------------------------------------------------------------------------------------------
CORPUS_SEED = 20261104
CORPUS_N = (1, 2, 3, 4, 6, 8, 13, 16, 19, 32, 64)
CORPUS_H = (1, 3, 6, 12, 32)
CORPUS_SMS = (132, 144, 148)
_RANDOM_KINDS = ("uniform", "lognormal0.5", "lognormal1.0", "lognormal1.5")
_FIXED_KINDS = ("equal", "onelong4", "onelong8", "onelong32", "half", "twolong")
CORPUS_DRAWS = 24          # lists per (random kind, N)


def _chunks_to_length(rng, chunks):
    """Token length of a sequence of ``chunks`` 64-token chunks; a random tail of 0 .. 63 tokens is added to every other sequence to exercise the tail."""
    return chunks * 64 + (rng.randint(0, 63) if rng.random() < 0.5 else 0)


def _list_of(kind, N, rng):
    """Chunk counts of one length list: lengths of the named distribution in 64-token units (8192 tokens = 128 chunks)."""
    if kind == "equal":
        return [128] * N
    if kind == "uniform":
        return [rng.randint(32, 128) for _ in range(N)]
    if kind.startswith("lognormal"):
        sigma = float(kind[len("lognormal"):])
        return [max(1, min(128, int(rng.lognormvariate(4.2, sigma)))) for _ in range(N)]
    if kind.startswith("onelong"):
        ratio = int(kind[len("onelong"):])
        return [128] + [max(1, 128 // ratio)] * (N - 1)
    if kind == "half":
        return [128] * ((N + 1) // 2) + [16] * (N // 2)
    if kind == "twolong":
        return ([128, 128] + [16] * (N - 2)) if N >= 3 else [128] * N
    raise ValueError(kind)


def corpus_p(seed=CORPUS_SEED):
    """The policy corpus: a list of dicts ``{"name", "kind", "N", "lengths"}`` (token lengths, N sequences each), deterministic from ``CORPUS_SEED``.
    Every (kind, N) of the fixed kinds once and ``CORPUS_DRAWS`` draws of every (random kind, N): at least 1000 lists; each is evaluated with every H of
    ``CORPUS_H`` and every SM count of ``CORPUS_SMS``."""
    rng = random.Random(seed)
    out = []
    for kind in _FIXED_KINDS + _RANDOM_KINDS:
        for N in CORPUS_N:
            draws = CORPUS_DRAWS if kind in _RANDOM_KINDS else 1
            for d in range(draws):
                chunks = _list_of(kind, N, rng)
                lengths = [_chunks_to_length(rng, ch) if kind in _RANDOM_KINDS else ch * 64 for ch in chunks]
                out.append({"name": "%s/N%d/%d" % (kind, N, d), "kind": kind, "N": N, "lengths": lengths})
    return out
