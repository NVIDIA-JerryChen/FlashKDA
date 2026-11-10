"""Runtime placement and bounded windows; no input metadata is read on host."""

import functools
from .schedule import SEGMENT


def _rows(n, h, sms):
    """Rows per lane from host shapes only (no device metadata)."""
    pairs = n * h
    cohorts = (pairs * 8 + sms - 1) // sms
    # Few request-head pairs: eight CTAs per pair spread the recurrence over
    # more SMs. Otherwise group CTAs into occupancy cohorts.
    if pairs * 32 <= sms * 7:
        return 2
    return 2 if cohorts in (1, 3) else 4


def _long_ranks(n, h, sms):
    """Leading (longest) requests that get the two-row engine while every CTA
    of the launch still fits one per SM; from host shapes only."""
    return max(0, min(n, (sms // h - 4 * n) // 4))


def _mixed_ranks(n, h, sms):
    """Number of requests using two-row consumers within two SM waves."""
    if 4 * n * h <= sms:
        return 0
    best = 0
    for count in range(1, n):
        if h * (8 * count + 4 * (n - count)) <= 2 * sms:
            best = count
    return best if best >= 2 else 0


def _fill_ranks(n, h, sms):
    """Fill the second wave with two-row consumers for the longest requests."""
    wave = 3 * sms
    ctas = 4 * n * h
    if not wave < ctas <= 2 * wave:
        return 0
    if n <= 4:
        return 1
    if h > 12:
        return 0
    return max(0, min(n - 1, (2 * wave - ctas) // (4 * h)))


def _window_cost(slots, segments, n):
    """Sum over windows of the busiest request's segment count (round-robin
    ranks, equal-length requests) plus a relaunch overhead per window."""
    cost = 0.0
    for start in range(0, segments, slots):
        cost += (min(slots, segments - start) + n - 1) // n + 0.5
    return cost


FIRST_ROUNDS = 2


FEW_REQUEST_FIRST_ROUNDS = 3


FEW_REQUESTS = 4


PROPORTIONAL_REQUESTS = 12


SPLIT_TOKEN_HEADS = 100000


@functools.lru_cache(maxsize=256)
def _plan_windows(limit, capacity, n, total, heads, many_waves=False):
    """Partition the bounded workspace into ordered, nonoverlapping adjacent windows."""
    bound = max(1, min(capacity, (total + 127 * n) // SEGMENT))
    if bound <= limit and total * heads < SPLIT_TOKEN_HEADS:
        return [(0, bound, 0, 1)]
    if limit < 2:
        return [(start, 1, 0, 1) for start in range(bound)]
    half = limit // 2
    rounds = FEW_REQUEST_FIRST_ROUNDS if n <= FEW_REQUESTS else FIRST_ROUNDS
    first = max(1, min(bound, rounds * n, half - half % n if half >= n else half))
    rest = bound - first
    if many_waves and rest - (limit - first) > min(n, first):
        # Grids beyond one resident wave whose rest needs several windows of
        # at most half the workspace: each of those windows' coefficients
        # hides behind a one-round scan, so the first window (exposed
        # coefficients) keeps one segment per request.
        first = max(1, min(bound, n, half))
        rest = bound - first
    sizes = [first]
    if 0 < rest <= limit - first:
        sizes.append(rest)
    elif 0 < rest - (limit - first) <= min(n, first):
        # The bound overshoots the real segment count by up to N - 1, so a
        # small overflow gets its own (usually empty) last window instead of
        # splitting the run into more full windows.
        sizes += [limit - first, rest - (limit - first)]
    elif rest > 0:
        # Windows of at most half the workspace. Windows take segments
        # round-robin across requests, so the size minimizing the modelled
        # cost (_window_cost: busiest request's segments plus a relaunch per
        # window) is chosen among even splits and multiples of N.
        least = (rest + half - 1) // half
        candidates = {half} | {(rest + k - 1) // k for k in range(least, least + 3)}
        candidates |= {n * j for j in range(max(1, half // n - 2), half // n + 1)}
        size = min(
            (c for c in candidates if 0 < c <= half),
            key=lambda c: (_window_cost(c, rest, n), -c),
        )
        windows = (rest + size - 1) // size
        sizes += [size] * (windows - 1) + [rest - size * (windows - 1)]
        if len(sizes) >= 3 and sizes[-1] + sizes[-2] <= half:
            sizes[-2:] = [sizes[-2] + sizes[-1]]
    plan, start = [], 0
    for index, count in enumerate(sizes):
        offset = 0 if index % 2 == 0 else limit - count
        wait = 1 if index == 0 or sizes[index - 1] + count > limit else 0
        plan.append((start, count, offset, wait))
        start += count
    return plan
