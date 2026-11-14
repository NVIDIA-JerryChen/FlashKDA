"""Bind bounded FP32 coefficient windows and ordered recurrent scans."""

from dataclasses import dataclass
import torch
from .aligned_induction_coeff import compiled_coeff_kernel
from .shared_stage_scan import (
    compiled_scan_kernel,
    KPACK,
    PAIR_ENGINE,
    LONG_LAST,
    OVERLAP,
    SMEM_BYTES,
    SMEM_TWO_PER_SM,
    TWO_AHEAD,
)
from .schedule import SEGMENT, MAX_REQUESTS, compiled_schedule
from .policy import _rows, _long_ranks, _mixed_ranks, _fill_ranks, _plan_windows

OCTA_OVERLAP_MAX = 10
OVERLAP_MAX = 7
LONG_REQUEST_TOKENS = 512
TAIL_WINDOW_ROUNDS = 5
OCTA_SPREAD_WINDOWS = 5


def _octa(n, h, sms):
    # Compare estimated resident-wave costs of four- and eight-row consumers.
    pairs, wave = n * h, 3 * sms
    if pairs <= wave:
        return False
    return 1.6 * (-(-2 * pairs // wave)) < 0.97 * (-(-4 * pairs // wave))


@dataclass(frozen=True)
class WindowPlan:
    calls: tuple
    workspace_bytes: int

    def run(self):
        for kernel, arguments in self.calls:
            kernel(*arguments)


@torch.no_grad()
def prepare(
    q,
    k,
    v,
    g,
    beta,
    a_log,
    dt_bias,
    state_in,
    cu_seqlens,
    out,
    state_out,
    workspace_limit_bytes,
    *,
    scale,
    eps,
    lower_bound,
):
    n, h = state_in.shape[:2]
    total = q.shape[1]
    if n > MAX_REQUESTS:
        raise ValueError(f"prefill supports at most {MAX_REQUESTS} requests")
    if n == 0:
        return WindowPlan((), 0)
    capacity = (total + SEGMENT - 1) // SEGMENT + n
    metadata_bytes = (5 * n + 1 + 4 * capacity + 8 * n * h + 32 * (capacity + 2)) * 4
    row_bytes = SEGMENT * h * 392 * 4
    slots = min(capacity, (workspace_limit_bytes - metadata_bytes) // row_bytes)
    if slots < 1:
        raise ValueError("workspace cannot hold metadata and one scheduled segment")
    capability = torch.cuda.get_device_capability(q.device)
    index = q.device.index
    state = dict(
        coeff=compiled_coeff_kernel(index, capability),
        scan=compiled_scan_kernel(index, capability),
        schedule=compiled_schedule(index, capability),
        workspace=torch.empty(
            slots * SEGMENT * h * 392, device=q.device, dtype=torch.float32
        ),
        prefix=torch.empty(n + 1, device=q.device, dtype=torch.int32),
        ranks=torch.empty(capacity, device=q.device, dtype=torch.int32),
        tasks=torch.zeros(capacity, 3, device=q.device, dtype=torch.int32),
        order=torch.empty(4 * n, device=q.device, dtype=torch.int32),
        flags=torch.zeros(
            8 * n * h + 32 * (capacity + 2), device=q.device, dtype=torch.int32
        ),
        capacity=capacity,
        max_requests=n,
        sms=torch.cuda.get_device_properties(q.device).multi_processor_count,
    )
    calls = bind(
        q,
        k,
        v,
        g,
        beta,
        a_log,
        dt_bias,
        state_in,
        cu_seqlens,
        state,
        out,
        state_out,
        float(scale),
        float(eps),
        float(lower_bound),
    )
    # Bound tensor views in calls retain all allocations throughout graph lifetime.
    return WindowPlan(calls, slots * row_bytes + metadata_bytes)


def bind(
    q,
    k,
    v,
    g,
    beta,
    a_log,
    dt_bias,
    state_in,
    cu_seqlens,
    state,
    output,
    state_out,
    scale,
    eps,
    lower_bound,
):
    calls = []
    p = state
    n, h = state_in.shape[:2]
    total = q.shape[1]
    capacity = (total + SEGMENT - 1) // SEGMENT + n
    assert capacity <= p["capacity"] and n <= p["max_requests"]
    prefix, order = (p["prefix"][: n + 1], p["order"][: 4 * n])
    slots = min(capacity, p["workspace"].numel() // (SEGMENT * h * 392))
    long_ranks = _long_ranks(n, h, p["sms"])
    if long_ranks == 0:
        long_ranks = _mixed_ranks(n, h, p["sms"])
    if long_ranks == 0 and _rows(n, h, p["sms"]) == 4:
        long_ranks = _fill_ranks(n, h, p["sms"])
    rows = 4 if long_ranks else _rows(n, h, p["sms"])
    long_pairs = long_ranks * h
    last = 0
    if (
        rows == 2
        and long_pairs == 0
        and (p["sms"] < 4 * n * h <= 2 * p["sms"])
        and ((n * h * 8 + p["sms"] - 1) // p["sms"] == 3)
    ):
        rows, long_pairs, last = (4, n * h - p["sms"] // 4, LONG_LAST)
    if rows == 4 and long_ranks == 0 and _octa(n, h, p["sms"]):
        rows = 8
    long_batch = total >= LONG_REQUEST_TOKENS * n
    if (
        rows == 4
        and long_pairs
        and (_long_ranks(n, h, p["sms"]) == 0)
        and (_mixed_ranks(n, h, p["sms"]) == 0)
        and (5 * p["sms"] < 4 * n * h <= 6 * p["sms"])
        and (
            sum(
                (
                    count >= n
                    for _, count, _, _ in _plan_windows(
                        slots, capacity, n, total, h, True
                    )
                )
            )
            >= (4 if long_batch else 9)
        )
    ):
        rows, long_pairs = (8, 0)
    if (
        rows == 4
        and long_pairs == 0
        and (6 * p["sms"] < 4 * n * h <= OCTA_OVERLAP_MAX * p["sms"])
        and (
            sum(
                (
                    count >= n
                    for _, count, _, _ in _plan_windows(
                        slots, capacity, n, total, h, True
                    )
                )
            )
            >= (4 if long_batch else 10)
        )
    ):
        rows = 8
    if (
        long_batch
        and rows == 4
        and long_pairs
        and (5 * p["sms"] < 4 * n * h <= 6 * p["sms"])
    ):
        rows, long_pairs = (8, 0)
        slots = min(slots, 4 * n)
    ctas = long_pairs * 8 + (n * h - long_pairs) * (128 // (8 * rows))
    tail = (
        long_pairs == 0
        and rows == 4
        and (4 * n * h > 2 * p["sms"])
        and (h * (4 * n + 4) <= 3 * p["sms"])
    )
    if tail:
        long_pairs, ctas = (h, h * (4 * n + 4))
    radix4 = 3 if ctas <= 2 * p["sms"] else 2 if rows == 4 else 4
    if tail:
        radix4 |= 16
    radix4 |= 64 | last
    if rows == 2 or long_pairs:
        radix4 |= TWO_AHEAD
    if rows == 4 and long_pairs == 0 and (ctas <= p["sms"]):
        radix4 |= KPACK
    if long_pairs and ctas <= p["sms"]:
        radix4 |= PAIR_ENGINE
    smem = SMEM_BYTES
    if ctas <= 2 * p["sms"] and (long_pairs == 0 or last):
        smem = SMEM_TWO_PER_SM
    plan = _plan_windows(slots, capacity, n, total, h, ctas > 3 * p["sms"])
    if tail and len(plan) >= 4 and (slots > TAIL_WINDOW_ROUNDS * 2 * n):
        slots = TAIL_WINDOW_ROUNDS * 2 * n
        plan = _plan_windows(slots, capacity, n, total, h, ctas > 3 * p["sms"])
    sms = p["sms"]
    if rows == 8 and ctas <= 3 * sms and (len(plan) <= OCTA_SPREAD_WINDOWS):
        smem = SMEM_TWO_PER_SM
    overlap = 2.5 * sms < ctas <= OVERLAP_MAX * sms and sum(
        (count >= n for _, count, _, _ in plan)
    ) >= (3 if ctas <= 6 * sms else 10)
    if overlap:
        radix4 |= OVERLAP
    flags = p["flags"][:ctas]
    base = -(-ctas // 32) * 32
    counters = p["flags"][base : base + 32 * len(plan)]
    calls.append(
        (
            p["schedule"],
            (
                cu_seqlens,
                prefix,
                p["ranks"],
                p["tasks"],
                order,
                1,  # Length-proportional ranks for every runtime batch.
                p["flags"],
                base + 32 * len(plan) if overlap else 0,
            ),
        )
    )
    row = SEGMENT * h * 392
    for index, (start, count, offset, wait) in enumerate(plan):
        workspace = p["workspace"][offset * row : (offset + count) * row]
        if overlap and index > 0:
            wait += 2
        calls.append(
            (
                p["coeff"],
                (
                    q,
                    k,
                    g,
                    beta,
                    a_log,
                    dt_bias.view(-1),
                    workspace,
                    scale,
                    eps,
                    lower_bound,
                    cu_seqlens,
                    prefix,
                    p["tasks"],
                    start,
                    count,
                    wait,
                    counters,
                    index,
                ),
            )
        )
        calls.append(
            (
                p["scan"],
                (
                    v,
                    workspace,
                    state_out,
                    state_in,
                    output,
                    cu_seqlens,
                    prefix,
                    p["ranks"],
                    order,
                    start,
                    count,
                    rows,
                    long_pairs,
                    radix4,
                    flags,
                    counters,
                    index,
                    smem,
                ),
            )
        )
    return tuple(calls)
