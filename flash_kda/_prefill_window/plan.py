"""Bind the recurrent prefill pipeline and its bounded, caller-independent workspace."""

from dataclasses import dataclass
import torch

from .aligned_induction_coeff import compiled_coeff_kernel
from .shared_stage_scan import compiled_scan_kernel
from .schedule import SEGMENT, MAX_REQUESTS, compiled_schedule
from .policy import (
    _rows,
    _long_ranks,
    _mixed_ranks,
    _fill_ranks,
    _plan_windows,
    PROPORTIONAL_REQUESTS,
)


@dataclass(frozen=True)
class WindowPlan:
    calls: tuple
    workspace: torch.Tensor
    workspace_bytes: int
    out: torch.Tensor
    state_out: torch.Tensor

    def run(self):
        for kernel, arguments in self.calls:
            kernel(*arguments)
        return self.out, self.state_out


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
    workspace_limit_bytes=256 * 1024 * 1024,
    *,
    scale=128**-0.5,
    eps=1e-6,
    lower_bound=-5.0,
):
    """Prepare on the tensor's device; the public API validates the buffers."""
    _, total, h, _ = q.shape
    n = state_in.shape[0]
    if n > MAX_REQUESTS:
        raise ValueError(f"prefill supports at most {MAX_REQUESTS} requests")
    if n == 0:
        workspace = torch.empty(0, device=q.device, dtype=torch.float32)
        return WindowPlan((), workspace, 0, out, state_out)

    capacity = (total + SEGMENT - 1) // SEGMENT + n
    # prefix[N+1], ranks[C], tasks[C,3], order[4N], all Int32.
    metadata_bytes = (5 * n + 1 + 4 * capacity) * 4
    row_bytes = SEGMENT * h * 392 * 4
    slots = min(capacity, (workspace_limit_bytes - metadata_bytes) // row_bytes)
    if slots < 1:
        raise ValueError("workspace cannot hold metadata and one scheduled segment")
    workspace = torch.empty(
        slots * SEGMENT * h * 392, device=q.device, dtype=torch.float32
    )
    prefix = torch.empty(n + 1, device=q.device, dtype=torch.int32)
    ranks = torch.empty(capacity, device=q.device, dtype=torch.int32)
    tasks = torch.empty(capacity, 3, device=q.device, dtype=torch.int32)
    order = torch.empty(4 * n, device=q.device, dtype=torch.int32)
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    capability = torch.cuda.get_device_capability(q.device)
    device = q.device.index
    coeff = compiled_coeff_kernel(device, capability)
    scan = compiled_scan_kernel(device, capability)
    schedule = compiled_schedule(device, capability)

    long_ranks = _long_ranks(n, h, sms)
    if long_ranks == 0:
        long_ranks = _mixed_ranks(n, h, sms)
    if long_ranks == 0 and _rows(n, h, sms) == 4:
        long_ranks = _fill_ranks(n, h, sms)
    rows = 4 if long_ranks else _rows(n, h, sms)
    long_pairs = long_ranks * h
    ctas = long_pairs * 8 + (n * h - long_pairs) * (8 if rows == 2 else 4)
    tail = long_pairs == 0 and rows == 4 and n >= 8 and h * (4 * n + 4) <= 3 * sms
    if tail:
        long_pairs = h
        ctas = h * (4 * n + 4)
    radix4 = 3 if ctas <= 2 * sms else (2 if rows == 4 else 4)
    if tail:
        radix4 |= 16
    radix4 |= 64

    calls = [
        (
            schedule,
            (
                cu_seqlens,
                prefix,
                ranks,
                tasks,
                order,
                1 if n <= PROPORTIONAL_REQUESTS else 0,
            ),
        )
    ]
    row = SEGMENT * h * 392
    for start, count, offset, wait in _plan_windows(
        slots, capacity, n, total, h, ctas > 3 * sms
    ):
        # Adjacent windows use disjoint regions until their predecessor completes.
        region = workspace[offset * row : (offset + count) * row]
        calls.append(
            (
                coeff,
                (
                    q,
                    k,
                    g,
                    beta,
                    a_log,
                    dt_bias.view(-1),
                    region,
                    float(scale),
                    float(eps),
                    float(lower_bound),
                    cu_seqlens,
                    prefix,
                    tasks,
                    start,
                    count,
                    wait,
                ),
            )
        )
        calls.append(
            (
                scan,
                (
                    v,
                    region,
                    state_out,
                    state_in,
                    out,
                    cu_seqlens,
                    prefix,
                    ranks,
                    order,
                    start,
                    count,
                    rows,
                    long_pairs,
                    radix4,
                ),
            )
        )
    return WindowPlan(
        tuple(calls), workspace, workspace.numel() * 4 + metadata_bytes, out, state_out
    )
