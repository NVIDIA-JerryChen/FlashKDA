"""Dense prefill and checkpoint forward under the recurrent numerical contract."""

import torch
from dataclasses import dataclass, field

from .consistent import _contract_scalars, _range


@dataclass(frozen=True)
class ConsistentPrefillPlan:
    """Buffer-bound launch prepared by prepare_consistent_prefill.

    Keep tensor addresses, metadata and scalar parameters unchanged. Tensor
    contents may change on the caller's stream. Recreate the plan to rebind
    buffers or shapes; compatible plans reuse the compiled kernel.
    """

    _calls: tuple = field(repr=False)
    workspace_bytes: int
    _requires_grad: bool = field(repr=False)
    out: torch.Tensor = field(repr=False)
    state_out: torch.Tensor | None = field(repr=False)
    checkpoint_states: torch.Tensor | None = field(default=None, repr=False)
    checkpoint_indices: torch.Tensor | None = field(default=None, repr=False)

    def run(self):
        """Enqueue the prepared sequence on the current stream without allocation."""
        if self._requires_grad and torch.is_grad_enabled():
            raise RuntimeError("consistent prefill supports no_grad forward only")
        for kernel, arguments in self._calls:
            kernel(*arguments)
        return self.out, self.state_out


def prepare_consistent_prefill(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    state_in,
    out,
    state_out,
    *,
    scale=128**-0.5,
    eps=1e-6,
    lower_bound=-5.0,
    mode="auto",
    workspace_limit_bytes=64 * 1024 * 1024,
    checkpoint_states=None,
    checkpoint_indices=None,
):
    """Validate and bind dense prefill buffers, then compile outside Graph capture.

    BF16 q/k/v/g/out: [B,L,H,128]; beta: [B,L,H]; state: [B,H,128,128].
    FP32 A_log: [H]; dt_bias: [H,128]. All tensors must be contiguous CUDA
    tensors on one device, with 32-byte aligned data pointers. State input and
    output may be identical without checkpoints; all overlap is forbidden with
    checkpoints. Optional BF16 checkpoint_states [P,H,128,128] and Int32
    checkpoint_indices [B,L] select unique valid slots. With checkpoints,
    state_out may be None to omit a separate final-state write. Buffers belong to
    the caller. Empty batches are no-ops. Prepare outside CUDA Graph capture.
    See docs/consistent-prefill.md for the complete contract.
    """
    has_checkpoints = checkpoint_states is not None
    if has_checkpoints != (checkpoint_indices is not None):
        raise ValueError(
            "checkpoint_states and checkpoint_indices must be provided together"
        )
    if state_out is None and not has_checkpoints:
        raise ValueError("state_out may be None only when checkpoints are enabled")
    tensors = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        state_in=state_in,
        out=out,
        state_out=state_out,
    )
    if state_out is None:
        del tensors["state_out"]
    if has_checkpoints:
        tensors.update(
            checkpoint_states=checkpoint_states, checkpoint_indices=checkpoint_indices
        )
    if not all(isinstance(t, torch.Tensor) for t in tensors.values()):
        raise TypeError("all inputs and output buffers must be torch.Tensor")
    if q.ndim != 4 or q.shape[2] < 1 or q.shape[3] != 128:
        raise ValueError("q must have shape [B,L,H,128], H > 0")
    b, length, h, _ = q.shape
    if length >= 2**31 or b * h * 64 >= 2**31:
        raise ValueError("prefill requires L < 2**31 and B * H * 64 < 2**31")
    shapes = dict(
        q=q.shape,
        k=q.shape,
        v=q.shape,
        g=q.shape,
        out=q.shape,
        beta=(b, length, h),
        A_log=(h,),
        dt_bias=(h, 128),
        state_in=(b, h, 128, 128),
        state_out=(b, h, 128, 128),
    )
    if has_checkpoints:
        if checkpoint_states.ndim != 4 or tuple(checkpoint_states.shape[1:]) != (
            h,
            128,
            128,
        ):
            raise ValueError("checkpoint_states must have shape [P,H,128,128]")
        capacity = checkpoint_states.shape[0]
        if capacity < b * length or capacity >= 2**31:
            raise ValueError("checkpoint pool requires B * L <= P < 2**31")
        shapes.update(
            checkpoint_states=(capacity, h, 128, 128), checkpoint_indices=(b, length)
        )
    for name, t in tensors.items():
        dtype = (
            torch.int32
            if name == "checkpoint_indices"
            else torch.float32 if name in ("A_log", "dt_bias") else torch.bfloat16
        )
        if t.device.type != "cuda" or t.device != q.device:
            raise ValueError(f"{name} must be on the same CUDA device as q")
        if t.dtype != dtype or tuple(t.shape) != tuple(shapes[name]):
            raise ValueError(f"{name} must have dtype {dtype} and shape {shapes[name]}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if t.numel() and t.data_ptr() % 32:
            raise ValueError(f"{name} data pointer must be 32-byte aligned")
        if torch.is_grad_enabled() and t.requires_grad:
            raise RuntimeError("consistent_prefill_fwd supports no_grad forward only")
    _contract_scalars(scale, eps, lower_bound)
    if mode not in ("auto", "direct", "staged"):
        raise ValueError("mode must be auto, direct or staged")
    if not isinstance(workspace_limit_bytes, int) or workspace_limit_bytes < 0:
        raise ValueError("workspace_limit_bytes must be a nonnegative integer")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "prepare_consistent_prefill must run outside CUDA Graph capture"
        )
    ranges = [(name, *_range(t)) for name, t in tensors.items()]
    for i, (name, lo, hi) in enumerate(ranges):
        for other, ulo, uhi in ranges[i + 1 :]:
            if max(lo, ulo) < min(hi, uhi):
                if (
                    not has_checkpoints
                    and {name, other} == {"state_in", "state_out"}
                    and (lo, hi)
                    == (
                        ulo,
                        uhi,
                    )
                ):
                    continue
                raise ValueError(f"overlapping tensors are forbidden: {name}, {other}")
    grad = any(t.requires_grad for t in tensors.values())
    if b == 0:
        return ConsistentPrefillPlan(
            (), 0, grad, out, state_out, checkpoint_states, checkpoint_indices
        )
    if mode == "auto" and not has_checkpoints:
        from .unified_prefill import prepare_unified_prefill

        plan = prepare_unified_prefill(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            state_in,
            out,
            state_out,
            scale=scale,
            eps=eps,
            lower_bound=lower_bound,
            workspace_limit_bytes=workspace_limit_bytes,
        )
        return ConsistentPrefillPlan(
            ((plan.run, ()),), plan.workspace_bytes, grad, out, state_out
        )
    from ._consistent_prefill_kernel import compiled_prefill_kernel

    calls = []
    used_workspace = 0
    final_buffer = state_out if state_out is not None else state_in
    tile = (2 if b * h <= 64 else 8) if has_checkpoints else 8
    with torch.cuda.device(q.device):
        device = q.device.index
        capability = torch.cuda.get_device_capability(q.device)
        arguments = (
            q,
            k,
            v,
            g,
            beta,
            final_buffer,
            state_in,
            out,
            A_log,
            dt_bias.view(-1),
            float(scale),
            float(eps),
            float(lower_bound),
        )
        if length == 0 or mode == "direct" or (mode == "auto" and length < 8):
            if has_checkpoints:
                from ._consistent_prefill_kernel import compiled_prefill_checkpoints

                kernel = compiled_prefill_checkpoints(
                    device, capability, tile, state_out is not None
                )
                arguments = (
                    arguments[:10]
                    + (checkpoint_states, checkpoint_indices)
                    + arguments[10:]
                )
            else:
                kernel = compiled_prefill_kernel(device, capability, tile)
            calls.append((kernel, arguments))
        else:
            from ._consistent_prefill_staged import (
                compiled_coeff_kernel,
                compiled_scan_kernel,
            )

            # Checkpoint mode may specialize on the host-known query length.
            long_scan = has_checkpoints and length >= 256
            if long_scan:
                from ._consistent_prefill_scan import compiled_scan_kernel
            row_bytes = b * h * 392 * 4
            workspace_budget = workspace_limit_bytes
            if long_scan:
                target_mib = 8 if b * h <= 8 else (32 if b * h <= 512 else 64)
                workspace_budget = min(
                    workspace_budget, max(row_bytes, target_mib * 1024 * 1024)
                )
            window = min(length, workspace_budget // row_bytes)
            if window < 1:
                raise ValueError(
                    "workspace limit cannot hold one token for the whole batch"
                )
            # Cap the number of rows consumed by the Int32 coefficient grid.
            window = min(window, (2**31 - 1) // (b * h))
            workspace = torch.empty(
                b * window * h * 392, device=q.device, dtype=torch.float32
            )
            used_workspace = workspace.numel() * workspace.element_size()
            coeff = compiled_coeff_kernel(device, capability)
            if has_checkpoints:
                if long_scan:
                    from ._consistent_prefill_scan import compiled_scan_checkpoints
                else:
                    from ._consistent_prefill_staged import compiled_scan_checkpoints

                scan = compiled_scan_checkpoints(
                    device, capability, tile, state_out is not None
                )
            else:
                scan = compiled_scan_kernel(device, capability, tile)
            for start in range(0, length, window):
                count = min(window, length - start)
                calls.append(
                    (
                        coeff,
                        (
                            q,
                            k,
                            g,
                            beta,
                            A_log,
                            dt_bias.view(-1),
                            workspace,
                            float(scale),
                            float(eps),
                            float(lower_bound),
                            start,
                            count,
                        ),
                    )
                )
                scan_arguments = (
                    v,
                    workspace,
                    final_buffer,
                    state_in if has_checkpoints or start == 0 else state_out,
                    out,
                )
                if has_checkpoints:
                    scan_arguments += (checkpoint_states, checkpoint_indices)
                calls.append((scan, scan_arguments + (start, count)))
    return ConsistentPrefillPlan(
        tuple(calls),
        used_workspace,
        grad,
        out,
        state_out,
        checkpoint_states,
        checkpoint_indices,
    )


def consistent_prefill_fwd(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    state_in,
    out,
    state_out,
    *,
    scale=128**-0.5,
    eps=1e-6,
    lower_bound=-5.0,
    mode="auto",
    workspace_limit_bytes=64 * 1024 * 1024,
    checkpoint_states=None,
    checkpoint_indices=None,
):
    """Validate and launch a dense sequence; use a prepared plan for hot loops."""
    return prepare_consistent_prefill(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        state_in,
        out,
        state_out,
        scale=scale,
        eps=eps,
        lower_bound=lower_bound,
        mode=mode,
        workspace_limit_bytes=workspace_limit_bytes,
        checkpoint_states=checkpoint_states,
        checkpoint_indices=checkpoint_indices,
    ).run()
