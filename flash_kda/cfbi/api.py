"""Python entry points of the CF-BI kernels (prefill, decode, training forward)."""
import os
from typing import Optional

import torch

from ._ext import load_ext
from .state import CfbiState

_ARCH_OK = {}


def _guard(device: torch.device):
    """The bitwise premise (MMA output independent of other rows' data) was validated on B300 only."""
    key = torch.device(device).index if torch.device(device).index is not None else torch.cuda.current_device()
    ok = _ARCH_OK.get(key)
    if ok is None:
        major, minor = torch.cuda.get_device_capability(key)
        ok = major >= 10 or os.environ.get("FLASH_KDA_CFBI_ALLOW_UNVALIDATED_ARCH", "0") == "1"
        if not ok:
            raise RuntimeError(
                f"flash_kda.cfbi was validated on compute capability >= 10.0 (B300) only, this device is {major}.{minor}. "
                "Set FLASH_KDA_CFBI_ALLOW_UNVALIDATED_ARCH=1 to run anyway (the bitwise guarantee is unverified there)."
            )
        _ARCH_OK[key] = ok


def prefill(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, state: CfbiState,
            cu_seqlens: Optional[torch.Tensor] = None, *, out_fp32: Optional[torch.Tensor] = None, _mode: int = 0,
            _impl: int = 0):
    """Append ``T`` tokens per sequence to the sequences' CF-BI state and write the outputs in place.

    Argument names/shapes follow ``flash_kda.fwd``: q, k, v, g bf16 ``[B, T, H, 128]``, beta bf16 raw logits
    ``[B, T, H]``, A_log fp32 ``[H]``, dt_bias fp32 ``[H, 128]``, ``lower_bound`` in ``[-5, 0]``; ``out`` bf16
    ``[B, T, H, 128]``. Without ``cu_seqlens`` every batch entry is one sequence (state entry ``b``); with
    ``cu_seqlens`` (int64 ``[N+1]``, B == 1) sequence ``n`` owns tokens ``cu_seqlens[n]:cu_seqlens[n+1]``.
    ``state`` is updated in place. The bits of ``out`` and ``state`` do not depend on how the sequence is cut
    into calls, nor on the other sequences of the batch.
    """
    _guard(q.device)
    load_ext().forward(q, k, v, g, beta, float(scale), out, A_log, dt_bias, float(lower_bound),
                       state.E, state.rows, state.pos, cu_seqlens, _mode, out_fp32, _impl)


def decode(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, state: CfbiState, *,
           out_fp32: Optional[torch.Tensor] = None, _impl: int = 0):
    """One token per sequence (``q`` bf16 ``[N, 1, H, 128]`` or ``[N, H, 128]``); same arithmetic as ``prefill``."""
    if q.dim() == 3:
        q, k, v, g, out = (x.unsqueeze(1) for x in (q, k, v, g, out))
        beta = beta.unsqueeze(1)
        if out_fp32 is not None:
            out_fp32 = out_fp32.unsqueeze(1)
    _guard(q.device)
    load_ext().forward(q, k, v, g, beta, float(scale), out, A_log, dt_bias, float(lower_bound),
                       state.E, state.rows, state.pos, None, 1, out_fp32, _impl)


def forward_train(q, k, v, g, beta, scale, A_log, dt_bias, lower_bound,
                  cu_seqlens: Optional[torch.Tensor] = None, *, out: Optional[torch.Tensor] = None,
                  out_fp32: Optional[torch.Tensor] = None, _impl: int = 0) -> torch.Tensor:
    """Training-forward schedule of the same program, from the empty state; nothing is persisted."""
    _guard(q.device)
    if out is None:
        out = torch.empty_like(q)
    load_ext().forward(q, k, v, g, beta, float(scale), out, A_log, dt_bias, float(lower_bound),
                       None, None, None, cu_seqlens, 0, out_fp32, _impl)
    return out
