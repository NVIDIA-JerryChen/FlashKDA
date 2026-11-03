"""Python entry points of the CF-BI kernels (prefill, decode, training forward)."""
import os
import warnings
from typing import Optional

import torch

from ._ext import load_ext
from .state import CfbiState
from .verify import verify_hardware

_GUARDED = {}   # device index -> True once the premise check passed (or was waived)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def _guard(device):
    """Refuse to run where the bitwise premise (an MMA output element is independent of the other rows / columns of
    its tile) does not hold. The first call per device runs the empirical battery of ``verify_hardware`` (well under a
    second); the verdict is cached. ``FLASH_KDA_CFBI_SKIP_VERIFY=1`` skips the battery,
    ``FLASH_KDA_CFBI_ALLOW_UNVALIDATED_ARCH=1`` runs it but only warns when it fails."""
    d = torch.device(device)
    key = d.index if d.index is not None else torch.cuda.current_device()
    if key in _GUARDED:
        return
    if not _env_flag("FLASH_KDA_CFBI_SKIP_VERIFY"):
        res = verify_hardware(key)
        if not res["ok"]:
            msg = (f"flash_kda.cfbi: the hardware premise could not be validated on cuda:{key} ({res['name']}, "
                   f"compute capability {res['capability'][0]}.{res['capability'][1]}): {res['reason']}. The bitwise "
                   "train/infer guarantee does not hold there.")
            if not _env_flag("FLASH_KDA_CFBI_ALLOW_UNVALIDATED_ARCH"):
                raise RuntimeError(msg + " Set FLASH_KDA_CFBI_ALLOW_UNVALIDATED_ARCH=1 to run anyway.")
            warnings.warn(msg + " Running anyway (FLASH_KDA_CFBI_ALLOW_UNVALIDATED_ARCH=1).", RuntimeWarning, stacklevel=3)
    _GUARDED[key] = True


def _impl_arg(_impl: Optional[int]) -> int:
    """``_impl`` (hidden, tests/diagnostics): bit 0 reference row engine, bit 1 reference K1/K2, bit 2 row engine for
    every row; 0 = the optimised kernels where the device can run them (the launcher falls back to the reference
    kernels below sm_90 and to the row engine where K1 does not fit in shared memory, see csrc/cfbi/cfbi_launch.h).
    ``None`` reads ``FLASH_KDA_CFBI_IMPL`` (default 0), which lets a whole test run be pinned to e.g. the reference."""
    return int(os.environ.get("FLASH_KDA_CFBI_IMPL", "0")) if _impl is None else int(_impl)


def prefill(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, state: CfbiState,
            cu_seqlens: Optional[torch.Tensor] = None, *, out_fp32: Optional[torch.Tensor] = None, _mode: int = 0,
            _impl: Optional[int] = None):
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
                       state.E, state.rows, state.pos, cu_seqlens, _mode, out_fp32, _impl_arg(_impl))


def decode(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, state: CfbiState, *,
           out_fp32: Optional[torch.Tensor] = None, _impl: Optional[int] = None):
    """One token per sequence (``q`` bf16 ``[N, 1, H, 128]`` or ``[N, H, 128]``); same arithmetic as ``prefill``."""
    if q.dim() == 3:
        q, k, v, g, out = (x.unsqueeze(1) for x in (q, k, v, g, out))
        beta = beta.unsqueeze(1)
        if out_fp32 is not None:
            out_fp32 = out_fp32.unsqueeze(1)
    _guard(q.device)
    load_ext().forward(q, k, v, g, beta, float(scale), out, A_log, dt_bias, float(lower_bound),
                       state.E, state.rows, state.pos, None, 1, out_fp32, _impl_arg(_impl))


def forward_train(q, k, v, g, beta, scale, A_log, dt_bias, lower_bound,
                  cu_seqlens: Optional[torch.Tensor] = None, *, out: Optional[torch.Tensor] = None,
                  out_fp32: Optional[torch.Tensor] = None, _impl: Optional[int] = None) -> torch.Tensor:
    """Training-forward schedule of the same program, from the empty state; nothing is persisted."""
    _guard(q.device)
    if out is None:
        out = torch.empty_like(q)
    load_ext().forward(q, k, v, g, beta, float(scale), out, A_log, dt_bias, float(lower_bound),
                       None, None, None, cu_seqlens, 0, out_fp32, _impl_arg(_impl))
    return out
