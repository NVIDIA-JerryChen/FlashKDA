"""CF-BI: bitwise train/infer-consistent KDA kernels.

One fixed-order chunk-form floating-point program (chunk 64 anchored at the sequence start); prefill and
the training forward evaluate many rows at once, decode evaluates one more row of the same program, and the
bits do not depend on call cuts, phase or batch composition. See docs/20261102-cfbi-numerical-contract.md.
"""
from ._ext import load_ext
from .api import PrefillPhase1, decode, forward_train, prefill, prefill_k1, prefill_k2
from .cp import cp_prefill
from .autograd import chunk_kda
from .state import CfbiState, alloc_state, state_equal, state_mismatch

__all__ = [
    "load_ext", "prefill", "decode", "forward_train", "chunk_kda", "prefill_k1", "prefill_k2", "PrefillPhase1", "cp_prefill", "CfbiState", "alloc_state", "state_equal", "state_mismatch",
]
