"""CF-BI: bitwise train/infer-consistent KDA kernels.

One fixed-order chunk-form floating-point program (chunk 64 anchored at the sequence start); prefill and
the training forward evaluate many rows at once, decode evaluates one more row of the same program, and the
bits do not depend on call cuts, phase or batch composition. See docs/20261102-cfbi-numerical-contract.md.
"""
from ._ext import load_conv_ext, load_ext
from .api import decode, forward_train, prefill
from .autograd import chunk_kda
from .conv import alloc_conv_state, conv_state_after, short_conv
from .linear import bi_linear
from .state import CfbiState, alloc_state, state_equal, state_mismatch

__all__ = [
    "load_ext", "load_conv_ext", "prefill", "decode", "forward_train", "chunk_kda", "CfbiState", "alloc_state", "state_equal",
    "state_mismatch", "short_conv", "alloc_conv_state", "conv_state_after", "bi_linear",
]
