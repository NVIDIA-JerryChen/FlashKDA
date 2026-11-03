"""CF-BI: bitwise train/infer-consistent KDA kernels.

One fixed-order chunk-form floating-point program (chunk 64 anchored at the sequence start); prefill and
the training forward evaluate many rows at once, decode evaluates one more row of the same program, and the
bits do not depend on call cuts, phase or batch composition. See docs/20261102-cfbi-numerical-contract.md.
"""
from ._ext import load_ext

__all__ = ["load_ext"]
