"""Lower-bounded sigmoid decay gate of the fixed FP32 program.

gate(x) = exp2(lb * log2(e) * sigmoid(exp(a_log) * x)), evaluated as
exp2(lb_log2e / (1 + exp2(-exp(a_log) * log2(e) * x))) with the program's
fast-math exp2 and reciprocal, so every caller produces the same bits.

Provenance: the formula and its evaluation order follow the recurrent contract's
gate helper (adapted from FlashInfer, Apache License 2.0), so the bits match
the fixed program.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import math as mlir_math

LOG2_E = 1.4426950408889634


@cute.jit
def decay_gate(x, rate, lower_bound):
    """x: biased raw gate (FP32); rate: exp(a_log) of the head (FP32)."""
    log2e = cutlass.Float32(LOG2_E)
    slope = -rate * log2e
    floor = lower_bound * log2e
    fast = mlir_arith.FastMathFlags.fast
    one = cutlass.Float32(1.0).ir_value()
    scaled = mlir_arith.mulf(slope.ir_value(), x.ir_value(), fastmath=fast)
    denominator = mlir_arith.addf(
        one, mlir_math.exp2(scaled, fastmath=fast), fastmath=fast
    )
    sigmoid = mlir_arith.divf(one, denominator, fastmath=fast)
    exponent = mlir_arith.mulf(floor.ir_value(), sigmoid, fastmath=fast)
    return cutlass.Float32(mlir_math.exp2(exponent, fastmath=fast))
