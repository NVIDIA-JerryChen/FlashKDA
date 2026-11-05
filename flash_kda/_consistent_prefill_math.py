"""Explicit FP32 contractions matching the pinned FlashInfer instruction path."""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def _mul_rn(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(), b.ir_value()],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _fma_rn(a, b, c, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(), b.ir_value(), c.ir_value()],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _prediction_dot8(state, row, key):
    # Match pinned FI's generated prediction contraction, which differs from
    # its output contraction: terms 2/3 round first, then terms 0/1 are fused.
    even = _mul_rn(state[row, 2], key[2])
    odd = _mul_rn(state[row, 3], key[3])
    even = _fma_rn(state[row, 0], key[0], even)
    odd = _fma_rn(state[row, 1], key[1], odd)
    for pair in cutlass.range_constexpr(2, 4):
        i = 2 * pair
        even = _fma_rn(state[row, i], key[i], even)
        odd = _fma_rn(state[row, i + 1], key[i + 1], odd)
    return even + odd


@cute.jit
def _output_dot8(state, row, query):
    even = _mul_rn(state[row, 0], query[0])
    odd = _mul_rn(state[row, 1], query[1])
    for pair in cutlass.range_constexpr(1, 4):
        i = 2 * pair
        even = _fma_rn(state[row, i], query[i], even)
        odd = _fma_rn(state[row, i + 1], query[i + 1], odd)
    return even + odd
