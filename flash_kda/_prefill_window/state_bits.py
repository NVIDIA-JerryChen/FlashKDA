"""Exact BF16 state helpers for the register-resident recurrent scans.

The state is kept in FP32 registers whose values are always BF16 values:
expanding BF16 bits and rounding with cvt.rn.bf16x2 are exact bit moves and
the same RN rounding the recurrence has always applied after every token.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm


@dsl_user_op
def _expand_state_bits(packed, *, loc=None, ip=None):
    # Reinterpret the two BF16 values as exact FP32 values, without arithmetic.
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32,f32)>"),
        [packed.ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 a, b; shl.b32 a, $2, 16; "
        "and.b32 b, $2, 0xffff0000; mov.b32 $0, a; mov.b32 $1, b; }",
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        cutlass.Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        cutlass.Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _round_state(value, zero, *, loc=None, ip=None):
    # cvt.rn.bf16x2 writes its first source to the upper half. With a zero
    # second source the result is the FP32 encoding of the RN BF16 value.
    result = llvm.inline_asm(
        T.f32(),
        [value.ir_value(loc=loc, ip=ip), zero.ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 t; cvt.rn.bf16x2.f32 t, $1, $2; mov.b32 $0, t; }",
        "=f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(result)


@dsl_user_op
def _pack_state(low, high, *, loc=None, ip=None):
    # Both inputs are already BF16 values; keep their upper halves (low first).
    result = llvm.inline_asm(
        T.i32(),
        [low.ir_value(loc=loc, ip=ip), high.ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 a, b; mov.b32 a, $1; mov.b32 b, $2; prmt.b32 $0, a, b, 0x7632; }",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint32(result)


@cute.jit
def _known_aligned(pointer, alignment: cutlass.Constexpr[int]):
    return cute.make_ptr(
        pointer.dtype, pointer.toint(), pointer.memspace, assumed_align=alignment
    )
