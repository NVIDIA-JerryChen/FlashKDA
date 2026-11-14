"""Preserve alignment information when forming vector access pointers."""

import cutlass
import cutlass.cute as cute


@cute.jit
def _known_aligned(pointer, alignment: cutlass.Constexpr[int]):
    return cute.make_ptr(
        pointer.dtype, pointer.toint(), pointer.memspace, assumed_align=alignment
    )
