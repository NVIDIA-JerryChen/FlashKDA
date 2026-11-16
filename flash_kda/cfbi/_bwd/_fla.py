"""The only place where flash_kda.cfbi touches flash-linear-attention (FLA).

The CF-BI backward needs correct gradients of the KDA math, nothing more (it is not bit-pinned), so it
recomputes the forward with FLA's Triton chunk kernels under autograd and back-propagates through them.

Packaging decision: the plan was to vendor FLA's chunk backward (MIT). Measured on the FLA checkout b0de6fc8,
the symbol-level import closure of ``fla/ops/kda/{chunk_bwd,chunk_fwd,chunk_intra,wy_fast,gate}.py`` is
~10k lines in 28 files (about 1.7k of it context-parallel code, 1.5k from ``gla/chunk.py``, plus
``fla.utils``/``fla.backends`` machinery), so the plan's fallback is used: a lazy import of an installed
``fla`` (``pip install flash-linear-attention>=0.5.0``; ``tests/test.sh`` already installs it). Vendoring later
only has to replace :func:`fla_chunk_kda`.
"""


def fla_chunk_kda():
    try:
        from fla.ops.kda import chunk_kda
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "flash_kda.cfbi.chunk_kda needs flash-linear-attention for its backward pass "
            "(pip install 'flash-linear-attention>=0.5.0'); the inference kernels (prefill/decode/forward_train) "
            "do not need it."
        ) from e
    return chunk_kda
