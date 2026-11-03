"""Loader for the ``flash_kda_cfbi_C`` extension.

Order: (1) the module built by ``pip install`` / ``setup.py`` (``FLASH_KDA_BUILD_EXTS=cfbi``), unless
``FLASH_KDA_CFBI_JIT=1``; (2) a JIT build with ``torch.utils.cpp_extension.load`` from ``csrc/cfbi``
(development loop; same flags as setup.py, no ``--use_fast_math``).

Env knobs: FLASH_KDA_CFBI_JIT=1, FLASH_KDA_CFBI_VERBOSE=1, FLASH_KDA_CFBI_BUILD_DIR, TORCH_CUDA_ARCH_LIST.
"""
import os
from pathlib import Path

import torch

_EXT = None

NVCC_FLAGS = [
    "-O3",
    "-fmad=false",       # no FMA contraction (numerics-critical code also uses explicit intrinsics)
    "-ftz=false",
    "-prec-div=true",
    "-prec-sqrt=true",
    "--expt-relaxed-constexpr",
    "-lineinfo",
    "--ptxas-options=-v",
]

_SOURCES = ("cfbi_binding.cpp", "cfbi_launch.cu", "cfbi_probe.cu")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _jit_build(verbose: bool, name: str = "flash_kda_cfbi_C_jit", defines: tuple = ()):
    from torch.utils.cpp_extension import load

    root = _repo_root()
    csrc = root / "csrc"
    # The extension uses tf32 mma.sync (sm_80+). Container images often preset TORCH_CUDA_ARCH_LIST with
    # old archs (e.g. 7.5), so build for the current device unless FLASH_KDA_CFBI_ARCH says otherwise.
    arch = os.environ.get("FLASH_KDA_CFBI_ARCH")
    if arch is None and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}.{minor}" + ("a" if major >= 9 else "")
    build_dir = os.environ.get("FLASH_KDA_CFBI_BUILD_DIR")
    if build_dir:
        os.makedirs(build_dir, exist_ok=True)
    saved = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch is not None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        return load(
            name=name,
            sources=[str(csrc / "cfbi" / s) for s in _SOURCES],
            extra_include_paths=[str(csrc)],
            extra_cflags=["-O3", "-Wno-psabi"],
            extra_cuda_cflags=NVCC_FLAGS + [f"-D{d}" for d in defines],
            build_directory=build_dir or None,
            verbose=verbose,
        )
    finally:
        if saved is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = saved


def load_ext(verbose: bool | None = None):
    """Return the CF-BI extension module (building it just in time if it was not installed)."""
    global _EXT
    if _EXT is not None:
        return _EXT
    if verbose is None:
        verbose = os.environ.get("FLASH_KDA_CFBI_VERBOSE", "0") == "1"
    if os.environ.get("FLASH_KDA_CFBI_JIT", "0") != "1":
        try:
            import flash_kda_cfbi_C

            _EXT = flash_kda_cfbi_C
            return _EXT
        except ImportError:
            pass
    _EXT = _jit_build(verbose)
    return _EXT


def load_variant(name: str, defines: tuple):
    """JIT-build a differently flagged copy of the extension under its own module name (tests only: the
    negative-control kernels that the bitwise harness must catch)."""
    return _jit_build(os.environ.get("FLASH_KDA_CFBI_VERBOSE", "0") == "1", name=f"flash_kda_cfbi_C_{name}",
                      defines=tuple(defines))
