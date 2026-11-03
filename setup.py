import os
import subprocess
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME

this_dir = os.path.dirname(os.path.abspath(__file__))
subprocess.run(["git", "submodule", "update", "--init", "cutlass"])


def is_flag_set(flag: str) -> bool:
    return os.getenv(flag, "FALSE").lower() in ["true", "1", "y", "yes"]


def get_nvcc_thread_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return ["--threads", nvcc_threads]


SUPPORTED_CUDA_ARCHS = ["90a", "100a", "103a", "120a"]
# The CF-BI extension additionally builds for Ampere / Ada (mma.sync TF32 needs sm_80+; the sm_90-only PTX of its
# optimised kernels is guarded in csrc/cfbi/cfbi_async.cuh and the launcher runs the reference kernels there).
CFBI_CUDA_ARCHS = ["80", "86", "89"] + SUPPORTED_CUDA_ARCHS


def detect_cuda_arch():
    import torch

    if not torch.cuda.is_available():
        return None

    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return f"{major}{minor}" + ("a" if major >= 9 else "")   # the "a" (arch-specific) variants start at sm_90


def get_arch_flags(supported=None):
    assert CUDA_HOME is not None, "PyTorch must be compiled with CUDA support"

    requested = os.getenv("FLASH_KDA_CUDA_ARCHS", "auto").lower()
    if requested == "auto":
        arch = detect_cuda_arch()
        if arch is None:
            raise RuntimeError(
                "FLASH_KDA_CUDA_ARCHS=auto requires a visible CUDA device. "
                "Set FLASH_KDA_CUDA_ARCHS=all to build all supported archs."
            )
        archs = [arch]
    elif requested == "all":
        archs = supported or SUPPORTED_CUDA_ARCHS
    else:
        archs = [arch.strip() for arch in requested.split(",") if arch.strip()]

    flags = []
    for arch in archs:
        flags.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
    return flags


# FLASH_KDA_BUILD_EXTS selects which extensions to build: "all" (default), or a
# comma list of "master" (flash_kda_C, the CUTLASS chunk-16 kernel), "cfbi"
# (flash_kda_cfbi_C, the bitwise train/infer-consistent CF-BI kernels, plus
# flash_kda_cfbi_conv_C) and "cfbi_conv" (only flash_kda_cfbi_conv_C: the pinned
# short causal conv and the batch-invariant bf16 linear).
_BUILD_EXTS = {
    s.strip() for s in os.getenv("FLASH_KDA_BUILD_EXTS", "all").lower().split(",") if s.strip()
}
if "all" in _BUILD_EXTS:
    _BUILD_EXTS = {"master", "cfbi"}
_unknown = _BUILD_EXTS - {"master", "cfbi", "cfbi_conv"}
assert not _unknown, f"unknown FLASH_KDA_BUILD_EXTS entries: {sorted(_unknown)}"

ext_modules = []

if "master" in _BUILD_EXTS:
    ext_modules.append(
        CUDAExtension(
            name='flash_kda_C',
            sources=[
                'csrc/flash_kda.cpp',
                'csrc/smxx/fwd_launch.cu',
            ],
            include_dirs=[
                os.path.join(this_dir, 'cutlass', 'include'),
                os.path.join(this_dir, 'cutlass', 'examples', 'common'),
                os.path.join(this_dir, 'cutlass', 'tools', 'util', 'include'),
                os.path.join(this_dir, 'csrc'),
            ],
            extra_compile_args={
                'cxx': ['-O3', '-Wno-psabi'],
                'nvcc': [
                    '-O3',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_HALF2_OPERATORS__',
                    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '--use_fast_math',
                    '--ptxas-options=-v,--register-usage-level=10,--warn-on-spills',
                    '-lineinfo',
                    *get_nvcc_thread_args(),
                    *get_arch_flags(),
                ],
            },
        )
    )

if "cfbi" in _BUILD_EXTS:
    # The CF-BI kernels fix the floating-point program bit for bit, so unlike
    # flash_kda_C this extension must NOT be built with --use_fast_math (that
    # enables FTZ, FMA contraction and approximate division/sqrt). All
    # numerics-critical operations also use explicit intrinsics in the source;
    # -fmad=false is a second line of defence. No CUTLASS is needed (pure PTX).
    ext_modules.append(
        CUDAExtension(
            name='flash_kda_cfbi_C',
            sources=[
                'csrc/cfbi/cfbi_binding.cpp',
                'csrc/cfbi/cfbi_launch.cu',
                'csrc/cfbi/cfbi_probe.cu',
            ],
            include_dirs=[os.path.join(this_dir, 'csrc')],
            extra_compile_args={
                'cxx': ['-O3', '-Wno-psabi'],
                'nvcc': [
                    '-O3',
                    '-fmad=false',
                    '-ftz=false',
                    '-prec-div=true',
                    '-prec-sqrt=true',
                    '--expt-relaxed-constexpr',
                    '--ptxas-options=-v',
                    '-lineinfo',
                    *get_nvcc_thread_args(),
                    *get_arch_flags(CFBI_CUDA_ARCHS),
                ],
            },
        )
    )

if {"cfbi", "cfbi_conv"} & _BUILD_EXTS:
    # The layers around the core (short causal conv + SiLU, batch-invariant bf16 linear): same flags, own extension.
    ext_modules.append(
        CUDAExtension(
            name='flash_kda_cfbi_conv_C',
            sources=[
                'csrc/cfbi/cfbi_convlin_binding.cpp',
                'csrc/cfbi/cfbi_convlin.cu',
            ],
            include_dirs=[os.path.join(this_dir, 'csrc')],
            extra_compile_args={
                'cxx': ['-O3', '-Wno-psabi'],
                'nvcc': [
                    '-O3',
                    '-fmad=false',
                    '-ftz=false',
                    '-prec-div=true',
                    '-prec-sqrt=true',
                    '--expt-relaxed-constexpr',
                    '--ptxas-options=-v',
                    '-lineinfo',
                    *get_nvcc_thread_args(),
                    *get_arch_flags(),
                ],
            },
        )
    )

cmdclass = {"build_ext": BuildExtension}

rev = os.getenv("FLASH_KDA_VERSION_SUFFIX", "")
if not rev:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        rev = "+" + subprocess.check_output(cmd, cwd=this_dir).decode("ascii").rstrip()
    except Exception:
        rev = ""

setup(
    name='flash_kda',
    version='0.0.1' + rev,
    description='FlashKDA: Flash Kimi Delta Attention',
    ext_modules=ext_modules,
    packages=['flash_kda', 'flash_kda.cfbi', 'flash_kda.cfbi._bwd'],
    cmdclass=cmdclass,
    zip_safe=False,
)
