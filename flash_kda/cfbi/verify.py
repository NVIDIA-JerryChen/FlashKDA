"""Empirical self-check of the one hardware premise behind CF-BI.

The bitwise guarantee (prefill == training forward == decode, any cut, any batch) rests on one property of the tensor
cores: an output element of ``mma.sync.m16n8k8.tf32`` (as used by the three-term ``dot3`` of the contract) depends only
on its own row vector, its own column vector and the K order - not on the slot of the 16x8 tile it sits in, nor on the
other rows / columns of the tile. Prefill fills up to 16 row slots, decode one; if the premise failed, they would
differ in bits.

``verify_hardware(device)`` runs a small battery of the extension's own ``probe_dot3`` on that device (well under a
second, once per device and process): the same (row, column) pairs are placed in every one of the 16 x 8 slots with
zero / random / wide-exponent neighbours, K = 24 and 128, four data regimes; the output bits must be identical to the
canonical placement. Two controls keep the check from being vacuous: the K-reversed product must differ in a visible
fraction of tiles (a check that cannot tell two programs apart proves nothing), and ``dot3`` must agree with the fp64
dot product to its accuracy class. It is the runtime counterpart of ``tests/cfbi/test_mma_probe.py``.

What it does NOT show: that the rest of the program (``ex2/rcp/rsqrt.approx``, FMA chains) is bit-stable on the device -
those are deterministic per instruction and exercised by the test-suite, not by this probe - and nothing about other
devices, drivers or compiler builds. Bits are only ever compared within one device architecture + build.
"""
import os
import time

import torch

from ._ext import load_ext

__all__ = ["verify_hardware"]

_RESULTS = {}   # device index -> result dict

_KS = (24, 128)                                            # 3 and 16 K-steps
_REGIMES = ("normal", "wide_exponents", "tiny_times_huge", "cancelling")
_N_PAIRS = 4                                               # distinct (row, column) pairs per regime
_N_FILLS = 3                                               # zero, random, wide-exponent neighbours
_NEG_CONTROL_MIN_FRACTION = 0.05                           # same threshold as tests/cfbi/test_mma_probe.py
_ACCURACY_MAX_REL = 2e-6                                   # |dot3 - exact| / sum|a*b| (B300: 3e-7)


def _impl_name(impl):
    """Human-readable name of the implementation bits returned by ``resolve_impl`` (csrc/cfbi/cfbi_launch.h)."""
    rows = "reference" if impl & 1 else "optimised"
    if impl & 4:
        return f"{rows} row engine only (no K1/K2)"
    chunk = "reference" if impl & 2 else "optimised"
    if impl & 8 and not impl & 2:
        chunk = "optimised, tcgen05 (UMMA) K2"
    return rows if rows == chunk else f"{rows} row engine + {chunk} K1/K2"


def _bits(x):
    return x.contiguous().view(torch.int32)


def _make_pair(regime, n, K, gen, device):
    def randn(*shape):
        return torch.randn(*shape, device=device, generator=gen)

    def pow2(*shape, lo, hi):
        return torch.exp2(torch.randint(lo, hi + 1, shape, device=device, generator=gen).float())

    a, b = randn(n, K), randn(n, K)
    if regime == "wide_exponents":
        a, b = a * pow2(n, K, lo=-30, hi=30), b * pow2(n, K, lo=-30, hi=30)
    elif regime == "cancelling":   # large terms of both signs that nearly cancel, small terms decide the result
        sign = torch.where(torch.arange(K, device=device) % 2 == 0, 1.0, -1.0)
        a = a.abs() * sign * 1e3
        b = b.abs() + 1e-3 * randn(n, K)
    elif regime == "tiny_times_huge":   # the exponent extremes of ks (<= 1) x kjr (<= 2^108) in a sub-block
        a, b = a * 2.0 ** -100, b * 2.0 ** 100
    return a.contiguous(), b.contiguous()


def _placement_battery(ext, a, b, gen, device):
    """Same pairs in all 16 x 8 slots x 3 neighbour patterns, one probe launch. Returns (mismatching outputs,
    outputs checked, canonical outputs [n])."""
    n, K = a.shape
    combos = [(i, j, f) for i in range(16) for j in range(8) for f in range(_N_FILLS)]   # combos[0] = (0, 0, zero)
    C = len(combos)
    rows = torch.tensor([c[0] for c in combos], device=device)
    cols = torch.tensor([c[1] for c in combos], device=device)
    kind = torch.tensor([c[2] for c in combos], device=device)

    def fill(*shape):
        x = torch.randn(C, n, *shape, device=device, generator=gen)
        wide = x * torch.exp2(torch.randint(-60, 61, (C, n, *shape), device=device, generator=gen).float())
        k = kind.view(C, *([1] * (len(shape) + 1)))
        return torch.where(k == 0, torch.zeros_like(x), torch.where(k == 1, x, wide))

    A, B = fill(16, K), fill(K, 8)
    ci, pn = torch.arange(C, device=device)[:, None], torch.arange(n, device=device)[None, :]
    A[ci, pn, rows[:, None]] = a[None].expand(C, n, K)
    B[ci, pn, :, cols[:, None]] = b[None].expand(C, n, K)
    D = ext.probe_dot3(A.view(C * n, 16, K), B.view(C * n, K, 8), False).view(C, n, 16, 8)
    got = D[ci, pn, rows[:, None], cols[:, None]]   # [C, n]: output element of the slot the pair sits in
    ref = got[0]
    bad = int((_bits(got) != _bits(ref)[None]).sum())
    if not bool(torch.isfinite(ref).all()):
        bad += 1   # a non-finite canonical result would make the comparison meaningless
    return bad, C * n, ref


def _run_battery(ext, device):
    gen = torch.Generator(device=device)
    gen.manual_seed(20261103)
    bad = checked = 0
    max_rel = 0.0
    for regime in _REGIMES:
        for K in _KS:
            a, b = _make_pair(regime, _N_PAIRS, K, gen, device)
            nb, nc, ref = _placement_battery(ext, a, b, gen, device)
            bad, checked = bad + nb, checked + nc
            if regime == "normal":   # accuracy class of the three-term product
                exact = (a.double() * b.double()).sum(-1)
                scale = (a.double().abs() * b.double().abs()).sum(-1)
                max_rel = max(max_rel, float(((ref.double() - exact).abs() / scale).max()))
    # negative control: reversed K order is a different program and must show in the bits
    a, b = _make_pair("normal", 256, 128, gen, device)
    A, B = torch.zeros(256, 16, 128, device=device), torch.zeros(256, 128, 8, device=device)
    A[:, 3, :], B[:, :, 2] = a, b
    fwd = ext.probe_dot3(A, B, False)[:, 3, 2]
    rev = ext.probe_dot3(A, B, True)[:, 3, 2]
    frac = float((_bits(fwd) != _bits(rev)).float().mean())
    return dict(mismatches=bad, placements_checked=checked, k_order_diff_fraction=frac, max_rel_err=max_rel)


def verify_hardware(device=None, *, force=False) -> dict:
    """Check the tensor-core premise on ``device`` (default: the current CUDA device); cached per device and process.

    Returns a dict: ``ok`` (bool, the verdict), ``reason`` (why not, else ``""``), ``premise_ok`` (identical bits in
    every slot / neighbour pattern), ``negative_control_ok`` (reversed K order differs in more than 5 % of tiles),
    ``accuracy_ok`` (dot3 within 2e-6 of the fp64 dot product), the raw numbers (``mismatches``,
    ``placements_checked``, ``k_order_diff_fraction``, ``max_rel_err``), the device (``device``, ``name``,
    ``capability``), which kernels run there (``kernels``, ``impl``: see ``resolve_impl`` in csrc/cfbi/cfbi_launch.h)
    and ``elapsed_s``. ``force=True`` repeats the battery. Must not be called while a CUDA graph is being captured
    (it synchronises); call it once beforehand, ``prefill`` / ``decode`` / ``forward_train`` do so on first use.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("verify_hardware needs a CUDA device")
    dev = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    if not force and idx in _RESULTS:
        return dict(_RESULTS[idx])
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("flash_kda.cfbi.verify_hardware() synchronises and cannot run during CUDA graph capture; "
                           "call it (or any cfbi entry point) once before capturing")
    major, minor = torch.cuda.get_device_capability(idx)
    res = dict(device=idx, name=torch.cuda.get_device_name(idx), capability=(major, minor), ok=False, reason="",
               premise_ok=False, negative_control_ok=False, accuracy_ok=False, mismatches=-1, placements_checked=0,
               k_order_diff_fraction=float("nan"), max_rel_err=float("nan"), kernels="none", impl=-1, elapsed_s=0.0)
    if major < 8:
        res["reason"] = f"compute capability {major}.{minor} has no TF32 tensor cores (mma.sync tf32 needs sm_80+)"
        _RESULTS[idx] = res
        return dict(res)
    ext = load_ext()   # (JIT build on first use: not part of elapsed_s)
    with torch.cuda.device(idx):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res.update(_run_battery(ext, torch.device("cuda", idx)))
        torch.cuda.synchronize()
        res["elapsed_s"] = time.perf_counter() - t0
    res["premise_ok"] = res["mismatches"] == 0
    res["negative_control_ok"] = res["k_order_diff_fraction"] > _NEG_CONTROL_MIN_FRACTION
    res["accuracy_ok"] = res["max_rel_err"] < _ACCURACY_MAX_REL
    try:
        res["impl"] = int(ext.resolve_impl(0, idx))
        res["kernels"] = _impl_name(res["impl"])
    except AttributeError:   # an extension built before resolve_impl existed
        pass
    why = []
    if not res["premise_ok"]:
        why.append(f"a tensor-core output element changed with its tile slot / neighbours in {res['mismatches']} of "
                   f"{res['placements_checked']} placements")
    if not res["negative_control_ok"]:
        why.append(f"the check is vacuous: reversed K order differs in only {res['k_order_diff_fraction']:.3%} of tiles")
    if not res["accuracy_ok"]:
        why.append(f"dot3 is {res['max_rel_err']:.2e} of sum|a*b| away from the fp64 product (limit {_ACCURACY_MAX_REL:.0e})")
    res["reason"] = "; ".join(why)
    res["ok"] = not why
    _RESULTS[idx] = res
    return dict(res)
