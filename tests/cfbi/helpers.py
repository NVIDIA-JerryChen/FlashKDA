"""Shared helpers of the CF-BI tests."""
import collections
import contextlib
import math

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

D = 128
REGIMES = ["authors", "unit", "heavy_tail", "long_memory", "no_decay", "eps_dominated", "saturated", "one_hot"]


def make_case(T_total, H, regime="authors", seed=0, device="cuda"):
    """Random inputs for ``T_total`` tokens as one ``[1, T_total, H, 128]`` stream (B = 1).

    The regimes stress different parts of the arithmetic: unit/authors-style data, heavy tails, almost no
    decay (long memory), exactly no decay (lower_bound = 0), eps-dominated norms, saturated gates, one-hot keys.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    def randn(*s):
        return torch.randn(*s, device=device, generator=gen)

    q, k, v, g = (randn(1, T_total, H, D) for _ in range(4))
    beta = randn(1, T_total, H)
    A_log = torch.rand(H, device=device, generator=gen)
    dt_bias = torch.rand(H, D, device=device, generator=gen)
    lower_bound = -5.0
    if regime == "authors":
        q, k = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)
    elif regime == "unit":
        pass
    elif regime == "heavy_tail":
        q, k, v = q * torch.exp(2 * randn(*q.shape)), k * torch.exp(2 * randn(*k.shape)), v * torch.exp(randn(*v.shape))
        g = g * 4
    elif regime == "long_memory":
        g = -12.0 + 2 * g
        A_log = torch.full_like(A_log, 0.5)
    elif regime == "no_decay":
        lower_bound = 0.0
    elif regime == "eps_dominated":
        q, k = q * 1e-5, k * 1e-5
    elif regime == "saturated":
        g, beta = 20.0 + g, 10.0 + beta
    elif regime == "one_hot":
        idx = torch.randint(0, D, (1, T_total, H), device=device, generator=gen)
        k = F.one_hot(idx, D).float()
        idx = torch.randint(0, D, (1, T_total, H), device=device, generator=gen)
        q = F.one_hot(idx, D).float()
        beta = beta + 3.0
    else:
        raise ValueError(regime)
    bf = torch.bfloat16
    return dict(q=q.to(bf).contiguous(), k=k.to(bf).contiguous(), v=v.to(bf).contiguous(), g=g.to(bf).contiguous(),
                beta=beta.to(bf).contiguous(), A_log=A_log, dt_bias=dt_bias, scale=1.0 / math.sqrt(D),
                lower_bound=lower_bound)


def split_tokens(case, lo, hi):
    """Slice of the token axis as a new case dict (views made contiguous)."""
    out = dict(case)
    for key in ("q", "k", "v", "g", "beta"):
        out[key] = case[key][:, lo:hi].contiguous()
    return out


def bits(x):
    return x.contiguous().view(torch.int16 if x.dtype == torch.bfloat16 else torch.int32)


def n_diff(a, b):
    return int((bits(a) != bits(b)).sum())


def rel_rms(x, ref):
    x, ref = x.double(), ref.double()
    return float(((x - ref) ** 2).mean().sqrt() / (ref ** 2).mean().sqrt())


def require_impl(ext, impl):
    """Skip a scenario that asks for the device-scheduled K2 (impl bit 16) on a device that drops the bit: the comparison would run another K2."""
    if impl & 16 and not ext.resolve_impl(impl, -1) & 16:
        import pytest
        pytest.skip("the device-scheduled K2 does not run on this device (impl bit 16 is dropped)")


ROWS_OPTIMISED = "cfbi::rows_kernel<false>"        # the optimised row engine behind the plan kernel
ROWS_DECODE = "cfbi::rows_kernel<true>"           # the single-launch decode (no plan kernel in front of it)
ROWS_REFERENCE = "cfbi::ref::rows_kernel"
WAVES = 2.5                    # the decode families run on grids of at least this many times the CTAs that are resident at once


# ---- observing the launches: one profiler session around a block of calls --------------------------------------------------------------------------------
def launch_counts(prof):
    """The CF-BI kernels of a finished profiler session: name (``cfbi::`` and the argument list removed, the template arguments kept: ``rows_kernel<true>`` and ``rows_kernel<false>`` are two
    names) -> launches."""
    cnt = collections.Counter()
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA and "cfbi::" in ev.name:
            name = ev.name.split("(")[0].strip()
            if name.startswith("void "):
                name = name[5:]
            cnt[name.replace("cfbi::", "", 1)] += 1
    return cnt


class Recording:
    """``counts`` is set when the ``recording()`` block ends."""
    counts = None


@contextlib.contextmanager
def recording():
    """``with recording() as rec: ...`` then ``rec.counts``: the CF-BI kernels that every call inside the block launched (see ``launch_counts``). The block is one profiler session, so a whole
    chain of calls is observed at once: a call in the middle that took another path shows in the counts."""
    rec = Recording()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        yield rec
        torch.cuda.synchronize()
    rec.counts = launch_counts(prof)


def rows_instance(ext, name=ROWS_OPTIMISED):
    """The occupancy record of one instantiation of the row engine from the hidden binding (``ext.rows_occupancy()``) and the device figures that go with it."""
    occ = ext.rows_occupancy()
    return next(i for i in occ["instances"] if i["name"] == name), occ["device"]


def resident_ctas(ext, name=ROWS_OPTIMISED):
    """CTAs of the row engine that are resident at once: SMs times the CTAs per SM that the occupancy API reports for the launch configuration in use (never a constant)."""
    inst, dev = rows_instance(ext, name)
    return dev["sm_count"] * inst["ctas_per_sm"]


def sequences_beyond_waves(capacity, heads, waves=WAVES):
    """The smallest number of sequences whose decode grid (one CTA per sequence and head) holds at least ``waves`` times ``capacity`` CTAs."""
    return max(1, -(-int(math.ceil(waves * capacity)) // heads))


def decode_sequences(ext, heads, waves=WAVES):
    """The number of sequences of a decode call beyond ``waves`` resident waves on this device."""
    return sequences_beyond_waves(resident_ctas(ext), heads, waves)


MUST_RUN_MAJOR = 10            # B200 and B300 (compute capability 10.x): the devices where every test of the row-engine suite must run


def must_run(capability):
    """Is this a device where the row-engine tests may not be skipped (a Blackwell data-centre part, compute capability 10.x)? ``capability`` is ``(major, minor)`` or None."""
    return capability is not None and capability[0] == MUST_RUN_MAJOR


def skip_or_fail(reason, capability=None):
    """``pytest.skip(reason)``, except on a device where the suite must run (B200, B300): the condition that would skip the test fails it there, so that an unsupported arch, a missing
    tool or a missing optimised kernel cannot turn a gate into a silent pass. ``capability`` defaults to the current device's (None without CUDA: a skip)."""
    import pytest

    if capability is None and torch.cuda.is_available():
        capability = tuple(torch.cuda.get_device_capability())
    if must_run(capability):
        pytest.fail("must run on compute capability %d.%d but would be skipped: %s" % (capability[0], capability[1], reason), pytrace=False)
    pytest.skip(reason)
