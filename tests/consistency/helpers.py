"""Shared test data and raw-bit diagnostics (no production arithmetic)."""
import torch
from flash_kda import consistent_fwd


def assert_bits(actual, expected, *, label, lengths=None, token=None):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    a, b = actual.detach().cpu(), expected.detach().cpu()
    for name, x in (("actual", a), ("expected", b)):
        bad = (~torch.isfinite(x)).nonzero()
        if bad.numel():
            raise AssertionError(f"{label}: {name} nonfinite at {bad[0].tolist()}")
    bad = (a.view(torch.int16) != b.view(torch.int16)).nonzero()
    if bad.numel():
        idx = tuple(bad[0].tolist())
        location = str(idx)
        if lengths is not None:
            packed = idx[1]
            for request, length in enumerate(lengths):
                if packed < length:
                    location = f"request={request}, head={idx[2]}, token={packed + (token or 0)}, V={idx[3]}"
                    break
                packed -= length
        elif a.ndim == 4 and a.shape[-2:] == (128, 128):
            location = f"request={idx[0]}, head={idx[1]}, token={token}, V={idx[2]}, K={idx[3]}"
        raise AssertionError(f"{label}: first mismatch {location}: {a[idx]} vs {b[idx]}, "
                             f"bits={a.view(torch.int16)[idx]} vs {b.view(torch.int16)[idx]}")


def make_case(lengths, h=12, *, seed=123, padding=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    t, n = sum(lengths) + padding, len(lengths)
    def rand(shape, dtype=torch.bfloat16, scale=0.2):
        return (torch.randn(shape, device="cuda", generator=gen) * scale).to(dtype)
    xs = [rand((1, t, h, 128)) for _ in range(4)]
    beta = rand((1, t, h))
    a = rand((h,), torch.float32)
    dt = rand((h, 128), torch.float32)
    state = rand((n, h, 128, 128))
    return dict(q=xs[0], k=xs[1], v=xs[2], g=xs[3], beta=beta,
                A_log=a, dt_bias=dt,
                cu_seqlens=torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()),
                                       device="cuda", dtype=torch.int64), state_in=state)


def run(case, **kwargs):
    out = torch.full_like(case["q"], 31)
    state = torch.empty_like(case["state_in"])
    consistent_fwd(**case, out=out, state_out=state, **kwargs)
    return out, state


def select(case, lengths, spans, order=None, state=None):
    """Pack arbitrary per-request intervals using host-known metadata."""
    order = list(range(len(lengths))) if order is None else list(order)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    result = {}
    for name in ("q", "k", "v", "g", "beta"):
        x = case[name]
        result[name] = torch.cat([x[:, offsets[r] + spans[r][0]:offsets[r] + spans[r][1]]
                                  for r in order], dim=1).contiguous()
    result.update(A_log=case["A_log"], dt_bias=case["dt_bias"])
    lens = [spans[r][1] - spans[r][0] for r in order]
    offsets = [0]
    for length in lens:
        offsets.append(offsets[-1] + length)
    result["cu_seqlens"] = torch.tensor(offsets, device="cuda", dtype=torch.int64)
    result["state_in"] = (case["state_in"][order].contiguous() if state is None else state)
    return result, lens
