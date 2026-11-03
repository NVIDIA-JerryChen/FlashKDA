"""Results must not depend on memory the program does not own: the K1 workspace and the not-yet-valid rows of the
open-chunk cache hold whatever the allocator left there (torch.empty), the kernels only read what they wrote or
what is part of the state. Filling both with NaN must leave every bit of the outputs and of the valid state alone."""
import random

import pytest
import torch

from flash_kda.cfbi import alloc_state, load_ext, prefill, state_equal, state_mismatch

from helpers import make_case, n_diff


def _cu(lens):
    cu = [0]
    for L in lens:
        cu.append(cu[-1] + L)
    return cu


def _calls(case, lens, takes):
    cu = _cu(lens)
    done = [0] * len(lens)
    for take in takes:
        pieces = [{k: case[k][:, cu[n] + done[n]:cu[n] + done[n] + take[n]] for k in ("q", "k", "v", "g", "beta")}
                  for n in range(len(lens))]
        sub = {k: torch.cat([p[k] for p in pieces], dim=1).contiguous() for k in ("q", "k", "v", "g", "beta")}
        yield sub, torch.tensor(_cu(take), dtype=torch.int64, device="cuda")
        done = [d + t for d, t in zip(done, take)]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nan_filled_scratch_changes_nothing(ext, seed):
    H = 3
    rng = random.Random(seed)
    lens = [rng.randint(70, 330) for _ in range(5)] + [64, 1]
    case = make_case(sum(lens), H, "authors", seed=100 + seed)
    takes = []
    rem = list(lens)
    for _ in range(3):
        t = [min(r, rng.choice([0, 1, 17, 63, 64, 65, 100, 129])) for r in rem]
        takes.append(t)
        rem = [r - x for r, x in zip(rem, t)]
    takes.append(rem)

    clean = alloc_state(len(lens), H)
    dirty = alloc_state(len(lens), H)
    for sub, cu in _calls(case, lens, takes):
        o_c = torch.zeros_like(sub["q"])
        f_c = torch.zeros(sub["q"].shape, dtype=torch.float32, device="cuda")
        prefill(sub["q"], sub["k"], sub["v"], sub["g"], sub["beta"], case["scale"], o_c, case["A_log"], case["dt_bias"],
                case["lower_bound"], clean, cu, out_fp32=f_c)
        # dirty run: NaN in the whole row cache beyond the valid rows, NaN in the K1 workspace (the split API takes it as an argument)
        valid = torch.arange(64, device="cuda")[None, :] < dirty.pos[:, None]            # [N, 64]
        mask = valid[:, None, None, :, None].expand_as(dirty.rows)
        dirty.rows.copy_(torch.where(mask, dirty.rows, torch.full_like(dirty.rows, float("nan"))))
        ext_mod = load_ext()
        N = len(lens)
        T = sub["q"].shape[1]
        ws_n, plan_n = ext_mod.workspace_sizes(T, N, H)
        ws = torch.full((ws_n,), float("nan"), dtype=torch.float32, device="cuda")
        plan = torch.empty(plan_n, dtype=torch.int32, device="cuda")
        o_d = torch.zeros_like(sub["q"])
        f_d = torch.zeros(sub["q"].shape, dtype=torch.float32, device="cuda")
        ext_mod.forward(sub["q"], sub["k"], sub["v"], sub["g"], sub["beta"], case["scale"], o_d, case["A_log"], case["dt_bias"],
                        case["lower_bound"], dirty.E, dirty.rows, dirty.pos, cu, 0, f_d, 0, None, ws, plan, 3)
        assert n_diff(o_c, o_d) == 0 and n_diff(f_c, f_d) == 0, "outputs depend on stale workspace / cache memory"
        assert state_equal(clean, dirty), state_mismatch(clean, dirty)
