"""The workloads that compute-sanitizer runs over the row engine, on a path of the decode that is named and asserted (plain functions of the device; the drivers of ``test_rows_sanitizer.py`` import them).

The profiler (CUPTI) cannot attach to a process that runs under compute-sanitizer (``CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED``: the record of a launch set comes back empty), so the launch
assertion of a path (``rows_workload.launched_exactly``: the single launch is one ``rows_kernel<true>`` and nothing else, the two launches are the plan kernel and ``rows_kernel<false>``) cannot be made
inside the tool. Every workload here is therefore one function that runs in both places, with ``assert_launches`` on in the test process (without the tool: every launch is asserted to be the one of
the path) and off in the driver (under the tool); what it returns is a *digest* of the plan that it executed (the sizes, the live CTAs of every launch, a checksum of the slots), which the test compares
with the digest that the driver printed: the tool ran the workload whose launches were asserted, on the path that the test names.
"""
import json
import random
import zlib

import torch

import helpers
import rows_workload as rw
from flash_kda.cfbi import alloc_state
from rows_workload import Family, Step

H_WAVES = 12
OOB_SLOTS = 12            # the pool of the out-of-bounds control: its pos is a tensor of this many ints, so an offset beyond OOB_SLOTS leaves the allocation
OOB_SEQUENCES, OOB_HEADS = 4, 2      # the launch of the control: four live sequences of H = 2 heads, so 8 CTAs, each with one arrival atomic, and one elected store per sequence
WAVES_FAMILIES = [Family("san_varlen", lens=[1, 7, 33, 62, 63, 64, 65, 70], pool=True, pad=0.3, empty=0.2),
                  Family("san_decode", fixed=1, pool=True, pad=0.3),
                  Family("san_multirow", fixed=5, pool=True, pad=0.3)]


def run(mod, path, pool, step, assert_launches):
    """One launch on ``path`` (the ``_impl`` of the path), asserted to launch exactly the kernels of the path when ``assert_launches`` (not possible under the sanitizer)."""
    if assert_launches:
        return rw.launched_exactly(path, lambda: rw.run_step(mod, path.impl, pool, step))
    return rw.run_step(mod, path.impl, pool, step)


def checksum(values):
    return zlib.crc32(json.dumps(list(values)).encode())


def waves_workload(ext, path, assert_launches=False):
    """Three mode-1 launches of the optimised row engine on a pool (``cu_seqlens`` of 1 .. 70 rows with open chunks at every position, chunk ends crossed and empty sequences; a decode launch; a
    five-row launch; padding entries, slot -1, in every one), each with at least 2.5 resident waves of LIVE CTAs of the instantiation that ``path`` launches (``path.instance``), asserted before the
    launch. Returns the digest of what was executed."""
    H = H_WAVES
    rng = random.Random(5)
    par = rw.make_params(H, 5)
    cap = helpers.resident_ctas(ext, path.instance)
    live = helpers.sequences_beyond_waves(cap, H)
    S = int(live * 1.5) + 19
    pool = alloc_state(S, H)
    digest = {"path": path.name, "impl": path.impl, "resident_ctas": cap, "pool": S, "launches": []}
    for i, fam in enumerate(WAVES_FAMILIES):
        step, _ = rw.draw_step(ext, fam, H, par, rng, 50 + i, pool_size=S, instance=path.instance)
        problems = rw.beyond_waves_problems([step.live_ctas()], {str(path.instance): cap})
        assert not problems, "; ".join(problems)
        run(ext, path, pool, step, assert_launches)
        digest["launches"].append({"family": fam.name, "entries": step.N, "live_ctas": step.live_ctas(), "padding": sum(1 for s in step.slots if s < 0), "rows": sum(step.lens),
                                   "slots": checksum(step.slots), "lens": checksum(step.lens)})
    torch.cuda.synchronize()
    return digest


def bounded_workload(ext, path, launches, heads, pool_slots, assert_launches=False):
    """The bounded racecheck workload (see ``test_rows_sanitizer.BOUNDED_LAUNCHES``) on ``path``; returns the final position of every slot and the digest."""
    par = rw.make_params(heads, 6)
    pool = alloc_state(pool_slots, heads)
    steps = []
    for i, (lens, slots) in enumerate(launches):
        step = Step(par, heads, lens, slots, len(set(lens)) == 1, 60 + i)
        run(ext, path, pool, step, assert_launches)
        steps.append({"entries": step.N, "live_ctas": step.live_ctas(), "slots": checksum(step.slots), "lens": checksum(step.lens)})
    torch.cuda.synchronize()
    return pool.pos.cpu().tolist(), {"path": path.name, "impl": path.impl, "launches": steps}


def oob_workload(mod, path, offset, accesses, assert_launches=False):
    """A decode launch of four sequences (H = 2, a pool of 12 slots) on ``path`` of the variant ``mod`` built with CFBI_ROWS_TEST_POS_OOB, with the position accesses selected by ``accesses``
    (bit 0: the fused store of the two-launch row engine, bit 1: the arrival atomic and bit 2: the elected store of the single launch) moved by ``offset``."""
    H, S = OOB_HEADS, OOB_SLOTS
    par = rw.make_params(H, 7)
    pool = alloc_state(S, H)
    mod.rows_test_pos_offset(offset, accesses)
    step = Step(par, H, [1] * OOB_SEQUENCES, [3, 7, 1, 9], True, 71)
    run(mod, path, pool, step, assert_launches)
    torch.cuda.synchronize()
    return {"path": path.name, "impl": path.impl, "offset": offset, "accesses": accesses}
