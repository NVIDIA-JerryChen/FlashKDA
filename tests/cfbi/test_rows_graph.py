"""CUDA-graph replay of a decode launch beyond 2.5 resident waves, with ``slot_ids`` rewritten between replays, bit for bit against the eager reference (``_impl=3``).

A serving engine captures one fixed-shape decode launch (inputs, ``slot_ids``, outputs and the state pool are static buffers) and replays it with new tokens, new slots and new padding
entries (slot -1) every step; the indirection is resolved on the device. The older test of this (``test_slot_ids.py``) replays a graph of five entries. Here the graph holds enough
sequences that its live CTAs fill at least 2.5 waves of resident CTAs of the optimised row engine (sized from the occupancy binding), the slots are permuted and padded anew at every replay,
the state starts from a warm pool with open chunks, and after every replay the outputs (bf16, fp32), E, the valid cache rows and ``pos`` are compared with an eager reference launch that
runs the same inputs and the same ``slot_ids`` on its own copy of the pool.

Control: a replay whose ``slot_ids`` buffer was not rewritten, compared with a reference that ran the new slots, differs: the comparison does see the slots.

A graph replay that never completes would hang the test from inside its own process, so the two cases run as one group in a child process under ``GROUP_TIMEOUT_S`` (``rows_bounded``): a
hang fails the test and kills the child's process group; the cases keep their ids and their assertions.
"""
import random
import types

import pytest
import torch

import helpers
import rows_bounded as rb
import rows_workload as rw
from flash_kda.cfbi import alloc_state
from helpers import make_case, skip_or_fail
from rows_workload import Outcome

H = 12
STEPS = 10
PAD = 0.3
GROUP_TIMEOUT_S = 600     # the two cases take a few seconds on a B200 (a first run adds the JIT build of the extension); a hang ends here


@pytest.fixture(autouse=True)
def _optimised_kernels_run_here(ext):
    if ext.resolve_impl(rw.OPTIMISED, -1) == ext.resolve_impl(rw.REFERENCE, -1):
        skip_or_fail("the optimised kernels do not run on this device (_impl=0 resolves to the reference kernels)")


class Replay:
    """The captured graph and everything it replays on; ``next_slots`` draws permuted slots with padding entries."""

    def __init__(self, ext, seed):
        self.ext, self.rng = ext, random.Random(seed)
        self.par = rw.make_params(H, seed)
        live = helpers.sequences_beyond_waves(helpers.resident_ctas(ext), H)
        self.n_pad = int(live * PAD)
        self.N = live + self.n_pad
        self.live_ctas = live * H
        self.S = self.N + rw.POOL_SPARE
        src = make_case(STEPS * self.N + self.N, H, "authors", seed=seed)
        self.src = {k: src[k].view(STEPS + 1, self.N, 1, H, 128) for k in ("q", "k", "v", "g")}
        self.src_beta = src["beta"].view(STEPS + 1, self.N, 1, H)
        self.bufs = {k: torch.zeros(self.N, 1, H, 128, dtype=torch.bfloat16, device="cuda") for k in ("q", "k", "v", "g")}
        self.beta = torch.zeros(self.N, 1, H, dtype=torch.bfloat16, device="cuda")
        self.out = torch.empty_like(self.bufs["q"])
        self.out32 = torch.empty(self.bufs["q"].shape, dtype=torch.float32, device="cuda")
        self.ids = torch.full((self.N,), -1, dtype=torch.int32, device="cuda")
        warm = rw.warm_pool(ext, self.S, H, rw.draw_fills(self.rng, self.S, "open"), self.par, seed)
        self.pool, self.ref_pool = warm.clone(), warm.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):       # warm-up on a scratch pool (attribute setup, allocations), as the capture protocol asks
            self.launch(alloc_state(self.S, H))
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.launch(self.pool)

    def launch(self, state):
        p = self.par
        self.ext.forward(self.bufs["q"], self.bufs["k"], self.bufs["v"], self.bufs["g"], self.beta, float(p["scale"]), self.out, p["A_log"], p["dt_bias"], float(p["lower_bound"]),
                         state.E, state.rows, state.pos, None, 1, self.out32, rw.OPTIMISED, self.ids)

    def next_slots(self):
        slots = self.rng.sample(range(self.S), self.N)
        pads = set(self.rng.sample(range(self.N), self.n_pad))
        return [-1 if n in pads else slots[n] for n in range(self.N)]

    def load_inputs(self, t):
        for k in self.bufs:
            self.bufs[k].copy_(self.src[k][t])
        self.beta.copy_(self.src_beta[t])

    def reference(self, slot_ids):
        """The eager reference launch of the current inputs with ``slot_ids`` on the reference pool: ``(out, out32)``."""
        view = types.SimpleNamespace(par=self.par, q=self.bufs["q"], k=self.bufs["k"], v=self.bufs["v"], g=self.bufs["g"], beta=self.beta, cu=None, slot_ids=slot_ids)
        return rw.run_step(self.ext, rw.REFERENCE, self.ref_pool, view)

    def replay(self, slots):
        self.ids.copy_(torch.tensor(slots, dtype=torch.int32))
        self.out.fill_(rw.NAN)
        self.out32.fill_(rw.NAN)
        self.graph.replay()
        torch.cuda.synchronize()


# ---- the cases (they run in the child process of their group) -------------------------------------------------------------------------------------------
def rewritten_case(ext):
    r = Replay(ext, 91)
    assert r.live_ctas >= helpers.WAVES * helpers.resident_ctas(ext), "the graph's live grid is not beyond %.1f resident waves" % helpers.WAVES
    seen = set()
    for t in range(STEPS):
        slots = r.next_slots()
        seen.add(tuple(slots))
        r.load_inputs(t)
        r.replay(slots)
        ref_out, ref_out32 = r.reference(torch.tensor(slots, dtype=torch.int32, device="cuda"))
        torch.cuda.synchronize()
        m = rw.mismatch(Outcome(r.out, r.out32, r.pool), Outcome(ref_out, ref_out32, r.ref_pool))
        assert all(v == 0 for v in m.values()), "replay %d differs from the reference: %s" % (t, m)
        assert 0 <= int(r.pool.pos.min()) and int(r.pool.pos.max()) < rw.BT
    assert len(seen) == STEPS, "the slots were not rewritten at every replay"


def stale_case(ext):
    r = Replay(ext, 92)
    old = r.next_slots()
    r.load_inputs(0)
    r.replay(old)
    r.reference(torch.tensor(old, dtype=torch.int32, device="cuda"))
    new = r.next_slots()
    assert new != old
    r.load_inputs(1)
    r.out.fill_(rw.NAN)
    r.out32.fill_(rw.NAN)
    r.graph.replay()                                     # the graph still reads the slots of the previous step: they were not rewritten
    ref_out, ref_out32 = r.reference(torch.tensor(new, dtype=torch.int32, device="cuda"))
    torch.cuda.synchronize()
    m = rw.mismatch(Outcome(r.out, r.out32, r.pool), Outcome(ref_out, ref_out32, r.ref_pool))
    assert sum(m.values()) > 0, "a replay with stale slot_ids matches a reference that ran the new slots: the comparison cannot see the slots (%s)" % m


# ---- the group (one child process) ------------------------------------------------------------------------------------------------------------------------
def group_graph(ext):
    return {"cases": {"rewritten_slot_ids": rb.capture(rewritten_case, ext), "stale_slot_ids": rb.capture(stale_case, ext)}, "reps": STEPS}


@pytest.fixture(scope="module")
def graph_group(ext):
    return rb.run_group("test_rows_graph", "group_graph", {}, GROUP_TIMEOUT_S)


def test_graph_replay_with_rewritten_slot_ids_is_bitwise_the_reference_beyond_2_5_waves(graph_group):
    assert graph_group["cases"]["rewritten_slot_ids"] is None, graph_group["cases"]["rewritten_slot_ids"]


def test_a_replay_with_stale_slot_ids_differs_from_the_reference(graph_group):
    assert graph_group["cases"]["stale_slot_ids"] is None, graph_group["cases"]["stale_slot_ids"]
