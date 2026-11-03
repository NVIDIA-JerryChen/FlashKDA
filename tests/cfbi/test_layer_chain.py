"""Integration gate: the whole KDA layer front end, projections -> short conv (+ SiLU) -> CF-BI core, is bitwise train/infer
consistent.

Two ways to run the same layer on the same tokens:
  * training-style: one call over the whole sequence (Linear on all T rows, conv with an empty state, ``forward_train``);
  * inference-style: a prefill call over the first P tokens with conv / core state, then one-token decode calls (Linear on
    ONE row, conv with state, ``decode``) - also inside a varlen batch of sequences with a scheduler-like mix of prefill
    and lock-step decode.
Everything is compared bit for bit: the projection outputs (bf16 and the fp32 values before the rounding), the conv outputs
(bf16 and fp32), the core outputs (bf16 and fp32), and the final conv / core states. Negative controls: the same harness
with the sabotaged builds (a decode path that adds the conv taps in another association / a small-M linear that walks K
backwards) must fail, and so must a chain whose projections go through cuBLAS where cuBLAS is not batch invariant.

The layer follows FLA's ``KimiDeltaAttention``: q/k/v = SiLU(depthwise_conv4(Linear(x))), g = Linear(Linear(x)) (low rank),
beta = Linear(x); the gated RMSNorm and o_proj after the core are not part of this check.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from flash_kda.cfbi import (alloc_conv_state, alloc_state, bi_linear, decode, forward_train, prefill, short_conv,
                            state_equal)

from convlin_helpers import cu_tensor, cx, cx_neg, n_diff  # noqa: F401

HD = 128   # head dim (K = V = 128)
W = 4      # conv kernel size
LB = -5.0


class Ops:
    """The two building blocks the chain calls, each returning (bf16 result, fp32 value before the rounding or None).
    ``lin_ext`` / ``conv_ext``: a differently built extension (negative controls) instead of the production wrappers;
    ``cublas``: F.linear for the projections."""

    def __init__(self, cublas=False, lin_ext=None, conv_ext=None):
        self.cublas, self.lin_ext, self.conv_ext = cublas, lin_ext, conv_ext

    def linear(self, x2d, w):
        if self.cublas:
            return F.linear(x2d, w), None
        o32 = torch.zeros(x2d.shape[0], w.shape[0], dtype=torch.float32, device=x2d.device)
        if self.lin_ext is None:
            return bi_linear(x2d, w, out_fp32=o32), o32
        out = torch.empty(x2d.shape[0], w.shape[0], dtype=torch.bfloat16, device=x2d.device)
        self.lin_ext.linear_forward(x2d.contiguous(), w, out, None, -1, o32)
        return out, o32

    def conv(self, x3d, w, state, cu):
        o32 = torch.zeros(x3d.shape, dtype=torch.float32, device=x3d.device)
        if self.conv_ext is None:
            return short_conv(x3d, w, None, state, "silu", cu, out_fp32=o32), o32
        out = torch.empty_like(x3d)
        self.conv_ext.conv_forward(x3d, w, out, None, state, cu, None, True, o32)
        return out, o32


class Layer:
    def __init__(self, Dm, H, seed=0):
        g = torch.Generator(device="cuda")
        g.manual_seed(seed)

        def rb(*shape, scale):
            return (torch.randn(*shape, device="cuda", generator=g) * scale).to(torch.bfloat16)

        self.Dm, self.H = Dm, H
        kd = H * HD
        s = 1.0 / math.sqrt(Dm)
        self.wq, self.wk, self.wv = rb(kd, Dm, scale=s), rb(kd, Dm, scale=s), rb(kd, Dm, scale=s)
        self.f1, self.f2 = rb(HD, Dm, scale=s), rb(kd, HD, scale=1.0 / math.sqrt(HD))
        self.bw = rb(H, Dm, scale=s)
        self.cq, self.ck, self.cv = (torch.randn(kd, W, device="cuda", generator=g) * 0.6 for _ in range(3))
        self.A_log = torch.rand(H, device="cuda", generator=g)
        self.dt_bias = torch.rand(H, HD, device="cuda", generator=g)
        self.scale = 1.0 / math.sqrt(HD)

    def alloc(self, N):
        kd = self.H * HD
        return dict(conv=[alloc_conv_state(N, kd, W) for _ in range(3)], core=alloc_state(N, self.H))

    def forward(self, ops, x, states, cu=None, mode="prefill"):
        """x: bf16 [B, T, Dm]. states None -> training forward. mode: "prefill" (state in/out) or "decode" (B = N, T = 1)."""
        B, T, Dm = x.shape
        H = self.H
        x2 = x.reshape(B * T, Dm)
        rec = {}
        rec["pq"], rec["pk"], rec["pv"] = (ops.linear(x2, w) for w in (self.wq, self.wk, self.wv))
        f1 = ops.linear(x2, self.f1)[0]
        rec["g"] = ops.linear(f1, self.f2)
        rec["beta"] = ops.linear(x2, self.bw)
        cs = states["conv"] if states is not None else [None] * 3
        rec["q"] = ops.conv(rec["pq"][0].view(B, T, -1), self.cq, cs[0], cu)
        rec["k"] = ops.conv(rec["pk"][0].view(B, T, -1), self.ck, cs[1], cu)
        rec["v"] = ops.conv(rec["pv"][0].view(B, T, -1), self.cv, cs[2], cu)
        q, k, v = (rec[n][0].view(B, T, H, HD) for n in "qkv")
        g = rec["g"][0].view(B, T, H, HD)
        beta = rec["beta"][0].view(B, T, H)
        out = torch.empty_like(q)
        o32 = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
        if states is None:
            forward_train(q, k, v, g, beta, self.scale, self.A_log, self.dt_bias, LB, cu, out=out, out_fp32=o32)
        elif mode == "decode":
            decode(q, k, v, g, beta, self.scale, out, self.A_log, self.dt_bias, LB, states["core"], out_fp32=o32)
        else:
            prefill(q, k, v, g, beta, self.scale, out, self.A_log, self.dt_bias, LB, states["core"], cu, out_fp32=o32)
        rec["core"] = (out, o32)
        return rec


KEYS = ("pq", "pk", "pv", "g", "beta", "q", "k", "v", "core")


def flat(t):
    """[.., C...] -> [tokens, features] with the token axes first (works for [B,T,D], [B,T,H,D], [M,N], [B,T,H])."""
    return t.reshape(t.shape[0] if t.dim() == 2 else t.shape[0] * t.shape[1], -1)


def cmp_records(whole, parts):
    """whole: record of the training-style call over the sequence; parts: list of (record, row_lo, row_hi) of the serving
    calls that cover the same tokens in order. Returns the number of differing elements per kind."""
    bad = {}
    for key in KEYS:
        for j, kind in enumerate(("bf16", "fp32")):
            a = whole[key][j]
            if a is None or any(r[key][j] is None for r, _, _ in parts):   # (cuBLAS has no fp32 side channel)
                continue
            ref = flat(a)
            got = torch.cat([flat(r[key][j])[lo:hi] for r, lo, hi in parts], dim=0)
            bad[(key, kind)] = n_diff(ref, got)
    return bad


def total(bad):
    return sum(bad.values())


def single_sequence(layer, ops_whole, ops_serve, T=150, P=110, seed=1):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    x = torch.randn(1, T, layer.Dm, device="cuda", generator=g).to(torch.bfloat16)
    whole = layer.forward(ops_whole, x, None)
    st = layer.alloc(1)
    parts = [(layer.forward(ops_serve, x[:, :P].contiguous(), st), 0, P)]
    for t in range(P, T):
        parts.append((layer.forward(ops_serve, x[:, t:t + 1].contiguous(), st, mode="decode"), 0, 1))
    bad = cmp_records(whole, parts)
    # final states: conv states = last W-1 pre-conv inputs, core state: compare with a whole prefill from the empty state
    ref_st = layer.alloc(1)
    layer.forward(ops_serve, x, ref_st)
    st_bad = {f"conv{i}": n_diff(st["conv"][i], ref_st["conv"][i]) for i in range(3)}
    st_bad["core"] = 0 if state_equal(st["core"], ref_st["core"]) else 1
    return bad, st_bad


@pytest.mark.parametrize("Dm,H", [(1024, 4), (4096, 16)])
def test_whole_sequence_vs_prefill_then_decode(cx, Dm, H):
    layer = Layer(Dm, H, seed=Dm)
    bad, st_bad = single_sequence(layer, Ops(), Ops())
    assert total(bad) == 0, {k: v for k, v in bad.items() if v}
    assert all(v == 0 for v in st_bad.values()), st_bad
    print(f"\n[chain Dm={Dm} H={H}] whole call vs prefill(110) + 40 decode calls: 0 differing elements in "
          f"{len(bad)} tensors (projections, conv, core; bf16 and fp32) and the final conv / core states")


def test_batched_prefill_and_lockstep_decode(cx):
    """Varlen prefill of N sequences (different lengths), then lock-step decode of the ones that are still running, with
    the active slots compacted into a batch of fewer rows as sequences finish; every sequence == its own whole call."""
    Dm, H = 1024, 4
    layer = Layer(Dm, H, seed=7)
    lens, pre = [150, 70, 33, 1], [110, 40, 13, 1]
    N = len(lens)
    g = torch.Generator(device="cuda")
    g.manual_seed(2)
    xs = [torch.randn(1, L, Dm, device="cuda", generator=g).to(torch.bfloat16) for L in lens]
    wholes = [layer.forward(Ops(), x, None) for x in xs]
    # prefill: one varlen call with the first pre[n] tokens of every sequence
    cu = [0]
    for p in pre:
        cu.append(cu[-1] + p)
    xp = torch.cat([xs[n][:, :pre[n]] for n in range(N)], dim=1).contiguous()
    st = layer.alloc(N)
    rec_p = layer.forward(Ops(), xp, st, cu=cu_tensor(cu))
    seqs_parts = [[(rec_p, cu[n], cu[n + 1])] for n in range(N)]
    done = list(pre)
    steps = 0
    while any(done[n] < lens[n] for n in range(N)):
        active = [n for n in range(N) if done[n] < lens[n]]
        ids = torch.tensor(active, device="cuda")
        sub = dict(conv=[c[ids].contiguous() for c in st["conv"]], core=st["core"].select(ids))
        xd = torch.cat([xs[n][:, done[n]:done[n] + 1] for n in active], dim=0).contiguous()   # [n_active, 1, Dm]
        rec = layer.forward(Ops(), xd, sub, mode="decode")
        for c, s in zip(st["conv"], sub["conv"]):
            c[ids] = s
        st["core"].E[ids], st["core"].rows[ids], st["core"].pos[ids] = sub["core"].E, sub["core"].rows, sub["core"].pos
        for j, n in enumerate(active):
            seqs_parts[n].append((rec, j, j + 1))
            done[n] += 1
        steps += 1
    for n in range(N):
        bad = cmp_records(wholes[n], seqs_parts[n])
        assert total(bad) == 0, (n, {k: v for k, v in bad.items() if v})
    print(f"\n[chain, batched] {N} sequences of {lens} tokens: varlen prefill {pre} + {steps} lock-step decode steps with "
          f"compaction: 0 differing elements in every projection / conv / core tensor (bf16 and fp32)")


def test_negative_controls_are_caught(cx, cx_neg):
    """The harness above must fail for a layer whose conv decode path sums the taps differently, whose small-M linear walks
    K backwards, or whose projections are cuBLAS matmuls (where cuBLAS is not batch invariant). The reference side of
    every comparison is the production build; only the serving side (prefill + decode calls) is sabotaged."""
    layer = Layer(4096, 16, seed=4096)
    out = {}
    for name, ops in (("conv: different decode association", Ops(conv_ext=cx_neg)),
                      ("linear: small-M walks K backwards", Ops(lin_ext=cx_neg)),
                      ("projections through cuBLAS", Ops(cublas=True))):
        bad, _ = single_sequence(layer, Ops(), ops, T=150, P=110)
        out[name] = {k: v for k, v in bad.items() if v}
    print("\n[chain negative controls] differing elements per tensor (the production chain: none):")
    for name, d in out.items():
        print(f"  {name}: {d if d else 'NONE'}")
    conv_bad = out["conv: different decode association"]
    assert any(k[0] in ("q", "k", "v") and k[1] == "fp32" for k in conv_bad), "conv association change not caught"
    assert not any(k[0] in ("pq", "pk", "pv", "g", "beta") for k in conv_bad)   # the projections were not touched
    lin_bad = out["linear: small-M walks K backwards"]
    assert any(k[0] in ("pq", "pk", "pv", "g", "beta") and k[1] == "fp32" for k in lin_bad), "linear K-order change not caught"
    if not out["projections through cuBLAS"]:
        print("  (cuBLAS happened to be batch invariant for these shapes: that control is inconclusive)")
