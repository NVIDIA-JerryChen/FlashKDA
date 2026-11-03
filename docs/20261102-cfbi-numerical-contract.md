# CF-BI KDA numerical contract v1

Status: implemented by `csrc/cfbi/` (extension `flash_kda_cfbi_C`), used by `flash_kda.cfbi`.
Validated only on B300 (sm_103a); see "What is and is not guaranteed".

This document is the *specification* of the floating-point program. Prefill, the training forward and
decode are three schedules of this one program; they differ only in how many rows are evaluated per
call. Any change to a formula below is a numerics change (new contract version), not an optimisation.

## 0. Notation

- Per sequence and head. K = V = 128, chunk length BT = 64, **anchored at the sequence start** (never
  at a call boundary), sub-block = 16 rows, reference row `r(t) = 16*floor(t/16)` (`t` = row index inside
  the chunk, 0..63).
- fp32 everywhere except tensor-core operands. "round" = IEEE round-to-nearest-even. `fma` = one rounding.
  Every `*`, `+`, `-` below is its own rounding; nothing is contracted into an FMA unless written `fma`.
- Approximations (each exactly one PTX instruction): `ex2` = `ex2.approx.ftz.f32` (flushes results below
  2^-126 to 0 - intended), `rcp` = `rcp.approx.ftz.f32`, `rsqrt` = `rsqrt.approx.ftz.f32`. Measured max relative
  error on B300: ex2 1.4e-7, rcp ~1.2e-7, rsqrt 1.2e-7 (`tanh.approx.f32`: 1.1e-5, not used).
- Inputs mirror `flash_kda.fwd`: `q, k, v, g` bf16 `[B,T,H,128]`, `beta` bf16 raw logits `[B,T,H]`,
  `A_log` fp32 `[H]`, `dt_bias` fp32 `[H,128]`, `scale`, `lower_bound` in `[-5, 0]`, `cu_seqlens` int64.

## 1. Per-token prep (position-local; one shared device function set)

```
l2inv(x):  16 partial sums p_j = fma chain over x[8j..8j+7] (starting from 0, ascending);
           butterfly p_j <- p_j + p_{j^o} for o = 8,4,2,1;  inv = rsqrt(p + 1e-6f)
khat = k * l2inv(k)                      (fp32, NOT rounded to bf16)
qhat = (q * l2inv(q)) * scale            (scale rounded to fp32)
gate scalar per head:  ale = ex2(A_log * LOG2E_f)         (LOG2E_f = fp32(log2 e))
a2[c]  = scale2 * sig(ale * (g[c] + dt_bias[c])),  scale2 = fp32(lower_bound * log2 e) (double product)
sig(x) = rcp(1 + ex2(-x * LOG2E_f))
beta   = sig(beta_logit)
```
`a2 <= 0` and `a2 >= scale2`, so the log2-decay of one row is in `[scale2, 0]` (>= -7.22 bits).

## 2. Chunk program (rows t = 0..63 of one chunk; E = state at chunk entry, fp32 `[V][K]`)

```
G2[0] = a2[0];  G2[t] = G2[t-1] + a2[t]          (sequential cumsum, per channel)
Gamma[t] = ex2(G2[t])                            (absolute inside the chunk; underflow to 0 is intended)
ks[t]  = khat[t] * ex2(G2[t] - G2[r(t)])         qs[t] = qhat[t] * ex2(G2[t] - G2[r(t)])
kjr[j;r] = khat[j] * ex2(G2[r] - G2[j])          (reference row r = r(t) of the consuming row t)
Akk[t][j] = beta[t] * dot3_K128(ks[t], kjr[j;r(t)])        j <  t
Aqk[t][j] =           dot3_K128(qs[t], kjr[j;r(t)])        j <= t
```
Exponent range: within a sub-block `ks <= khat`, `kjr <= khat * 2^108`; across sub-blocks `kjr <= khat`.

Triangular solves, `X in {UT, W}`, `b = r(t)`:
```
R_UT[t] = beta[t] * v[t]          R_W[t] = beta[t] * (Gamma[t] * khat[t])
X[t] = R[t]                                  if b == 0
X[t] = R[t] - dot3_Kb(Akk[t][0:b], X[0:b])   if b  > 0      (K = b, a multiple of 16)
for j = b .. t-1 ascending:  X[t] = fma(-Akk[t][j], X[j], X[t])
```
Entry-state products and output:
```
GQ[t] = Gamma[t] * qhat[t]
P1[t][v] = dot3_K128(W[t], E[v][:])          P2[t][v] = dot3_K128(GQ[t], E[v][:])
U[t] = UT[t] - P1[t]
O[t] = P2[t]                                  if b == 0
O[t] = P2[t] + dot3_Kb(Aqk[t][0:b], U[0:b])   if b  > 0
for j = b .. t ascending:  O[t] = fma(Aqk[t][j], U[j], O[t])
out[t] = bf16_rn(O[t])
```
Chunk end (row 63 completed): `G2e = G2[63]`
```
KD[k][j] = khat[j][k] * ex2(G2e[k] - G2[j][k])        GEND[k] = ex2(G2e[k])
E'[v][k] = (E[v][k] * GEND[k]) + dot3_K64(KD[k][0:64], U[0:64][v])
```

## 3. The `dot3` primitive and tile geometry

`dot3_Kn(x, y)` is the three-term TF32 product, K-steps of 8 in ascending order:
```
x_hi = bits(x) & 0xFFFFE000;   x_lo = bits(x - x_hi) & 0xFFFFE000      (same for y)
BIG = SMALL = -0.0
per K-step:  SMALL = mma(lo_x, hi_y, SMALL);  SMALL = mma(hi_x, lo_y, SMALL);  BIG = mma(hi_x, hi_y, BIG)
result = BIG + SMALL                                                    (after the last K-step)
```
with `mma` = `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`. There is **one** device template for this
(`dot3<>` in `cfbi_primitives.cuh`) used by every kernel, with one tile geometry:

| product | A operand (16 row slots) | B operand (8 column slots) | K index |
|---|---|---|---|
| Akk / Aqk | `ks[t]` / `qs[t]`, slot `t%16` | `kjr[j]`, slot `j%8` | channel (128) |
| triangular M | `Akk[t][0:b]`, slot `t%16` | `X[j][c]`, slot `c%8` | `j` (b) |
| P1 / P2 | `W[t]` / `GQ[t]`, slot `t%16` | `E[v][k]`, slot `v%8` | `k` (128) |
| output M | `Aqk[t][0:b]`, slot `t%16` | `U[j][v]`, slot `v%8` | `j` (b) |
| P3 | `KD[k][0:64]`, slot `k%16` | `U[j][v]`, slot `v%8` | `j` (64) |

Invalid slots (rows not yet existing, columns beyond the valid range) are zero-filled. Prefill fills up to
16 row slots, decode one; the output element must not depend on it - this is the **hardware premise**,
verified by `tests/cfbi/test_mma_probe.py` (same pair in every slot x neighbour pattern => identical bits;
reversed K order => different bits).

## 4. State

Per sequence and head: `E` fp32 `[V][K]` (same orientation as `flash_kda.fwd` states), the open-chunk row
cache (fp32 planes `khat, G2, UT, W, U`, each `[64][128]`, rows `0..pos-1` valid) and `pos` in `0..63`
(int32 per sequence). Rows are appended as they are computed; row 63 triggers the chunk-end update of `E` and
`pos = 0`. Rows `>= pos` are stale and not part of the state.

Bit-identical across schedules (whole prefill, any call cuts, T=1 decode, mixed batches, head sharding, any
batch composition): the outputs, `E`, the valid cache rows and `pos`.

## 5. Differences from the CF-BI report (decisions made here)

- The report does not specify the elementwise approximations, the l2norm reduction order, the gate/beta
  forms or whether `khat/qhat` are rounded; section 1 above fixes them (taken from master's `torch_ref.py`,
  except that `khat/qhat` stay fp32 and **the sigmoid is `rcp(1 + ex2(-x log2 e))` instead of master's
  `fma(tanh.approx(x/2), .5, .5)`**). Reason, measured on B300: `tanh.approx.f32` has a max relative error of
  1.1e-5 (sigmoid abs error up to 3.9e-6) and made the program's fp32 error 2.4e-6 against 0.7e-6 for the same
  program with exact transcendentals; with ex2+rcp the plan's accuracy target is met (see the design doc).
  This is the one deliberate departure from the plan's text, which pinned master's tanh form.
- For `b == 0` the report takes `R[t]` / `P2[t]` directly (as here); no `-0.0` subtract is executed.
- Performance-only deviations (no effect on bits): `mma.sync` everywhere (no tcgen05/UMMA), global fp32
  workspace between K1 and K2, no warp specialisation, whole-head CTA for decode.

## 6. What is and is not guaranteed

Guaranteed (and gated by tests): on one GPU architecture + one compiler build, the same sequence gives the
same bits however it is cut or batched, and the training forward equals prefill equals decode.

Not guaranteed: other GPU architectures or compiler versions (the hardware premise was only validated on
B300; `flash_kda.cfbi` refuses `compute capability < 10.0` unless `FLASH_KDA_CFBI_ALLOW_UNVALIDATED_ARCH=1`),
other implementations of the same math, TF32 denormal handling of the tensor cores (hardware-defined, but
identical on every path), context parallelism, conv/projection/gated-norm alignment. The backward pass is
**not** bit-pinned: it recomputes the math from the inputs and only has to produce correct gradients.
