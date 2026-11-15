#!/usr/bin/env python3
"""Host lane / offset model of the once-per-row A fragments of the row engine (the staged tiles of ``FragSmem`` in csrc/cfbi/cfbi_rows.cuh). CPU only, no CUDA, no GPU.

usage: afrag_model.py [--seed N] [--states N] [--quiet]        (the tests call the functions; the command line prints the report)

What it models. The row engine before the staging rebuilt, in EVERY warp and for every k-step, three families of tensor-core A fragments (TF32 hi / lo words of
m16n8k8, lane = 4 g + t, a0..a3 = (g, t), (g + 8, t), (g, t + 4), (g + 8, t + 4), element ii = row g + 8 (ii & 1), column t + 4 (ii >> 1)):
  AB  phase B (lines 390-399): ks / qs rows, A row slot = ks, slot ^ 8 = qs, 16 k-steps, dense for the ``mine`` lanes (g == slot & 7), zero for the others
  AM  rows_m_tiles (lines 162-170): the akk row and the aqk row (two tiles, each with ONLY row ``slot`` non-zero), KS = 2 s k-steps (s = pos >> 4, s > 0)
  AD  phase D (lines 478-483): W / GQ rows, same shape as AB, 16 k-steps
The design stages each family once per row into shared memory (uint4 per (k-step, [which,] hi/lo, t): the four mine lanes read 64 contiguous bytes), written by one producer thread per word,
read with LDS.128 by the mine lanes and from a zero tile (the same relative offsets, 64 mod 128 bytes away) by the others. This script (1) transcribes T's expressions, (2) implements the
producer -> tile -> consumer protocol on a PERSISTENT shared memory that is poisoned at kernel start (a word that a row does not rewrite is read stale), (3) compares every register word bit
for bit for every pos 0..63 (and the chunk-end wrap) on random operands and special values, (4) checks unique writers / full coverage per row, that the staged side really observed its reads, the
order of every read after its write by the barrier plan, the reuse of a tile by the next row, 16-byte / 128-byte alignment, (5) reports the shared-memory wavefronts of the loads and the
stores, (6) rejects its negative controls: six mutations x 3 families (a shifted lane order, the wrong row parity, a missing word, a double writer, a missing barrier, non-mine lanes reading
the tile) and one AM-only mutation (the zero-parity words are stored once at kernel start and not rewritten per row: invisible until slot_hi flips; AB and AD are dense, they have no zero words
to skip) - each must be rejected for each family it applies to.
The model never evaluates the MMA or the hardware ex2: both sides call the same Python ``ex2`` (a deterministic float32 function), because the claim under test is the MAPPING and the
placement of the same operands, not the arithmetic of the instruction.
"""
import argparse
import sys

import numpy as np

F32 = np.float32
LANES = np.arange(32)
G, T4 = LANES >> 2, LANES & 3
KS_ALL = np.arange(16)
II = np.arange(4)
POISON = np.uint32(0xA5A5A5A5)
FAMILIES = ("AB", "AM", "AD")

# ---- the layout (design F v1): byte offsets inside the static shared memory of the CTA ----------------------------------------------------------------------------------------------------
ROWSMEM_T = 7684                       # sizeof(RowSmem) at T: 6 x 512 (k q g2 gref gq gend) + 1024 (x) + 1536 (m) + 1536 (p1 p2 u) + 512 (akk aqk) + 4 (beta)
DEC_POS0 = 4                           # the int of rows_kernel<true>
AB_WORDS, AM_WORDS, AD_WORDS = 16 * 2 * 4 * 4, 6 * 2 * 2 * 4 * 4, 16 * 2 * 4 * 4          # words (uint32) of the tiles: 512, 384, 512


def up(x, a):
    return (x + a - 1) // a * a


ROWSMEM_F = ROWSMEM_T - 2 * 512        # F deletes S.gq and S.x[1] (they only fed the phase-D A fragments, lines 449 / 451 / 480): RowSmem of F v1 = 6660 B
# Offsets INSIDE ``struct alignas(128) FragSmem`` (the struct, not the CTA's shared window, fixes the geometry: where the compiler places the struct is its business, the 128-byte alignment is the contract).
# The tiles start at 128-byte boundaries (the 64-byte groups of the mine lanes then sit in one half of the 32 banks). The zero page sits FIRST (at 64 mod 128), the three tiles form the
# SUFFIX of the struct, so that a later tile (the chunk-end tile) can grow the overlay arena upward without ever covering the zero page.
ZERO_WORDS = max(AB_WORDS, AM_WORDS, AD_WORDS)         # the +0 page: as big as the largest tile, because a load takes the SAME immediate offset for every lane (a 16-byte page would not do)
ZERO_OFF = 64                                          # 64 mod 128: the zero twin of a 64-byte group lies in the other half of the banks (a 64-byte pad in front of it)
AB_OFF = up(ZERO_OFF + 4 * ZERO_WORDS, 128)            # 2176: the 64-byte pad after the zero page restores the 128-byte boundary
AM_OFF = AB_OFF + 4 * AB_WORDS
AD_OFF = AM_OFF + 4 * AM_WORDS
END_OFF = AD_OFF + 4 * AD_WORDS
ARENA = (AB_OFF, END_OFF)                              # the overlay arena of the three tiles (live from the AB stores to the end of phase D), a suffix of the struct
FRAG_SIZE = up(END_OFF, 128)                           # sizeof(FragSmem): a multiple of its alignment
STATIC_F = up(ROWSMEM_F, 128) + FRAG_SIZE + DEC_POS0   # worst case: RowSmem first, FragSmem after its padding, then the int of rows_kernel<true>; the compiler's own figure is the one that counts
OFF = {"AB": AB_OFF, "AM": AM_OFF, "AD": AD_OFF}
LAYOUT = {"AB": AB_OFF, "AM": AM_OFF, "AD": AD_OFF, "ZERO": ZERO_OFF}


# ---- float32 / bit primitives (the same functions on both sides) ----------------------------------------------------------------------------------------------------------------------------
def bits(x):
    return np.ascontiguousarray(x, dtype=F32).view(np.uint32)


def flt(u):
    return np.ascontiguousarray(u, dtype=np.uint32).view(F32)


def fmul(a, b):
    with np.errstate(all="ignore"):
        return (a.astype(F32) * b.astype(F32)).astype(F32)


def fsub(a, b):
    with np.errstate(all="ignore"):
        return (a.astype(F32) - b.astype(F32)).astype(F32)


def ex2(x):
    with np.errstate(all="ignore"):
        return np.exp2(x.astype(F32)).astype(F32)


def split(x):
    """cfbi_primitives.cuh split_tf32: hi = bits & 0xFFFFE000, lo = bits(fsub(x, hi)) & 0xFFFFE000."""
    hi = bits(x) & np.uint32(0xFFFFE000)
    lo = bits(fsub(x, flt(hi))) & np.uint32(0xFFFFE000)
    return hi, lo


# ---- random state of one row with special values ---------------------------------------------------------------------------------------------------------------------------------------------
SPECIALS = [0.0, -0.0, np.nan, np.inf, -np.inf, 1e-45, -1e-45, 3.4e38, -3.4e38, 1.0, -1.0, 1.17549435e-38]


def rand_vec(rng, n, special):
    v = (rng.standard_normal(n) * 10.0 ** rng.uniform(-3, 3, n)).astype(F32)
    if special:
        for _ in range(max(1, n // 16)):
            v[rng.integers(0, n)] = F32(SPECIALS[rng.integers(0, len(SPECIALS))])
    return v


def make_state(rng, special):
    g2 = (-np.abs(rand_vec(rng, 128, False)) * F32(0.05)).astype(F32)
    gref = (-np.abs(rand_vec(rng, 128, False)) * F32(0.05)).astype(F32)
    if special:
        g2 = rand_vec(rng, 128, True)
        gref = rand_vec(rng, 128, True)
    return {"k": rand_vec(rng, 128, special), "q": rand_vec(rng, 128, special), "g2": g2, "gref": gref, "akk": rand_vec(rng, 64, special), "aqk": rand_vec(rng, 64, special),
            "W": rand_vec(rng, 128, special)}


def gq_of(S):
    return fmul(ex2(S["g2"]), S["q"])        # phase C: S.gq[col] = fmul(ex2_ftz(S.g2[col]), S.q[col])


# ---- T: the register words that the kernel builds in every lane (transcribed from the source) --------------------------------------------------------------------------------------
def row_params(pos):
    slot = pos & 15
    return {"s": pos >> 4, "slot": slot, "slot_hi": slot >> 3, "mine": (G == (slot & 7))}


def t_AB(S, pos):
    """-> (hi, lo) uint32 [lane][ks][ii]   (lines 390-399)"""
    P = row_params(pos)
    ch = 8 * KS_ALL[None, :, None] + T4[:, None, None] + 4 * (II[None, None, :] >> 1)          # [lane][ks][ii]
    f = ex2(fsub(S["g2"][ch], S["gref"][ch]))
    ksv, qsv = fmul(S["k"][ch], f), fmul(S["q"][ch], f)
    pick = (II[None, None, :] & 1) == P["slot_hi"]
    v = np.where(P["mine"][:, None, None], np.where(pick, ksv, qsv), F32(0)).astype(F32)
    return split(v)


def t_AM(S, pos):
    """-> ({which: (hi, lo)} uint32 [lane][ks < KS][ii], KS); which 0 = akk (kh, kl), 1 = aqk (qh, ql)   (lines 163-170); (None, 0) for s = 0"""
    P = row_params(pos)
    KS = 2 * P["s"]
    if KS == 0:
        return None, 0
    row = G[:, None, None] + 8 * (II[None, None, :] & 1)
    kk = 8 * np.arange(KS)[None, :, None] + T4[:, None, None] + 4 * (II[None, None, :] >> 1)
    out = {}
    for which, arr in ((0, S["akk"]), (1, S["aqk"])):
        v = np.where(row == P["slot"], arr[kk], F32(0)).astype(F32)
        out[which] = split(v)
    return out, KS


def t_AD(S, pos):
    """-> (hi, lo) uint32 [lane][ks][ii]   (lines 478-483)"""
    P = row_params(pos)
    kk = 8 * KS_ALL[None, :, None] + T4[:, None, None] + 4 * (II[None, None, :] >> 1)
    gq = gq_of(S)
    pick = (II[None, None, :] & 1) == P["slot_hi"]
    v = np.where(P["mine"][:, None, None], np.where(pick, S["W"][kk], gq[kk]), F32(0)).astype(F32)
    return split(v)


# ---- the staging protocol: persistent shared memory, producers, consumers --------------------------------------------------------------------------------------------------------------------
def idx_ab(ks, t, hl, ii):                                # AB / AD: [ks 16][hl 2][t 4][ii 4]
    return ((ks * 2 + hl) * 4 + t) * 4 + ii


def idx_am(ks, t, which, hl, ii):                         # AM: [ks 6][which 2][hl 2][t 4][ii 4]
    return (((ks * 2 + which) * 2 + hl) * 4 + t) * 4 + ii


# the barrier plan of one row, as the epochs in which the accesses happen (every barrier is executed by all 8 warps): epoch 1 follows barrier 1, epoch 2 follows barrier 1b (the one
# added barrier), epoch 3 barrier 2, epoch 4 barrier 3 (s > 0), epoch 5 barrier 4 ...; in the kernel the numbering of barriers is static, the epoch numbers below are the order only.
EPOCH = {"AB_write": 1, "AB_read": 2, "AM_write": 2, "AM_read": 3, "AD_write": 4, "AD_read": 5}
EPOCHS_PER_ROW = 8                                        # barriers 1, 1b, 2, 3, 4, 5, (6), 7 of a row: the next row's epochs are shifted by this


class Smem:
    """The shared memory of one CTA: the three tiles (poisoned at kernel start, then persistent over the rows) and the zero tile; the write log and the read log are per row."""

    def __init__(self):
        self.w = {"AB": np.full(AB_WORDS, POISON, np.uint32), "AM": np.full(AM_WORDS, POISON, np.uint32), "AD": np.full(AD_WORDS, POISON, np.uint32)}
        self.zero = np.zeros(ZERO_WORDS, np.uint32)
        self.begin_row()

    def begin_row(self):
        self.log = {k: [] for k in self.w}                # (word index, thread, epoch)
        self.reads = []                                   # (tile, word index, lane, epoch)

    def write(self, tile, idx, val, tid, epoch):
        self.w[tile][idx] = val
        self.log[tile].append((int(idx), int(tid), int(epoch)))


def active(mut, fam):
    """The mutation name for this family, or None (a control mutates one family at a time, so that no family is hidden by another)."""
    return mut[0] if mut and mut[1] == fam else None


def produce_AB(T, S, pos, mut):
    P, m = row_params(pos), active(mut, "AB")
    for ch in range(128):                                 # producer thread tid = ch (tid < 128)
        f = ex2(fsub(S["g2"][ch:ch + 1], S["gref"][ch:ch + 1]))
        ksv, qsv = fmul(S["k"][ch:ch + 1], f), fmul(S["q"][ch:ch + 1], f)
        ks, tt, half = ch >> 3, ch & 3, (ch >> 2) & 1
        if m == "shift_lane":
            tt = (tt + 1) & 3
        ii_k, ii_q = P["slot_hi"] + 2 * half, (1 - P["slot_hi"]) + 2 * half
        if m == "wrong_parity":
            ii_k, ii_q = ii_q, ii_k
        for val, ii in ((ksv, ii_k), (qsv, ii_q)):
            hi, lo = split(val)
            for hl, w in ((0, hi), (1, lo)):
                if m == "missing_word" and (ks, hl, tt, ii) == (3, 1, 2, ii_q):
                    continue
                T.write("AB", idx_ab(ks, tt, hl, ii), w[0], ch, EPOCH["AB_write"])
    if m == "double_writer":                              # thread 1 also writes a word of thread 0
        f = ex2(fsub(S["g2"][0:1], S["gref"][0:1]))
        hi, lo = split(fmul(S["k"][0:1], f))
        T.write("AB", idx_ab(0, 0, 0, P["slot_hi"]), hi[0], 1, EPOCH["AB_write"])


def produce_AM(T, S, pos, mut):
    P, m = row_params(pos), active(mut, "AM")
    if P["s"] == 0:
        return
    for w in range(8):                                    # warp w (guard: w <= pos >> 3 and w < 2 s: the tile has 6 k-steps)
        if not (w <= (pos >> 3) and w < 2 * P["s"]):
            continue
        for t in range(4):
            tid = 32 * w + 4 * (P["slot"] & 7) + t        # the mine lane of this t
            for e in range(2):
                c = 8 * w + 2 * t + e
                r = c & 7
                tt, half = r & 3, r >> 2
                if m == "shift_lane":
                    tt = (tt + 1) & 3
                ii_v, ii_z = P["slot_hi"] + 2 * half, (1 - P["slot_hi"]) + 2 * half
                if m == "wrong_parity":
                    ii_v, ii_z = ii_z, ii_v
                for which, arr in ((0, S["akk"]), (1, S["aqk"])):
                    hi, lo = split(arr[c:c + 1])
                    for hl, word in ((0, hi[0]), (1, lo[0])):
                        if not (m == "missing_word" and (w, which, hl, tt, ii_v) == (1, 0, 1, 2, ii_v)):
                            T.write("AM", idx_am(w, tt, which, hl, ii_v), word, tid, EPOCH["AM_write"])
                        if m != "stale_zero":
                            T.write("AM", idx_am(w, tt, which, hl, ii_z), np.uint32(0), tid, EPOCH["AM_write"])
    if m == "double_writer":
        hi, lo = split(S["akk"][0:1])
        T.write("AM", idx_am(0, 0, 0, 0, P["slot_hi"]), hi[0], 7, EPOCH["AM_write"])


def produce_AD(T, S, pos, mut):
    P, m = row_params(pos), active(mut, "AD")
    gq = gq_of(S)
    for tid in range(256):                                # which = tid >> 7, col = tid & 127
        which, col = tid >> 7, tid & 127
        ks, tt, half = col >> 3, col & 3, (col >> 2) & 1
        if m == "shift_lane":
            tt = (tt + 1) & 3
        if which == 1:
            val, ii = S["W"][col:col + 1], P["slot_hi"] + 2 * half
        else:
            val, ii = gq[col:col + 1], (1 - P["slot_hi"]) + 2 * half
        if m == "wrong_parity":
            ii ^= 1                                       # the other row parity: W / GQ swap rows
        hi, lo = split(val)
        for hl, word in ((0, hi[0]), (1, lo[0])):
            if m == "missing_word" and (ks, hl, tt, which) == (3, 1, 2, 1):
                continue
            T.write("AD", idx_ab(ks, tt, hl, ii), word, tid, EPOCH["AD_write"])
    if m == "double_writer":
        hi, lo = split(S["W"][0:1])
        T.write("AD", idx_ab(0, 0, 0, P["slot_hi"]), hi[0], 129, EPOCH["AD_write"])


def consume(T, fam, pos, ks_list, which_list, mut):
    """Words the lanes load: [lane][ks][which][hl][ii] (the tile for the mine lanes, the zero tile for the others); logs the reads of the tile with the epoch of the read."""
    P = row_params(pos)
    out = np.zeros((32, len(ks_list), len(which_list), 2, 4), np.uint32)
    ep = EPOCH[fam + "_read"] - (1 if active(mut, fam) == "missing_barrier" else 0)
    all_read_tile = active(mut, fam) == "nonmine_tile"
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        mine = g == (P["slot"] & 7)
        for ki, ks in enumerate(ks_list):
            for wi, which in enumerate(which_list):
                for hl in range(2):
                    for ii in range(4):
                        idx = idx_am(ks, t, which, hl, ii) if fam == "AM" else idx_ab(ks, t, hl, ii)
                        if mine or all_read_tile:
                            out[lane, ki, wi, hl, ii] = T.w[fam][idx]
                            T.reads.append((fam, idx, lane, ep))
                        else:
                            out[lane, ki, wi, hl, ii] = T.zero[idx]
    return out


# ---- the checks --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def consumed_mask(fam, pos, nks):
    """bool [lane][ks][ii]: the words of the A rows whose MMA output rows reach a result. The other rows of an A tile only produce MMA rows that the kernel discards (tile-neighbour
    independence, the contract's hardware premise), so a wrong word there cannot change an output. AB and AD consume both rows of the lanes that are ``mine`` (ks / qs, W / GQ: T 411-418 and 494-503
    store both); AM only row ``slot`` (the akk and the aqk row are two tiles with ONE row each, T 187); the non-mine lanes hold rows that are never read."""
    P = row_params(pos)
    mine = np.broadcast_to(P["mine"][:, None, None], (32, nks, 4))
    if fam == "AM":
        return mine & ((II[None, None, :] & 1) == P["slot_hi"])
    return mine.copy()


def compare(name, ref_hi, ref_lo, got, problems, pos, mask=None, stats=None, fam=None):
    """ref [lane][ks][ii] hi / lo, got [lane][ks][which=0][hl][ii]: the words as uint32 (a NaN or an infinity never passes by float equality). With ``stats`` the words that differ are counted per family:
    ``stats[fam] = [all differing words, those inside the consumed A rows (``mask``)]``."""
    first = None
    for hl, ref in ((0, ref_hi), (1, ref_lo)):
        g = got[:, :, 0, hl, :]
        diff = ref != g
        if stats is not None and fam is not None:
            s = stats.setdefault(fam, [0, 0])
            s[0] += int(np.count_nonzero(diff))
            if mask is not None:
                s[1] += int(np.count_nonzero(diff & mask))
        if first is None and diff.any():
            bad = np.argwhere(diff)[0]
            first = "%s pos %d: word differs at lane %d ks %d ii %d (hl %d): T %08x staged %08x" % (name, pos, bad[0], bad[1], bad[2], hl, ref[tuple(bad)], g[tuple(bad)])
    if first is not None:
        problems.append(first)


def check_writers(T, fam, expect_words, problems, pos):
    """Every word of the used part of the tile is written exactly once per row by exactly one thread (so a stale or poisoned word cannot be read)."""
    counts = {}
    for idx, tid, ep in T.log[fam]:
        counts.setdefault(idx, []).append(tid)
    for idx, ws in counts.items():
        if len(ws) != 1:
            problems.append("%s pos %d: word %d has %d writers %s" % (fam, pos, idx, len(ws), sorted(set(ws))[:3]))
            return
    missing = sorted(set(expect_words) - set(counts))
    if missing:
        problems.append("%s pos %d: %d word(s) never written in this row, first %d" % (fam, pos, len(missing), missing[0]))


def check_order(T, fam, problems, pos):
    """RAW: every read follows the write of its word by at least one barrier (epoch of the read > epoch of the write)."""
    written = {idx: ep for idx, tid, ep in T.log[fam]}
    for f, idx, lane, ep in T.reads:
        if f != fam:
            continue
        if idx not in written:
            problems.append("%s pos %d: word %d is read but not written in this row (stale)" % (fam, pos, idx))
            return
        if ep <= written[idx]:
            problems.append("%s pos %d: word %d is read in epoch %d, written in epoch %d: no barrier between (race)" % (fam, pos, idx, ep, written[idx]))
            return


def check_reuse():
    """WAR across rows: the next row writes a tile in an epoch after every read of this row (the epochs of a row are shifted by EPOCHS_PER_ROW)."""
    problems = []
    for fam in FAMILIES:
        if EPOCH[fam + "_write"] + EPOCHS_PER_ROW <= EPOCH[fam + "_read"]:
            problems.append("%s: the next row writes it before this row has read it" % fam)
    return problems


def alignment_problems():
    out = []
    for name, off in LAYOUT.items():
        if off % 16:
            out.append("%s offset %d is not 16-byte aligned" % (name, off))
    for name in FAMILIES:
        if LAYOUT[name] % 128:
            out.append("%s offset %d is not 128-byte aligned" % (name, LAYOUT[name]))
        if (ZERO_OFF - LAYOUT[name]) % 128 != 64:
            out.append("the zero tile is not 64 mod 128 bytes from %s" % name)
    if 4 * ZERO_WORDS < max(4 * AB_WORDS, 4 * AM_WORDS, 4 * AD_WORDS):
        out.append("the zero tile is smaller than the largest tile")
    if not (AB_OFF + 4 * AB_WORDS == AM_OFF and AM_OFF + 4 * AM_WORDS == AD_OFF and ARENA == (AB_OFF, AD_OFF + 4 * AD_WORDS)):
        out.append("the three tiles are not one contiguous overlay arena")
    if not (ZERO_OFF + 4 * ZERO_WORDS <= ARENA[0] and ARENA[1] <= FRAG_SIZE):
        out.append("the zero page overlaps the overlay arena, or the arena is not the suffix of the struct (a later tile could not grow the arena upward without covering the zero page)")
    for ks in range(16):
        for t in range(4):
            for hl in range(2):
                if (AB_OFF + 4 * idx_ab(ks, t, hl, 0)) % 16 or (AD_OFF + 4 * idx_ab(ks, t, hl, 0)) % 16 or (ZERO_OFF + 4 * idx_ab(ks, t, hl, 0)) % 16:
                    out.append("a uint4 of AB / AD / zero is misaligned at ks %d t %d hl %d" % (ks, t, hl))
    for ks in range(6):
        for t in range(4):
            for wh in range(2):
                for hl in range(2):
                    if (AM_OFF + 4 * idx_am(ks, t, wh, hl, 0)) % 16:
                        out.append("a uint4 of AM is misaligned at ks %d t %d" % (ks, t))
    return out


def lds128_wavefronts(addrs):
    """A 128-bit load is processed by quarter warps; a quarter takes as many wavefronts as the most loaded bank has distinct 16-byte words (4 banks each). Conflict-free = 4 in all."""
    total = 0
    for q in range(4):
        words = sorted(set(addrs[8 * q:8 * q + 8]))
        per_bank = {}
        for a in words:
            for b in range(4):
                per_bank.setdefault(((a // 4) + b) % 32, set()).add(a)
        total += max(len(v) for v in per_bank.values())
    return total


def sts32_wavefronts(addrs):
    per_bank = {}
    for a in set(addrs):
        per_bank.setdefault((a // 4) % 32, set()).add(a)
    return max(len(v) for v in per_bank.values())


def bank_stats():
    worst = {"AB_load": 0, "AM_load": 0, "AD_load": 0}
    for pos in range(64):
        P = row_params(pos)
        mine = P["mine"]
        for ks in range(16):
            for hl in range(2):
                for name, off in (("AB_load", AB_OFF), ("AD_load", AD_OFF)):
                    a = [(off if mine[l] else ZERO_OFF) + 4 * idx_ab(ks, int(T4[l]), hl, 0) for l in range(32)]
                    worst[name] = max(worst[name], lds128_wavefronts(a))
        for ks in range(2 * P["s"]):
            for wh in range(2):
                for hl in range(2):
                    a = [(AM_OFF if mine[l] else ZERO_OFF) + 4 * idx_am(ks, int(T4[l]), wh, hl, 0) for l in range(32)]
                    worst["AM_load"] = max(worst["AM_load"], lds128_wavefronts(a))
    stores = {"AB_store": 0, "AD_store": 0}
    for slot_hi in (0, 1):                                # one warp of producers (channels 32 w .. 32 w + 31), the four STS.32 of a thread
        for hl in range(2):
            for kind in range(2):
                a = []
                for l in range(32):
                    ks, tt, half = l >> 3, l & 3, (l >> 2) & 1
                    ii = (slot_hi if kind == 0 else 1 - slot_hi) + 2 * half
                    a.append(AB_OFF + 4 * idx_ab(ks, tt, hl, ii))
                stores["AB_store"] = max(stores["AB_store"], sts32_wavefronts(a))
                stores["AD_store"] = stores["AB_store"]
    am_store = 0                                          # AM producers: the four mine lanes (t = 0 .. 3) of one warp store the words of their two columns (e = 0, 1); the other lanes are idle
    for slot_hi in (0, 1):
        for w in range(6):
            for e in range(2):
                for which in range(2):
                    for hl in range(2):
                        for kind in range(2):
                            a = []
                            for t in range(4):
                                r = 2 * t + e
                                ii = (slot_hi if kind == 0 else 1 - slot_hi) + 2 * (r >> 2)
                                a.append(AM_OFF + 4 * idx_am(w, r & 3, which, hl, ii))
                            am_store = max(am_store, sts32_wavefronts(a))
    stores["AM_store"] = am_store
    worst.update(stores)
    return worst


# ---- one row, all checks ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
def expected_reads(fam, KS, all_lanes):
    return (32 if all_lanes else 4) * (KS if fam == "AM" else 16) * (2 if fam == "AM" else 1) * 2 * 4


def run_row(T, S, pos, mut=None, stats=None):
    problems = []
    T.begin_row()
    produce_AB(T, S, pos, mut)
    produce_AM(T, S, pos, mut)
    produce_AD(T, S, pos, mut)
    ab_got = consume(T, "AB", pos, list(range(16)), [0], mut)
    compare("AB", *t_AB(S, pos), ab_got, problems, pos, consumed_mask("AB", pos, 16), stats, "AB")
    ad_got = consume(T, "AD", pos, list(range(16)), [0], mut)
    compare("AD", *t_AD(S, pos), ad_got, problems, pos, consumed_mask("AD", pos, 16), stats, "AD")
    am_ref, KS = t_AM(S, pos)
    if am_ref is not None:
        am_got = consume(T, "AM", pos, list(range(KS)), [0, 1], mut)
        for which in (0, 1):
            compare("AM%d" % which, am_ref[which][0], am_ref[which][1], am_got[:, :, which:which + 1, :, :], problems, pos, consumed_mask("AM", pos, KS), stats, "AM")
    used = {"AB": [idx_ab(ks, t, hl, ii) for ks in range(16) for t in range(4) for hl in range(2) for ii in range(4)],
            "AD": [idx_ab(ks, t, hl, ii) for ks in range(16) for t in range(4) for hl in range(2) for ii in range(4)],
            "AM": [idx_am(ks, t, wh, hl, ii) for ks in range(KS) for t in range(4) for wh in range(2) for hl in range(2) for ii in range(4)]}
    for fam in FAMILIES:
        if fam == "AM" and am_ref is None:
            continue
        check_writers(T, fam, used[fam], problems, pos)
        check_order(T, fam, problems, pos)
        seen = sum(1 for r in T.reads if r[0] == fam)
        want = expected_reads(fam, KS, active(mut, fam) == "nonmine_tile")
        if seen != want and active(mut, fam) is None:
            problems.append("%s pos %d: the staged side observed %d tile reads, %d expected (the equality would compare the expression with itself)" % (fam, pos, seen, want))
    return problems


def run_sequence(poss, rng, special_every=2, mut=None, stats=None):
    """Rows at the positions ``poss`` in one CTA: persistent poisoned shared memory, a fresh random state per row (every other one with special values)."""
    T, problems = Smem(), []
    if mut and mut[0] == "stale_zero":
        T.w[mut[1]][:] = 0                                 # the zero words were stored once at kernel start (the poison would be seen at once, which proves less than a flip does)
    for i, pos in enumerate(poss):
        problems += run_row(T, make_state(rng, special=(i % special_every == 1)), pos, mut, stats)
    return problems


# mutation -> the families it applies to (stale_zero: only AM has zero-parity words that a producer can skip; AB / AD tiles are dense, a skipped word there is ``missing_word``)
MUTATIONS = {"shift_lane": FAMILIES, "wrong_parity": FAMILIES, "missing_word": FAMILIES, "double_writer": FAMILIES, "missing_barrier": FAMILIES, "stale_zero": ("AM",), "nonmine_tile": FAMILIES}
# the rows of a control: they cross a slot_hi flip inside a sub-block (7 -> 8) and, for AM, s = 1 and s = 3 rows; the wrap at the chunk end is in the positive sequence
CONTROL_ROWS = [21, 22, 23, 24, 25, 62, 63, 0, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--states", type=int, default=2, help="how many times the positive sequence of 80 rows is run (a new random state per row)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    say = (lambda *x: None) if a.quiet else print
    say("layout inside struct alignas(128) FragSmem (zero page first): zero tile %d (%d B), AB %d (%d B), AM %d (%d B), AD %d (%d B), end %d, sizeof(FragSmem) = %d B; overlay arena AB u AM u AD = [%d, %d) = %d B" % (
        ZERO_OFF, 4 * ZERO_WORDS, AB_OFF, 4 * AB_WORDS, AM_OFF, 4 * AM_WORDS, AD_OFF, 4 * AD_WORDS, END_OFF, FRAG_SIZE, ARENA[0], ARENA[1], ARENA[1] - ARENA[0]))
    say("RowSmem: T %d B, F v1 %d B (S.gq and S.x[1] deleted); static shared memory (worst-case placement: RowSmem, padding to 128, FragSmem, the int of rows_kernel<true>) = %d B" % (ROWSMEM_T, ROWSMEM_F, STATIC_F))
    blackwell, hopper = up(STATIC_F, 1024), up(STATIC_F, 128)
    say("static shared memory the compiler reports (allocation unit 1 KiB on Blackwell, 128 B on Hopper), predicted: %d B / %d B (T: 8192 / 7808); margin rule: %d - %d = %d < 1024: %s" % (
        blackwell, hopper, blackwell, STATIC_F, blackwell - STATIC_F, blackwell - STATIC_F < 1024))
    say("two CTAs per SM: 2 x (%d + 1024 reserve) = %d B <= 233472 (228 KiB): %s; base carve-out 65536 B (measured on B200 and B300 for T): %s" % (
        blackwell, 2 * (blackwell + 1024), 2 * (blackwell + 1024) <= 233472, "fits" if 2 * (blackwell + 1024) <= 65536 else "DOES NOT FIT"))
    fails = alignment_problems() + check_reuse()
    for _ in range(a.states):                              # pos 0..63 and the wrap at the chunk end (0..15 again): the rows of one CTA that appends 80 rows
        fails += run_sequence(list(range(64)) + list(range(16)), rng)
    say("positive checks: %d sequence(s) of 80 rows (pos 0..63, then the chunk end and 0..15 again): %s" % (a.states, "PASS (0 problems)" if not fails else "FAIL (%d problem(s)): %s" % (len(fails), fails[:3])))
    stats = bank_stats()
    extra = (stats["AB_store"] - 1) * 4 * 4 + (stats["AD_store"] - 1) * 8 * 2          # AB: 4 producer warps x 4 STS.32; AD: 8 warps x 2 STS.32 (hi, lo)
    say("shared-memory wavefronts: AB load %d, AM load %d, AD load %d (LDS.128 per warp, the minimum is 4: conflict-free); producer stores AB %d, AD %d (STS.32 per warp, the minimum is 1: a %d-way conflict "
        "because the 4 k-steps of a warp sit 128 B apart; %d extra shared-memory wavefronts per row and CTA in total: 4 warps x 4 STS (AB) + 8 warps x 2 STS (AD), once per row), AM %d (4 mine lanes: conflict-free = 1)" % (
            stats["AB_load"], stats["AM_load"], stats["AD_load"], stats["AB_store"], stats["AD_store"], max(stats["AB_store"], stats["AD_store"]), extra, stats["AM_store"]))
    rejected, total = 0, 0
    classes = {}                                           # (mutation, family) -> "visible" / "invisible" / "log-only"
    for mut, fams in MUTATIONS.items():
        line = []
        for fam in FAMILIES:
            if fam not in fams:
                line.append("%s n/a (no zero-parity words in a dense tile)" % fam)
                continue
            st = {}
            got = run_sequence(CONTROL_ROWS, np.random.default_rng(a.seed + 1), special_every=99, mut=(mut, fam), stats=st)
            total += 1
            rejected += bool(got)
            d_all, d_cons = st.get(fam, [0, 0])
            cls = "visible" if d_cons else ("invisible" if d_all else "log-only")
            classes[(mut, fam)] = cls
            line.append("%s %s [%s]" % (fam, "rejected" if got else "ACCEPTED (not seen)", {"visible": "words differ in the consumed A rows: output-visible", "invisible": "words differ ONLY in rows that no output reads: output-INVISIBLE",
                                                                                         "log-only": "no word differs: seen by the protocol checks only"}[cls]))
        say("  control %-16s %s" % (mut, "; ".join(line)))
    # the classification is itself a claim about the design: a control that the outputs cannot show needs a fragment-level oracle on the device; pin it so that a change of the design shows
    expect_invisible = {("nonmine_tile", f) for f in FAMILIES} | {("stale_zero", "AM")}
    observed_invisible = {k for k, v in classes.items() if v == "invisible"}
    class_ok = observed_invisible == expect_invisible
    say("output-INVISIBLE controls (need the fragment-level oracle on the device): %s%s" % (sorted(observed_invisible), "" if class_ok else "  != expected %s" % sorted(expect_invisible)))
    good = (not fails and rejected == total and class_ok and all(stats[k] == 4 for k in ("AB_load", "AM_load", "AD_load")) and stats["AB_store"] <= 4 and stats["AD_store"] <= 4 and stats["AM_store"] == 1)
    say("VERDICT: %s (positives %s, controls rejected %d of %d, load wavefronts %s, store wavefronts AB/AM/AD %s)" % (
        "PASS" if good else "FAIL", "pass" if not fails else "FAIL", rejected, total, (stats["AB_load"], stats["AM_load"], stats["AD_load"]), (stats["AB_store"], stats["AM_store"], stats["AD_store"])))
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
