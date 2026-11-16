"""Persistent state of the CF-BI program (what a decode call continues from)."""
from dataclasses import dataclass

import torch

NPLANES = 5   # khat, G2, UT, W, U
PLANE_NAMES = ("khat", "G2", "UT", "W", "U")


@dataclass
class CfbiState:
    """Per (sequence, head): chunk-entry state ``E`` and the open-chunk row cache.

    E:    fp32 ``[N, H, V=128, K=128]``  state at the start of the open chunk (same orientation as
          ``flash_kda.fwd`` states)
    rows: fp32 ``[N, H, 5, 64, 128]``    planes (khat, G2, UT, W, U) of the rows computed so far in the
          open chunk; only rows ``< pos`` are meaningful
    pos:  int32 ``[N]``                  number of rows in the open chunk (0..63); the chunk grid is
          anchored at the sequence start
    """

    E: torch.Tensor
    rows: torch.Tensor
    pos: torch.Tensor

    @property
    def N(self) -> int:
        return self.E.shape[0]

    @property
    def H(self) -> int:
        return self.E.shape[1]

    def clone(self) -> "CfbiState":
        return CfbiState(self.E.clone(), self.rows.clone(), self.pos.clone())

    def select(self, seq_ids, heads=None) -> "CfbiState":
        """A copy restricted to some sequences (and optionally a head slice)."""
        hs = slice(None) if heads is None else heads
        return CfbiState(self.E[seq_ids][:, hs].clone(), self.rows[seq_ids][:, hs].clone(), self.pos[seq_ids].clone())


def alloc_state(N: int, H: int, device="cuda") -> CfbiState:
    """Zero state for ``N`` fresh sequences and ``H`` heads (K = V = 128)."""
    return CfbiState(
        E=torch.zeros(N, H, 128, 128, dtype=torch.float32, device=device),
        rows=torch.zeros(N, H, NPLANES, 64, 128, dtype=torch.float32, device=device),
        pos=torch.zeros(N, dtype=torch.int32, device=device),
    )


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int32)


def state_mismatch(a: CfbiState, b: CfbiState) -> dict:
    """Bitwise comparison of two states: counts of differing E elements, differing *valid* cache rows
    (rows >= pos are stale and not part of the state) and differing pos entries."""
    assert a.E.shape == b.E.shape
    out = {"E": int((_bits(a.E) != _bits(b.E)).sum()), "pos": int((a.pos != b.pos).sum())}
    n_bad_rows = 0
    if int((a.pos != b.pos).sum()) == 0:
        valid = torch.arange(64, device=a.rows.device)[None, :] < a.pos[:, None]   # [N, 64]
        diff = (_bits(a.rows) != _bits(b.rows)).any(dim=-1)                          # [N, H, 5, 64]
        n_bad_rows = int((diff & valid[:, None, None, :]).sum())
    out["rows"] = n_bad_rows
    return out


def state_equal(a: CfbiState, b: CfbiState) -> bool:
    return all(v == 0 for v in state_mismatch(a, b).values())
