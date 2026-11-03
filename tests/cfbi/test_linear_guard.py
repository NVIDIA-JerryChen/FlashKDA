"""``bi_linear`` rests on the tile-slot independence of the bf16 ``m16n8k16`` MMA, probed on Blackwell only: it must
refuse to run on a device below compute capability 10 unless the user waives the check explicitly."""
import pytest
import torch

import flash_kda.cfbi.linear as linear_mod
from flash_kda.cfbi import bi_linear

from convlin_helpers import cx  # noqa: F401  (session fixture: builds / loads the conv-linear extension)


def _inputs():
    x = torch.randn(5, 64, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    return x, w


def test_bi_linear_refuses_pre_blackwell_unless_waived(cx, monkeypatch):
    x, w = _inputs()
    good = bi_linear(x, w)
    monkeypatch.setattr(linear_mod, "_BF16_GUARDED", {})
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    monkeypatch.delenv("FLASH_KDA_CFBI_ALLOW_UNVALIDATED_ARCH", raising=False)
    with pytest.raises(RuntimeError, match="validated on Blackwell"):
        bi_linear(x, w)
    monkeypatch.setenv("FLASH_KDA_CFBI_ALLOW_UNVALIDATED_ARCH", "1")
    with pytest.warns(RuntimeWarning, match="validated on Blackwell"):
        again = bi_linear(x, w)
    assert torch.equal(again.view(torch.int16), good.view(torch.int16))
