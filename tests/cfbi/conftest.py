import os
import sys
from pathlib import Path

import pytest
import torch

# tests/ (for torch_ref etc.) and the repo root (for flash_kda) on sys.path
_TESTS = Path(__file__).resolve().parents[1]
for p in (str(_TESTS), str(_TESTS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="session")
def ext():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from flash_kda.cfbi import load_ext

    return load_ext()


@pytest.fixture()
def gen():
    g = torch.Generator(device="cuda")
    g.manual_seed(20261102)
    return g
