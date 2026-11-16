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
        if os.environ.get("CFBI_REQUIRE_GPU", "0") == "1":     # set by the runs on allocated nodes: no device there is an error of the run, not a skipped test
            pytest.fail("CUDA is required here (CFBI_REQUIRE_GPU=1) and no CUDA device is available", pytrace=False)
        pytest.skip("CUDA required")
    from flash_kda.cfbi import load_ext

    return load_ext()


@pytest.fixture()
def gen():
    g = torch.Generator(device="cuda")
    g.manual_seed(20261102)
    return g


@pytest.fixture()
def forced(ext):
    """The hidden calibration control on: every decode call of the optimised row engine without impl bit 64 is ONE launch, whatever the device and the size of the grid. The tests that
    compare the single launch with the reference run through it, so they test the single launch on every device that runs the optimised row engine, and each asserts its launch set."""
    previous = ext.decode_single_force(True)
    try:
        yield
    finally:
        ext.decode_single_force(previous)
