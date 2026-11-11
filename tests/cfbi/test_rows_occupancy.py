"""The hidden occupancy binding of the row engine (``ext.rows_occupancy()``): what the occupancy API reports for the launch configuration in use of every instantiation of
``rows_kernel`` on the current device. It is not part of the documented API; the grid sizing of the decode tests (``helpers.decode_sequences``) reads it, so does the budget table and
the resource gate.

* every instantiation that a launch can use is reported (the optimised row engine and the frozen reference) with the threads, static and dynamic shared memory and registers of its
  launch, and the occupancy API's CTAs per SM;
* the figures obey the arithmetic limits of the device (registers, shared memory with the per-CTA reserve, threads);
* the optimised row engine keeps two CTAs per SM (``__launch_bounds__(256, 2)``) on the devices it runs on;
* the grid of the decode families is derived from the binding, never from a constant.
"""
import pytest
import torch

import helpers
from helpers import ROWS_OPTIMISED, ROWS_REFERENCE, rows_instance, sequences_beyond_waves


@pytest.fixture(scope="module")
def occupancy(ext):
    return ext.rows_occupancy()


def test_the_binding_reports_every_instantiation_of_the_row_engine(occupancy):
    assert [i["name"] for i in occupancy["instances"]] == [ROWS_OPTIMISED, ROWS_REFERENCE]
    for i in occupancy["instances"]:
        assert i["threads"] == 256 and i["dynamic_smem"] == 0, i
        assert i["static_smem"] > 0 and 0 < i["registers"] <= 255 and i["ctas_per_sm"] >= 1, i
    assert occupancy["device"]["sm_count"] == torch.cuda.get_device_properties(0).multi_processor_count


def test_the_figures_obey_the_arithmetic_limits_of_the_device(occupancy):
    dev = occupancy["device"]
    for i in occupancy["instances"]:
        assert i["ctas_per_sm"] <= dev["regs_per_sm"] // (i["registers"] * i["threads"]), i
        assert i["ctas_per_sm"] <= dev["smem_per_sm"] // (i["static_smem"] + i["dynamic_smem"] + dev["smem_reserved_per_block"]), i
        assert i["ctas_per_sm"] <= dev["max_threads_per_sm"] // i["threads"], i


def test_the_optimised_row_engine_keeps_two_ctas_per_sm_where_it_runs(ext):
    if ext.resolve_impl(0, -1) == ext.resolve_impl(3, -1):
        pytest.skip("the optimised kernels do not run on this device (_impl=0 resolves to the reference kernels)")
    inst, dev = rows_instance(ext)
    assert inst["registers"] <= 128, "__launch_bounds__(256, 2) allows 128 registers"
    assert inst["ctas_per_sm"] >= 2, inst


def test_the_binding_is_not_part_of_the_documented_api(ext):
    import flash_kda.cfbi as api

    assert hasattr(ext, "rows_occupancy")
    assert not hasattr(api, "rows_occupancy") and "rows_occupancy" not in api.__all__


def test_the_decode_grid_is_derived_from_the_device_not_from_a_constant(ext):
    heads = 12
    cap = helpers.resident_ctas(ext)
    n = helpers.decode_sequences(ext, heads)
    assert n * heads >= helpers.WAVES * cap and (n - 1) * heads < helpers.WAVES * cap
    inst, dev = rows_instance(ext)
    assert cap == dev["sm_count"] * inst["ctas_per_sm"]


@pytest.mark.parametrize("capacity,heads,expected", [(296, 12, 62), (264, 12, 55), (296, 1, 740), (296, 740, 1), (1, 1, 3), (100, 7, 36)])
def test_the_grid_arithmetic(capacity, heads, expected):
    """At least 2.5 waves, no more than one sequence beyond: 296 CTAs (148 SMs, two per SM) and 12 heads need 62 sequences (744 CTAs, 2.51 waves)."""
    assert sequences_beyond_waves(capacity, heads) == expected
    assert sequences_beyond_waves(capacity, heads) * heads >= 2.5 * capacity > (sequences_beyond_waves(capacity, heads) - 1) * heads
