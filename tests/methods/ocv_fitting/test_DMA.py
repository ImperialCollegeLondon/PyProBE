"""Tests for the DMA method."""
import numpy as np
import polars as pl
import pytest

from pyprobe.methods.ocv_fitting.DMA import DMA
from pyprobe.methods.ocv_fitting.Simple_OCV_fit import Simple_OCV_fit
from pyprobe.result import Result


@pytest.fixture
def bol_ne_limits_fixture():
    """Return the anode stoichiometry limits."""
    return np.array([0.02, 0.99])


@pytest.fixture
def bol_pe_limits_fixture():
    """Return the cathode stoichiometry limits."""
    return np.array([0.05, 0.9])


@pytest.fixture
def eol_ne_limits_fixture():
    """Return the anode stoichiometry limits."""
    return np.array([0.3, 0.99])


@pytest.fixture
def eol_pe_limits_fixture():
    """Return the cathode stoichiometry limits."""
    return np.array([0.05, 0.9])


@pytest.fixture
def bol_capacity_fixture(bol_ne_limits_fixture, bol_pe_limits_fixture):
    """Return the cell and electrode capacities."""
    cell_capacity = 5
    pe_capacity, ne_capacity, li_inventory = Simple_OCV_fit.calc_electrode_capacities(
        bol_pe_limits_fixture[0],
        bol_pe_limits_fixture[1],
        bol_ne_limits_fixture[0],
        bol_ne_limits_fixture[1],
        cell_capacity,
    )
    return [cell_capacity, pe_capacity, ne_capacity, li_inventory]


@pytest.fixture
def eol_capacity_fixture(eol_ne_limits_fixture, eol_pe_limits_fixture):
    """Return the cell and electrode capacities."""
    cell_capacity = 4.5
    pe_capacity, ne_capacity, li_inventory = Simple_OCV_fit.calc_electrode_capacities(
        eol_pe_limits_fixture[0],
        eol_pe_limits_fixture[1],
        eol_ne_limits_fixture[0],
        eol_ne_limits_fixture[1],
        cell_capacity,
    )
    return [cell_capacity, pe_capacity, ne_capacity, li_inventory]


@pytest.fixture
def bol_result_fixture(bol_capacity_fixture):
    """Return a Result instance."""
    return Result(
        _data=pl.LazyFrame(
            {
                "Cell Capacity": bol_capacity_fixture[0],
                "Cathode Capacity": bol_capacity_fixture[1],
                "Anode Capacity": bol_capacity_fixture[2],
                "Li Inventory": bol_capacity_fixture[3],
            }
        ),
        info={},
    )


@pytest.fixture
def eol_result_fixture(eol_capacity_fixture):
    """Return a Result instance."""
    return Result(
        _data=pl.LazyFrame(
            {
                "Cell Capacity": eol_capacity_fixture[0],
                "Cathode Capacity": eol_capacity_fixture[1],
                "Anode Capacity": eol_capacity_fixture[2],
                "Li Inventory": eol_capacity_fixture[3],
            }
        ),
        info={},
    )


@pytest.fixture
def dma_fixture(bol_result_fixture, eol_result_fixture):
    """Return a DMA instance."""
    return DMA([bol_result_fixture, eol_result_fixture])


def test_init(dma_fixture, bol_result_fixture, eol_result_fixture):
    """Test the __init__ method."""
    np.testing.assert_allclose(dma_fixture.cell_capacity, [[5], [4.5]])
    np.testing.assert_allclose(
        dma_fixture.pe_capacity,
        [
            bol_result_fixture.data["Cathode Capacity"].to_numpy(),
            eol_result_fixture.data["Cathode Capacity"].to_numpy(),
        ],
    )


def test_calculate_dma_parameters(
    bol_capacity_fixture, eol_capacity_fixture, dma_fixture
):
    """Test the calculate_dma_parameters method."""
    expected_SOH = eol_capacity_fixture[0] / bol_capacity_fixture[0]
    expected_LAM_pe = 1 - eol_capacity_fixture[1] / bol_capacity_fixture[1]
    expected_LAM_ne = 1 - eol_capacity_fixture[2] / bol_capacity_fixture[2]
    expected_LLI = (
        bol_capacity_fixture[3] - eol_capacity_fixture[3]
    ) / bol_capacity_fixture[3]

    result = dma_fixture.result
    assert result.data["SOH"].to_numpy()[1] == expected_SOH
    assert result.data["LAM_pe"].to_numpy()[1] == expected_LAM_pe
    assert result.data["LAM_ne"].to_numpy()[1] == expected_LAM_ne
    assert result.data["LLI"].to_numpy()[1] == expected_LLI
    assert result.data.columns == ["SOH", "LAM_pe", "LAM_ne", "LLI"]
