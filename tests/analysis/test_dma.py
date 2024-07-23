"""Tests for the degradation mode analysis module."""
import numpy as np
import polars as pl
import pytest

import pyprobe.analysis.base.degradation_mode_analysis_functions as dma_functions
import pyprobe.analysis.degradation_mode_analysis as dma
from pyprobe.result import Result


def graphite_LGM50_ocp_Chen2020(sto):
    """Chen2020 graphite ocp fit."""
    u_eq = (
        1.9793 * np.exp(-39.3631 * sto)
        + 0.2482
        - 0.0909 * np.tanh(29.8538 * (sto - 0.1234))
        - 0.04478 * np.tanh(14.9159 * (sto - 0.2769))
        - 0.0205 * np.tanh(30.4444 * (sto - 0.6103))
    )

    return u_eq


def nmc_LGM50_ocp_Chen2020(sto):
    """Chen2020 nmc ocp fit."""
    u_eq = (
        -0.8090 * sto
        + 4.4875
        - 0.0428 * np.tanh(18.5138 * (sto - 0.5542))
        - 17.7326 * np.tanh(15.7890 * (sto - 0.3117))
        + 17.5842 * np.tanh(15.9308 * (sto - 0.3120))
    )

    return u_eq


def test_fit_ocv():
    """Test the fit_ocv method."""
    n_points = 1000
    capacity = np.linspace(0, 1, n_points)
    x_real = [0.8, 0.1, 0.1, 0.7]
    x_pe_real = np.linspace(x_real[0], x_real[1], n_points)
    x_ne_real = np.linspace(x_real[2], x_real[3], n_points)

    # test charge
    voltage = nmc_LGM50_ocp_Chen2020(x_pe_real) - graphite_LGM50_ocp_Chen2020(x_ne_real)
    z = np.linspace(0, 1, n_points)
    ocp_pe = nmc_LGM50_ocp_Chen2020(z)
    ocp_ne = graphite_LGM50_ocp_Chen2020(z)

    x_guess = [0.8, 0.4, 0.2, 0.6]
    result = Result(
        pl.DataFrame({"Voltage [V]": voltage, "Capacity [Ah]": capacity}), {}
    )
    params, fit = dma.fit_ocv(
        result, x_ne=z, x_pe=z, ocp_ne=ocp_ne, ocp_pe=ocp_pe, x_guess=x_guess
    )
    assert isinstance(params, Result)
    assert params.data.columns == [
        "x_pe low SOC",
        "x_pe high SOC",
        "x_ne low SOC",
        "x_ne high SOC",
        "Cell Capacity [Ah]",
        "Cathode Capacity [Ah]",
        "Anode Capacity [Ah]",
        "Li Inventory [Ah]",
    ]

    assert isinstance(fit, Result)
    assert fit.data.columns == [
        "Capacity [Ah]",
        "SOC",
        "Input Voltage [V]",
        "Fitted Voltage [V]",
        "Input dSOCdV [1/V]",
        "Fitted dSOCdV [1/V]",
        "Input dVdSOC [V]",
        "Fitted dVdSOC [V]",
    ]

    param_values = list(params.data.row(0))
    np.testing.assert_allclose(
        np.array(param_values)[:4],
        np.array(x_real),
        rtol=1e-4,
    )

    # test discharge
    voltage = np.flip(voltage)
    capacity = -1 * capacity
    result = Result(
        pl.DataFrame({"Voltage [V]": voltage, "Capacity [Ah]": capacity}), {}
    )
    params, _ = dma.fit_ocv(
        result, x_ne=z, x_pe=z, ocp_ne=ocp_ne, ocp_pe=ocp_pe, x_guess=x_guess
    )

    param_values = list(params.data.row(0))
    np.testing.assert_allclose(
        np.array(param_values)[:4],
        np.array(x_real),
        rtol=1e-4,
    )


def test_fit_ocv_discharge():
    """Test the fit_ocv method for a discharge curve."""
    n_points = 1000
    capacity = np.linspace(1, 0, n_points)
    x_real = [0.8, 0.1, 0.1, 0.7]
    x_pe_real = np.linspace(x_real[1], x_real[0], n_points)
    x_ne_real = np.linspace(x_real[3], x_real[2], n_points)
    voltage = nmc_LGM50_ocp_Chen2020(x_pe_real) - graphite_LGM50_ocp_Chen2020(x_ne_real)

    z = np.linspace(0, 1, n_points)
    ocp_pe = nmc_LGM50_ocp_Chen2020(z)
    ocp_ne = graphite_LGM50_ocp_Chen2020(z)

    x_guess = [0.8, 0.4, 0.2, 0.6]
    result = Result(
        pl.DataFrame({"Voltage [V]": voltage, "Capacity [Ah]": capacity}), {}
    )
    params, _ = dma.fit_ocv(
        result, x_ne=z, x_pe=z, ocp_ne=ocp_ne, ocp_pe=ocp_pe, x_guess=x_guess
    )

    param_values = list(params.data.row(0))
    np.testing.assert_allclose(
        np.array(param_values)[:4],
        np.array(x_real),
        rtol=1e-4,
    )


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
    pe_capacity, ne_capacity, li_inventory = dma_functions.calc_electrode_capacities(
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
    pe_capacity, ne_capacity, li_inventory = dma_functions.calc_electrode_capacities(
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
                "Cell Capacity [Ah]": bol_capacity_fixture[0],
                "Cathode Capacity [Ah]": bol_capacity_fixture[1],
                "Anode Capacity [Ah]": bol_capacity_fixture[2],
                "Li Inventory [Ah]": bol_capacity_fixture[3],
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
                "Cell Capacity [Ah]": eol_capacity_fixture[0],
                "Cathode Capacity [Ah]": eol_capacity_fixture[1],
                "Anode Capacity [Ah]": eol_capacity_fixture[2],
                "Li Inventory [Ah]": eol_capacity_fixture[3],
            }
        ),
        info={},
    )


def test_calculate_dma_parameters(
    bol_capacity_fixture, eol_capacity_fixture, bol_result_fixture, eol_result_fixture
):
    """Test the calculate_dma_parameters method."""
    expected_SOH = eol_capacity_fixture[0] / bol_capacity_fixture[0]
    expected_LAM_pe = 1 - eol_capacity_fixture[1] / bol_capacity_fixture[1]
    expected_LAM_ne = 1 - eol_capacity_fixture[2] / bol_capacity_fixture[2]
    expected_LLI = (
        bol_capacity_fixture[3] - eol_capacity_fixture[3]
    ) / bol_capacity_fixture[3]

    result = dma.quantify_degradation_modes([bol_result_fixture, eol_result_fixture])
    assert result.data["SOH"].to_numpy()[1] == expected_SOH
    assert result.data["LAM_pe"].to_numpy()[1] == expected_LAM_pe
    assert result.data["LAM_ne"].to_numpy()[1] == expected_LAM_ne
    assert result.data["LLI"].to_numpy()[1] == expected_LLI
    assert result.data.columns == ["SOH", "LAM_pe", "LAM_ne", "LLI"]
