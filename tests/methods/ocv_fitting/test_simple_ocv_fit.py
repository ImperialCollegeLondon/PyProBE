"""Tests for the Simple_OCV_fit module."""
import numpy as np
import polars as pl
import pytest

from pyprobe.methods.ocv_fitting.Simple_OCV_fit import Simple_OCV_fit
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
    voltage = nmc_LGM50_ocp_Chen2020(x_pe_real) - graphite_LGM50_ocp_Chen2020(x_ne_real)

    z = np.linspace(0, 1, n_points)
    ocp_pe = nmc_LGM50_ocp_Chen2020(z)
    ocp_ne = graphite_LGM50_ocp_Chen2020(z)

    x_guess = [0.8, 0.4, 0.2, 0.6]
    x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi, _, _, _, _ = Simple_OCV_fit.fit_ocv(
        capacity, voltage, x_ne=z, ocp_ne=ocp_ne, x_pe=z, ocp_pe=ocp_pe, x_guess=x_guess
    )

    np.testing.assert_allclose(
        np.concatenate((x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi)),
        np.array(x_real),
        rtol=1e-4,
    )


@pytest.fixture
def ocp_ne_fixture():
    """Return the anode OCP data."""
    z = np.linspace(0, 1, 1000)
    return graphite_LGM50_ocp_Chen2020(z)


@pytest.fixture
def ocp_pe_fixture():
    """Return the cathode OCP data."""
    z = np.linspace(0, 1, 1000)
    return nmc_LGM50_ocp_Chen2020(z)


@pytest.fixture
def full_cell_voltage_fixture():
    """Return the full cell voltage data."""
    x_real = [0.9, 0.2, 0.1, 0.7]
    n_points = 1000
    x_pe_real = np.linspace(x_real[0], x_real[1], n_points)
    x_ne_real = np.linspace(x_real[2], x_real[3], n_points)
    voltage = nmc_LGM50_ocp_Chen2020(x_pe_real) - graphite_LGM50_ocp_Chen2020(x_ne_real)
    return voltage


@pytest.fixture
def OCP_result_fixture():
    """Return a Result instance."""
    x_real = [0.9, 0.2, 0.1, 0.7]
    n_points = 1000
    x_pe_real = np.linspace(x_real[0], x_real[1], n_points)
    x_ne_real = np.linspace(x_real[2], x_real[3], n_points)
    voltage = nmc_LGM50_ocp_Chen2020(x_pe_real) - graphite_LGM50_ocp_Chen2020(x_ne_real)
    z = np.linspace(0, 1, n_points)
    _data = pl.DataFrame({"Voltage [V]": voltage, "Capacity [Ah]": z})
    return Result(_data=_data, info={})


def test_init(
    OCP_result_fixture, ocp_ne_fixture, ocp_pe_fixture, full_cell_voltage_fixture
):
    """Test the __init__ method."""
    result = OCP_result_fixture
    parameters = {
        "Anode Stoichiometry": np.linspace(0, 1, 1000),
        "Cathode Stoichiometry": np.linspace(0, 1, 1000),
        "Anode OCP [V]": ocp_ne_fixture,
        "Cathode OCP [V]": ocp_pe_fixture,
        "Initial Guess": [0.9, 0.1, 0.1, 0.9],
    }
    method = Simple_OCV_fit(result, parameters)

    np.testing.assert_allclose(method.voltage, full_cell_voltage_fixture)
    np.testing.assert_allclose(method.capacity, np.linspace(0, 1, 1000))
    np.testing.assert_allclose(method.x_ne, np.linspace(0, 1, 1000))
    np.testing.assert_allclose(method.x_pe, np.linspace(0, 1, 1000))
    np.testing.assert_allclose(method.ocp_ne, ocp_ne_fixture)
    np.testing.assert_allclose(method.ocp_pe, ocp_pe_fixture)
    np.testing.assert_allclose(method.x_guess, [0.9, 0.1, 0.1, 0.9])
    assert method.variable_list == ["Voltage [V]", "Capacity [Ah]"]
    assert method.variable_list == ["Voltage [V]", "Capacity [Ah]"]
    assert method.parameter_list == [
        "Anode Stoichiometry",
        "Cathode Stoichiometry",
        "Anode OCP [V]",
        "Cathode OCP [V]",
        "Initial Guess",
    ]


def test_result(OCP_result_fixture, ocp_ne_fixture, ocp_pe_fixture):
    """Test the result method."""
    parameters = {
        "Anode Stoichiometry": np.linspace(0, 1, 1000),
        "Cathode Stoichiometry": np.linspace(0, 1, 1000),
        "Anode OCP [V]": ocp_ne_fixture,
        "Cathode OCP [V]": ocp_pe_fixture,
        "Initial Guess": [0.8, 0.4, 0.2, 0.6],
    }
    method = Simple_OCV_fit(OCP_result_fixture, parameters)
    assert isinstance(method.stoichiometry_limits, Result)
    assert method.stoichiometry_limits.data.columns == [
        "x_pe low SOC",
        "x_pe high SOC",
        "x_ne low SOC",
        "x_ne high SOC",
        "Cell Capacity",
        "Cathode Capacity",
        "Anode Capacity",
        "Li Inventory",
    ]
