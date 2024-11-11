"""Tests for the degradation mode analysis module."""
import math

import numpy as np
import polars as pl
import pytest
import sympy as sp
from pydantic import ValidationError
from scipy.interpolate import PPoly

import pyprobe.analysis.base.degradation_mode_analysis_functions as dma_functions
from pyprobe.analysis import smoothing
from pyprobe.analysis.degradation_mode_analysis import (
    DMA,
    OCP,
    BatchDMA,
    CompositeOCP,
    _get_gradient,
)
from pyprobe.result import Result

"""Tests for the OCP class in the degradation mode analysis module."""


@pytest.fixture
def stoichiometry_data():
    """Sample stoichiometry data."""
    return np.linspace(0, np.pi, 1000)


@pytest.fixture
def ocp_data():
    """Sample ocp data."""
    return np.sin(np.linspace(0, np.pi, 1000))


def test_set_from_data(stoichiometry_data, ocp_data):
    """Test the set_from_data method."""
    ocp = OCP.from_data(stoichiometry_data, ocp_data)

    pts = [0, 1, 2]
    assert isinstance(ocp.ocp_function, PPoly)
    np.testing.assert_allclose(
        ocp.ocp_function(pts),
        smoothing._LinearInterpolator(stoichiometry_data, ocp_data)(pts),
    )


def test_from_expression():
    """Test the set_from_expression method."""
    x = sp.symbols("x")
    expression = 2 * x**2 + 3 * x + 1
    ocp = OCP.from_expression(expression)
    assert ocp.ocp_function == expression


def test_eval():
    """Test the _get_eval method."""
    # Test with sympy expression
    x = sp.symbols("x")
    expression = 2 * x**2 + 3 * x + 1
    ocp = OCP(ocp_function=expression)
    x_pts = np.linspace(0, 10, 10)
    np.testing.assert_allclose(ocp.eval(x_pts), 2 * x_pts**2 + 3 * x_pts + 1)

    # Test with standard function
    def f_x(x):
        return 2 * x**2 + 3 * x + 1

    ocp = OCP(ocp_function=f_x)
    np.testing.assert_allclose(ocp.eval(x_pts), 2 * x_pts**2 + 3 * x_pts + 1)


def test_get_gradient_ppoly():
    """Test the _get_gradient method with a PPoly object."""
    # Create a PPoly object
    c = np.array([[1, 2], [3, 4]])  # Coefficients
    x = np.array([0, 1, 2])  # Breakpoints
    ppoly = PPoly(c, x)
    gradient = _get_gradient(ppoly)
    # Expected derivative
    expected_gradient = ppoly.derivative()
    # Test points
    test_x = np.linspace(0, 2, 50)
    np.testing.assert_allclose(gradient(test_x), expected_gradient(test_x))


def test_get_gradient_sympy_expr():
    """Test the _get_gradient method with a SymPy expression."""
    # Create a SymPy expression
    sto = sp.Symbol("x")
    expr = sto**3 + 2 * sto**2 + sto
    gradient = _get_gradient(expr)
    # Expected derivative
    expected_gradient = sp.lambdify(sto, sp.diff(expr, sto), "numpy")
    test_x = np.linspace(-10, 10, 100)
    np.testing.assert_allclose(gradient(test_x), expected_gradient(test_x))


def test_get_gradient_callable():
    """Test the _get_gradient method with a callable function."""
    test_x = np.linspace(0, 2 * np.pi, 100)

    # Define a callable function
    def any_function(x):
        return np.sin(x)

    gradient = _get_gradient(any_function)

    # Expected derivative
    def expected_gradient(x):
        return np.cos(x)

    np.testing.assert_allclose(gradient(test_x), expected_gradient(test_x), rtol=1e-3)


def test_get_gradient_invalid_input():
    """Test the _get_gradient method with an invalid input."""
    with pytest.raises(ValueError):
        _get_gradient(42)


def test_comp_from_data(stoichiometry_data):
    """Test the from_data method."""
    ocp_data = 2 * stoichiometry_data**2 + 3 * stoichiometry_data + 1
    ocp = CompositeOCP.from_data(
        stoichiometry_data, ocp_data, stoichiometry_data, ocp_data
    )
    assert isinstance(ocp.ocp_list, list)

    assert len(ocp.ocp_list) == 2
    pts = [0, 1, 2]
    for ocp_i in ocp.ocp_list:
        assert isinstance(ocp_i, PPoly)
        np.testing.assert_allclose(
            ocp_i(pts),
            smoothing._LinearInterpolator(ocp_data, stoichiometry_data)(pts),
        )


def test_comp_eval_and_grad():
    """Test the eval and grad methods."""
    n_points = 10

    # Create data arrays
    z = np.linspace(0, 1, n_points)
    x_c1, ocp_c1 = z, np.linspace(1.5, 0.05, n_points)
    x_c2, ocp_c2 = z, np.linspace(2.3, 0.15, n_points)

    ocp = CompositeOCP.from_data(x_c1, ocp_c1, x_c2, ocp_c2)

    np.testing.assert_allclose(ocp.eval(0.5), 0.95625)
    np.testing.assert_allclose(ocp.grad(0.5), -1.73194444)


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


@pytest.fixture
def ne_ocp_fixture():
    """Fixture for the negative electrode OCP."""
    ocp_ne = OCP(graphite_LGM50_ocp_Chen2020)
    return ocp_ne


@pytest.fixture
def pe_ocp_fixture():
    """Fixture for the positive electrode OCP."""
    ocp_pe = OCP(nmc_LGM50_ocp_Chen2020)
    return ocp_pe


def test_f_OCV(ne_ocp_fixture, pe_ocp_fixture):
    """Test the f_OCV method."""
    dma = DMA(
        input_data=Result(base_dataframe=pl.DataFrame({}), info={}),
        ocp_ne=ne_ocp_fixture,
        ocp_pe=pe_ocp_fixture,
    )
    x_pe_lo = 0.8
    x_pe_hi = 0.1
    x_ne_lo = 0.1
    x_ne_hi = 0.7
    soc = np.linspace(0, 1, 100)
    ocv = dma._f_OCV(
        soc,
        x_pe_lo,
        x_pe_hi,
        x_ne_lo,
        x_ne_hi,
    )

    x_pe = np.linspace(x_pe_lo, x_pe_hi, 100)
    x_ne = np.linspace(x_ne_lo, x_ne_hi, 100)
    ocv_expected = nmc_LGM50_ocp_Chen2020(x_pe) - graphite_LGM50_ocp_Chen2020(x_ne)
    np.testing.assert_allclose(ocv, ocv_expected)


def test_f_grad_OCV(ne_ocp_fixture, pe_ocp_fixture):
    """Test the f_grad_OCV method."""
    dma = DMA(
        input_data=Result(base_dataframe=pl.DataFrame({}), info={}),
        ocp_ne=ne_ocp_fixture,
        ocp_pe=pe_ocp_fixture,
    )
    soc = np.linspace(0, 1, 1000)
    x_pe_lo = 0.8
    x_pe_hi = 0.1
    x_ne_lo = 0.1
    x_ne_hi = 0.7
    d_ocv = dma._f_grad_OCV(soc, x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi)
    ocv_pts = dma._f_OCV(soc, x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi)
    numerical_d_ocv = np.gradient(ocv_pts, soc)
    np.testing.assert_allclose(d_ocv, numerical_d_ocv)


def test_run_ocv_curve_fit(ne_ocp_fixture, pe_ocp_fixture):
    """Test the run_ocv_curve_fit method."""
    x_pe_lo = 0.8
    x_pe_hi = 0.1
    x_ne_lo = 0.1
    x_ne_hi = 0.7
    x_pe = np.linspace(x_pe_lo, x_pe_hi, 10000)
    x_ne = np.linspace(x_ne_lo, x_ne_hi, 10000)
    ocv_pe = nmc_LGM50_ocp_Chen2020(x_pe)
    ocv_ne = graphite_LGM50_ocp_Chen2020(x_ne)
    soc = np.linspace(0, 1, 10000)
    ocv_target = ocv_pe - ocv_ne
    dma = DMA(
        input_data=Result(
            base_dataframe=pl.DataFrame(
                {"Voltage [V]": ocv_target, "Capacity [Ah]": soc}
            ),
            info={},
        ),
        ocp_ne=ne_ocp_fixture,
        ocp_pe=pe_ocp_fixture,
    )

    d_ocv_target = np.gradient(ocv_target, soc)
    limits, fit = dma.run_ocv_curve_fit(
        fitting_target="OCV",
        optimizer="differential_evolution",
        optimizer_options={
            "bounds": [
                (x_pe_lo - 0.05, x_pe_lo + 0.05),
                (x_pe_hi - 0.05, x_pe_hi + 0.05),
                (x_ne_lo - 0.05, x_ne_lo + 0.05),
                (x_ne_hi - 0.05, x_ne_hi + 0.05),
            ]
        },
    )
    assert isinstance(limits, Result)
    assert limits.data.columns == [
        "x_pe low SOC",
        "x_pe high SOC",
        "x_ne low SOC",
        "x_ne high SOC",
        "Cell Capacity [Ah]",
        "Cathode Capacity [Ah]",
        "Anode Capacity [Ah]",
        "Li Inventory [Ah]",
    ]
    np.testing.assert_allclose(limits.data["x_pe low SOC"].to_numpy()[0], x_pe_lo)
    np.testing.assert_allclose(limits.data["x_pe high SOC"].to_numpy()[0], x_pe_hi)
    np.testing.assert_allclose(limits.data["x_ne low SOC"].to_numpy()[0], x_ne_lo)
    np.testing.assert_allclose(limits.data["x_ne high SOC"].to_numpy()[0], x_ne_hi)

    np.testing.assert_allclose(fit.data["Fitted Voltage [V]"].to_numpy(), ocv_target)
    np.testing.assert_allclose(
        fit.data["Fitted dVdSOC [V]"].to_numpy(), d_ocv_target, rtol=0.005, atol=0.005
    )


def test_run_ocv_curve_fit_dQdV(ne_ocp_fixture, pe_ocp_fixture):
    """Test the run_ocv_curve_fit method with target dQdV."""
    x_pe_lo = 0.9
    x_pe_hi = 0.1
    x_ne_lo = 0.1
    x_ne_hi = 0.6
    x_pe = np.linspace(x_pe_lo, x_pe_hi, 10000)
    x_ne = np.linspace(x_ne_lo, x_ne_hi, 10000)
    ocv_pe = nmc_LGM50_ocp_Chen2020(x_pe)
    ocv_ne = graphite_LGM50_ocp_Chen2020(x_ne)
    soc = np.linspace(0, 1, 10000)
    ocv_target = ocv_pe - ocv_ne
    d_ocv_target = np.gradient(ocv_target, soc)
    dQdV_target = 1 / d_ocv_target

    dma = DMA(
        input_data=Result(
            base_dataframe=pl.DataFrame(
                {"Voltage [V]": ocv_target, "Capacity [Ah]": soc}
            ),
            info={},
        ),
        ocp_ne=ne_ocp_fixture,
        ocp_pe=pe_ocp_fixture,
    )

    limits, fit = dma.run_ocv_curve_fit(
        fitting_target="dQdV",
        optimizer="differential_evolution",
        optimizer_options={
            "bounds": [
                (x_pe_lo - 0.05, x_pe_lo + 0.05),
                (x_pe_hi - 0.05, x_pe_hi + 0.05),
                (x_ne_lo - 0.05, x_ne_lo + 0.05),
                (x_ne_hi - 0.05, x_ne_hi + 0.05),
            ]
        },
    )
    assert isinstance(limits, Result)
    assert limits.data.columns == [
        "x_pe low SOC",
        "x_pe high SOC",
        "x_ne low SOC",
        "x_ne high SOC",
        "Cell Capacity [Ah]",
        "Cathode Capacity [Ah]",
        "Anode Capacity [Ah]",
        "Li Inventory [Ah]",
    ]
    np.testing.assert_allclose(
        limits.data["x_pe low SOC"].to_numpy()[0], x_pe_lo, rtol=1e-5
    )
    np.testing.assert_allclose(
        limits.data["x_pe high SOC"].to_numpy()[0], x_pe_hi, rtol=1e-5
    )
    np.testing.assert_allclose(
        limits.data["x_ne low SOC"].to_numpy()[0], x_ne_lo, rtol=1e-5
    )
    np.testing.assert_allclose(
        limits.data["x_ne high SOC"].to_numpy()[0], x_ne_hi, rtol=1e-5
    )

    np.testing.assert_allclose(
        fit.data["Fitted Voltage [V]"].to_numpy(), ocv_target, rtol=1e-6
    )
    np.testing.assert_allclose(
        fit.data["Fitted dSOCdV [1/V]"].to_numpy(), dQdV_target, rtol=0.005, atol=0.005
    )


def test_run_ocv_curve_fit_dVdQ(ne_ocp_fixture, pe_ocp_fixture):
    """Test the run_ocv_curve_fit method with target dVdQ."""
    x_pe_lo = 0.8
    x_pe_hi = 0.1
    x_ne_lo = 0.1
    x_ne_hi = 0.7
    x_pe = np.linspace(x_pe_lo, x_pe_hi, 100000)
    x_ne = np.linspace(x_ne_lo, x_ne_hi, 100000)
    ocv_pe = nmc_LGM50_ocp_Chen2020(x_pe)
    ocv_ne = graphite_LGM50_ocp_Chen2020(x_ne)
    soc = np.linspace(0, 1, 100000)
    ocv_target = ocv_pe - ocv_ne
    d_ocv_target = np.gradient(ocv_target, soc)
    dVdQ_target = d_ocv_target

    dma = DMA(
        input_data=Result(
            base_dataframe=pl.DataFrame(
                {"Voltage [V]": ocv_target, "Capacity [Ah]": soc}
            ),
            info={},
        ),
        ocp_ne=ne_ocp_fixture,
        ocp_pe=pe_ocp_fixture,
    )

    limits, fit = dma.run_ocv_curve_fit(
        fitting_target="dVdQ",
        optimizer="differential_evolution",
        optimizer_options={
            "bounds": [
                (x_pe_lo - 0.05, x_pe_lo + 0.05),
                (x_pe_hi - 0.05, x_pe_hi + 0.05),
                (x_ne_lo - 0.05, x_ne_lo + 0.05),
                (x_ne_hi - 0.05, x_ne_hi + 0.05),
            ]
        },
    )
    assert isinstance(limits, Result)
    assert limits.data.columns == [
        "x_pe low SOC",
        "x_pe high SOC",
        "x_ne low SOC",
        "x_ne high SOC",
        "Cell Capacity [Ah]",
        "Cathode Capacity [Ah]",
        "Anode Capacity [Ah]",
        "Li Inventory [Ah]",
    ]
    np.testing.assert_allclose(limits.data["x_pe low SOC"].to_numpy()[0], x_pe_lo)
    np.testing.assert_allclose(limits.data["x_pe high SOC"].to_numpy()[0], x_pe_hi)
    np.testing.assert_allclose(limits.data["x_ne low SOC"].to_numpy()[0], x_ne_lo)
    np.testing.assert_allclose(limits.data["x_ne high SOC"].to_numpy()[0], x_ne_hi)

    np.testing.assert_allclose(fit.data["Fitted Voltage [V]"].to_numpy(), ocv_target)
    np.testing.assert_allclose(
        fit.data["Fitted dVdSOC [V]"].to_numpy(), dVdQ_target, rtol=0.005, atol=0.005
    )


def get_sample_ocv_data(sto_limits, n_points=1000):
    """Get sample OCV data."""
    x_pe_lo = sto_limits[0]
    x_pe_hi = sto_limits[1]
    x_ne_lo = sto_limits[2]
    x_ne_hi = sto_limits[3]
    x_pe = np.linspace(x_pe_lo, x_pe_hi, n_points)
    x_ne = np.linspace(x_ne_lo, x_ne_hi, n_points)
    ocv_pe = nmc_LGM50_ocp_Chen2020(x_pe)
    ocv_ne = graphite_LGM50_ocp_Chen2020(x_ne)
    return ocv_pe - ocv_ne


def test_run_batch_dma():
    """Test the run_batch_dma_parallel method."""
    soc = np.linspace(0, 1, 1000)

    ocv_target_list = [
        get_sample_ocv_data([0.83, 0.1, 0.1, 0.73]),
        get_sample_ocv_data([0.7, 0.2, 0.1, 0.70]),
        get_sample_ocv_data([0.6, 0.3, 0.3, 0.65]),
    ]
    input_data_list = [
        Result(
            base_dataframe=pl.DataFrame(
                {"Voltage [V]": ocv_target, "Capacity [Ah]": soc}
            ),
            info={},
            column_definitions={"Voltage [V]": "OCV", "Capacity [Ah]": "SOC"},
        )
        for ocv_target in ocv_target_list
    ]
    dma = BatchDMA(
        input_data=input_data_list,
    )
    x_pe = np.linspace(0, 1, 1000)
    x_ne = np.linspace(0, 1, 1000)
    ocv_pe = nmc_LGM50_ocp_Chen2020(x_pe)
    ocv_ne = graphite_LGM50_ocp_Chen2020(x_ne)
    dma.add_ocp_from_data(x_pe, ocv_pe, electrode="pe")
    dma.add_ocp_from_data(x_ne, ocv_ne, electrode="ne")
    dma_result, fitted_ocvs = dma.run_batch_dma_parallel(
        fitting_target="OCV",
        optimizer="differential_evolution",
        optimizer_options={
            "bounds": [(0.6, 0.85), (0.05, 0.4), (0.05, 0.4), (0.6, 0.75)]
        },
    )
    np.testing.assert_allclose(dma_result.data["Index"], [0, 1, 2])
    np.testing.assert_allclose(
        dma_result.data["x_pe low SOC"], [0.83, 0.7, 0.6], rtol=1e-6
    )
    np.testing.assert_allclose(
        dma_result.data["x_pe high SOC"], [0.1, 0.2, 0.3], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        dma_result.data["x_ne low SOC"], [0.1, 0.1, 0.3], rtol=1e-5, atol=5e-5
    )
    np.testing.assert_allclose(
        dma_result.data["x_ne high SOC"], [0.73, 0.7, 0.65], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(dma_result.data["Cell Capacity [Ah]"], [1, 1, 1])
    np.testing.assert_allclose(
        dma_result.data["Cathode Capacity [Ah]"], [1.3699, 2, 3.333], rtol=5e-4
    )
    np.testing.assert_allclose(
        dma_result.data["LAM_pe"], [0, -0.45996, -1.43327], rtol=5e-4
    )

    np.testing.assert_allclose(
        fitted_ocvs[0].data["Input Voltage [V]"], ocv_target_list[0]
    )
    np.testing.assert_allclose(
        fitted_ocvs[1].data["Input Voltage [V]"], ocv_target_list[1]
    )
    np.testing.assert_allclose(
        fitted_ocvs[2].data["Input Voltage [V]"], ocv_target_list[2]
    )


n_points = 1000
z = np.linspace(0, 1, 1000)


def test_fit_ocv():
    """Test the fit_ocv method."""
    capacity = np.linspace(0, 1, n_points)
    x_real = [0.8, 0.1, 0.1, 0.7]
    x_pe_real = np.linspace(x_real[0], x_real[1], n_points)
    x_ne_real = np.linspace(x_real[2], x_real[3], n_points)

    # test charge
    voltage = nmc_LGM50_ocp_Chen2020(x_pe_real) - graphite_LGM50_ocp_Chen2020(x_ne_real)

    result = Result(
        base_dataframe=pl.DataFrame(
            {"Voltage [V]": voltage, "Capacity [Ah]": capacity}
        ),
        info={},
    )
    dma = DMA(input_data=result)
    x_guess = [0.8, 0.4, 0.2, 0.6]
    params, fit = dma.fit_ocv(
        x_ne=z,
        x_pe=z,
        ocp_ne=graphite_LGM50_ocp_Chen2020(z),
        ocp_pe=nmc_LGM50_ocp_Chen2020(z),
        x_guess=x_guess,
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
        base_dataframe=pl.DataFrame(
            {"Voltage [V]": voltage, "Capacity [Ah]": capacity}
        ),
        info={},
    )
    dma = DMA(input_data=result)
    params, _ = dma.fit_ocv(
        x_ne=z,
        x_pe=z,
        ocp_ne=graphite_LGM50_ocp_Chen2020(z),
        ocp_pe=nmc_LGM50_ocp_Chen2020(z),
        x_guess=x_guess,
    )

    param_values = list(params.data.row(0))
    np.testing.assert_allclose(
        np.array(param_values)[:4],
        np.array(x_real),
        rtol=1e-4,
    )

    result = Result(
        base_dataframe=pl.DataFrame({"Voltage [V]": voltage, "Time [s]": capacity}),
        info={},
    )
    dma = DMA(input_data=result)
    with pytest.raises(ValidationError):
        dma.fit_ocv(
            x_ne=z,
            x_pe=z,
            ocp_ne=graphite_LGM50_ocp_Chen2020(z),
            ocp_pe=nmc_LGM50_ocp_Chen2020(z),
            x_guess=x_guess,
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
        base_dataframe=pl.DataFrame(
            {"Voltage [V]": voltage, "Capacity [Ah]": capacity}
        ),
        info={},
    )
    dma = DMA(input_data=result)
    params, _ = dma.fit_ocv(
        x_ne=z, x_pe=z, ocp_ne=ocp_ne, ocp_pe=ocp_pe, x_guess=x_guess
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
    voltage = np.linspace(0, 1, n_points)
    capacity = np.linspace(0, 1, n_points)
    result = Result(
        base_dataframe=pl.DataFrame(
            {"Voltage [V]": voltage, "Capacity [Ah]": capacity}
        ),
        info={},
    )
    dma = DMA(input_data=result)
    dma.stoichiometry_limits = Result(
        base_dataframe=pl.LazyFrame(
            {
                "Cell Capacity [Ah]": bol_capacity_fixture[0],
                "Cathode Capacity [Ah]": bol_capacity_fixture[1],
                "Anode Capacity [Ah]": bol_capacity_fixture[2],
                "Li Inventory [Ah]": bol_capacity_fixture[3],
            }
        ),
        info={},
    )
    return dma


@pytest.fixture
def eol_result_fixture(eol_capacity_fixture):
    """Return a Result instance."""
    voltage = np.linspace(0, 1, n_points)
    capacity = np.linspace(0, 1, n_points)
    result = Result(
        base_dataframe=pl.DataFrame(
            {"Voltage [V]": voltage, "Capacity [Ah]": capacity}
        ),
        info={},
    )
    dma = DMA(input_data=result)
    dma.stoichiometry_limits = Result(
        base_dataframe=pl.LazyFrame(
            {
                "Cell Capacity [Ah]": eol_capacity_fixture[0],
                "Cathode Capacity [Ah]": eol_capacity_fixture[1],
                "Anode Capacity [Ah]": eol_capacity_fixture[2],
                "Li Inventory [Ah]": eol_capacity_fixture[3],
            }
        ),
        info={},
    )
    return dma


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

    result = eol_result_fixture.quantify_degradation_modes(
        bol_result_fixture.stoichiometry_limits
    )
    assert result.data["SOH"].to_numpy()[1] == expected_SOH
    assert result.data["LAM_pe"].to_numpy()[1] == expected_LAM_pe
    assert result.data["LAM_ne"].to_numpy()[1] == expected_LAM_ne
    assert result.data["LLI"].to_numpy()[1] == expected_LLI
    assert result.data.columns == ["SOH", "LAM_pe", "LAM_ne", "LLI"]

    # test with missing or incorrect input data
    result = Result(
        base_dataframe=pl.DataFrame(
            {
                "Voltage [V]": np.linspace(0, 1, 10),
                "Capacity [Ah]": np.linspace(0, 1, 10),
            }
        ),
        info={},
    )


def test_average_ocvs(BreakinCycles_fixture):
    """Test the average_ocvs method."""
    break_in = BreakinCycles_fixture.cycle(0)
    break_in.set_SOC()
    dma = DMA.average_ocvs(input_data=break_in, charge_filter="constant_current(1)")
    assert math.isclose(dma.input_data.get_only("Voltage [V]")[0], 3.14476284763849)
    assert math.isclose(dma.input_data.get_only("Voltage [V]")[-1], 4.170649780122139)
    np.testing.assert_allclose(
        dma.input_data.get_only("SOC"), break_in.constant_current(1).get_only("SOC")
    )
    # test invalid input
    with pytest.raises(ValueError):
        DMA.average_ocvs(input_data=break_in.charge(0))


def test_calc_full_cell_ocv_composite():
    """Test the composite_full_cell_ocv method."""
    # Sample data
    n_points = 10
    params = [0.05, 0.01, 0.95, 0.9, 0.85]

    # Create data arrays
    z = np.linspace(0, 1, n_points)
    x_c1, ocp_c1 = z, np.linspace(1.5, 0.05, n_points)
    x_c2, ocp_c2 = z, np.linspace(2.3, 0.15, n_points)
    x_pe, ocp_pe = z, np.linspace(4, 2, n_points)
    cell_SOC = np.linspace(0, 1, n_points)

    # Run the function
    soc, y_pred = dma_functions.calc_full_cell_OCV_composite(
        SOC=cell_SOC,
        z_pe_lo=params[0],
        z_pe_hi=params[2],
        z_ne_lo=params[1],
        z_ne_hi=params[3],
        x_pe=x_pe,
        ocp_pe=ocp_pe,
        x_c1=x_c1,
        ocp_c1=ocp_c1,
        x_c2=x_c2,
        ocp_c2=ocp_c2,
        comp1_frac=params[4],
    )

    # Expected outcomes
    expected_soc = np.linspace(0, 1, n_points)
    expected_y_pred = np.array(
        [
            2.4,
            2.28091008,
            2.23166123,
            2.18241239,
            2.13316354,
            2.0839147,
            2.03466585,
            1.98541701,
            1.93616816,
            1.88691932,
        ]
    )

    # Assertions
    np.testing.assert_array_almost_equal(soc, expected_soc, decimal=8)
    np.testing.assert_array_almost_equal(y_pred, expected_y_pred, decimal=8)


def test_downsample_ocv():
    """Test the downsample_ocv method."""
    times = np.linspace(0, 100, 101)
    values = times
    min_distance = 10
    df = pl.DataFrame({"Time [s]": times, "Voltage [V]": values})
    dma = DMA(input_data=Result(base_dataframe=df, info={}))
    downsampled = dma.downsample_ocv(min_distance)
    assert isinstance(downsampled, Result)
    np.testing.assert_array_equal(
        dma.input_data.data["Voltage [V]"].to_numpy(),
        np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
    )
