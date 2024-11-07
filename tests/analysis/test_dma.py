"""Tests for the degradation mode analysis module."""
import math

import numpy as np
import polars as pl
import pytest
import sympy as sp
from pydantic import ValidationError

import pyprobe.analysis.base.degradation_mode_analysis_functions as dma_functions
from pyprobe.analysis import smoothing
from pyprobe.analysis.degradation_mode_analysis import DMA, BatchDMA
from pyprobe.result import Result


@pytest.fixture
def stoichiometry_data():
    """Sample stoichiometry data."""
    return np.linspace(0, np.pi, 1000)


@pytest.fixture
def ocp_data():
    """Sample ocp data."""
    return np.sin(np.linspace(0, np.pi, 1000))


def test_set_ocp_from_data_pe(stoichiometry_data, ocp_data):
    """Test the set_ocp_from_data method."""
    dma = DMA(input_data=Result(base_dataframe=pl.DataFrame({}), info={}))
    dma.set_ocp_from_data(
        stoichiometry_data, ocp_data, electrode="pe", interpolation_method="cubic"
    )
    assert dma.ocp_pe[0] is not None
    assert callable(dma.ocp_pe[0])
    assert np.isclose(dma.ocp_pe[0](0.4), np.sin(0.4))


def test_set_ocp_from_data_ne(stoichiometry_data, ocp_data):
    """Test the set_ocp_from_data method."""
    dma = DMA(input_data=Result(base_dataframe=pl.DataFrame({}), info={}))
    dma.set_ocp_from_data(stoichiometry_data, ocp_data, electrode="ne")
    assert dma.ocp_ne[0] is not None
    assert callable(dma.ocp_ne[0])
    assert np.isclose(dma.ocp_ne[0](0.1), np.sin(0.1))


def test_set_ocp_from_data_linear_interpolation(stoichiometry_data, ocp_data):
    """Test the set_ocp_from_data method with linear interpolation."""
    dma = DMA(input_data=Result(base_dataframe=pl.DataFrame({}), info={}))
    dma.set_ocp_from_data(
        stoichiometry_data, ocp_data, electrode="pe", interpolation_method="linear"
    )
    assert dma.ocp_pe[0] is not None
    assert callable(dma.ocp_pe[0])
    assert np.isclose(dma.ocp_pe[0](0.4), np.sin(0.4))


def test_set_ocp_from_data_cubic_interpolation(stoichiometry_data, ocp_data):
    """Test the set_ocp_from_data method with cubic interpolation."""
    dma = DMA(input_data=Result(base_dataframe=pl.DataFrame({}), info={}))
    dma.set_ocp_from_data(
        stoichiometry_data, ocp_data, electrode="pe", interpolation_method="cubic"
    )
    assert dma.ocp_pe[0] is not None
    assert callable(dma.ocp_pe[0])
    assert np.isclose(dma.ocp_pe[0](0.4), np.sin(0.4))


def test_set_ocp_from_data_multiple_components(stoichiometry_data, ocp_data):
    """Test the set_ocp_from_data method with multiple components."""
    dma = DMA(input_data=Result(base_dataframe=pl.DataFrame({}), info={}))
    dma.set_ocp_from_data(
        stoichiometry_data,
        ocp_data,
        electrode="pe",
        component_index=0,
        total_electrode_components=2,
    )
    assert dma.ocp_pe[0] is not None
    assert callable(dma.ocp_pe[0])
    assert len(dma.ocp_pe) == 2
    assert np.isclose(dma.ocp_pe[0](0.4), np.sin(0.4))
    assert dma.ocp_pe[1] is None
    dma.set_ocp_from_data(
        stoichiometry_data,
        ocp_data,
        electrode="pe",
        component_index=1,
        total_electrode_components=2,
    )
    assert np.isclose(dma.ocp_pe[1](0.8), np.sin(0.8))


def test_set_ocp_from_expression():
    """Test the set_ocp_from_expression method."""
    dma = DMA(input_data=Result(base_dataframe=pl.DataFrame({}), info={}))
    x = sp.symbols("x")
    expression = 2 * x**2 + 3 * x + 1
    dma.set_ocp_from_expression(expression, electrode="pe")
    assert dma._ocp_pe[0] == expression
    assert dma.ocp_pe[0] is not None
    assert callable(dma.ocp_pe[0])
    assert np.isclose(dma.ocp_pe[0](0.4), 2 * 0.4**2 + 3 * 0.4 + 1)


def test_ocp_derivative_ppoly():
    """Test _ocp_derivative with a PPoly object."""
    x = np.array([0, 1, 2, 3])
    y = np.array([0, 1, 0, -1])
    ppoly_ocp = smoothing.linear_interpolator(x, y)
    dma = DMA(input_data=Result(base_dataframe=pl.DataFrame({}), info={}))
    derivative = dma._ocp_derivative([ppoly_ocp])[0]
    assert callable(derivative)
    x = np.array([0, 1, 2, 3])
    np.testing.assert_allclose(derivative(x), np.array([1, -1, -1, -1]))


def test_ocp_derivative_sympy():
    """Test _ocp_derivative with a sympy expression."""
    dma = DMA(input_data=Result(base_dataframe=pl.DataFrame({}), info={}))
    x = sp.symbols("x")
    sympy_ocp = 2 * x**2 + 3 * x + 1
    derivative = dma._ocp_derivative([sympy_ocp])[0]
    assert callable(derivative)
    x = np.array([0, 1, 2, 3])
    np.testing.assert_allclose(derivative(x), np.array([3, 7, 11, 15]))


def test_ocp_derivative_invalid():
    """Test _ocp_derivative with an invalid input."""
    dma = DMA(input_data=Result(base_dataframe=pl.DataFrame({}), info={}))
    with pytest.raises(ValueError, match="OCP is not in a differentiable format."):
        dma._ocp_derivative("invalid_ocp")


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


def test_f_OCV():
    """Test the f_OCV method."""
    dma = DMA(input_data=Result(base_dataframe=pl.DataFrame({}), info={}))
    dma.ocp_ne = [graphite_LGM50_ocp_Chen2020]
    dma.ocp_pe = [nmc_LGM50_ocp_Chen2020]
    x_pe_lo = 0.8
    x_pe_hi = 0.1
    x_ne_lo = 0.1
    x_ne_hi = 0.7
    soc = np.linspace(0, 1, 100)
    ocv = dma._f_OCV(soc, x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi)

    x_pe = np.linspace(x_pe_lo, x_pe_hi, 100)
    x_ne = np.linspace(x_ne_lo, x_ne_hi, 100)
    ocv_expected = nmc_LGM50_ocp_Chen2020(x_pe) - graphite_LGM50_ocp_Chen2020(x_ne)
    np.testing.assert_allclose(ocv, ocv_expected)


def test_f_grad_OCV():
    """Test the f_grad_OCV method."""
    dma = DMA(input_data=Result(base_dataframe=pl.DataFrame({}), info={}))
    x_pts = np.linspace(0, 1, 100)
    ocp_pe_pts = 2 * x_pts**2
    ocp_pe = smoothing.cubic_interpolator(x=x_pts, y=ocp_pe_pts)
    ocp_ne_pts = 3 * x_pts**3
    ocp_ne = smoothing.cubic_interpolator(x=x_pts, y=ocp_ne_pts)
    dma.ocp_ne = [ocp_ne]
    dma.ocp_pe = [ocp_pe]
    x_pe_lo = 0
    x_pe_hi = 1
    x_ne_lo = 0
    x_ne_hi = 1
    d_ocv = dma._f_grad_OCV(x_pts, x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi)
    x_pe = np.linspace(x_pe_lo, x_pe_hi, 100)
    x_ne = np.linspace(x_ne_lo, x_ne_hi, 100)
    d_ocv_expected = 4 * x_pe - 9 * x_ne**2
    np.testing.assert_allclose(d_ocv, d_ocv_expected, atol=1e-12)

    x_pe_lo = 0.8
    x_pe_hi = 0.1
    x_ne_lo = 0.1
    x_ne_hi = 0.7
    d_ocv = dma._f_grad_OCV(x_pts, x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi)
    ocv_pts = dma._f_OCV(x_pts, x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi)
    numerical_d_ocv = np.gradient(ocv_pts, x_pts)
    np.testing.assert_allclose(d_ocv, numerical_d_ocv, rtol=1e-3, atol=0.02)


def test_curve_fit_ocv():
    """Test the curve_fit_ocv method."""
    dma = DMA(input_data=Result(base_dataframe=pl.DataFrame({}), info={}))
    dma.ocp_ne = [graphite_LGM50_ocp_Chen2020]
    dma.ocp_pe = [nmc_LGM50_ocp_Chen2020]
    x_pe_lo = 0.8
    x_pe_hi = 0.1
    x_ne_lo = 0.1
    x_ne_hi = 0.7
    soc = np.linspace(0, 1, 100)
    ocv_target = dma._f_OCV(soc, x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi)
    fit = dma._curve_fit_ocv(
        soc,
        ocv_target,
        "OCV",
        optimizer="minimize",
        optimizer_options={"x0": [0.8, 0.4, 0.2, 0.6]},
    )
    np.testing.assert_allclose(fit, [x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi], rtol=1e-6)


def test_curve_fit_ocv_target_dVdQ():
    """Test the curve_fit_ocv method with target dVdQ."""
    dma = DMA(input_data=Result(base_dataframe=pl.DataFrame({}), info={}))
    x_pe_lo = 0.8
    x_pe_hi = 0.1
    x_ne_lo = 0.1
    x_ne_hi = 0.7
    x_pe = np.linspace(x_pe_lo, x_pe_hi, 10000)
    x_ne = np.linspace(x_ne_lo, x_ne_hi, 10000)
    ocv_pe = nmc_LGM50_ocp_Chen2020(x_pe)
    ocv_ne = graphite_LGM50_ocp_Chen2020(x_ne)
    dma.set_ocp_from_data(x_pe, ocv_pe, electrode="pe")
    dma.set_ocp_from_data(x_ne, ocv_ne, electrode="ne")
    ocv_target = ocv_pe - ocv_ne
    soc = np.linspace(0, 1, 10000)
    d_ocv_target = np.gradient(ocv_target, soc)

    fit = dma._curve_fit_ocv(
        soc,
        d_ocv_target,
        "dVdQ",
        optimizer="minimize",
        optimizer_options={
            "x0": [0.8, 0.1, 0.1, 0.7],
            "bounds": [(0, 1), (0, 1), (0, 1), (0, 1)],
        },
    )
    np.testing.assert_allclose(fit, [x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi], rtol=5e-6)

    fit = dma._curve_fit_ocv(
        soc,
        d_ocv_target,
        "dVdQ",
        optimizer="differential_evolution",
        optimizer_options={
            "bounds": [(0.75, 0.85), (0.05, 15), (0.05, 0.15), (0.65, 0.75)]
        },
    )
    np.testing.assert_allclose(
        fit, [x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi], rtol=5e-4, atol=2e-4
    )


def test_curve_fit_ocv_target_dQdV():
    """Test the curve_fit_ocv method with target dQdV."""
    dma = DMA(input_data=Result(base_dataframe=pl.DataFrame({}), info={}))
    x_pe_lo = 0.8
    x_pe_hi = 0.1
    x_ne_lo = 0.1
    x_ne_hi = 0.6
    x_pe = np.linspace(x_pe_lo, x_pe_hi, 10000)
    x_ne = np.linspace(x_ne_lo, x_ne_hi, 10000)
    ocv_pe = nmc_LGM50_ocp_Chen2020(x_pe)
    ocv_ne = graphite_LGM50_ocp_Chen2020(x_ne)
    z = np.linspace(0, 1, 10000)
    dma.set_ocp_from_data(z, nmc_LGM50_ocp_Chen2020(z), electrode="pe")
    dma.set_ocp_from_data(z, graphite_LGM50_ocp_Chen2020(z), electrode="ne")
    ocv_target = ocv_pe - ocv_ne
    soc = np.linspace(0, 1, 10000)
    d_ocv_target = np.gradient(ocv_target, soc)

    fit = dma._curve_fit_ocv(
        soc,
        1 / d_ocv_target,
        "dQdV",
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
    np.testing.assert_allclose(
        fit, [x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi], rtol=5e-4, atol=1e-5
    )


def test_run_ocv_curve_fit():
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
        )
    )
    dma.set_ocp_from_data(x_pe, ocv_pe, electrode="pe")
    dma.set_ocp_from_data(x_ne, ocv_ne, electrode="ne")

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
    dma.set_ocp_from_data(x_pe, ocv_pe, electrode="pe")
    dma.set_ocp_from_data(x_ne, ocv_ne, electrode="ne")
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
