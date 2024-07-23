"""Tests for the degradation mode analysis functions."""
import math

import numpy as np

import pyprobe.methods.base.degradation_mode_analysis_functions as dma_functions


def test_calc_full_cell_OCV():
    """Test the calc_full_cell_OCV function."""
    SOC = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    x_pe_lo = 0.2
    x_pe_hi = 0.8
    x_ne_lo = 0.1
    x_ne_hi = 0.9
    x_pe = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ocp_pe = np.array([3.0, 3.2, 3.4, 3.6, 3.8, 4.0])
    x_ne = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ocp_ne = np.array([2.0, 2.2, 2.4, 2.6, 2.8, 3.0])

    expected_result = np.array([1.1, 1.06, 1.02, 0.98, 0.94, 0.9])

    result = dma_functions.calc_full_cell_OCV(
        SOC, x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi, x_pe, ocp_pe, x_ne, ocp_ne
    )
    np.testing.assert_allclose(result, expected_result)


def test_calc_electrode_capacities():
    """Test the calc_electrode_capacities function."""
    x_pe_lo = 0.9
    x_pe_hi = 0.1
    x_ne_lo = 0.2
    x_ne_hi = 0.8
    cell_capacity = 100.0

    pe_capacity, ne_capacity, li_inventory = dma_functions.calc_electrode_capacities(
        x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi, cell_capacity
    )

    assert math.isclose(pe_capacity, 125.0)
    assert math.isclose(ne_capacity, 100 / 0.6)
    assert math.isclose(li_inventory, 875 / 6)


def test_calculate_dma_parameters():
    """Test the calculate_dma_parameters function."""
    cell_capacity = np.array([100, 90, 80, 70])
    pe_capacity = np.array([50, 45, 40, 35])
    ne_capacity = np.array([30, 27, 24, 21])
    li_inventory = np.array([20, 18, 16, 14])

    SOH, LAM_pe, LAM_ne, LLI = dma_functions.calculate_dma_parameters(
        cell_capacity, pe_capacity, ne_capacity, li_inventory
    )

    np.testing.assert_allclose(SOH, [1.0, 0.9, 0.8, 0.7])
    np.testing.assert_allclose(LAM_pe, [0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(LAM_ne, [0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(LLI, [0, 0.1, 0.2, 0.3])
