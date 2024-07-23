"""Tests for the differentiation module."""

import numpy as np
import polars as pl

import pyprobe.methods.differentiation as diff
from pyprobe.result import Result


def test_differentiate_FD():
    """Test the finite difference differentiation method."""
    # Test case 1: dydx gradient
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([2, 4, 6, 8, 10])
    rawdata = Result(pl.DataFrame({"x": x_data, "y": y_data}), {})
    rawdata.column_definitions = {"x": "The x data", "y": "The y data"}
    expected_gradient = np.array([2, 2, 2, 2, 2])
    result = diff.differentiate_FD(rawdata, "x", "y", gradient="dydx")
    assert isinstance(result, Result)
    assert np.allclose(result.get("d(y)/d(x)"), expected_gradient)

    # Test case 2: dxdy gradient
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([2, 4, 6, 8, 10])
    rawdata = Result(pl.DataFrame({"x": x_data, "y": y_data}), {})
    rawdata.column_definitions = {"x": "The x data", "y": "The y data"}
    expected_gradient = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    result = diff.differentiate_FD(rawdata, "x", "y", gradient="dxdy")
    assert isinstance(result, Result)
    assert np.allclose(result.get("d(x)/d(y)"), expected_gradient)

    # Test case 3: Invalid gradient option
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([2, 4, 6, 8, 10])
    rawdata = Result(pl.DataFrame({"x": x_data, "y": y_data}), {})
    rawdata.column_definitions = {"x": "The x data", "y": "The y data"}
    try:
        result = diff.differentiate_FD(rawdata, "x", "y", gradient="invalid")
        assert False  # Should raise a ValueError
    except ValueError:
        assert True


def test_differentiate_LEAN():
    """Test the LEAN differentiation method."""
    # Test case 1: dydx gradient
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([2, 4, 6, 8, 10])
    rawdata = Result(pl.DataFrame({"x": x_data, "y": y_data}), {})
    rawdata.column_definitions = {"x": "The x data", "y": "The y data"}
    result = diff.differentiate_LEAN(rawdata, "x", "y", gradient="dydx")
    assert isinstance(result, Result)
