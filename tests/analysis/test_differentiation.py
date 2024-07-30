"""Tests for the differentiation module."""

import numpy as np
import polars as pl
import pytest

from pyprobe.analysis.differentiation import Differentiation
from pyprobe.rawdata import RawData
from pyprobe.result import Result

x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([2, 4, 6, 8, 10])


@pytest.fixture
def differentiation_fixture():
    """Return a Differentiation instance."""
    input_data = RawData(
        base_dataframe=pl.DataFrame({"x": x_data, "y": y_data}), info={}
    )
    input_data.column_definitions = {"x": "The x data", "y": "The y data"}
    return Differentiation(input_data=input_data)


def test_differentiate_FD(differentiation_fixture):
    """Test the finite difference differentiation method."""
    # Test case 1: dydx gradient
    result = differentiation_fixture.differentiate_FD("x", "y", gradient="dydx")
    expected_gradient = np.array([2, 2, 2, 2, 2])
    assert isinstance(result, Result)
    assert np.allclose(result.get_only("d(y)/d(x)"), expected_gradient)

    # Test case 2: dxdy gradient
    result = differentiation_fixture.differentiate_FD("x", "y", gradient="dxdy")
    expected_gradient = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    assert isinstance(result, Result)
    assert np.allclose(result.get_only("d(x)/d(y)"), expected_gradient)

    # Test case 3: Invalid gradient option
    try:
        result = differentiation_fixture.differentiate_FD("x", "y", gradient="invalid")
        assert False  # Should raise a ValueError
    except ValueError:
        assert True


def test_differentiate_LEAN(differentiation_fixture):
    """Test the LEAN differentiation method."""
    # Test case 1: dydx gradient
    result = differentiation_fixture.differentiate_LEAN("x", "y", gradient="dydx")
    assert isinstance(result, Result)
