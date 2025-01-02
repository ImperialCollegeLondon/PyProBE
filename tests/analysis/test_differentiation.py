"""Tests for the differentiation module."""

import numpy as np
import polars as pl
import pytest

from pyprobe.analysis import differentiation
from pyprobe.result import Result

x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([2, 4, 6, 8, 10])


@pytest.fixture
def differentiation_fixture():
    """Return a Differentiation instance."""
    input_data = Result(
        base_dataframe=pl.DataFrame({"x": x_data, "y": y_data}), info={}
    )
    input_data.column_definitions = {"x": "The x data", "y": "The y data"}
    return input_data


def test_gradient(differentiation_fixture):
    """Test the finite difference differentiation method."""
    result = differentiation.gradient(differentiation_fixture, "x", "y")
    expected_gradient = np.array([2, 2, 2, 2, 2])
    assert isinstance(result, Result)
    assert np.allclose(result.get("d(y)/d(x)"), expected_gradient)


def test_differentiate_LEAN(differentiation_fixture):
    """Test the LEAN differentiation method."""
    # Test case 1: dydx gradient
    result = differentiation.differentiate_LEAN(
        differentiation_fixture, "x", "y", gradient="dydx"
    )
    assert isinstance(result, Result)
