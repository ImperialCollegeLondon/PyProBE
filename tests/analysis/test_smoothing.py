"""Tests for the smoothing analysis module."""
import numpy as np
import polars as pl
import pytest

from pyprobe.analysis.smoothing import Smoothing
from pyprobe.result import Result

np.random.seed(42)
x = np.arange(1, 6, 0.01)
y = x**2 + np.random.normal(0, 0.1, size=x.size)  # y = x^2 with noise


@pytest.fixture
def noisy_data():
    """Generate noisy data."""
    return Result(
        base_dataframe=pl.DataFrame({"x": x, "y": y}),
        info={},
        column_definitions={"x": "The x data", "y": "The y data"},
    )


@pytest.fixture
def noisy_data_reversed():
    """Generate noisy data."""
    flipped_x = np.flip(x)
    flipped_y = np.flip(y)
    return Result(
        base_dataframe=pl.DataFrame({"x": flipped_x, "y": flipped_y}),
        info={},
        column_definitions={"x": "The x data", "y": "The y data"},
    )


def test_spline_smoothing(noisy_data, noisy_data_reversed):
    """Test the spline smoothing method with noisy data."""
    smoothing = Smoothing(rawdata=noisy_data)
    result = smoothing.spline_smoothing("x", "y")
    x = np.arange(1, 6, 0.01)
    expected_y = x**2
    np.testing.assert_allclose(result.get_only("y"), expected_y, atol=0.1)

    expected_dydx = 2 * x
    np.testing.assert_allclose(result.get_only("d(y)/d(x)"), expected_dydx, atol=0.5)

    # reverse the data
    smoothing = Smoothing(rawdata=noisy_data_reversed)
    flipped_x = np.flip(x)
    result = smoothing.spline_smoothing("x", "y")
    flipped_expected_y = flipped_x**2
    np.testing.assert_allclose(result.get_only("y"), flipped_expected_y, atol=0.1)
