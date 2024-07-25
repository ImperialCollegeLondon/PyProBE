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
        pl.DataFrame({"x": x, "y": y}),
        {},
        column_definitions={"x": "The x data", "y": "The y data"},
    )


@pytest.fixture
def noisy_data_reversed():
    """Generate noisy data."""
    flipped_x = np.flip(x)
    flipped_y = np.flip(y)
    return Result(
        pl.DataFrame({"x": flipped_x, "y": flipped_y}),
        {},
        column_definitions={"x": "The x data", "y": "The y data"},
    )


def test_spline_smoothing(noisy_data, noisy_data_reversed):
    """Test the spline smoothing method with noisy data."""
    smoothing = Smoothing(noisy_data)
    result = smoothing.spline_smoothing("x", "y")
    x = np.arange(1, 6, 0.01)
    expected_y = x**2
    np.testing.assert_allclose(result.get("y"), expected_y, atol=0.1)

    expected_dydx = 2 * x
    np.testing.assert_allclose(result.get("d(y)/d(x)"), expected_dydx, atol=0.5)
    import matplotlib.pyplot as plt

    plt.plot(result.get("x"), result.get("y"), label="Smoothed")
    plt.plot(result.get("x"), noisy_data.get("y"), label="Noisy")
    plt.plot(result.get("x"), expected_y, label="Expected")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(result.get("x"), result.get("d(y)/d(x)"), label="Smoothed Gradient")
    plt.plot(result.get("x"), expected_dydx, label="Expected Gradient")
    plt.show()
    # reverse the data
    smoothing = Smoothing(noisy_data_reversed)
    flipped_x = np.flip(x)
    result = smoothing.spline_smoothing("x", "y")
    flipped_expected_y = flipped_x**2
    np.testing.assert_allclose(result.get("y"), flipped_expected_y, atol=0.1)
