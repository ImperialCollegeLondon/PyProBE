"""Tests for the smoothing analysis module."""
import numpy as np
import polars as pl
import polars.testing as pl_testing
import pytest

from pyprobe.analysis.smoothing import Smoothing
from pyprobe.result import Result


@pytest.fixture
def noisy_data():
    """Generate noisy data."""
    np.random.seed(42)
    x = np.arange(1, 6, 0.01)
    y = x**2 + np.random.normal(0, 0.1, size=x.size)  # y = x^2 with noise

    return Result(
        base_dataframe=pl.DataFrame({"x": x, "y": y}),
        info={},
        column_definitions={"x": "The x data", "y": "The y data"},
    )


@pytest.fixture
def noisy_data_reversed():
    """Generate noisy data."""
    np.random.seed(42)
    x = np.arange(1, 6, 0.01)
    y = x**2 + np.random.normal(0, 0.5, size=x.size)  # y = x^2 with noise
    flipped_x = np.flip(x)
    flipped_y = np.flip(y)
    return Result(
        base_dataframe=pl.DataFrame({"x": flipped_x, "y": flipped_y}),
        info={},
        column_definitions={"x": "The x data", "y": "The y data"},
    )


def test_spline_smoothing(noisy_data, noisy_data_reversed, benchmark):
    """Test the spline smoothing method with noisy data."""
    smoothing = Smoothing(input_data=noisy_data)

    def smooth():
        return smoothing.spline_smoothing(x="x", target_column="y")

    result = benchmark(smooth)
    x = np.arange(1, 6, 0.01)
    expected_y = x**2

    np.testing.assert_allclose(result.get_only("y"), expected_y, rtol=0.2)

    input_data_columns = set(noisy_data.column_list + ["d(y)/d(x)"])
    result_columns = set(result.column_list)
    assert input_data_columns == result_columns

    expected_dydx = 2 * x

    np.testing.assert_allclose(result.get_only("d(y)/d(x)"), expected_dydx, rtol=0.2)

    # reverse the data
    smoothing = Smoothing(input_data=noisy_data_reversed)
    flipped_x = np.flip(x)
    result = smoothing.spline_smoothing(x="x", target_column="y")
    flipped_expected_y = flipped_x**2
    np.testing.assert_allclose(result.get_only("y"), flipped_expected_y, rtol=0.2)


def test_level_smoothing(noisy_data, noisy_data_reversed, benchmark):
    """Test the level smoothing method with noisy data."""
    smoothing = Smoothing(input_data=noisy_data)

    def smooth():
        return smoothing.level_smoothing(target_column="y", interval=1, monotonic=True)

    result = benchmark(smooth)

    assert result.data.select(pl.col("y").diff().min().alias("min_diff")).item(0, 0) > 1
    all_data = result.data.join(noisy_data.data, on="y")
    assert len(all_data) == len(result.data)
    pl_testing.assert_frame_equal(
        all_data.select("x"), all_data.select(pl.col("x_right").alias("x"))
    )
    assert set(result.column_list) == set(noisy_data.column_list)

    # reverse the data
    smoothing = Smoothing(input_data=noisy_data_reversed)
    result = smoothing.level_smoothing(target_column="y", interval=1)
    assert (
        result.data.select(pl.col("y").diff().abs().min().alias("min_diff")).item(0, 0)
        > 1
    )
    all_data = result.data.join(noisy_data_reversed.data, on="y")
    assert len(all_data) == len(result.data)
    pl_testing.assert_frame_equal(
        all_data.select("x"), all_data.select(pl.col("x_right").alias("x"))
    )


def test_savgol_smoothing(noisy_data, noisy_data_reversed, benchmark):
    """Test the Savgol smoothing method."""
    smoothing = Smoothing(input_data=noisy_data)

    def smooth():
        return smoothing.savgol_smoothing(
            target_column="y", window_length=100, polyorder=2
        )

    result = benchmark(smooth)
    x = np.arange(1, 6, 0.01)
    expected_y = x**2

    np.testing.assert_allclose(result.get_only("y"), expected_y, rtol=0.2)
    assert set(result.column_list) == set(noisy_data.column_list)
