"""Tests for the smoothing analysis module."""
import numpy as np
import polars as pl
import polars.testing as pl_testing
import pytest
from scipy import interpolate

from pyprobe.analysis import smoothing
from pyprobe.analysis.smoothing import Smoothing
from pyprobe.result import Result


@pytest.fixture
def noisy_data():
    """Generate noisy data."""
    np.random.seed(42)
    x = np.arange(1, 6, 0.01)
    y = x**2 + np.random.normal(0, 0.1, size=x.size)  # y = x^2 with noise

    return Result(
        base_dataframe=pl.LazyFrame({"x": x, "y": y}),
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
        base_dataframe=pl.LazyFrame({"x": flipped_x, "y": flipped_y}),
        info={},
        column_definitions={"x": "The x data", "y": "The y data"},
    )


def test_spline_smoothing(noisy_data, noisy_data_reversed, benchmark):
    """Test the spline smoothing method with noisy data."""
    smoothing = Smoothing(input_data=noisy_data)

    def smooth():
        return smoothing.spline_smoothing(x="x", target_column="y").get_only("y")

    benchmark(smooth)

    result = smoothing.spline_smoothing(x="x", target_column="y")
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

    result = smoothing.level_smoothing(target_column="y", interval=1, monotonic=True)

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
        ).get_only("y")

    benchmark(smooth)

    result = smoothing.savgol_smoothing(
        target_column="y", window_length=100, polyorder=2
    )
    x = np.arange(1, 6, 0.01)
    expected_y = x**2

    np.testing.assert_allclose(result.get_only("y"), expected_y, rtol=0.2)
    assert set(result.column_list) == set(noisy_data.column_list)


def test_linear_interpolator():
    """Test _LinearInterpolator initialization with valid x and y."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    interpolator = smoothing._LinearInterpolator(x, y)
    x_new = np.array([1.5, 2.5])
    y_new = interpolator(x_new)
    np.testing.assert_array_equal(y_new, np.array([4.5, 5.5]))


def test_create_interpolator_linear():
    """Test _create_interpolator with linear interpolation."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    interpolator = smoothing._create_interpolator(smoothing._LinearInterpolator, x, y)
    assert type(interpolator) == smoothing._LinearInterpolator
    assert isinstance(interpolator, interpolate.PPoly)
    x_new = np.array([1.5, 2.5])
    y_new = interpolator(x_new)
    assert y_new.shape == x_new.shape


def test_create_interpolator_cubic():
    """Test _create_interpolator with cubic interpolation."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    interpolator = smoothing._create_interpolator(interpolate.CubicSpline, x, y)
    assert type(interpolator) == interpolate.CubicSpline
    x_new = np.array([1.5, 2.5])
    y_new = interpolator(x_new)
    assert y_new.shape == x_new.shape


def test_create_interpolator_pchip():
    """Test _create_interpolator with Pchip interpolation."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    interpolator = smoothing._create_interpolator(interpolate.PchipInterpolator, x, y)
    assert type(interpolator) == interpolate.PchipInterpolator
    x_new = np.array([1.5, 2.5])
    y_new = interpolator(x_new)
    assert y_new.shape == x_new.shape


def test_create_interpolator_akima():
    """Test _create_interpolator with Akima interpolation."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    interpolator = smoothing._create_interpolator(interpolate.Akima1DInterpolator, x, y)
    assert type(interpolator) == interpolate.Akima1DInterpolator
    x_new = np.array([1.5, 2.5])
    y_new = interpolator(x_new)
    assert y_new.shape == x_new.shape


def test_validate_interp_input_vectors_valid():
    """Test _validate_interp_input_vectors with valid input vectors."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    try:
        x, y = smoothing._validate_interp_input_vectors(x, y)
    except ValueError:
        pytest.fail("Unexpected ValueError raised")


def test_validate_interp_input_vectors_invalid_length():
    """Test _validate_interp_input_vectors with vectors of different lengths."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5])
    with pytest.raises(ValueError, match="x and y must have the same length"):
        x, y = smoothing._validate_interp_input_vectors(x, y)


def test_validate_interp_input_vectors_non_strictly_increasing():
    """Test _validate_interp_input_vectors with non-strictly increasing x."""
    x = np.array([1, 2, 2])
    y = np.array([4, 5, 6])
    with pytest.raises(ValueError, match="x must be strictly increasing or decreasing"):
        x, y = smoothing._validate_interp_input_vectors(x, y)


def test_validate_interp_input_vectors_non_strictly_decreasing():
    """Test _validate_interp_input_vectors with non-strictly decreasing x."""
    x = np.array([3, 2, 2])
    y = np.array([6, 5, 4])
    with pytest.raises(ValueError, match="x must be strictly increasing or decreasing"):
        x, y = smoothing._validate_interp_input_vectors(x, y)


def test_validate_interp_input_vectors_flip():
    """Test _validate_interp_input_vectors with decreasing x."""
    x = np.array([3, 2, 1])
    y = np.array([6, 5, 4])
    try:
        x, y = smoothing._validate_interp_input_vectors(x, y)
    except ValueError:
        pytest.fail("Unexpected ValueError raised")
    np.testing.assert_array_equal(x, np.array([1, 2, 3]))
    np.testing.assert_array_equal(y, np.array([4, 5, 6]))


def test_downsample_df(benchmark):
    """Test downsample_data with different occurrences."""
    times = np.linspace(0, 100, 101)
    values = times
    min_distance = 10
    df = pl.DataFrame({"Time [s]": times, "values": values})

    def smooth_data():
        return smoothing.downsample_data(df, "values", min_distance, occurrence="first")

    resampled_first = benchmark(smooth_data)["values"].to_numpy()
    resampled_last = smoothing.downsample_data(
        df, "values", min_distance, occurrence="last"
    )["values"].to_numpy()
    resampled_middle = smoothing.downsample_data(
        df, "values", min_distance, occurrence="middle"
    )["values"].to_numpy()

    np.testing.assert_array_equal(
        resampled_first, np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    )
    np.testing.assert_array_equal(
        resampled_last, np.array([9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 100])
    )
    np.testing.assert_array_equal(
        resampled_middle, np.array([5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 100])
    )

    # Test with non-monotonic data
    times = np.append(times, np.linspace(101, 200, 100))
    values = np.append(values, np.linspace(99, 0, 100))
    df = pl.DataFrame({"Time [s]": times, "values": values})
    resampled_first = smoothing.downsample_data(
        df, "values", min_distance, occurrence="first"
    )["values"].to_numpy()
    resampled_last = smoothing.downsample_data(
        df, "values", min_distance, occurrence="last"
    )["values"].to_numpy()
    resampled_middle = smoothing.downsample_data(
        df, "values", min_distance, occurrence="middle"
    )["values"].to_numpy()

    np.testing.assert_array_equal(
        resampled_first,
        np.array(
            [
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                90,
                80,
                70,
                60,
                50,
                40,
                30,
                20,
                10,
                0,
            ]
        ),
    )

    np.testing.assert_array_equal(
        resampled_last,
        np.array(
            [
                9,
                19,
                29,
                39,
                49,
                59,
                69,
                79,
                89,
                99,
                91,
                81,
                71,
                61,
                51,
                41,
                31,
                21,
                11,
                1,
                0,
            ]
        ),
    )

    np.testing.assert_array_equal(
        resampled_middle,
        np.array(
            [
                5,
                15,
                25,
                35,
                45,
                55,
                65,
                75,
                85,
                95,
                95,
                85,
                75,
                65,
                55,
                45,
                35,
                25,
                15,
                5,
                0,
            ]
        ),
    )
