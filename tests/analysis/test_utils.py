"""Tests for the utils class."""

import numpy as np
import polars as pl
import pytest
from scipy import interpolate

import pyprobe.analysis.utils as utils
from pyprobe.filters import Experiment
from pyprobe.result import Result


@pytest.fixture
def input_data_fixture():
    """Return a Result instance."""
    return Result(
        base_dataframe=pl.LazyFrame(
            {"x": [1, 2, 3], "y": [4, 5, 6], "Units [Ah]": [7, 8, 9]}
        ),
        info={},
        column_definitions={
            "x": "x definition",
            "y": "y definition",
            "Units [Ah]": "Units definition",
        },
    )


def test_assemble_array(input_data_fixture):
    """Test the assemble array method."""
    array = utils.assemble_array([input_data_fixture, input_data_fixture], "x")
    assert np.array_equal(array, np.array([[1, 2, 3], [1, 2, 3]]))


def test_base_analysis(input_data_fixture):
    """Test the base analysis class."""
    analysis = utils.AnalysisValidator(
        input_data=input_data_fixture, required_columns=["x"]
    )
    assert analysis.validate_input_data_type() == analysis
    assert analysis.validate_required_columns() == analysis

    with pytest.raises(ValueError):
        utils.AnalysisValidator(input_data=input_data_fixture, required_columns=["z"])

    np.testing.assert_array_equal(analysis.variables, np.array([1, 2, 3]))

    analysis = utils.AnalysisValidator(
        input_data=input_data_fixture, required_columns=["x", "y"]
    )
    x, y = analysis.variables
    np.testing.assert_array_equal(x, np.array([1, 2, 3]))
    np.testing.assert_array_equal(y, np.array([4, 5, 6]))

    with pytest.raises(ValueError):
        analysis = utils.AnalysisValidator(
            input_data=input_data_fixture,
            required_columns=["z"],
            required_type=Experiment,
        )

    analysis = utils.AnalysisValidator(
        input_data=input_data_fixture, required_columns=["Units [mAh]"]
    )
    np.testing.assert_array_equal(analysis.variables, np.array([7, 8, 9]) * 1000)


def test_linear_interpolator():
    """Test _LinearInterpolator initialization with valid x and y."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    interpolator = utils._LinearInterpolator(x, y)
    x_new = np.array([1.5, 2.5])
    y_new = interpolator(x_new)
    np.testing.assert_array_equal(y_new, np.array([4.5, 5.5]))


def test_create_interpolator_linear():
    """Test _create_interpolator with linear interpolation."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    interpolator = utils._create_interpolator(utils._LinearInterpolator, x, y)
    assert type(interpolator) == utils._LinearInterpolator
    assert isinstance(interpolator, interpolate.PPoly)
    x_new = np.array([1.5, 2.5])
    y_new = interpolator(x_new)
    assert y_new.shape == x_new.shape


def test_create_interpolator_cubic():
    """Test _create_interpolator with cubic interpolation."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    interpolator = utils._create_interpolator(interpolate.CubicSpline, x, y)
    assert type(interpolator) == interpolate.CubicSpline
    x_new = np.array([1.5, 2.5])
    y_new = interpolator(x_new)
    assert y_new.shape == x_new.shape


def test_create_interpolator_pchip():
    """Test _create_interpolator with Pchip interpolation."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    interpolator = utils._create_interpolator(interpolate.PchipInterpolator, x, y)
    assert type(interpolator) == interpolate.PchipInterpolator
    x_new = np.array([1.5, 2.5])
    y_new = interpolator(x_new)
    assert y_new.shape == x_new.shape


def test_create_interpolator_akima():
    """Test _create_interpolator with Akima interpolation."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    interpolator = utils._create_interpolator(interpolate.Akima1DInterpolator, x, y)
    assert type(interpolator) == interpolate.Akima1DInterpolator
    x_new = np.array([1.5, 2.5])
    y_new = interpolator(x_new)
    assert y_new.shape == x_new.shape


def test_interpolators_dict():
    """Test the interpolators dictionary."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    assert (
        type(utils.interpolators["Akima"](x=x, y=y)) == interpolate.Akima1DInterpolator
    )


def test_validate_interp_input_vectors_valid():
    """Test _validate_interp_input_vectors with valid input vectors."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    try:
        x, y = utils._validate_interp_input_vectors(x, y)
    except ValueError:
        pytest.fail("Unexpected ValueError raised")


def test_validate_interp_input_vectors_invalid_length():
    """Test _validate_interp_input_vectors with vectors of different lengths."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5])
    with pytest.raises(ValueError, match="x and y must have the same length"):
        x, y = utils._validate_interp_input_vectors(x, y)


def test_validate_interp_input_vectors_non_strictly_increasing():
    """Test _validate_interp_input_vectors with non-strictly increasing x."""
    x = np.array([1, 2, 2])
    y = np.array([4, 5, 6])
    with pytest.raises(ValueError, match="x must be strictly increasing or decreasing"):
        x, y = utils._validate_interp_input_vectors(x, y)


def test_validate_interp_input_vectors_non_strictly_decreasing():
    """Test _validate_interp_input_vectors with non-strictly decreasing x."""
    x = np.array([3, 2, 2])
    y = np.array([6, 5, 4])
    with pytest.raises(ValueError, match="x must be strictly increasing or decreasing"):
        x, y = utils._validate_interp_input_vectors(x, y)


def test_validate_interp_input_vectors_flip():
    """Test _validate_interp_input_vectors with decreasing x."""
    x = np.array([3, 2, 1])
    y = np.array([6, 5, 4])
    try:
        x, y = utils._validate_interp_input_vectors(x, y)
    except ValueError:
        pytest.fail("Unexpected ValueError raised")
    np.testing.assert_array_equal(x, np.array([1, 2, 3]))
    np.testing.assert_array_equal(y, np.array([4, 5, 6]))
