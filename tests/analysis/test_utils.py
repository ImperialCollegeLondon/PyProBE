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


def test_base_interpolator_model_valid():
    """Test _BaseInterpolatorModel with valid x and y."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    model = utils._BaseInterpolatorModel(x=x, y=y)
    np.testing.assert_array_equal(model.x, x)
    np.testing.assert_array_equal(model.y, y)


def test_base_interpolator_model_x_not_strictly_increasing_or_decreasing():
    """Test _BaseInterpolatorModel with x not strictly increasing or decreasing."""
    x = np.array([1, 3, 2])
    y = np.array([4, 6, 5])
    with pytest.raises(ValueError, match="x must be strictly increasing or decreasing"):
        utils._BaseInterpolatorModel(x=x, y=y)


def test_base_interpolator_model_x_and_y_different_lengths():
    """Test _BaseInterpolatorModel with x and y of different lengths."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5])
    with pytest.raises(ValueError, match="x and y must have the same length"):
        utils._BaseInterpolatorModel(x=x, y=y)


def test_base_interpolator_model_x_strictly_increasing():
    """Test _BaseInterpolatorModel with x strictly increasing."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    model = utils._BaseInterpolatorModel(x=x, y=y)
    np.testing.assert_array_equal(model.x, x)
    np.testing.assert_array_equal(model.y, y)


def test_base_interpolator_model_x_strictly_decreasing():
    """Test _BaseInterpolatorModel with x strictly decreasing."""
    x = np.array([3, 2, 1])
    y = np.array([6, 5, 4])
    model = utils._BaseInterpolatorModel(x=x, y=y)
    np.testing.assert_array_equal(model.x, x)
    np.testing.assert_array_equal(model.y, y)


def test_linear_interpolator_init():
    """Test _LinearInterpolator initialization with valid x and y."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    interpolator = utils._LinearInterpolator(x=x, y=y)
    np.testing.assert_array_equal(interpolator.x, x)
    np.testing.assert_array_equal(interpolator.y, y)


def test_create_interpolator_linear():
    """Test _create_interpolator with linear interpolation."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    LinearInterpolator = utils._create_interpolator(utils._LinearInterpolator)
    interpolator = LinearInterpolator(x=x, y=y)
    assert type(interpolator._interpolator) == utils._LinearInterpolator
    x_new = np.array([1.5, 2.5])
    y_new = interpolator(x_new)
    assert y_new.shape == x_new.shape


def test_create_interpolator_cubic():
    """Test _create_interpolator with cubic interpolation."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    CubicInterpolator = utils._create_interpolator(interpolate.CubicSpline)
    interpolator = CubicInterpolator(x=x, y=y)
    assert type(interpolator._interpolator) == interpolate.CubicSpline
    x_new = np.array([1.5, 2.5])
    y_new = interpolator(x_new)
    assert y_new.shape == x_new.shape


def test_create_interpolator_pchip():
    """Test _create_interpolator with Pchip interpolation."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    PchipInterpolator = utils._create_interpolator(interpolate.PchipInterpolator)
    interpolator = PchipInterpolator(x=x, y=y)
    assert type(interpolator._interpolator) == interpolate.PchipInterpolator
    x_new = np.array([1.5, 2.5])
    y_new = interpolator(x_new)
    assert y_new.shape == x_new.shape


def test_create_interpolator_akima():
    """Test _create_interpolator with Akima interpolation."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    AkimaInterpolator = utils._create_interpolator(interpolate.Akima1DInterpolator)
    interpolator = AkimaInterpolator(x=x, y=y)
    assert type(interpolator._interpolator) == interpolate.Akima1DInterpolator
    x_new = np.array([1.5, 2.5])
    y_new = interpolator(x_new)
    assert y_new.shape == x_new.shape


def test_interpolators_dict():
    """Test the interpolators dictionary."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    assert (
        type(utils.interpolators["Akima"](x=x, y=y)._interpolator)
        == interpolate.Akima1DInterpolator
    )
