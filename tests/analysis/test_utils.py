"""Tests for the utils class."""

import numpy as np
import polars as pl
import pytest

import pyprobe.analysis.utils as utils
from pyprobe.filters import Experiment
from pyprobe.result import Result


@pytest.fixture
def input_data_fixture():
    """Return a Result instance."""
    return Result(
        base_dataframe=pl.LazyFrame({"x": [1, 2, 3], "y": [4, 5, 6]}), info={}
    )


def test_assemble_array(input_data_fixture):
    """Test the assemble array method."""
    array = utils.assemble_array([input_data_fixture, input_data_fixture], "x")
    assert np.array_equal(array, np.array([[1, 2, 3], [1, 2, 3]]))


def test_base_analysis(input_data_fixture):
    """Test the base analysis class."""
    analysis = utils.BaseAnalysis(input_data=input_data_fixture, required_columns=["x"])
    assert analysis.validate_input_data_type() == analysis
    assert analysis.validate_required_columns() == analysis

    with pytest.raises(ValueError):
        utils.BaseAnalysis(input_data=input_data_fixture, required_columns=["z"])

    np.testing.assert_array_equal(analysis.variables, np.array([1, 2, 3]))

    analysis = utils.BaseAnalysis(
        input_data=input_data_fixture, required_columns=["x", "y"]
    )
    x, y = analysis.variables
    np.testing.assert_array_equal(x, np.array([1, 2, 3]))
    np.testing.assert_array_equal(y, np.array([4, 5, 6]))

    with pytest.raises(ValueError):
        analysis = utils.BaseAnalysis(
            input_data=input_data_fixture,
            required_columns=["z"],
            required_type=Experiment,
        )
