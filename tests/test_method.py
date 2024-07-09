"""Tests for the BaseMethod class."""

import numpy as np
import polars as pl
import pytest

from pyprobe.methods.basemethod import BaseMethod
from pyprobe.result import Result


@pytest.fixture
def input_data_fixture():
    """Return a Result instance."""
    return Result(_data=pl.LazyFrame({"x": [1, 2, 3]}), info={})


@pytest.fixture
def method_fixture(input_data_fixture):
    """Return a BaseMethod instance."""
    return BaseMethod(input_data_fixture)


def test_init(method_fixture):
    """Test the __init__ method."""
    assert isinstance(method_fixture, BaseMethod)
    assert method_fixture.variable_list == []


def test_variable(method_fixture, input_data_fixture):
    """Test the variable method."""
    assert np.array_equal(method_fixture.variable("x"), np.array([1, 2, 3]))
    assert method_fixture.variable_list == ["x"]

    method = BaseMethod([input_data_fixture, input_data_fixture])
    assert np.array_equal(method.variable("x"), np.array([[1, 2, 3], [1, 2, 3]]))


def test_assign_outputs(method_fixture):
    """Test the make_result method."""
    result = method_fixture.make_result(
        {"y": np.array([4, 5, 6]), "z": np.array([7, 8, 9])}
    )
    expected_result = pl.DataFrame({"y": np.array([4, 5, 6]), "z": np.array([7, 8, 9])})
    pl.testing.assert_frame_equal(result.data, expected_result)

    result = method_fixture.make_result(
        {"y": np.array([[10, 11]]), "z": np.array([12])}
    )
    expected_result = pl.DataFrame({"y": np.array([[10, 11]]), "z": np.array([12])})
    pl.testing.assert_frame_equal(result.data, expected_result)
