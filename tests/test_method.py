"""Tests for the Method class."""

import numpy as np
import polars as pl
import pytest

from pyprobe.method import Method
from pyprobe.result import Result


@pytest.fixture
def input_data_fixture():
    """Return a Result instance."""
    return Result(_data=pl.LazyFrame({"x": [1, 2, 3]}), info={})


@pytest.fixture
def parameters_fixture():
    """Return a dictionary of parameters."""
    return {"a": 1, "b": 2}


@pytest.fixture
def method_fixture(input_data_fixture, parameters_fixture):
    """Return a Method instance."""
    return Method(input_data_fixture, parameters_fixture)


def test_init(method_fixture):
    """Test the __init__ method."""
    assert isinstance(method_fixture, Method)
    assert method_fixture.parameters == {"a": 1, "b": 2}
    assert method_fixture.variable_list == []
    assert method_fixture.parameter_list == []
    assert method_fixture.output_dict == {}


def test_variable(method_fixture, input_data_fixture, parameters_fixture):
    """Test the variable method."""
    assert np.array_equal(method_fixture.variable("x"), np.array([1, 2, 3]))
    assert method_fixture.variable_list == ["x"]

    method = Method([input_data_fixture, input_data_fixture], parameters_fixture)
    assert np.array_equal(method.variable("x"), np.array([[1, 2, 3], [1, 2, 3]]))


def test_parameter(method_fixture):
    """Test the parameter method."""
    assert method_fixture.parameter("a") == 1
    assert method_fixture.parameter_list == ["a"]


def test_assign_outputs(method_fixture):
    """Test the assign_outputs method."""
    result = method_fixture.assign_outputs(
        ["y", "z"], (np.array([4, 5, 6]), np.array([7, 8, 9]))
    )
    expected_result = pl.DataFrame({"y": np.array([4, 5, 6]), "z": np.array([7, 8, 9])})
    assert method_fixture.output_dict.keys() == {"y", "z"}
    assert np.array_equal(method_fixture.output_dict["y"], np.array([4, 5, 6]))
    assert np.array_equal(method_fixture.output_dict["z"], np.array([7, 8, 9]))
    pl.testing.assert_frame_equal(result.data, expected_result)

    result = method_fixture.assign_outputs(
        ["y", "z"], (np.array([[10, 11]]), np.array([12]))
    )
    expected_result = pl.DataFrame({"y": np.array([[10, 11]]), "z": np.array([12])})
    assert (method_fixture.output_dict["y"] == np.array([[10, 11]])).all()
    assert (method_fixture.output_dict["z"] == np.array([12])).all()
    pl.testing.assert_frame_equal(result.data, expected_result)
