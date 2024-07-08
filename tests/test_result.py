"""Tests for the result module."""

import numpy.testing as np_testing
import polars as pl
import polars.testing as pl_testing
import pytest

from pyprobe.result import Result


@pytest.fixture
def Result_fixture(lazyframe_fixture, info_fixture):
    """Return a Result instance."""
    return Result(lazyframe_fixture, info_fixture)


def test_init(Result_fixture):
    """Test the __init__ method."""
    assert isinstance(Result_fixture, Result)
    assert isinstance(Result_fixture._data, pl.LazyFrame)
    assert isinstance(Result_fixture.info, dict)


def test_call(Result_fixture):
    """Test the __call__ method."""
    current = Result_fixture("Current [A]")
    np_testing.assert_array_equal(
        current, Result_fixture.data["Current [A]"].to_numpy()
    )
    current_mA = Result_fixture("Current [mA]")
    np_testing.assert_array_equal(current_mA, current * 1000)


def test_data(Result_fixture):
    """Test the data property."""
    assert isinstance(Result_fixture._data, pl.LazyFrame)
    assert isinstance(Result_fixture.data, pl.DataFrame)
    assert isinstance(Result_fixture._data, pl.DataFrame)
    pl_testing.assert_frame_equal(Result_fixture.data, Result_fixture._data)


def test_check_units(Result_fixture):
    """Test the check_units method."""
    assert "Current [mA]" not in Result_fixture.data.columns
    Result_fixture.check_units("Current [mA]")
    assert "Current [mA]" in Result_fixture.data.columns


def test_print(Result_fixture, capsys):
    """Test the print method."""
    Result_fixture.print()
    captured = capsys.readouterr()
    assert captured.out.strip() == str(Result_fixture.data)


def test_print_definitions(Result_fixture, capsys):
    """Test the print_definitions method."""
    Result_fixture.define_column("Voltage [V]", "Voltage across the circuit")
    Result_fixture.define_column("Resistance [Ohm]", "Resistance of the circuit")
    Result_fixture.print_definitions()
    captured = capsys.readouterr()
    expected_output = (
        "{'Resistance [Ohm]': 'Resistance of the circuit'"
        ",\n 'Voltage [V]': 'Voltage across the circuit'}"
    )
    assert captured.out.strip() == expected_output
