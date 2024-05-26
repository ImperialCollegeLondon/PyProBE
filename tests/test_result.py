"""Tests for the result module."""

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
