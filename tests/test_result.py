"""Tests for the result module."""

from datetime import datetime, timedelta

import numpy.testing as np_testing
import polars as pl
import polars.testing as pl_testing
import pytest

from pyprobe.result import Result


@pytest.fixture
def Result_fixture(lazyframe_fixture, info_fixture):
    """Return a Result instance."""
    return Result(
        base_dataframe=lazyframe_fixture,
        info=info_fixture,
        column_definitions={
            "Current [A]": "Current definition",
        },
    )


def test_init(Result_fixture):
    """Test the __init__ method."""
    assert isinstance(Result_fixture, Result)
    assert isinstance(Result_fixture.base_dataframe, pl.LazyFrame)
    assert isinstance(Result_fixture.info, dict)


def test_call(Result_fixture):
    """Test the __call__ method."""
    current = Result_fixture("Current [A]")
    np_testing.assert_array_equal(
        current, Result_fixture.data["Current [A]"].to_numpy()
    )
    current_mA = Result_fixture("Current [mA]")
    np_testing.assert_array_equal(current_mA, current * 1000)


def test_get(Result_fixture):
    """Test the get method."""
    current = Result_fixture.get("Current [A]")
    np_testing.assert_array_equal(
        current, Result_fixture.data["Current [A]"].to_numpy()
    )
    current_mA = Result_fixture.get("Current [mA]")
    np_testing.assert_array_equal(current_mA, current * 1000)

    current, voltage = Result_fixture.get("Current [A]", "Voltage [V]")
    np_testing.assert_array_equal(
        current, Result_fixture.data["Current [A]"].to_numpy()
    )
    np_testing.assert_array_equal(
        voltage, Result_fixture.data["Voltage [V]"].to_numpy()
    )


def test_get_only(Result_fixture):
    """Test the get_only method."""
    current = Result_fixture.get_only("Current [A]")
    np_testing.assert_array_equal(
        current, Result_fixture.data["Current [A]"].to_numpy()
    )
    current_mA = Result_fixture.get_only("Current [mA]")
    np_testing.assert_array_equal(current_mA, current * 1000)


def test_array(Result_fixture):
    """Test the array method."""
    array = Result_fixture.array()
    np_testing.assert_array_equal(array, Result_fixture.data.to_numpy())

    filtered_array = Result_fixture.array("Current [A]", "Voltage [V]")
    np_testing.assert_array_equal(
        filtered_array,
        Result_fixture.data.select("Current [A]", "Voltage [V]").to_numpy(),
    )


def test_getitem(Result_fixture):
    """Test the __getitem__ method."""
    current = Result_fixture["Current [A]"]
    assert "Current [A]" in current.column_list
    assert isinstance(current, Result)
    pl_testing.assert_frame_equal(
        current.data, Result_fixture.data.select("Current [A]")
    )
    current_mA = Result_fixture["Current [mA]"]
    assert "Current [mA]" in current_mA.column_list
    assert "Current [A]" not in current_mA.column_list
    np_testing.assert_allclose(
        current_mA.get_only("Current [mA]"), Result_fixture.get_only("Current [mA]")
    )


def test_data(Result_fixture):
    """Test the data property."""
    assert isinstance(Result_fixture.base_dataframe, pl.LazyFrame)
    assert isinstance(Result_fixture.data, pl.DataFrame)
    assert isinstance(Result_fixture.base_dataframe, pl.DataFrame)
    pl_testing.assert_frame_equal(Result_fixture.data, Result_fixture.base_dataframe)


def test_quantities(Result_fixture):
    """Test the quantities property."""
    assert set(Result_fixture.quantities) == set(
        ["Time", "Current", "Voltage", "Capacity"]
    )


def test_check_units(Result_fixture):
    """Test the _check_units method."""
    assert "Current [mA]" not in Result_fixture.data.columns
    Result_fixture._check_units("Current [mA]")
    assert "Current [mA]" in Result_fixture.data.columns
    assert "Current [mA]" in Result_fixture.column_definitions.keys()
    assert (
        Result_fixture.column_definitions["Current [mA]"]
        == Result_fixture.column_definitions["Current [A]"]
    )

    result = Result(
        base_dataframe=pl.DataFrame({"Invented quantity [V]": [1, 2, 3]}),
        info={},
        column_definitions={"Invented quantity [V]": "Invented quantity definition"},
    )
    assert "Invented quantity [mV]" not in result.data.columns
    result._check_units("Invented quantity [mV]")
    assert "Invented quantity [mV]" in result.data.columns


def test_print_definitions(Result_fixture, capsys):
    """Test the print_definitions method."""
    Result_fixture.define_column("Voltage [V]", "Voltage across the circuit")
    Result_fixture.define_column("Resistance [Ohm]", "Resistance of the circuit")
    Result_fixture.print_definitions()
    captured = capsys.readouterr()
    expected_output = (
        "{'Current [A]': 'Current definition'"
        ",\n 'Resistance [Ohm]': 'Resistance of the circuit'"
        ",\n 'Voltage [V]': 'Voltage across the circuit'}"
    )
    assert captured.out.strip() == expected_output


def test_build():
    """Test the build method."""
    data1 = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    data2 = pl.DataFrame({"x": [7, 8, 9], "y": [10, 11, 12]})
    info = {"test": "info"}
    result = Result.build([data1, data2], info)
    assert isinstance(result, Result)
    expected_data = pl.DataFrame(
        {
            "x": [1, 2, 3, 7, 8, 9],
            "y": [4, 5, 6, 10, 11, 12],
            "Step": [0, 0, 0, 1, 1, 1],
            "Cycle": [0, 0, 0, 0, 0, 0],
        }
    )
    pl_testing.assert_frame_equal(
        result.data, expected_data, check_column_order=False, check_dtype=False
    )


def test_add_new_data_columns():
    """Test the add_new_data_columns method."""
    existing_data = pl.LazyFrame(
        {
            "Date": pl.datetime_range(
                datetime(1985, 1, 1, 0, 0, 0),
                datetime(1985, 1, 1, 0, 0, 5),
                timedelta(seconds=1),
                time_unit="ms",
                eager=True,
            ).alias("datetime"),
            "Data": [2, 4, 6, 8, 10, 12],
        }
    )
    new_data = pl.LazyFrame(
        {
            "DateTime": pl.datetime_range(
                datetime(1985, 1, 1, 0, 0, 2, 500000),
                datetime(1985, 1, 1, 0, 0, 7, 500000),
                timedelta(seconds=1),
                time_unit="ms",
                eager=True,
            ).alias("datetime"),
            "Data 1": [2, 4, 6, 8, 10, 12],
            "Data 2": [4, 8, 12, 16, 20, 24],
        }
    )
    result_object = Result(base_dataframe=existing_data, info={})
    result_object.add_new_data_columns(new_data, date_column_name="DateTime")
    expected_data = pl.DataFrame(
        {
            "Date": pl.datetime_range(
                datetime(1985, 1, 1, 0, 0, 0),
                datetime(1985, 1, 1, 0, 0, 5),
                timedelta(seconds=1),
                time_unit="ms",
                eager=True,
            )
            .dt.cast_time_unit("us")
            .alias("datetime"),
            "Data": [2, 4, 6, 8, 10, 12],
            "Data 1": [None, None, None, 3.0, 5.0, 7.0],
            "Data 2": [None, None, None, 6.0, 10.0, 14.0],
        }
    )
    pl_testing.assert_frame_equal(result_object.data, expected_data)
