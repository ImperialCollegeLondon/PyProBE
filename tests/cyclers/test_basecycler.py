"""Test the basecycler module."""
import copy
import os
import re

import numpy as np
import polars as pl
import polars.testing as pl_testing
import pytest
from polars.testing import assert_frame_equal

from pyprobe.cyclers.basecycler import BaseCycler


@pytest.fixture
def sample_dataframe():
    """A sample dataframe."""
    return pl.DataFrame(
        {
            "T [s]": [1.0, 2.0, 3.0],
            "V [V]": [4.0, 5.0, 6.0],
            "I [mA]": [7.0, 8.0, 9.0],
            "Q [Ah]": [10.0, 11.0, 12.0],
            "Count": [1, 2, 3],
            "Temp [C]": [13.0, 14.0, 15.0],
            "DateTime": [
                "2022-02-02 02:02:00.000000",
                "2022-02-02 02:02:01.000000",
                "2022-02-02 02:02:02.000000",
            ],
        }
    )


@pytest.fixture()
def sample_pyprobe_dataframe():
    """A sample dataframe in PyProBE format."""
    return pl.DataFrame(
        {
            "Date": [
                "2022-02-02 02:02:00.000000",
                "2022-02-02 02:02:01.000000",
                "2022-02-02 02:02:02.000000",
            ],
            "Time [s]": [1.0, 2.0, 3.0],
            "Step": [1, 2, 3],
            "Event": [0, 1, 2],
            "Current [A]": [7.0e-3, 8.0e-3, 9.0e-3],
            "Voltage [V]": [4.0, 5.0, 6.0],
            "Capacity [Ah]": [10.0, 11.0, 12.0],
            "Temperature [C]": [13.0, 14.0, 15.0],
        }
    ).with_columns(pl.col("Date").str.to_datetime())


@pytest.fixture
def column_dict():
    """A sample column dictionary."""
    return {
        "DateTime": "Date",
        "T [*]": "Time [*]",
        "V [*]": "Voltage [*]",
        "I [*]": "Current [*]",
        "Q [*]": "Capacity [*]",
        "Count": "Step",
        "Temp [*]": "Temperature [*]",
        "Q_ch [*]": "Charge Capacity [*]",
        "Q_dis [*]": "Discharge Capacity [*]",
    }


@pytest.fixture
def sample_cycler_instance(sample_dataframe, column_dict):
    """A sample cycler instance."""
    sample_dataframe.write_csv("tests/sample_data/test_data.csv")
    return BaseCycler(
        input_data_path="tests/sample_data/test_data.csv",
        column_dict=column_dict,
    )


@pytest.fixture
def sample_column_map():
    """A sample column map."""
    return {
        "Date": {
            "Cycler column name": "DateTime",
            "PyProBE column name": "Date",
            "Unit": "",
            "Type": pl.String,
        },
        "Time": {
            "Cycler column name": "T [s]",
            "PyProBE column name": "Time [s]",
            "Unit": "s",
            "Type": pl.Float64,
        },
        "Voltage": {
            "Cycler column name": "V [V]",
            "PyProBE column name": "Voltage [V]",
            "Unit": "V",
            "Type": pl.Float64,
        },
        "Current": {
            "Cycler column name": "I [mA]",
            "PyProBE column name": "Current [mA]",
            "Unit": "mA",
            "Type": pl.Float64,
        },
        "Capacity": {
            "Cycler column name": "Q [Ah]",
            "PyProBE column name": "Capacity [Ah]",
            "Unit": "Ah",
            "Type": pl.Float64,
        },
        "Step": {
            "Cycler column name": "Count",
            "PyProBE column name": "Step",
            "Unit": "",
            "Type": pl.Int64,
        },
        "Temperature": {
            "Cycler column name": "Temp [C]",
            "PyProBE column name": "Temperature [C]",
            "Unit": "C",
            "Type": pl.Float64,
        },
    }


def helper_read_and_process(
    benchmark,
    cycler_instance,
    expected_final_row,
    expected_events,
    expected_columns=[
        "Date",
        "Time [s]",
        "Step",
        "Event",
        "Current [A]",
        "Voltage [V]",
        "Capacity [Ah]",
        "Temperature [C]",
    ],
):
    """A helper function for other cyclers to test processing raw data files."""

    def read_and_process():
        result = cycler_instance.pyprobe_dataframe
        if isinstance(result, pl.LazyFrame):
            return result.collect()
        else:
            return result

    pyprobe_dataframe = benchmark(read_and_process)
    assert set(pyprobe_dataframe.columns) == set(expected_columns)
    assert (
        set(pyprobe_dataframe.select("Event").unique().to_series().to_list())
        == expected_events
    )
    assert_frame_equal(expected_final_row, pyprobe_dataframe.tail(1))
    return pyprobe_dataframe


def test_input_data_path_validator():
    """Test the input data path validator."""
    # test with invalid path
    path = "invalid_path"
    with pytest.raises(ValueError, match=f"File not found: path {path} does not exist"):
        BaseCycler._check_input_data_path(path)

    path = "invalid_path*"
    with pytest.raises(ValueError, match=f"No files found with the pattern {path}."):
        BaseCycler._check_input_data_path(path)

    # test with valid path
    assert (
        BaseCycler._check_input_data_path(
            "tests/sample_data/neware/sample_data_neware.csv"
        )
        == "tests/sample_data/neware/sample_data_neware.csv"
    )


def test_column_dict_validator(column_dict):
    """Test the column dictionary validator."""
    # test with missing columns
    column_dict.pop("I [*]")
    column_dict.pop("Q [*]")
    expected_message = re.escape(
        "The column dictionary is missing one or more required columns: "
        "{'Current [*]'}."
    )

    with pytest.raises(ValueError, match=expected_message):
        BaseCycler._check_column_dict(column_dict)

    column_dict.pop("Q_ch [*]")
    column_dict.pop("Q_dis [*]")
    with pytest.raises(
        ValueError,
    ):
        BaseCycler._check_column_dict(column_dict)


def test_map_columns(column_dict, sample_column_map):
    """Test initialising the basecycler."""
    # test with single file
    dict_with_extra = copy.deepcopy(column_dict)
    dict_with_extra["Ecell [*]"] = "Voltage [*]"
    column_list = [
        "DateTime",
        "T [s]",
        "V [V]",
        "I [mA]",
        "Q [Ah]",
        "Count",
        "Temp [C]",
    ]
    expected_map = sample_column_map

    assert BaseCycler._map_columns(dict_with_extra, column_list) == expected_map

    # missing columns
    column_list = ["DateTime", "T [s]", "V [V]", "I [mA]", "Q [Ah]", "Count"]
    expected_map.pop("Temperature")
    assert BaseCycler._map_columns(dict_with_extra, column_list) == expected_map


def test_check_missing_columns(sample_column_map, column_dict):
    """Test the check missing columns method."""
    sample_column_map.pop("Current")
    expected_message = (
        "PyProBE cannot find the following columns, please check your data: ['I [*]']."
    )
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        BaseCycler._check_missing_columns(column_dict, sample_column_map)


def test_tabulate_column_map(sample_column_map):
    """Test tabulating the column map."""
    column_map_table = BaseCycler._tabulate_column_map(sample_column_map)
    expected_dataframe = pl.DataFrame(
        {
            "Quantity": [
                "Date",
                "Time",
                "Voltage",
                "Current",
                "Capacity",
                "Step",
                "Temperature",
            ],
            "Cycler column name": [
                "DateTime",
                "T [s]",
                "V [V]",
                "I [mA]",
                "Q [Ah]",
                "Count",
                "Temp [C]",
            ],
            "PyProBE column name": [
                "Date",
                "Time [s]",
                "Voltage [V]",
                "Current [mA]",
                "Capacity [Ah]",
                "Step",
                "Temperature [C]",
            ],
        }
    )
    pl_testing.assert_frame_equal(column_map_table, expected_dataframe)


def test_init(sample_cycler_instance, sample_dataframe):
    """Test initialising the basecycler."""
    df = sample_dataframe.with_columns(
        pl.col("DateTime").str.to_datetime().alias("DateTime")
    )
    pl_testing.assert_frame_equal(
        sample_cycler_instance._imported_dataframe.collect(),
        df.with_columns(pl.all().cast(str)),
    )
    os.remove("tests/sample_data/test_data.csv")


def test_pyprobe_dataframe(sample_cycler_instance, sample_pyprobe_dataframe):
    """Test the pyprobe dataframe."""
    pl_testing.assert_frame_equal(
        sample_cycler_instance.pyprobe_dataframe.collect(), sample_pyprobe_dataframe
    )
    os.remove("tests/sample_data/test_data.csv")


def test_multiple_files(sample_dataframe, column_dict):
    """Test reading multiple files."""
    # test with multiple files
    sample_dataframe.write_csv("tests/sample_data/test_data_1.csv")
    sample_dataframe.write_csv("tests/sample_data/test_data_2.csv")
    base_cycler = BaseCycler(
        input_data_path="tests/sample_data/test_data_*.csv",
        column_dict=column_dict,
    )

    df = sample_dataframe.with_columns(
        pl.col("DateTime").str.to_datetime().alias("DateTime")
    )
    expected_dataframe = pl.concat([df, df])
    pl_testing.assert_frame_equal(
        base_cycler._imported_dataframe.collect(),
        expected_dataframe.with_columns(pl.all().cast(str)),
    )
    os.remove("tests/sample_data/test_data_1.csv")
    os.remove("tests/sample_data/test_data_2.csv")


def test_missing_columns(sample_dataframe, sample_pyprobe_dataframe, column_dict):
    """Test with a dataframe missing columns."""
    df = copy.deepcopy(sample_dataframe)
    df = df.drop("DateTime")
    df.write_csv("tests/sample_data/test_data.csv")
    base_cycler = BaseCycler(
        input_data_path="tests/sample_data/test_data.csv",
        column_dict=column_dict,
    )

    pl_testing.assert_frame_equal(
        base_cycler._imported_dataframe.collect(), df.with_columns(pl.all().cast(str))
    )

    sample_pyprobe_dataframe = sample_pyprobe_dataframe.drop("Date")
    pl_testing.assert_frame_equal(
        base_cycler.pyprobe_dataframe.collect(), sample_pyprobe_dataframe
    )
    os.remove("tests/sample_data/test_data.csv")


def test_ch_dis_capacity(sample_dataframe, sample_pyprobe_dataframe, column_dict):
    """Test with a dataframe containing charge and discharge capacity."""
    df = copy.deepcopy(sample_dataframe)
    df = df.drop("Q [Ah]")

    ch_dis = pl.DataFrame(
        {"Q_ch [Ah]": [8.0, 9.0, 10.0], "Q_dis [Ah]": [0.0, 0.0, 0.0]}
    )
    df = df.hstack(ch_dis)
    df.write_csv("tests/sample_data/test_data.csv")
    base_cycler = BaseCycler(
        input_data_path="tests/sample_data/test_data.csv", column_dict=column_dict
    )
    pl_testing.assert_frame_equal(
        base_cycler.pyprobe_dataframe.collect(), sample_pyprobe_dataframe
    )
    os.remove("tests/sample_data/test_data.csv")


def test_with_missing_columns(sample_dataframe):
    """Test with a dataframe missing columns."""
    sample_dataframe.write_csv("tests/sample_data/test_data.csv")
    df = copy.deepcopy(sample_dataframe)
    df = df.drop("I [mA]")
    df.write_csv("tests/sample_data/test_data1.csv")
    base_cycler = BaseCycler(
        input_data_path="tests/sample_data/test_data*.csv",
        column_dict={
            "DateTime": "Date",
            "T [*]": "Time [*]",
            "V [*]": "Voltage [*]",
            "I [*]": "Current [*]",
            "Q [*]": "Capacity [*]",
            "Count": "Step",
            "Temp [*]": "Temperature [*]",
            "Q_ch [*]": "Charge Capacity [*]",
            "Q_dis [*]": "Discharge Capacity [*]",
        },
    )
    assert np.all(
        np.isnan(
            base_cycler.pyprobe_dataframe.collect().select("Current [A]").to_numpy()[3:]
        )
    )
    os.remove("tests/sample_data/test_data.csv")
    os.remove("tests/sample_data/test_data1.csv")
