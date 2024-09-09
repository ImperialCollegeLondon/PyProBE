"""Test the basecycler module."""
import copy

import polars as pl
import polars.testing as pl_testing
import pytest

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
            "Cycle": [0, 0, 0],
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


def test_map_columns(column_dict):
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
    expected_map = {
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

    assert BaseCycler.map_columns(dict_with_extra, column_list) == expected_map

    # missing columns
    column_list = ["DateTime", "T [s]", "V [V]", "I [mA]", "Q [Ah]", "Count"]
    expected_map.pop("Temperature")
    assert BaseCycler.map_columns(dict_with_extra, column_list) == expected_map


def test_init(sample_cycler_instance, sample_dataframe):
    """Test initialising the basecycler."""
    df = sample_dataframe.with_columns(
        pl.col("DateTime").str.to_datetime().alias("DateTime")
    )
    pl_testing.assert_frame_equal(
        sample_cycler_instance._imported_dataframe.collect(),
        df.with_columns(pl.all().cast(str)),
    )


def test_pyprobe_dataframe(sample_cycler_instance, sample_pyprobe_dataframe):
    """Test the pyprobe dataframe."""
    pl_testing.assert_frame_equal(
        sample_cycler_instance.pyprobe_dataframe.collect(), sample_pyprobe_dataframe
    )


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

    sample_pyprobe_dataframe = sample_pyprobe_dataframe.with_columns(
        pl.lit(None).alias("Date")
    )
    pl_testing.assert_frame_equal(
        base_cycler.pyprobe_dataframe.collect(), sample_pyprobe_dataframe
    )


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
