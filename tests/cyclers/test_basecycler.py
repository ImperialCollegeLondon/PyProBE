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


def test_init(sample_dataframe, sample_pyprobe_dataframe):
    """Test initialising the basecycler."""
    # test with single file
    sample_dataframe.write_csv("tests/sample_data/test_data.csv")
    base_cycler = BaseCycler(
        input_data_path="tests/sample_data/test_data.csv",
        column_name_pattern=r"([\w\s]+?)\s*\[(\w+)\]",
        column_dict={
            "Date": "DateTime",
            "Time": "T",
            "Voltage": "V",
            "Current": "I",
            "Capacity": "Q",
            "Step": "Count",
            "Temperature": "Temp",
        },
        common_suffix="",
    )

    df = sample_dataframe.with_columns(
        pl.col("DateTime").str.to_datetime().alias("DateTime")
    )
    pl_testing.assert_frame_equal(
        base_cycler._imported_dataframe.collect(), df.with_columns(pl.all().cast(str))
    )

    pl_testing.assert_frame_equal(
        base_cycler.pyprobe_dataframe.collect(), sample_pyprobe_dataframe
    )


def test_multiple_files(sample_dataframe):
    """Test reading multiple files."""
    # test with multiple files
    sample_dataframe.write_csv("tests/sample_data/test_data_1.csv")
    sample_dataframe.write_csv("tests/sample_data/test_data_2.csv")
    base_cycler = BaseCycler(
        input_data_path="tests/sample_data/test_data_*.csv",
        column_name_pattern=r"([\w\s]+?)\s*\[(\w+)\]",
        column_dict={
            "Date": "DateTime",
            "Time": "T",
            "Voltage": "V",
            "Current": "I",
            "Capacity": "Q",
            "Step": "Count",
            "Temperature": "Temp",
        },
    )

    df = sample_dataframe.with_columns(
        pl.col("DateTime").str.to_datetime().alias("DateTime")
    )
    expected_dataframe = pl.concat([df, df])
    pl_testing.assert_frame_equal(
        base_cycler._imported_dataframe.collect(),
        expected_dataframe.with_columns(pl.all().cast(str)),
    )


def test_missing_columns(sample_dataframe, sample_pyprobe_dataframe):
    """Test with a dataframe missing columns."""
    df = copy.deepcopy(sample_dataframe)
    df = df.drop("DateTime")
    df.write_csv("tests/sample_data/test_data.csv")
    base_cycler = BaseCycler(
        input_data_path="tests/sample_data/test_data.csv",
        column_name_pattern=r"([\w\s]+?)\s*\[(\w+)\]",
        column_dict={
            "Date": "DateTime",
            "Time": "T",
            "Voltage": "V",
            "Current": "I",
            "Capacity": "Q",
            "Step": "Count",
            "Temperature": "Temp",
        },
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


def test_ch_dis_capacity(sample_dataframe, sample_pyprobe_dataframe):
    """Test with a dataframe containing charge and discharge capacity."""
    df = copy.deepcopy(sample_dataframe)
    df = df.drop("Q [Ah]")

    ch_dis = pl.DataFrame(
        {"Q_ch [Ah]": [8.0, 9.0, 10.0], "Q_dis [Ah]": [0.0, 0.0, 0.0]}
    )
    df = df.hstack(ch_dis)
    df.write_csv("tests/sample_data/test_data.csv")
    base_cycler = BaseCycler(
        input_data_path="tests/sample_data/test_data.csv",
        column_name_pattern=r"([\w\s]+?)\s*\[(\w+)\]",
        column_dict={
            "Date": "DateTime",
            "Time": "T",
            "Voltage": "V",
            "Current": "I",
            "Step": "Count",
            "Temperature": "Temp",
            "Charge Capacity": "Q_ch",
            "Discharge Capacity": "Q_dis",
        },
    )
    pl_testing.assert_frame_equal(
        base_cycler.pyprobe_dataframe.collect(), sample_pyprobe_dataframe
    )
