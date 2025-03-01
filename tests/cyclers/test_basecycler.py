"""Test the basecycler module."""

import datetime
import logging
import os
import re

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pyprobe.cyclers.basecycler import (
    BaseCycler,
    CapacityFromChDch,
    CapacityFromCurrentSign,
    CastAndRename,
    ColumnMap,
    ConvertTemperature,
    ConvertUnits,
    DateTime,
    TimeFromDate,
)


class TestColumnMap(ColumnMap):
    """Test implementation of ColumnMap."""

    @property
    def expr(self) -> pl.Expr:
        """Implement abstract method."""
        return None


def test_match_columns():
    """Test the match_columns static method."""
    available_columns = ["Time [s]", "Current [mA]", "Voltage [V]", "Count"]
    column_map_instance = TestColumnMap("", [""])
    # Test exact matches
    patterns = ["Count"]
    result = column_map_instance.match_columns(available_columns, patterns)
    assert result == {"Count": {"Cycler name": "Count", "Cycler unit": ""}}

    # Test wildcard matches
    patterns = ["Current [*]"]
    result = column_map_instance.match_columns(available_columns, patterns)
    assert result == {
        "Current [*]": {"Cycler name": "Current [mA]", "Cycler unit": "mA"},
    }

    # Test multiple patterns
    patterns = ["Count", "Current [*]", "Voltage [*]"]
    result = column_map_instance.match_columns(available_columns, patterns)
    assert result == {
        "Count": {"Cycler name": "Count", "Cycler unit": ""},
        "Current [*]": {"Cycler name": "Current [mA]", "Cycler unit": "mA"},
        "Voltage [*]": {"Cycler name": "Voltage [V]", "Cycler unit": "V"},
    }

    # Test non-existent columns
    patterns = ["NonExistent"]
    result = column_map_instance.match_columns(available_columns, patterns)
    assert result == {}

    # Test invalid unit
    available_columns = ["Current [invalid]"]
    patterns = ["Current [*]"]
    result = column_map_instance.match_columns(available_columns, patterns)
    assert result == {}

    # Test empty inputs
    assert column_map_instance.match_columns([], []) == {}
    assert column_map_instance.match_columns(available_columns, []) == {}
    assert column_map_instance.match_columns([], patterns) == {}

    # test a different format
    available_columns = ["Time/s", "Current/mA", "Voltage/time/V", "Count", "Voltage/V"]
    patterns = ["Count", "Current/*", "Voltage/*"]
    result = column_map_instance.match_columns(available_columns, patterns)
    assert result == {
        "Count": {"Cycler name": "Count", "Cycler unit": ""},
        "Current/*": {"Cycler name": "Current/mA", "Cycler unit": "mA"},
        "Voltage/*": {"Cycler name": "Voltage/V", "Cycler unit": "V"},
    }


def test_ColumnMap_validate():
    """Test the ColumnMap class."""
    column_map = TestColumnMap("Date", ["DateTime"])
    column_map.validate(["DateTime"])
    assert column_map.pyprobe_name == "Date"
    assert column_map.column_map == {
        "DateTime": {"Cycler name": "DateTime", "Cycler unit": ""},
    }
    assert column_map.columns_validated

    column_map = TestColumnMap("Date", ["DateTime", "Current [*]"])
    column_map.validate(["DateTime", "Current [mA]", "Voltage [V]"])
    assert column_map.pyprobe_name == "Date"
    assert column_map.column_map == {
        "DateTime": {"Cycler name": "DateTime", "Cycler unit": ""},
        "Current [*]": {"Cycler name": "Current [mA]", "Cycler unit": "mA"},
    }
    assert column_map.columns_validated

    column_map = TestColumnMap("Date", ["DateTime"])
    column_map.validate(["Date"])
    assert column_map.pyprobe_name == "Date"
    assert not column_map.columns_validated
    assert column_map.column_map == {}

    column_map = TestColumnMap("Date", ["DateTime", "Current [*]"])
    column_map.validate(["Date", "Current [mA]", "Voltage [V]"])
    assert column_map.pyprobe_name == "Date"
    assert column_map.column_map == {
        "Current [*]": {"Cycler name": "Current [mA]", "Cycler unit": "mA"},
    }
    assert not column_map.columns_validated


def test_ColumnMap_get():
    """Test the get method of the ColumnMap class."""
    column_map = TestColumnMap("Date", ["DateTime", "Current [*]"])
    column_map.validate(["DateTime", "Current [mA]", "Voltage [V]"])
    assert str(column_map.get("DateTime")) == str(pl.col("DateTime"))
    assert str(column_map.get("Current [*]")) == str(pl.col("Current [mA]"))


def test_CastAndRename():
    """Test the CastAndRename class."""
    column_map = CastAndRename("Date", "DateTime", pl.String)
    column_map.validate(["DateTime"])
    assert str(column_map.expr) == str(pl.col("DateTime").cast(pl.String).alias("Date"))


def test_ConvertUnits():
    """Test the ConvertUnits class."""
    column_map = ConvertUnits("Current [A]", "I [*]")
    column_map.validate(["I [mA]"])
    df = pl.DataFrame({"I [mA]": [1.0, 2.0, 3.0]})
    assert_frame_equal(
        df.select(column_map.expr),
        df.select((pl.col("I [mA]") / 1000).alias("Current [A]")),
    )


def test_ConvertTemperature():
    """Test the ConvertTemperature class."""
    column_map = ConvertTemperature("Temperature/*")
    column_map.validate(["Temperature/K"])
    df = pl.DataFrame({"Temperature/K": [300, 305, 310]})
    expected_df = pl.DataFrame({"Temperature [C]": [26.85, 31.85, 36.85]})
    assert_frame_equal(df.select(column_map.expr), expected_df)


def test_CapacityFromChDch():
    """Test the CapacityFromChDch class."""
    column_map = CapacityFromChDch("Q_ch [*]", "Q_dis [*]")
    column_map.validate(["Q_ch [mAh]", "Q_dis [Ah]"])
    df = pl.DataFrame({"Q_ch [mAh]": [1.0, 0.0, 0.0], "Q_dis [Ah]": [0.0, 1, 2]})
    expected_df = pl.DataFrame({"Capacity [Ah]": [0.001, -0.999, -1.999]})
    assert_frame_equal(df.select(column_map.expr), expected_df)


def test_CapacityFromCurrentSign():
    """Test the CapacityFromCurrentSign class."""
    column_map = CapacityFromCurrentSign(
        "Q [*]",
        "I [*]",
    )
    df = pl.DataFrame({"I [mA]": [1.0, -1.0, -1.0], "Q [Ah]": [1.0, 1.0, 2.0]})
    column_map.validate(df.columns)
    expected_df = pl.DataFrame({"Capacity [Ah]": [1.0, 0.0, -1.0]})
    assert_frame_equal(df.select(column_map.expr), expected_df)


def test_DateTime():
    """Test the DateTime class."""
    column_map = DateTime("DateTime", "%Y-%m-%d %H:%M:%S%.f")
    column_map.validate(["DateTime"])
    df = pl.DataFrame(
        {
            "DateTime": [
                "2022-04-02 02:06:00.000000",
                "2022-04-02 02:06:01.000000",
                "2022-04-02 02:06:02.000000",
            ],
        },
    )
    expected_df = pl.DataFrame(
        {
            "Date": [
                datetime.datetime(2022, 4, 2, 2, 6),
                datetime.datetime(2022, 4, 2, 2, 6, 1),
                datetime.datetime(2022, 4, 2, 2, 6, 2),
            ],
        },
    )
    assert_frame_equal(df.select(column_map.expr), expected_df)


def test_TimeFromDate():
    """Test the TimeFromDate class."""
    column_map = TimeFromDate("DateTime", "%Y-%m-%d %H:%M:%S%.f")
    column_map.validate(["DateTime"])
    df = pl.DataFrame(
        {
            "DateTime": [
                "2022-04-02 02:06:00.000000",
                "2022-04-02 02:06:01.000000",
                "2022-04-02 02:06:02.000000",
            ],
        },
    )
    expected_df = pl.DataFrame({"Time [s]": [0.0, 1.0, 2.0]})
    assert_frame_equal(df.select(column_map.expr), expected_df)


@pytest.fixture
def sample_dataframe():
    """A sample dataframe."""
    return pl.DataFrame(
        {
            "T [s]": [1.0, 2.0, 3.0],
            "V [V]": [4.0, 5.0, 6.0],
            "I [mA]": [7.0, 8.0, 9.0],
            "Q [Ah]": [1.0, 0.5, -1.5],
            "Count": [1, 2, 3],
            "Temp [C]": [13.0, 14.0, 15.0],
            "DateTime": [
                "2022-02-02 02:02:00.000000",
                "2022-02-02 02:02:01.000000",
                "2022-02-02 02:02:02.000000",
            ],
            "Q_ch [mAh]": [1.0, 0.0, 0.0],
            "Q_dis [Ah]": [0.0, 0.5, 2],
        },
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
            "Capacity [Ah]": [1.0, 0.5, -1.5],
            "Temperature [C]": [13.0, 14.0, 15.0],
        },
    ).with_columns(pl.col("Date").str.to_datetime())


@pytest.fixture
def column_importer_fixture():
    """A sample column importer."""
    return [
        CastAndRename("Step", "Count", pl.Int64),
        ConvertUnits("Current [A]", "I [*]"),
        ConvertUnits("Voltage [V]", "V [*]"),
        ConvertUnits("Capacity [Ah]", "Q [*]"),
        CapacityFromChDch("Q_ch [*]", "Q_dis [*]"),
        ConvertTemperature("Temp [*]"),
        DateTime("DateTime", "%Y-%m-%d %H:%M:%S%.f"),
        ConvertUnits("Time [s]", "T [*]"),
    ]


def test_basecycler_init(caplog):
    """Test input validation in the BaseCycler class."""
    cycler = BaseCycler(
        input_data_path="tests/sample_data/neware/sample_data_neware.csv",
        output_data_path="tests/sample_data/sample_data1.parquet",
        column_importers=[],
    )
    assert cycler.input_data_path == "tests/sample_data/neware/sample_data_neware.csv"
    assert cycler.output_data_path == "tests/sample_data/sample_data1.parquet"
    assert cycler.column_importers == []
    assert cycler.compression_priority == "performance"
    assert not cycler.overwrite_existing
    assert cycler.header_row_index == 0

    # Test with invalid input_data_path
    with pytest.raises(ValueError, match="Input file not found: invalid_path"):
        BaseCycler(
            input_data_path="invalid_path",
            output_data_path="tests/sample_data/sample_data1.parquet",
            column_importers=[],
        )
    with pytest.raises(
        ValueError, match="No files found matching pattern: invalid_path*"
    ):
        BaseCycler(
            input_data_path="invalid_path*",
            output_data_path="tests/sample_data/sample_data1.parquet",
            column_importers=[],
        )

    # Test with invalid output_data_path
    with pytest.raises(
        ValueError, match="Output directory does not exist: invalid_path"
    ):
        BaseCycler(
            input_data_path="tests/sample_data/neware/sample_data_neware.csv",
            output_data_path="invalid_path/sample_data1.parquet",
            column_importers=[],
        )

    # Test with missing output_data_path
    cycler = BaseCycler(
        input_data_path="tests/sample_data/neware/sample_data_neware.csv",
        column_importers=[],
    )
    assert (
        cycler.output_data_path == "tests/sample_data/neware/sample_data_neware.parquet"
    )

    # Test with missing parquet extension
    with caplog.at_level(logging.INFO):
        cycler = BaseCycler(
            input_data_path="tests/sample_data/neware/sample_data_neware.csv",
            output_data_path="tests/sample_data/sample_data1",
            column_importers=[],
        )
        assert cycler.output_data_path == "tests/sample_data/sample_data1.parquet"
        assert (
            caplog.messages[-1]
            == "Output file has no extension, will be given .parquet"
        )

    # Test with incorrect extension
    with caplog.at_level(logging.WARNING):
        cycler = BaseCycler(
            input_data_path="tests/sample_data/neware/sample_data_neware.csv",
            output_data_path="tests/sample_data/sample_data1.txt",
            column_importers=[],
        )
        assert cycler.output_data_path == "tests/sample_data/sample_data1.parquet"
        assert (
            caplog.messages[-1]
            == "Output file extension .txt will be replaced with .parquet"
        )


def test_extra_column_importers():
    """Test the extra_column_importers method."""
    cycler_instance = BaseCycler(
        input_data_path="tests/sample_data/neware/sample_data_neware.csv",
        output_data_path="tests/sample_data/sample_data.parquet",
        column_importers=[CastAndRename("Step", "Step", pl.Int64)],
        extra_column_importers=[CastAndRename("Cycle", "Cycle", pl.Int64)],
    )
    assert "Cycle" in cycler_instance.get_pyprobe_dataframe().columns


def test_process(mocker, caplog):
    """Test the process method."""
    cycler_instance = BaseCycler(
        input_data_path="tests/sample_data/neware/sample_data_neware.csv",
        output_data_path="tests/sample_data/sample_data.parquet",
        column_importers=[],
    )
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    )
    mocker.patch(
        "pyprobe.cyclers.basecycler.BaseCycler.get_pyprobe_dataframe", return_value=df
    )
    with caplog.at_level(logging.INFO):
        cycler_instance.process()
        message = caplog.messages[-1]
        assert re.match(r"parquet written in \d+(\.\d+)? seconds\.", message)
        assert_frame_equal(pl.read_parquet("tests/sample_data/sample_data.parquet"), df)
        cycler_instance.process()
        assert (
            caplog.messages[-1]
            == "File tests/sample_data/sample_data.parquet already exists. Skipping."
        )
    os.remove("tests/sample_data/sample_data.parquet")

    cycler_instance = BaseCycler(
        input_data_path="tests/sample_data/neware/sample_data_neware.csv",
        output_data_path="tests/sample_data/sample_data.parquet",
        column_importers=[],
        overwrite_existing=True,
    )
    with caplog.at_level(logging.INFO):
        cycler_instance.process()
        message = caplog.messages[-1]
        assert re.match(r"parquet written in \d+(\.\d+)? seconds\.", message)
        assert_frame_equal(pl.read_parquet("tests/sample_data/sample_data.parquet"), df)
        cycler_instance.process()
        assert re.match(r"parquet written in \d+(\.\d+)? seconds\.", message)
    os.remove("tests/sample_data/sample_data.parquet")


def test_get_pyprobe_dataframe(
    sample_dataframe,
    sample_pyprobe_dataframe,
    column_importer_fixture,
):
    """Test the get_pyprobe_dataframe method."""
    sample_dataframe.write_csv("tests/sample_data/sample_data.csv")
    cycler_instance = BaseCycler(
        input_data_path="tests/sample_data/sample_data.csv",
        column_importers=column_importer_fixture,
    )
    pyprobe_dataframe = cycler_instance.get_pyprobe_dataframe()
    assert_frame_equal(
        pyprobe_dataframe,
        sample_pyprobe_dataframe,
        check_column_order=False,
    )
    os.remove("tests/sample_data/sample_data.csv")


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
        result = cycler_instance.get_pyprobe_dataframe()
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
    assert_frame_equal(
        expected_final_row,
        pyprobe_dataframe.tail(1),
        check_column_order=False,
    )
    return pyprobe_dataframe
