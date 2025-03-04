"""Test the basecycler module."""

import logging
import os
import re

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pyprobe.cyclers.basecycler import BaseCycler
from pyprobe.cyclers.column_importers import (
    CapacityFromChDch,
    CastAndRename,
    ConvertTemperature,
    ConvertUnits,
    DateTime,
)


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
        schema={
            "Date": pl.String,
            "Time [s]": pl.Float64,
            "Step": pl.UInt64,
            "Event": pl.UInt64,
            "Current [A]": pl.Float64,
            "Voltage [V]": pl.Float64,
            "Capacity [Ah]": pl.Float64,
            "Temperature [C]": pl.Float64,
        },
    ).with_columns(pl.col("Date").str.to_datetime())


@pytest.fixture
def column_importer_fixture():
    """A sample column importer."""
    return [
        CastAndRename("Step", "Count", pl.UInt64),
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

    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    )
    df.write_csv("tests/sample_data/sample_basecyclerinit_data1.csv")
    df.write_csv("tests/sample_data/sample_basecyclerinit_data2.csv")
    # Test with wildcards in input_data_path
    cycler = BaseCycler(
        input_data_path="tests/sample_data/sample_basecyclerinit_data*.csv",
        column_importers=[],
    )
    assert (
        cycler.output_data_path
        == "tests/sample_data/sample_basecyclerinit_datax.parquet"
    )
    os.remove("tests/sample_data/sample_basecyclerinit_data1.csv")
    os.remove("tests/sample_data/sample_basecyclerinit_data2.csv")


def test_extra_column_importers():
    """Test the extra_column_importers method."""
    cycler_instance = BaseCycler(
        input_data_path="tests/sample_data/neware/sample_data_neware.csv",
        output_data_path="tests/sample_data/sample_data.parquet",
        column_importers=[CastAndRename("Step", "Step", pl.UInt64)],
        extra_column_importers=[CastAndRename("Cycle", "Cycle", pl.UInt64)],
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
        check_dtypes=False,
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
        check_dtypes=False,
    )
    return pyprobe_dataframe


def test_event_expr():
    """Test the event_expr method."""
    df = pl.DataFrame({"Step": [1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5]})
    expected_df = pl.DataFrame(
        {
            "Step": [1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5],
            "Event": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7],
        }
    )
    assert_frame_equal(
        df.with_columns(BaseCycler.event_expr()), expected_df, check_dtypes=False
    )


def test_get_dataframe_list(caplog):
    """Test the get_dataframe_list method."""
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    )
    df2 = pl.DataFrame(
        {
            "a": [7, 8, 9],
            "b": [10, 11, 12],
        }
    )
    df.write_csv("tests/sample_data/sample_dataframelist_data1.csv")
    df2.write_csv("tests/sample_data/sample_dataframelist_data2.csv")
    cycler = BaseCycler(
        input_data_path="tests/sample_data/sample_dataframelist_data1.csv",
        column_importers=[],
    )
    assert len(cycler._get_dataframe_list()) == 1
    assert_frame_equal(
        cycler._get_dataframe_list()[0],
        df.lazy().with_columns(pl.all().cast(pl.String)),
    )

    cycler = BaseCycler(
        input_data_path="tests/sample_data/sample_dataframelist_data*.csv",
        column_importers=[],
    )
    assert len(cycler._get_dataframe_list()) == 2
    assert_frame_equal(
        cycler._get_dataframe_list()[0],
        df.lazy().with_columns(pl.all().cast(pl.String)),
    )
    assert_frame_equal(
        cycler._get_dataframe_list()[1],
        df2.lazy().with_columns(pl.all().cast(pl.String)),
    )

    df3 = pl.DataFrame(
        {
            "a": [13, 14, 15],
        }
    )
    df3.write_csv("tests/sample_data/sample_dataframelist_data3.csv")
    cycler = BaseCycler(
        input_data_path="tests/sample_data/sample_dataframelist_data*.csv",
        column_importers=[],
    )

    with caplog.at_level(logging.WARNING):
        cycler._get_dataframe_list()
        assert (
            caplog.messages[-1]
            == "File sample_dataframelist_data3.csv has missing columns, "
            "these have been filled with null values."
        )
    os.remove("tests/sample_data/sample_dataframelist_data1.csv")
    os.remove("tests/sample_data/sample_dataframelist_data2.csv")
    os.remove("tests/sample_data/sample_dataframelist_data3.csv")


def test_read_file_csv(tmp_path):
    """Test reading a CSV file."""
    df = pl.DataFrame(
        {
            "header1": ["1", "4"],
            "header2": ["2", "5"],
            "header3": ["3", "6"],
        }
    )
    df.write_csv(tmp_path / "test.csv")
    result = BaseCycler.read_file(str(tmp_path / "test.csv"))
    assert_frame_equal(result, df.lazy().with_columns(pl.all().cast(pl.String)))


def test_read_file_csv_with_header_row(tmp_path):
    """Test reading a CSV file with a custom header row index."""
    # Create a temporary CSV file with a header on the second row
    csv_path = tmp_path / "test_header.csv"
    test_data = "ignore this row\nheader1,header2,header3\n1,2,3\n4,5,6"
    with open(csv_path, "w") as f:
        f.write(test_data)

    # Test reading the file with header_row_index=1
    result = BaseCycler.read_file(str(csv_path), header_row_index=1)
    result = result.collect()

    # Verify the result
    assert result.shape == (2, 3)
    assert result.columns == ["header1", "header2", "header3"]
    assert result["header1"].to_list() == ["1", "4"]


@pytest.mark.parametrize("file_ext", [".txt", ".dat", ".json", ".pdf"])
def test_read_file_unsupported_extension(tmp_path, file_ext):
    """Test reading a file with an unsupported extension."""
    # Create a temporary file with unsupported extension
    test_file = tmp_path / f"test{file_ext}"
    test_file.write_text("some content")

    # Test that reading the file raises a ValueError
    with pytest.raises(ValueError, match=f"Unsupported file extension: {file_ext}"):
        BaseCycler.read_file(str(test_file))


def test_read_file_excel(tmp_path, mocker):
    """Test reading an Excel file."""
    # Mock the polars.read_excel function since creating actual Excel files is complex
    mock_excel = mocker.patch("polars.read_excel")
    mock_df = pl.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    mock_excel.return_value = mock_df

    # Create a simple .xlsx file (content doesn't matter as we're mocking the read)
    excel_path = tmp_path / "test.xlsx"
    with open(excel_path, "wb") as f:
        f.write(b"dummy content")

    # Test reading the Excel file
    result = BaseCycler.read_file(str(excel_path))

    # Verify that read_excel was called with the correct arguments
    mock_excel.assert_called_once_with(
        str(excel_path),
        engine="calamine",
        infer_schema_length=0,
        read_options={"header_row": 0},
    )

    # Verify the result
    assert result is mock_df


def test_read_file_excel_with_header_row(tmp_path, mocker):
    """Test reading an Excel file with a custom header row index."""
    # Mock the polars.read_excel function
    mock_excel = mocker.patch("polars.read_excel")
    mock_df = pl.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    mock_excel.return_value = mock_df

    # Create a simple .xlsx file
    excel_path = tmp_path / "test_header.xlsx"
    with open(excel_path, "wb") as f:
        f.write(b"dummy content")

    # Test reading the Excel file with header_row_index=2
    result = BaseCycler.read_file(str(excel_path), header_row_index=2)

    # Verify that read_excel was called with the correct header_row argument
    mock_excel.assert_called_once_with(
        str(excel_path),
        engine="calamine",
        infer_schema_length=0,
        read_options={"header_row": 2},
    )

    # Verify the result
    assert result is mock_df
