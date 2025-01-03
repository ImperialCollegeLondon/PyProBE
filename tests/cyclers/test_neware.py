"""Tests for the neware module."""
import os
from datetime import datetime

import polars as pl
import polars.testing as pl_testing
import pytest

from pyprobe.cyclers.neware import Neware

from .test_basecycler import helper_read_and_process


@pytest.fixture
def neware_cycler():
    """Create a Neware cycler object."""
    return Neware(input_data_path="tests/sample_data/neware/sample_data_neware.xlsx")


def test_read_file(neware_cycler):
    """Test the read_file method."""
    unprocessed_dataframe = neware_cycler.read_file(
        "tests/sample_data/neware/sample_data_neware.xlsx"
    )
    assert isinstance(unprocessed_dataframe, pl.DataFrame)
    # Test that Time and Total time are read correctly
    expected_start = pl.DataFrame(
        {
            "Total Time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    pl_testing.assert_frame_equal(
        neware_cycler._imported_dataframe.select("Total Time").head(6),
        expected_start,
    )
    assert neware_cycler._imported_dataframe["Total Time"][-1] == 562784.5


def test_sort_files(neware_cycler):
    """Test the _sort_files method."""
    file_list = [
        "test_2_experiment_3_file_5_1.xlsx",
        "test_2_experiment_3_file_5_3.xlsx",
        "test_2_experiment_3_file_5.xlsx",
        "test_2_experiment_3_file_5_2.xlsx",
    ]
    file_list.sort()
    assert file_list == [
        "test_2_experiment_3_file_5.xlsx",
        "test_2_experiment_3_file_5_1.xlsx",
        "test_2_experiment_3_file_5_2.xlsx",
        "test_2_experiment_3_file_5_3.xlsx",
    ]


def test_read_multiple_files(neware_cycler):
    """Test the read_file method with multiple files."""
    unprocessed_dataframe = neware_cycler._imported_dataframe
    assert isinstance(unprocessed_dataframe, pl.DataFrame)


def test_process_dataframe():
    """Test the neware method."""
    mock_dataframe = pl.DataFrame(
        {
            "Date": [
                datetime(2022, 2, 2, 2, 2, 0),
                datetime(2022, 2, 2, 2, 2, 1),
                datetime(2022, 2, 2, 2, 2, 2),
                datetime(2022, 2, 2, 2, 2, 3),
                datetime(2022, 2, 2, 2, 2, 4),
                datetime(2022, 2, 2, 2, 2, 5, 100000),
            ],
            "Total Time": [
                datetime(2022, 2, 2, 2, 2, 0),
                datetime(2022, 2, 2, 2, 2, 1),
                datetime(2022, 2, 2, 2, 2, 2),
                datetime(2022, 2, 2, 2, 2, 3),
                datetime(2022, 2, 2, 2, 2, 4, 100000),
                datetime(2022, 2, 2, 2, 2, 5),
            ],
            "Step Index": [1, 2, 1, 2, 4, 5],
            "Current(mA)": [1, 2, -3, -4, 0, 0],
            "Voltage(V)": [4, 5, 6, 7, 8, 9],
            "Chg. Cap.(mAh)": [
                0,
                20,
                0,
                0,
                0,
                0,
            ],
            "DChg. Cap.(mAh)": [0, 0, 10, 20, 0, 0],
            "Capacity(mAh)": [0, 20, 10, 20, 0, 0],
            "T1(℃)": [25, 25, 25, 25, 25, 25],
        }
    )
    mock_dataframe.write_excel(
        "tests/sample_data/mock_dataframe.xlsx", worksheet="record"
    )
    neware_cycler = Neware(input_data_path="tests/sample_data/mock_dataframe.xlsx")
    pyprobe_dataframe = neware_cycler.pyprobe_dataframe

    pyprobe_dataframe = pyprobe_dataframe.select(
        [
            "Time [s]",
            "Step",
            "Current [A]",
            "Voltage [V]",
            "Capacity [Ah]",
            "Temperature [C]",
        ]
    )
    expected_dataframe = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.1],
            "Step": [1, 2, 1, 2, 4, 5],
            "Current [A]": [1e-3, 2e-3, -3e-3, -4e-3, 0, 0],
            "Voltage [V]": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "Capacity [Ah]": [20.0e-3, 40.0e-3, 30.0e-3, 20.0e-3, 20.0e-3, 20.0e-3],
            "Temperature [C]": [25.0, 25.0, 25.0, 25.0, 25.0, 25.0],
        }
    )
    pl_testing.assert_frame_equal(pyprobe_dataframe, expected_dataframe)
    os.remove("tests/sample_data/mock_dataframe.xlsx")

    # Test with a dataframe that does not contain a Charge or Discharge Capacity column
    mock_dataframe = mock_dataframe.drop("Chg. Cap.(mAh)")
    mock_dataframe = mock_dataframe.drop("DChg. Cap.(mAh)")
    mock_dataframe.write_excel(
        "tests/sample_data/mock_dataframe.xlsx", worksheet="record"
    )
    neware_cycler = Neware(input_data_path="tests/sample_data/mock_dataframe.xlsx")
    pyprobe_dataframe = neware_cycler.pyprobe_dataframe
    pyprobe_dataframe = pyprobe_dataframe.select(
        [
            "Time [s]",
            "Step",
            "Current [A]",
            "Voltage [V]",
            "Capacity [Ah]",
            "Temperature [C]",
        ]
    )
    pl_testing.assert_frame_equal(pyprobe_dataframe, expected_dataframe)

    # Test with a dataframe that does not contain a "Date" column
    mock_dataframe = mock_dataframe.drop("Date")
    mock_dataframe.write_excel(
        "tests/sample_data/mock_dataframe.xlsx", worksheet="record"
    )
    neware_cycler = Neware(input_data_path="tests/sample_data/mock_dataframe.xlsx")
    pyprobe_dataframe = neware_cycler.pyprobe_dataframe
    pyprobe_dataframe = pyprobe_dataframe.select(
        [
            "Time [s]",
            "Step",
            "Current [A]",
            "Voltage [V]",
            "Capacity [Ah]",
            "Temperature [C]",
        ]
    )
    expected_dataframe = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.1, 5.0],
            "Step": [1, 2, 1, 2, 4, 5],
            "Current [A]": [1e-3, 2e-3, -3e-3, -4e-3, 0, 0],
            "Voltage [V]": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "Capacity [Ah]": [20.0e-3, 40.0e-3, 30.0e-3, 20.0e-3, 20.0e-3, 20.0e-3],
            "Temperature [C]": [25.0, 25.0, 25.0, 25.0, 25.0, 25.0],
        }
    )
    pl_testing.assert_frame_equal(pyprobe_dataframe, expected_dataframe)
    os.remove("tests/sample_data/mock_dataframe.xlsx")


def test_read_and_process_neware(benchmark, neware_cycler):
    """Test the full process of reading and processing a file."""
    last_row = pl.DataFrame(
        {
            "Date": [datetime(2024, 3, 6, 21, 39, 38, 591000)],
            "Time [s]": [562749.497],
            "Step": [12],
            "Event": [61],
            "Current [A]": [0.0],
            "Voltage [V]": [3.4513],
            "Capacity [Ah]": [0.022805],
        }
    )
    expected_events = set(range(62))
    expected_columns = [
        "Date",
        "Time [s]",
        "Step",
        "Event",
        "Current [A]",
        "Voltage [V]",
        "Capacity [Ah]",
    ]
    helper_read_and_process(
        benchmark, neware_cycler, last_row, expected_events, expected_columns
    )


def test_read_and_process_neware_multi_file(benchmark):
    """Test the full process of reading and processing multiple Neware files."""
    neware_cycler = Neware(
        input_data_path="tests/sample_data/neware/sample_data_neware*.xlsx"
    )

    last_row = pl.DataFrame(
        {
            "Date": [datetime(2024, 3, 6, 21, 39, 38, 591000)],
            "Time [s]": [562749.497],
            "Step": [12],
            "Event": [123],
            "Current [A]": [0.0],
            "Voltage [V]": [3.4513],
            "Capacity [Ah]": [0.004219859999949997],
        }
    )
    expected_events = set(range(124))
    expected_columns = [
        "Date",
        "Time [s]",
        "Step",
        "Event",
        "Current [A]",
        "Voltage [V]",
        "Capacity [Ah]",
    ]
    helper_read_and_process(
        benchmark, neware_cycler, last_row, expected_events, expected_columns
    )


def test_convert_neware_time_format():
    """Test the _convert_neware_time_format method."""
    data = pl.DataFrame(
        {
            "Total Time": [
                "2022-02-02 02:02:00",
                "2022-02-02 02:02:01",
                "2022-02-02 02:02:02",
                "2022-02-02 02:02:03",
                "2022-02-02 02:02:04",
                "2022-02-02 02:02:05",
            ]
        }
    )
    converted_data = Neware._convert_neware_time_format(data, "Total Time")
    expected_data = pl.DataFrame({"Total Time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
    pl_testing.assert_frame_equal(converted_data, expected_data)

    data = pl.DataFrame(
        {
            "Total Time": [
                "00:00:00",
                "00:00:01",
                "00:00:02",
                "00:00:03",
                "00:00:04",
                "00:00:05",
            ]
        }
    )
    converted_data = Neware._convert_neware_time_format(data, "Total Time")
    expected_data = pl.DataFrame({"Total Time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
    pl_testing.assert_frame_equal(converted_data, expected_data)
