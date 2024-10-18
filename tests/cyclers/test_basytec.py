"""Tests for the Basytec cycler class."""

from datetime import datetime

import polars as pl
from polars.testing import assert_frame_equal

from pyprobe.cyclers.basytec import Basytec


def test_read_file_basytec():
    """Test reading a Basytec file."""
    dataframe = Basytec.read_file(
        "tests/sample_data/basytec/sample_data_basytec.txt"
    ).collect()
    assert "Date" in dataframe.columns
    assert dataframe["Date"][0] == "2023-06-19 17:56:53.000000"
    assert dataframe["Date"][2] == "2023-06-19 17:56:54.002823"


def test_read_and_process_basytec():
    """Test the full process of reading and processing a file."""
    basytec_cycler = Basytec(
        input_data_path="tests/sample_data/basytec/sample_data_basytec.txt"
    )
    pyprobe_dataframe = basytec_cycler.pyprobe_dataframe
    expected_columns = [
        "Date",
        "Time [s]",
        "Step",
        "Cycle",
        "Event",
        "Current [A]",
        "Voltage [V]",
        "Capacity [Ah]",
        "Temperature [C]",
    ]
    assert set(pyprobe_dataframe.columns) == set(expected_columns)
    last_row = pl.LazyFrame(
        {
            "Date": datetime(2023, 6, 19, 17, 58, 3, 235803),
            "Time [s]": [70.235804],
            "Cycle": [0],
            "Step": [4],
            "Event": [1],
            "Current [A]": [0.449602],
            "Voltage [V]": [3.53285],
            "Capacity [Ah]": [0.001248916998009],
            "Temperature [C]": [25.47953],
        }
    )
    assert_frame_equal(pyprobe_dataframe.tail(1), last_row)
