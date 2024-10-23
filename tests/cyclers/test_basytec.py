"""Tests for the Basytec cycler class."""

from datetime import datetime

import polars as pl

from pyprobe.cyclers.basytec import Basytec

from .test_basecycler import helper_read_and_process


def test_read_file_basytec():
    """Test reading a Basytec file."""
    dataframe = Basytec.read_file(
        "tests/sample_data/basytec/sample_data_basytec.txt"
    ).collect()
    assert "Date" in dataframe.columns
    assert dataframe["Date"][0] == "2023-06-19 17:56:53.000000"
    assert dataframe["Date"][2] == "2023-06-19 17:56:54.002823"


def test_read_and_process_basytec(benchmark):
    """Test the full process of reading and processing a file."""
    basytec_cycler = Basytec(
        input_data_path="tests/sample_data/basytec/sample_data_basytec.txt"
    )
    last_row = pl.DataFrame(
        {
            "Date": datetime(2023, 6, 19, 17, 58, 3, 235803),
            "Time [s]": [70.235804],
            "Step": [4],
            "Event": [1],
            "Current [A]": [0.449602],
            "Voltage [V]": [3.53285],
            "Capacity [Ah]": [0.001248916998009],
            "Temperature [C]": [25.47953],
        }
    )
    helper_read_and_process(
        benchmark,
        basytec_cycler,
        expected_final_row=last_row,
        expected_events=set([0, 1]),
    )
