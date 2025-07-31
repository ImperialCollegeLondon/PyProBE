"""Tests for the Basytec cycler class."""

from datetime import datetime

import polars as pl

from pyprobe.cyclers.novonix import Novonix

from .test_basecycler import helper_read_and_process


def test_read_file_novonix():
    """Test reading a Basytec file."""
    dataframe = Novonix.read_file(
        "tests/sample_data/novonix/Novonix_Test.csv",
    )


def test_read_and_process_novonix(benchmark):
    """Test the full process of reading and processing a file."""
    novonix_cycler = Novonix(
        input_data_path="tests/sample_data/novonix/Novonix_Test.csv",
    )
    last_row = pl.DataFrame(
        {
            "Date": datetime(2025, 7, 19, 18, 51, 8),
            "Time [s]": [12287.48004],
            "Step": [1],
            "Event": [0],
            "Current [A]": [0.49999387],
            "Voltage [V]": [4.12864581],
            "Capacity [Ah]": [1.70652976],
            "Temperature [C]": [24.792],
        },
    )
    helper_read_and_process(
        benchmark,
        novonix_cycler,
        expected_final_row=last_row,
        expected_events={0},
    )
