"""Tests for the Maccor cycler class."""

from datetime import datetime

import polars as pl

from pyprobe.cyclers.maccor import Maccor

from .test_basecycler import helper_read_and_process


def test_read_and_process_maccor(benchmark):
    """Test reading and processing a sample Maccor file."""
    maccor_cycler = Maccor(
        input_data_path="tests/sample_data/maccor/sample_data_maccor.csv"
    )
    last_row = pl.DataFrame(
        {
            "Date": datetime(2023, 11, 23, 15, 56, 24, 60000),
            "Time [s]": [13.06],
            "Step": [2],
            "Event": [1],
            "Current [A]": [28.798],
            "Voltage [V]": [3.716],
            "Capacity [Ah]": [0.048],
            "Temperature [C]": [22.2591],
        }
    )
    helper_read_and_process(
        benchmark,
        maccor_cycler,
        expected_final_row=last_row,
        expected_events=set([0, 1]),
    )
