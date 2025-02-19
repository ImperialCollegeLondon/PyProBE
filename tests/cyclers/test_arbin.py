"""Tests for the Arbin cycler class."""

from datetime import datetime

import polars as pl

from pyprobe.cyclers.arbin import Arbin

from .test_basecycler import helper_read_and_process


def test_read_and_process_arbin(benchmark):
    """Test the full process of reading and processing a file."""
    arbin_cycler = Arbin(
        input_data_path="tests/sample_data/arbin/sample_data_arbin.csv",
    )
    expected_df = pl.DataFrame(
        {
            "Date": [datetime(2024, 9, 20, 8, 37, 5, 772000)],
            "Time [s]": [301.214],
            "Step": [3],
            "Event": [2],
            "Current [A]": [2.650138],
            "Voltage [V]": [3.599601],
            "Capacity [Ah]": [0.0007812400999999999],
            "Temperature [C]": [24.68785],
        },
    )
    expected_events = set([0, 1, 2])
    helper_read_and_process(
        benchmark,
        arbin_cycler,
        expected_final_row=expected_df,
        expected_events=expected_events,
    )
