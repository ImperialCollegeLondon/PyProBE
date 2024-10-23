"""Tests for the Arbin cycler class."""

from datetime import datetime

import polars as pl
from polars.testing import assert_frame_equal

from pyprobe.cyclers.arbin import Arbin


def test_read_and_process_arbin(benchmark):
    """Test the full process of reading and processing a file."""
    arbin_cycler = Arbin(
        input_data_path="tests/sample_data/arbin/sample_data_arbin.csv"
    )

    def read_and_process_arbin():
        return arbin_cycler.pyprobe_dataframe.collect()

    pyprobe_dataframe = benchmark(read_and_process_arbin)
    expected_columns = [
        "Date",
        "Time [s]",
        "Step",
        "Event",
        "Current [A]",
        "Voltage [V]",
        "Capacity [Ah]",
        "Temperature [C]",
    ]
    assert set(pyprobe_dataframe.columns) == set(expected_columns)
    assert set(pyprobe_dataframe.select("Event").unique().to_series().to_list()) == set(
        [0, 1, 2]
    )
    expected_df = pl.DataFrame(
        {
            "Date": [datetime(2024, 9, 20, 8, 32, 34, 558000)],
            "Time [s]": [30.0005],
            "Step": [1],
            "Event": [0],
            "Current [A]": [0.0],
            "Voltage [V]": [3.534595],
            "Capacity [Ah]": [0.000400839],
            "Temperature [C]": [24.66422],
        }
    )
    assert_frame_equal(expected_df, pyprobe_dataframe.head(1))
