"""Tests for the neware module."""
from datetime import datetime

import polars as pl

from pyprobe.cyclers.neware import process_dataframe, read_file


def test_read_file():
    """Test the read_file method."""
    unprocessed_dataframe = read_file(
        "tests/sample_data_neware/sample_data_neware.xlsx"
    )
    assert isinstance(unprocessed_dataframe, pl.DataFrame)


def test_process_dataframe():
    """Test the neware method."""
    dataframe = pl.DataFrame(
        {
            "Date": [
                datetime(2022, 2, 2, 2, 2, 0),
                datetime(2022, 2, 2, 2, 2, 1),
                datetime(2022, 2, 2, 2, 2, 2),
                datetime(2022, 2, 2, 2, 2, 3),
            ],
            "Cycle Index": [1, 1, 1, 1],
            "Step Index": [1, 2, 3, 4],
            "Current(mA)": [1, 2, 3, 4],
            "Voltage(V)": [4, 5, 6, 7],
            "Chg. Cap.(Ah)": [0, 20, 0, 0],
            "DChg. Cap.(Ah)": [0, 0, 10, 20],
        }
    )
    processed_dataframe = process_dataframe(dataframe)
    processed_dataframe = processed_dataframe.select(
        ["Time [s]", "Cycle", "Step", "Current [A]", "Voltage [V]", "Capacity [Ah]"]
    )
    expected_dataframe = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0],
            "Cycle": [1, 1, 1, 1],
            "Step": [1, 2, 3, 4],
            "Current [A]": [1e-3, 2e-3, 3e-3, 4e-3],
            "Voltage [V]": [4, 5, 6, 7],
            "Capacity [Ah]": [20, 40, 30, 20],
        }
    )

    pl.testing.assert_frame_equal(processed_dataframe, expected_dataframe)
