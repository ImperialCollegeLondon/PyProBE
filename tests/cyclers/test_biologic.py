"""Tests for the neware module."""

import polars as pl

from pyprobe.cyclers.biologic import process_dataframe, read_file


def test_read_file():
    """Test the read_file method."""
    unprocessed_dataframe = read_file("tests/sample_data_biologic/interim.mpt")
    assert isinstance(unprocessed_dataframe, pl.DataFrame)


def test_process_dataframe():
    """Test the Biologic method."""
    dataframe = pl.DataFrame(
        {
            "time/s": [0.0, 1.0, 2.0, 3.0],
            "cycle number": [1, 1, 1, 1],
            "counter inc.": [1, 1, 1, 1],
            "Ns": [1, 2, 3, 4],
            "I/mA": [1, 2, 3, 4],
            "Ecell/V": [4, 5, 6, 7],
            "Q charge/mA.h": [0, 20, 0, 0],
            "Q discharge/mA.h": [0, 0, 10, 20],
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
            "Capacity [Ah]": [0.020, 0.040, 0.030, 0.020],
        }
    )

    pl.testing.assert_frame_equal(processed_dataframe, expected_dataframe)
