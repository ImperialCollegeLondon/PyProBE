"""Tests for the BaseCycler class."""

import polars as pl
import polars.testing as pl_testing

from pyprobe.cyclers.basecycler import BaseCycler


def test_get_cycle_and_event():
    """Test the get_cycle_and_event method."""
    dataframe = pl.DataFrame(
        {
            "Step": [1, 2, 3, 1, 2, 3, 5, 6, 7],
        }
    )
    new_dataframe = BaseCycler.get_cycle_and_event(dataframe)
    expected_dataframe = pl.DataFrame(
        {
            "Step": [1, 2, 3, 1, 2, 3, 5, 6, 7],
            "Cycle": [0, 0, 0, 1, 1, 1, 1, 1, 1],
            "Event": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    pl_testing.assert_frame_equal(new_dataframe, expected_dataframe)
