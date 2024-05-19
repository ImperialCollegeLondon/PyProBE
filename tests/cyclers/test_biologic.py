"""Tests for the neware module."""

import polars as pl

from pyprobe.cyclers.biologic import read_file


def test_read_file():
    """Test the read_file method."""
    unprocessed_dataframe = read_file(
        "tests/sample_data_biologic/interim_sample_data_biologic.mpt"
    )
    print(unprocessed_dataframe)
    assert isinstance(unprocessed_dataframe, pl.DataFrame)
    assert 0 == 1
