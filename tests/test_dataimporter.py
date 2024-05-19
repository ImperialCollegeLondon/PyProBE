"""Tests for the DataImporter module."""
import polars as pl
import pytest

from pyprobe.dataimporter import DataImporter


@pytest.fixture
def neware_dataimporter():
    """Pytest fixture for a neware DataImporter object."""
    return DataImporter("neware")


@pytest.fixture
def unprocessed_dataframe(neware_dataimporter):
    """Pytest fixture for an unprocessed DataFrame."""
    return neware_dataimporter.read_file(
        "tests/sample_data_neware/sample_data_neware.xlsx"
    )


def test_read_file(unprocessed_dataframe):
    """Test the read_file method."""
    assert isinstance(unprocessed_dataframe, pl.DataFrame)


def test_process_dataframe(
    neware_dataimporter, unprocessed_dataframe, lazyframe_fixture
):
    """Test the process_dataframe method."""
    processed_dataframe = neware_dataimporter.process_dataframe(unprocessed_dataframe)
    pl.testing.assert_frame_equal(processed_dataframe, lazyframe_fixture.collect())
