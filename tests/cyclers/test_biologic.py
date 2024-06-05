"""Tests for the neware module."""

import polars as pl
import polars.testing as pl_testing
import pytest

from pyprobe.cyclers.biologic import Biologic


@pytest.fixture
def biologic_cycler():
    """Create a Biologic cycler object."""
    return Biologic("tests/sample_data/biologic/Sample_data_biologic_*_MB_CA1.txt")


def test_read_file(biologic_cycler):
    """Test the read_file method."""
    unprocessed_dataframe = biologic_cycler.raw_dataframe
    assert isinstance(unprocessed_dataframe, pl.DataFrame)
    assert unprocessed_dataframe.columns == [
        "Ns changes",
        "Ns",
        "time/s",
        "Ecell/V",
        "I/mA",
        "step time/s",
        "Q discharge/mA.h",
        "Q charge/mA.h",
        "Date",
    ]


def test_sort_files(biologic_cycler):
    """Test the sort_files method."""
    file_list = [
        "test_2_experiment_3_03_MB_file.xlsx",
        "test_2_experiment_3_01_MB_file.xlsx",
        "test_2_experiment_3_04_MB_file.xlsx",
        "test_2_experiment_3_02_MB_file.xlsx",
    ]
    sorted_files = biologic_cycler.sort_files(file_list)
    assert sorted_files == [
        "test_2_experiment_3_01_MB_file.xlsx",
        "test_2_experiment_3_02_MB_file.xlsx",
        "test_2_experiment_3_03_MB_file.xlsx",
        "test_2_experiment_3_04_MB_file.xlsx",
    ]


def test_read_and_process(benchmark, biologic_cycler):
    """Test the full process of reading and processing a file."""

    def read_and_process():
        return biologic_cycler.processed_dataframe

    processed_dataframe = benchmark(read_and_process)
    expected_columns = [
        "Date",
        "Time [s]",
        "Step",
        "Current [A]",
        "Voltage [V]",
        "Capacity [Ah]",
    ]
    assert isinstance(processed_dataframe, pl.DataFrame)
    assert all(col in processed_dataframe.columns for col in expected_columns)
    processed_dataframe = processed_dataframe.with_columns(
        [
            pl.col("Time [s]").diff().fill_null(strategy="zero").alias("dt"),
            pl.col("Date").diff().fill_null(strategy="zero").alias("dd"),
        ]
    )
    assert not any(processed_dataframe.select(pl.col("dt") < 0).to_numpy())
    assert not any(processed_dataframe.select(pl.col("dd") < 0).to_numpy())


def test_process_dataframe(monkeypatch):
    """Test the Biologic method."""

    def mock_dataframe(self):
        return pl.DataFrame(
            {
                "Date": [0, 1, 2, 3],
                "time/s": [0.0, 1.0, 2.0, 3.0],
                "Ns": [0, 1, 2, 3],
                "I/mA": [1, 2, 3, 4],
                "Ecell/V": [4, 5, 6, 7],
                "Q charge/mA.h": [0, 20, 0, 0],
                "Q discharge/mA.h": [0, 0, 10, 20],
            }
        )

    monkeypatch.setattr(
        "pyprobe.cyclers.biologic.Biologic.raw_dataframe", property(mock_dataframe)
    )
    biologic_cycler = Biologic(
        "tests/sample_data/biologic/Sample_data_biologic_*_MB_CA1.txt"
    )
    processed_dataframe = biologic_cycler.processed_dataframe.select(
        ["Time [s]", "Step", "Current [A]", "Voltage [V]", "Capacity [Ah]"]
    )
    expected_dataframe = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0],
            "Step": [1, 2, 3, 4],
            "Current [A]": [1e-3, 2e-3, 3e-3, 4e-3],
            "Voltage [V]": [4, 5, 6, 7],
            "Capacity [Ah]": [0.020, 0.040, 0.030, 0.020],
        }
    )

    pl_testing.assert_frame_equal(processed_dataframe, expected_dataframe)
