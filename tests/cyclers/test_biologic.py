"""Tests for the neware module."""

import polars as pl
import polars.testing as pl_testing

from pyprobe.cyclers.biologic import process_dataframe, read_all_files, sort_files


def test_read_file():
    """Test the read_file method."""
    unprocessed_dataframe = read_all_files(
        "tests/sample_data_biologic/Sample_data_biologic_*_MB_CA1.txt"
    )
    assert isinstance(unprocessed_dataframe, pl.DataFrame)


def test_sort_files():
    """Test the sort_files method."""
    file_list = [
        "test_2_experiment_3_03_MB_file.xlsx",
        "test_2_experiment_3_01_MB_file.xlsx",
        "test_2_experiment_3_04_MB_file.xlsx",
        "test_2_experiment_3_02_MB_file.xlsx",
    ]
    sorted_files = sort_files(file_list)
    assert sorted_files == [
        "test_2_experiment_3_01_MB_file.xlsx",
        "test_2_experiment_3_02_MB_file.xlsx",
        "test_2_experiment_3_03_MB_file.xlsx",
        "test_2_experiment_3_04_MB_file.xlsx",
    ]


def test_read_and_process(benchmark):
    """Test the full process of reading and processing a file."""

    def read_and_process():
        unprocessed_dataframe = read_all_files(
            "tests/sample_data_biologic/Sample_data_biologic_*_MB_CA1.txt"
        )
        processed_dataframe = process_dataframe(unprocessed_dataframe)
        return processed_dataframe

    processed_dataframe = benchmark(read_and_process)
    expected_columns = [
        "Date",
        "Time [s]",
        "Cycle",
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


def test_process_dataframe():
    """Test the Biologic method."""
    dataframe = pl.DataFrame(
        {
            "Date": [0, 1, 2, 3],
            "time/s": [0.0, 1.0, 2.0, 3.0],
            "cycle number": [1, 1, 1, 1],
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

    pl_testing.assert_frame_equal(processed_dataframe, expected_dataframe)
