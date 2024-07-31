"""Tests for the neware module."""
from datetime import datetime

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
    unprocessed_dataframe = biologic_cycler.imported_dataframe
    assert isinstance(unprocessed_dataframe, pl.LazyFrame)
    start_time = "2024-05-13 11:19:51.602000"
    print(unprocessed_dataframe.select(pl.col("Date")).collect().item(0, 0))
    assert (
        str(unprocessed_dataframe.select(pl.col("Date")).collect().item(0, 0))
        == start_time
    )


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
        return biologic_cycler.pyprobe_dataframe

    pyprobe_dataframe = benchmark(read_and_process)

    expected_columns = [
        "Date",
        "Time [s]",
        "Step",
        "Cycle",
        "Event",
        "Current [A]",
        "Voltage [V]",
        "Capacity [Ah]",
    ]
    assert set(pyprobe_dataframe.columns) == set(expected_columns)
    pyprobe_dataframe = pyprobe_dataframe.with_columns(
        [
            pl.col("Time [s]").diff().fill_null(strategy="zero").alias("dt"),
            pl.col("Date").diff().fill_null(strategy="zero").alias("dd"),
            pl.col("Step").diff().fill_null(strategy="zero").alias("ds"),
        ]
    )
    assert not any(pyprobe_dataframe.select(pl.col("dt") < 0).collect().to_numpy())
    assert not any(pyprobe_dataframe.select(pl.col("dd") < 0).collect().to_numpy())
    assert not any(pyprobe_dataframe.select(pl.col("ds") < 0).collect().to_numpy())
    steps = list(
        pyprobe_dataframe.select(pl.col("Step")).collect().unique().to_numpy().flatten()
    )
    assert set(steps) == set([1, 2, 3, 4, 5, 6])


def test_process_dataframe(monkeypatch):
    """Test the Biologic method."""

    def mock_dataframe(self):
        return pl.DataFrame(
            {
                "Date": [
                    datetime(2022, 2, 2, 2, 2, 0),
                    datetime(2022, 2, 2, 2, 2, 1),
                    datetime(2022, 2, 2, 2, 2, 2),
                    datetime(2022, 2, 2, 2, 2, 3),
                    datetime(2022, 2, 2, 2, 2, 4),
                    datetime(2022, 2, 2, 2, 2, 5),
                    datetime(2022, 2, 2, 2, 2, 6),
                ],
                "time/s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "Ns": [0, 0, 1, 1, 1, 0, 0],
                "I/mA": [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0],
                "Ecell/V": [4.0, 5.0, 6.0, 7.0, 0.0, 0.0, 0.0],
                "Q charge/mA.h": [0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "Q discharge/mA.h": [0.0, 0.0, 10.0, 20.0, 0.0, 0.0, 0.0],
            }
        )

    monkeypatch.setattr(
        "pyprobe.cyclers.biologic.Biologic.imported_dataframe", property(mock_dataframe)
    )
    biologic_cycler = Biologic(
        "tests/sample_data/biologic/Sample_data_biologic_*_MB_CA1.txt"
    )
    pyprobe_dataframe = biologic_cycler.pyprobe_dataframe.select(
        ["Time [s]", "Step", "Current [A]", "Voltage [V]", "Capacity [Ah]"]
    )
    expected_dataframe = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "Step": [1, 1, 2, 2, 2, 1, 1],
            "Current [A]": [1e-3, 2e-3, 3e-3, 4e-3, 0, 0, 0],
            "Voltage [V]": [4.0, 5.0, 6.0, 7.0, 0.0, 0.0, 0.0],
            "Capacity [Ah]": [0.020, 0.040, 0.030, 0.020, 0.020, 0.020, 0.020],
        }
    )
    pl_testing.assert_frame_equal(pyprobe_dataframe, expected_dataframe)


def test_apply_step_correction():
    """Test the apply_step_correction method."""
    df = pl.DataFrame(
        {
            "Ns": [
                0,
                1,
                2,
                3,
                0,
                1,
                2,
                3,
                0,
                1,
                2,
                0,
                1,
                2,
                3,
                4,
                0,
                1,
                0,
                1,
                2,
                3,
                4,
                2,
                3,
                4,
            ],
            "MB File": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
            ],
        }
    )

    expected_step_col = pl.DataFrame(
        {
            "Ns": [
                0,
                1,
                2,
                3,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                9,
                10,
                11,
                12,
                13,
                11,
                12,
                13,
            ]
        }
    )

    step_col = Biologic.apply_step_correction(df).select("Ns")
    pl_testing.assert_frame_equal(step_col, expected_step_col)
