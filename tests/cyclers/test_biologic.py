"""Tests for the neware module."""
from datetime import datetime

import numpy as np
import polars as pl
import polars.testing as pl_testing
import pytest

from pyprobe.cyclers.biologic import Biologic, BiologicMB

from .test_basecycler import helper_read_and_process


@pytest.fixture
def biologic_MB_cycler():
    """Create a Biologic cycler object."""
    return BiologicMB(
        input_data_path="tests/sample_data/biologic/Sample_data_biologic_*_MB_CA1.txt"
    )


@pytest.fixture
def biologic_cycler():
    """Create a Biologic cycler object."""
    return Biologic(
        input_data_path="tests/sample_data/biologic/Sample_data_biologic_CA1.txt"
    )


def test_read_file(biologic_cycler, biologic_MB_cycler):
    """Test the read_file method."""
    unprocessed_dataframe = biologic_cycler._imported_dataframe
    assert isinstance(unprocessed_dataframe, pl.LazyFrame)
    start_time = "2024-05-13 11:19:51.602000"
    assert (
        str(unprocessed_dataframe.select(pl.col("Date")).collect().item(0, 0))
        == start_time
    )

    unprocessed_dataframe = biologic_MB_cycler._imported_dataframe
    assert isinstance(unprocessed_dataframe, pl.LazyFrame)
    start_time = "2024-05-13 11:19:51.602000"
    assert (
        str(unprocessed_dataframe.select(pl.col("Date")).collect().item(0, 0))
        == start_time
    )


def test_read_file_timestamp():
    """Test the read_file method."""
    unprocessed_dataframe = Biologic(
        input_data_path="tests/sample_data/biologic/"
        "Sample_data_biologic_timestamped.txt"
    )._imported_dataframe
    assert isinstance(unprocessed_dataframe, pl.LazyFrame)
    time_s = (
        unprocessed_dataframe.select(pl.col("time/s"))
        .cast(pl.Float64)
        .collect()
        .to_numpy()
    )
    np.testing.assert_allclose(
        time_s.flatten(),
        np.array([0, 6.464, 7.464, 8.464, 9.464, 10.464, 11.464, 12.464]),
    )


def test_sort_files():
    """Test the _sort_files method."""
    file_list = [
        "test_2_experiment_3_03_MB_file.xlsx",
        "test_2_experiment_3_01_MB_file.xlsx",
        "test_2_experiment_3_04_MB_file.xlsx",
        "test_2_experiment_3_02_MB_file.xlsx",
    ]
    file_list.sort()
    assert file_list == [
        "test_2_experiment_3_01_MB_file.xlsx",
        "test_2_experiment_3_02_MB_file.xlsx",
        "test_2_experiment_3_03_MB_file.xlsx",
        "test_2_experiment_3_04_MB_file.xlsx",
    ]


def test_read_and_process_biologic(benchmark, biologic_cycler):
    """Test the full process of reading and processing a file."""
    last_row = pl.DataFrame(
        {
            "Date": [datetime(2024, 5, 13, 11, 19, 51, 602139)],
            "Time [s]": [139.524007],
            "Step": [1],
            "Event": [1],
            "Current [A]": [-0.899826],
            "Voltage [V]": [3.4854481],
            "Capacity [Ah]": [-0.03237135133365209],
            "Temperature [C]": [23.029291],
        }
    )
    pyprobe_dataframe = helper_read_and_process(
        benchmark,
        biologic_cycler,
        expected_final_row=last_row,
        expected_events=set([0, 1]),
    )
    pyprobe_dataframe = pyprobe_dataframe.with_columns(
        [
            pl.col("Time [s]").diff().fill_null(strategy="zero").alias("dt"),
            pl.col("Date").diff().fill_null(strategy="zero").alias("dd"),
            pl.col("Step").diff().fill_null(strategy="zero").alias("ds"),
        ]
    )
    assert not any(pyprobe_dataframe.select(pl.col("dt") < 0).to_numpy())
    assert not any(pyprobe_dataframe.select(pl.col("dd") < 0).to_numpy())
    assert not any(pyprobe_dataframe.select(pl.col("ds") < 0).to_numpy())


def test_read_and_process_biologic_MB(benchmark, biologic_MB_cycler):
    """Test the full process of reading and processing modulo bat files."""
    last_row = pl.DataFrame(
        {
            "Date": [datetime(2024, 5, 13, 11, 19, 51, 858016)],
            "Time [s]": [256016.11344],
            "Step": [5],
            "Event": [5],
            "Current [A]": [0.450135],
            "Voltage [V]": [3.062546],
            "Capacity [Ah]": [0.307727],
            "Temperature [C]": [22.989878],
        }
    )
    pyprobe_dataframe = helper_read_and_process(
        benchmark,
        biologic_MB_cycler,
        expected_final_row=last_row,
        expected_events=set([0, 1, 2, 3, 4, 5]),
    )
    pyprobe_dataframe = pyprobe_dataframe.with_columns(
        [
            pl.col("Time [s]").diff().fill_null(strategy="zero").alias("dt"),
            pl.col("Date").diff().fill_null(strategy="zero").alias("dd"),
            pl.col("Step").diff().fill_null(strategy="zero").alias("ds"),
        ]
    )
    assert not any(pyprobe_dataframe.select(pl.col("dt") < 0).to_numpy())
    assert not any(pyprobe_dataframe.select(pl.col("dd") < 0).to_numpy())
    assert not any(pyprobe_dataframe.select(pl.col("ds") < 0).to_numpy())


def test_read_and_process_biologic_no_header(benchmark):
    """Test reading a Biologic file without a header."""
    cycler = Biologic(
        input_data_path="tests/sample_data/biologic/Sample_data_biologic_no_header.mpt"
    )
    last_row = pl.DataFrame(
        {
            "Time [s]": [281792.50213],
            "Step": [0],
            "Event": [0],
            "Current [A]": [0.0],
            "Voltage [V]": [2.9814022],
            "Capacity [Ah]": [0.0],
            "Temperature [C]": [24.506462],
        }
    )
    helper_read_and_process(
        benchmark,
        cycler,
        expected_final_row=last_row,
        expected_events=set([0]),
        expected_columns=[
            "Time [s]",
            "Step",
            "Event",
            "Current [A]",
            "Voltage [V]",
            "Capacity [Ah]",
            "Temperature [C]",
        ],
    )


def test_process_dataframe(monkeypatch):
    """Test the Biologic method."""
    mock_dataframe = pl.DataFrame(
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
            "Temperature/�C": [25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0],
        }
    )

    biologic_cycler = Biologic(
        input_data_path="tests/sample_data/biologic/Sample_data_biologic_CA1.txt"
    )
    biologic_cycler._imported_dataframe = mock_dataframe
    pyprobe_dataframe = biologic_cycler.pyprobe_dataframe.select(
        [
            "Time [s]",
            "Step",
            "Current [A]",
            "Voltage [V]",
            "Capacity [Ah]",
            "Temperature [C]",
        ]
    )
    expected_dataframe = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "Step": [0, 0, 1, 1, 1, 0, 0],
            "Current [A]": [1e-3, 2e-3, 3e-3, 4e-3, 0, 0, 0],
            "Voltage [V]": [4.0, 5.0, 6.0, 7.0, 0.0, 0.0, 0.0],
            "Capacity [Ah]": [0.020, 0.040, 0.030, 0.020, 0.020, 0.020, 0.020],
            "Temperature [C]": [25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0],
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

    step_col = BiologicMB.apply_step_correction(df).select("Ns")
    pl_testing.assert_frame_equal(step_col, expected_step_col)
