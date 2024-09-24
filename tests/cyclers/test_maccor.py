"""Tests for the Maccor cycler class."""
from datetime import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pyprobe.cyclers.maccor import Maccor


def test_read_and_process_maccor():
    """Test reading and processing a sample Maccor file."""
    maccor_cycler = Maccor(
        input_data_path="tests/sample_data/maccor/sample_data_maccor.csv"
    )
    pyprobe_dataframe = maccor_cycler.pyprobe_dataframe
    expected_columns = [
        "Date",
        "Time [s]",
        "Step",
        "Cycle",
        "Event",
        "Current [A]",
        "Voltage [V]",
        "Capacity [Ah]",
        "Temperature [C]",
    ]
    assert set(pyprobe_dataframe.columns) == set(expected_columns)
    last_row = pl.LazyFrame(
        {
            "Date": datetime(2023, 11, 23, 15, 56, 24, 60000),
            "Time [s]": [13.06],
            "Cycle": [0],
            "Step": [2],
            "Event": [1],
            "Current [A]": [28.798],
            "Voltage [V]": [3.716],
            "Capacity [Ah]": [0.048],
            "Temperature [C]": [22.2591],
        }
    )
    assert_frame_equal(pyprobe_dataframe.tail(1), last_row)


@pytest.fixture
def sample_dataframe():
    """Return a sample DataFrame."""
    return pl.DataFrame(
        {
            "Date": [
                "01-Jan-21 11:00:00 PM",
                "01-Jan-21 11:00:01 PM",
                "01-Jan-21 11:00:02 PM",
                "01-Jan-21 11:00:03 PM",
                "01-Jan-21 11:00:04 PM",
                "01-Jan-21 11:00:05 PM",
            ],
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.1],
            "Capacity [Ah]": [1, 2, 3, 4, 5, 6],
            "Current [A]": [1, 1, 0, -1, -1, 0],
        }
    )


def test_date(sample_dataframe):
    """Test the date property."""
    maccor_cycler = Maccor(
        input_data_path="tests/sample_data/maccor/sample_data_maccor.csv"
    )
    date = maccor_cycler.date
    expected_dataframe = pl.DataFrame(
        {
            "Date": [
                "2021-01-01 23:00:00.0",
                "2021-01-01 23:00:01.0",
                "2021-01-01 23:00:02.0",
                "2021-01-01 23:00:03.0",
                "2021-01-01 23:00:04.0",
                "2021-01-01 23:00:05.1",
            ]
        }
    ).with_columns(
        pl.col("Date").str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f", time_unit="us")
    )
    assert_frame_equal(sample_dataframe.select(date), expected_dataframe)


def test_charge_capacity(sample_dataframe):
    """Test the charge_capacity property."""
    maccor_cycler = Maccor(
        input_data_path="tests/sample_data/maccor/sample_data_maccor.csv"
    )
    charge_capacity = maccor_cycler.charge_capacity
    assert_frame_equal(
        sample_dataframe.select(charge_capacity),
        pl.DataFrame({"Charge Capacity [Ah]": [1, 2, 0, 0, 0, 0]}),
    )


def test_discharge_capacity(sample_dataframe):
    """Test the discharge_capacity property."""
    maccor_cycler = Maccor(
        input_data_path="tests/sample_data/maccor/sample_data_maccor.csv"
    )
    discharge_capacity = maccor_cycler.discharge_capacity
    assert_frame_equal(
        sample_dataframe.select(discharge_capacity),
        pl.DataFrame({"Discharge Capacity [Ah]": [0, 0, 0, 4, 5, 0]}),
    )


def test_capacity(sample_dataframe):
    """Test the capacity property."""
    maccor_cycler = Maccor(
        input_data_path="tests/sample_data/maccor/sample_data_maccor.csv"
    )
    capacity = maccor_cycler.capacity
    dataframe = pl.DataFrame(
        {
            "Capacity [Ah]": [0.0, 20.0, 10.0, 20.0, 20.0, 20.0],
            "Current [A]": [0.0, 3.0, -2.0, -2.0, 0.0, 0.0],
        }
    )
    assert_frame_equal(
        dataframe.select(capacity),
        pl.DataFrame({"Capacity [Ah]": [20.0, 40.0, 30.0, 20.0, 20.0, 20.0]}),
    )
