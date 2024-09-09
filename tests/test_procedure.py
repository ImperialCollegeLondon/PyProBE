"""Module containing tests of the procedure class."""

import os

import numpy as np
import pandas as pd
import polars as pl


def test_experiment(procedure_fixture, cycles_fixture, steps_fixture, benchmark):
    """Test creating an experiment."""

    def make_experiment():
        return procedure_fixture.experiment("Break-in Cycles")

    experiment = benchmark(make_experiment)
    assert experiment.data["Cycle"].unique().to_list() == cycles_fixture[1]
    assert experiment.data["Step"].unique().to_list() == steps_fixture[1]

    experiment = procedure_fixture.experiment("Discharge Pulses")
    assert experiment.data["Cycle"].unique().to_list() == cycles_fixture[2]
    assert experiment.data["Step"].unique().to_list() == steps_fixture[2]

    """Test filtering by multiple experiment names."""
    experiment = procedure_fixture.experiment("Break-in Cycles", "Discharge Pulses")
    assert set(experiment.data["Cycle"].unique().to_list()) == set(
        cycles_fixture[1] + cycles_fixture[2]
    )
    assert set(experiment.data["Step"].unique().to_list()) == set(
        steps_fixture[1] + steps_fixture[2]
    )

    assert experiment.data["Experiment Time [s]"][0] == 0
    assert experiment.data["Experiment Capacity [Ah]"][0] == 0


def test_experiment_names(procedure_fixture, titles_fixture):
    """Test the experiment_names method."""
    assert procedure_fixture.experiment_names == titles_fixture


def test_flatten(procedure_fixture):
    """Test flattening lists."""
    lst = [[1, 2, 3], [4, 5], 6]
    flat_list = procedure_fixture._flatten(lst)
    assert flat_list == [1, 2, 3, 4, 5, 6]


def test_zero_columns(procedure_fixture):
    """Test methods to set the first value of columns to zero."""
    assert procedure_fixture.data["Procedure Time [s]"][0] == 0
    assert procedure_fixture.data["Procedure Capacity [Ah]"][0] == 0


def test_add_external_data(procedure_fixture):
    """Test adding external data to the procedure."""
    # Create external data
    data = pl.read_excel("tests/sample_data/neware/sample_data_neware.xlsx").to_pandas()
    start_date = data["Date"][0] - pd.Timedelta(seconds=30.54)
    end_date = data["Date"].iloc[-1] - pd.Timedelta(seconds=67.54)
    date_range = pd.date_range(start=start_date, end=end_date, freq="1min")
    seconds_passed = (date_range - start_date).total_seconds()
    value = 10 * np.sin(0.001 * seconds_passed)
    dataframe = pl.DataFrame({"Date": date_range, "Value": value})
    dataframe.write_csv("tests/sample_data/neware/external_data.csv")

    procedure_fixture.add_external_data(
        filepath="tests/sample_data/neware/external_data.csv",
        importing_columns=["Value"],
        date_column_name="Date",
    )
    assert "Value" in procedure_fixture.column_list
    assert procedure_fixture.data.select(
        pl.col("Value").tail(69).is_null()
    ).unique().to_numpy() == np.array([True])

    procedure_fixture.add_external_data(
        filepath="tests/sample_data/neware/external_data.csv",
        importing_columns={"Value": "new column"},
    )
    assert "new column" in procedure_fixture.column_list

    time = procedure_fixture.data["Time [s]"].to_numpy() + 30.54
    value = 10 * np.sin(0.001 * time)
    data = procedure_fixture.data["new column"].to_numpy()
    nan_mask = np.isnan(data)

    # Filter out NaNs
    value = value[~nan_mask]
    data = data[~nan_mask]
    assert np.allclose(data, value, atol=0.005)

    os.remove("tests/sample_data/neware/external_data.csv")
