"""Module containing tests of the procedure class."""

import copy

import numpy as np
import pandas as pd
import polars as pl
import pytest

from pyprobe.cell import Cell


def test_experiment(procedure_fixture, steps_fixture, benchmark):
    """Test creating an experiment."""

    def make_experiment():
        return procedure_fixture.experiment("Break-in Cycles")

    experiment = benchmark(make_experiment)
    assert experiment.data["Step"].unique().to_list() == steps_fixture[1]
    assert experiment.cycle_info == [(4, 7, 5)]

    experiment = procedure_fixture.experiment("Discharge Pulses")
    assert experiment.data["Step"].unique().to_list() == steps_fixture[2]
    assert experiment.cycle_info == [(9, 12, 10)]

    """Test filtering by multiple experiment names."""
    with pytest.warns(UserWarning):
        experiment = procedure_fixture.experiment("Break-in Cycles", "Discharge Pulses")

    assert experiment.data["Experiment Time [s]"][0] == 0
    assert experiment.data["Experiment Capacity [Ah]"][0] == 0
    assert experiment.cycle_info == []


def test_remove_experiment(procedure_fixture):
    """Test removing an experiment."""
    procedure_fixture.remove_experiment("Break-in Cycles")
    assert "Break-in Cycles" not in procedure_fixture.experiment_names
    assert procedure_fixture.data["Step"].unique().to_list() == [2, 3, 9, 10, 11, 12]
    assert procedure_fixture.step_descriptions["Step"] == [1, 2, 3, 9, 10, 11, 12]


def test_init(procedure_fixture, step_descriptions_fixture):
    """Test initialising a procedure."""
    assert procedure_fixture.step_descriptions == step_descriptions_fixture


def test_experiment_no_description():
    """Test creating a procedure with no step descriptions."""
    cell = Cell(info={})
    cell.add_procedure(
        "sample",
        "tests/sample_data/neware/",
        "sample_data_neware.parquet",
        readme_name="README_total_steps.yaml",
    )
    assert np.all(np.isnan(cell.procedure["sample"].step_descriptions["Description"]))


def test_experiment_names(procedure_fixture, titles_fixture):
    """Test the experiment_names method."""
    assert procedure_fixture.experiment_names == titles_fixture


def test_zero_columns(procedure_fixture):
    """Test methods to set the first value of columns to zero."""
    assert procedure_fixture.data["Procedure Time [s]"][0] == 0
    assert procedure_fixture.data["Procedure Capacity [Ah]"][0] == 0


def test_add_external_data(procedure_fixture, tmp_path):
    """Test adding external data to the procedure."""
    # Create external data
    data = pl.read_excel("tests/sample_data/neware/sample_data_neware.xlsx").to_pandas()
    start_date = data["Date"][0] - pd.Timedelta(seconds=30.54)
    end_date = data["Date"].iloc[-1] - pd.Timedelta(seconds=67.54)
    date_range = pd.date_range(start=start_date, end=end_date, freq="1min")
    seconds_passed = (date_range - start_date).total_seconds()
    value = 10 * np.sin(0.001 * seconds_passed)
    dataframe = pl.DataFrame({"Date": date_range, "Value": value})
    external_data_path = tmp_path / "external_data.csv"
    dataframe.write_csv(external_data_path)

    procedure1 = copy.deepcopy(procedure_fixture)
    procedure1.add_external_data(
        filepath=str(external_data_path),
        importing_columns=["Value"],
        date_column_name="Date",
    )
    assert "Value" in procedure1.column_list
    assert procedure1.data.select(
        pl.col("Value").tail(69).is_null(),
    ).unique().to_numpy() == np.array([True])

    procedure2 = copy.deepcopy(procedure_fixture)
    procedure2.add_external_data(
        filepath=str(external_data_path),
        importing_columns={"Value": "new column"},
    )
    assert "new column" in procedure2.column_list

    time = procedure2.data["Time [s]"].to_numpy() + 30.54
    value = 10 * np.sin(0.001 * time)
    data = procedure2.data["new column"].to_numpy()
    nan_mask = np.isnan(data)

    # Filter out NaNs
    value = value[~nan_mask]
    data = data[~nan_mask]
    assert np.allclose(data, value, atol=0.005)
