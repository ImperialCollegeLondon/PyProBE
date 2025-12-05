"""Tests for the RawData class."""

import copy
import random

import numpy as np
import polars as pl
import pytest

from pyprobe.rawdata import RawData


@pytest.fixture
def RawData_fixture(lazyframe_fixture, info_fixture, step_descriptions_fixture):
    """Return a Result instance."""
    return RawData(
        lf=lazyframe_fixture,
        info=info_fixture,
        step_descriptions=step_descriptions_fixture,
    )


def test_init(RawData_fixture, step_descriptions_fixture):
    """Test the __init__ method."""
    assert isinstance(RawData_fixture, RawData)
    assert isinstance(RawData_fixture.lf, pl.LazyFrame)
    assert isinstance(RawData_fixture.info, dict)
    assert RawData_fixture.step_descriptions == step_descriptions_fixture

    # test with incorrect data
    data = pl.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    with pytest.raises(ValueError):
        RawData(lf=data, info={"test": 1})


def test_data(RawData_fixture):
    """Test the data property."""
    columns = copy.deepcopy(RawData_fixture.data.collect_schema().names())
    random.shuffle(columns)
    RawData_fixture.lf = RawData_fixture.lf.select(columns)
    assert RawData_fixture.data.columns == [
        "Time [s]",
        "Step",
        "Event",
        "Current [A]",
        "Voltage [V]",
        "Capacity [Ah]",
        "Date",
    ]


def test_capacity(BreakinCycles_fixture):
    """Test the capacity property."""
    capacity = BreakinCycles_fixture.cycle(0).charge(0).capacity
    assert np.isclose(capacity, 41.08565 / 1000)


def test_set_SOC(BreakinCycles_fixture):
    """Test the set_soc method."""
    with_charge_specified = copy.deepcopy(BreakinCycles_fixture)
    with_charge_specified.set_soc(0.04, BreakinCycles_fixture.cycle(-1).charge(-1))
    assert isinstance(with_charge_specified.lf, pl.LazyFrame)
    assert "Capacity [Ah]_right" not in with_charge_specified.data.columns
    with_charge_specified = with_charge_specified.data["SOC"]

    without_charge_specified = copy.deepcopy(BreakinCycles_fixture)
    without_charge_specified.set_soc(0.04)
    assert isinstance(without_charge_specified.lf, pl.LazyFrame)
    without_charge_specified = without_charge_specified.data["SOC"]

    assert (with_charge_specified == without_charge_specified).all()
    assert max(without_charge_specified) == 1
    assert max(with_charge_specified) == 1


def test_SOC_ref_as_dataframe(BreakinCycles_fixture):
    """Test the set_soc method with the reference charge collected into a dataframe."""
    with_charge_specified = BreakinCycles_fixture
    assert isinstance(with_charge_specified.lf, pl.LazyFrame)
    BreakinCycles_fixture.cycle(-1).charge(-1).data
    with_charge_specified.set_soc(0.04, BreakinCycles_fixture.cycle(-1).charge(-1))
    assert isinstance(with_charge_specified.lf, pl.LazyFrame)


def test_SOC_with_base_as_dataframe(BreakinCycles_fixture):
    """Test the set_soc method with the base dataframe collected into a dataframe."""
    with_charge_specified = BreakinCycles_fixture
    with_charge_specified.data
    with_charge_specified.set_soc(0.04, BreakinCycles_fixture.cycle(-1).charge(-1))
    assert "SOC" in with_charge_specified.columns


def test_deprecated_set_SOC(BreakinCycles_fixture, mocker):
    """Test the deprecated set_SOC method."""
    mocker.patch("pyprobe.rawdata.RawData.set_soc")
    BreakinCycles_fixture.set_SOC(0.04)
    BreakinCycles_fixture.set_soc.assert_called_once_with(0.04, None)


def test_set_reference_capacity(BreakinCycles_fixture):
    """Test the set_reference_capacity method."""
    procedure1 = copy.deepcopy(BreakinCycles_fixture)
    procedure1.set_reference_capacity()
    assert procedure1.get("Capacity - Referenced [Ah]").min() == 0
    assert np.isclose(
        procedure1.get("Capacity - Referenced [Ah]").max(),
        procedure1.capacity,
    )

    procedure2 = copy.deepcopy(BreakinCycles_fixture)
    procedure2.set_reference_capacity(0.04)
    assert np.isclose(
        procedure2.get("Capacity - Referenced [Ah]").min(),
        0.04 - procedure2.capacity,
    )
    assert procedure2.get("Capacity - Referenced [Ah]").max() == 0.04


def test_zero_column(RawData_fixture):
    """Test method for zeroing the first value of a selected column."""
    RawData_fixture.zero_column(
        "Capacity [Ah]",
        "Zeroed Capacity [Ah]",
        "Capacity column with first value zeroed.",
    )
    assert RawData_fixture.data["Zeroed Capacity [Ah]"][0] == 0
    assert RawData_fixture.column_definitions["Zeroed Capacity"] == (
        "Capacity column with first value zeroed."
    )


def test_definitions(lazyframe_fixture, info_fixture, step_descriptions_fixture):
    """Test that the definitions have been correctly set."""
    rawdata = RawData(
        lf=lazyframe_fixture,
        info=info_fixture,
        step_descriptions=step_descriptions_fixture,
    )
    definition_keys = list(rawdata.column_definitions.keys())
    assert set(definition_keys) == {
        "Time",
        "Current",
        "Voltage",
        "Capacity",
        "Cycle",
        "Step",
        "Event",
        "Date",
        "Temperature",
    }


def test_pybamm_experiment():
    """Test successful creation of PyBaMM experiment list."""
    # Create test data
    test_data = pl.DataFrame(
        {
            "Time [s]": [1, 2, 3],
            "Step": [1, 2, 2],
            "Event": [1, 2, 2],
            "Current [A]": [0.1, 0.2, 0.3],
            "Voltage [V]": [3.0, 3.1, 3.2],
            "Capacity [Ah]": [0.1, 0.2, 0.3],
        },
    )

    step_descriptions = {
        "Step": [1, 2],
        "Description": ["Rest for 1 hour", "Charge at 1C until 4.2V"],
    }

    raw_data = RawData(
        lf=test_data,
        info={},
        step_descriptions=step_descriptions,
    )

    result = raw_data.pybamm_experiment
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == "Rest for 1 hour"
    assert result[1] == "Charge at 1C until 4.2V"


def test_pybamm_experiment_missing_descriptions():
    """Test error handling when step descriptions are missing."""
    test_data = pl.DataFrame(
        {
            "Time [s]": [1, 2, 3],
            "Step": [1, 2, 3],
            "Event": [1, 2, 3],
            "Current [A]": [0.1, 0.2, 0.3],
            "Voltage [V]": [3.0, 3.1, 3.2],
            "Capacity [Ah]": [0.1, 0.2, 0.3],
        },
    )

    step_descriptions = {
        "Step": [1, 2, 3],
        "Description": ["Rest for 1 hour", None, "Charge at 1C"],
    }

    raw_data = RawData(
        lf=test_data,
        info={},
        step_descriptions=step_descriptions,
    )

    with pytest.raises(ValueError, match="Descriptions for steps.*are missing"):
        raw_data.pybamm_experiment


def test_pybamm_experiment_multiple_conditions():
    """Test handling of steps with multiple comma-separated conditions."""
    test_data = pl.DataFrame(
        {
            "Time [s]": [1, 2],
            "Step": [1, 2],
            "Event": [1, 2],
            "Current [A]": [0.1, 0.2],
            "Voltage [V]": [3.0, 3.1],
            "Capacity [Ah]": [0.1, 0.2],
        },
    )

    step_descriptions = {
        "Step": [1, 2],
        "Description": [
            "Charge at 1C until 4.2V, Hold at 4.2V until C/20",
            "Rest for 1 hour",
        ],
    }

    raw_data = RawData(
        lf=test_data,
        info={},
        step_descriptions=step_descriptions,
    )

    result = raw_data.pybamm_experiment
    assert len(result) == 3
    assert result[0] == "Charge at 1C until 4.2V"
    assert result[1] == "Hold at 4.2V until C/20"
    assert result[2] == "Rest for 1 hour"


def test_pybamm_experiment_with_loops():
    """Test pybamm_experiment property handles repeated steps correctly."""
    # Create test data with repeated steps: 1->2->1->2
    base_df = pl.DataFrame(
        {
            "Step": [1, 1, 1, 2, 2, 1, 1, 2, 2],
            "Time [s]": range(9),
            "Voltage [V]": [3.0] * 9,
            "Current [A]": [0.1] * 9,
            "Capacity [Ah]": [0.1] * 9,
            "Event": [1, 1, 1, 2, 2, 3, 3, 4, 4],
        },
    )

    step_descriptions = {
        "Step": [1, 2],
        "Description": ["Discharge at C/10", "Rest for 1 hour"],
    }

    data = RawData(lf=base_df, info={}, step_descriptions=step_descriptions)

    expected = [
        "Discharge at C/10",  # Step 1
        "Rest for 1 hour",  # Step 2
        "Discharge at C/10",  # Step 1 again
        "Rest for 1 hour",  # Step 2 again
    ]

    assert data.pybamm_experiment == expected
