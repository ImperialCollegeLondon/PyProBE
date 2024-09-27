"""Tests for the RawData class."""

import copy

import numpy as np
import polars as pl
import pybamm
import pytest
from polars.testing import assert_frame_equal

from pyprobe.rawdata import RawData


@pytest.fixture
def RawData_fixture(lazyframe_fixture, info_fixture, step_descriptions_fixture):
    """Return a Result instance."""
    return RawData(
        base_dataframe=lazyframe_fixture,
        info=info_fixture,
        step_descriptions=step_descriptions_fixture,
    )


def test_init(RawData_fixture, step_descriptions_fixture):
    """Test the __init__ method."""
    assert isinstance(RawData_fixture, RawData)
    assert isinstance(RawData_fixture.base_dataframe, pl.LazyFrame)
    assert isinstance(RawData_fixture.info, dict)
    assert_frame_equal(RawData_fixture.step_descriptions, step_descriptions_fixture)

    # test with incorrect data
    data = pl.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    with pytest.raises(ValueError):
        RawData(base_dataframe=data, info={"test": 1})


def test_pybamm_experiment(RawData_fixture):
    """Test the pybamm_experiment method."""
    assert isinstance(RawData_fixture.pybamm_experiment, pybamm.Experiment)
    assert (
        RawData_fixture.pybamm_experiment.steps[-1].description == "Rest for 1.5 hours"
    )

    RawData_fixture.step_descriptions = pl.LazyFrame(
        {
            "Step": [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12],
            "Description": [
                "Rest for 4 hours",
                "Charge at 4mA until 4.2 V, Hold at 4.2 V until 0.04 A",
                "Rest for 2 hours",
                "Discharge at 4 mA until 3 V",
                None,
                "Charge at 4 mA until 4.2 V, Hold at 4.2 V until 0.04 A",
                "Rest for 2 hours",
                "Rest for 10 seconds",
                None,
                "Rest for 30 minutes",
                "Rest for 1.5 hours",
            ],
        }
    )
    with pytest.raises(ValueError):
        RawData_fixture.pybamm_experiment


def test_capacity(BreakinCycles_fixture):
    """Test the capacity property."""
    capacity = BreakinCycles_fixture.cycle(0).charge(0).capacity
    assert np.isclose(capacity, 41.08565 / 1000)


def test_set_SOC(BreakinCycles_fixture):
    """Test the set_SOC method."""
    with_charge_specified = copy.deepcopy(BreakinCycles_fixture)
    with_charge_specified.set_SOC(0.04, BreakinCycles_fixture.cycle(-1).charge(-1))
    assert isinstance(with_charge_specified.base_dataframe, pl.LazyFrame)
    assert "Capacity [Ah]_right" not in with_charge_specified.data.columns
    with_charge_specified = with_charge_specified.data["SOC"]

    without_charge_specified = copy.deepcopy(BreakinCycles_fixture)
    without_charge_specified.set_SOC(0.04)
    assert isinstance(without_charge_specified.base_dataframe, pl.LazyFrame)
    without_charge_specified = without_charge_specified.data["SOC"]

    assert (with_charge_specified == without_charge_specified).all()
    assert max(without_charge_specified) == 1
    assert max(with_charge_specified) == 1


def test_SOC_ref_as_dataframe(BreakinCycles_fixture):
    """Test the set_SOC method with the reference charge collected into a dataframe."""
    with_charge_specified = BreakinCycles_fixture
    assert isinstance(with_charge_specified.base_dataframe, pl.LazyFrame)
    BreakinCycles_fixture.cycle(-1).charge(-1).data
    with_charge_specified.set_SOC(0.04, BreakinCycles_fixture.cycle(-1).charge(-1))
    assert isinstance(with_charge_specified.base_dataframe, pl.LazyFrame)


def test_SOC_with_base_as_dataframe(BreakinCycles_fixture):
    """Test the set_SOC method with the base dataframe collected into a dataframe."""
    with_charge_specified = BreakinCycles_fixture
    with_charge_specified.data
    assert isinstance(with_charge_specified.base_dataframe, pl.DataFrame)
    with_charge_specified.set_SOC(0.04, BreakinCycles_fixture.cycle(-1).charge(-1))
    assert isinstance(with_charge_specified.base_dataframe, pl.DataFrame)


def test_set_reference_capacity(BreakinCycles_fixture):
    """Test the set_reference_capacity method."""
    BreakinCycles_fixture.set_reference_capacity()
    assert BreakinCycles_fixture("Capacity - Referenced [Ah]").min() == 0
    assert np.isclose(
        BreakinCycles_fixture("Capacity - Referenced [Ah]").max(),
        BreakinCycles_fixture.capacity,
    )

    BreakinCycles_fixture.set_reference_capacity(0.04)
    assert np.isclose(
        BreakinCycles_fixture("Capacity - Referenced [Ah]").min(),
        0.04 - BreakinCycles_fixture.capacity,
    )
    assert BreakinCycles_fixture("Capacity - Referenced [Ah]").max() == 0.04


def test_zero_column(RawData_fixture):
    """Test method for zeroing the first value of a selected column."""
    RawData_fixture.zero_column(
        "Capacity [Ah]",
        "Zeroed Capacity [Ah]",
        "Capacity column with first value zeroed.",
    )
    assert RawData_fixture.data["Zeroed Capacity [Ah]"][0] == 0
    assert RawData_fixture.column_definitions["Zeroed Capacity [Ah]"] == (
        "Capacity column with first value zeroed."
    )


def test_definitions(lazyframe_fixture, info_fixture, step_descriptions_fixture):
    """Test that the definitions have been correctly set."""
    rawdata = RawData(
        base_dataframe=lazyframe_fixture,
        info=info_fixture,
        step_descriptions=step_descriptions_fixture,
    )
    definition_keys = list(rawdata.column_definitions.keys())
    assert set(definition_keys) == set(
        [
            "Time [s]",
            "Current [A]",
            "Voltage [V]",
            "Capacity [Ah]",
            "Cycle",
            "Step",
            "Event",
            "Date",
            "Temperature [C]",
        ]
    )
