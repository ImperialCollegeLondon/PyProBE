"""Tests for the RawData class."""

import numpy as np
import polars as pl
import polars.testing as pl_testing
import pytest

from pyprobe.rawdata import RawData
from pyprobe.result import Result


@pytest.fixture
def RawData_fixture(lazyframe_fixture, info_fixture):
    """Return a Result instance."""
    return RawData(lazyframe_fixture, info_fixture)


def test_init(RawData_fixture):
    """Test the __init__ method."""
    assert isinstance(RawData_fixture, RawData)
    assert isinstance(RawData_fixture._data, pl.LazyFrame)
    assert isinstance(RawData_fixture.info, dict)


def test_data(RawData_fixture):
    """Test the data property."""
    assert isinstance(RawData_fixture._data, pl.LazyFrame)
    assert isinstance(RawData_fixture.data, pl.DataFrame)
    assert isinstance(RawData_fixture._data, pl.DataFrame)
    pl_testing.assert_frame_equal(RawData_fixture.data, RawData_fixture._data)

    for column in ["Capacity [Ah]", "Time [s]"]:
        assert RawData_fixture.data[column][0] == 0


def test_gradient(BreakinCycles_fixture):
    """Test the gradient property."""
    discharge = BreakinCycles_fixture.cycle(0).discharge(0)
    gradient = discharge.gradient("LEAN", "Capacity [Ah]", "Voltage [V]", 1, "dxdy")
    assert isinstance(gradient, Result)
    assert isinstance(gradient.data, pl.DataFrame)
    assert gradient.data.columns == [
        "Capacity [Ah]",
        "Voltage [V]",
        "d(Capacity [Ah])/d(Voltage [V])",
    ]


def test_capacity(BreakinCycles_fixture):
    """Test the capacity property."""
    capacity = BreakinCycles_fixture.cycle(0).charge(0).capacity
    assert np.isclose(capacity, 41.08565 / 1000)


def test_set_SOC(BreakinCycles_fixture, benchmark):
    """Test the set_SOC method."""
    with_charge_specified = BreakinCycles_fixture

    with_charge_specified.set_SOC(0.04, BreakinCycles_fixture.cycle(-1).charge(-1))

    without_charge_specified = BreakinCycles_fixture

    def set_SOC():
        return without_charge_specified.set_SOC(0.04)

    benchmark(set_SOC)

    assert (
        with_charge_specified.data["SOC"] == without_charge_specified.data["SOC"]
    ).all()
    assert max(without_charge_specified.data["SOC"]) == 1
    assert max(with_charge_specified.data["SOC"]) == 1


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
    RawData_fixture.zero_column("Capacity [Ah]", "Zeroed Capacity [Ah]")
    assert RawData_fixture.data["Zeroed Capacity [Ah]"][0] == 0


def test_definitions(RawData_fixture):
    """Test that the definitions have been correctly set."""
    definition_keys = list(RawData_fixture._column_definitions.keys())
    assert set(definition_keys) == set(
        [
            "Date",
            "Time [s]",
            "Current [A]",
            "Voltage [V]",
            "Capacity [Ah]",
        ]
    )
