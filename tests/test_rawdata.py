"""Tests for the RawData class."""

import numpy as np
import polars as pl
import pytest

from pyprobe.rawdata import RawData


@pytest.fixture
def RawData_fixture(lazyframe_fixture, info_fixture):
    """Return a Result instance."""
    return RawData(base_dataframe=lazyframe_fixture, info=info_fixture)


def test_init(RawData_fixture):
    """Test the __init__ method."""
    assert isinstance(RawData_fixture, RawData)
    assert isinstance(RawData_fixture.base_dataframe, pl.LazyFrame)
    assert isinstance(RawData_fixture.info, dict)

    # test with incorrect data
    data = pl.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    with pytest.raises(ValueError):
        RawData(base_dataframe=data, info={"test": 1})


# def test_gradient(BreakinCycles_fixture):
#     """Test the gradient property."""
#     discharge = BreakinCycles_fixture.cycle(0).discharge(0)
#     gradient = discharge.gradient("Capacity [Ah]", "Voltage [V]", "LEAN", 1, "dxdy")
#     assert isinstance(gradient, Result)
#     assert isinstance(gradient.data, pl.DataFrame)
#     assert gradient.data.columns == [
#         "Capacity [Ah]",
#         "Voltage [V]",
#         "d(Capacity [Ah])/d(Voltage [V])",
#     ]


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
    RawData_fixture.zero_column(
        "Capacity [Ah]",
        "Zeroed Capacity [Ah]",
        "Capacity column with first value zeroed.",
    )
    assert RawData_fixture.data["Zeroed Capacity [Ah]"][0] == 0
    assert RawData_fixture.column_definitions["Zeroed Capacity [Ah]"] == (
        "Capacity column with first value zeroed."
    )


def test_definitions(lazyframe_fixture, info_fixture):
    """Test that the definitions have been correctly set."""
    rawdata = RawData(base_dataframe=lazyframe_fixture, info=info_fixture)
    definition_keys = list(rawdata.column_definitions.keys())
    print(rawdata.column_definitions)
    assert set(definition_keys) == set(
        [
            "Date",
            "Time [s]",
            "Current [A]",
            "Voltage [V]",
            "Capacity [Ah]",
        ]
    )
