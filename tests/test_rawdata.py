"""Tests for the RawData class."""
import math

import numpy as np
import polars as pl
import pytest

from pyprobe.rawdata import RawData


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
    pl.testing.assert_frame_equal(RawData_fixture.data, RawData_fixture._data)

    for column in ["Capacity [Ah]", "Time [s]"]:
        assert RawData_fixture.data[column][0] == 0


def test_capacity(BreakinCycles_fixture):
    """Test the capacity property."""
    capacity = BreakinCycles_fixture.cycle(0).charge(0).capacity
    assert np.isclose(capacity, 41.08565 / 1000)


def test_dQdV(BreakinCycles_fixture):
    """Test the dQdV method."""
    dQdV = BreakinCycles_fixture.cycle(0).charge(0).dQdV("feng_2020", 0.006)
    dQdV_data = dQdV.data.filter(pl.col("Voltage [V]") < 4)
    assert dQdV_data.columns == ["Voltage [V]", "IC [Ah/V]"]
    assert math.isclose(dQdV_data["IC [Ah/V]"].max(), 3.04952, rel_tol=1e-5)


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
