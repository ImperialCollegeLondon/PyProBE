"""Tests for the cycling experiment class."""

import math

import pytest

from pybatdata.experiments.cycling import Cycling
from pybatdata.result import Result


@pytest.fixture
def Cycling_fixture(lazyframe_fixture, info_fixture):
    """Return a Cycling instance."""
    return Cycling(lazyframe_fixture, info_fixture)


def test_set_capacity_throughput(Cycling_fixture):
    """Test the set_capacity_throughput method."""
    assert "Capacity Throughput [Ah]" in Cycling_fixture.data.columns
    assert Cycling_fixture.data["Capacity Throughput [Ah]"].head(1)[0] == 0
    assert (
        Cycling_fixture.step(0).data["Capacity Throughput [Ah]"].max()
        == Cycling_fixture.step(0).capacity
    )
    assert math.isclose(
        Cycling_fixture.data["Capacity Throughput [Ah]"].tail(1)[0],
        0.472115,
        rel_tol=1e-5,
    )


def test_SOH_capacity(BreakinCycles_fixture):
    """Test the SOH_capacity property."""
    assert isinstance(BreakinCycles_fixture.SOH_capacity, Result)
    columns = BreakinCycles_fixture.SOH_capacity.data.columns
    required_columns = [
        "Capacity Throughput [Ah]",
        "Time [s]",
        "Charge Capacity [Ah]",
        "Discharge Capacity [Ah]",
        "SOH Charge [%]",
        "SOH Discharge [%]",
    ]
    assert all(item in columns for item in required_columns)
    assert BreakinCycles_fixture.SOH_capacity.data.shape[0] == 5
    assert BreakinCycles_fixture.SOH_capacity.data["SOH Charge [%]"].head(1)[0] == 100
    assert (
        BreakinCycles_fixture.SOH_capacity.data["SOH Discharge [%]"].head(1)[0] == 100
    )
    assert math.isclose(
        BreakinCycles_fixture.SOH_capacity.data["Charge Capacity [Ah]"].tail(1)[0],
        0.04139,
        rel_tol=1e-5,
    )
    assert math.isclose(
        BreakinCycles_fixture.SOH_capacity.data["Discharge Capacity [Ah]"].tail(1)[0],
        0.0413295,
        rel_tol=1e-5,
    )
