"""Tests for the cycling class."""

import math

import pytest

from pyprobe.analysis.cycling import Cycling
from pyprobe.filters import Experiment
from pyprobe.result import Result


@pytest.fixture
def Cycling_fixture(lazyframe_fixture, info_fixture):
    """Return a Cycling instance."""
    input_data = Experiment(base_dataframe=lazyframe_fixture, info=info_fixture)
    return Cycling(input_data=input_data)


def test_init():
    """Test the __init__ method."""
    with pytest.raises(ValueError):
        Cycling(input_data=Result(base_dataframe=None, info=None))


def test_set_capacity_throughput(Cycling_fixture):
    """Test the set_capacity_throughput method."""
    Cycling_fixture._create_capacity_throughput()
    assert "Capacity Throughput [Ah]" in Cycling_fixture.input_data.data.columns
    assert Cycling_fixture.input_data.data["Capacity Throughput [Ah]"].head(1)[0] == 0
    assert (
        Cycling_fixture.input_data.step(0).data["Capacity Throughput [Ah]"].max()
        == Cycling_fixture.input_data.step(0).capacity
    )
    assert math.isclose(
        Cycling_fixture.input_data.data["Capacity Throughput [Ah]"].tail(1)[0],
        0.472115,
        rel_tol=1e-5,
    )


def test_summary(BreakinCycles_fixture):
    """Test the summary property."""
    cycling_instance = Cycling(input_data=BreakinCycles_fixture)
    assert isinstance(cycling_instance.summary(), Result)
    columns = cycling_instance.summary().data.columns
    required_columns = [
        "Capacity Throughput [Ah]",
        "Time [s]",
        "Charge Capacity [Ah]",
        "Discharge Capacity [Ah]",
        "SOH Charge [%]",
        "SOH Discharge [%]",
        "Coulombic Efficiency",
    ]
    assert all(item in columns for item in required_columns)
    assert cycling_instance.summary().data.shape[0] == 5
    assert cycling_instance.summary().data["SOH Charge [%]"].head(1)[0] == 100
    assert cycling_instance.summary().data["SOH Discharge [%]"].head(1)[0] == 100
    assert math.isclose(
        cycling_instance.summary().data["Charge Capacity [Ah]"].tail(1)[0],
        0.04139,
        rel_tol=1e-5,
    )
    assert math.isclose(
        cycling_instance.summary().data["Discharge Capacity [Ah]"].tail(1)[0],
        0.0413295,
        rel_tol=1e-5,
    )

    assert math.isclose(
        cycling_instance.summary().data["Coulombic Efficiency"].tail(1)[0],
        0.999212,
        rel_tol=1e-7,
    )
