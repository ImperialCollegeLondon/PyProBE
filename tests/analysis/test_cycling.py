"""Tests for the cycling class."""

import math

import pytest

from pyprobe.analysis import cycling
from pyprobe.filters import Experiment
from pyprobe.result import Result


@pytest.fixture
def Cycling_fixture(lazyframe_fixture, info_fixture, step_descriptions_fixture):
    """Return a Cycling instance."""
    input_data = Experiment(
        base_dataframe=lazyframe_fixture,
        info=info_fixture,
        step_descriptions=step_descriptions_fixture,
        cycle_info=[],
    )
    return input_data


def test_set_capacity_throughput(Cycling_fixture):
    """Test the set_capacity_throughput method."""
    result = cycling._create_capacity_throughput(
        Cycling_fixture.base_dataframe,
    ).collect()
    assert "Capacity Throughput [Ah]" in result.columns
    assert result["Capacity Throughput [Ah]"].head(1)[0] == 0
    assert math.isclose(
        result["Capacity Throughput [Ah]"].tail(1)[0],
        0.472115,
        rel_tol=1e-5,
    )


def test_summary(BreakinCycles_fixture):
    """Test the summary property."""
    summary = cycling.summary(BreakinCycles_fixture)
    assert isinstance(summary, Result)
    columns = summary.data.columns
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
    assert summary.data.shape[0] == 5
    assert summary.data["SOH Charge [%]"].head(1)[0] == 100
    assert summary.data["SOH Discharge [%]"].head(1)[0] == 100
    assert math.isclose(
        summary.data["Charge Capacity [Ah]"].tail(1)[0],
        0.04139,
        rel_tol=1e-2,
    )
    assert math.isclose(
        summary.data["Discharge Capacity [Ah]"].tail(1)[0],
        0.0413295,
        rel_tol=1e-2,
    )

    assert math.isclose(
        summary.data["Coulombic Efficiency"].tail(1)[0],
        0.999212,
        rel_tol=1e-2,
    )
