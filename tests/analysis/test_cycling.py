"""Tests for the cycling class."""

import math

from pyprobe.analysis import cycling
from pyprobe.result import Result


def test_summary(BreakinCycles_fixture):
    """Test the summary property."""
    summary = cycling.summary(BreakinCycles_fixture)
    assert isinstance(summary, Result)
    columns = summary.data.columns
    required_columns = [
        "Capacity Throughput / Ah",
        "Test Time / s",
        "Charge Capacity / Ah",
        "Discharge Capacity / Ah",
        "SOH Charge / %",
        "SOH Discharge / %",
        "Coulombic Efficiency",
        "Cycle Count / 1",
    ]
    assert set(required_columns) == set(columns)
    assert summary.data.shape[0] == 5
    assert summary.data["SOH Charge / %"].head(1)[0] == 100
    assert summary.data["SOH Discharge / %"].head(1)[0] == 100
    assert math.isclose(
        summary.data["Charge Capacity / Ah"].tail(1)[0],
        0.04139,
        rel_tol=1e-2,
    )
    assert math.isclose(
        summary.data["Discharge Capacity / Ah"].tail(1)[0],
        0.0413295,
        rel_tol=1e-2,
    )

    assert math.isclose(
        summary.data["Coulombic Efficiency"].tail(1)[0],
        0.999212,
        rel_tol=1e-2,
    )

    cycle_counts = summary.data["Cycle Count / 1"].to_list()
    assert cycle_counts == sorted(cycle_counts)
