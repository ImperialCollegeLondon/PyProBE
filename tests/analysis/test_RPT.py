"""Tests for the RPT analysis module."""

import polars as pl
import pytest
from polars import testing as pl_testing

from pyprobe.analysis.RPT import RPT
from pyprobe.result import Result


@pytest.fixture
def RPT_fixture(procedure_fixture):
    """Return an RPT object."""
    return RPT(input_data=[procedure_fixture, procedure_fixture, procedure_fixture])


def test_init(RPT_fixture, procedure_fixture):
    """Test the initialization of the RPT object."""
    assert isinstance(RPT_fixture.rpt_summary, Result)
    expected_df = pl.DataFrame(
        {
            "RPT Number": list(range(3)),
        }
    )
    pl_testing.assert_frame_equal(RPT_fixture.rpt_summary.data, expected_df)
    assert RPT_fixture.rpt_summary.column_definitions == {
        "RPT Number": "The RPT number.",
    }

    # Test with incorrect input data
    with pytest.raises(ValueError):
        RPT(input_data=[procedure_fixture.discharge(1)])


def test_process_cell_capacity(RPT_fixture, procedure_fixture):
    """Test the process_cell_capacity method."""
    RPT_fixture.process_cell_capacity(
        "experiment('Break-in Cycles').discharge(-1)",
        name="Last discharge capacity [Ah]",
    )
    print(RPT_fixture.rpt_summary.data)
    known_discharge_capacity = (
        procedure_fixture.experiment("Break-in Cycles").discharge(-1).capacity
    )
    expected_df = pl.DataFrame(
        {
            "RPT Number": list(range(3)),
            "Last discharge capacity [Ah]": [known_discharge_capacity] * 3,
        }
    )
    pl_testing.assert_frame_equal(RPT_fixture.rpt_summary.data, expected_df)
    assert RPT_fixture.rpt_summary.column_definitions == {
        "RPT Number": "The RPT number.",
        "Last discharge capacity [Ah]": "The cell capacity.",
    }


def test_process_soh(RPT_fixture):
    """Test the process_soh method."""
    RPT_fixture.process_soh("experiment('Break-in Cycles').discharge(-1)", name="SOH")
    expected_df = pl.DataFrame(
        {
            "RPT Number": list(range(3)),
            "SOH": [1.0] * 3,
        }
    )
    pl_testing.assert_frame_equal(RPT_fixture.rpt_summary.data, expected_df)
    assert RPT_fixture.rpt_summary.column_definitions == {
        "RPT Number": "The RPT number.",
        "SOH": "The cell SOH.",
    }
