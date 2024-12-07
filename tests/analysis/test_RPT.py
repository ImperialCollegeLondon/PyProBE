"""Tests for the RPT analysis module."""

import polars as pl
import pytest
from polars import testing as pl_testing

from pyprobe.analysis import pulsing
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


def test_process_pulse_resistance(RPT_fixture, procedure_fixture):
    """Test the process_pulse_resistance method."""
    RPT_fixture.process_pulse_resistance("experiment('Discharge Pulses')")

    # test R0
    known_r0 = (
        pulsing.get_resistances(procedure_fixture.experiment("Discharge Pulses"))
        .data["R0 [Ohms]"]
        .to_numpy()
    )
    expected_df = pl.DataFrame(
        {
            "RPT Number": list(range(3)),
            "R0 [Ohms]": [known_r0] * 3,
        }
    )
    pl_testing.assert_frame_equal(RPT_fixture.rpt_summary.data, expected_df)

    # test R_5s
    RPT_fixture.process_pulse_resistance(
        "experiment('Discharge Pulses')", eval_time=5.0
    )
    known_r5s = (
        pulsing.get_resistances(procedure_fixture.experiment("Discharge Pulses"), [5.0])
        .data["R_5.0s [Ohms]"]
        .to_numpy()
    )
    expected_df = pl.DataFrame(
        {
            "RPT Number": list(range(3)),
            "R0 [Ohms]": [known_r0] * 3,
            "R_5.0s [Ohms]": [known_r5s] * 3,
        }
    )

    # test single pulse number
    RPT_fixture.process_pulse_resistance(
        "experiment('Discharge Pulses')",
        pulse_number=4,
        eval_time=5.0,
        name="R_5s_pulse_1",
    )
    known_r5s_pulse_1 = (
        pulsing.get_resistances(procedure_fixture.experiment("Discharge Pulses"), [5.0])
        .data["R_5.0s [Ohms]"]
        .to_numpy()[4]
    )
    expected_df = pl.DataFrame(
        {
            "RPT Number": list(range(3)),
            "R0 [Ohms]": [known_r0] * 3,
            "R_5.0s [Ohms]": [known_r5s] * 3,
            "R_5s_pulse_1": [known_r5s_pulse_1] * 3,
        }
    )
