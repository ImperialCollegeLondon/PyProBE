"""Tests for the pulsing experiment."""

import numpy as np
import pytest

import pyprobe.analysis.pulsing as pulsing
from pyprobe.analysis.pulsing import Pulsing
from pyprobe.result import Result


@pytest.fixture
def Pulsing_fixture(procedure_fixture):
    """Pytest fixture for example pulsing experiment."""
    procedure_fixture.set_soc(
        reference_charge=procedure_fixture.experiment("Break-in Cycles").charge(-1),
    )
    return procedure_fixture.experiment("Discharge Pulses")


def test_pulse(Pulsing_fixture):
    """Test the pulse method."""
    pulse_obj = Pulsing(input_data=Pulsing_fixture)
    pulse = pulse_obj.pulse(0)
    assert pulse.data["Time [s]"][0] == 483572.397
    assert (pulse.data["Step"] == 10).all()

    pulse = pulse_obj.pulse(6)
    assert pulse.data["Time [s]"][0] == 531149.401
    assert (pulse.data["Step"] == 10).all()


def test_get_resistances(Pulsing_fixture):
    """Test the get_resistances method."""
    resistances = pulsing.get_resistances(Pulsing_fixture, [10])
    assert isinstance(resistances, Result)
    assert resistances.get("R0 [Ohms]")[0] == (4.1558 - 4.1919) / -0.0199936
    assert resistances.get("R_10s [Ohms]")[0] == (4.1337 - 4.1919) / -0.0199936


def test_get_ocv_curve(Pulsing_fixture):
    """Test the get_ocv_curve method."""
    result = pulsing.get_ocv_curve(Pulsing_fixture)
    expected_ocv_points = [
        4.1919,
        4.0949,
        3.9934,
        3.8987,
        3.8022,
        3.7114,
        3.665,
        3.6334,
        3.5866,
        3.5164,
        3.4513,
    ]
    assert isinstance(result, Result)
    assert result.column_list == Pulsing_fixture.column_list
    assert np.allclose(result.get("Voltage [V]"), expected_ocv_points)
