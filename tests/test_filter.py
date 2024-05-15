"""Tests for the filter module."""
import pytest

from pyprobe.filter import Filter


def test_get_events(lazyframe_fixture):
    """Test the _get_events method."""
    result = Filter._get_events(lazyframe_fixture)

    # Check that the result has the expected columns
    expected_columns = ["_cycle", "_cycle_reversed", "_step", "_step_reversed"]
    assert all(column in result.columns for column in expected_columns)
    result = result.collect()
    # Check that '_cycle' contains only values from 0 to 13
    assert result["_cycle"].min() == 0
    assert result["_cycle"].max() == 13

    # Check that '_step' contains only values from 0 to 61
    assert result["_step"].min() == 0
    assert result["_step"].max() == 61

    # Check that '_cycle_reversed' contains only values from -14 to -1
    assert result["_cycle_reversed"].min() == -14
    assert result["_cycle_reversed"].max() == -1

    # Check that '_step_reversed' contains only values from -62 to -1
    assert result["_step_reversed"].min() == -62
    assert result["_step_reversed"].max() == -1


def test_step(BreakinCycles_fixture):
    """Test the step method."""
    step = BreakinCycles_fixture.cycle(0).step(1)
    assert (step.data["Step"] == 5).all()


def test_multi_step(BreakinCycles_fixture):
    """Test the step method."""
    step = BreakinCycles_fixture.cycle(0).step(range(1, 4))
    assert (step.data["Step"].unique() == [5, 6, 7]).all()


def test_charge(BreakinCycles_fixture):
    """Test the charge method."""
    charge = BreakinCycles_fixture.cycle(0).charge(0)
    assert (charge.data["Step"] == 6).all()
    assert (charge.data["Current [A]"] > 0).all()


def test_discharge(BreakinCycles_fixture):
    """Test the discharge method."""
    discharge = BreakinCycles_fixture.cycle(0).discharge(0)
    assert (discharge.data["Step"] == 4).all()
    assert (discharge.data["Current [A]"] < 0).all()

    # test invalid input
    with pytest.raises(ValueError):
        BreakinCycles_fixture.cycle(6).data


def test_chargeordischarge(BreakinCycles_fixture):
    """Test the chargeordischarge method."""
    charge = BreakinCycles_fixture.cycle(0).chargeordischarge(0)
    assert (charge.data["Step"] == 4).all()
    assert (charge.data["Current [A]"] < 0).all()

    discharge = BreakinCycles_fixture.cycle(0).chargeordischarge(1)
    assert (discharge.data["Step"] == 6).all()
    assert (discharge.data["Current [A]"] > 0).all()


def test_rest(BreakinCycles_fixture):
    """Test the rest method."""
    rest = BreakinCycles_fixture.cycle(0).rest(0)
    assert (rest.data["Step"] == 5).all()
    assert (rest.data["Current [A]"] == 0).all()

    rest = BreakinCycles_fixture.cycle(0).rest(1)
    assert (rest.data["Step"] == 7).all()
    assert (rest.data["Current [A]"] == 0).all()


def test_negative_cycle_index(BreakinCycles_fixture):
    """Test the negative index."""
    cycle = BreakinCycles_fixture.cycle(-1)
    assert (cycle.data["Cycle"] == 5).all()
    assert (cycle.data["Step"].unique() == [4, 5, 6, 7]).all()


def test_negative_step_index(BreakinCycles_fixture):
    """Test the negative index."""
    step = BreakinCycles_fixture.cycle(0).step(-1)
    assert (step.data["Step"] == 7).all()


def test_cycle(BreakinCycles_fixture):
    """Test the cycle method."""
    cycle = BreakinCycles_fixture.cycle(0)
    assert (cycle.data["Cycle"] == 1).all()
    assert (cycle.data["Step"].unique() == [4, 5, 6, 7]).all()


def test_all_steps(BreakinCycles_fixture):
    """Test the all_steps method."""
    all_steps = BreakinCycles_fixture.cycle(0).step()
    assert (all_steps.data["Cycle"] == 1).all()
    assert (all_steps.data["Step"].unique() == [4, 5, 6, 7]).all()
