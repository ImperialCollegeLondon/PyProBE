"""Tests for the filter module."""
import pytest


def test_step(BreakinCycles_fixture, benchmark):
    """Test the step method."""

    def step():
        return BreakinCycles_fixture.cycle(0).step(1).data

    data = benchmark(step)
    assert (data["Step"] == 5).all()


def test_multi_step(BreakinCycles_fixture, benchmark):
    """Test the step method."""

    def multi_step():
        return BreakinCycles_fixture.cycle(0).step(range(1, 4)).data

    data = benchmark(multi_step)
    assert (data["Step"].unique() == [5, 6, 7]).all()


def test_charge(BreakinCycles_fixture, benchmark):
    """Test the charge method."""

    def charge():
        return BreakinCycles_fixture.cycle(0).charge(0).data

    data = benchmark(charge)
    assert (data["Step"] == 6).all()
    assert (data["Current [A]"] > 0).all()


def test_discharge(BreakinCycles_fixture, benchmark):
    """Test the discharge method."""

    def discharge():
        return BreakinCycles_fixture.cycle(0).discharge(0).data

    data = benchmark(discharge)
    assert (data["Step"] == 4).all()
    assert (data["Current [A]"] < 0).all()

    # test invalid input
    with pytest.raises(ValueError):
        BreakinCycles_fixture.cycle(6).data


def test_chargeordischarge(BreakinCycles_fixture, benchmark):
    """Test the chargeordischarge method."""

    def chargeordischarge():
        return BreakinCycles_fixture.cycle(0).chargeordischarge(0).data

    data = benchmark(chargeordischarge)
    assert (data["Step"] == 4).all()
    assert (data["Current [A]"] < 0).all()

    data = BreakinCycles_fixture.cycle(0).chargeordischarge(1).data
    assert (data["Step"] == 6).all()
    assert (data["Current [A]"] > 0).all()


def test_rest(BreakinCycles_fixture, benchmark):
    """Test the rest method."""

    def rest():
        return BreakinCycles_fixture.cycle(0).rest(0).data

    data = benchmark(rest)
    assert (data["Step"] == 5).all()
    assert (data["Current [A]"] == 0).all()

    data = BreakinCycles_fixture.cycle(0).rest(1).data
    assert (data["Step"] == 7).all()
    assert (data["Current [A]"] == 0).all()


def test_negative_cycle_index(BreakinCycles_fixture, benchmark):
    """Test the negative index."""

    def negative_cycle_index():
        return BreakinCycles_fixture.cycle(-1).data

    data = benchmark(negative_cycle_index)
    assert (data["Cycle"] == 4).all()
    assert (data["Step"].unique() == [4, 5, 6, 7]).all()


def test_negative_step_index(BreakinCycles_fixture, benchmark):
    """Test the negative index."""

    def negative_step_index():
        return BreakinCycles_fixture.cycle(0).step(-1).data

    data = benchmark(negative_step_index)
    assert (data["Step"] == 7).all()


def test_cycle(BreakinCycles_fixture, benchmark):
    """Test the cycle method."""

    def cycle():
        return BreakinCycles_fixture.cycle(2).data

    data = benchmark(cycle)
    assert (data["Cycle"] == 2).all()
    assert (data["Step"].unique() == [4, 5, 6, 7]).all()

    assert data["Cycle Time [s]"][0] == 0
    assert data["Cycle Capacity [Ah]"][0] == 0


def test_all_steps(BreakinCycles_fixture, benchmark):
    """Test the all_steps method."""

    def all_steps():
        return BreakinCycles_fixture.cycle(0).step().data

    data = benchmark(all_steps)
    assert (data["Cycle"] == 0).all()
    assert (data["Step"].unique() == [4, 5, 6, 7]).all()
