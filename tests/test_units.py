"""Unit tests for the Units class."""

from pybatdata.units import Units


def test_set_zero(BreakinCycles_fixture):
    """Test the set_zero method."""
    lf = BreakinCycles_fixture.cycle(1).chargeordischarge(1).lazyframe
    instruction = Units.set_zero("Capacity [Ah]")
    lf = lf.with_columns(instruction).collect()
    assert lf["Capacity [Ah]"][0] == 0
