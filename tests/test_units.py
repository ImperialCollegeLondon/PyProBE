"""Unit tests for the Units class."""


from pybatdata.units import Units


def test_set_zero(BreakinCycles_fixture):
    """Test the set_zero method."""
    lf = BreakinCycles_fixture.cycle(1).chargeordischarge(1)._data
    instruction = Units.set_zero("Capacity [Ah]")
    lf = lf.with_columns(instruction).collect()
    assert lf["Capacity [Ah]"][0] == 0

    assert Units.set_zero("Voltage [V]") is None
    assert Units.set_zero("Step") is None


def test_extract_quantity_and_unit():
    """Test the extract_quantity_and_unit method."""
    assert Units.extract_quantity_and_unit("Capacity [Ah]") == ("Capacity", "Ah")
    assert Units.extract_quantity_and_unit("Step") == ("Step", None)


def test_convert_units(lazyframe_fixture):
    """Test the convert_units method."""
    instruction_list = Units.convert_units("Capacity [Ah]")
    lf = lazyframe_fixture.with_columns(instruction_list)
    assert "Capacity [mAh]" in lf.columns

    instruction_list = Units.convert_units("Step")
    assert instruction_list is None
