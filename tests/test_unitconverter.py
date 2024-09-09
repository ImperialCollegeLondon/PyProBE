"""Test the Units class."""
import numpy as np
import polars as pl

from pyprobe.units import Units


def test_from_regexp():
    """Test the get_quantity_and_unit method."""
    name = "Capacity [Ah]"
    unit_object = Units.from_regexp(name)
    assert unit_object.input_quantity == "Capacity"
    assert unit_object.input_unit == "Ah"

    name = "Two names [Ah]"
    unit_object = Units.from_regexp(name)
    assert unit_object.input_quantity == "Two names"
    assert unit_object.input_unit == "Ah"

    name = "Current/mA"
    pattern = r"(\w+)/(\w+)"
    unit_object = Units.from_regexp(name, pattern)
    assert unit_object.input_quantity == "Current"
    assert unit_object.input_unit == "mA"


def test_init():
    """Test the __init__ method."""
    current_quantity = Units("Current", "A")
    assert current_quantity.input_quantity == "Current"
    assert current_quantity.default_quantity == "Current"
    assert current_quantity.default_unit == "A"
    assert current_quantity.prefix is None

    capacity_quantity = Units("Capacity", "mAh")
    assert capacity_quantity.input_quantity == "Capacity"
    assert capacity_quantity.default_unit == "Ah"
    assert capacity_quantity.prefix == "m"

    time_quantity = Units("Time", "hr")
    assert time_quantity.input_quantity == "Time"
    assert time_quantity.default_quantity == "Time"
    assert time_quantity.default_unit == "s"
    assert time_quantity.prefix is None


def test_from_default_unit():
    """Test the from_default_unit method."""
    instruction = Units("Current", "mA").from_default_unit()
    original_frame = pl.DataFrame({"Current [A]": [1.0, 2.0, 3.0]})
    updated_frame = original_frame.with_columns(instruction)
    assert "Current [mA]" in updated_frame.columns
    assert np.allclose(
        updated_frame["Current [mA]"].to_numpy(),
        original_frame["Current [A]"].to_numpy() * 1000,
    )

    # Test for Current
    instruction = Units("Current", "mA").from_default_unit()
    original_frame = pl.DataFrame({"Current [A]": [1.0, 2.0, 3.0]})
    updated_frame = original_frame.with_columns(instruction)
    assert "Current [mA]" in updated_frame.columns
    assert np.allclose(
        updated_frame["Current [mA]"].to_numpy(),
        original_frame["Current [A]"].to_numpy() * 1000,
    )

    # Test for Capacity
    instruction = Units("Capacity", "mAh").from_default_unit()
    original_frame = pl.DataFrame({"Capacity [Ah]": [1.0, 2.0, 3.0]})
    updated_frame = original_frame.with_columns(instruction)
    assert "Capacity [mAh]" in updated_frame.columns
    assert np.allclose(
        updated_frame["Capacity [mAh]"].to_numpy(),
        original_frame["Capacity [Ah]"].to_numpy() * 1000,
    )

    # Test for Time
    instruction = Units("Time", "hr").from_default_unit()
    original_frame = pl.DataFrame({"Time [s]": [1.0, 2.0, 3.0]})
    updated_frame = original_frame.with_columns(instruction)
    assert "Time [hr]" in updated_frame.columns
    assert np.allclose(
        updated_frame["Time [hr]"].to_numpy(),
        original_frame["Time [s]"].to_numpy() / 3600,
    )


def test_to_default_unit():
    """Test the to_default_unit method."""
    # Test for Current
    instruction = Units("Current", "mA").to_default_unit()
    original_frame = pl.DataFrame({"Current [mA]": [1000.0, 2000.0, 3000.0]})
    updated_frame = original_frame.with_columns(instruction)
    assert "Current [A]" in updated_frame.columns
    assert np.allclose(
        updated_frame["Current [A]"].to_numpy(),
        original_frame["Current [mA]"].to_numpy() / 1000,
    )

    # Test for Capacity
    instruction = Units("Capacity", "mAh").to_default_unit()
    original_frame = pl.DataFrame({"Capacity [mAh]": [1000.0, 2000.0, 3000.0]})
    updated_frame = original_frame.with_columns(instruction)
    assert "Capacity [Ah]" in updated_frame.columns
    assert np.allclose(
        updated_frame["Capacity [Ah]"].to_numpy(),
        original_frame["Capacity [mAh]"].to_numpy() / 1000,
    )

    # Test for Time
    instruction = Units("Time", "hr").to_default_unit()
    original_frame = pl.DataFrame({"Time [hr]": [1.0, 2.0, 3.0]})
    updated_frame = original_frame.with_columns(instruction)
    assert "Time [s]" in updated_frame.columns
    assert np.allclose(
        updated_frame["Time [s]"].to_numpy(),
        original_frame["Time [hr]"].to_numpy() * 3600,
    )
