"""Test the UnitConverter class."""
import polars as pl
import polars.testing as pl_testing
import pytest

from pyprobe.unitconverter import UnitConverter


@pytest.fixture
def current_quantity():
    """Return a UnitConverter instance for current."""
    return UnitConverter("Current [A]")


@pytest.fixture
def capacity_quantity():
    """Return a UnitConverter instance for capacity."""
    return UnitConverter("Capacity [mAh]")


@pytest.fixture
def time_quantity():
    """Return a UnitConverter instance for time."""
    return UnitConverter("Time [hr]")


@pytest.fixture
def current_from_cycler_quantity():
    """Return a UnitConverter instance for current from a cycler."""
    pattern = r"(\w+)/(\w+)"
    return UnitConverter("Current/A", pattern)


@pytest.fixture
def I_from_cycler_quantity():
    """Return a UnitConverter instance for I from a cycler."""
    pattern = r"(\w+)/(\w+)"
    return UnitConverter("I/mA", pattern)


def test_get_quantity_and_unit():
    """Test the get_quantity_and_unit method."""
    name = "Capacity [Ah]"
    quantity, unit = UnitConverter.get_quantity_and_unit(name)
    assert quantity == "Capacity"
    assert unit == "Ah"

    name = "Two names [Ah]"
    quantity, unit = UnitConverter.get_quantity_and_unit(name)
    assert quantity == "Two names"
    assert unit == "Ah"

    name = "Step"
    pattern = r"(\w+)\s*\[(\w+)\]"
    with pytest.raises(ValueError):
        quantity, unit = UnitConverter.get_quantity_and_unit(name)

    name = "Current/mA"
    pattern = r"(\w+)/(\w+)"
    quantity, unit = UnitConverter.get_quantity_and_unit(name, pattern)
    assert quantity == "Current"
    assert unit == "mA"


def test_init(
    current_quantity,
    capacity_quantity,
    time_quantity,
    current_from_cycler_quantity,
    I_from_cycler_quantity,
):
    """Test the __init__ method."""
    assert current_quantity.name == "Current [A]"
    assert current_quantity.default_quantity == "Current"
    assert current_quantity.default_unit == "A"
    assert current_quantity.prefix is None
    assert current_quantity.default_name == "Current [A]"

    assert capacity_quantity.name == "Capacity [mAh]"
    assert capacity_quantity.default_unit == "Ah"
    assert capacity_quantity.prefix == "m"
    assert capacity_quantity.default_name == "Capacity [Ah]"

    assert time_quantity.name == "Time [hr]"
    assert time_quantity.default_quantity == "Time"
    assert time_quantity.default_unit == "s"
    assert time_quantity.prefix is None
    assert time_quantity.default_name == "Time [s]"

    assert current_from_cycler_quantity.name == "Current/A"
    assert current_from_cycler_quantity.default_quantity == "Current"
    assert current_from_cycler_quantity.default_unit == "A"
    assert current_from_cycler_quantity.prefix is None
    assert current_from_cycler_quantity.default_name == "Current [A]"

    assert I_from_cycler_quantity.name == "I/mA"
    assert I_from_cycler_quantity.default_quantity == "Current"
    assert I_from_cycler_quantity.default_unit == "A"
    assert I_from_cycler_quantity.prefix == "m"
    assert I_from_cycler_quantity.default_name == "Current [A]"

    with pytest.raises(ValueError):
        UnitConverter("Step")
    with pytest.raises(ValueError):
        UnitConverter("Current/A")


def test_from_default(current_quantity, capacity_quantity, time_quantity):
    """Test the from_default method."""
    instruction = current_quantity.from_default()
    original_frame = pl.DataFrame({"Current [A]": [1.0, 2.0, 3.0]})
    updated_frame = original_frame.with_columns(instruction)
    assert "Current [A]" in updated_frame.columns
    pl_testing.assert_series_equal(
        updated_frame["Current [A]"], original_frame["Current [A]"]
    )

    instruction = capacity_quantity.from_default()
    original_frame = pl.DataFrame({"Capacity [Ah]": [1.0, 2.0, 3.0]})
    updated_frame = original_frame.with_columns(instruction)
    assert "Capacity [mAh]" in updated_frame.columns
    pl_testing.assert_series_equal(
        updated_frame["Capacity [mAh]"],
        original_frame["Capacity [Ah]"] / 1e-3,
        check_names=False,
    )

    instruction = time_quantity.from_default()
    print(time_quantity.default_quantity)
    print(time_quantity.factor)
    original_frame = pl.DataFrame({"Time [s]": [1.0, 2.0, 3.0]})
    updated_frame = original_frame.with_columns(instruction)
    assert "Time [hr]" in updated_frame.columns
    pl_testing.assert_series_equal(
        updated_frame["Time [hr]"], original_frame["Time [s]"] / 3600, check_names=False
    )


def test_to_default(I_from_cycler_quantity):
    """Test the to_default method."""
    original_frame = pl.DataFrame({"I/mA": [1.0, 2.0, 3.0]})
    instruction = I_from_cycler_quantity.to_default()
    updated_frame = original_frame.with_columns(instruction)
    assert "Current [A]" in updated_frame.columns
    pl_testing.assert_series_equal(
        updated_frame["Current [A]"], original_frame["I/mA"] * 1e-3, check_names=False
    )

    original_frame = pl.DataFrame({"Chg. Cap.(Ah)": [1.0, 2.0, 3.0]})
    instruction = UnitConverter("Chg. Cap.(Ah)", r"(.+)\((.+)\)").to_default(
        keep_name=True
    )
    updated_frame = original_frame.with_columns(instruction)
    assert "Chg. Cap. [Ah]" in updated_frame.columns
    pl_testing.assert_series_equal(
        updated_frame["Chg. Cap. [Ah]"],
        original_frame["Chg. Cap.(Ah)"],
        check_names=False,
    )
