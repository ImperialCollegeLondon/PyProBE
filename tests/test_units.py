"""Test the Units class."""

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pyprobe.units import Units, get_unit_scaling, split_quantity_unit, unit_from_regexp


def test_split_quantity_unit():
    """Test the split_quantity_unit method."""
    name = "Date"
    quantity, unit = split_quantity_unit(name)
    assert quantity == "Date"
    assert unit == ""

    name = "SOC"
    quantity, unit = split_quantity_unit(name)
    assert quantity == "SOC"
    assert unit == ""

    name = "Capacity [Ah]"
    quantity, unit = split_quantity_unit(name)
    assert quantity == "Capacity"
    assert unit == "Ah"

    name = "Percentage [%]"
    quantity, unit = split_quantity_unit(name)
    assert quantity == "Percentage"
    assert unit == "%"


def test_get_unit_scaling():
    """Test the get_unit_scaling method."""
    # Test for SI prefixes
    assert get_unit_scaling("mA") == (1e-3, "A")
    assert get_unit_scaling("µA.h") == (1e-6, "A.h")
    assert get_unit_scaling("nV") == (1e-9, "V")
    assert get_unit_scaling("ps") == (1e-12, "s")
    assert get_unit_scaling("kA") == (1e3, "A")
    assert get_unit_scaling("MV") == (1e6, "V")

    # Test for time units
    assert get_unit_scaling("s") == (1, "s")
    assert get_unit_scaling("min") == (60, "s")
    assert get_unit_scaling("hr") == (3600, "s")
    assert get_unit_scaling("Seconds") == (1, "s")

    # Test for single value unit
    assert get_unit_scaling("A") == (1, "A")


def test_UnitsExpr_to_default():
    """Test the to_default method."""
    df = pl.DataFrame({"Current [mA]": [1.0, 2.0, 3.0]})
    expected_df = df.with_columns(
        (pl.col("Current [mA]").cast(pl.Float64) * 1e-3).alias("Current [A]")
    )
    assert_frame_equal(
        df.with_columns(pl.col("Current [mA]").units.to_default()), expected_df
    )

    df = pl.DataFrame({"Capacity 1 [mAh]": [1.0, 2.0, 3.0]})
    expected_df = df.with_columns(
        (pl.col("Capacity 1 [mAh]").cast(pl.Float64) * 1e-3).alias("Capacity 1 [Ah]")
    )
    assert_frame_equal(
        df.with_columns(pl.col("Capacity 1 [mAh]").units.to_default()), expected_df
    )

    df = pl.DataFrame({"Time [hr]": [1.0, 2.0, 3.0]})
    expected_df = df.with_columns(
        (pl.col("Time [hr]").cast(pl.Float64) * 3600).alias("Time [s]")
    )
    assert_frame_equal(
        df.with_columns(pl.col("Time [hr]").units.to_default()), expected_df
    )

    df = pl.DataFrame({"Time [ms]": [1.0, 2.0, 3.0]})
    expected_df = df.with_columns(
        (pl.col("Time [ms]").cast(pl.Float64) * 1e-3).alias("Time [s]")
    )
    assert_frame_equal(
        df.with_columns(pl.col("Time [ms]").units.to_default()), expected_df
    )

    df = pl.DataFrame({"Current/voltage [mA/V]": [1.0, 2.0, 3.0]})
    expected_df = df.with_columns(
        (pl.col("Current/voltage [mA/V]").cast(pl.Float64) * 1e-3).alias(
            "Current/voltage [A/V]"
        )
    )
    assert_frame_equal(
        df.with_columns(pl.col("Current/voltage [mA/V]").units.to_default()),
        expected_df,
    )

    df = pl.DataFrame({"Current/voltage/V": [1.0, 2.0, 3.0]})
    expected_df = df.with_columns(
        (pl.col("Current/voltage/V").cast(pl.Float64) * 1).alias("Current/voltage [V]")
    )
    assert_frame_equal(
        df.with_columns(
            pl.col("Current/voltage/V").units.to_default(r"^(.+)/([^/]+)$")
        ),
        expected_df,
    )

    df = pl.DataFrame({"Current/voltage/mV": [1.0, 2.0, 3.0]})
    expected_df = df.with_columns(
        (pl.col("Current/voltage/mV").cast(pl.Float64) * 1e-3).alias(
            "Current/voltage [V]"
        )
    )


def test_UnitsExpr_to_si():
    """Test the to_si method."""
    df = pl.DataFrame({"Current [mA]": [1.0, 2.0, 3.0]})
    expected_df = df.with_columns(pl.col("Current [mA]").cast(pl.Float64) * 1e-3)
    assert_frame_equal(
        df.with_columns(pl.col("Current [mA]").units.to_si("mA")), expected_df
    )

    with pytest.raises(ValueError):
        df.with_columns(pl.col("Current [mA]").units.to_si("pA"))

    with pytest.raises(ValueError):
        df.with_columns(pl.col("Current [mA]").units.to_si("mV"))

    df = pl.DataFrame({"Capacity 1 [mAh]": [1.0, 2.0, 3.0]})
    expected_df = df.with_columns(pl.col("Capacity 1 [mAh]").cast(pl.Float64) * 1e-3)
    assert_frame_equal(
        df.with_columns(pl.col("Capacity 1 [mAh]").units.to_si("mAh")), expected_df
    )

    df = pl.DataFrame({"Time [hr]": [1.0, 2.0, 3.0]})
    expected_df = df.with_columns(pl.col("Time [hr]").cast(pl.Float64) * 3600)
    assert_frame_equal(
        df.with_columns(pl.col("Time [hr]").units.to_si("hr")), expected_df
    )

    df = pl.DataFrame({"Time [ms]": [1.0, 2.0, 3.0]})
    expected_df = df.with_columns(pl.col("Time [ms]").cast(pl.Float64) * 1e-3)
    assert_frame_equal(
        df.with_columns(pl.col("Time [ms]").units.to_si("ms")), expected_df
    )

    df = pl.DataFrame({"Current/voltage [mA/V]": [1.0, 2.0, 3.0]})
    expected_df = df.with_columns(
        pl.col("Current/voltage [mA/V]").cast(pl.Float64) * 1e-3
    )
    assert_frame_equal(
        df.with_columns(pl.col("Current/voltage [mA/V]").units.to_si("mA/V")),
        expected_df,
    )

    df = pl.DataFrame({"Current/voltage/V": [1.0, 2.0, 3.0]})
    expected_df = df.with_columns(pl.col("Current/voltage/V").cast(pl.Float64) * 1)
    assert_frame_equal(
        df.with_columns(pl.col("Current/voltage/V").units.to_si("V")), expected_df
    )

    df = pl.DataFrame({"Current/voltage/mV": [1.0, 2.0, 3.0]})
    expected_df = df.with_columns(pl.col("Current/voltage/mV").cast(pl.Float64) * 1e-3)
    assert_frame_equal(
        df.with_columns(pl.col("Current/voltage/mV").units.to_si("mV")), expected_df
    )


def test_from_regexp():
    """Test the get_quantity_and_unit method."""
    name = "Capacity [Ah]"
    unit_object = unit_from_regexp(name)
    assert unit_object.input_quantity == "Capacity"
    assert unit_object.input_unit == "Ah"

    name = "Two names [Ah]"
    unit_object = unit_from_regexp(name)
    assert unit_object.input_quantity == "Two names"
    assert unit_object.input_unit == "Ah"

    name = "Current/mA"
    pattern = r"(\w+)/(\w+)"
    unit_object = unit_from_regexp(name, pattern)
    assert unit_object.input_quantity == "Current"
    assert unit_object.input_unit == "mA"

    name = "Percentage [%]"
    unit_object = unit_from_regexp(name)
    assert unit_object.input_quantity == "Percentage"
    assert unit_object.input_unit == "%"


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
