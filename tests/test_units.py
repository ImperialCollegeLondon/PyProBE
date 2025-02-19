"""Test the Units class."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pyprobe.units import get_unit_scaling, split_quantity_unit


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


def test_UnitsExpr_to_base_unit():
    """Test the to_base_unit method."""
    df = pl.DataFrame({"Current [mA]": [1.0, 2.0, 3.0]})
    expected_df = df.with_columns(pl.col("Current [mA]").cast(pl.Float64) * 1e-3)
    assert_frame_equal(
        df.with_columns(pl.col("Current [mA]").units.to_base_unit("mA")),
        expected_df,
    )

    with pytest.raises(ValueError):
        df.with_columns(pl.col("Current [mA]").units.to_base_unit("pA"))

    with pytest.raises(ValueError):
        df.with_columns(pl.col("Current [mA]").units.to_base_unit("mV"))

    df = pl.DataFrame({"Capacity 1 [mAh]": [1.0, 2.0, 3.0]})
    expected_df = df.with_columns(pl.col("Capacity 1 [mAh]").cast(pl.Float64) * 1e-3)
    assert_frame_equal(
        df.with_columns(pl.col("Capacity 1 [mAh]").units.to_base_unit("mAh")),
        expected_df,
    )

    df = pl.DataFrame({"Time [hr]": [1.0, 2.0, 3.0]})
    expected_df = df.with_columns(pl.col("Time [hr]").cast(pl.Float64) * 3600)
    assert_frame_equal(
        df.with_columns(pl.col("Time [hr]").units.to_base_unit("hr")),
        expected_df,
    )

    df = pl.DataFrame({"Time [ms]": [1.0, 2.0, 3.0]})
    expected_df = df.with_columns(pl.col("Time [ms]").cast(pl.Float64) * 1e-3)
    assert_frame_equal(
        df.with_columns(pl.col("Time [ms]").units.to_base_unit("ms")),
        expected_df,
    )

    df = pl.DataFrame({"Current/voltage [mA/V]": [1.0, 2.0, 3.0]})
    expected_df = df.with_columns(
        pl.col("Current/voltage [mA/V]").cast(pl.Float64) * 1e-3,
    )
    assert_frame_equal(
        df.with_columns(pl.col("Current/voltage [mA/V]").units.to_base_unit("mA/V")),
        expected_df,
    )

    df = pl.DataFrame({"Current/voltage/V": [1.0, 2.0, 3.0]})
    expected_df = df.with_columns(pl.col("Current/voltage/V").cast(pl.Float64) * 1)
    assert_frame_equal(
        df.with_columns(pl.col("Current/voltage/V").units.to_base_unit("V")),
        expected_df,
    )

    df = pl.DataFrame({"Current/voltage/mV": [1.0, 2.0, 3.0]})
    expected_df = pl.DataFrame({"Current/voltage/V": [1.0, 2.0, 3.0]})
    expected_df = df.with_columns(pl.col("Current/voltage/mV").cast(pl.Float64) * 1e-3)
    assert_frame_equal(
        df.with_columns(pl.col("Current/voltage/mV").units.to_base_unit("mV")),
        expected_df,
    )


def test_UnitsExpr_to_unit():
    """Test the to_unit method."""
    df = pl.DataFrame({"Current [A]": [1.0, 2.0, 3.0]})
    expected_df = pl.DataFrame({"Current [mA]": [1e3, 2e3, 3e3]})
    assert_frame_equal(
        df.select(pl.col("Current [A]").units.to_unit("mA")),
        expected_df,
    )

    df = pl.DataFrame({"Capacity 1 [mAh]": [1.0, 2.0, 3.0]})
    expected_df = pl.DataFrame({"Capacity 1 [mAh]": [1.0, 2.0, 3.0]})
    assert_frame_equal(
        df.select(pl.col("Capacity 1 [mAh]").units.to_unit("mAh")),
        expected_df,
    )

    df = pl.DataFrame({"Time [hr]": [1.0, 2.0, 3.0]})
    expected_df = pl.DataFrame({"Time [min]": [60.0, 120.0, 180.0]})
    assert_frame_equal(
        df.select(pl.col("Time [hr]").units.to_unit("min")),
        expected_df,
    )

    df = pl.DataFrame({"Time [ms]": [1.0, 2.0, 3.0]})
    expected_df = pl.DataFrame({"Time [s]": [1e-3, 2e-3, 3e-3]})
    assert_frame_equal(
        df.select(pl.col("Time [ms]").units.to_unit("s")),
        expected_df,
    )

    df = pl.DataFrame({"Current [mA]": [1.0, 2.0, 3.0]})
    expected_df = pl.DataFrame({"Current [kA]": [1e-6, 2e-6, 3e-6]})
    assert_frame_equal(
        df.select(pl.col("Current [mA]").units.to_unit("kA")),
        expected_df,
    )
