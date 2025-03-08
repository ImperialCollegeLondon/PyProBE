"""Tests for the column importers module."""

import datetime

import polars as pl
from polars.testing import assert_frame_equal

from pyprobe.cyclers.column_maps import (
    CapacityFromChDch,
    CapacityFromCurrentSign,
    CastAndRename,
    ColumnMap,
    ConvertTemperature,
    ConvertUnits,
    DateTime,
    StepFromCategorical,
    TimeFromDate,
)


class TestColumnMap(ColumnMap):
    """Test implementation of ColumnMap."""

    @property
    def expr(self) -> pl.Expr:
        """Implement abstract method."""
        return None


def test_match_columns():
    """Test the match_columns static method."""
    available_columns = ["Time [s]", "Current [mA]", "Voltage [V]", "Count"]
    column_map_instance = TestColumnMap("", [""])
    # Test exact matches
    patterns = ["Count"]
    result = column_map_instance.match_columns(available_columns, patterns)
    assert result == {"Count": {"Cycler name": "Count", "Cycler unit": ""}}

    # Test wildcard matches
    patterns = ["Current [*]"]
    result = column_map_instance.match_columns(available_columns, patterns)
    assert result == {
        "Current [*]": {"Cycler name": "Current [mA]", "Cycler unit": "mA"},
    }

    # Test multiple patterns
    patterns = ["Count", "Current [*]", "Voltage [*]"]
    result = column_map_instance.match_columns(available_columns, patterns)
    assert result == {
        "Count": {"Cycler name": "Count", "Cycler unit": ""},
        "Current [*]": {"Cycler name": "Current [mA]", "Cycler unit": "mA"},
        "Voltage [*]": {"Cycler name": "Voltage [V]", "Cycler unit": "V"},
    }

    # Test non-existent columns
    patterns = ["NonExistent"]
    result = column_map_instance.match_columns(available_columns, patterns)
    assert result == {}

    # Test invalid unit
    available_columns = ["Current [invalid]"]
    patterns = ["Current [*]"]
    result = column_map_instance.match_columns(available_columns, patterns)
    assert result == {}

    # Test empty inputs
    assert column_map_instance.match_columns([], []) == {}
    assert column_map_instance.match_columns(available_columns, []) == {}
    assert column_map_instance.match_columns([], patterns) == {}

    # test a different format
    available_columns = ["Time/s", "Current/mA", "Voltage/time/V", "Count", "Voltage/V"]
    patterns = ["Count", "Current/*", "Voltage/*"]
    result = column_map_instance.match_columns(available_columns, patterns)
    assert result == {
        "Count": {"Cycler name": "Count", "Cycler unit": ""},
        "Current/*": {"Cycler name": "Current/mA", "Cycler unit": "mA"},
        "Voltage/*": {"Cycler name": "Voltage/V", "Cycler unit": "V"},
    }


def test_ColumnMap_validate():
    """Test the ColumnMap class."""
    column_map = TestColumnMap("Date", ["DateTime"])
    column_map.validate(["DateTime"])
    assert column_map.pyprobe_name == "Date"
    assert column_map.column_map == {
        "DateTime": {"Cycler name": "DateTime", "Cycler unit": ""},
    }
    assert column_map.columns_validated

    column_map = TestColumnMap("Date", ["DateTime", "Current [*]"])
    column_map.validate(["DateTime", "Current [mA]", "Voltage [V]"])
    assert column_map.pyprobe_name == "Date"
    assert column_map.column_map == {
        "DateTime": {"Cycler name": "DateTime", "Cycler unit": ""},
        "Current [*]": {"Cycler name": "Current [mA]", "Cycler unit": "mA"},
    }
    assert column_map.columns_validated

    column_map = TestColumnMap("Date", ["DateTime"])
    column_map.validate(["Date"])
    assert column_map.pyprobe_name == "Date"
    assert not column_map.columns_validated
    assert column_map.column_map == {}

    column_map = TestColumnMap("Date", ["DateTime", "Current [*]"])
    column_map.validate(["Date", "Current [mA]", "Voltage [V]"])
    assert column_map.pyprobe_name == "Date"
    assert column_map.column_map == {
        "Current [*]": {"Cycler name": "Current [mA]", "Cycler unit": "mA"},
    }
    assert not column_map.columns_validated


def test_ColumnMap_get():
    """Test the get method of the ColumnMap class."""
    column_map = TestColumnMap("Date", ["DateTime", "Current [*]"])
    column_map.validate(["DateTime", "Current [mA]", "Voltage [V]"])
    assert str(column_map.get("DateTime")) == str(pl.col("DateTime"))
    assert str(column_map.get("Current [*]")) == str(pl.col("Current [mA]"))


def test_CastAndRename():
    """Test the CastAndRename class."""
    column_map = CastAndRename("Date", "DateTime", pl.String)
    column_map.validate(["DateTime"])
    assert str(column_map.expr) == str(pl.col("DateTime").cast(pl.String).alias("Date"))


def test_ConvertUnits():
    """Test the ConvertUnits class."""
    column_map = ConvertUnits("Current [A]", "I [*]")
    column_map.validate(["I [mA]"])
    df = pl.DataFrame({"I [mA]": [1.0, 2.0, 3.0]})
    assert_frame_equal(
        df.select(column_map.expr),
        df.select((pl.col("I [mA]") / 1000).alias("Current [A]")),
    )


def test_ConvertTemperature():
    """Test the ConvertTemperature class."""
    column_map = ConvertTemperature("Temperature/*")
    column_map.validate(["Temperature/K"])
    df = pl.DataFrame({"Temperature/K": [300, 305, 310]})
    expected_df = pl.DataFrame({"Temperature [C]": [26.85, 31.85, 36.85]})
    assert_frame_equal(df.select(column_map.expr), expected_df)


def test_CapacityFromChDch():
    """Test the CapacityFromChDch class."""
    column_map = CapacityFromChDch("Q_ch [*]", "Q_dis [*]")
    column_map.validate(["Q_ch [mAh]", "Q_dis [Ah]"])
    df = pl.DataFrame({"Q_ch [mAh]": [1.0, 0.0, 0.0], "Q_dis [Ah]": [0.0, 1, 2]})
    expected_df = pl.DataFrame({"Capacity [Ah]": [0.001, -0.999, -1.999]})
    assert_frame_equal(df.select(column_map.expr), expected_df)


def test_CapacityFromCurrentSign():
    """Test the CapacityFromCurrentSign class."""
    column_map = CapacityFromCurrentSign(
        "Q [*]",
        "I [*]",
    )
    df = pl.DataFrame({"I [mA]": [1.0, -1.0, -1.0], "Q [Ah]": [1.0, 1.0, 2.0]})
    column_map.validate(df.columns)
    expected_df = pl.DataFrame({"Capacity [Ah]": [1.0, 0.0, -1.0]})
    assert_frame_equal(df.select(column_map.expr), expected_df)


def test_DateTime():
    """Test the DateTime class."""
    column_map = DateTime("DateTime", "%Y-%m-%d %H:%M:%S%.f")
    column_map.validate(["DateTime"])
    df = pl.DataFrame(
        {
            "DateTime": [
                "2022-04-02 02:06:00.000000",
                "2022-04-02 02:06:01.000000",
                "2022-04-02 02:06:02.000000",
            ],
        },
    )
    expected_df = pl.DataFrame(
        {
            "Date": [
                datetime.datetime(2022, 4, 2, 2, 6),
                datetime.datetime(2022, 4, 2, 2, 6, 1),
                datetime.datetime(2022, 4, 2, 2, 6, 2),
            ],
        },
    )
    assert_frame_equal(df.select(column_map.expr), expected_df)


def test_TimeFromDate():
    """Test the TimeFromDate class."""
    column_map = TimeFromDate("DateTime", "%Y-%m-%d %H:%M:%S%.f")
    column_map.validate(["DateTime"])
    df = pl.DataFrame(
        {
            "DateTime": [
                "2022-04-02 02:06:00.000000",
                "2022-04-02 02:06:01.000000",
                "2022-04-02 02:06:02.000000",
            ],
        },
    )
    expected_df = pl.DataFrame({"Time [s]": [0.0, 1.0, 2.0]})
    assert_frame_equal(df.select(column_map.expr), expected_df)


def test_StepFromCategorical():
    """Test the StepFromCategorical class."""
    column_map = StepFromCategorical("Step Type")
    column_map.validate(["Step Type"])
    df = pl.DataFrame(
        {"Step Type": ["Charge", "Charge", "Discharge", "Discharge", "Charge"]}
    )
    expected_df = pl.DataFrame({"Step": [0, 0, 1, 1, 2]}, schema={"Step": pl.UInt32})
    assert_frame_equal(df.select(column_map.expr), expected_df)
