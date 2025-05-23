"""Tests for the column importers module."""

import datetime

import polars as pl
from polars.testing import assert_frame_equal

from pyprobe.cyclers.column_maps import (
    CapacityFromChDchMap,
    CapacityFromCurrentSignMap,
    CastAndRenameMap,
    ColumnMap,
    ConvertTemperatureMap,
    ConvertUnitsMap,
    DateTimeMap,
    StepFromCategoricalMap,
    TimeFromDateMap,
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


def test_ColumnMap_validate(caplog):
    """Test the ColumnMap class."""
    # Test case 1: All columns are found
    column_map = TestColumnMap("Date", ["DateTime"])
    with caplog.at_level("INFO"):
        column_map.validate(["DateTime"])
        expected_name = "Date"
        expected_map = {
            "DateTime": {"Cycler name": "DateTime", "Cycler unit": ""},
        }
        assert column_map.pyprobe_name == expected_name
        assert column_map.column_map == expected_map
        assert column_map.columns_validated
        assert caplog.messages[-1] == (
            "Column mapping validated: Date -> {'DateTime': {'Cycler name': "
            "'DateTime', 'Cycler unit': ''}}"
        )

    # Test case 2: Multiple columns, all found
    column_map = TestColumnMap("Date", ["DateTime", "Current [*]"])
    with caplog.at_level("INFO"):
        caplog.clear()
        column_map.validate(["DateTime", "Current [mA]", "Voltage [V]"])
        expected_name = "Date"
        expected_map = {
            "DateTime": {"Cycler name": "DateTime", "Cycler unit": ""},
            "Current [*]": {"Cycler name": "Current [mA]", "Cycler unit": "mA"},
        }
        assert column_map.pyprobe_name == expected_name
        assert column_map.column_map == expected_map
        assert column_map.columns_validated
        assert caplog.messages[-1] == (
            "Column mapping validated: Date -> {'DateTime': {'Cycler name': "
            "'DateTime', 'Cycler unit': ''}, 'Current [*]': {'Cycler name': "
            "'Current [mA]', 'Cycler unit': 'mA'}}"
        )

    # Test case 3: No columns found
    column_map = TestColumnMap("Date", ["DateTime"])
    with caplog.at_level("INFO"):
        caplog.clear()
        column_map.validate(["Date"])
        expected_name = "Date"
        expected_map = {}
        assert column_map.pyprobe_name == expected_name
        assert column_map.column_map == expected_map
        assert not column_map.columns_validated
        assert (
            caplog.messages[-1]
            == "Failed to find required columns for Date. Missing: {'DateTime'}"
        )

    # Test case 4: Some columns found, others missing
    column_map = TestColumnMap("Date", ["DateTime", "Current [*]"])
    with caplog.at_level("INFO"):
        column_map.validate(["Date", "Current [mA]", "Voltage [V]"])
        expected_name = "Date"
        expected_map = {
            "Current [*]": {"Cycler name": "Current [mA]", "Cycler unit": "mA"},
        }
        assert column_map.pyprobe_name == expected_name
        assert column_map.column_map == expected_map
        assert not column_map.columns_validated
        assert (
            caplog.messages[-1]
            == "Failed to find required columns for Date. Missing: {'DateTime'}"
        )


def test_ColumnMap_get():
    """Test the get method of the ColumnMap class."""
    column_map = TestColumnMap("Date", ["DateTime", "Current [*]"])
    column_map.validate(["DateTime", "Current [mA]", "Voltage [V]"])
    assert str(column_map.get("DateTime")) == str(pl.col("DateTime"))
    assert str(column_map.get("Current [*]")) == str(pl.col("Current [mA]"))


def test_CastAndRename():
    """Test the CastAndRenameMap class."""
    column_map = CastAndRenameMap("Date", "DateTime", pl.String)
    column_map.validate(["DateTime"])
    assert str(column_map.expr) == str(pl.col("DateTime").cast(pl.String).alias("Date"))


def test_ConvertUnits():
    """Test the ConvertUnitsMap class."""
    column_map = ConvertUnitsMap("Current [A]", "I [*]")
    column_map.validate(["I [mA]"])
    df = pl.DataFrame({"I [mA]": [1.0, 2.0, 3.0]})
    assert_frame_equal(
        df.select(column_map.expr),
        df.select((pl.col("I [mA]") / 1000).alias("Current [A]")),
    )


def test_ConvertTemperature():
    """Test the ConvertTemperatureMap class."""
    column_map = ConvertTemperatureMap("Temperature/*")
    column_map.validate(["Temperature/K"])
    df = pl.DataFrame({"Temperature/K": [300, 305, 310]})
    expected_df = pl.DataFrame({"Temperature [C]": [26.85, 31.85, 36.85]})
    assert_frame_equal(df.select(column_map.expr), expected_df)


def test_CapacityFromChDch():
    """Test the CapacityFromChDchMap class."""
    column_map = CapacityFromChDchMap("Q_ch [*]", "Q_dis [*]")
    column_map.validate(["Q_ch [mAh]", "Q_dis [Ah]"])
    df = pl.DataFrame({"Q_ch [mAh]": [1.0, 0.0, 0.0], "Q_dis [Ah]": [0.0, 1, 2]})
    expected_df = pl.DataFrame({"Capacity [Ah]": [0.001, -0.999, -1.999]})
    assert_frame_equal(df.select(column_map.expr), expected_df)


def test_CapacityFromCurrentSign():
    """Test the CapacityFromCurrentSignMap class."""
    column_map = CapacityFromCurrentSignMap(
        "Q [*]",
        "I [*]",
    )
    df = pl.DataFrame({"I [mA]": [1.0, -1.0, -1.0], "Q [Ah]": [1.0, 1.0, 2.0]})
    column_map.validate(df.columns)
    expected_df = pl.DataFrame({"Capacity [Ah]": [1.0, 0.0, -1.0]})
    assert_frame_equal(df.select(column_map.expr), expected_df)


def test_DateTime():
    """Test the DateTimeMap class."""
    column_map = DateTimeMap("DateTime", "%Y-%m-%d %H:%M:%S%.f")
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
    """Test the TimeFromDateMap class."""
    column_map = TimeFromDateMap("DateTime", "%Y-%m-%d %H:%M:%S%.f")
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


def test_StepFromCategorical(caplog):
    """Test the StepFromCategoricalMap class."""
    column_map = StepFromCategoricalMap("Step Type")
    column_map.validate(["Step Type"])
    df = pl.DataFrame(
        {"Step Type": ["Charge", "Charge", "Discharge", "Discharge", "Charge"]}
    )
    expected_df = pl.DataFrame({"Step": [0, 0, 1, 1, 2]}, schema={"Step": pl.UInt32})
    with caplog.at_level("WARNING"):
        assert_frame_equal(df.select(column_map.expr), expected_df)
    assert caplog.messages[-1] == (
        "Step number is being inferred from the categorical column Step Type. "
        "A new step will be counted each time the column changes. This means that "
        "it will not be possible to filter by cycle."
    )
