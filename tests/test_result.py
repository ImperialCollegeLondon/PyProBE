"""Tests for the result module - organized into logical test classes."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import numpy as np
import numpy.testing as np_testing
import polars as pl
import polars.testing as pl_testing
import pytest
from scipy.io import loadmat
from tzlocal import get_localzone

from pyprobe.result import (
    Result,
    _validate_timezone,
    combine_results,
)


@pytest.fixture
def Result_fixture(lazyframe_fixture, info_fixture):
    """Return a Result instance."""
    return Result(
        lf=lazyframe_fixture,
        metadata=info_fixture,
        column_definitions={
            "Current": "Current definition",
        },
    )


@pytest.fixture
def reduced_result_fixture():
    """Return a Result instance with reduced data."""
    data = pl.DataFrame(
        {
            "Current [A]": [1, 2, 3],
            "Voltage [V]": [1, 2, 3],
        },
    )
    return Result(
        lf=data.lazy(),
        metadata={"test": "metadata"},
        column_definitions={
            "Voltage": "Voltage definition",
            "Current": "Current definition",
        },
    )


class TestResultInit:
    """Test Result initialization."""

    def test_init(self, Result_fixture):
        """Test the __init__ method."""
        assert isinstance(Result_fixture, Result)
        assert isinstance(Result_fixture.lf, pl.LazyFrame)
        assert isinstance(Result_fixture.metadata, dict)

    def test_init_accepts_dataframe(self):
        """Test that DataFrame input is converted to LazyFrame at construction."""
        result = Result(lf=pl.DataFrame({"a": [1, 2, 3]}), metadata={})
        assert isinstance(result.lf, pl.LazyFrame)
        pl_testing.assert_frame_equal(result.data, pl.DataFrame({"a": [1, 2, 3]}))


class TestResultDataFrameProperty:
    """Test DataFrame property and setter."""

    def test_df(self, Result_fixture):
        """Test the df property."""
        df = Result_fixture.df
        assert isinstance(df, pl.DataFrame)
        pl_testing.assert_frame_equal(df, Result_fixture.lf.collect())

    def test_df_setter(self, Result_fixture):
        """Test the df setter."""
        new_df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        Result_fixture.df = new_df
        assert isinstance(Result_fixture.lf, pl.LazyFrame)
        pl_testing.assert_frame_equal(Result_fixture.lf.collect(), new_df)
        pl_testing.assert_frame_equal(Result_fixture.df, new_df)

    def test_collect(self, Result_fixture):
        """Test the collect method."""
        collected_df = Result_fixture.collect()
        assert isinstance(collected_df, pl.DataFrame)
        pl_testing.assert_frame_equal(collected_df, Result_fixture.data)
        assert isinstance(Result_fixture.lf, pl.LazyFrame)


class TestResultColumnResolution:
    """Test column resolution and unit conversion."""

    def test_can_resolve_valid(self, Result_fixture):
        """Test that known BDF columns are resolvable via ColumnSet."""
        col_set = Result_fixture.columns
        assert col_set.can_resolve("Current / A")
        assert col_set.can_resolve("Voltage / V")

    def test_can_resolve_missing(self, Result_fixture):
        """Test that an unknown column is not resolvable via ColumnSet."""
        col_set = Result_fixture.columns
        assert not col_set.can_resolve("NonExistent / A")

    def test_get_unit_conversion(self, Result_fixture):
        """Test that get() performs BDF-aware unit conversion."""
        current_ma = Result_fixture.get("Current / mA")
        np_testing.assert_allclose(
            current_ma,
            Result_fixture.data["Current / A"].to_numpy() * 1000,
            rtol=1e-5,
        )

    def test_get_missing_column_raises(self, Result_fixture):
        """Test that get() raises ValueError for nonexistent columns."""
        with pytest.raises(ValueError, match="Cannot resolve"):
            Result_fixture.get("NonExistent / A")

    def test_getitem_unit_conversion(self, Result_fixture):
        """Test that __getitem__() supports unit conversion via ColumnSet."""
        current_ma = Result_fixture["Current / mA"]
        assert isinstance(current_ma, Result)
        assert "Current / mA" in current_ma.columns
        np_testing.assert_allclose(
            current_ma.data["Current / mA"].to_numpy(),
            Result_fixture.data["Current / A"].to_numpy() * 1000,
            rtol=1e-5,
        )

    def test_getitem_missing_column_raises(self, Result_fixture):
        """Test that __getitem__() raises ValueError for nonexistent columns."""
        with pytest.raises(ValueError, match="Cannot resolve"):
            _ = Result_fixture["NonExistent / A"]

    def test_getitem_does_not_mutate_columns(self, Result_fixture):
        """Test that __getitem__() with unit conversion doesn't add column to result."""
        original_columns = set(Result_fixture.data.columns)
        _ = Result_fixture["Current / mA"]
        assert set(Result_fixture.data.columns) == original_columns

    def test_get(self, Result_fixture):
        """Test the get method."""
        current = Result_fixture.get("Current / A")
        np_testing.assert_array_equal(
            current,
            Result_fixture.data["Current / A"].to_numpy(),
        )

        current, voltage = Result_fixture.get("Current / A", "Voltage / V")
        np_testing.assert_array_equal(
            current,
            Result_fixture.data["Current / A"].to_numpy(),
        )
        np_testing.assert_array_equal(
            voltage,
            Result_fixture.data["Voltage / V"].to_numpy(),
        )

    def test_getitem(self, Result_fixture):
        """Test the __getitem__ method."""
        current = Result_fixture["Current / A"]
        assert "Current / A" in current.columns
        assert isinstance(current, Result)
        pl_testing.assert_frame_equal(
            current.data,
            Result_fixture.data.select("Current / A"),
        )


class TestResultDataProperty:
    """Test data property and metadata."""

    def test_data(self, Result_fixture):
        """Test the data property."""
        assert isinstance(Result_fixture.lf, pl.LazyFrame)
        assert isinstance(Result_fixture.data, pl.DataFrame)
        pl_testing.assert_frame_equal(Result_fixture.data, Result_fixture.lf.collect())

    def test_quantities(self, Result_fixture):
        """Test the quantities property."""
        assert set(Result_fixture.columns.quantities) == {
            "Test Time",
            "Current",
            "Voltage",
            "Net Capacity",
            "Step Count",
            "Step Index",
            "Unix Time",
        }

    def test_print_definitions(self, Result_fixture, capsys):
        """Test the print_definitions method."""
        Result_fixture.define_column("Voltage", "Voltage across the circuit")
        Result_fixture.define_column("Resistance", "Resistance of the circuit")
        Result_fixture.print_definitions()
        captured = capsys.readouterr()
        expected_output = (
            "{'Current': 'Current definition'"
            ",\n 'Resistance': 'Resistance of the circuit'"
            ",\n 'Voltage': 'Voltage across the circuit'}"
        )
        assert captured.out.strip() == expected_output


class TestResultBuild:
    """Test Result.build method."""

    def test_build(self):
        """Test the build method."""
        data1 = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        data2 = pl.DataFrame({"x": [7, 8, 9], "y": [10, 11, 12]})
        metadata = {"test": "metadata"}
        result = Result.build([data1, data2], metadata)
        assert isinstance(result, Result)
        expected_data = pl.DataFrame(
            {
                "x": [1, 2, 3, 7, 8, 9],
                "y": [4, 5, 6, 10, 11, 12],
                "Step": [0, 0, 0, 1, 1, 1],
                "Cycle": [0, 0, 0, 0, 0, 0],
            },
        )
        pl_testing.assert_frame_equal(
            result.data,
            expected_data,
            check_column_order=False,
            check_dtype=False,
        )


class TestAddDataBasic:
    """Test basic add_data functionality."""

    def test_add_data(self):
        """Test the add_data method."""
        base_time = datetime(1985, 1, 1, 0, 0, 0, tzinfo=UTC).timestamp()
        existing_data = pl.LazyFrame(
            {
                "Unix Time / s": np.array([base_time + i for i in range(6)]),
                "Data": [2, 4, 6, 8, 10, 12],
            },
        )
        new_data = pl.LazyFrame(
            {
                "DateTime": [
                    datetime(1985, 1, 1, 0, 0, 0),
                    datetime(1985, 1, 1, 0, 0, 1),
                    datetime(1985, 1, 1, 0, 0, 2),
                    datetime(1985, 1, 1, 0, 0, 3),
                    datetime(1985, 1, 1, 0, 0, 4),
                    datetime(1985, 1, 1, 0, 0, 5),
                ],
                "Data 1": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
                "Data 2": [4.0, 8.0, 12.0, 16.0, 20.0, 24.0],
            },
        )
        result_object = Result(lf=existing_data, metadata={})
        result_object.add_data(
            new_data,
            time_column_name="DateTime",
            timezone="UTC",
        )
        expected_data = pl.DataFrame(
            {
                "Unix Time / s": np.array([base_time + i for i in range(6)]),
                "Data": [2, 4, 6, 8, 10, 12],
                "Data 1": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
                "Data 2": [4.0, 8.0, 12.0, 16.0, 20.0, 24.0],
            },
        )
        pl_testing.assert_frame_equal(
            result_object.data,
            expected_data,
            check_column_order=False,
        )

    def test_add_data_with_format(self):
        """Test add_data with datetime format string."""
        base_time = datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC).timestamp()
        existing_data = pl.LazyFrame(
            {"Unix Time / s": np.array([base_time]), "Value": [1]}
        )

        new_data = pl.LazyFrame({"DateStr": ["2023/01/01 10:00:00"], "Ext": [10]})

        result = Result(lf=existing_data, metadata={})
        result.add_data(
            new_data,
            time_column_name="DateStr",
            datetime_format="%Y/%m/%d %H:%M:%S",
            timezone="UTC",
        )

        schema = result.lf.collect_schema()
        assert schema["Unix Time / s"] == pl.Float64

        data = result.data
        assert "Ext" in data.columns
        assert data["Ext"][0] == 10


class TestAddDataTimezoneHandling:
    """Test timezone handling with time difference verification."""

    def test_validate_timezone_valid(self):
        """Test _validate_timezone with valid timezone strings."""
        assert _validate_timezone("UTC") == "UTC"
        assert _validate_timezone("Europe/London") == "Europe/London"
        assert _validate_timezone("America/New_York") == "America/New_York"
        assert _validate_timezone("Asia/Tokyo") == "Asia/Tokyo"

    def test_validate_timezone_invalid(self):
        """Test _validate_timezone raises error for invalid timezone strings."""
        with pytest.raises(ValueError, match="Invalid timezone"):
            _validate_timezone("Invalid/Timezone")

        with pytest.raises(ValueError, match="Invalid timezone"):
            _validate_timezone("NotATimezone")

        with pytest.raises(ValueError, match="Invalid timezone"):
            _validate_timezone("GMT+5")

    def test_add_data_timezone_handling(self):
        """Test timezone handling in add_data."""
        base_time = datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC).timestamp()

        existing_data = pl.LazyFrame(
            {"Unix Time / s": np.array([base_time]), "Value": [1]}
        )

        new_data = pl.LazyFrame(
            {
                "DateUTC": [datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC)],
                "Ext": [10],
            }
        )

        result = Result(lf=existing_data, metadata={})
        result.add_data(new_data, time_column_name="DateUTC", timezone="UTC")

        schema = result.lf.collect_schema()
        assert schema["Unix Time / s"] == pl.Float64
        assert "Ext" in schema

    def test_add_data_timezone_difference_utc_vs_london(self):
        """Test time difference calculation between UTC and Europe/London."""
        # June 21, 2023: London is in BST (UTC+1)
        base_time_utc = datetime(2023, 6, 21, 12, 0, 0, tzinfo=UTC).timestamp()
        existing_data = pl.LazyFrame(
            {"Unix Time / s": np.array([base_time_utc]), "Value": [1]}
        )

        # Same wall clock time interpreted as London time
        new_data_london = pl.LazyFrame(
            {
                "DateTime": [datetime(2023, 6, 21, 12, 0, 0)],
                "Data": [10],
            }
        )

        result = Result(lf=existing_data, metadata={})
        result.add_data(
            new_data_london,
            time_column_name="DateTime",
            timezone="Europe/London",
            join_strategy="keep_both",
        )

        data = result.data
        unix_times = data["Unix Time / s"].to_numpy()

        # Verify time offset is correctly applied (3600 seconds in this direction)
        time_diff = unix_times[1] - unix_times[0]
        assert time_diff == pytest.approx(3600, abs=1)

    def test_add_data_timezone_difference_utc_vs_newyork(self):
        """Test time difference calculation between UTC and America/New_York."""
        # June 21, 2023: New York is in EDT (UTC-4)
        base_time_utc = datetime(2023, 6, 21, 12, 0, 0, tzinfo=UTC).timestamp()
        existing_data = pl.LazyFrame(
            {"Unix Time / s": np.array([base_time_utc]), "Value": [1]}
        )

        # Same wall clock time interpreted as New York time
        new_data_newyork = pl.LazyFrame(
            {
                "DateTime": [datetime(2023, 6, 21, 12, 0, 0)],
                "Data": [10],
            }
        )

        result = Result(lf=existing_data, metadata={})
        result.add_data(
            new_data_newyork,
            time_column_name="DateTime",
            timezone="America/New_York",
            join_strategy="keep_both",
        )

        data = result.data
        unix_times = data["Unix Time / s"].to_numpy()

        # Verify time offset is correctly applied
        time_diff = unix_times[1] - unix_times[0]
        assert abs(time_diff) == pytest.approx(4 * 3600, abs=1)

    def test_add_data_timezone_difference_multiple_timezones(self):
        """Test that times in different timezones are correctly aligned."""
        # March 21, 2023: New York is in EST (UTC-5)
        utc_time = datetime(2023, 3, 21, 12, 0, 0, tzinfo=UTC)
        base_time_utc = utc_time.timestamp()

        existing_data = pl.LazyFrame(
            {"Unix Time / s": np.array([base_time_utc]), "Value": [1]}
        )

        naive_time = datetime(2023, 3, 21, 12, 0, 0)

        new_data = pl.LazyFrame({"DateTime": [naive_time], "Data": [10]})

        result = Result(lf=existing_data, metadata={})
        result.add_data(
            new_data,
            time_column_name="DateTime",
            timezone="America/New_York",
            join_strategy="keep_both",
        )

        data = result.data
        unix_times = data["Unix Time / s"].to_numpy()

        # Verify timezone handling produces a significant time difference
        time_diff = unix_times[1] - unix_times[0]
        assert abs(time_diff) == pytest.approx(4 * 3600, abs=1)

    def test_add_data_timezone_difference_with_data_joining(self):
        """Test that timezone conversion is applied correctly during data joining."""
        utc_noon = datetime(2023, 6, 21, 12, 0, 0, tzinfo=UTC).timestamp()
        utc_1pm = datetime(2023, 6, 21, 13, 0, 0, tzinfo=UTC).timestamp()

        existing_data = pl.LazyFrame(
            {
                "Unix Time / s": np.array([utc_noon, utc_1pm]),
                "Temperature_UTC": [20.0, 21.0],
            }
        )

        # London BST is UTC+1, so 13:00 BST = 12:00 UTC and 14:00 BST = 13:00 UTC
        new_data = pl.LazyFrame(
            {
                "DateTime": [
                    datetime(2023, 6, 21, 13, 0, 0),  # 13:00 BST = 12:00 UTC
                    datetime(2023, 6, 21, 14, 0, 0),  # 14:00 BST = 13:00 UTC
                ],
                "Temperature_London": [20.0, 21.0],
            }
        )

        result = Result(lf=existing_data, metadata={})
        result.add_data(
            new_data,
            time_column_name="DateTime",
            timezone="Europe/London",
            join_strategy="keep_existing",
            fill_strategy=None,
        )

        data = result.data
        london_col = data["Temperature_London"]
        assert london_col[0] == 20.0
        assert london_col[1] == 21.0

    def test_add_data_invalid_timezone(self):
        """Test add_data raises error for invalid timezone."""
        base_time = datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC).timestamp()
        existing_data = pl.LazyFrame(
            {"Unix Time / s": np.array([base_time]), "Value": [1]}
        )
        new_data = pl.LazyFrame(
            {"DateNew": [datetime(2023, 1, 1, 10, 0, 0)], "Ext": [10]}
        )
        result = Result(lf=existing_data, metadata={})

        with pytest.raises(ValueError, match="Invalid timezone"):
            result.add_data(
                new_data,
                time_column_name="DateNew",
                timezone="Invalid/Timezone",
            )

    def test_tzlocal_returns_valid_timezone(self):
        """Test that tzlocal returns a valid IANA timezone that can be used."""
        local_tz = str(get_localzone())
        zone = ZoneInfo(local_tz)
        assert zone is not None

        df = pl.DataFrame({"Date": [datetime(2023, 1, 1, 10, 0, 0)]})
        df_with_tz = df.with_columns(pl.col("Date").dt.replace_time_zone(local_tz))
        assert df_with_tz["Date"].dtype.time_zone == local_tz

    def test_add_data_uses_local_timezone_when_not_specified(self):
        """Test that add_data uses UTC timezone behavior when converting datetimes."""
        base_time = datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC).timestamp()
        existing_data = pl.LazyFrame(
            {"Unix Time / s": np.array([base_time]), "Value": [1]}
        )
        new_data = pl.LazyFrame(
            {
                "DateUTC": [datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC)],
                "Ext": [10],
            }
        )

        result = Result(lf=existing_data, metadata={})
        result.add_data(new_data, time_column_name="DateUTC")

        schema = result.lf.collect_schema()
        assert schema["Unix Time / s"] == pl.Float64
        data = result.data
        assert len(data) > 0
        assert "Ext" in data.columns


class TestAddDataJoinStrategies:
    """Test add_data with different join strategies."""

    def test_add_data_join_strategy_keep_existing(self):
        """Test add_data with join_strategy='keep_existing'."""
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC).timestamp()
        existing_data = pl.LazyFrame(
            {
                "Unix Time / s": np.array([base_time + i for i in range(5)]),
                "Temperature": [20.0, 21.0, 22.0, 23.0, 24.0],
            },
        )
        new_data = pl.LazyFrame(
            {
                "DateTime": [
                    datetime(2024, 1, 1, 0, 0, 0),
                    datetime(2024, 1, 1, 0, 0, 2),
                    datetime(2024, 1, 1, 0, 0, 4),
                ],
                "Voltage": [3.6, 3.8, 4.0],
            },
        )

        result = Result(lf=existing_data, metadata={})
        result.add_data(
            new_data,
            time_column_name="DateTime",
            join_strategy="keep_existing",
            fill_strategy="interpolate",
            timezone="UTC",
        )

        data = result.data
        assert len(data) == 5
        assert "Temperature" in data.columns
        assert "Voltage" in data.columns

        assert data["Voltage"][0] == pytest.approx(3.6)
        assert data["Voltage"][1] == pytest.approx(3.7)
        assert data["Voltage"][2] == pytest.approx(3.8)
        assert data["Voltage"][3] == pytest.approx(3.9)
        assert data["Voltage"][4] == pytest.approx(4.0)

    def test_add_data_join_strategy_keep_new(self):
        """Test add_data with join_strategy='keep_new'."""
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC).timestamp()
        existing_data = pl.LazyFrame(
            {
                "Unix Time / s": np.array([base_time + i * 2 for i in range(3)]),
                "Temperature": [20.0, 22.0, 24.0],
            },
        )
        new_data = pl.LazyFrame(
            {
                "DateTime": [
                    datetime(2024, 1, 1, 0, 0, 0),
                    datetime(2024, 1, 1, 0, 0, 1),
                    datetime(2024, 1, 1, 0, 0, 2),
                    datetime(2024, 1, 1, 0, 0, 3),
                    datetime(2024, 1, 1, 0, 0, 4),
                ],
                "Voltage": [3.6, 3.7, 3.8, 3.9, 4.0],
            },
        )

        result = Result(lf=existing_data, metadata={})
        result.add_data(
            new_data,
            time_column_name="DateTime",
            join_strategy="keep_new",
            fill_strategy="interpolate",
            timezone="UTC",
        )

        data = result.data
        assert len(data) == 5
        assert data["Temperature"][0] == 20.0
        assert data["Temperature"][1] == 21.0
        assert data["Temperature"][2] == 22.0
        assert data["Temperature"][3] == 23.0
        assert data["Temperature"][4] == 24.0

    def test_add_data_join_strategy_keep_both(self):
        """Test add_data with join_strategy='keep_both'."""
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC).timestamp()
        existing_data = pl.LazyFrame(
            {
                "Unix Time / s": np.array([base_time + i for i in range(3)]),
                "Temperature": [20.0, 21.0, 22.0],
            },
        )
        new_data = pl.LazyFrame(
            {
                "DateTime": [
                    datetime(2024, 1, 1, 0, 0, 0, 500000),
                    datetime(2024, 1, 1, 0, 0, 1, 500000),
                ],
                "Voltage": [3.65, 3.85],
            },
        )

        result = Result(lf=existing_data, metadata={})
        result.add_data(
            new_data,
            time_column_name="DateTime",
            join_strategy="keep_both",
            fill_strategy="interpolate",
            timezone="UTC",
        )

        data = result.data
        assert len(data) >= 3
        assert "Temperature" in data.columns
        assert "Voltage" in data.columns

        assert data["Temperature"].null_count() < len(data)
        assert data["Voltage"].null_count() < len(data)


class TestAddDataFillStrategies:
    """Test add_data with different fill strategies."""

    def test_add_data_fill_strategy_forward_fill(self):
        """Test add_data with fill_strategy='forward_fill'."""
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC).timestamp()
        existing_data = pl.LazyFrame(
            {
                "Unix Time / s": np.array([base_time + i for i in range(6)]),
                "Temperature": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
            },
        )
        new_data = pl.LazyFrame(
            {
                "DateTime": [
                    datetime(2024, 1, 1, 0, 0, 1),
                    datetime(2024, 1, 1, 0, 0, 4),
                ],
                "Voltage": [3.7, 4.0],
            },
        )

        result = Result(lf=existing_data, metadata={})
        result.add_data(
            new_data,
            time_column_name="DateTime",
            join_strategy="keep_existing",
            fill_strategy="forward_fill",
            timezone="UTC",
        )

        data = result.data
        assert data["Voltage"][0] is None
        assert data["Voltage"][1] == 3.7
        assert data["Voltage"][2] == 3.7
        assert data["Voltage"][3] == 3.7
        assert data["Voltage"][4] == 4.0
        assert data["Voltage"][5] == 4.0

    def test_add_data_fill_strategy_backward_fill(self):
        """Test add_data with fill_strategy='backward_fill'."""
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC).timestamp()
        existing_data = pl.LazyFrame(
            {
                "Unix Time / s": np.array([base_time + i for i in range(6)]),
                "Temperature": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
            },
        )
        new_data = pl.LazyFrame(
            {
                "DateTime": [
                    datetime(2024, 1, 1, 0, 0, 1),
                    datetime(2024, 1, 1, 0, 0, 4),
                ],
                "Voltage": [3.7, 4.0],
            },
        )

        result = Result(lf=existing_data, metadata={})
        result.add_data(
            new_data,
            time_column_name="DateTime",
            join_strategy="keep_existing",
            fill_strategy="backward_fill",
            timezone="UTC",
        )

        data = result.data
        assert data["Voltage"][0] == 3.7
        assert data["Voltage"][1] == 3.7
        assert data["Voltage"][2] == 4.0
        assert data["Voltage"][3] == 4.0
        assert data["Voltage"][4] == 4.0
        assert data["Voltage"][5] is None

    def test_add_data_fill_strategy_none(self):
        """Test add_data with fill_strategy=None."""
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC).timestamp()
        existing_data = pl.LazyFrame(
            {
                "Unix Time / s": np.array([base_time + i for i in range(5)]),
                "Temperature": [20.0, 21.0, 22.0, 23.0, 24.0],
            },
        )
        new_data = pl.LazyFrame(
            {
                "DateTime": [
                    datetime(2024, 1, 1, 0, 0, 0),
                    datetime(2024, 1, 1, 0, 0, 2),
                    datetime(2024, 1, 1, 0, 0, 4),
                ],
                "Voltage": [3.6, 3.8, 4.0],
            },
        )

        result = Result(lf=existing_data, metadata={})
        result.add_data(
            new_data,
            time_column_name="DateTime",
            join_strategy="keep_existing",
            fill_strategy=None,
            timezone="UTC",
        )

        data = result.data
        assert data["Voltage"][0] == 3.6
        assert data["Voltage"][1] is None
        assert data["Voltage"][2] == 3.8
        assert data["Voltage"][3] is None
        assert data["Voltage"][4] == 4.0


class TestAddDataValidation:
    """Test add_data validation and error handling."""

    def test_add_data_invalid_join_strategy_raises(self):
        """Test add_data with an invalid join strategy."""
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC).timestamp()
        existing_data = pl.LazyFrame(
            {
                "Unix Time / s": np.array([base_time]),
                "Temperature": [20.0],
            },
        )
        new_data = pl.LazyFrame(
            {
                "DateTime": [datetime(2024, 1, 1, 0, 0, 0)],
                "Voltage": [3.7],
            },
        )

        result = Result(lf=existing_data, metadata={})
        with pytest.raises(
            ValueError,
            match=(
                r"^Unsupported join_strategy: 'bad_strategy'\. "
                r"Expected one of: 'keep_existing', 'keep_new', 'keep_both'\.$"
            ),
        ):
            result.add_data(
                new_data,
                time_column_name="DateTime",
                join_strategy="bad_strategy",
                timezone="UTC",
            )

    def test_add_data_invalid_fill_strategy_raises(self):
        """Test add_data with an invalid fill strategy."""
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC).timestamp()
        existing_data = pl.LazyFrame(
            {
                "Unix Time / s": np.array([base_time]),
                "Temperature": [20.0],
            },
        )
        new_data = pl.LazyFrame(
            {
                "DateTime": [datetime(2024, 1, 1, 0, 0, 0)],
                "Voltage": [3.7],
            },
        )

        result = Result(lf=existing_data, metadata={})
        with pytest.raises(
            ValueError,
            match=(
                r"^Unsupported fill_strategy: 'bad_strategy'\. "
                r"Valid options are None, 'interpolate', 'forward_fill', "
                r"'backward_fill'\.$"
            ),
        ):
            result.add_data(
                new_data,
                time_column_name="DateTime",
                fill_strategy="bad_strategy",
                timezone="UTC",
            )


class TestAddDataComplexScenarios:
    """Test add_data with complex scenarios."""

    def test_add_data_combined_strategies(self):
        """Test add_data with combined join and fill strategies."""
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC).timestamp()
        existing_data = pl.LazyFrame(
            {
                "Unix Time / s": np.array([base_time + i * 2 for i in range(3)]),
                "Temperature": [20.0, 22.0, 24.0],
            },
        )
        new_data = pl.LazyFrame(
            {
                "DateTime": [
                    datetime(2024, 1, 1, 0, 0, 1),
                    datetime(2024, 1, 1, 0, 0, 3),
                    datetime(2024, 1, 1, 0, 0, 5),
                ],
                "Voltage": [3.7, 3.9, 4.1],
            },
        )

        result = Result(lf=existing_data, metadata={})
        result.add_data(
            new_data,
            time_column_name="DateTime",
            join_strategy="keep_both",
            fill_strategy="forward_fill",
            timezone="UTC",
        )

        data = result.data
        assert len(data) == 6

    @pytest.mark.parametrize(
        (
            "join_strategy",
            "fill_strategy",
            "expected_length",
            "check_column",
            "check_second",
            "expected_value",
        ),
        [
            ("keep_existing", "interpolate", 3, "Voltage", 2, 3.8),
            ("keep_existing", "forward_fill", 3, "Voltage", 2, 3.7),
            ("keep_existing", "backward_fill", 3, "Voltage", 2, 3.9),
            ("keep_existing", None, 3, "Voltage", 2, None),
            ("keep_new", "interpolate", 3, "Temperature", 3, 23.0),
            ("keep_new", "forward_fill", 3, "Temperature", 3, 22.0),
            ("keep_new", "backward_fill", 3, "Temperature", 3, 24.0),
            ("keep_new", None, 3, "Temperature", 3, None),
            ("keep_both", "interpolate", 6, "Voltage", 2, 3.8),
            ("keep_both", "forward_fill", 6, "Voltage", 2, 3.7),
            ("keep_both", "backward_fill", 6, "Voltage", 2, 3.9),
            ("keep_both", None, 6, "Voltage", 2, None),
        ],
    )
    def test_add_data_all_join_fill_strategy_combinations(
        self,
        join_strategy,
        fill_strategy,
        expected_length,
        check_column,
        check_second,
        expected_value,
    ):
        """Test all join_strategy x fill_strategy combinations for add_data."""
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC).timestamp()
        existing_data = pl.LazyFrame(
            {
                "Unix Time / s": np.array([base_time, base_time + 2, base_time + 4]),
                "Temperature": [20.0, 22.0, 24.0],
            },
        )
        new_data = pl.LazyFrame(
            {
                "DateTime": [
                    datetime(2024, 1, 1, 0, 0, 1),
                    datetime(2024, 1, 1, 0, 0, 3),
                    datetime(2024, 1, 1, 0, 0, 5),
                ],
                "Voltage": [3.7, 3.9, 4.1],
            },
        )

        result = Result(lf=existing_data, metadata={})
        result.add_data(
            new_data,
            time_column_name="DateTime",
            join_strategy=join_strategy,
            fill_strategy=fill_strategy,
            timezone="UTC",
        )

        data = result.data
        assert len(data) == expected_length

        check_time = base_time + check_second
        row = data.filter(
            (pl.col("Unix Time / s") >= check_time - 0.1)
            & (pl.col("Unix Time / s") <= check_time + 0.1)
        )
        assert len(row) >= 1
        actual_value = row[check_column][0]
        if expected_value is None:
            assert actual_value is None or np.isnan(actual_value)
        else:
            assert actual_value == pytest.approx(expected_value, abs=0.2)


class TestAddDataColumnMapping:
    """Test add_data with column mapping."""

    def test_add_data_with_column_map(self):
        """Test add_data with column_map parameter."""
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC).timestamp()
        existing_data = pl.LazyFrame(
            {
                "Unix Time / s": np.array([base_time + i for i in range(5)]),
                "Voltage / V": [3.6, 3.7, 3.8, 3.9, 4.0],
            },
        )

        new_data = pl.LazyFrame(
            {
                "DateTime": [
                    datetime(2024, 1, 1, 0, 0, 0),
                    datetime(2024, 1, 1, 0, 0, 1),
                    datetime(2024, 1, 1, 0, 0, 2),
                    datetime(2024, 1, 1, 0, 0, 3),
                    datetime(2024, 1, 1, 0, 0, 4),
                ],
                "RawCurrent": [0.1, 0.2, 0.3, 0.4, 0.5],
                "RawTemperature": [20.0, 20.5, 21.0, 21.5, 22.0],
            },
        )

        result = Result(lf=existing_data, metadata={})
        result.add_data(
            new_data,
            time_column_name="DateTime",
            column_map={
                "Current / A": "RawCurrent",
                "Temperature / degC": "RawTemperature",
            },
            timezone="UTC",
        )

        data = result.data
        assert "Current / A" in data.columns
        assert "Temperature / degC" in data.columns
        assert "RawCurrent" not in data.columns
        assert "RawTemperature" not in data.columns

        assert data["Current / A"][0] == 0.1
        assert data["Temperature / degC"][0] == 20.0

    def test_add_data_with_column_map_interpolation(self):
        """Test add_data with column_map combined with interpolation."""
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC).timestamp()
        existing_data = pl.LazyFrame(
            {
                "Unix Time / s": np.array([base_time + i for i in range(6)]),
                "Voltage / V": [3.6, 3.7, 3.8, 3.9, 4.0, 4.1],
            },
        )

        new_data = pl.LazyFrame(
            {
                "DateTime": [
                    datetime(2024, 1, 1, 0, 0, 0),
                    datetime(2024, 1, 1, 0, 0, 2),
                    datetime(2024, 1, 1, 0, 0, 4),
                ],
                "SensorValue": [20.0, 22.0, 24.0],
            },
        )

        result = Result(lf=existing_data, metadata={})
        result.add_data(
            new_data,
            time_column_name="DateTime",
            column_map={"Temperature / degC": "SensorValue"},
            join_strategy="keep_existing",
            fill_strategy="interpolate",
            timezone="UTC",
        )

        data = result.data
        assert "Temperature / degC" in data.columns
        assert len(data) == 6

        temp = data["Temperature / degC"]
        assert temp[0] == 20.0
        assert temp[1] == pytest.approx(21.0)
        assert temp[2] == 22.0
        assert temp[3] == pytest.approx(23.0)
        assert temp[4] == 24.0
        assert temp[5] is None

    def test_add_data_with_multiple_column_maps(self):
        """Test add_data with multiple column mappings."""
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC).timestamp()
        existing_data = pl.LazyFrame(
            {
                "Unix Time / s": np.array([base_time + i for i in range(3)]),
                "Voltage / V": [3.6, 3.8, 4.0],
            },
        )

        new_data = pl.LazyFrame(
            {
                "DateTime": [
                    datetime(2024, 1, 1, 0, 0, 0),
                    datetime(2024, 1, 1, 0, 0, 1),
                    datetime(2024, 1, 1, 0, 0, 2),
                ],
                "I": [0.1, 0.2, 0.3],
                "T": [20.0, 21.0, 22.0],
                "P": [100.0, 101.0, 102.0],
            },
        )

        result = Result(lf=existing_data, metadata={})
        result.add_data(
            new_data,
            time_column_name="DateTime",
            column_map={
                "Current / A": "I",
                "Temperature / degC": "T",
                "Pressure / Pa": "P",
            },
            timezone="UTC",
        )

        data = result.data
        assert "Current / A" in data.columns
        assert "Temperature / degC" in data.columns
        assert "Pressure / Pa" in data.columns
        assert "I" not in data.columns
        assert "T" not in data.columns
        assert "P" not in data.columns


class TestAddDataAlignment:
    """Test add_data with alignment parameters."""

    def test_add_data_with_alignment(self):
        """Test add_data with the align_on parameter."""
        base_df = pl.DataFrame(
            {
                "Unix Time / s": [0.0, 1.0, 2.0],
                "Value [V]": [1.0, 2.0, 3.0],
            }
        )

        new_df = pl.DataFrame(
            {
                "Time [s]": [
                    datetime(1970, 1, 1, 0, 0, 0, 500000),
                    datetime(1970, 1, 1, 0, 0, 1, 500000),
                    datetime(1970, 1, 1, 0, 0, 2, 500000),
                ],
                "Other [A]": [1.5, 2.5, 3.5],
            }
        )

        result = Result(lf=base_df.lazy(), metadata={})

        result.add_data(
            new_df,
            time_column_name="Time [s]",
            timezone="UTC",
        )

        combined_df = result.data

        assert "Other [A]" in combined_df.columns
        assert len(combined_df) > 0

    def test_add_data_with_alignment_error(self):
        """Test add_data with invalid align_on columns."""
        base_df = pl.DataFrame(
            {
                "Test Time [s]": [0.0],
                "Value [V]": [1.0],
            }
        )
        new_df = pl.DataFrame(
            {
                "Time [s]": [0.0],
                "Other [A]": [1.0],
            }
        )
        result = Result(lf=base_df.lazy(), metadata={})

        with pytest.raises(ValueError):
            result.add_data(
                new_df,
                time_column_name="Time [s]",
                align_on=("NonExistent [V]", "Other [A]"),
                timezone="UTC",
            )

        with pytest.raises(ValueError):
            result.add_data(
                new_df,
                time_column_name="Time [s]",
                align_on=("Value [V]", "NonExistent [A]"),
                timezone="UTC",
            )


class TestResultFrameOperations:
    """Test Result frame operations like join, extend, combine."""

    def test_verify_compatible_frames(self):
        """Test the _verify_compatible_frames method."""
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"b": [4, 5, 6]})
        lazy_df1 = df1.lazy()
        lazy_df2 = df2.lazy()

        result1, result2 = Result._verify_compatible_frames(df1, [df2])
        assert isinstance(result1, pl.DataFrame)
        assert isinstance(result2[0], pl.DataFrame)

        result1, result2 = Result._verify_compatible_frames(df1, [lazy_df2])
        assert isinstance(result1, pl.DataFrame)
        assert isinstance(result2[0], pl.DataFrame)

        result1, result2 = Result._verify_compatible_frames(lazy_df1, [df2])
        assert isinstance(result1, pl.DataFrame)
        assert isinstance(result2[0], pl.DataFrame)

        result1, result2 = Result._verify_compatible_frames(
            lazy_df1,
            [lazy_df2],
            mode="collect all",
        )
        assert isinstance(result1, pl.LazyFrame)
        assert isinstance(result2[0], pl.LazyFrame)

        result1, result2 = Result._verify_compatible_frames(
            lazy_df1, [df2], mode="match 1"
        )
        assert isinstance(result1, pl.LazyFrame)
        assert isinstance(result2[0], pl.LazyFrame)

        result1, result2 = Result._verify_compatible_frames(
            df1, [lazy_df2], mode="match 1"
        )
        assert isinstance(result1, pl.DataFrame)
        assert isinstance(result2[0], pl.DataFrame)

        result1, result2 = Result._verify_compatible_frames(df1, [df2, lazy_df2])
        assert isinstance(result1, pl.DataFrame)
        assert isinstance(result2[0], pl.DataFrame)
        assert isinstance(result2[1], pl.DataFrame)

        result1, result2 = Result._verify_compatible_frames(
            lazy_df1,
            [df2, lazy_df2],
            mode="match 1",
        )
        assert isinstance(result1, pl.LazyFrame)
        assert isinstance(result2[0], pl.LazyFrame)
        assert isinstance(result2[1], pl.LazyFrame)

    def test_join_left(self, reduced_result_fixture):
        """Test the join method with left join."""
        other_data = pl.DataFrame(
            {
                "Current [A]": [1, 2, 3],
                "Capacity [Ah]": [4, 5, 6],
            },
        )
        other_result = Result(
            lf=other_data.lazy(),
            metadata={"test": "metadata"},
            column_definitions={"Voltage": "Voltage definition"},
        )
        reduced_result_fixture.join(other_result, on="Current [A]", how="left")
        expected_data = pl.DataFrame(
            {
                "Current [A]": [1, 2, 3],
                "Voltage [V]": [1, 2, 3],
                "Capacity [Ah]": [4, 5, 6],
            },
        )
        pl_testing.assert_frame_equal(
            reduced_result_fixture.data,
            expected_data,
            check_column_order=False,
        )
        assert (
            reduced_result_fixture.column_definitions["Voltage"] == "Voltage definition"
        )

    def test_extend(self, reduced_result_fixture):
        """Test the extend method."""
        other_data = pl.DataFrame(
            {
                "Current [A]": [4, 5, 6],
                "Voltage [V]": [4, 5, 6],
            },
        )
        other_result = Result(
            lf=other_data.lazy(),
            metadata={"test": "metadata"},
            column_definitions={"Voltage": "Voltage definition"},
        )
        reduced_result_fixture.extend(other_result)
        expected_data = pl.DataFrame(
            {
                "Current [A]": [1, 2, 3, 4, 5, 6],
                "Voltage [V]": [1, 2, 3, 4, 5, 6],
            },
        )
        pl_testing.assert_frame_equal(
            reduced_result_fixture.data,
            expected_data,
            check_column_order=False,
        )
        assert (
            reduced_result_fixture.column_definitions["Voltage"] == "Voltage definition"
        )

    def test_extend_with_new_columns(self, reduced_result_fixture):
        """Test the extend method with new columns."""
        other_data = pl.DataFrame(
            {
                "Current [A]": [4, 5, 6],
                "Voltage [V]": [4, 5, 6],
                "Capacity [Ah]": [8, 9, 10],
            },
        )
        other_result = Result(
            lf=other_data.lazy(),
            metadata={"test": "metadata"},
            column_definitions={
                "Voltage": "New voltage definition",
                "Capacity": "Capacity definition",
                "Current": "Current definition",
            },
        )
        reduced_result_fixture.extend(other_result)
        expected_data = pl.DataFrame(
            {
                "Current [A]": [1, 2, 3, 4, 5, 6],
                "Voltage [V]": [1, 2, 3, 4, 5, 6],
                "Capacity [Ah]": [None, None, None, 8, 9, 10],
            },
        )
        pl_testing.assert_frame_equal(
            reduced_result_fixture.data,
            expected_data,
            check_column_order=False,
        )
        assert (
            reduced_result_fixture.column_definitions["Voltage"] == "Voltage definition"
        )
        assert (
            reduced_result_fixture.column_definitions["Capacity"]
            == "Capacity definition"
        )
        assert (
            reduced_result_fixture.column_definitions["Current"] == "Current definition"
        )

    def test_combine_results(self):
        """Test the combine results method."""
        result1 = Result(
            lf=pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).lazy(),
            metadata={"test index": 1.0},
        )
        result2 = Result(
            lf=pl.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]}).lazy(),
            metadata={"test index": 2.0},
        )
        combined_result = combine_results([result1, result2])
        expected_data = pl.DataFrame(
            {
                "a": [1, 2, 3, 7, 8, 9],
                "b": [4, 5, 6, 10, 11, 12],
                "test index": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            },
        )
        pl_testing.assert_frame_equal(
            combined_result.data,
            expected_data,
            check_column_order=False,
        )


class TestResultCleanCopy:
    """Test Result.clean_copy method."""

    def test_clean_copy(self, reduced_result_fixture):
        """Test the clean_copy method."""
        clean_result = reduced_result_fixture.clean_copy()
        assert isinstance(clean_result, Result)
        assert clean_result.lf.collect().is_empty()
        assert clean_result.metadata == reduced_result_fixture.metadata
        assert clean_result.column_definitions == {}

        new_df = pl.DataFrame({"Test [V]": [1, 2, 3]})
        clean_result = reduced_result_fixture.clean_copy(dataframe=new_df)
        assert isinstance(clean_result, Result)
        pl_testing.assert_frame_equal(clean_result.data, new_df)
        assert clean_result.metadata == reduced_result_fixture.metadata
        assert clean_result.column_definitions == {}

        new_defs = {"New Column [A]": "New definition"}
        clean_result = reduced_result_fixture.clean_copy(column_definitions=new_defs)
        assert isinstance(clean_result, Result)
        assert clean_result.lf.collect().is_empty()
        assert clean_result.metadata == reduced_result_fixture.metadata
        assert clean_result.column_definitions == new_defs

        clean_result = reduced_result_fixture.clean_copy(
            dataframe=new_df,
            column_definitions=new_defs,
        )
        assert isinstance(clean_result, Result)
        pl_testing.assert_frame_equal(clean_result.data, new_df)
        assert clean_result.metadata == reduced_result_fixture.metadata
        assert clean_result.column_definitions == new_defs

        lazy_df = new_df.lazy()
        clean_result = reduced_result_fixture.clean_copy(dataframe=lazy_df)
        assert isinstance(clean_result, Result)
        assert isinstance(clean_result.lf, pl.LazyFrame)
        pl_testing.assert_frame_equal(clean_result.data, new_df)


class TestResultExport:
    """Test Result export methods."""

    def test_export_to_mat(self, Result_fixture, tmp_path):
        """Test the export to mat function."""
        mat_path = tmp_path / "test_mat.mat"
        Result_fixture.export_to_mat(str(mat_path))
        saved_data = loadmat(str(mat_path))
        assert "data" in saved_data
        assert "metadata" in saved_data
        expected_columns = {
            "Current___A",
            "Voltage___V",
            "Test_Time___s",
            "Net_Capacity___Ah",
            "Step_Count___1",
            "Step_Index___1",
            "Unix_Time___s",
        }
        actual_columns = set(saved_data["data"].dtype.names)
        assert actual_columns == expected_columns


class TestResultPolarsIO:
    """Test Result Polars I/O methods."""

    def test_from_polars_io(self, tmp_path):
        """Test the from_polars_io method."""
        test_df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        csv_path = tmp_path / "test_data.csv"
        test_df.write_csv(csv_path)

        result = Result.from_polars_io(
            metadata={"test": "metadata"},
            column_definitions={"a": "Column A"},
            polars_io_func=pl.read_csv,
            source=str(csv_path),
        )
        assert isinstance(result, Result)
        assert result.metadata == {"test": "metadata"}
        assert result.column_definitions == {"a": "Column A"}
        pl_testing.assert_frame_equal(result.data, test_df)

        result_lazy = Result.from_polars_io(
            metadata={"test": "lazy"},
            column_definitions={},
            polars_io_func=pl.scan_csv,
            source=str(csv_path),
        )
        assert isinstance(result_lazy, Result)
        assert isinstance(result_lazy.lf, pl.LazyFrame)

        result_with_kwargs = Result.from_polars_io(
            metadata={"test": "kwargs"},
            column_definitions={"a": "Column A with kwargs"},
            polars_io_func=pl.read_csv,
            source=str(csv_path),
            has_header=True,
            skip_rows=0,
        )
        assert isinstance(result_with_kwargs, Result)
        pl_testing.assert_frame_equal(result_with_kwargs.data, test_df)

    @pytest.mark.parametrize(
        "io_function,expected_type",
        [
            (pl.read_csv, pl.DataFrame),
            (pl.scan_csv, pl.LazyFrame),
            (pl.read_parquet, pl.DataFrame),
            (pl.scan_parquet, pl.LazyFrame),
        ],
    )
    def test_from_polars_io_different_formats(
        self, io_function, expected_type, tmp_path
    ):
        """Test from_polars_io with different polars I/O functions."""
        test_df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        if "csv" in io_function.__name__:
            test_file = tmp_path / "test.csv"
            test_df.write_csv(test_file)
        else:
            test_file = tmp_path / "test.parquet"
            test_df.write_parquet(test_file)

        metadata = {"source": io_function.__name__}

        result = Result.from_polars_io(
            polars_io_func=io_function,
            source=test_file,
            metadata=metadata,
            column_definitions={},
        )

        assert isinstance(result, Result)
        assert isinstance(result.lf, pl.LazyFrame)
        assert result.metadata == metadata
        pl_testing.assert_frame_equal(result.data, test_df, check_column_order=False)

    def test_from_polars_io_python_object(self):
        """Test from_polars_io with a Python object."""
        test_df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        metadata = {"source": "python_object"}

        result = Result.from_polars_io(
            polars_io_func=pl.from_pandas,
            data=test_df.to_pandas(),
            metadata=metadata,
            column_definitions={},
        )

        assert isinstance(result, Result)
        assert isinstance(result.lf, pl.LazyFrame)
        assert result.metadata == metadata
        pl_testing.assert_frame_equal(result.data, test_df, check_column_order=False)

        result = Result.from_polars_io(
            polars_io_func=pl.from_numpy,
            schema=["a", "b"],
            data=test_df.to_numpy(),
            metadata=metadata,
            column_definitions={},
        )

        assert isinstance(result, Result)
        assert isinstance(result.lf, pl.LazyFrame)
        assert result.metadata == metadata
        pl_testing.assert_frame_equal(result.data, test_df, check_column_order=False)


class TestDeprecatedProperties:
    """Test deprecated Result properties."""

    def test_base_dataframe_deprecated_property(self, Result_fixture, caplog):
        """Test that base_dataframe property is deprecated."""
        import logging

        with caplog.at_level(logging.WARNING):
            _ = Result_fixture.base_dataframe
        assert "base_dataframe" in caplog.text
        assert "deprecated" in caplog.text

    def test_base_dataframe_setter_deprecated(self, Result_fixture, caplog):
        """Test that base_dataframe setter is deprecated."""
        import logging

        new_lf = pl.LazyFrame({"a": [1, 2, 3]})
        with caplog.at_level(logging.WARNING):
            Result_fixture.base_dataframe = new_lf
        assert "base_dataframe" in caplog.text
        assert "deprecated" in caplog.text

    def test_live_dataframe_deprecated_property(self, Result_fixture, caplog):
        """Test that live_dataframe property is deprecated."""
        import logging

        with caplog.at_level(logging.WARNING):
            _ = Result_fixture.live_dataframe
        assert "live_dataframe" in caplog.text
        assert "deprecated" in caplog.text

    def test_live_dataframe_setter_deprecated(self, Result_fixture, caplog):
        """Test that live_dataframe setter is deprecated."""
        import logging

        new_lf = pl.LazyFrame({"a": [1, 2, 3]})
        with caplog.at_level(logging.WARNING):
            Result_fixture.live_dataframe = new_lf
        assert "live_dataframe" in caplog.text
        assert "deprecated" in caplog.text
