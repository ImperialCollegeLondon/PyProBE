"""Tests for the result module - organized into logical test classes."""

import warnings
from datetime import UTC, datetime

import bdf
import numpy as np
import numpy.testing as np_testing
import polars as pl
import polars.testing as pl_testing
import pytest
from scipy.interpolate import (
    Akima1DInterpolator,
    CubicSpline,
    PchipInterpolator,
    make_smoothing_spline,
)
from scipy.io import loadmat

from pyprobe.analysis import differentiation, smoothing
from pyprobe.columns import BDF, Column, ColumnResolutionError
from pyprobe.rawdata import CyclingData, RawData
from pyprobe.result import (
    Curve,
    Quantified,
    Result,
    Table,
    combine_results,
)
from tests.metadata_helpers import build_metadata, read_extras


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
        metadata=build_metadata(test="metadata"),
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
        assert isinstance(Result_fixture.metadata, bdf.Metadata)

    def test_init_accepts_dataframe(self):
        """Test that DataFrame input is converted to LazyFrame at construction."""
        result = Result(lf=pl.DataFrame({"a": [1, 2, 3]}), metadata=build_metadata())
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
        """Test that known BDF columns are resolvable via ColumnDict."""
        col_set = Result_fixture.columns
        assert col_set.can_resolve("Current / A")
        assert col_set.can_resolve("Voltage / V")

    def test_can_resolve_missing(self, Result_fixture):
        """Test that an unknown column is not resolvable via ColumnDict."""
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
        """Test that __getitem__() supports unit conversion via ColumnDict."""
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

    def test_get_with_column_instance(self, Result_fixture):
        """Test that get() accepts Column and BDF instances."""
        current_str = Result_fixture.get("Current / A")
        current_bdf = Result_fixture.get(BDF.CURRENT_AMPERE)
        current_col = Result_fixture.get(Column("Current", "A"))
        np_testing.assert_array_equal(current_bdf, current_str)
        np_testing.assert_array_equal(current_col, current_str)

    def test_getitem_with_column_instance(self, Result_fixture):
        """Test that __getitem__() accepts Column and BDF instances."""
        by_str = Result_fixture["Current / A"]
        by_bdf = Result_fixture[BDF.CURRENT_AMPERE]
        pl_testing.assert_frame_equal(by_bdf.data, by_str.data)

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
            "Unix Time",
            "Test Time",
            "Current",
            "Voltage",
            "Net Capacity",
            "Step Count",
            "Step ID",
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
        result_object = Result(lf=existing_data, metadata=build_metadata())
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

        result = Result(lf=existing_data, metadata=build_metadata())
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

        result = Result(lf=existing_data, metadata=build_metadata())
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

        result = Result(lf=existing_data, metadata=build_metadata())
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

        result = Result(lf=existing_data, metadata=build_metadata())
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

        result = Result(lf=existing_data, metadata=build_metadata())
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

        result = Result(lf=existing_data, metadata=build_metadata())
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
        result = Result(lf=existing_data, metadata=build_metadata())

        with pytest.raises(ValueError, match="Invalid timezone"):
            result.add_data(
                new_data,
                time_column_name="DateNew",
                timezone="Invalid/Timezone",
            )

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

        result = Result(lf=existing_data, metadata=build_metadata())
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

        result = Result(lf=existing_data, metadata=build_metadata())
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

        result = Result(lf=existing_data, metadata=build_metadata())
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

        result = Result(lf=existing_data, metadata=build_metadata())
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

        result = Result(lf=existing_data, metadata=build_metadata())
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

        result = Result(lf=existing_data, metadata=build_metadata())
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

        result = Result(lf=existing_data, metadata=build_metadata())
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

        result = Result(lf=existing_data, metadata=build_metadata())
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

        result = Result(lf=existing_data, metadata=build_metadata())
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

        result = Result(lf=existing_data, metadata=build_metadata())
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

        result = Result(lf=existing_data, metadata=build_metadata())
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

        result = Result(lf=existing_data, metadata=build_metadata())
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

        result = Result(lf=existing_data, metadata=build_metadata())
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

        result = Result(lf=existing_data, metadata=build_metadata())
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

        result = Result(lf=base_df.lazy(), metadata=build_metadata())

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
        result = Result(lf=base_df.lazy(), metadata=build_metadata())

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
            metadata=build_metadata(test="metadata"),
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
            metadata=build_metadata(test="metadata"),
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
            metadata=build_metadata(test="metadata"),
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
            metadata=build_metadata(**{"test index": 1.0}),
        )
        result2 = Result(
            lf=pl.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]}).lazy(),
            metadata=build_metadata(**{"test index": 2.0}),
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
            "Step_ID",
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
            metadata=build_metadata(test="metadata"),
            column_definitions={"a": "Column A"},
            polars_io_func=pl.read_csv,
            source=str(csv_path),
        )
        assert isinstance(result, Result)
        assert result.info == {"test": "metadata"}
        assert result.column_definitions == {"a": "Column A"}
        pl_testing.assert_frame_equal(result.data, test_df)

        result_lazy = Result.from_polars_io(
            metadata=build_metadata(test="lazy"),
            column_definitions={},
            polars_io_func=pl.scan_csv,
            source=str(csv_path),
        )
        assert isinstance(result_lazy, Result)
        assert isinstance(result_lazy.lf, pl.LazyFrame)

        result_with_kwargs = Result.from_polars_io(
            metadata=build_metadata(test="kwargs"),
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

        metadata = build_metadata(source=io_function.__name__)

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

        metadata = build_metadata(source="python_object")

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

    def test_base_dataframe_deprecated_property(self, Result_fixture):
        """Test that base_dataframe property is deprecated."""
        with pytest.warns(DeprecationWarning, match="base_dataframe"):
            _ = Result_fixture.base_dataframe

    def test_base_dataframe_setter_deprecated(self, Result_fixture):
        """Test that base_dataframe setter is deprecated."""
        new_lf = pl.LazyFrame({"a": [1, 2, 3]})
        with pytest.warns(DeprecationWarning, match="base_dataframe"):
            Result_fixture.base_dataframe = new_lf

    def test_live_dataframe_deprecated_property(self, Result_fixture):
        """Test that live_dataframe property is deprecated."""
        with pytest.warns(DeprecationWarning, match="live_dataframe"):
            _ = Result_fixture.live_dataframe

    def test_live_dataframe_setter_deprecated(self, Result_fixture):
        """Test that live_dataframe setter is deprecated."""
        new_lf = pl.LazyFrame({"a": [1, 2, 3]})
        with pytest.warns(DeprecationWarning, match="live_dataframe"):
            Result_fixture.live_dataframe = new_lf


class TestQuantifiedTraitAndAlias:
    """Tests for the Quantified trait and the deprecated Result alias."""

    def test_table_satisfies_quantified(self):
        """A Table exposes columns, metadata, and column_definitions."""
        table = Table(lf=pl.LazyFrame({"Current / A": [1.0, 2.0]}))
        assert isinstance(table, Quantified)
        assert table.columns.names == ("Current / A",)
        assert isinstance(table.metadata, bdf.Metadata)
        assert isinstance(table.column_definitions, dict)

    def test_result_alias_resolves_to_table_and_warns(self):
        """Constructing via Result warns but yields a working Table."""
        with pytest.warns(DeprecationWarning):
            result = Result(lf=pl.LazyFrame({"Current / A": [1.0]}))
        assert isinstance(result, Table)
        assert result.columns.names == ("Current / A",)

    def test_table_instance_is_result_instance(self):
        """isinstance(table, Result) stays True for the deprecated alias."""
        table = Table(lf=pl.LazyFrame({"Current / A": [1.0]}))
        assert isinstance(table, Result)


@pytest.fixture
def reduction_table():
    """Table with numeric, integer, and string columns for reduction tests."""
    return Table(
        lf=pl.LazyFrame(
            {
                "Net Capacity / Ah": [0.0, 1.0, 0.5, 1.5],
                "Voltage / V": [3.0, 3.5, 4.0, 3.8],
                "Step Count / 1": pl.Series([0, 1, 1, 2], dtype=pl.UInt64),
                "Step ID": pl.Series([1, 1, 2, 2], dtype=pl.UInt64),
                "Step Type": ["rest", "charge", "charge", "discharge"],
            }
        )
    )


@pytest.fixture
def table():
    """A Table with a monotonic time axis and a smooth voltage signal."""
    time = np.linspace(0.0, 10.0, 51)
    voltage = 3.0 + 0.1 * time
    return Table(
        lf=pl.DataFrame(
            {
                BDF.TEST_TIME_SECOND.name: time,
                BDF.VOLTAGE_VOLT.name: voltage,
            }
        ).lazy(),
        metadata=build_metadata(cell_id="test"),
    )


@pytest.fixture
def cycling_data():
    """A CyclingData object with net capacity and net energy columns."""
    capacity = np.array([0.0, 1.0, 2.0, 1.5, 3.0])
    return CyclingData(
        lf=pl.DataFrame(
            {
                BDF.UNIX_TIME_SECOND.name: [0.0, 1.0, 2.0, 3.0, 4.0],
                BDF.CURRENT_AMPERE.name: [1.0] * 5,
                BDF.VOLTAGE_VOLT.name: [3.0] * 5,
                BDF.NET_CAPACITY_AH.name: capacity,
                BDF.NET_ENERGY_WH.name: capacity * 3.7,
            }
        ).lazy(),
        metadata=build_metadata(),
    )


class TestTableReducibleColumns:
    """Tests for _reducible_columns on Table."""

    def test_reducible_columns_excludes_step_id_and_strings(
        self, reduction_table: Table
    ) -> None:
        """Numeric columns included; Step ID and string columns excluded."""
        cols = reduction_table._reducible_columns()
        assert "Net Capacity / Ah" in cols
        assert "Voltage / V" in cols
        assert "Step Count / 1" in cols
        assert "Step ID" not in cols
        assert "Step Type" not in cols


class TestTableReductions:
    """Tests for delta, range, mean, maximum, minimum, start, and end on Table."""

    def test_delta_equals_last_minus_first(self, reduction_table: Table) -> None:
        """delta() returns last − first for each column."""
        result = reduction_table.delta()
        assert result.data.shape[0] == 1
        assert result.get("Net Capacity / Ah") == pytest.approx([1.5 - 0.0])

    def test_delta_chains_after_filter(self, BreakinCycles_fixture) -> None:
        """delta() on a filtered discharge slice returns signed net capacity delta."""
        discharge = BreakinCycles_fixture.cycle(0).discharge(0)
        result = discharge.delta("Net Capacity / Ah")
        assert result.data.shape[0] == 1
        cap = result.get("Net Capacity / Ah")
        assert cap[0] < 0

    def test_delta_no_arg_excludes_step_id_and_step_type(
        self, reduction_table: Table
    ) -> None:
        """No-arg delta includes Step Count but excludes Step ID and Step Type."""
        result = reduction_table.delta()
        assert "Step Count / 1" in result.data.columns
        assert "Step ID" not in result.data.columns
        assert "Step Type" not in result.data.columns

    def test_delta_named_cumulative_column_materialises_recipe(self) -> None:
        """delta('Cumulative Capacity / Ah') materialises recipe before reducing."""
        t = Table(lf=pl.LazyFrame({"Net Capacity / Ah": [0.0, 1.0, 0.5, 1.5, 0.0]}))
        result = t.delta("Cumulative Capacity / Ah")
        assert result.data.columns == ["Cumulative Capacity / Ah"]
        assert result.get("Cumulative Capacity / Ah") == pytest.approx([4.0])

    def test_delta_named_column_with_unit_conversion(self) -> None:
        """delta('Cumulative Capacity / mAh') applies unit conversion."""
        t = Table(lf=pl.LazyFrame({"Net Capacity / Ah": [0.0, 1.0, 0.5, 1.5, 0.0]}))
        result = t.delta("Cumulative Capacity / mAh")
        assert result.data.columns == ["Cumulative Capacity / mAh"]
        assert result.get("Cumulative Capacity / mAh") == pytest.approx([4000.0])

    def test_delta_named_column_only_that_column_in_result(
        self, reduction_table: Table
    ) -> None:
        """Explicit column arg produces result with only that column."""
        result = reduction_table.delta("Net Capacity / Ah")
        assert result.data.columns == ["Net Capacity / Ah"]

    def test_delta_accepts_column_objects(self) -> None:
        """Reducer methods accept Column-like objects directly."""
        table = Table(lf=pl.LazyFrame({"Net Capacity / Ah": [0.0, 1.0, 0.5, 1.5, 0.0]}))
        result = table.delta(BDF.CUMULATIVE_CAPACITY_AH)
        assert result.data.columns == [BDF.CUMULATIVE_CAPACITY_AH.name]
        assert result.item() == pytest.approx(4.0)

    def test_range_equals_absolute_delta_on_monotonic_discharge(
        self, BreakinCycles_fixture
    ) -> None:
        """range() is non-negative while delta() retains discharge sign."""
        discharge = BreakinCycles_fixture.cycle(0).discharge(0)
        range_value = discharge.range("Net Capacity / Ah").item()
        delta_value = discharge.delta("Net Capacity / Ah").item()
        assert range_value == pytest.approx(abs(delta_value))
        assert range_value >= 0
        assert delta_value < 0

    def test_range_over_full_cycle_is_non_zero_when_delta_is_near_zero(self) -> None:
        """range() captures extent over a full cycle even when delta cancels out."""
        table = Table(
            lf=pl.LazyFrame({"Net Capacity / Ah": [0.0, 1.2, 0.1, 1.3, 0.0]}),
        )
        assert table.delta("Net Capacity / Ah").item() == pytest.approx(0.0)
        assert table.range("Net Capacity / Ah").item() == pytest.approx(1.3)

    def test_range_no_arg_reduces_all_reducible_columns(
        self, reduction_table: Table
    ) -> None:
        """No-arg range includes reducible numeric columns only."""
        result = reduction_table.range()
        assert "Net Capacity / Ah" in result.data.columns
        assert "Voltage / V" in result.data.columns
        assert "Step Count / 1" in result.data.columns
        assert "Step ID" not in result.data.columns
        assert "Step Type" not in result.data.columns

    def test_range_unknown_column_raises_column_resolution_error(self) -> None:
        """Unknown columns still fail through the shared resolution path."""
        table = Table(lf=pl.LazyFrame({"Net Capacity / Ah": [0.0, 1.0]}))
        with pytest.raises(ColumnResolutionError, match="Cannot resolve"):
            table.range("Missing / Ah")

    def test_mean_returns_correct_single_row(self, reduction_table: Table) -> None:
        """mean() returns column-wise mean as a single row."""
        result = reduction_table.mean("Net Capacity / Ah")
        assert result.data.shape[0] == 1
        assert result.get("Net Capacity / Ah") == pytest.approx(
            [np.mean([0.0, 1.0, 0.5, 1.5])]
        )

    def test_maximum_returns_correct_single_row(self, reduction_table: Table) -> None:
        """maximum() returns column-wise maximum as a single row."""
        result = reduction_table.maximum("Net Capacity / Ah")
        assert result.get("Net Capacity / Ah") == pytest.approx([1.5])

    def test_minimum_returns_correct_single_row(self, reduction_table: Table) -> None:
        """minimum() returns column-wise minimum as a single row."""
        result = reduction_table.minimum("Net Capacity / Ah")
        assert result.get("Net Capacity / Ah") == pytest.approx([0.0])

    def test_first_returns_first_value(self, reduction_table: Table) -> None:
        """first() returns the first value of each column."""
        result = reduction_table.first("Net Capacity / Ah")
        assert result.get("Net Capacity / Ah") == pytest.approx([0.0])

    def test_last_returns_last_value(self, reduction_table: Table) -> None:
        """last() returns the last value of each column."""
        result = reduction_table.last("Net Capacity / Ah")
        assert result.get("Net Capacity / Ah") == pytest.approx([1.5])

    def test_delta_result_is_gettable_via_get(self, reduction_table: Table) -> None:
        """Single-row result of delta flows through get() as a length-1 array."""
        result = reduction_table.delta("Voltage / V")
        arr = result.get("Voltage / V")
        assert len(arr) == 1
        assert arr[0] == pytest.approx(3.8 - 3.0)

    def test_item_returns_scalar_from_single_column_reduction(self) -> None:
        """item() returns a float from a single-column reduction chain."""
        table = Table(lf=pl.LazyFrame({"Net Capacity / Ah": [0.0, 1.0, 0.5, 1.5]}))
        assert table.range("Net Capacity / Ah").item() == pytest.approx(1.5)

    def test_item_returns_named_column_from_multi_column_reduction(
        self, reduction_table: Table
    ) -> None:
        """item(column) extracts a named value from a multi-column reduction."""
        assert reduction_table.delta().item("Net Capacity / Ah") == pytest.approx(1.5)

    def test_item_rejects_multi_row_tables(self, reduction_table: Table) -> None:
        """item() rejects tables that have more than one row."""
        with pytest.raises(ValueError, match="exactly one row"):
            reduction_table.item()

    def test_item_rejects_ambiguous_column_selection(
        self, reduction_table: Table
    ) -> None:
        """item() names the available columns when column is ambiguous."""
        reduction = reduction_table.delta()
        with pytest.raises(ValueError, match="Available columns:"):
            reduction.item()

    def test_item_raises_key_error_for_missing_column(
        self, reduction_table: Table
    ) -> None:
        """item() requires an exact schema column name."""
        with pytest.raises(KeyError, match="Missing / Ah"):
            reduction_table.delta().item("Missing / Ah")

    def test_item_raises_type_error_for_non_numeric_values(self) -> None:
        """item() rejects non-numeric single-row values."""
        table = Table(lf=pl.LazyFrame({"Label": ["charge"]}))
        with pytest.raises(TypeError, match="numeric value"):
            table.item()


class TestTableSummary:
    """Tests for Table.summary grouped multi-statistic reduction."""

    def test_summary_groups_by_step_count(self) -> None:
        """summary() groups by Step Count / 1 by default, one row per group."""
        t = Table(
            lf=pl.LazyFrame(
                {
                    "Net Capacity / Ah": [0.0, 0.5, 0.5, 1.0],
                    "Voltage / V": [3.0, 3.5, 4.0, 3.8],
                    "Step Count / 1": pl.Series([0, 0, 1, 1], dtype=pl.UInt64),
                }
            )
        )
        result = t.summary()
        assert result.data.shape[0] == 2
        assert "Step Count / 1" in result.data.columns
        assert "delta Net Capacity / Ah" in result.data.columns
        assert "range Net Capacity / Ah" in result.data.columns
        assert "mean Net Capacity / Ah" in result.data.columns

    def test_summary_explicit_column_and_by(self) -> None:
        """Summary with explicit column and by='Cycle Count / 1'."""
        t = Table(
            lf=pl.LazyFrame(
                {
                    "Net Capacity / Ah": [0.0, 1.0, 0.5, 2.0],
                    "Cycle Count / 1": pl.Series([0, 0, 1, 1], dtype=pl.UInt64),
                }
            )
        )
        result = t.summary("Net Capacity / Ah", by="Cycle Count / 1")
        assert result.data.shape[0] == 2
        assert "delta Net Capacity / Ah" in result.data.columns

    def test_summary_capacity_delta_is_last_minus_first(self) -> None:
        """Capacity delta in summary equals last − first, not max − min."""
        t = Table(
            lf=pl.LazyFrame(
                {
                    "Net Capacity / Ah": [0.0, 1.0, 0.8],
                    "Step Count / 1": pl.Series([0, 0, 0], dtype=pl.UInt64),
                }
            )
        )
        result = t.summary("Net Capacity / Ah")
        delta_val = result.data["delta Net Capacity / Ah"][0]
        assert delta_val == pytest.approx(0.8)

    def test_summary_retains_step_id_as_descriptor(self) -> None:
        """Step ID appears in summary result as first value per group."""
        t = Table(
            lf=pl.LazyFrame(
                {
                    "Net Capacity / Ah": [0.0, 1.0, 0.5, 2.0],
                    "Step Count / 1": pl.Series([0, 0, 1, 1], dtype=pl.UInt64),
                    "Step ID": pl.Series([1, 1, 2, 2], dtype=pl.UInt64),
                }
            )
        )
        result = t.summary("Net Capacity / Ah")
        assert "Step ID" in result.data.columns
        assert "delta Step ID" not in result.data.columns

    def test_summary_throughput_from_cumulative_recipe(self) -> None:
        """Summary with Cumulative Capacity materialises recipe before grouping."""
        t = Table(
            lf=pl.LazyFrame(
                {
                    "Net Capacity / Ah": [0.0, 1.0, 0.5, 1.5],
                    "Step Count / 1": pl.Series([0, 0, 1, 1], dtype=pl.UInt64),
                }
            )
        )
        result = t.summary("Cumulative Capacity / Ah")
        assert "delta Cumulative Capacity / Ah" in result.data.columns

    def test_summary_includes_range_with_groupwise_extent(self) -> None:
        """summary() includes range columns computed as max minus min."""
        table = Table(
            lf=pl.LazyFrame(
                {
                    "Net Capacity / Ah": [0.0, 0.4, 0.2, 1.0, 0.6],
                    "Step Count / 1": pl.Series([0, 0, 0, 1, 1], dtype=pl.UInt64),
                }
            )
        )
        result = table.summary("Net Capacity / Ah")
        assert result.data["range Net Capacity / Ah"].to_list() == pytest.approx(
            [0.4, 0.4]
        )


class TestTableCurveOperations:
    """Tests for flat Table operations that delegate to analysis helpers."""

    @pytest.mark.parametrize(
        "fit",
        [PchipInterpolator, CubicSpline, Akima1DInterpolator, make_smoothing_spline],
    )
    def test_to_curve_returns_labelled_curve(self, table: Table, fit) -> None:
        """Each scipy fit callable returns a labelled Curve."""
        curve = table.to_curve(BDF.VOLTAGE_VOLT, x=BDF.TEST_TIME_SECOND, fit=fit)
        assert isinstance(curve, Curve)
        assert curve.columns.x.name == BDF.TEST_TIME_SECOND.name
        assert curve.columns.y.name == BDF.VOLTAGE_VOLT.name

    def test_to_curve_default_fit_is_pchip(self, table: Table) -> None:
        """The default fit is PchipInterpolator, recorded in metadata."""
        curve = table.to_curve(BDF.VOLTAGE_VOLT, x=BDF.TEST_TIME_SECOND)
        assert isinstance(curve, Curve)
        assert read_extras(curve)["curve_method"] == "PchipInterpolator"

    def test_to_curve_carries_source_extras(self, table: Table) -> None:
        """A fitted curve carries the source table's extras, not its record."""
        curve = table.to_curve(BDF.VOLTAGE_VOLT, x=BDF.TEST_TIME_SECOND)
        assert curve.metadata["cell_id"] == "test"

    def test_to_curve_interpolator_passes_through_points(self, table: Table) -> None:
        """An interpolating curve passes through the supplied data points."""
        x, y = table.get(BDF.TEST_TIME_SECOND.name, BDF.VOLTAGE_VOLT.name)
        curve = table.to_curve(
            BDF.VOLTAGE_VOLT,
            x=BDF.TEST_TIME_SECOND,
            fit=CubicSpline,
        )
        np.testing.assert_allclose(curve(x), y, atol=1e-6)

    def test_to_curve_forwards_kwargs(self, table: Table) -> None:
        """Extra kwargs are forwarded to the fit callable."""
        curve = table.to_curve(
            BDF.VOLTAGE_VOLT,
            x=BDF.TEST_TIME_SECOND,
            fit=CubicSpline,
            bc_type="natural",
        )
        assert isinstance(curve, Curve)

    def test_to_curve_non_conforming_fit_raises_typeerror(self, table: Table) -> None:
        """A fit returning neither PPoly nor BSpline raises TypeError."""
        with pytest.raises(TypeError):
            table.to_curve(
                BDF.VOLTAGE_VOLT,
                x=BDF.TEST_TIME_SECOND,
                fit=lambda x, y: "not a poly",
            )

    def test_savgol_returns_table_matching_standalone(self, table: Table) -> None:
        """Savgol returns a Table equal to the standalone function."""
        result = table.savgol(BDF.VOLTAGE_VOLT.name, window_length=5, polyorder=2)
        expected = smoothing.savgol_smoothing(
            table, BDF.VOLTAGE_VOLT.name, window_length=5, polyorder=2
        )
        assert isinstance(result, Table)
        pl_testing.assert_frame_equal(result.data, expected.data)

    def test_downsample_returns_table_matching_standalone(self, table: Table) -> None:
        """Downsample returns a Table equal to the standalone function."""
        result = table.downsample(BDF.TEST_TIME_SECOND.name, sampling_interval=1.0)
        expected = smoothing.downsample(
            table, BDF.TEST_TIME_SECOND.name, sampling_interval=1.0
        )
        assert isinstance(result, Table)
        pl_testing.assert_frame_equal(result.data, expected.data)

    def test_gradient_returns_table_matching_standalone(self, table: Table) -> None:
        """Gradient returns a Table matching the standalone function."""
        result = table.gradient(y=BDF.VOLTAGE_VOLT.name, x=BDF.TEST_TIME_SECOND.name)
        expected = differentiation.gradient(
            table, x=BDF.TEST_TIME_SECOND.name, y=BDF.VOLTAGE_VOLT.name
        )
        assert isinstance(result, Table)
        pl_testing.assert_frame_equal(result.data, expected.data)


class TestCyclingDataReductions:
    """Generic reduction helpers cover the old CyclingData scalar quantities."""

    def test_range_net_capacity_is_extent(self, cycling_data: CyclingData) -> None:
        """range().item() returns max minus min of the net capacity."""
        capacity = np.asarray(cycling_data.get(BDF.NET_CAPACITY_AH.name))
        value = cycling_data.range(BDF.NET_CAPACITY_AH).item()
        assert isinstance(value, float)
        assert value == pytest.approx(capacity.max() - capacity.min())

    def test_range_net_energy_is_extent(self, cycling_data: CyclingData) -> None:
        """range().item() returns max minus min of the net energy."""
        energy = np.asarray(cycling_data.get(BDF.NET_ENERGY_WH.name))
        assert cycling_data.range(BDF.NET_ENERGY_WH).item() == pytest.approx(
            energy.max() - energy.min()
        )

    def test_delta_capacity_throughput_uses_cumulative_recipe(
        self, cycling_data: CyclingData
    ) -> None:
        """delta().item() returns the cumulative capacity throughput."""
        capacity = cycling_data.get(BDF.NET_CAPACITY_AH.name)
        assert cycling_data.delta(BDF.CUMULATIVE_CAPACITY_AH).item() == pytest.approx(
            np.abs(np.diff(capacity)).sum()
        )

    def test_delta_energy_throughput_uses_cumulative_recipe(
        self, cycling_data: CyclingData
    ) -> None:
        """delta().item() returns the cumulative energy throughput."""
        energy = cycling_data.get(BDF.NET_ENERGY_WH.name)
        assert cycling_data.delta(BDF.CUMULATIVE_ENERGY_WH).item() == pytest.approx(
            np.abs(np.diff(energy)).sum()
        )


class TestRawDataAlias:
    """RawData is a deprecated alias of CyclingData."""

    def test_isinstance_holds_for_any_cycling_data(
        self, cycling_data: CyclingData
    ) -> None:
        """isinstance(obj, RawData) is True for any CyclingData instance."""
        assert isinstance(cycling_data, RawData)

    def test_construction_warns(self, cycling_data: CyclingData) -> None:
        """Constructing RawData directly emits a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            RawData(lf=cycling_data.lf, metadata=build_metadata())
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_capacity_matches_range_and_warns(self, cycling_data: CyclingData) -> None:
        """The deprecated capacity property returns the generic range and warns."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = cycling_data.capacity
        assert value == cycling_data.range(BDF.NET_CAPACITY_AH).item()
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
