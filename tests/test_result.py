"""Tests for the result module."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import numpy as np
import numpy.testing as np_testing
import polars as pl
import polars.testing as pl_testing
import pytest
from scipy.io import loadmat

from pyprobe.result import Result, _PolarsColumnCache, combine_results


def test__PolarsColumnCache_lazyframe():
    """Test the _PolarsColumnCache class."""
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    cache = _PolarsColumnCache(lf)
    assert cache.cache == {}
    pl_testing.assert_frame_equal(cache.base_dataframe, lf)


def test__PolarsColumnCache_dataframe():
    """Test the _PolarsColumnCache class with a DataFrame."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    cache = _PolarsColumnCache(df)
    pl_testing.assert_frame_equal(cache.cached_dataframe, df)
    expected_a = df.select("a")["a"]
    expected_b = df.select("b")["b"]
    expected_c = df.select("c")["c"]
    assert cache.cache["a"].to_list() == expected_a.to_list()
    assert cache.cache["b"].to_list() == expected_b.to_list()
    assert cache.cache["c"].to_list() == expected_c.to_list()


def test_collect_columns():
    """Test the collect_columns method."""
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    cache = _PolarsColumnCache(lf)

    # Test single column collection
    cache.collect_columns("a")
    expected_a = lf.select("a").collect()["a"]
    assert cache.cache["a"].to_list() == expected_a.to_list()
    pl_testing.assert_frame_equal(cache.cached_dataframe, lf.select("a").collect())

    # Test making a second collection
    cache.collect_columns("b")
    pl.testing.assert_frame_equal(cache.cached_dataframe, lf.select("a", "b").collect())

    # Test multiple column collection
    cache = _PolarsColumnCache(lf)
    cache.collect_columns("a", "b")
    expected_a = lf.select("a").collect()["a"]
    expected_b = lf.select("b").collect()["b"]
    assert cache.cache["a"].to_list() == expected_a.to_list()
    assert cache.cache["b"].to_list() == expected_b.to_list()
    pl_testing.assert_frame_equal(
        cache.cached_dataframe,
        lf.select("a", "b").collect(),
        check_column_order=False,
    )

    # Test unit conversion
    lf = pl.LazyFrame(
        {
            "Current [A]": [1, 2, 3],
            "Voltage [V]": [4, 5, 6],
            "Date": [5, 6, 7],
        },
    )
    cache = _PolarsColumnCache(lf)
    cache.collect_columns("Current [mA]")
    expected_current = pl.Series("Current [mA]", [1000, 2000, 3000])
    assert cache.cache["Current [mA]"].to_list() == expected_current.to_list()


def test_cached_dataframe():
    """Test the cached_dataframe property."""
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    cache = _PolarsColumnCache(lf)
    assert cache._cached_dataframe.is_empty()
    assert cache.cached_dataframe.is_empty()

    cache.collect_columns("a")
    assert cache._cached_dataframe.is_empty()
    pl.testing.assert_frame_equal(cache.cached_dataframe, lf.select("a").collect())
    pl.testing.assert_frame_equal(cache._cached_dataframe, lf.select("a").collect())

    cache.collect_columns("b")
    pl.testing.assert_frame_equal(cache._cached_dataframe, lf.select("a").collect())
    pl.testing.assert_frame_equal(cache.cached_dataframe, lf.select("a", "b").collect())
    pl.testing.assert_frame_equal(
        cache._cached_dataframe,
        lf.select("a", "b").collect(),
    )


def test_live_dataframe():
    """Test the live_dataframe property."""
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    result_object = Result(base_dataframe=lf, info={})
    pl_testing.assert_frame_equal(result_object.live_dataframe, lf)
    assert result_object._polars_cache.columns == ["a", "b", "c"]
    assert result_object._polars_cache.quantities == {"a", "b", "c"}

    # test updating a column of the live_dataframe
    result_object.live_dataframe = result_object.live_dataframe.with_columns(
        (pl.col("a") * 10).alias("a"),
    )
    result_object._polars_cache.collect_columns("a")
    pl_testing.assert_frame_equal(
        result_object.live_dataframe,
        lf.with_columns(pl.col("a") * 10),
    )

    result_object = Result(base_dataframe=lf, info={})
    result_object._polars_cache.collect_columns("a")
    result_object.live_dataframe = result_object.live_dataframe.with_columns(
        (pl.col("a") * 10).alias("d"),
    )
    pl.testing.assert_frame_equal(
        result_object.live_dataframe,
        lf.with_columns((pl.col("a") * 10).alias("d")),
    )


@pytest.fixture
def Result_fixture(lazyframe_fixture, info_fixture):
    """Return a Result instance."""
    return Result(
        base_dataframe=lazyframe_fixture,
        info=info_fixture,
        column_definitions={
            "Current": "Current definition",
        },
    )


def test_init(Result_fixture):
    """Test the __init__ method."""
    assert isinstance(Result_fixture, Result)
    assert isinstance(Result_fixture.base_dataframe, pl.LazyFrame)
    assert isinstance(Result_fixture.info, dict)


def test_cache_columns():
    """Test the collect method."""
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    result_object = Result(base_dataframe=lf, info={})
    result_object.cache_columns("a")
    pl_testing.assert_frame_equal(
        result_object._polars_cache.cached_dataframe,
        lf.select("a").collect(),
    )

    result_object = Result(base_dataframe=lf, info={})
    result_object.cache_columns("a", "b")
    pl_testing.assert_frame_equal(
        result_object._polars_cache.cached_dataframe,
        lf.select("a", "b").collect(),
        check_column_order=False,
    )

    result_object = Result(base_dataframe=lf, info={})
    result_object.cache_columns()
    pl_testing.assert_frame_equal(
        result_object._polars_cache.cached_dataframe,
        lf.collect(),
        check_column_order=False,
    )


def test_get(Result_fixture):
    """Test the get method."""
    current = Result_fixture.get("Current [A]")
    np_testing.assert_array_equal(
        current,
        Result_fixture.data["Current [A]"].to_numpy(),
    )
    current_mA = Result_fixture.get("Current [mA]")
    np_testing.assert_array_equal(current_mA, current * 1000)

    current, voltage = Result_fixture.get("Current [A]", "Voltage [V]")
    np_testing.assert_array_equal(
        current,
        Result_fixture.data["Current [A]"].to_numpy(),
    )
    np_testing.assert_array_equal(
        voltage,
        Result_fixture.data["Voltage [V]"].to_numpy(),
    )


def test_get_only(Result_fixture):
    """Test the get_only method."""
    current = Result_fixture.get("Current [A]")
    np_testing.assert_array_equal(
        current,
        Result_fixture.data["Current [A]"].to_numpy(),
    )
    current_mA = Result_fixture.get("Current [mA]")
    np_testing.assert_array_equal(current_mA, current * 1000)


def test_getitem(Result_fixture):
    """Test the __getitem__ method."""
    current = Result_fixture["Current [A]"]
    assert "Current [A]" in current.column_list
    assert isinstance(current, Result)
    pl_testing.assert_frame_equal(
        current.data,
        Result_fixture.data.select("Current [A]"),
    )
    current_mA = Result_fixture["Current [mA]"]
    assert "Current [mA]" in current_mA.column_list
    assert "Current [A]" not in current_mA.column_list
    np_testing.assert_allclose(
        current_mA.get("Current [mA]"),
        Result_fixture.get("Current [mA]"),
    )


def test_data(Result_fixture):
    """Test the data property."""
    assert isinstance(Result_fixture.base_dataframe, pl.LazyFrame)
    assert isinstance(Result_fixture.data, pl.DataFrame)
    assert isinstance(Result_fixture.live_dataframe, pl.DataFrame)
    pl_testing.assert_frame_equal(Result_fixture.data, Result_fixture.live_dataframe)


def test_quantities(Result_fixture):
    """Test the quantities property."""
    assert set(Result_fixture.quantities) == {
        "Time",
        "Current",
        "Voltage",
        "Capacity",
        "Event",
        "Date",
        "Step",
    }


def test_print_definitions(Result_fixture, capsys):
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


def test_build():
    """Test the build method."""
    data1 = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    data2 = pl.DataFrame({"x": [7, 8, 9], "y": [10, 11, 12]})
    info = {"test": "info"}
    result = Result.build([data1, data2], info)
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


def test_add_data():
    """Test the add_data method."""
    existing_data = pl.LazyFrame(
        {
            "Date": pl.datetime_range(
                datetime(1985, 1, 1, 0, 0, 0),
                datetime(1985, 1, 1, 0, 0, 5),
                timedelta(seconds=1),
                time_unit="ms",
                eager=True,
            ).alias("datetime"),
            "Data": [2, 4, 6, 8, 10, 12],
        },
    )
    new_data = pl.LazyFrame(
        {
            "DateTime": pl.datetime_range(
                datetime(1985, 1, 1, 0, 0, 2, 500000),
                datetime(1985, 1, 1, 0, 0, 7, 500000),
                timedelta(seconds=1),
                time_unit="ms",
                eager=True,
            ).alias("datetime"),
            "Data 1": [2, 4, 6, 8, 10, 12],
            "Data 2": [4, 8, 12, 16, 20, 24],
        },
    )
    result_object = Result(base_dataframe=existing_data, info={})
    result_object.add_data(
        new_data,
        date_column_name="DateTime",
        existing_data_timezone="GMT",
    )
    expected_data = pl.DataFrame(
        {
            "Date": pl.datetime_range(
                datetime(1985, 1, 1, 0, 0, 0),
                datetime(1985, 1, 1, 0, 0, 5),
                timedelta(seconds=1),
                time_unit="ms",
                eager=True,
            )
            .dt.cast_time_unit("us")
            .dt.replace_time_zone("GMT")
            .alias("datetime"),
            "Data": [2, 4, 6, 8, 10, 12],
            "Data 1": [None, None, None, 3.0, 5.0, 7.0],
            "Data 2": [None, None, None, 6.0, 10.0, 14.0],
        },
    )
    pl_testing.assert_frame_equal(
        result_object.data,
        expected_data,
        check_column_order=False,
    )


def test_add_new_data_columns_deprecated():
    """Test that add_new_data_columns works but is deprecated."""
    existing_data = pl.LazyFrame(
        {
            "Date": pl.datetime_range(
                datetime(1985, 1, 1, 0, 0, 0),
                datetime(1985, 1, 1, 0, 0, 5),
                timedelta(seconds=1),
                time_unit="ms",
                eager=True,
            ).alias("datetime"),
            "Data": [2, 4, 6, 8, 10, 12],
        },
    )
    new_data = pl.LazyFrame(
        {
            "DateTime": pl.datetime_range(
                datetime(1985, 1, 1, 0, 0, 2, 500000),
                datetime(1985, 1, 1, 0, 0, 7, 500000),
                timedelta(seconds=1),
                time_unit="ms",
                eager=True,
            ).alias("datetime"),
            "Data 1": [2, 4, 6, 8, 10, 12],
        },
    )
    result_object = Result(base_dataframe=existing_data, info={})

    with patch("pyprobe.utils.logger.warning") as mock_warning:
        result_object.add_new_data_columns(new_data, date_column_name="DateTime")
        mock_warning.assert_called_with("Deprecation Warning: Use add_data instead.")

    assert "Data 1" in result_object.column_list


def test_add_data_timezone_handling():
    """Test timezone handling in add_data."""
    # Case 1: Existing data is naive, new data is aware (UTC)
    # Should default to local timezone (or London) for existing, and convert new to that
    existing_data = pl.LazyFrame(
        {"Date": [datetime(2023, 1, 1, 10, 0, 0)], "Value": [1]}
    )

    new_data = pl.LazyFrame(
        {"DateUTC": [datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC)], "Ext": [10]}
    )

    result = Result(base_dataframe=existing_data, info={})
    result.add_data(new_data, date_column_name="DateUTC")

    schema = result.live_dataframe.collect_schema()
    assert isinstance(schema["Date"], pl.Datetime)
    assert schema["Date"].time_zone is not None

    # Case 2: Explicit timezones
    existing_data_naive = pl.LazyFrame(
        {"Date": [datetime(2023, 1, 1, 10, 0, 0)], "Value": [1]}
    )

    new_data_naive = pl.LazyFrame(
        {"DateNew": [datetime(2023, 1, 1, 10, 0, 0)], "Ext": [10]}
    )

    result2 = Result(base_dataframe=existing_data_naive, info={})
    result2.add_data(
        new_data_naive,
        date_column_name="DateNew",
        existing_data_timezone="UTC",
        new_data_timezone="Europe/Paris",
    )

    schema2 = result2.live_dataframe.collect_schema()
    assert schema2["Date"].time_zone == "UTC"


def test_add_data_with_format():
    """Test add_data with datetime format string."""
    existing_data = pl.LazyFrame(
        {"Date": [datetime(2023, 1, 1, 10, 0, 0)], "Value": [1]}
    )

    new_data = pl.LazyFrame({"DateStr": ["2023/01/01 10:00:00"], "Ext": [10]})

    result = Result(base_dataframe=existing_data, info={})
    result.add_data(
        new_data, date_column_name="DateStr", datetime_format="%Y/%m/%d %H:%M:%S"
    )

    schema = result.live_dataframe.collect_schema()
    assert isinstance(schema["Date"], pl.Datetime)

    data = result.data
    assert "Ext" in data.columns
    assert data["Ext"][0] == 10


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
        base_dataframe=data,
        info={"test": "info"},
        column_definitions={
            "Voltage": "Voltage definition",
            "Current": "Current definition",
        },
    )


def test_verify_compatible_frames():
    """Test the _verify_compatible_frames method."""
    df1 = pl.DataFrame({"a": [1, 2, 3]})
    df2 = pl.DataFrame({"b": [4, 5, 6]})
    lazy_df1 = df1.lazy()
    lazy_df2 = df2.lazy()

    # Test with two DataFrames
    result1, result2 = Result._verify_compatible_frames(df1, [df2])
    assert isinstance(result1, pl.DataFrame)
    assert isinstance(result2[0], pl.DataFrame)

    # Test with DataFrame and LazyFrame
    result1, result2 = Result._verify_compatible_frames(df1, [lazy_df2])
    assert isinstance(result1, pl.DataFrame)
    assert isinstance(result2[0], pl.DataFrame)

    # Test with LazyFrame and DataFrame
    result1, result2 = Result._verify_compatible_frames(lazy_df1, [df2])
    assert isinstance(result1, pl.DataFrame)
    assert isinstance(result2[0], pl.DataFrame)

    # Test with two LazyFrames
    result1, result2 = Result._verify_compatible_frames(
        lazy_df1,
        [lazy_df2],
        mode="collect all",
    )
    assert isinstance(result1, pl.LazyFrame)
    assert isinstance(result2[0], pl.LazyFrame)

    # Test with matching the first df
    result1, result2 = Result._verify_compatible_frames(lazy_df1, [df2], mode="match 1")
    assert isinstance(result1, pl.LazyFrame)
    assert isinstance(result2[0], pl.LazyFrame)

    result1, result2 = Result._verify_compatible_frames(df1, [lazy_df2], mode="match 1")
    assert isinstance(result1, pl.DataFrame)
    assert isinstance(result2[0], pl.DataFrame)

    # Test with a list of frames
    result1, result2 = Result._verify_compatible_frames(df1, [df2, lazy_df2])
    assert isinstance(result1, pl.DataFrame)
    assert isinstance(result2[0], pl.DataFrame)
    assert isinstance(result2[1], pl.DataFrame)

    # Test matching the first df with a list of frames
    result1, result2 = Result._verify_compatible_frames(
        lazy_df1,
        [df2, lazy_df2],
        mode="match 1",
    )
    assert isinstance(result1, pl.LazyFrame)
    assert isinstance(result2[0], pl.LazyFrame)
    assert isinstance(result2[1], pl.LazyFrame)


def test_join_left(reduced_result_fixture):
    """Test the join method with left join."""
    other_data = pl.DataFrame(
        {
            "Current [A]": [1, 2, 3],
            "Capacity [Ah]": [4, 5, 6],
        },
    )
    other_result = Result(
        base_dataframe=other_data,
        info={"test": "info"},
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
    assert reduced_result_fixture.column_definitions["Voltage"] == "Voltage definition"


def test_extend(reduced_result_fixture):
    """Test the extend method."""
    other_data = pl.DataFrame(
        {
            "Current [A]": [4, 5, 6],
            "Voltage [V]": [4, 5, 6],
        },
    )
    other_result = Result(
        base_dataframe=other_data,
        info={"test": "info"},
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
    assert reduced_result_fixture.column_definitions["Voltage"] == "Voltage definition"


def test_extend_with_new_columns(reduced_result_fixture):
    """Test the extend method with new columns."""
    other_data = pl.DataFrame(
        {
            "Current [A]": [4, 5, 6],
            "Voltage [V]": [4, 5, 6],
            "Capacity [Ah]": [8, 9, 10],
        },
    )
    other_result = Result(
        base_dataframe=other_data,
        info={"test": "info"},
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
    assert reduced_result_fixture.column_definitions["Voltage"] == "Voltage definition"
    assert (
        reduced_result_fixture.column_definitions["Capacity"] == "Capacity definition"
    )
    assert reduced_result_fixture.column_definitions["Current"] == "Current definition"


def test_clean_copy(reduced_result_fixture):
    """Test the clean_copy method."""
    # Test default parameters (empty dataframe)
    clean_result = reduced_result_fixture.clean_copy()
    assert isinstance(clean_result, Result)
    assert clean_result.base_dataframe.is_empty()
    assert clean_result.info == reduced_result_fixture.info
    assert clean_result.column_definitions == {}

    # Test with new dataframe
    new_df = pl.DataFrame({"Test [V]": [1, 2, 3]})
    clean_result = reduced_result_fixture.clean_copy(dataframe=new_df)
    assert isinstance(clean_result, Result)
    pl_testing.assert_frame_equal(clean_result.data, new_df)
    assert clean_result.info == reduced_result_fixture.info
    assert clean_result.column_definitions == {}

    # Test with new column definitions
    new_defs = {"New Column [A]": "New definition"}
    clean_result = reduced_result_fixture.clean_copy(column_definitions=new_defs)
    assert isinstance(clean_result, Result)
    assert clean_result.base_dataframe.is_empty()
    assert clean_result.info == reduced_result_fixture.info
    assert clean_result.column_definitions == new_defs

    # Test with both new dataframe and column definitions
    clean_result = reduced_result_fixture.clean_copy(
        dataframe=new_df,
        column_definitions=new_defs,
    )
    assert isinstance(clean_result, Result)
    pl_testing.assert_frame_equal(clean_result.data, new_df)
    assert clean_result.info == reduced_result_fixture.info
    assert clean_result.column_definitions == new_defs

    # Test with LazyFrame
    lazy_df = new_df.lazy()
    clean_result = reduced_result_fixture.clean_copy(dataframe=lazy_df)
    assert isinstance(clean_result, Result)
    assert isinstance(clean_result.base_dataframe, pl.LazyFrame)
    pl_testing.assert_frame_equal(clean_result.data, new_df)


def test_combine_results():
    """Test the combine results method."""
    result1 = Result(
        base_dataframe=pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
        info={"test index": 1.0},
    )
    result2 = Result(
        base_dataframe=pl.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]}),
        info={"test index": 2.0},
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


def test_export_to_mat(Result_fixture, tmp_path):
    """Test the export to mat function."""
    mat_path = tmp_path / "test_mat.mat"
    Result_fixture.export_to_mat(str(mat_path))
    saved_data = loadmat(str(mat_path))
    assert "data" in saved_data
    assert "info" in saved_data
    expected_columns = {
        "Current__A_",
        "Step",
        "Event",
        "Time__s_",
        "Capacity__Ah_",
        "Voltage__V_",
        "Date",
    }
    actual_columns = set(saved_data["data"].dtype.names)
    assert actual_columns == expected_columns


def test_from_polars_io(tmp_path):
    """Test the from_polars_io method."""
    # Test with read_csv function
    test_df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    csv_path = tmp_path / "test_data.csv"
    test_df.write_csv(csv_path)

    # Test with basic parameters
    result = Result.from_polars_io(
        info={"test": "info"},
        column_definitions={"a": "Column A"},
        polars_io_func=pl.read_csv,
        source=str(csv_path),
    )
    assert isinstance(result, Result)
    assert result.info == {"test": "info"}
    assert result.column_definitions == {"a": "Column A"}
    pl_testing.assert_frame_equal(result.data, test_df)

    # Test with LazyFrame function
    result_lazy = Result.from_polars_io(
        info={"test": "lazy"},
        column_definitions={},
        polars_io_func=pl.scan_csv,
        source=str(csv_path),
    )
    assert isinstance(result_lazy, Result)
    assert isinstance(result_lazy.base_dataframe, pl.LazyFrame)

    # Test with keyword arguments
    result_with_kwargs = Result.from_polars_io(
        info={"test": "kwargs"},
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
def test_from_polars_io_different_formats(io_function, expected_type, tmp_path):
    """Test from_polars_io with different polars I/O functions."""
    # Create test data
    test_df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # Create appropriate test file based on function
    if "csv" in io_function.__name__:
        test_file = tmp_path / "test.csv"
        test_df.write_csv(test_file)
    else:  # parquet
        test_file = tmp_path / "test.parquet"
        test_df.write_parquet(test_file)

    # Mock info for testing
    info = {"source": io_function.__name__}

    # Create result using the function
    result = Result.from_polars_io(
        polars_io_func=io_function, source=test_file, info=info, column_definitions={}
    )

    # Check the result
    assert isinstance(result, Result)
    assert isinstance(result.base_dataframe, expected_type)
    assert result.info == info
    pl_testing.assert_frame_equal(result.data, test_df, check_column_order=False)


def test_from_polars_io_python_object():
    """Test from_polars_io with a Python object."""
    # Create a test DataFrame
    test_df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # Mock info for testing
    info = {"source": "python_object"}

    # Create result using the function
    result = Result.from_polars_io(
        polars_io_func=pl.from_pandas,
        data=test_df.to_pandas(),
        info=info,
        column_definitions={},
    )

    # Check the result
    assert isinstance(result, Result)
    assert isinstance(result.base_dataframe, pl.DataFrame)
    assert result.info == info
    pl_testing.assert_frame_equal(result.data, test_df, check_column_order=False)

    result = Result.from_polars_io(
        polars_io_func=pl.from_numpy,
        schema=["a", "b"],
        data=test_df.to_numpy(),
        info=info,
        column_definitions={},
    )

    # Check the result
    assert isinstance(result, Result)
    assert isinstance(result.base_dataframe, pl.DataFrame)
    assert result.info == info
    pl_testing.assert_frame_equal(result.data, test_df, check_column_order=False)


def test_add_data_with_alignment():
    """Test add_data with the align_on parameter."""
    # Create base data: Square wave signals by sampling continuous signals
    # This simulates real data where edge timing is preserved in sample values
    dt = 0.1
    t = np.arange(0, 20, dt)

    t_continuous = np.linspace(0, 20, 100000)
    y_continuous = np.zeros_like(t_continuous)
    y_continuous[t_continuous >= 5.0] = 1.0
    y_continuous[t_continuous >= 10.0] = 0.0
    y_continuous[t_continuous >= 12.0] = -1.0
    y_continuous[t_continuous >= 17.0] = 0.0

    # Sample the continuous signal
    y = np.interp(t, t_continuous, y_continuous)

    start_time = datetime(2023, 1, 1, 10, 0, 0)

    base_df = pl.DataFrame(
        {"Date": [start_time + timedelta(seconds=float(val)) for val in t], "Signal": y}
    )

    # Create new data: Same signal but shifted
    shift = 2.35
    y_shifted_continuous = np.zeros_like(t_continuous)
    y_shifted_continuous[t_continuous >= (5.0 + shift)] = 1.0
    y_shifted_continuous[t_continuous >= (10.0 + shift)] = 0.0
    y_shifted_continuous[t_continuous >= (12.0 + shift)] = -1.0
    y_shifted_continuous[t_continuous >= (17.0 + shift)] = 0.0

    y_shifted = np.interp(t, t_continuous, y_shifted_continuous)

    new_df = pl.DataFrame(
        {
            "DateNew": [start_time + timedelta(seconds=float(val)) for val in t],
            "SignalNew": y_shifted,
        }
    )

    result = Result(base_dataframe=base_df, info={})

    # Add data with alignment
    result.add_data(
        new_df, date_column_name="DateNew", align_on=("Signal", "SignalNew")
    )

    combined_df = result.data

    # Check that SignalNew is aligned with Signal
    s1 = combined_df["Signal"].to_numpy()
    s2 = combined_df["SignalNew"].to_numpy()

    # Filter out NaNs (due to shifting, some points might not overlap)
    mask = ~np.isnan(s2)

    # Assert that the signals are close (alignment worked)
    # Tolerance of 0.5 accounts for edge transition differences after interpolation
    np_testing.assert_allclose(s1[mask], s2[mask], atol=0.5)


def test_add_data_with_alignment_error():
    """Test add_data with invalid align_on columns."""
    start_time = datetime(2023, 1, 1, 10, 0, 0)
    base_df = pl.DataFrame({"Date": [start_time], "Signal": [1.0]})
    new_df = pl.DataFrame({"DateNew": [start_time], "SignalNew": [1.0]})
    result = Result(base_dataframe=base_df, info={})

    # Test with missing column in base data
    with pytest.raises(ValueError):
        result.add_data(
            new_df, date_column_name="DateNew", align_on=("NonExistent", "SignalNew")
        )

    # Test with missing column in new data
    with pytest.raises(ValueError):
        result.add_data(
            new_df, date_column_name="DateNew", align_on=("Signal", "NonExistent")
        )
