"""Tests for the result module."""

from datetime import datetime, timedelta

import numpy.testing as np_testing
import polars as pl
import polars.testing as pl_testing
import pytest

from pyprobe.result import PolarsColumnCache, Result, combine_results


def test_PolarsColumnCache_lazyframe():
    """Test the PolarsColumnCache class."""
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    cache = PolarsColumnCache(lf)
    assert cache.cache == {}
    pl_testing.assert_frame_equal(cache.base_dataframe, lf)


def test_PolarsColumnCache_dataframe():
    """Test the PolarsColumnCache class with a DataFrame."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    cache = PolarsColumnCache(df)
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
    cache = PolarsColumnCache(lf)

    # Test single column collection
    cache.collect_columns("a")
    expected_a = lf.select("a").collect()["a"]
    assert cache.cache["a"].to_list() == expected_a.to_list()
    pl_testing.assert_frame_equal(cache.cached_dataframe, lf.select("a").collect())

    # Test multiple column collection
    cache = PolarsColumnCache(lf)
    cache.collect_columns("a", "b")
    expected_a = lf.select("a").collect()["a"]
    expected_b = lf.select("b").collect()["b"]
    assert cache.cache["a"].to_list() == expected_a.to_list()
    assert cache.cache["b"].to_list() == expected_b.to_list()
    pl_testing.assert_frame_equal(
        cache.cached_dataframe, lf.select("a", "b").collect(), check_column_order=False
    )

    # Test unit conversion
    lf = pl.LazyFrame(
        {
            "Current [A]": [1, 2, 3],
            "Voltage [V]": [4, 5, 6],
            "Date": [5, 6, 7],
        }
    )
    cache = PolarsColumnCache(lf)
    cache.collect_columns("Current [mA]")
    expected_current = pl.Series("Current [mA]", [1000, 2000, 3000])
    assert cache.cache["Current [mA]"].to_list() == expected_current.to_list()


def test_live_dataframe():
    """Test the live_dataframe property."""
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    result_object = Result(base_dataframe=lf, info={})
    pl_testing.assert_frame_equal(result_object.live_dataframe, lf)
    assert result_object._polars_cache.columns == ["a", "b", "c"]
    assert result_object._polars_cache.quantities == {"a", "b", "c"}

    # test updating a column of the live_dataframe
    result_object.live_dataframe = result_object.live_dataframe.with_columns(
        (pl.col("a") * 10).alias("a")
    )
    result_object._polars_cache.collect_columns("a")
    pl_testing.assert_frame_equal(
        result_object.live_dataframe, lf.with_columns((pl.col("a") * 10))
    )

    result_object = Result(base_dataframe=lf, info={})
    result_object._polars_cache.collect_columns("a")
    result_object.live_dataframe = result_object.live_dataframe.with_columns(
        (pl.col("a") * 10).alias("d")
    )
    pl.testing.assert_frame_equal(
        result_object.live_dataframe, lf.with_columns((pl.col("a") * 10).alias("d"))
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


def test_get(Result_fixture):
    """Test the get method."""
    current = Result_fixture.get("Current [A]")
    np_testing.assert_array_equal(
        current, Result_fixture.data["Current [A]"].to_numpy()
    )
    current_mA = Result_fixture.get("Current [mA]")
    np_testing.assert_array_equal(current_mA, current * 1000)

    current, voltage = Result_fixture.get("Current [A]", "Voltage [V]")
    np_testing.assert_array_equal(
        current, Result_fixture.data["Current [A]"].to_numpy()
    )
    np_testing.assert_array_equal(
        voltage, Result_fixture.data["Voltage [V]"].to_numpy()
    )


def test_get_only(Result_fixture):
    """Test the get_only method."""
    current = Result_fixture.get("Current [A]")
    np_testing.assert_array_equal(
        current, Result_fixture.data["Current [A]"].to_numpy()
    )
    current_mA = Result_fixture.get("Current [mA]")
    np_testing.assert_array_equal(current_mA, current * 1000)


def test_getitem(Result_fixture):
    """Test the __getitem__ method."""
    current = Result_fixture["Current [A]"]
    assert "Current [A]" in current.column_list
    assert isinstance(current, Result)
    pl_testing.assert_frame_equal(
        current.data, Result_fixture.data.select("Current [A]")
    )
    current_mA = Result_fixture["Current [mA]"]
    assert "Current [mA]" in current_mA.column_list
    assert "Current [A]" not in current_mA.column_list
    np_testing.assert_allclose(
        current_mA.get("Current [mA]"), Result_fixture.get("Current [mA]")
    )


def test_data(Result_fixture):
    """Test the data property."""
    assert isinstance(Result_fixture.base_dataframe, pl.LazyFrame)
    assert isinstance(Result_fixture.data, pl.DataFrame)
    assert isinstance(Result_fixture.live_dataframe, pl.DataFrame)
    pl_testing.assert_frame_equal(Result_fixture.data, Result_fixture.live_dataframe)


def test_quantities(Result_fixture):
    """Test the quantities property."""
    assert set(Result_fixture.quantities) == set(
        ["Time", "Current", "Voltage", "Capacity", "Event", "Date", "Step"]
    )


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
        }
    )
    pl_testing.assert_frame_equal(
        result.data, expected_data, check_column_order=False, check_dtype=False
    )


def test_add_new_data_columns():
    """Test the add_new_data_columns method."""
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
        }
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
        }
    )
    result_object = Result(base_dataframe=existing_data, info={})
    result_object.add_new_data_columns(new_data, date_column_name="DateTime")
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
            .alias("datetime"),
            "Data": [2, 4, 6, 8, 10, 12],
            "Data 1": [None, None, None, 3.0, 5.0, 7.0],
            "Data 2": [None, None, None, 6.0, 10.0, 14.0],
        }
    )
    pl_testing.assert_frame_equal(
        result_object.data, expected_data, check_column_order=False
    )


@pytest.fixture
def reduced_result_fixture():
    """Return a Result instance with reduced data."""
    data = pl.DataFrame(
        {
            "Current [A]": [1, 2, 3],
            "Voltage [V]": [1, 2, 3],
        }
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
        lazy_df1, [lazy_df2], mode="collect all"
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
        lazy_df1, [df2, lazy_df2], mode="match 1"
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
        }
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
        }
    )
    pl_testing.assert_frame_equal(
        reduced_result_fixture.data, expected_data, check_column_order=False
    )
    assert reduced_result_fixture.column_definitions["Voltage"] == "Voltage definition"


def test_extend(reduced_result_fixture):
    """Test the extend method."""
    other_data = pl.DataFrame(
        {
            "Current [A]": [4, 5, 6],
            "Voltage [V]": [4, 5, 6],
        }
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
        }
    )
    pl_testing.assert_frame_equal(
        reduced_result_fixture.data, expected_data, check_column_order=False
    )
    assert reduced_result_fixture.column_definitions["Voltage"] == "Voltage definition"


def test_extend_with_new_columns(reduced_result_fixture):
    """Test the extend method with new columns."""
    other_data = pl.DataFrame(
        {
            "Current [A]": [4, 5, 6],
            "Voltage [V]": [4, 5, 6],
            "Capacity [Ah]": [8, 9, 10],
        }
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
        }
    )
    pl_testing.assert_frame_equal(
        reduced_result_fixture.data, expected_data, check_column_order=False
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
        dataframe=new_df, column_definitions=new_defs
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
        }
    )
    pl_testing.assert_frame_equal(
        combined_result.data, expected_data, check_column_order=False
    )
