"""Tests for the Plot class."""

import polars as pl
import polars.testing as pl_testing
import pytest

from pyprobe import plot
from pyprobe.result import Result


def test_retrieve_relevant_columns_args():
    """Test _retrieve_relevant_columns with positional arguments."""
    # Set up test data
    data = pl.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]})
    result = Result(base_dataframe=data, info={})

    # Test with args only
    args = ["col1", "col2"]
    kwargs = {}
    output = plot._retrieve_relevant_columns(result, args, kwargs)

    assert isinstance(output, pl.DataFrame)
    assert set(output.columns) == {"col1", "col2"}
    assert output.shape == (3, 2)


def test_retrieve_relevant_columns_kwargs():
    """Test _retrieve_relevant_columns with keyword arguments."""
    data = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]})
    result = Result(base_dataframe=data, info={})

    # Test with kwargs only
    args = []
    kwargs = {"x_col": "x", "y_col": "y"}
    output = plot._retrieve_relevant_columns(result, args, kwargs)

    assert isinstance(output, pl.DataFrame)
    assert set(output.columns) == {"x", "y"}
    assert output.shape == (3, 2)


def test_retrieve_relevant_columns_mixed():
    """Test _retrieve_relevant_columns with both args and kwargs."""
    data = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    result = Result(base_dataframe=data, info={})

    args = ["a"]
    kwargs = {"col": "b"}
    output = plot._retrieve_relevant_columns(result, args, kwargs)

    assert isinstance(output, pl.DataFrame)
    assert set(output.columns) == {"a", "b"}
    assert output.shape == (3, 2)


def test_retrieve_relevant_columns_lazy():
    """Test _retrieve_relevant_columns with LazyFrame."""
    data = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}).lazy()
    result = Result(base_dataframe=data, info={})

    args = ["x"]
    kwargs = {"y_col": "y"}
    output = plot._retrieve_relevant_columns(result, args, kwargs)

    assert isinstance(output, pl.DataFrame)  # Should be collected
    assert not isinstance(output, pl.LazyFrame)
    assert set(output.columns) == {"x", "y"}


def test_retrieve_relevant_columns_intersection():
    """Test _retrieve_relevant_columns column intersection behavior."""
    data = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = Result(base_dataframe=data, info={})

    # Request columns including ones that don't exist
    args = ["a", "nonexistent1"]
    kwargs = {"col": "b", "missing": "nonexistent2"}
    output = plot._retrieve_relevant_columns(result, args, kwargs)

    assert isinstance(output, pl.DataFrame)
    assert set(output.columns) == {"a", "b"}  # Only existing columns
    assert output.shape == (3, 2)


def test_retrieve_relevant_columns_no_columns():
    """Test _retrieve_relevant_columns with no columns."""
    data = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = Result(base_dataframe=data, info={})

    # Request columns that don't exist
    args = ["nonexistent1"]
    kwargs = {"missing": "nonexistent2"}

    with pytest.raises(ValueError):
        plot._retrieve_relevant_columns(result, args, kwargs)


def test_retrieve_relevant_columns_with_unit_conversion():
    """Test _retrieve_relevant_columns with unit conversion."""
    data = pl.DataFrame({"I [A]": [1, 2, 3], "V [V]": [4, 5, 6]})
    result = Result(
        base_dataframe=data,
        info={},
        column_definitions={"I": "Current", "V": "Voltage"},
    )

    args = ["I [mA]"]
    kwargs = {"y_col": "V [kV]"}
    output = plot._retrieve_relevant_columns(result, args, kwargs)

    expected_data = pl.DataFrame(
        {"I [mA]": [1e3, 2e3, 3e3], "V [kV]": [4e-3, 5e-3, 6e-3]}
    )
    pl_testing.assert_frame_equal(output, expected_data, check_column_order=False)


def test_seaborn_wrapper_creation():
    """Test basic seaborn wrapper creation."""
    pytest.importorskip("seaborn")
    wrapper = plot._create_seaborn_wrapper()
    assert wrapper is not None
    assert isinstance(wrapper, object)


def test_seaborn_wrapper_data_conversion(mocker):
    """Test that wrapped functions convert data correctly."""
    sns = pytest.importorskip("seaborn")
    result = Result(
        base_dataframe=pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}),
        info={},
        column_definitions={"x": "int", "y": "int"},
    )
    data = result.data.to_pandas()
    pyprobe_seaborn_plot = plot.seaborn.lineplot(data=result, x="x", y="y")
    seaborn_lineplot = sns.lineplot(data=data, x="x", y="y")
    assert pyprobe_seaborn_plot == seaborn_lineplot


def test_seaborn_wrapper_function_call():
    """Test that wrapped functions produce same output."""
    sns = pytest.importorskip("seaborn")
    wrapper = plot._create_seaborn_wrapper()

    assert wrapper.set_theme() == sns.set_theme()

    colors1 = wrapper.color_palette()
    colors2 = sns.color_palette()
    assert colors1 == colors2

    # Test with specific parameters
    palette1 = wrapper.color_palette("husl", 8)
    palette2 = sns.color_palette("husl", 8)
    assert palette1 == palette2


def test_seaborn_wrapper_function_properties():
    """Test that wrapped functions maintain original properties."""
    sns = pytest.importorskip("seaborn")
    wrapper = plot._create_seaborn_wrapper()
    original_func = sns.lineplot
    wrapped_func = wrapper.lineplot

    assert wrapped_func.__name__ == original_func.__name__
    assert wrapped_func.__doc__ == original_func.__doc__


def test_seaborn_wrapper_complete_coverage():
    """Test that all public seaborn attributes are wrapped."""
    sns = pytest.importorskip("seaborn")
    wrapper = plot._create_seaborn_wrapper()
    sns_attrs = {attr for attr in dir(sns) if not attr.startswith("_")}
    wrapper_attrs = {attr for attr in dir(wrapper) if not attr.startswith("_")}
    assert sns_attrs == wrapper_attrs
