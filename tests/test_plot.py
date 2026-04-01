"""Tests for the Plot class."""

import polars as pl
import polars.testing as pl_testing
import pytest

from pyprobe import plot
from pyprobe.result import Result


def test_get_plotting_data_args():
    """Test get_plotting_data with positional arguments."""
    # Set up test data with BDF format columns
    data = pl.DataFrame(
        {
            "Current / A": [1, 2, 3],
            "Voltage / V": [4, 5, 6],
            "Time / s": [7, 8, 9],
        }
    )
    result = Result(lf=data, metadata={})

    # Test with args only
    args = ["Current / A", "Voltage / V"]
    kwargs = {}
    output = result.get_plotting_data(args, kwargs)

    assert isinstance(output, pl.DataFrame)
    assert set(output.columns) == {"Current / A", "Voltage / V"}
    assert output.shape == (3, 2)


def test_get_plotting_data_kwargs():
    """Test get_plotting_data with keyword arguments."""
    data = pl.DataFrame(
        {
            "Time / s": [1, 2, 3],
            "Current / A": [4, 5, 6],
            "Voltage / V": [7, 8, 9],
        }
    )
    result = Result(lf=data, metadata={})

    # Test with kwargs only
    args = []
    kwargs = {"x_col": "Time / s", "y_col": "Current / A"}
    output = result.get_plotting_data(args, kwargs)

    assert isinstance(output, pl.DataFrame)
    assert set(output.columns) == {"Time / s", "Current / A"}
    assert output.shape == (3, 2)


def test_get_plotting_data_mixed():
    """Test get_plotting_data with both args and kwargs."""
    data = pl.DataFrame(
        {
            "Current / A": [1, 2, 3],
            "Voltage / V": [4, 5, 6],
            "Capacity / Ah": [7, 8, 9],
        }
    )
    result = Result(lf=data, metadata={})

    args = ["Current / A"]
    kwargs = {"col": "Voltage / V"}
    output = result.get_plotting_data(args, kwargs)

    assert isinstance(output, pl.DataFrame)
    assert set(output.columns) == {"Current / A", "Voltage / V"}
    assert output.shape == (3, 2)


def test_get_plotting_data_lazy():
    """Test get_plotting_data with LazyFrame."""
    data = pl.DataFrame(
        {
            "Time / s": [1, 2, 3],
            "Current / A": [4, 5, 6],
        }
    ).lazy()
    result = Result(lf=data, metadata={})

    args = ["Time / s"]
    kwargs = {"y_col": "Current / A"}
    output = result.get_plotting_data(args, kwargs)

    assert isinstance(output, pl.DataFrame)  # Should be collected
    assert not isinstance(output, pl.LazyFrame)
    assert set(output.columns) == {"Time / s", "Current / A"}


def test_get_plotting_data_intersection():
    """Test get_plotting_data column intersection behavior."""
    data = pl.DataFrame(
        {
            "Current / A": [1, 2, 3],
            "Voltage / V": [4, 5, 6],
        }
    )
    result = Result(lf=data, metadata={})

    # Request columns including ones that don't exist
    args = ["Current / A", "Nonexistent / A"]
    kwargs = {"col": "Voltage / V", "missing": "Missing / 1"}
    output = result.get_plotting_data(args, kwargs)

    assert isinstance(output, pl.DataFrame)
    assert set(output.columns) == {
        "Current / A",
        "Voltage / V",
    }  # Only existing columns
    assert output.shape == (3, 2)


def test_get_plotting_data_no_columns():
    """Test get_plotting_data with no columns."""
    data = pl.DataFrame(
        {
            "Current / A": [1, 2, 3],
            "Voltage / V": [4, 5, 6],
        }
    )
    result = Result(lf=data, metadata={})

    # Request columns that don't exist
    args = ["Nonexistent / A"]
    kwargs = {"missing": "Missing / 1"}

    with pytest.raises(ValueError):
        result.get_plotting_data(args, kwargs)


def test_get_plotting_data_with_unit_conversion():
    """Test get_plotting_data with unit conversion."""
    data = pl.DataFrame(
        {
            "Current / A": [1.0, 2.0, 3.0],
            "Voltage / V": [4.0, 5.0, 6.0],
        }
    )
    result = Result(lf=data, metadata={})

    args = ["Current / mA"]
    kwargs = {"y_col": "Voltage / kV"}
    output = result.get_plotting_data(args, kwargs)

    expected_data = pl.DataFrame(
        {
            "Current / mA": [1e3, 2e3, 3e3],
            "Voltage / kV": [4e-3, 5e-3, 6e-3],
        },
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
        lf=pl.DataFrame(
            {
                "Time / s": [1, 2, 3],
                "Current / A": [4, 5, 6],
            }
        ),
        metadata={},
    )
    data = result.data.to_pandas()
    pyprobe_seaborn_plot = plot.seaborn.lineplot(
        data=result,
        x="Time / s",
        y="Current / A",
    )
    seaborn_lineplot = sns.lineplot(
        data=data,
        x="Time / s",
        y="Current / A",
    )
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


def test_result_plot_method():
    """Test Result.plot() method."""
    pytest.importorskip("pandas")
    result = Result(
        lf=pl.DataFrame(
            {
                "Time / s": [1, 2, 3],
                "Current / A": [4, 5, 6],
                "Voltage / V": [7, 8, 9],
            }
        ),
        metadata={},
    )

    # Basic plot call should work
    ax = result.plot(x="Time / s", y="Current / A")
    assert ax is not None


def test_result_plot_method_with_lazy():
    """Test Result.plot() method with LazyFrame."""
    pytest.importorskip("pandas")
    result = Result(
        lf=pl.DataFrame(
            {
                "Time / s": [1, 2, 3],
                "Current / A": [4, 5, 6],
            }
        ).lazy(),
        metadata={},
    )

    # Plot should work with LazyFrame too
    ax = result.plot(x="Time / s", y="Current / A")
    assert ax is not None


def test_result_plot_method_missing_column():
    """Test Result.plot() raises KeyError for missing columns (from pandas)."""
    pytest.importorskip("pandas")
    result = Result(
        lf=pl.DataFrame(
            {
                "Time / s": [1, 2, 3],
                "Current / A": [4, 5, 6],
            }
        ),
        metadata={},
    )

    # Should raise KeyError from pandas when column doesn't exist
    with pytest.raises(KeyError):
        result.plot(x="Nonexistent / A", y="Current / A")


def test_result_hvplot_method():
    """Test Result.hvplot() method."""
    pytest.importorskip("hvplot")
    result = Result(
        lf=pl.DataFrame(
            {
                "Time / s": [1, 2, 3],
                "Current / A": [4, 5, 6],
                "Voltage / V": [7, 8, 9],
            }
        ),
        metadata={},
    )

    # Basic hvplot call should work
    plot_obj = result.hvplot(x="Time / s", y="Current / A")
    assert plot_obj is not None


def test_result_hvplot_method_with_lazy():
    """Test Result.hvplot() method with LazyFrame."""
    pytest.importorskip("hvplot")
    result = Result(
        lf=pl.DataFrame(
            {
                "Time / s": [1, 2, 3],
                "Current / A": [4, 5, 6],
            }
        ).lazy(),
        metadata={},
    )

    # hvplot should work with LazyFrame too
    plot_obj = result.hvplot(x="Time / s", y="Current / A")
    assert plot_obj is not None


def test_result_hvplot_method_missing_column():
    """Test Result.hvplot() raises ValueError for missing columns."""
    pytest.importorskip("hvplot")
    result = Result(
        lf=pl.DataFrame(
            {
                "Time / s": [1, 2, 3],
                "Current / A": [4, 5, 6],
            }
        ),
        metadata={},
    )

    # Should raise ValueError if column doesn't exist
    with pytest.raises(ValueError):
        result.hvplot(x="Nonexistent / A", y="Current / A")
