"""Tests for the Plot class."""

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest
from plotly.express.colors import sample_colorscale
from sklearn.preprocessing import minmax_scale

from pyprobe.plot import Plot
from pyprobe.result import Result


@pytest.fixture
def Plot_fixture():
    """Return a Plot instance."""
    return Plot()


def test_init(Plot_fixture):
    """Test the __init__ method."""
    assert isinstance(Plot_fixture, Plot)
    assert Plot_fixture.layout == Plot.default_layout
    assert isinstance(Plot_fixture._fig, go.Figure)


def test_check_limits(Plot_fixture):
    """Test the _check_limits method."""
    data = pl.DataFrame({"x": [5, 6, 7, 8], "y": [1, 2, 3, 4]})
    x = data["x"]
    y = data["y"]
    Plot_fixture._check_limits(x, y)
    assert Plot_fixture.x_max == 8
    assert Plot_fixture.x_min == 5
    assert Plot_fixture.y_max == 4
    assert Plot_fixture.y_min == 1

    data = pl.DataFrame({"x": [-4, 6, 14, 12], "y": [2, 2, 3, 3]})
    x = data["x"]
    y = data["y"]

    Plot_fixture._check_limits(x, y)
    assert Plot_fixture.x_max == 14
    assert Plot_fixture.x_min == -4
    assert Plot_fixture.y_max == 4
    assert Plot_fixture.y_min == 1

    Plot_fixture._check_limits(x, y, secondary_y=True)
    assert Plot_fixture.y2_max == 3
    assert Plot_fixture.y2_min == 2


def test_ranges(Plot_fixture):
    """Test the ranges method."""
    Plot_fixture.x_max = 14
    Plot_fixture.x_min = -4
    Plot_fixture.y_max = 4
    Plot_fixture.y_min = 1
    Plot_fixture.y2_max = 3
    Plot_fixture.y2_min = 2

    assert Plot_fixture.x_range == [-4.9, 14.9]
    assert Plot_fixture.y_range == [0.85, 4.15]
    assert Plot_fixture.y2_range == [1.95, 3.05]


@pytest.fixture
def plot_result_fixture():
    """Return a Plot instance."""
    info = {"color": "blue", "Name": "Line 1"}
    data = pl.DataFrame(
        {"x": [1, 2, 3, 4], "y": [5, 6, 7, 8], "secondary_y": [2, 3, 4, 5]}
    )

    return Result(base_dataframe=data, info=info)


def test_make_colorscale(Plot_fixture):
    """Test the make_colorscale method."""
    colormap = "viridis"
    points = np.array([0, 3, 1, 2])
    colors = Plot_fixture.make_colorscale(points, colormap)

    assert isinstance(colors, list)
    assert colors == [
        "rgb(68, 1, 84)",
        "rgb(253, 231, 37)",
        "rgb(49, 104, 142)",
        "rgb(53, 183, 121)",
    ]


def test_add_line(Plot_fixture, plot_result_fixture):
    """Test the add_line method."""
    result = plot_result_fixture
    x = "x"
    y = "y"
    secondary_y = None
    color = None
    label = None
    showlegend = True

    Plot_fixture.add_line(
        result,
        x,
        y,
        secondary_y=secondary_y,
        color=color,
        label=label,
        showlegend=showlegend,
    )

    assert Plot_fixture.xaxis_title == x
    assert Plot_fixture.yaxis_title == y

    expected_trace = go.Scatter(
        x=result.data[x],
        y=result.data[y],
        mode="lines",
        line=dict(color=result.info["color"], dash="solid"),
        name=result.info["Name"],
        showlegend=showlegend,
    )
    assert Plot_fixture._fig.data[0] == expected_trace

    assert Plot_fixture.secondary_y is None


def test_add_line_with_secondary_y(Plot_fixture, plot_result_fixture):
    """Test the add_line method with secondary_y."""
    result = plot_result_fixture
    x = "x"
    y = "y"
    secondary_y = "secondary_y"
    color = None
    label = None
    showlegend = True

    Plot_fixture.add_line(
        result,
        x,
        y,
        secondary_y=secondary_y,
        color=color,
        label=label,
        showlegend=showlegend,
    )

    assert Plot_fixture.secondary_y == secondary_y

    expected_trace = go.Scatter(
        x=result.data[x],
        y=result.data[y],
        mode="lines",
        line=dict(color=result.info["color"], dash="solid"),
        name=result.info["Name"],
        showlegend=showlegend,
    )
    assert Plot_fixture._fig.data[0] == expected_trace

    expected_secondary_trace = go.Scatter(
        x=result.data[x],
        y=result.data[secondary_y],
        mode="lines",
        line=dict(color=result.info["color"], dash="dash"),
        name=result.info["Name"],
        yaxis="y2",
        showlegend=False,
        xaxis="x",
    )
    assert Plot_fixture._fig.data[1] == expected_secondary_trace


def test_add_colorscaled_line_with_colorbar(Plot_fixture, plot_result_fixture):
    """Test the add_colorscaled_line method."""
    result = plot_result_fixture
    x = "x"
    y = "y"
    color_by = "secondary_y"
    colormap = "viridis"

    Plot_fixture.add_colorscaled_line(result, x, y, color_by, colormap)

    assert Plot_fixture.xaxis_title == x
    assert Plot_fixture.yaxis_title == y

    unique_colors = result.data[color_by].unique(maintain_order=True).to_numpy()
    colors = sample_colorscale(colormap, minmax_scale(unique_colors))

    for i, condition in enumerate(unique_colors):
        subset = result.data.filter(pl.col(color_by) == condition)
        expected_trace = go.Scatter(
            x=subset[x],
            y=subset[y],
            mode="lines",
            line=dict(color=colors[i]),
            name=str(unique_colors[i]),
            showlegend=False,
        )
        assert Plot_fixture._fig.data[i] == expected_trace


def test_add_colorbar(Plot_fixture, plot_result_fixture):
    """Test the add_colorbar method."""
    result = plot_result_fixture
    color_by = "secondary_y"
    colormap = "viridis"
    unique_colors = result.data[color_by].unique(maintain_order=True).to_numpy()

    Plot_fixture.add_colorbar(
        [unique_colors.min(), unique_colors.max()], color_by, colormap
    )

    unique_colors = result.data[color_by].unique(maintain_order=True).to_numpy()

    expected_trace = go.Heatmap(
        z=[
            [unique_colors.min(), unique_colors.max()]
        ],  # Set z to the range of the color scale
        colorscale=colormap,  # choose a colorscale
        colorbar=dict(title=color_by),  # add colorbar
        opacity=0,
    )
    assert Plot_fixture._fig.data[0] == expected_trace


def test_add_secondary_y_legend(Plot_fixture):
    """Test the _add_secondary_y_legend method."""
    Plot_fixture.secondary_y = "secondary_y"
    Plot_fixture._add_secondary_y_legend("secondary_y")

    expected_trace = go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line=dict(color="black", dash="dash"),
        name="secondary_y",
        showlegend=True,
    )
    assert Plot_fixture._fig.data[0] == expected_trace


def test_fig(Plot_fixture, plot_result_fixture):
    """Test the fig method."""
    assert isinstance(Plot_fixture.fig, go.Figure)
    Plot_fixture.add_line(
        plot_result_fixture,
        "x",
        "y",
        "secondary_y",
        color="blue",
        label="Line 1",
        showlegend=True,
    )
    Plot_fixture.x_min = -4
    Plot_fixture.x_max = 14
    Plot_fixture.y_min = 1
    Plot_fixture.y_max = 4
    Plot_fixture.y2_min = 2
    Plot_fixture.y2_max = 3

    expected_xaxis_range = (-4.9, 14.9)
    expected_yaxis_range = (0.85, 4.15)
    expected_y2_range = (1.95, 3.05)

    expected_x_layout = go.layout.XAxis(
        anchor="y",
        domain=[0.0, 0.94],
        range=expected_xaxis_range,
        tickfont=dict(size=Plot.axis_font_size),
        title={"text": "x", "font": {"size": Plot.title_font_size}},
    )

    expected_y_layout = go.layout.YAxis(
        anchor="x",
        domain=[0.0, 1.0],
        range=expected_yaxis_range,
        tickfont=dict(size=Plot.axis_font_size),
        title={"text": "y", "font": {"size": Plot.title_font_size}},
    )

    expected_y2_layout = go.layout.YAxis(
        anchor="x",
        overlaying="y",
        range=expected_y2_range,
        side="right",
        tickfont=dict(size=Plot.axis_font_size),
        title={"text": "secondary_y", "font": {"size": Plot.title_font_size}},
    )

    expected_legend_layout = go.layout.Legend({"font": {"size": 14}, "x": 1.2})

    result = Plot_fixture.fig

    # Assertions
    assert result == Plot_fixture._fig
    assert result.layout.xaxis.range == expected_xaxis_range
    assert result.layout.yaxis.range == expected_yaxis_range
    assert result.layout.yaxis2.range == expected_y2_range

    assert result.layout.xaxis == expected_x_layout
    assert result.layout.yaxis == expected_y_layout
    assert result.layout.yaxis2 == expected_y2_layout
    assert result.layout.legend == expected_legend_layout


def test_show(Plot_fixture, mocker):
    """Test the show method."""
    mocker.patch.object(Plot_fixture.fig, "show")

    Plot_fixture.show()

    Plot_fixture.fig.show.assert_called_once()
