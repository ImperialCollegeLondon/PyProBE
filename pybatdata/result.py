"""A module for the Result class."""
from typing import Dict, Optional

import plotly.graph_objects as go
import polars as pl
from plotly.express.colors import sample_colorscale
from sklearn.preprocessing import minmax_scale

from pybatdata.units import Units


class Result:
    """A result object for returning data and plotting.

    Attributes:
        lazyframe (pl.LazyFrame): The filtered lazyframe.
        dataframe (Optional[pl.DataFrame]): The data as a polars DataFrame.
        info (Dict[str, str | int | float]): A dictionary containing test info.
    """

    def __init__(
        self, lazyframe: pl.LazyFrame, info: Dict[str, str | int | float]
    ) -> None:
        """Initialize the Result object.

        Args:
            lazyframe (pl.LazyFrame): The filtered lazyframe.
            info (dict): A dictionary containing info about the test.
        """
        self.lazyframe = lazyframe
        self.dataframe: Optional[pl.DataFrame] = None
        self.info = info

    @property
    def data(self) -> pl.DataFrame:
        """Return the data as a polars DataFrame.

        Returns:
            pl.DataFrame: The data as a polars DataFrame.
        """
        if self.dataframe is None:
            instruction_list = []
            for column in self.lazyframe.columns:
                new_instruction = Units.set_zero(column)
                if new_instruction is not None:
                    instruction_list.extend(new_instruction)
                new_instruction = Units.convert_units(column)
                if new_instruction is not None:
                    instruction_list.extend(new_instruction)
            self.lazyframe = self.lazyframe.with_columns(instruction_list)
            self.dataframe = self.lazyframe.collect()
            if self.dataframe.shape[0] == 0:
                raise ValueError("No data exists for this filter.")
        return self.dataframe

    def print(self) -> None:
        """Print the data."""
        print(self.data)

    def plot(
        self,
        fig: go.Figure,
        x: str,
        y: str,
        secondary_y: Optional[str] = None,
        color_by: Optional[str] = None,
        label: Optional[str] = None,
        legend_by: str = "Name",
        colormap: str = "viridis",
    ) -> go.Figure:
        """Plot the selected columns of the data.

        Args:
            fig (go.Figure): The plotly figure to update.
            x (str): The x-axis column.
            y (str): The y-axis column.
            secondary_y (Optional[str]): The secondary y-axis column.
            color_by (Optional[str]): The column to color by.
            label (Optional[str]): The label to use for the legend.
            legend_by (str): The column to use for the legend.
            colormap (str): The colormap to use.
        """
        title_font_size = 18
        axis_font_size = 14
        plot_theme = "simple_white"
        x_range = [self.data[x].min(), self.data[x].max()]
        y_range = [self.data[y].min(), self.data[y].max()]
        x_buffer = 0.05 * (x_range[1] - x_range[0])  # type: ignore
        y_buffer = 0.05 * (y_range[1] - y_range[0])  # type: ignore

        if color_by is not None:
            if color_by in self.data.columns:
                unique_colors = self.data[color_by].unique().to_numpy()
                colors = sample_colorscale(colormap, minmax_scale(unique_colors))
                for i, condition in enumerate(unique_colors):
                    subset = self.data.filter(pl.col(color_by) == condition)
                    fig.add_trace(
                        go.Scatter(
                            x=subset[x],
                            y=subset[y],
                            mode="lines",
                            line=dict(color=colors[i]),
                            name=str(unique_colors[i]),
                            showlegend=False,
                        )
                    )
                if secondary_y is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=subset[x],
                            y=subset[secondary_y],
                            mode="lines",
                            line=dict(color=colors[i], dash="dash"),
                            name=str(unique_colors[i]),
                            yaxis="y2",
                            showlegend=False,
                        )
                    )

                # Dummy heatmap for colorbar
                fig.add_trace(
                    go.Heatmap(
                        z=[
                            [unique_colors.min(), unique_colors.max()]
                        ],  # Set z to the range of the color scale
                        colorscale=colormap,  # choose a colorscale
                        colorbar=dict(title=color_by),  # add colorbar
                        opacity=0,
                    )
                )
                fig.update_xaxes(
                    range=[x_range[0] - x_buffer, x_range[1] + x_buffer]
                )  # type: ignore
                fig.update_yaxes(
                    range=[y_range[0] - y_buffer, y_range[1] + y_buffer]
                )  # type: ignore
        else:
            color = self.info["color"]
            fig.add_trace(
                go.Scatter(
                    x=self.data[x],
                    y=self.data[y],
                    mode="lines",
                    line=dict(color=color),
                    name=self.info[legend_by] if label is None else label,
                )
            )
            if secondary_y is not None:
                fig.add_trace(
                    go.Scatter(
                        x=self.data[x],
                        y=self.data[secondary_y],
                        mode="lines",
                        line=dict(color=color, dash="dash"),
                        name=self.info[legend_by] if label is None else label,
                        yaxis="y2",
                        showlegend=False,
                    )
                )

            fig.update_layout(
                showlegend=True, legend=dict(font=dict(size=axis_font_size))
            )

        fig.update_layout(
            xaxis_title=x,
            yaxis_title=y,
            template=plot_theme,
            title_font=dict(size=title_font_size),
            xaxis_title_font=dict(size=title_font_size),
            yaxis_title_font=dict(size=title_font_size),
            xaxis_tickfont=dict(size=axis_font_size),
            yaxis_tickfont=dict(size=axis_font_size),
        )

        if secondary_y is not None:
            fig.update_layout(
                yaxis2=dict(title=secondary_y, overlaying="y", side="right"),
                yaxis2_tickfont=dict(size=axis_font_size),
                yaxis2_title_font=dict(size=title_font_size),
                legend=dict(x=1.2),
            )

        return fig
