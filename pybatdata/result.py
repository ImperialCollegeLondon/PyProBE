"""A module for the Result class."""
from typing import Dict

import plotly.graph_objects as go
import polars as pl

from pybatdata.units import Units


class Result:
    """A result object for returning data and plotting.

    Attributes:
        _data (pl.LazyFrame | pl.DataFrame): The filtered _data.
        dataframe (Optional[pl.DataFrame]): The data as a polars DataFrame.
        info (Dict[str, str | int | float]): A dictionary containing test info.
    """

    def __init__(
        self,
        _data: pl.LazyFrame | pl.DataFrame,
        info: Dict[str, str | int | float],
    ) -> None:
        """Initialize the Result object.

        Args:
            _data (pl.LazyFrame | pl.DataFrame): The filtered _data.
            info (Dict[str, str | int | float]): A dict containing test info.
        """
        self._data = _data
        self._dataframe = None
        self.data_property_called = False
        self.info = info

    title_font_size = 18
    axis_font_size = 14
    plot_theme = "simple_white"
    plotly_layout = go.Layout(
        template=plot_theme,
        title_font=dict(size=title_font_size),
        xaxis_title_font=dict(size=title_font_size),
        yaxis_title_font=dict(size=title_font_size),
        xaxis_tickfont=dict(size=axis_font_size),
        yaxis_tickfont=dict(size=axis_font_size),
        width=800,
        height=600,
    )

    @property
    def data(self) -> pl.DataFrame:
        """Return the data as a polars DataFrame.

        Returns:
            pl.DataFrame: The data as a polars DataFrame.
        """
        if self.data_property_called is False:
            instruction_list = []
            for column in self._data.columns:
                new_instruction = Units.set_zero(column)
                if new_instruction is not None:
                    instruction_list.extend(new_instruction)
                new_instruction = Units.convert_units(column)
                if new_instruction is not None:
                    instruction_list.extend(new_instruction)
            self._data = self._data.with_columns(instruction_list)
        if isinstance(self._data, pl.LazyFrame):
            self._data = self._data.collect()
        self._dataframe = self._data
        if self._dataframe.shape[0] == 0:
            raise ValueError("No data exists for this filter.")
        self.data_property_called = True
        return self._dataframe

    def print(self) -> None:
        """Print the data."""
        print(self.data)
