"""A module for the Result class."""
from pprint import pprint
from typing import Dict, Optional

import numpy as np
import polars as pl
from numpy.typing import NDArray

from pyprobe.unitconverter import UnitConverter


class Result:
    """A result object for returning data and plotting.

    Attributes:
        _data (pl.LazyFrame | pl.DataFrame): The data as a polars DataFrame or
            LazyFrame.
        data (Optional[pl.DataFrame]): The data as a polars DataFrame.
        info (Dict[str, str | int | float]): A dictionary containing test info.
    """

    def __init__(
        self,
        _data: pl.LazyFrame | pl.DataFrame,
        info: Dict[str, str | int | float],
        column_definitions: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize the Result object.

        Args:
            _data (pl.LazyFrame | pl.DataFrame): The filtered _data.
            info (Dict[str, str | int | float]): A dict containing test info.
            column_definitions(Optional[Dict[str, str]]):
                A dict containing the definitions of the columns in _data.
        """
        self._data = _data
        self.info = info
        if column_definitions is None:
            self.column_definitions: Dict[str, str] = {}
        else:
            self.column_definitions = column_definitions

    def __call__(self, column_name: str) -> NDArray[np.float64]:
        """Return columns of the data as numpy arrays.

        Args:
            column_name (str): The column names to return.

        Returns:
            Union[NDArray[np.float64]:
                The column as a numpy array.
        """
        self.check_units(column_name)
        if column_name not in self.data.columns:
            raise ValueError(f"Column '{column_name}' not in data.")
        else:
            return self.data[column_name].to_numpy()

    @property
    def data(self) -> pl.DataFrame:
        """Return the data as a polars DataFrame.

        Returns:
            pl.DataFrame: The data as a polars DataFrame.
        """
        if isinstance(self._data, pl.LazyFrame):
            self._data = self._data.collect()
        return self._data

    def print(self) -> None:
        """Print the data."""
        print(self.data)

    def check_units(self, column_name: str) -> None:
        """Check if a column exists and convert the units if it does not.

        Args:
            column_name (str): The column name to convert to.
        """
        if column_name not in self.data.columns:
            instruction = UnitConverter(column_name).from_default()
            self._data = self.data.with_columns(instruction)

    def define_column(self, column_name: str, definition: str) -> None:
        """Define a new column when it is added to the dataframe.

        Args:
            column_name (str): The name of the column.
            definition (str): The definition of the quantity stored in the column
        """
        self.column_definitions[column_name] = definition

    def print_definitions(self) -> None:
        """Print the definitions of the columns stored in this result object."""
        pprint(self.column_definitions)
