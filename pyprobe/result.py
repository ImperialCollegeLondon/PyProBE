"""A module for the Result class."""
import warnings
from pprint import pprint
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from numpy.typing import NDArray

from pyprobe.unitconverter import UnitConverter


class Result:
    """A result object for returning data and plotting.

    Attributes:
        dataframe (pl.LazyFrame | pl.DataFrame): The data as a polars DataFrame or
            LazyFrame.
        info (Dict[str, str | int | float]): A dictionary containing test info.
    """

    def __init__(
        self,
        dataframe: pl.LazyFrame | pl.DataFrame,
        info: Dict[str, str | int | float],
        column_definitions: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize the Result object.

        Args:
            dataframe (pl.LazyFrame | pl.DataFrame): The filtered data.
            info (Dict[str, str | int | float]): A dict containing test info.
            column_definitions(Optional[Dict[str, str]]):
                A dict containing the definitions of the columns in data.
        """
        self._data = dataframe
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
        Deprecated:
            This method will be removed in a future version. Use `array` instead.
        """
        warnings.warn(
            "The __call__ method is deprecated and will be removed in a future version."
            "Use `array` instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.check_units(column_name)
        if column_name not in self.data.columns:
            raise ValueError(f"Column '{column_name}' not in data.")
        else:
            return self.data[column_name].to_numpy()

    def __getitem__(self, *column_name: str) -> "Result":
        """Return a new result object with the specified columns.

        Args:
            *column_name (str): The columns to include in the new result object.

        Returns:
            Result: A new result object with the specified columns.
        """
        column_names = list(column_name)
        for col in column_names:
            self.check_units(col)
        if not all(col in self.data.columns for col in column_names):
            raise ValueError("One or more columns not in data.")
        else:
            return Result(self.data.select(column_names), self.info)

    def get(
        self, *column_names: str
    ) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], ...]]:
        """Return one or more columns of the data as numpy arrays.

        Args:
            column_names (str): The column name(s) to return.

        Returns:
            Union[NDArray[np.float64], Tuple[NDArray[np.float64], ...]]:
                The column(s) as numpy array(s).
        """
        if not column_names:
            raise ValueError("At least one column name must be provided.")

        for column_name in column_names:
            self.check_units(column_name)
            if column_name not in self.data.columns:
                raise ValueError(f"Column '{column_name}' not in data.")

        arrays = tuple(
            self.data[column_name].to_numpy() for column_name in column_names
        )
        return arrays if len(arrays) > 1 else arrays[0]

    def get_only(self, column_name: str) -> NDArray[np.float64]:
        """Return a single column of the data as a numpy array.

        Args:
            column_name (str): The column name to return.

        Returns:
            NDArray[np.float64]: The column as a numpy array.
        """
        data = self.get(column_name)
        if not isinstance(data, np.ndarray):
            raise ValueError("More than one column returned.")
        return data

    def array(self, *filtering_column_names: str) -> NDArray[np.float64]:
        """Return the data as a numpy array.

        Args:
            *filtering_column_names (str): The column names to return.

        Returns:
            NDArray[np.float64]: The data as a numpy array.
        """
        print("filtering_column_names", filtering_column_names)
        if len(filtering_column_names) == 0:
            return self.data.to_numpy()
        else:
            column_names = list(filtering_column_names)
            for col in column_names:
                self.check_units(col)
            if not all(col in self.data.columns for col in column_names):
                raise ValueError("One or more columns not in data.")
            else:
                data_to_convert = self.data.select(column_names)
                return data_to_convert.to_numpy()

    @property
    def data(self) -> pl.DataFrame:
        """Return the data as a polars DataFrame.

        Returns:
            pl.DataFrame: The data as a polars DataFrame.
        """
        if isinstance(self._data, pl.LazyFrame):
            self._data = self._data.collect()
        if self._data.is_empty():
            raise ValueError("No data exists for this filter.")
        return self._data

    def check_units(self, column_name: str) -> None:
        """Check if a column exists and convert the units if it does not.

        Args:
            column_name (str): The column name to convert to.
        """
        if column_name not in self.data.columns:
            converter_object = UnitConverter(column_name)
            if converter_object.input_quantity in self.quantities:
                instruction = converter_object.from_default()
                self._data = self.data.with_columns(instruction)
                self.define_column(
                    column_name,
                    self.column_definitions[
                        f"{converter_object.input_quantity} "
                        f"[{converter_object.default_unit}]"
                    ],
                )

    @property
    def quantities(self) -> List[str]:
        """Return the quantities of the data, with unit information removed."""
        _quantities = []
        for _, column in enumerate(self.column_list):
            try:
                quantity, _ = UnitConverter.get_quantity_and_unit(column)
                _quantities.append(quantity)
            except ValueError:
                continue
        return _quantities

    @property
    def column_list(self) -> List[str]:
        """Return a list of the columns in the data."""
        return self.data.columns

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

    def clean_copy(
        self,
        dataframe: Optional[Dict[str, NDArray[np.float64]]] = {},
        column_definitions: Optional[Dict[str, str]] = None,
    ) -> "Result":
        """Create a copy of the result object with info dictionary but without data.

        Args:
            dataframe (Optional[Dict[str, NDArray[np.float64]]):
                The data to include in the new result object.
            column_definitions (Optional[Dict[str, str]]):
                The definitions of the columns in the new result object.

        Returns:
            Result: A new result object with the specified data.
        """
        return Result(pl.DataFrame(dataframe), self.info, column_definitions)

    @classmethod
    def build(
        cls,
        data_list: List[
            pl.LazyFrame
            | pl.DataFrame
            | Dict[str, NDArray[np.float64] | List[float]]
            | List[
                pl.LazyFrame
                | pl.DataFrame
                | Dict[str, NDArray[np.float64] | List[float]]
            ]
        ],
        info: Dict[str, str | int | float],
    ) -> "Result":
        """Build a Result object from a list of dataframes.

        Args:
            data_list (List[List[pl.LazyFrame | pl.DataFrame | Dict]]):
                The data to include in the new result object.
                The first index indicates the cycle and the second index indicates the
                step.
            info (Dict[str, str | int | float]): A dict containing test info.

        Returns:
            Result: A new result object with the specified data.
        """
        cycles_and_steps_given = all(isinstance(item, list) for item in data_list)
        if not cycles_and_steps_given:
            data_list = [data_list]
        data = []
        for cycle, cycle_data in enumerate(data_list):
            for step, step_data in enumerate(cycle_data):
                if isinstance(step_data, dict):
                    step_data = pl.DataFrame(step_data)
                step_data = step_data.with_columns(
                    pl.lit(cycle).alias("Cycle"), pl.lit(step).alias("Step")
                )
                data.append(step_data)
        data = pl.concat(data)
        return cls(data, info)
