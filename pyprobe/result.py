"""A module for the Result class."""
import warnings
from pprint import pprint
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from pyprobe.units import Units


class Result(BaseModel):
    """A result object for returning data and plotting.

    Args:
        base_dataframe (Union[pl.LazyFrame, pl.DataFrame]):
            The data as a polars DataFrame or LazyFrame.
        info (Dict[str, Union[str, int, float]]):
            A dictionary containing test info.
        column_definitions (Dict[str, str], optional):
            A dictionary containing the definitions of the columns in the data.
    """

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    base_dataframe: Union[pl.LazyFrame, pl.DataFrame]
    """The data as a polars DataFrame or LazyFrame."""
    info: Dict[str, Optional[str | int | float]]
    """A dictionary containing test info."""
    column_definitions: Dict[str, str] = Field(default_factory=dict)
    """A dictionary containing the definitions of the columns in the data."""

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

        self._check_units(column_name)
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
            self._check_units(col)
        if not all(col in self.data.columns for col in column_names):
            raise ValueError("One or more columns not in data.")
        else:
            return Result(base_dataframe=self.data.select(column_names), info=self.info)

    @property
    def data(self) -> pl.DataFrame:
        """Return the data as a polars DataFrame.

        Returns:
            pl.DataFrame: The data as a polars DataFrame.
        """
        if isinstance(self.base_dataframe, pl.LazyFrame):
            self.base_dataframe = self.base_dataframe.collect()
        if self.base_dataframe.is_empty():
            raise ValueError("No data exists for this filter.")
        return self.base_dataframe

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

        full_array = self._get_filtered_array(column_names)
        seperated_columns = tuple(full_array.T)
        if len(seperated_columns) == 1:
            return seperated_columns[0]
        else:
            return seperated_columns

    def get_only(self, column_name: str) -> NDArray[np.float64]:
        """Return a single column of the data as a numpy array.

        Args:
            column_name (str): The column name to return.

        Returns:
            NDArray[np.float64]: The column as a numpy array.
        """
        column = self.get(column_name)
        if not isinstance(column, np.ndarray):
            raise ValueError("More than one column returned.")
        return column

    def array(self, *filtering_column_names: str) -> NDArray[np.float64]:
        """Return the data as a numpy array.

        Args:
            *filtering_column_names (str): The column names to return.

        Returns:
            NDArray[np.float64]: The data as a numpy array.
        """
        if len(filtering_column_names) == 0:
            return self.data.to_numpy()
        else:
            return self._get_filtered_array(filtering_column_names)

    def _get_filtered_array(
        self, filtering_column_names: Tuple[str, ...]
    ) -> NDArray[np.float64]:
        for column_name in filtering_column_names:
            self._check_units(column_name)
            if column_name not in self.base_dataframe.columns:
                raise ValueError(f"Column '{column_name}' not in data.")
        frame_to_return = self.base_dataframe.select(filtering_column_names)
        if isinstance(frame_to_return, pl.LazyFrame):
            frame_to_return = frame_to_return.collect()
        return frame_to_return.to_numpy()

    def _check_units(self, column_name: str) -> None:
        """Check if a column exists and convert the units if it does not.

        Adds a new column to the dataframe with the desired unit.

        Args:
            column_name (str): The column name to convert to.
        """
        if column_name not in self.base_dataframe.columns:
            converter_object = Units(column_name)
            if converter_object.input_quantity in self.quantities:
                instruction = converter_object.from_default_unit()
                self.base_dataframe = self.base_dataframe.with_columns(instruction)
                self.define_column(
                    column_name,
                    self.column_definitions[
                        f"{converter_object.input_quantity} "
                        f"[{converter_object.default_unit}]"
                    ],
                )
            else:
                raise ValueError(
                    f"Column with quantity'{converter_object.input_quantity}' not in"
                    " data."
                )

    @property
    def quantities(self) -> List[str]:
        """Return the quantities of the data, with unit information removed."""
        _quantities = []
        for _, column in enumerate(self.column_list):
            try:
                quantity, _ = Units.get_quantity_and_unit(column)
                _quantities.append(quantity)
            except ValueError:
                continue
        return _quantities

    @property
    def column_list(self) -> List[str]:
        """Return a list of the columns in the data."""
        return self.base_dataframe.columns

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
        column_definitions: Dict[str, str] = {},
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
        return Result(
            base_dataframe=pl.DataFrame(dataframe),
            info=self.info,
            column_definitions=column_definitions,
        )

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
        info: Dict[str, Optional[str | int | float]],
    ) -> "Result":
        """Build a Result object from a list of dataframes.

        Args:
            data_list (List[List[pl.LazyFrame | pl.DataFrame | Dict]]):
                The data to include in the new result object.
                The first index indicates the cycle and the second index indicates the
                step.
            info (Dict[str, Optional[str | int | float]]): A dict containing test info.

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
        return cls(base_dataframe=data, info=info)
