"""A module for the Result class."""
import logging
import warnings
from pprint import pprint
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import BaseModel, Field, model_validator

from pyprobe.units import unit_from_regexp

logger = logging.getLogger(__name__)


class Result(BaseModel):
    """A class for holding any data in PyProBE.

    A Result object is the base type for every data object in PyProBE. This class
    includes all of the main methods for returning and describing any data in PyProBE.

    Key attributes for returning data:
        - :attr:`data`: The data as a Polars DataFrame.
        - :meth:`get`: Get a column from the data as a NumPy array.

    Key attributes for describing the data:
        - :attr:`info`: A dictionary containing information about the cell.
        - :attr:`column_definitions`: A dictionary of column definitions.
        - :meth:`print_definitions`: Print the column definitions.
        - :attr:`column_list`: A list of column names.
    """

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    base_dataframe: Union[pl.LazyFrame, pl.DataFrame]
    """The data as a polars DataFrame or LazyFrame."""
    info: Dict[str, Optional[str | int | float | Dict[Any, Any]]]
    """Dictionary containing information about the cell."""
    column_definitions: Dict[str, str] = Field(default_factory=dict)
    """A dictionary containing the definitions of the columns in the data."""

    @model_validator(mode="before")
    @classmethod
    def _load_base_dataframe(cls, data: Any) -> Any:
        """Load the base dataframe from a file if provided as a string."""
        if "base_dataframe" in data and isinstance(data["base_dataframe"], str):
            data["base_dataframe"] = pl.scan_parquet(data["base_dataframe"])
        return data

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
        if column_name not in self.data.collect_schema().names():
            error_msg = f"Column '{column_name}' not in data."
            logger.error(error_msg)
            raise ValueError(error_msg)
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
        if not all(col in self.data.collect_schema().names() for col in column_names):
            error_msg = "One or more columns not in data."
            logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            return Result(base_dataframe=self.data.select(column_names), info=self.info)

    @property
    def data(self) -> pl.DataFrame:
        """Return the data as a polars DataFrame.

        Returns:
            pl.DataFrame: The data as a polars DataFrame.

        Raises:
            ValueError: If no data exists for this filter.
        """
        if isinstance(self.base_dataframe, pl.LazyFrame):
            self.base_dataframe = self.base_dataframe.collect()
        if self.base_dataframe.is_empty():
            error_msg = "No data exists for this filter."
            logger.error(error_msg)
            raise ValueError(error_msg)
        return self.base_dataframe

    @property
    def contains_lazyframe(self) -> bool:
        """Return whether the data is a LazyFrame.

        Returns:
            bool: True if the data is a LazyFrame, False otherwise.
        """
        return isinstance(self.base_dataframe, pl.LazyFrame)

    def get(
        self, *column_names: str
    ) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], ...]]:
        """Return one or more columns of the data as separate 1D numpy arrays.

        Args:
            column_names (str): The column name(s) to return.

        Returns:
            Union[NDArray[np.float64], Tuple[NDArray[np.float64], ...]]:
                The column(s) as numpy array(s).

        Raises:
            ValueError: If no column names are provided.
            ValueError: If a column name is not in the data.
        """
        if not column_names:
            error_msg = "At least one column name must be provided."
            logger.error(error_msg)
            raise ValueError(error_msg)

        full_array = self._get_filtered_array(column_names)
        separated_columns = tuple(full_array.T)
        if len(separated_columns) == 1:
            return separated_columns[0]
        else:
            return separated_columns

    def get_only(self, column_name: str) -> NDArray[np.float64]:
        """Return a single column of the data as a numpy array.

        Args:
            column_name (str): The column name to return.

        Returns:
            NDArray[np.float64]: The column as a numpy array.

        Raises:
            ValueError: If the column name is not in the data.
            ValueError: If no column name is provided.
        """
        column = self.get(column_name)
        if not isinstance(column, np.ndarray):
            error_msg = "More than one column returned."
            logger.error(error_msg)
            raise ValueError(error_msg)
        return column

    def array(self, *filtering_column_names: str) -> NDArray[np.float64]:
        """Return the data as a single numpy array.

        Args:
            *filtering_column_names (str): The column names to return.

        Returns:
            NDArray[np.float64]: The data as a single numpy array.

        Raises:
            ValueError: If a column name is not in the data.
        """
        if len(filtering_column_names) == 0:
            return self.data.to_numpy()
        else:
            return self._get_filtered_array(filtering_column_names)

    def _get_filtered_array(
        self, filtering_column_names: Tuple[str, ...]
    ) -> NDArray[np.float64]:
        """Return the data as a single numpy array from a list of column names.

        Args:
            filtering_column_names (Tuple[str, ...]): The column names to return.

        Returns:
            NDArray[np.float64]: The data as a single numpy array.

        Raises:
            ValueError: If a column name is not in the data.
        """
        for column_name in filtering_column_names:
            self._check_units(column_name)
            if column_name not in self.base_dataframe.collect_schema().names():
                error_msg = f"Column '{column_name}' not in data."
                logger.error(error_msg)
                raise ValueError(error_msg)
        frame_to_return = self.base_dataframe.select(filtering_column_names)
        if isinstance(frame_to_return, pl.LazyFrame):
            frame_to_return = frame_to_return.collect()
        return frame_to_return.to_numpy()

    def _check_units(self, column_name: str) -> None:
        """Check if a column exists and convert the units if it does not.

        Adds a new column to the dataframe with the desired unit.

        Args:
            column_name (str): The column name to convert to.

        Raises:
            ValueError: If the column name is not in the data.
        """
        if column_name not in self.base_dataframe.collect_schema().names():
            converter_object = unit_from_regexp(column_name)
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
                error_msg = f"Column with quantity '{converter_object.input_quantity}'"
                " not in data."
                logger.error(error_msg)
                raise ValueError(error_msg)

    @property
    def quantities(self) -> List[str]:
        """The quantities of the data, with unit information removed.

        Returns:
            List[str]: The quantities of the data.
        """
        _quantities = []
        for _, column in enumerate(self.column_list):
            try:
                quantity = unit_from_regexp(column).input_quantity
                _quantities.append(quantity)
            except ValueError:
                continue
        return _quantities

    @property
    def column_list(self) -> List[str]:
        """The columns in the data.

        Returns:
            List[str]: The columns in the data.
        """
        return self.base_dataframe.collect_schema().names()

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
        dataframe: Optional[Union[pl.DataFrame, pl.LazyFrame]] = None,
        column_definitions: Optional[Dict[str, str]] = None,
    ) -> "Result":
        """Create a copy of the result object with info dictionary but without data.

        Args:
            dataframe (Optional[Union[pl.DataFrame, pl.LazyFrame]):
                The data to include in the new Result object.
            column_definitions (Optional[Dict[str, str]]):
                The definitions of the columns in the new result object.

        Returns:
            Result: A new result object with the specified data.
        """
        if dataframe is None:
            dataframe = pl.DataFrame({})
        if column_definitions is None:
            column_definitions = {}
        return Result(
            base_dataframe=dataframe,
            info=self.info,
            column_definitions=column_definitions,
        )

    @staticmethod
    def _verify_compatible_frames(
        base_frame: Union[pl.DataFrame, pl.LazyFrame],
        frames: List[Union[pl.DataFrame, pl.LazyFrame]],
        mode: Literal["match 1", "collect all"] = "collect all",
    ) -> Tuple[
        Union[pl.DataFrame, pl.LazyFrame], List[Union[pl.DataFrame, pl.LazyFrame]]
    ]:
        """Verify that frames are compatible and return them as DataFrames.

        Args:
            base_frame (pl.DataFrame | pl.LazyFrame): The first frame to verify.
            frames (List[pl.DataFrame | pl.LazyFrame]): The list of frames to verify.

        Returns:
            Tuple[pl.DataFrame | pl.LazyFrame, List[pl.DataFrame | pl.LazyFrame]]:
                The first frame and the list of verified frames as DataFrames.
        """
        verified_frames = []
        for frame in frames:
            if isinstance(base_frame, pl.LazyFrame) and isinstance(frame, pl.DataFrame):
                if mode == "match 1":
                    frame = frame.lazy()
                elif mode == "collect all":
                    base_frame = base_frame.collect()
            elif isinstance(base_frame, pl.DataFrame) and isinstance(
                frame, pl.LazyFrame
            ):
                frame = frame.collect()
            verified_frames.append(frame)

        return base_frame, verified_frames

    def add_new_data_columns(
        self, new_data: pl.DataFrame | pl.LazyFrame, date_column_name: str
    ) -> None:
        """Add new data columns to the result object.

        The data must be time series data with a date column. The new data is joined to
        the base dataframe on the date column, and the new data columns are interpolated
        to fill in missing values.

        Args:
            new_data (pl.DataFrame | pl.LazyFrame):
                The new data to add to the result object.
            date_column_name (str):
                The name of the column in the new data containing the date.

        Raises:
            ValueError: If the base dataframe has no date column.
        """
        if "Date" not in self.column_list:
            error_msg = "No date column in the base dataframe."
            logger.error(error_msg)
            raise ValueError(error_msg)
        # get the columns of the new data
        new_data_cols = new_data.collect_schema().names()
        new_data_cols.remove(date_column_name)
        # check if the new data is lazyframe or not
        _, new_data = self._verify_compatible_frames(
            self.base_dataframe, [new_data], mode="match 1"
        )
        new_data = new_data[0]
        if (
            new_data.dtypes[new_data.collect_schema().names().index(date_column_name)]
            != pl.Datetime
        ):
            new_data = new_data.with_columns(pl.col(date_column_name).str.to_datetime())

        # Ensure both DataFrames have DateTime columns in the same unit
        new_data = new_data.with_columns(
            pl.col(date_column_name).dt.cast_time_unit("us")
        )
        self.base_dataframe = self.base_dataframe.with_columns(
            pl.col("Date").dt.cast_time_unit("us")
        )

        new_data = self.base_dataframe.join(
            new_data,
            left_on="Date",
            right_on=date_column_name,
            how="full",
            coalesce=True,
        )
        new_data = new_data.with_columns(
            pl.col(new_data_cols).interpolate_by("Date")
        ).select(pl.col(["Date"] + new_data_cols))

        self.base_dataframe = self.base_dataframe.join(
            new_data, on="Date", how="left", coalesce=True
        )

    def join(
        self,
        other: "Result",
        on: Union[str, List[str]],
        how: str = "inner",
        coalesce: bool = True,
    ) -> None:
        """Join two Result objects on a column. A wrapper around the polars join method.

        This will extend the data in the Result object horizontally. The column
        definitions of the two Result objects are combined, if there are any conflicts
        the column definitions of the calling Result object will take precedence.

        Args:
            other (Result): The other Result object to join with.
            on (Union[str, List[str]]): The column(s) to join on.
            how (str): The type of join to perform. Default is 'inner'.
            coalesce (bool): Whether to coalesce the columns. Default is True.
        """
        _, other_frame = self._verify_compatible_frames(
            self.base_dataframe, [other.base_dataframe], mode="match 1"
        )
        if isinstance(on, str):
            on = [on]
        self.base_dataframe = self.base_dataframe.join(
            other_frame[0], on=on, how=how, coalesce=coalesce
        )
        self.column_definitions = {
            **other.column_definitions,
            **self.column_definitions,
        }

    def extend(
        self, other: "Result" | List["Result"], concat_method: str = "diagonal"
    ) -> None:
        """Extend the data in this Result object with the data in another Result object.

        This method will concatenate the data in the two Result objects, with the Result
        object calling the method above the other Result object. The column definitions
        of the two Result objects are combined, if there are any conflicts the column
        definitions of the calling Result object will take precedence.

        Args:
            other (Result | List[Result]): The other Result object(s) to extend with.
            concat_method (str):
                The method to use for concatenation. Default is 'diagonal'. See the
                polars.concat method documentation for more information.
        """
        if not isinstance(other, list):
            other = [other]
        other_frame_list = [other_result.base_dataframe for other_result in other]
        self.base_dataframe, other_frame_list = self._verify_compatible_frames(
            self.base_dataframe, other_frame_list, mode="collect all"
        )
        self.base_dataframe = pl.concat(
            [self.base_dataframe] + other_frame_list, how=concat_method
        )
        original_column_definitions = self.column_definitions.copy()
        for other_result in other:
            self.column_definitions.update(other_result.column_definitions)
        self.column_definitions.update(original_column_definitions)

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
        info: Dict[str, Optional[str | int | float | Dict[Any, Any]]],
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
