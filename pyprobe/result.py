"""A module for the Result class."""

import logging
import re
from functools import wraps
from pprint import pprint
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from deprecated import deprecated
from numpy.typing import NDArray
from pydantic import BaseModel, Field, model_validator
from scipy.io import savemat

from pyprobe.plot import _retrieve_relevant_columns
from pyprobe.units import split_quantity_unit, unit_from_regexp

logger = logging.getLogger(__name__)

try:
    import hvplot.polars
except ImportError:
    hvplot = None


class PolarsColumnCache:
    """A class to cache columns from a Polars DataFrame.

    Args:
        base_dataframe (pl.LazyFrame | pl.DataFrame):
            The base dataframe to cache columns from.
    """

    def __init__(self, base_dataframe: pl.LazyFrame | pl.DataFrame) -> None:
        """Initialize the PolarsColumnCache object."""
        self.cache: Dict[str, pl.Series] = {}
        self._cached_dataframe = None
        self._base_dataframe = base_dataframe
        if isinstance(base_dataframe, pl.DataFrame):
            self.cached_dataframe = base_dataframe

    @property
    def base_dataframe(self) -> pl.LazyFrame | pl.DataFrame:
        """The base dataframe.

        Returns:
            pl.LazyFrame | pl.DataFrame: The base dataframe.
        """
        return self._base_dataframe

    @base_dataframe.setter
    def base_dataframe(self, value: pl.LazyFrame | pl.DataFrame) -> None:
        """Set the base dataframe."""
        self.clear_cache()
        self._base_dataframe = value

    @property
    def columns(self) -> List[str]:
        """The columns in the data.

        Returns:
            List[str]: The columns in the data.
        """
        return self.base_dataframe.collect_schema().names()

    @property
    def quantities(self) -> Set[str]:
        """The quantities of the data, with unit information removed.

        Returns:
            Set[str]: The quantities of the data.
        """
        return self.get_quantities(self.columns)

    def collect_columns(self, *columns: str) -> None:
        """Collect columns from the base dataframe and add to the cache.

        This method will check if the columns are in the cache. If they are not, it will
        check if they are in the base dataframe. If they are not, it will attempt to
        convert the column to the requested units and add to the lazyframe.

        Args:
            *columns (str): The columns to collect.

        Raises:
            ValueError:
                If the requested columns are not in the base dataframe and cannot
                be converted.
        """
        missing_from_cache = list(set(columns) - set(self.cache.keys()))
        if missing_from_cache:
            missing_from_data = list(set(missing_from_cache) - set(self.columns))
            if missing_from_data:
                # if missing from cache and data, may be a candidate for conversion
                requested_quantities = self.get_quantities(missing_from_data)
                missing_quantities = requested_quantities - self.quantities
                if missing_quantities:
                    # not a candidate for conversion as quantities are not in data
                    raise ValueError(f"Quantities {missing_quantities} not in data.")
                # convert the missing columns to the requested units and add to the
                # lazyframe
                for col in missing_from_data:
                    converter_object = unit_from_regexp(col)
                    instruction = converter_object.from_default_unit()
                    self.base_dataframe = self.base_dataframe.with_columns(instruction)
            # collect any missing columns and add to the cache
            if isinstance(self.base_dataframe, pl.LazyFrame):
                dataframe = self.base_dataframe.select(missing_from_cache).collect()
            else:
                dataframe = self.base_dataframe.select(missing_from_cache)
            for col in missing_from_cache:
                self.cache[col] = dataframe[col]

    def clear_cache(self) -> None:
        """Clear the cache."""
        self.cache = {}
        self._cached_dataframe = None

    @property
    def cached_dataframe(self) -> pl.DataFrame:
        """Return the cached dataframe as a Polars DataFrame."""
        if self._cached_dataframe is None:
            self._cached_dataframe = pl.DataFrame(self.cache)
        return pl.DataFrame(self.cache)

    @cached_dataframe.setter
    def cached_dataframe(self, value: pl.DataFrame) -> None:
        """Set the cached dataframe."""
        self.cache = value.to_dict()
        self._cached_dataframe = value

    @staticmethod
    def get_quantities(column_list: List[str]) -> Set[str]:
        """The quantities of the data, with unit information removed.

        Args:
            column_list (List[str]): The columns to get the quantities of.

        Returns:
            Set[str]: The quantities of the data.
        """
        _quantities: List[str] = []
        for _, column in enumerate(column_list):
            try:
                quantity, _ = split_quantity_unit(column)
                _quantities.append(quantity)
            except ValueError:
                continue
        return set(_quantities)


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
    info: dict[str, Optional[Any]]
    """Dictionary containing information about the cell."""
    column_definitions: Dict[str, str] = Field(default_factory=dict)
    """A dictionary containing the definitions of the columns in the data."""

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization method for the Pydantic model."""
        self._polars_cache = PolarsColumnCache(self.base_dataframe)

    @model_validator(mode="before")
    @classmethod
    def _load_base_dataframe(cls, data: Any) -> Any:
        """Load the base dataframe from a file if provided as a string."""
        if "base_dataframe" in data and isinstance(data["base_dataframe"], str):
            data["base_dataframe"] = pl.scan_parquet(data["base_dataframe"])
        return data

    @property
    def live_dataframe(self) -> pl.DataFrame:
        """Return the data as a polars DataFrame."""
        return self._polars_cache.base_dataframe

    @live_dataframe.setter
    def live_dataframe(self, value: pl.DataFrame) -> None:
        """Set the data as a polars DataFrame."""
        self._polars_cache.base_dataframe = value

    @property
    def data(self) -> pl.DataFrame:
        """Return the data as a polars DataFrame.

        Returns:
            pl.DataFrame: The data as a polars DataFrame.

        Raises:
            ValueError: If no data exists for this filter.
        """
        all_columns = self._polars_cache.columns
        self._polars_cache.collect_columns(*all_columns)
        complete_dataframe = self._polars_cache.cached_dataframe
        self._polars_cache.base_dataframe = complete_dataframe
        if complete_dataframe.is_empty():
            raise ValueError("No data exists for this filter.")
        return complete_dataframe

    @wraps(pd.DataFrame.plot)
    def plot(self, *args: Any, **kwargs: Any) -> None:
        """Wrapper for plotting using the pandas library."""
        data_to_plot = _retrieve_relevant_columns(self, args, kwargs)
        return data_to_plot.to_pandas().plot(*args, **kwargs)

    plot.__doc__ = (
        "This is a wrapper around the pandas plot method. It will perform"
        "exactly as you would expect the pandas plot method to perform"
        "when called on a DataFrame.\n\n" + (plot.__doc__ or "")
    )

    if hvplot is not None:

        @wraps(hvplot.hvPlot)
        def hvplot(self, *args: Any, **kwargs: Any) -> None:
            """Wrapper for plotting using the hvplot library."""
            data_to_plot = _retrieve_relevant_columns(self, args, kwargs)
            return data_to_plot.hvplot(*args, **kwargs)

    else:

        def hvplot(self, *args: Any, **kwargs: Any) -> None:
            """Wrapper for plotting using the hvplot library."""
            raise ImportError(
                "Optional dependency hvplot is not installed. Please install it via "
                "'pip install hvplot' or by installing PyProBE with hvplot as an "
                "optional dependency: pip install 'PyProBE-Data[hvplot]'."
            )

    hvplot.__doc__ = (
        "HvPlot is a library for creating fast and interactive plots.\n\n"
        "This method requires the hvplot library to be installed as an optional "
        "dependency. You can install it with PyProBE by running "
        ":code:`pip install 'PyProBE-Data[hvplot]'`, or install it seperately with "
        ":code:`pip install hvplot`.\n\n"
        "The default backend is bokeh, which can be changed by setting the backend "
        "with :code:`hvplot.extension('matplotlib')` or "
        ":code:`hvplot.extension('plotly')`.\n\n" + (hvplot.__doc__ or "")
    )

    def _get_data_subset(self, *column_names: str) -> pl.DataFrame:
        """Return a subset of the data with the specified columns.

        Args:
            *column_names: The columns to include in the new result object.

        Returns:
            A subset of the data with the specified columns.
        """
        self._polars_cache.collect_columns(*column_names)
        return self._polars_cache.cached_dataframe.select(column_names)

    def __getitem__(self, *column_names: str) -> "Result":
        """Return a new result object with the specified columns.

        Args:
            *column_names (str): The columns to include in the new result object.

        Returns:
            Result: A new result object with the specified columns.
        """
        return Result(
            base_dataframe=self._get_data_subset(*column_names), info=self.info
        )

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
        array = self._get_data_subset(*column_names).to_numpy()
        if len(column_names) == 0:
            error_msg = "At least one column name must be provided."
            logger.error(error_msg)
            raise ValueError(error_msg)
        elif len(column_names) == 1:
            return array.T[0]
        else:
            return tuple(array.T)

    @property
    def contains_lazyframe(self) -> bool:
        """Return whether the data is a LazyFrame.

        Returns:
            bool: True if the data is a LazyFrame, False otherwise.
        """
        return isinstance(self.live_dataframe, pl.LazyFrame)

    @deprecated(
        reason="The get_only method is deprecated. Use the get method instead.",
        version="1.2.0",
    )
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

    @property
    def quantities(self) -> Set[str]:
        """The quantities of the data, with unit information removed.

        Returns:
            List[str]: The quantities of the data.
        """
        return self._polars_cache.quantities

    @property
    def column_list(self) -> List[str]:
        """The columns in the data.

        Returns:
            List[str]: The columns in the data.
        """
        return self.live_dataframe.collect_schema().names()

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
            mode:
                The mode to use for verification. Either 'match 1' or 'collect all'.
                'match 1' will convert the frames to match the base frame. 'collect all'
                will collect all frames to DataFrames.

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
            self.live_dataframe, [new_data], mode="match 1"
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
        self.live_dataframe = self.live_dataframe.with_columns(
            pl.col("Date").dt.cast_time_unit("us")
        )

        all_data = self.live_dataframe.clone().join(
            new_data,
            left_on="Date",
            right_on=date_column_name,
            how="full",
            coalesce=True,
        )
        interpolated = all_data.with_columns(
            pl.col(new_data_cols).interpolate_by("Date")
        ).select(pl.col(["Date"] + new_data_cols))
        self.live_dataframe = self.live_dataframe.join(
            interpolated, on="Date", how="left", coalesce=True
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
            self.live_dataframe, [other.live_dataframe], mode="match 1"
        )
        if isinstance(on, str):
            on = [on]
        self.live_dataframe = self.live_dataframe.join(
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
        self._polars_cache.clear_cache()
        if not isinstance(other, list):
            other = [other]
        other_frame_list = [other_result.live_dataframe for other_result in other]
        self.live_dataframe, other_frame_list = self._verify_compatible_frames(
            self.live_dataframe, other_frame_list, mode="collect all"
        )
        self.live_dataframe = pl.concat(
            [self.live_dataframe] + other_frame_list, how=concat_method
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
        info: Dict[str, Optional[Any]],
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

    def export_to_mat(self, filename: str) -> None:
        """Export the data to a .mat file.

        This method will export the data and info dictionary to a .mat file. The
        variables in the .mat file will be named 'data' and 'info'. Column names and
        dictionary keys will have any non-alphanumeric characters replaced with an
        underscore, to comply with MATLAB variable naming rules.

        Args:
            filename: The name of the file to export to.
        """
        # Replace any non-alphanumeric character with an underscore in the DataFrame columns
        renamed_data = self.data.rename(
            {col: re.sub(r"\W", "_", col) for col in self.data.columns}
        )

        # Replace any non-alphanumeric character with an underscore in the info dictionary keys
        renamed_info = {
            re.sub(r"\W", "_", key): value for key, value in self.info.items()
        }

        variable_dict = {
            "data": renamed_data.to_dict(),
            "info": renamed_info,
        }
        savemat(filename, variable_dict, oned_as="column")


def combine_results(
    results: List[Result],
    concat_method: str = "diagonal",
) -> Result:
    """Combine multiple Result objects into a single Result object.

    This method should be used to combine multiple Result objects that have different
    entries in their info dictionaries. The info dictionaries of the Result objects will
    be integrated into the dataframe of the new Result object

    Args:
        results (List[Result]): The Result objects to combine.
        concat_method (str):
            The method to use for concatenation. Default is 'diagonal'. See the
            polars.concat method documentation for more information.

    Returns:
        Result: A new result object with the combined data.
    """
    for result in results:
        instructions = [
            pl.lit(result.info[key]).alias(key) for key in result.info.keys()
        ]
        result.live_dataframe = result.live_dataframe.with_columns(instructions)
    results[0].extend(results[1:], concat_method=concat_method)
    return results[0]
