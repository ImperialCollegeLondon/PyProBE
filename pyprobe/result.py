"""A module for the Result class."""

import os
import re
import warnings
from collections.abc import Callable
from functools import wraps
from pprint import pprint
from typing import Any, Literal, Union

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.io import savemat

from pyprobe.columns import Column, ColumnSet
from pyprobe.utils import catch_pydantic_validation, deprecated, validate_timezone

try:
    import hvplot.polars  # noqa: F401

    hvplot_exists = True
except ImportError:
    hvplot_exists = False


class Result:
    """A class for holding any data in PyProBE.

    A Result object is the base type for every data object in PyProBE. This class
    includes all of the main methods for returning and describing any data in PyProBE.

    Key attributes for returning data:
        - :attr:`data`: The data as a Polars DataFrame.
        - :meth:`get`: Get a column from the data as a NumPy array.

    Key attributes for describing the data:
        - :attr:`metadata`: A dictionary containing metadata about the cell and
          data source.
        - :attr:`column_definitions`: A dictionary of column definitions.
        - :meth:`print_definitions`: Print the column definitions.
        - :attr:`columns`: A :class:`~pyprobe.columns.ColumnSet` object providing
          column name access (via ``.names``) and BDF-aware resolution (via
          ``.resolve()`` and ``.can_resolve()``).
    """

    def __init__(
        self,
        lf: pl.LazyFrame | pl.DataFrame | str,
        metadata: dict[str, Any | None] = {},
        column_definitions: dict[str, str] | None = None,
    ) -> None:
        """Create a Result with explicit constructor validation.

        Args:
            lf: A LazyFrame, DataFrame, or a path to a parquet file.
            metadata: Dictionary containing metadata about the result.
            column_definitions: Optional definitions for data columns.

        Raises:
            ValueError: If constructor inputs do not match expected types.
        """
        if isinstance(lf, str):
            lf = pl.scan_parquet(lf)
        if not isinstance(lf, pl.LazyFrame):
            if isinstance(lf, pl.DataFrame):
                lf = lf.lazy()
            elif isinstance(lf, str):
                lf = pl.scan_parquet(lf)
            else:
                raise ValueError(
                    "lf must be a polars DataFrame, LazyFrame, or a parquet file path."
                )
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary.")
        if column_definitions is None:
            column_definitions = {}
        elif not isinstance(column_definitions, dict):
            raise ValueError("column_definitions must be a dictionary.")

        self.lf: pl.LazyFrame = lf
        self.metadata = metadata
        self.column_definitions = column_definitions.copy()

    def collect(self) -> pl.DataFrame:
        """Collect the lazy dataframe into a polars DataFrame.

        Use this method to resolve the lazy computations in the Result object. This can
        improve performance if you are reading a large amount of data from disk, and
        will be performing multiple calls to access the data.

        Returns:
            pl.DataFrame: The collected dataframe.
        """
        lf = self.lf.collect()
        self.lf = lf.lazy()
        return lf

    @property
    def columns(self) -> ColumnSet:
        """The columns in the data as a ColumnSet.

        Returns a :class:`~pyprobe.columns.ColumnSet` object that provides
        both simple column name access and BDF-aware resolution:

        - :attr:`~pyprobe.columns.ColumnSet.names`: tuple of column name strings.
        - :attr:`~pyprobe.columns.ColumnSet.quantities`: tuple of quantity strings.
        - :meth:`~pyprobe.columns.ColumnSet.resolve`: resolve a column by name
          or quantity, with optional unit conversion.
        - :meth:`~pyprobe.columns.ColumnSet.can_resolve`: check if a column
          or BDF quantity is available.

        Returns:
            ColumnSet: A column introspection and resolution object.

        Examples:
            >>> import polars as pl
            >>> from pyprobe.result import Result
            >>> r = Result(lf=pl.LazyFrame({"Current / A": [1.0]}))
            >>> r.columns.names
            ('Current / A',)
            >>> r.columns.quantities
            ('Current',)
        """
        return ColumnSet(self.lf.collect_schema().names())

    @property
    def info(self) -> dict[str, Any | None]:
        """Backward compatibility alias for metadata.

        Returns:
            dict: The metadata dictionary.
        """
        return self.metadata

    @property
    def df(self) -> pl.DataFrame:
        """Return the data as a Polars DataFrame.

        Returns:
            pl.DataFrame: The data as a Polars DataFrame.
        """
        return self.collect()

    @df.setter
    def df(self, dataframe: pl.DataFrame) -> None:
        """Set the data as a Polars DataFrame.

        Args:
            dataframe (pl.DataFrame): The data as a Polars DataFrame.
        """
        self.lf = dataframe.lazy()

    @property
    def data(self) -> pl.DataFrame:
        """Return the data as a polars DataFrame.

        Returns:
            pl.DataFrame: The data as a polars DataFrame.

        Raises:
            ValueError: If no data exists for this filter.
        """
        df = self.collect()
        if df.is_empty():
            raise ValueError("No data exists for this filter.")
        return df

    @wraps(pd.DataFrame.plot)
    def plot(self, *args: Any, **kwargs: Any) -> Axes | NDArray[Axes]:
        """Wrapper for plotting using the pandas library."""
        data_to_plot = self.get_plotting_data(args, kwargs)
        return data_to_plot.to_pandas().plot(*args, **kwargs)

    plot.__doc__ = """Plot the data using the pandas plot method.

    Call this method on a Result object in the same way you would call the pandas plot
    method on a DataFrame. For example:

    .. code-block:: python

        result.plot(x="Time [s]", y="Current [A]")

    Refer to the `pandas documentation \
    <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html>`_
    for detailed information and examples.
    """

    if hvplot_exists is True:

        @wraps(hvplot.hvPlot)
        def hvplot(self, *args: Any, **kwargs: Any) -> Any:
            """Wrapper for plotting using the hvplot library."""
            data_to_plot = self.get_plotting_data(args, kwargs)
            return data_to_plot.hvplot(*args, **kwargs)

    else:

        def hvplot(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore
            """Wrapper for plotting using the hvplot library."""
            raise ImportError(
                "Optional dependency hvplot is not installed. Please install it via "
                "'pip install hvplot' or by installing PyProBE with hvplot as an "
                "optional dependency: pip install 'PyProBE-Data[hvplot]'.",
            )

    hvplot.__doc__ = """HvPlot is a library for creating fast and interactive plots.
        This method requires the hvplot library to be installed as an optional
        dependency. You can install it with PyProBE by running
        :code:`pip install 'PyProBE-Data[hvplot]'`, or install it seperately with
        :code:`pip install hvplot`.

        The default backend is bokeh, which can be changed by setting the backend
        with :code:`hvplot.extension('matplotlib')` or
        :code:`hvplot.extension('plotly')`.

        Example usage:

        .. code-block:: python

            result.hvplot(x="Time [s]", y="Current [A]", kind="scatter")

        This method is not compatible with the inline syntax for hvplot:
        :code:`result.hvplot.scatter(...)`.

        See the `hvplot documentation
        <https://hvplot.holoviz.org/user_guide/Plotting.html>`_ for information
        and examples.
        """

    def __getitem__(self, *column_names: str | Column) -> "Result":
        """Return a new result object with the specified columns.

        Args:
            *column_names (str | Column):
                The columns to include in the new result object.

        Returns:
            Result: A new result object with the specified columns.
        """
        col_set = self.columns
        exprs = [col_set.resolve(name) for name in column_names]
        return Result(
            lf=self.lf.select(*exprs),
            metadata=self.metadata,
        )

    def get(
        self,
        *column_names: str | Column,
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], ...]:
        """Return one or more columns of the data as separate 1D numpy arrays.

        Args:
            column_names (str | Column): The column name(s) to return.

        Returns:
            Union[NDArray[np.float64], Tuple[NDArray[np.float64], ...]]:
                The column(s) as numpy array(s).

        Raises:
            ValueError: If no column names are provided.
            ValueError: If a column name is not in the data.
        """
        if len(column_names) == 0:
            error_msg = "At least one column name must be provided."
            logger.error(error_msg)
            raise ValueError(error_msg)
        col_set = self.columns
        exprs = [col_set.resolve(name) for name in column_names]
        array = self.lf.select(*exprs).collect().to_numpy()
        if len(column_names) == 1:
            return array.T[0]
        else:
            return tuple(array.T)

    @deprecated(
        reason="The get_only method is deprecated. Use the get method instead.",
        version="1.2.0",
    )
    def get_only(self, column_name: str | Column) -> NDArray[np.float64]:
        """Return a single column of the data as a numpy array.

        Args:
            column_name (str | Column): The column name to return.

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

    def get_plotting_data(
        self,
        args: tuple[Any, ...],
        kwargs: dict[Any, Any],
    ) -> pl.DataFrame:
        """Extract and resolve columns for plotting from function arguments.

        This method analyzes the arguments passed to a plotting function and
        retrieves the used columns as a DataFrame. It extracts column names from
        positional and keyword arguments, resolves them using the ColumnSet
        (which handles unit conversions and BDF-aware resolution), and returns
        a collected DataFrame suitable for passing to plotting libraries.

        Args:
            args: Positional arguments from the plotting function.
            kwargs: Keyword arguments from the plotting function.

        Returns:
            pl.DataFrame: A collected DataFrame containing the requested columns.

        Raises:
            ValueError: If none of the requested columns are present in the data.

        Examples:
            >>> result = Result(lf=pl.LazyFrame({"Current / A": [1.0, 2.0]}))
            >>> df = result.get_plotting_data(["Current / mA"], {})
            >>> df.shape
            (2, 1)
        """
        kwargs_values = [
            v
            for k, v in kwargs.items()
            if isinstance(v, (str, Column)) and k != "label"
        ]
        args_values = [v for v in args if isinstance(v, (str, Column))]
        all_args = set(kwargs_values + args_values)
        relevant_columns = []
        col_set = self.columns

        for arg in all_args:
            if col_set.can_resolve(arg):
                relevant_columns.append(arg)

        if len(relevant_columns) == 0:
            raise ValueError(
                f"None of the columns in {all_args} are present in the Result object.",
            )

        # Resolve columns using ColumnSet to handle unit conversions
        exprs = [col_set.resolve(col) for col in relevant_columns]
        return self.lf.select(*exprs).collect()

    def define_column(self, column_name: str, definition: str) -> None:
        """Define a new column when it is added to the dataframe.

        Args:
            column_name (str): The name of the column.
            definition (str): The definition of the quantity stored in the column
        """
        self.column_definitions[column_name] = definition

    def print_definitions(self) -> None:
        """Print the definitions of the columns stored in this result object."""
        pprint(self.column_definitions)  # noqa: T203

    def clean_copy(
        self,
        dataframe: pl.DataFrame | pl.LazyFrame | None = None,
        column_definitions: dict[str, str] | None = None,
    ) -> "Result":
        """Create a copy of the result object with info dictionary but without data.

        Args:
            dataframe (Optional[Union[pl.DataFrame, pl.LazyFrame]):
                The data to include in the new Result object.
            column_definitions (Optional[dict[str, str]]):
                The definitions of the columns in the new result object.

        Returns:
            Result: A new result object with the specified data.
        """
        if dataframe is None:
            dataframe = pl.LazyFrame({})
        elif isinstance(dataframe, pl.DataFrame):
            dataframe = dataframe.lazy()
        if column_definitions is None:
            column_definitions = {}
        return Result(
            lf=dataframe,
            metadata=self.metadata,
            column_definitions=column_definitions,
        )

    @staticmethod
    def _verify_compatible_frames(
        base_frame: pl.DataFrame | pl.LazyFrame,
        frames: list[pl.DataFrame | pl.LazyFrame],
        mode: Literal["match 1", "collect all"] = "collect all",
    ) -> tuple[pl.DataFrame | pl.LazyFrame, list[pl.DataFrame | pl.LazyFrame]]:
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
                frame,
                pl.LazyFrame,
            ):
                frame = frame.collect()
            verified_frames.append(frame)

        return base_frame, verified_frames

    def load_external_file(self, filepath: str) -> pl.LazyFrame:
        """Load an external file into a LazyFrame.

        Supported file types are CSV, Parquet, and Excel. For maximum performance,
        consider using Parquet files. If you have an Excel file, consider converting
        it to CSV before loading.

        Args:
            filepath (str): The path to the external file.
        """
        file = os.path.basename(filepath)
        file_ext = os.path.splitext(file)[1]
        match file_ext:
            case ".csv":
                return pl.scan_csv(filepath)
            case ".parquet":
                return pl.scan_parquet(filepath)
            case ".xlsx":
                warnings.warn("Excel reading is slow. Consider converting to CSV.")
                return pl.read_excel(filepath).lazy()
            case _:
                error_msg = f"Unsupported file type: {file_ext}"
                logger.error(error_msg)
                raise ValueError(error_msg)

    def add_data(
        self,
        new_data: pl.DataFrame | pl.LazyFrame | str,
        time_column_name: str,
        column_map: dict[str, str] | None = None,
        datetime_format: str | None = None,
        timezone: str = "UTC",
        align_on: tuple[str, str] | None = None,
        join_strategy: Literal[
            "keep_existing", "keep_new", "keep_both"
        ] = "keep_existing",
        fill_strategy: Literal["interpolate", "forward_fill", "backward_fill"]
        | None = "interpolate",
    ) -> None:
        """Add new data columns to the result object using Unix Time as the join key.

        The data must be time series data with a time column. The new data is joined to
        the base dataframe on the "Unix Time / s" column. Choose which dates to keep
        with the join strategy, and how to fill missing values with the fill strategy.

        Args:
            new_data:
                The new data to add to the result object. Can be a DataFrame, LazyFrame,
                or a path to a file (CSV, Parquet, Excel).
            time_column_name:
                The name of the column in the new data containing the time. Can be a
                datetime column (which will be auto-converted to UTC unix seconds), a
                numeric column (assumed to be UTC unix seconds), or a string column
                (which will be parsed then converted).
            column_map:
                Mapping from output names to source column names:
                {output_name: source_name}.
                Only the columns in this dict will be imported. If None, all columns
                (except time_column_name) will be imported. Output names do not need to
                follow "Quantity / unit" format.
            datetime_format:
                The format string for parsing the time column if it is a string.
                Defaults to None (auto-detect).
            timezone:
                The timezone of the new data's time column, as an IANA string
                (e.g. ``"UTC"``, ``"Europe/Berlin"``).  Applied only to tz-naive
                datetime columns; tz-aware columns are converted to UTC directly.
                Defaults to ``"UTC"``.
            align_on:
                A tuple of column names to use for aligning the new data with the
                existing data. The first element is the column name in the existing
                data, and the second element is the column name in the new data.
                The new data will be shifted in time to maximize the cross-correlation
                between the two columns. Defaults to None.
            join_strategy:
                The strategy for which times to keep in the result:
                - "keep_existing": Keep only times from existing data
                - "keep_new": Keep only times from new data
                - "keep_both": Keep all times from both datasets
                Defaults to "keep_existing".
            fill_strategy:
                The strategy for filling missing values in the merged dataset columns
                after applying the join strategy (this may affect both existing and
                new columns):
                - "interpolate": Interpolate missing values by unix time
                - "forward_fill": Forward fill missing values
                - "backward_fill": Backward fill missing values
                - None: Don't fill missing values
                Defaults to "interpolate".

        Raises:
            ValueError: If the base dataframe has no "Unix Time / s" column.
            ValueError: If an invalid timezone string is provided.
        """
        # Load external file if needed
        if isinstance(new_data, str):
            new_data = self.load_external_file(new_data)

        # Apply column_map (select and rename columns)
        if column_map is not None:
            cols_to_select = [time_column_name] + list(column_map.values())
            new_data = new_data.select(cols_to_select)
            rename_map = {src: dest for dest, src in column_map.items()}
            new_data = new_data.rename(rename_map)

        # Validate base dataframe has Unix Time column
        if "Unix Time / s" not in self.lf.collect_schema().names():
            error_msg = "No 'Unix Time / s' column in the base dataframe."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Convert new_data to match the type of lf
        _, new_data = self._verify_compatible_frames(
            self.lf,
            [new_data],
            mode="match 1",
        )
        new_data = new_data[0]

        # Convert time column to "Unix Time / s" Float64
        schema = new_data.collect_schema()
        time_dtype = schema[time_column_name]

        # Handle String dtype: parse to datetime first
        if isinstance(time_dtype, pl.String):
            new_data = new_data.with_columns(
                pl.col(time_column_name).str.to_datetime(format=datetime_format)
            )
            time_dtype = pl.Datetime(time_unit="us")  # Update dtype after conversion

        # Handle Datetime dtype: convert to UTC unix seconds
        if isinstance(time_dtype, pl.Datetime):
            col_tz = time_dtype.time_zone
            if col_tz is None:
                # Tz-naive: interpret as the specified timezone (default "UTC")
                validate_timezone(timezone)
                col = pl.col(time_column_name).dt.replace_time_zone(timezone)
            else:
                # Tz-aware: convert to UTC directly
                col = pl.col(time_column_name).dt.convert_time_zone("UTC")

            new_data = new_data.with_columns(
                col.dt.epoch(time_unit="s").cast(pl.Float64).alias(time_column_name)
            )
        # Handle numeric dtype: cast to Float64 (assumed UTC unix seconds)
        elif isinstance(time_dtype, (pl.Float32, pl.Float64, pl.Int32, pl.Int64)):
            new_data = new_data.with_columns(pl.col(time_column_name).cast(pl.Float64))
        else:
            error_msg = (
                f"Unsupported dtype for time column: {time_dtype}. "
                "Must be String, Datetime, or numeric."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Rename time column to "Unix Time / s"
        new_data = new_data.rename({time_column_name: "Unix Time / s"})
        if isinstance(new_data, pl.DataFrame):
            new_data = new_data.lazy()
        new_result = Result(lf=new_data, metadata={})

        # Collect new data column names (excluding unix time)
        new_data_cols = [
            col for col in new_data.collect_schema().names() if col != "Unix Time / s"
        ]

        # Optionally align the new data with existing data
        if align_on is not None:
            from pyprobe.analysis.time_series import align_data

            col_existing, col_new = align_on
            _, new_result = align_data(self, new_result, col_existing, col_new)

        new_data = new_result.lf

        # Join all data to prepare for filling
        all_data = (
            self.lf.clone()
            .join(
                new_data,
                on="Unix Time / s",
                how="full",
                coalesce=True,
            )
            .sort("Unix Time / s")
        )

        # Get all non-Unix Time columns for filling
        all_cols_except_time = [
            col for col in all_data.collect_schema().names() if col != "Unix Time / s"
        ]
        # Restrict interpolation to numeric columns only, since interpolate_by
        # is not supported for non-numeric dtypes.
        schema = all_data.collect_schema()
        numeric_cols_except_time = [
            name
            for name, dtype in zip(schema.names(), schema.dtypes())
            if name != "Unix Time / s" and dtype in pl.NUMERIC_DTYPES
        ]

        # Apply fill strategy to all columns (both existing and new)
        valid_fill_strategies = {None, "interpolate", "forward_fill", "backward_fill"}
        if fill_strategy not in valid_fill_strategies:
            raise ValueError(
                f"Unsupported fill_strategy: {fill_strategy!r}. "
                "Valid options are None, 'interpolate', 'forward_fill', "
                "'backward_fill'."
            )
        if fill_strategy == "interpolate":
            if numeric_cols_except_time:
                filled = all_data.with_columns(
                    pl.col(numeric_cols_except_time).interpolate_by("Unix Time / s"),
                )
            else:
                # No numeric columns to interpolate; leave data unchanged.
                filled = all_data
        elif fill_strategy == "forward_fill":
            filled = all_data.with_columns(
                pl.col(all_cols_except_time).forward_fill(),
            )
        elif fill_strategy == "backward_fill":
            filled = all_data.with_columns(
                pl.col(all_cols_except_time).backward_fill(),
            )
        else:  # fill_strategy is None
            filled = all_data

        # Apply join strategy
        if join_strategy == "keep_existing":
            # Keep only existing times
            filled_new_cols = filled.select(pl.col(["Unix Time / s"] + new_data_cols))
            self.lf = self.lf.join(
                filled_new_cols,
                on="Unix Time / s",
                how="left",
                coalesce=True,
            )
        elif join_strategy == "keep_new":
            # Keep only new times
            # Filter filled to only times that exist in new_data
            self.lf = filled.join(
                new_data.select(["Unix Time / s"]),
                on="Unix Time / s",
                how="inner",
            )
        elif join_strategy == "keep_both":
            # Keep all times from both datasets
            self.lf = filled
        else:
            raise ValueError(
                f"Unsupported join_strategy: {join_strategy!r}. "
                "Expected one of: 'keep_existing', 'keep_new', 'keep_both'."
            )

    @deprecated(
        reason="Use add_data instead.",
        version="2.3.1",
    )
    def add_new_data_columns(
        self,
        new_data: pl.DataFrame | pl.LazyFrame,
        date_column_name: str,
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
        raise NotImplementedError("This method is deprecated. Use add_data instead.")

    def join(
        self,
        other: "Result",
        on: str | list[str],
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
            self.lf,
            [other.lf],
            mode="match 1",
        )
        if isinstance(on, str):
            on = [on]
        self.lf = self.lf.join(
            other_frame[0],
            on=on,
            how=how,
            coalesce=coalesce,
        )
        self.column_definitions = {
            **other.column_definitions,
            **self.column_definitions,
        }

    def extend(
        self,
        other: Union["Result", list["Result"]],  # noqa: UP007
        concat_method: str = "diagonal",
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
        other_frame_list = [other_result.lf for other_result in other]
        self.lf, other_frame_list = self._verify_compatible_frames(
            self.lf,
            other_frame_list,
            mode="collect all",
        )
        self.lf = pl.concat(
            [self.lf] + other_frame_list,
            how=concat_method,
        )
        original_column_definitions = self.column_definitions.copy()
        for other_result in other:
            self.column_definitions.update(other_result.column_definitions)
        self.column_definitions.update(original_column_definitions)

    @classmethod
    def build(
        cls,
        data_list: list[
            pl.LazyFrame
            | pl.DataFrame
            | dict[str, NDArray[np.float64] | list[float]]
            | list[
                pl.LazyFrame
                | pl.DataFrame
                | dict[str, NDArray[np.float64] | list[float]]
            ]
        ],
        info: dict[str, Any | None],
    ) -> "Result":
        """Build a Result object from a list of dataframes.

        Args:
            data_list (List[List[pl.LazyFrame | pl.DataFrame | dict]]):
                The data to include in the new result object.
                The first index indicates the cycle and the second index indicates the
                step.
            info (dict[str, Optional[str | int | float]]): A dict containing test info.

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
                    pl.lit(cycle).alias("Cycle"),
                    pl.lit(step).alias("Step"),
                )
                data.append(step_data)
        data = pl.concat(data)
        if isinstance(data, pl.DataFrame):
            data = data.lazy()
        return cls(lf=data, metadata=info)

    def export_to_mat(self, filename: str) -> None:
        """Export the data to a .mat file.

        This method will export the data and metadata dictionary to a .mat file. The
        variables in the .mat file will be named 'data' and 'metadata'. Column names and
        dictionary keys will have any non-alphanumeric characters replaced with an
        underscore, to comply with MATLAB variable naming rules.

        Args:
            filename: The name of the file to export to.
        """
        # Replace any non-alphanumeric character with an underscore in the DataFrame
        # columns
        renamed_data = self.data.rename(
            {col: re.sub(r"\W", "_", col) for col in self.data.columns},
        )

        # Replace any non-alphanumeric character with an underscore in the metadata
        # dictionary keys
        renamed_metadata = {
            re.sub(r"\W", "_", key): value for key, value in self.metadata.items()
        }

        variable_dict = {
            "data": renamed_data.to_dict(),
            "metadata": renamed_metadata,
        }
        savemat(filename, variable_dict, oned_as="column")

    @catch_pydantic_validation
    @staticmethod
    def from_polars_io(
        polars_io_func: Callable[..., pl.DataFrame | pl.LazyFrame],
        metadata: dict[str, Any | None] = {},
        column_definitions: dict[str, str] = {},
        **kwargs: Any,
    ) -> "Result":
        """Create a new Result object with data from a Polars IO function.

        Refer to the Polars documentation for a list of available IO functions:

        - `External file import functions \
            <https://docs.pola.rs/api/python/stable/reference/io.html>`_
        - `Python object conversion functions \
            <https://docs.pola.rs/api/python/stable/reference/functions.html>`_

        Args:
            polars_io_func (Callable[..., pl.DataFrame | pl.LazyFrame]):
                The Polars IO function to use to create the data.
            metadata (dict[str, Any | None]):
                The metadata dictionary for the new Result object. Empty by default.
            column_definitions (dict[str, str]):
                The column definitions for the new Result object. Empty by default.
            **kwargs: The keyword arguments to pass to the Polars IO function.

        Returns:
            Result: A new Result object with the specified data and info.

        Example:
            From a saved .csv file:

            .. code-block:: python

            result = Result.from_polars_io(
                pl.scan_csv,
                metadata={"test": "test"},
                column_definitions={},
                source="data.csv",
            )

            From a pandas DataFrame:

            .. code-block:: python

            result = Result.from_polars_io(
                pl.from_pandas,
                metadata={"test": "test"},
                column_definitions={},
                data=pd.DataFrame({"a": [1, 2, 3]}),
            )

            From a numpy array:

            .. code-block:: python

            result = Result.from_polars_io(
                pl.from_numpy,
                metadata={"test": "test"},
                column_definitions={},
                data=np.array([[1, 2, 3], [4, 5, 6]]),
                schema=["a", "b"]
            )

        """
        lf = polars_io_func(**kwargs)
        if isinstance(lf, pl.DataFrame):
            lf = lf.lazy()
        return Result(lf=lf, metadata=metadata, column_definitions=column_definitions)

    @property
    @deprecated(
        reason=(
            "The live_dataframe property is deprecated. Use the lf property instead."
        ),
        version="2.4.0",
    )
    def live_dataframe(self) -> pl.LazyFrame:
        """The base dataframe as a LazyFrame.

        Returns:
            pl.LazyFrame: The base dataframe as a LazyFrame.
        """
        return self.lf

    @live_dataframe.setter
    @deprecated(
        reason=(
            "The live_dataframe property is deprecated. Use the lf property instead."
        ),
        version="2.4.0",
    )
    def live_dataframe(self, value: pl.LazyFrame) -> None:
        self.lf = value

    @property
    @deprecated(
        reason=(
            "The base_dataframe property is deprecated. Use the lf property instead."
        ),
        version="2.4.0",
    )
    def base_dataframe(self) -> pl.LazyFrame:
        """The base dataframe as a LazyFrame.

        Returns:
            pl.LazyFrame: The base dataframe as a LazyFrame.
        """
        return self.lf

    @base_dataframe.setter
    @deprecated(
        reason=(
            "The base_dataframe property is deprecated. Use the lf property instead."
        ),
        version="2.4.0",
    )
    def base_dataframe(self, value: pl.LazyFrame) -> None:
        self.lf = value


def combine_results(
    results: list[Result],
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
            pl.lit(result.metadata[key]).alias(key) for key in result.metadata
        ]
        result.lf = result.lf.with_columns(instructions)
    results[0].extend(results[1:], concat_method=concat_method)
    return results[0]
