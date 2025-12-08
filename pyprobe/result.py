"""A module for the Result class."""

import os
import re
import warnings
from collections.abc import Callable
from functools import wraps
from pprint import pprint
from typing import Any, Literal, Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator, model_validator
from scipy.io import savemat
from tzlocal import get_localzone

from pyprobe.plot import _retrieve_relevant_columns
from pyprobe.units import get_unit_scaling, split_quantity_unit
from pyprobe.utils import catch_pydantic_validation, deprecated

try:
    import hvplot.polars  # noqa: F401

    hvplot_exists = True
except ImportError:
    hvplot_exists = False


def _validate_timezone(timezone: str) -> str:
    """Validate that a timezone string is a valid IANA timezone.

    Args:
        timezone: The timezone string to validate.

    Returns:
        The validated timezone string.

    Raises:
        ValueError: If the timezone string is not valid.
    """
    try:
        ZoneInfo(timezone)
        return timezone
    except ZoneInfoNotFoundError as e:
        error_msg = f"Invalid timezone: '{timezone}'. Must be a valid IANA timezone."
        logger.error(error_msg)
        raise ValueError(error_msg) from e


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
        - :attr:`columns`: A list of column names.
    """

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    lf: pl.LazyFrame
    info: dict[str, Any | None]
    """Dictionary containing information about the cell."""
    column_definitions: dict[str, str] = Field(default_factory=dict)
    """A dictionary containing the definitions of the columns in the data."""

    @model_validator(mode="before")
    @classmethod
    def _load_base_dataframe(cls, data: Any) -> Any:
        """Load the base dataframe from a file if provided as a string."""
        if "base_dataframe" in data:
            data["lf"] = data.pop("base_dataframe")
            warning_msg = "'base_dataframe' is deprecated. Please use 'lf' instead."
            logger.warning(
                warning_msg,
            )
            warnings.warn(
                warning_msg,
                DeprecationWarning,
            )
        return data

    @field_validator("lf", mode="before")
    @classmethod
    def _validate_lf(cls, data: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
        """Validate that the base dataframe is a LazyFrame."""
        if isinstance(data, pl.DataFrame):
            data = data.lazy()
        return data

    @model_validator(mode="before")
    @classmethod
    def _load_lf(cls, data: Any) -> Any:
        """Load the base dataframe from a file if provided as a string."""
        if "lf" in data and isinstance(data["lf"], str):
            data["lf"] = pl.scan_parquet(data["lf"])
        return data

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
    def columns(self) -> list[str]:
        """The columns in the data.

        Returns:
            List[str]: The columns in the data.
        """
        return self.lf.collect_schema().names()

    @staticmethod
    def _get_quantities(columns: list[str]) -> list[str]:
        """The quantities of the data, with unit information removed.

        Args:
            columns (List[str]): The columns to get the quantities of.

        Returns:
            List[str]: The quantities of the data.
        """
        _quantities: set[str] = set()
        for _, column in enumerate(columns):
            try:
                quantity, _ = split_quantity_unit(column)
                _quantities.add(quantity)
            except ValueError:
                continue
        return list(_quantities)

    @property
    def quantities(self) -> list[str]:
        """The quantities of the data, with unit information removed.

        Returns:
            List[str]: The quantities of the data.
        """
        return self._get_quantities(self.columns)

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

    def check_columns(self, columns: list[str]) -> None:
        """Check whether a column exists in the data.

        Convert units if selected quantity exists in data with different unit.

        Args:
            columns (List[str]): The columns to check.

        Raises:
            ValueError: If a column does not exist in the data.
        """
        missing_columns = set(columns) - set(self.columns)
        if missing_columns:
            logger.info("Missing columns: {}", missing_columns)
            # check if missing columns can be converted from existing quantities
            quantities = set(self._get_quantities(list(missing_columns)))
            missing_quantities = set(quantities) - set(self.quantities)
            if missing_quantities:
                raise ValueError(f"Quantities {missing_quantities} not in data.")
            # convert missing columns to requested units
            for col in missing_columns:
                quantity, unit = split_quantity_unit(col)
                if unit == "":
                    continue
                _, base_unit = get_unit_scaling(unit)
                self.lf = self.lf.with_columns(
                    (pl.col(f"{quantity} [{base_unit}]").units.to_unit(unit)),
                )
                logger.info(f"Converted column {col} from {base_unit} to {unit}.")

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
        data_to_plot = _retrieve_relevant_columns(self, args, kwargs)
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
            data_to_plot = _retrieve_relevant_columns(self, args, kwargs)
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

    def __getitem__(self, *column_names: str) -> "Result":
        """Return a new result object with the specified columns.

        Args:
            *column_names (str): The columns to include in the new result object.

        Returns:
            Result: A new result object with the specified columns.
        """
        self.check_columns(list(column_names))
        return Result(
            lf=self.lf.select(*column_names),
            info=self.info,
        )

    def get(
        self,
        *column_names: str,
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], ...]:
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
        if len(column_names) == 0:
            error_msg = "At least one column name must be provided."
            logger.error(error_msg)
            raise ValueError(error_msg)
        self.check_columns(list(column_names))
        array = self.lf.select(*column_names).collect().to_numpy()
        if len(column_names) == 1:
            return array.T[0]
        else:
            return tuple(array.T)

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
            info=self.info,
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
        date_column_name: str,
        datetime_format: str | None = None,
        importing_columns: list[str] | dict[str, str] | None = None,
        existing_data_timezone: str | None = None,
        new_data_timezone: str | None = None,
        align_on: tuple[str, str] | None = None,
    ) -> None:
        """Add new data columns to the result object.

        The data must be time series data with a date column. The new data is joined to
        the base dataframe on the date column, and the new data columns are interpolated
        to fill in missing values.

        Args:
            new_data (pl.DataFrame | pl.LazyFrame | str):
                The new data to add to the result object. Can be a DataFrame, LazyFrame,
                or a path to a file (CSV, Parquet, Excel).
            date_column_name (str):
                The name of the column in the new data containing the date.
            datetime_format (Optional[str]):
                The format string for parsing the date column if it is a string.
                Defaults to None.
            importing_columns (Optional[List[str] | dict[str, str]]):
                The columns to import from the external file. If a list, the columns
                will be imported as is. If a dict, the keys are the columns in the data
                you want to import and the values are the columns you want to rename
                them to. If None, all columns will be imported. Defaults to None.
            existing_data_timezone (Optional[str]):
                The timezone of the existing data. If None, the timezone is inferred
                from the local machine. Defaults to None.
            new_data_timezone (Optional[str]):
                The timezone of the new data. If None, and the new data is naive, it is
                assumed to be in the same timezone as the existing data. Defaults to
                None.
            align_on (Optional[Tuple[str, str]]):
                A tuple of column names to use for aligning the new data with the
                existing data. The first element is the column name in the existing
                data, and the second element is the column name in the new data.
                The new data will be shifted in time to maximize the cross-correlation
                between the two columns. Defaults to None.

        Raises:
            ValueError: If the base dataframe has no date column.
            ValueError: If an invalid timezone string is provided.
        """
        # Validate timezone inputs
        if existing_data_timezone is not None:
            _validate_timezone(existing_data_timezone)
        if new_data_timezone is not None:
            _validate_timezone(new_data_timezone)

        if isinstance(new_data, str):
            new_data = self.load_external_file(new_data)

        if isinstance(importing_columns, dict):
            new_data = new_data.select(
                [date_column_name] + list(importing_columns.keys()),
            )
            new_data = new_data.rename(importing_columns)
        elif isinstance(importing_columns, list):
            new_data = new_data.select([date_column_name] + importing_columns)

        if "Date" not in self.columns:
            error_msg = "No date column in the base dataframe."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Convert new_data to match the type of lf
        _, new_data = self._verify_compatible_frames(
            self.lf,
            [new_data],
            mode="match 1",
        )
        new_data = new_data[0]
        if not isinstance(
            new_data.collect_schema().dtypes()[
                new_data.collect_schema().names().index(date_column_name)
            ],
            pl.Datetime,
        ):
            new_data = new_data.with_columns(
                pl.col(date_column_name).str.to_datetime(format=datetime_format),
            )

        # Ensure both DataFrames have DateTime columns in the same unit
        new_data = new_data.with_columns(
            pl.col(date_column_name).dt.cast_time_unit("us"),
        )
        self.lf = self.lf.with_columns(
            pl.col("Date").dt.cast_time_unit("us"),
        )

        # Check for timezone mismatch and harmonize to self.lf's timezone
        live_schema = self.lf.collect_schema()
        new_schema = new_data.collect_schema()

        live_dtype = live_schema["Date"]
        new_dtype = new_schema[date_column_name]

        if isinstance(live_dtype, pl.Datetime) and isinstance(new_dtype, pl.Datetime):
            live_tz = live_dtype.time_zone
            new_tz = new_dtype.time_zone

            if live_tz is None:
                if existing_data_timezone is not None:
                    local_tz = existing_data_timezone
                else:
                    local_tz = str(get_localzone())
                self.lf = self.lf.with_columns(
                    pl.col("Date").dt.replace_time_zone(local_tz),
                )
                live_tz = local_tz

            if new_tz is None and new_data_timezone is not None:
                new_data = new_data.with_columns(
                    pl.col(date_column_name).dt.replace_time_zone(new_data_timezone),
                )
                new_tz = new_data_timezone

            if live_tz != new_tz:
                if new_tz is None:
                    # New is naive, assume it is in live_tz
                    new_data = new_data.with_columns(
                        pl.col(date_column_name).dt.replace_time_zone(live_tz),
                    )
                else:
                    # Both aware, convert new to live_tz
                    new_data = new_data.with_columns(
                        pl.col(date_column_name).dt.convert_time_zone(live_tz),
                    )

        # Rename date column to "Date"
        new_data = new_data.rename({date_column_name: "Date"})
        new_result = Result(lf=new_data, info={})

        if align_on is not None:
            from pyprobe.analysis.time_series import align_data

            col_existing, col_new = align_on
            _, new_result = align_data(self, new_result, col_existing, col_new)

        new_data = new_result.lf
        new_data_cols = [
            col for col in new_data.collect_schema().names() if col != "Date"
        ]
        all_data = self.lf.clone().join(
            new_data,
            on="Date",
            how="full",
            coalesce=True,
        )
        interpolated = all_data.with_columns(
            pl.col(new_data_cols).interpolate_by("Date"),
        ).select(pl.col(["Date"] + new_data_cols))
        self.lf = self.lf.join(
            interpolated,
            on="Date",
            how="left",
            coalesce=True,
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
        if "Date" not in self.columns:
            error_msg = "No date column in the base dataframe."
            logger.error(error_msg)
            raise ValueError(error_msg)
        # get the columns of the new data
        new_data_cols = new_data.collect_schema().names()
        new_data_cols.remove(date_column_name)
        # check if the new data is lazyframe or not
        _, new_data = self._verify_compatible_frames(
            self.lf,
            [new_data],
            mode="match 1",
        )
        new_data = new_data[0]
        if (
            new_data.dtypes[new_data.collect_schema().names().index(date_column_name)]
            != pl.Datetime
        ):
            new_data = new_data.with_columns(pl.col(date_column_name).str.to_datetime())

        # Ensure both DataFrames have DateTime columns in the same unit
        new_data = new_data.with_columns(
            pl.col(date_column_name).dt.cast_time_unit("us"),
        )
        self.lf = self.lf.with_columns(
            pl.col("Date").dt.cast_time_unit("us"),
        )

        all_data = self.lf.clone().join(
            new_data,
            left_on="Date",
            right_on=date_column_name,
            how="full",
            coalesce=True,
        )
        interpolated = all_data.with_columns(
            pl.col(new_data_cols).interpolate_by("Date"),
        ).select(pl.col(["Date"] + new_data_cols))
        self.lf = self.lf.join(
            interpolated,
            on="Date",
            how="left",
            coalesce=True,
        )

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
        return cls(lf=data, info=info)

    def export_to_mat(self, filename: str) -> None:
        """Export the data to a .mat file.

        This method will export the data and info dictionary to a .mat file. The
        variables in the .mat file will be named 'data' and 'info'. Column names and
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

        # Replace any non-alphanumeric character with an underscore in the info
        # dictionary keys
        renamed_info = {
            re.sub(r"\W", "_", key): value for key, value in self.info.items()
        }

        variable_dict = {
            "data": renamed_data.to_dict(),
            "info": renamed_info,
        }
        savemat(filename, variable_dict, oned_as="column")

    @catch_pydantic_validation
    @staticmethod
    def from_polars_io(
        polars_io_func: Callable[..., pl.DataFrame | pl.LazyFrame],
        info: dict[str, Any | None] = {},
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
            info (dict[str, Any | None]):
                The info dictionary for the new Result object. Empty by default.
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
                info={"test": "test"},
                column_definitions={},
                source="data.csv",
            )

            From a pandas DataFrame:

            .. code-block:: python

            result = Result.from_polars_io(
                pl.from_pandas,
                info={"test": "test"},
                column_definitions={},
                data=pd.DataFrame({"a": [1, 2, 3]}),
            )

            From a numpy array:

            .. code-block:: python

            result = Result.from_polars_io(
                pl.from_numpy,
                info={"test": "test"},
                column_definitions={},
                data=np.array([[1, 2, 3], [4, 5, 6]]),
                schema=["a", "b"]
            )

        """
        return Result(
            lf=polars_io_func(**kwargs),
            info=info,
            column_definitions=column_definitions,
        )

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
        instructions = [pl.lit(result.info[key]).alias(key) for key in result.info]
        result.lf = result.lf.with_columns(instructions)
    results[0].extend(results[1:], concat_method=concat_method)
    return results[0]
