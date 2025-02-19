"""A module to load and process battery cycler data."""

import glob
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import polars as pl
from pydantic import BaseModel, GetCoreSchemaHandler, field_validator, model_validator
from pydantic_core import CoreSchema, core_schema

from pyprobe.units import valid_units

logger = logging.getLogger(__name__)


class ColumnMap(ABC):
    """A class to map cycler columns to PyProBE columns."""

    def __init__(self, pyprobe_name: str, required_cycler_cols: list[str]) -> None:
        """Initialize the ColumnMap class."""
        self.pyprobe_name = pyprobe_name
        self.required_cycler_cols = required_cycler_cols

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        """Get the Pydantic core schema for the ColumnMap class."""
        # Checks only that the value is an instance of ColumnMap.
        return core_schema.is_instance_schema(cls)

    @property
    @abstractmethod
    def expr(self) -> pl.Expr:
        """Get the polars expression for the column mapping."""
        pass

    def get(self, column: str) -> pl.Expr:
        """Get a column from a cycler DataFrame using the keys in required_cycler_cols.

        Args:
            column: The column name, as given in required_cycler_cols.

        Returns:
            A polars expression for the column in the cycler data.
        """
        return pl.col(self.column_map[column]["Cycler name"])

    @property
    def cycler_col(self) -> str:
        """Get the single cycler column name, if only one is required."""
        if len(self.column_map) != 1:
            raise ValueError("Method only valid for single column mappings")
        return list(self.column_map.keys())[0]

    def match_columns(
        self,
        available_columns: list[str],
        required_patterns: list[str],
    ) -> dict[str, dict[str, str]]:
        """Find columns that match the required patterns, handling wildcards.

        Args:
            available_columns: List of actual column names in the data
            required_patterns: List of required column patterns with * wildcards

        Returns:
            dict[str, dict[str, str]]: Nested mapping of pattern to column details:
                - pattern -> {
                    "Cycler name": full column name,
                    "Cycler unit": extracted unit (if pattern contains *)
                }
        """
        available_set = set(available_columns)
        matches = {}

        for pattern in required_patterns:
            if "*" not in pattern:
                # Direct match required
                if pattern in available_set:
                    matches[pattern] = {"Cycler name": pattern, "Cycler unit": ""}
            else:
                # Split pattern into prefix and suffix
                prefix, suffix = pattern.split("*")
                # Find first matching column
                for col in available_set:
                    if col.startswith(prefix) and col.endswith(suffix):
                        unit = (
                            col[len(prefix) : -len(suffix)]
                            if suffix
                            else col[len(prefix) :]
                        )
                        if (
                            unit not in valid_units
                            and self.pyprobe_name != "Temperature [C]"
                        ):
                            continue
                        matches[pattern] = {"Cycler name": col, "Cycler unit": unit}

        return matches

    def validate(self, column_list: list[str]) -> None:
        """Validate the column mapping.

        This method checks if the required columns are present in the cycler data. It
        fills the attribute :code:`column_map` with the name of the columns in the
        cycler data that match the required patterns and the extracted units. It also
        sets the attribute :code:`columns_validated` to True if all required columns are
        present.

        Args:
            column_list: The list of columns in the cycler data.
        """
        self.column_map = self.match_columns(column_list, self.required_cycler_cols)
        self.columns_validated = len(self.column_map) == len(self.required_cycler_cols)


class CastAndRename(ColumnMap):
    """A template mapping for simple column renaming.

    Args:
        pyprobe_name: The name of the PyProBE column.
        required_cycler_col: The name of the required cycler column.
    """

    def __init__(
        self,
        pyprobe_name: str,
        required_cycler_col: str,
        data_type: pl.DataType,
    ) -> None:
        """Initialize the CastAndRename class."""
        super().__init__(pyprobe_name, [required_cycler_col])
        self.data_type = data_type

    @property
    def expr(self) -> pl.Expr:
        """Get the polars expression for the column mapping."""
        return pl.col(self.cycler_col).cast(self.data_type).alias(self.pyprobe_name)


class DateTime(ColumnMap):
    """A template mapping for datetime columns.

    Args:
        required_cycler_col: The name of the required cycler column.
    """

    def __init__(self, required_cycler_col: str, datetime_format: str) -> None:
        """Initialize the DateTime class."""
        pyprobe_name = "Date"
        super().__init__(pyprobe_name, [required_cycler_col])
        self.required_cycler_col = required_cycler_col
        self.datetime_format = datetime_format

    @property
    def expr(self) -> pl.Expr:
        """Get the polars expression for the column mapping."""
        return (
            self.get(self.cycler_col)
            .str.strip_chars()
            .str.to_datetime(format=self.datetime_format, time_unit="us")
        ).alias(self.pyprobe_name)


class TimeFromDate(ColumnMap):
    """A template mapping for extracting time from a date column.

    Args:
        required_cycler_col: The name of the required cycler column.
    """

    def __init__(self, required_cycler_col: str, datetime_format: str) -> None:
        """Initialize the TimeFromDate class."""
        pyprobe_name = "Time [s]"
        self.datetime_format = datetime_format
        super().__init__(pyprobe_name, [required_cycler_col])

    @property
    def expr(self) -> pl.Expr:
        """Get the polars expression for the column mapping."""
        return (
            (
                self.get(self.cycler_col)
                .str.to_datetime(format=self.datetime_format, time_unit="us")
                .diff()
                .dt.total_microseconds()
                .cum_sum()
                / 1e6
            )
            .fill_null(strategy="zero")
            .alias(self.pyprobe_name)
        )


class ConvertUnits(ColumnMap):
    """A template mapping for converting units.

    Args:
        pyprobe_name: The name of the PyProBE column.
        required_cycler_col: The name of the required cycler column.
    """

    def __init__(self, pyprobe_name: str, required_cycler_col: str) -> None:
        """Initialize the ConvertUnits class."""
        super().__init__(pyprobe_name, [required_cycler_col])

    @property
    def expr(self) -> pl.Expr:
        """Get the polars expression for the column mapping."""
        unit = self.column_map[self.cycler_col]["Cycler unit"]
        return self.get(self.cycler_col).units.to_si(unit).alias(self.pyprobe_name)


class ConvertTemperature(ColumnMap):
    """A template mapping for converting temperature units.

    Args:
        required_cycler_col: The name of the required cycler column.
    """

    def __init__(self, required_cycler_col: str) -> None:
        """Initialize the ConvertTemperature class."""
        pyprobe_name = "Temperature [C]"
        super().__init__(pyprobe_name, [required_cycler_col])

    @property
    def expr(self) -> pl.Expr:
        """Get the polars expression for the column mapping."""
        unit = self.column_map[self.cycler_col]["Cycler unit"]
        if unit == "K":
            return (
                (self.get(self.cycler_col) - 273.15)
                .cast(pl.Float64)
                .alias(self.pyprobe_name)
            )
        else:
            return self.get(self.cycler_col).cast(pl.Float64).alias(self.pyprobe_name)


class CapacityFromChDch(ColumnMap):
    """A template mapping for calculating capacity from charge and discharge columns.

    Args:
        charge_capacity_col: The name of the charge capacity column.
        discharge_capacity_col: The name of the discharge capacity column.
    """

    def __init__(self, charge_capacity_col: str, discharge_capacity_col: str) -> None:
        """Initialize the CapacityFromChDch class."""
        pyprobe_name = "Capacity [Ah]"
        self.charge_capacity_col = charge_capacity_col
        self.discharge_capacity_col = discharge_capacity_col
        required_cycler_cols = [self.charge_capacity_col, self.discharge_capacity_col]
        super().__init__(pyprobe_name, required_cycler_cols)

    @property
    def expr(self) -> pl.Expr:
        """Get the polars expression for the column mapping."""
        charge_capacity_unit = self.column_map[self.charge_capacity_col]["Cycler unit"]
        discharge_capacity_unit = self.column_map[self.discharge_capacity_col][
            "Cycler unit"
        ]
        charge_capacity = self.get(self.charge_capacity_col).units.to_si(
            charge_capacity_unit,
        )
        discharge_capacity = self.get(self.discharge_capacity_col).units.to_si(
            discharge_capacity_unit,
        )
        diff_charge_capacity = (
            charge_capacity.diff().clip(lower_bound=0).fill_null(strategy="zero")
        )

        diff_discharge_capacity = (
            discharge_capacity.diff().clip(lower_bound=0).fill_null(strategy="zero")
        )
        return (
            (diff_charge_capacity - diff_discharge_capacity).cum_sum()
            + charge_capacity.max()
        ).alias(self.pyprobe_name)


class CapacityFromCurrentSign(ColumnMap):
    """A template mapping for calculating capacity from current and time columns.

    Args:
        capacity_col: The name of the capacity column.
        current_col: The name of the current column.
    """

    def __init__(self, capacity_col: str, current_col: str) -> None:
        """Initialize the CapacityFromCurrentSign class."""
        pyprobe_name = "Capacity [Ah]"
        self.current_col = current_col
        self.capacity_col = capacity_col
        required_cycler_cols = [self.current_col, self.capacity_col]
        super().__init__(pyprobe_name, required_cycler_cols)

    @property
    def capacity(self) -> pl.Expr:
        """Get the capacity column."""
        return self.get(self.capacity_col).units.to_si(
            self.column_map[self.capacity_col]["Cycler unit"],
        )

    @property
    def current(self) -> pl.Expr:
        """Get the current column."""
        return self.get(self.current_col).cast(pl.Float64)

    @property
    def expr(self) -> pl.Expr:
        """Get the polars expression for the column mapping."""
        current_direction = self.current.sign()
        charge_capacity = self.capacity * current_direction.replace(-1, 0).abs()
        discharge_capacity = self.capacity * current_direction.replace(1, 0).abs()
        diff_charge_capacity = (
            charge_capacity.diff().clip(lower_bound=0).fill_null(strategy="zero")
        )

        diff_discharge_capacity = (
            discharge_capacity.diff().clip(lower_bound=0).fill_null(strategy="zero")
        )
        return (
            (diff_charge_capacity - diff_discharge_capacity).cum_sum()
            + charge_capacity.max()
        ).alias(self.pyprobe_name)


class BaseCycler(BaseModel):
    """A class to load and process battery cycler data."""

    input_data_path: str
    """The path to the input data."""
    header_row_index: int = 0
    """The index of the header row in the data file."""

    column_importers: list[ColumnMap]
    """A list of :class:`ColumnMap` objects to map cycler columns to PyProBE columns."""

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    @field_validator("input_data_path")
    @classmethod
    def _check_input_data_path(cls, value: str) -> str:
        """Check if the input data path is valid.

        Args:
            value (str): The input data path.

        Returns:
            str: The input data path.
        """
        if "*" in value:
            files = glob.glob(value)
            if len(files) == 0:
                error_msg = f"No files found with the pattern {value}."
                logger.error(error_msg)
                raise ValueError(error_msg)
        elif not os.path.exists(value):
            error_msg = f"File not found: path {value} does not exist."
            logger.error(error_msg)
            raise ValueError(error_msg)
        return value

    @staticmethod
    def read_file(
        filepath: str,
        header_row_index: int = 0,
    ) -> pl.DataFrame | pl.LazyFrame:
        """Read a battery cycler file into a DataFrame.

        Args:
            filepath: The path to the file.
            header_row_index: The index of the header row.
            header_row_index: The index of the header row.

        Returns:
            pl.DataFrame | pl.LazyFrame: The DataFrame.
        """
        file = os.path.basename(filepath)
        file_ext = os.path.splitext(file)[1]
        match file_ext.lower():
            case ".xlsx":
                return pl.read_excel(
                    filepath,
                    engine="calamine",
                    infer_schema_length=0,
                    read_options={"header_row": header_row_index},
                )
            case ".csv":
                return pl.scan_csv(
                    filepath,
                    infer_schema=False,
                    skip_rows=header_row_index,
                )
            case _:
                error_msg = f"Unsupported file extension: {file_ext}"
                logger.error(error_msg)
                raise ValueError(error_msg)

    def _get_dataframe_list(self) -> list[pl.DataFrame | pl.LazyFrame]:
        """Return a list of all the imported dataframes.

        Args:
            input_data_path (str): The path to the input data.

        Returns:
            List[DataFrame]: A list of DataFrames.
        """
        files = glob.glob(self.input_data_path)
        files.sort()
        df_list = [self.read_file(file, self.header_row_index) for file in files]
        all_columns = {col for df in df_list for col in df.collect_schema().names()}
        for i in range(len(df_list)):
            if len(df_list[i].collect_schema().names()) < len(all_columns):
                logger.warning(
                    f"File {os.path.basename(files[i])} has missing columns, "
                    "these have been filled with null values.",
                )
        return df_list

    def get_imported_dataframe(
        self,
        dataframe_list: list[pl.DataFrame],
    ) -> pl.DataFrame:
        """Return a single DataFrame from a list of DataFrames.

        Args:
            dataframe_list: A list of DataFrames.

        Returns:
            DataFrame: A single DataFrame.
        """
        return pl.concat(dataframe_list, how="diagonal", rechunk=True)

    @model_validator(mode="after")
    def import_and_validate_data(self) -> "BaseCycler":
        """Import the data and validate the column mapping."""
        dataframe_list = self._get_dataframe_list()
        self._imported_dataframe = self.get_imported_dataframe(dataframe_list)
        for column_importer in self.column_importers:
            column_importer.validate(self._imported_dataframe.collect_schema().names())
        return self

    def get_pyprobe_dataframe(self) -> pl.DataFrame:
        """Return the PyProBE DataFrame."""
        imported_columns = set()
        importers: list[ColumnMap] = []
        for importer in self.column_importers:
            if (
                importer.columns_validated
                and importer.pyprobe_name not in imported_columns
            ):
                importers.append(importer.expr)
                imported_columns.add(importer.pyprobe_name)
        event_expr = (
            (pl.col("Step").cast(pl.Int64) - pl.col("Step").cast(pl.Int64).shift() != 0)
            .fill_null(strategy="zero")
            .cum_sum()
            .alias("Event")
            .cast(pl.Int64)
        )
        return (
            self._imported_dataframe.select(importers)
            .with_columns(event_expr)
            .collect()
        )

    # column_dict = {
    #     "Step": ["Count"],
    # }
    # column_map = [
    #     CastAndRename("Step", "Count", pl.Int64),
    # ]

    # def cast_and_rename(cycler_column, pyprobe_column, data_type):
    #     return pl.col(cycler_column).cast(data_type).alias(pyprobe_column)

    # def convert_units(cycler_column):
    #     return pl.col(cycler_column).units.to_default()
