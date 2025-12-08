"""A module of classes to map cycler columns to PyProBE columns."""

from abc import ABC, abstractmethod
from typing import Any

import polars as pl
from loguru import logger
from pydantic import (
    GetCoreSchemaHandler,
)
from pydantic_core import CoreSchema, core_schema

from pyprobe.units import valid_units


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

    def validate(self, columns: list[str]) -> None:
        """Validate the column mapping.

        This method checks if the required columns are present in the cycler data. It
        fills the attribute :code:`column_map` with the name of the columns in the
        cycler data that match the required patterns and the extracted units. It also
        sets the attribute :code:`columns_validated` to True if all required columns are
        present.

        Args:
            columns: The list of columns in the cycler data.
        """
        self.column_map = self.match_columns(columns, self.required_cycler_cols)
        self.columns_validated = len(self.column_map) == len(self.required_cycler_cols)
        with logger.contextualize(
            column_importer=self.__class__.__name__,
        ):
            if self.columns_validated:
                logger.info(
                    f"Column mapping validated: {self.pyprobe_name} -> "
                    f"{self.column_map}"
                )
            else:
                missing_columns = set(self.required_cycler_cols) - set(
                    self.column_map.keys()
                )
                logger.info(
                    f"Failed to find required columns for {self.pyprobe_name}. "
                    f"Missing: {missing_columns}"
                )


class CastAndRenameMap(ColumnMap):
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
        """Initialize the CastAndRenameMap class."""
        super().__init__(pyprobe_name, [required_cycler_col])
        self.data_type = data_type

    @property
    def expr(self) -> pl.Expr:
        """Get the polars expression for the column mapping."""
        return pl.col(self.cycler_col).cast(self.data_type).alias(self.pyprobe_name)


class DateTimeMap(ColumnMap):
    """A template mapping for datetime columns.

    Args:
        required_cycler_col: The name of the required cycler column.
        datetime_format: The format of the datetime string.
    """

    def __init__(self, required_cycler_col: str, datetime_format: str) -> None:
        """Initialize the DateTimeMap class."""
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


class TimeFromDateMap(ColumnMap):
    """A template mapping for extracting time from a date column.

    Args:
        required_cycler_col: The name of the required cycler column.
    """

    def __init__(self, required_cycler_col: str, datetime_format: str) -> None:
        """Initialize the TimeFromDateMap class."""
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


class ConvertUnitsMap(ColumnMap):
    """A template mapping for converting units.

    Args:
        pyprobe_name: The name of the PyProBE column.
        required_cycler_col: The name of the required cycler column.
    """

    def __init__(self, pyprobe_name: str, required_cycler_col: str) -> None:
        """Initialize the ConvertUnitsMap class."""
        super().__init__(pyprobe_name, [required_cycler_col])

    @property
    def expr(self) -> pl.Expr:
        """Get the polars expression for the column mapping."""
        unit = self.column_map[self.cycler_col]["Cycler unit"]
        return (
            self.get(self.cycler_col).units.to_base_unit(unit).alias(self.pyprobe_name)
        )


class ConvertTemperatureMap(ColumnMap):
    """A template mapping for converting temperature units.

    Args:
        required_cycler_col: The name of the required cycler column.
    """

    def __init__(self, required_cycler_col: str) -> None:
        """Initialize the ConvertTemperatureMap class."""
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


class CapacityFromChDchMap(ColumnMap):
    """A template mapping for calculating capacity from charge and discharge columns.

    Args:
        charge_capacity_col: The name of the charge capacity column.
        discharge_capacity_col: The name of the discharge capacity column.
    """

    def __init__(self, charge_capacity_col: str, discharge_capacity_col: str) -> None:
        """Initialize the CapacityFromChDchMap class."""
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
        charge_capacity = self.get(self.charge_capacity_col).units.to_base_unit(
            charge_capacity_unit,
        )
        discharge_capacity = self.get(self.discharge_capacity_col).units.to_base_unit(
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


class CapacityFromCurrentSignMap(ColumnMap):
    """A template mapping for calculating capacity from current and time columns.

    Args:
        capacity_col: The name of the capacity column.
        current_col: The name of the current column.
    """

    def __init__(self, capacity_col: str, current_col: str) -> None:
        """Initialize the CapacityFromCurrentSignMap class."""
        pyprobe_name = "Capacity [Ah]"
        self.current_col = current_col
        self.capacity_col = capacity_col
        required_cycler_cols = [self.current_col, self.capacity_col]
        super().__init__(pyprobe_name, required_cycler_cols)

    @property
    def capacity(self) -> pl.Expr:
        """Get the capacity column."""
        return self.get(self.capacity_col).units.to_base_unit(
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


class StepFromCategoricalMap(ColumnMap):
    """A template mapping for calculating step from categorical columns.

    An example of a categorical column is one that describes the type of step, e.g.
    "CC-CV", "CV", "OCV", etc. This method will fill the step column with an
    incrementing count each time the categorical column changes.

    Args:
        categorical_col: The name of the categorical column.
    """

    def __init__(self, categorical_col: str) -> None:
        """Initialize the StepFromCategoricalMap class."""
        pyprobe_name = "Step"
        super().__init__(pyprobe_name, [categorical_col])
        self.categorical_col = categorical_col

    @property
    def expr(self) -> pl.Expr:
        """Get the polars expression for the column mapping."""
        logger.warning(
            f"Step number is being inferred from the categorical column "
            f"{self.categorical_col}. "
            "A new step will be counted each time the column changes. This means that "
            "it will not be possible to filter by cycle.",
        )
        return (
            # Compare current value with previous value
            self.get(self.cycler_col)
            .ne(self.get(self.cycler_col).shift(1))
            # First value is always a new step (no previous value)
            .fill_null(False)
            # Convert boolean to integer (1 when changed, 0 when unchanged)
            .cast(pl.UInt32)
            # Create cumulative step counter
            .cum_sum()
            .alias(self.pyprobe_name)
        )
