"""A module to load and process battery cycler data."""

import glob
import logging
import os
import warnings
from typing import Dict, List, Optional

import polars as pl
from pydantic import BaseModel, field_validator, model_validator

from pyprobe.units import Units

logger = logging.getLogger(__name__)


class BaseCycler(BaseModel):
    """A class to load and process battery cycler data."""

    input_data_path: str
    """The path to the input data."""
    column_dict: Dict[str, str]
    """A dictionary mapping the column name format of the cycler to the PyProBE format.
    Units are indicated by an asterisk (*)."""
    datetime_format: Optional[str] = None
    """The string format of the date column if present. See the
    `chrono crate <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_
    documentation for more information on the format string.
    """
    header_row_index: int = 0
    """The index of the header row in the data file."""

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

    @field_validator("column_dict")
    @classmethod
    def _check_column_dict(cls, value: Dict[str, str]) -> Dict[str, str]:
        """Check if the column dictionary is valid.

        Args:
            value (Dict[str, str]): The column dictionary.

        Returns:
            Dict[str, str]: The column dictionary.
        """
        pyprobe_data_columns = {value for value in value.values()}
        pyprobe_required_columns = {
            "Time [*]",
            "Current [*]",
            "Voltage [*]",
            "Step",
            "Capacity [*]",
        }
        missing_columns = pyprobe_required_columns - pyprobe_data_columns
        extra_error_message = ""
        if "Capacity [*]" in missing_columns:
            if {"Charge Capacity [*]", "Discharge Capacity [*]"}.issubset(
                pyprobe_data_columns
            ):
                missing_columns.remove("Capacity [*]")
            else:
                missing_columns.add("Charge Capacity [*]")
                missing_columns.add("Discharge Capacity [*]")
                extra_error_message = (
                    " Capacity can be specified as 'Capacity [*]' or "
                    "'Charge Capacity [*]' and 'Discharge Capacity [*]'."
                )
        if len(missing_columns) > 0:
            error_msg = (
                f"The column dictionary is missing one or more required columns: "
                f"{missing_columns}." + extra_error_message
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        return value

    @model_validator(mode="after")
    def import_and_validate_data(self) -> "BaseCycler":
        """Import the data and validate the column mapping."""
        dataframe_list = self._get_dataframe_list()
        self._imported_dataframe = self.get_imported_dataframe(dataframe_list)
        self._column_map = self._map_columns(
            self.column_dict, self._imported_dataframe.collect_schema().names()
        )
        self._check_missing_columns(self.column_dict, self._column_map)
        return self

    @staticmethod
    def _check_missing_columns(
        column_dict: Dict[str, str], column_map: Dict[str, Dict[str, str | pl.DataType]]
    ) -> None:
        """Check for missing columns in the imported data.

        Args:
            column_dict:
                A dictionary mapping the column name format of the cycler to the
                PyProBE format.
            column_map:
                A dictionary mapping the column name format of the cycler to the PyProBE
                format, for the imported data including units.

        Raises:
            ValueError:
                If any of ["Time", "Current", "Voltage", "Capacity", "Step"]
                are missing.
        """
        pyprobe_required_columns = set(
            [
                "Time",
                "Current",
                "Voltage",
                "Capacity",
                "Step",
            ]
        )
        missing_columns = pyprobe_required_columns - set(column_map.keys())
        if missing_columns:
            if "Capacity" in missing_columns:
                if (
                    "Charge Capacity" in column_map.keys()
                    and "Discharge Capacity" in column_map.keys()
                ):
                    missing_columns.remove("Capacity")
                else:
                    missing_columns.add("Charge Capacity")
                    missing_columns.add("Discharge Capacity")
        if len(missing_columns) > 0:
            search_names = []
            for column in missing_columns:
                if column != "Step":
                    full_name = column + " [*]"
                else:
                    full_name = column
                for cycler_name, pyprobe_name in column_dict.items():
                    if pyprobe_name == full_name:
                        search_names.append(cycler_name)
            error_msg = (
                f"PyProBE cannot find the following columns, please check your data: "
                f"{search_names}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    @staticmethod
    def read_file(
        filepath: str, header_row_index: int = 0
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
                    filepath, infer_schema=False, skip_rows=header_row_index
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
        list = [self.read_file(file, self.header_row_index) for file in files]
        all_columns = set([col for df in list for col in df.collect_schema().names()])
        for i in range(len(list)):
            if len(list[i].collect_schema().names()) < len(all_columns):
                logger.warning(
                    f"File {os.path.basename(files[i])} has missing columns, "
                    "these have been filled with null values."
                )
        return list

    def get_imported_dataframe(
        self, dataframe_list: List[pl.DataFrame]
    ) -> pl.DataFrame:
        """Return a single DataFrame from a list of DataFrames.

        Args:
            dataframe_list: A list of DataFrames.

        Returns:
            DataFrame: A single DataFrame.
        """
        return pl.concat(dataframe_list, how="diagonal", rechunk=True)

    @staticmethod
    def _match_unit(column_name: str, pattern: str) -> Optional[str]:
        """Return the unit of a column name in the place of an asterisk.

        Args:
            column_name (str): The column name.
            pattern (str): The pattern to match.

        Returns:
            Optional[str]:
                The unit or None if the column name does not match the pattern.
        """
        if "*" not in pattern:
            if column_name == pattern:
                return ""
            else:
                return None
        else:
            pattern_parts = pattern.split("*")
            if column_name.startswith(pattern_parts[0]) and column_name.endswith(
                pattern_parts[1]
            ):
                unit = (
                    column_name[len(pattern_parts[0]) : -len(pattern_parts[1])]
                    if pattern_parts[1]
                    else column_name[len(pattern_parts[0]) :]
                )
                return unit
            else:
                return None

    @classmethod
    def _map_columns(
        cls, column_dict: Dict[str, str], dataframe_columns: List[str]
    ) -> Dict[str, Dict[str, str | pl.DataType]]:
        """Map the columns of the imported dataframe to the PyProBE format.

        Args:
            column_dict (Dict[str, str]):
                A dictionary mapping the column name format of the cycler to the PyProBE
                format.
            dataframe_columns (List[str]): The columns of the imported dataframe.

        Returns:
            Dict[str, Dict[str, str | pl.DataType]]:
                A dictionary mapping the column name format of the cycler to the PyProBE
                format.

                Fields (for each quantity):
                    Cycler column name (str): The name of the column in the cycler data.
                    PyProBE column name (str):
                        The name of the column in the PyProBE data.
                    Unit (str): The unit of the column.
                    Type (pl.DataType): The data type of the column.
        """
        column_map: Dict[str, Dict[str, str | pl.DataType]] = {}
        for cycler_format, pyprobe_format in column_dict.items():
            for cycler_column_name in dataframe_columns:
                unit = cls._match_unit(cycler_column_name, cycler_format)
                if unit is not None:
                    quantity = pyprobe_format.replace(" [*]", "")
                    default_units = {
                        "Time": "s",
                        "Current": "A",
                        "Voltage": "V",
                        "Capacity": "Ah",
                        "Charge Capacity": "Ah",
                        "Discharge Capacity": "Ah",
                        "Temperature": "C",
                    }

                    if quantity == "Temperature" and unit != "K":
                        unit = "C"
                    elif unit == "" and quantity in default_units:
                        unit = default_units[quantity]

                    pyprobe_column_name = pyprobe_format.replace("*", unit)

                    column_map[quantity] = {}
                    column_map[quantity]["Cycler column name"] = cycler_column_name
                    column_map[quantity]["PyProBE column name"] = pyprobe_column_name
                    column_map[quantity]["Unit"] = unit
                    if quantity == "Date":
                        column_map[quantity]["Type"] = pl.String
                    elif quantity == "Step":
                        column_map[quantity]["Type"] = pl.Int64
                    else:
                        column_map[quantity]["Type"] = pl.Float64
        return column_map

    def _assign_instructions(self) -> None:
        instruction_dict = {
            "Date": self.date,
            "Time": self.time,
            "Current": self.current,
            "Voltage": self.voltage,
            "Capacity": self.capacity,
            "Temperature": self.temperature,
            "Step": self.step,
            "Event": self.event,
        }
        for quantity in self._column_map.keys():
            self._column_map[quantity]["Instruction"] = instruction_dict[quantity]

    @staticmethod
    def _tabulate_column_map(
        column_map: Dict[str, Dict[str, str | pl.DataType]],
    ) -> str:
        data = {
            "Quantity": list(column_map.keys()),
            "Cycler column name": [
                v["Cycler column name"] for v in column_map.values()
            ],
            "PyProBE column name": [
                v["PyProBE column name"] for v in column_map.values()
            ],
        }

        return pl.DataFrame(data)

    def _convert_names(self, quantity: str) -> pl.Expr:
        """Write a column in the PyProBE column name format and convert its type.

        Args:
            quantity (str): The quantity to convert.

        Returns:
            pl.Expr:
                A polars expression to convert the name to the PyProBE format and cast
                to the correct data type.
        """
        column = self._column_map[quantity]
        # cast to type and rename
        return (
            pl.col(column["Cycler column name"])
            .cast(column["Type"])
            .alias(column["PyProBE column name"])
        )

    @property
    def pyprobe_dataframe(self) -> pl.DataFrame:
        """The DataFrame containing the required columns.

        Returns:
            pl.DataFrame: The DataFrame.
        """
        required_columns = [
            self.date if "Date" in self._column_map.keys() else None,
            self.time,
            self.step,
            self.event,
            self.current,
            self.voltage,
            self.capacity
            if "Capacity" in self._column_map.keys()
            else self.capacity_from_ch_dch,
            self.temperature if "Temperature" in self._column_map.keys() else None,
        ]
        name_converters = [
            self._convert_names(quantity) for quantity in self._column_map.keys()
        ]
        imported_dataframe = self._imported_dataframe.with_columns(name_converters)
        required_columns = [col for col in required_columns if col is not None]
        return imported_dataframe.select(required_columns)

    @property
    def date(self) -> pl.Expr:
        """Identify and format the date column.

        Returns:
            pl.Expr: A polars expression for the date column.
        """
        return (
            pl.col("Date")
            .str.strip_chars()
            .str.to_datetime(format=self.datetime_format, time_unit="us")
        )

    @property
    def time(self) -> pl.Expr:
        """Identify and format the time column.

        Returns:
            pl.Expr: A polars expression for the time column.
        """
        return Units("Time", self._column_map["Time"]["Unit"]).to_default_unit()

    @property
    def current(self) -> pl.Expr:
        """Identify and format the current column.

        Returns:
            pl.Expr: A polars expression for the current column.
        """
        return Units("Current", self._column_map["Current"]["Unit"]).to_default_unit()

    @property
    def voltage(self) -> pl.Expr:
        """Identify and format the voltage column.

        Returns:
            pl.Expr: A polars expression for the voltage column.
        """
        return Units("Voltage", self._column_map["Voltage"]["Unit"]).to_default_unit()

    @property
    def charge_capacity(self) -> pl.Expr:
        """Identify and format the charge capacity column.

        Returns:
            pl.Expr: A polars expression for the charge capacity column.
        """
        return Units(
            "Charge Capacity", self._column_map["Charge Capacity"]["Unit"]
        ).to_default_unit()

    @property
    def discharge_capacity(self) -> pl.Expr:
        """Identify and format the discharge capacity column.

        Returns:
            pl.Expr: A polars expression for the discharge capacity column.
        """
        return Units(
            "Discharge Capacity", self._column_map["Discharge Capacity"]["Unit"]
        ).to_default_unit()

    @property
    def capacity_from_ch_dch(self) -> pl.Expr:
        """Calculate the capacity from charge and discharge capacities.

        Returns:
            pl.Expr: A polars expression for the capacity column.
        """
        diff_charge_capacity = (
            self.charge_capacity.diff().clip(lower_bound=0).fill_null(strategy="zero")
        )

        diff_discharge_capacity = (
            self.discharge_capacity.diff()
            .clip(lower_bound=0)
            .fill_null(strategy="zero")
        )
        return (
            (diff_charge_capacity - diff_discharge_capacity).cum_sum()
            + self.charge_capacity.max()
        ).alias("Capacity [Ah]")

    @property
    def capacity(self) -> pl.Expr:
        """Identify and format the capacity column.

        Returns:
            pl.Expr: A polars expression for the capacity column.
        """
        return Units("Capacity", self._column_map["Capacity"]["Unit"]).to_default_unit()

    @property
    def temperature(self) -> pl.Expr:
        """Identify and format the temperature column.

        Returns:
            pl.Expr: A polars expression for the temperature column.
        """
        return Units(
            "Temperature", self._column_map["Temperature"]["Unit"]
        ).to_default_unit()

    @property
    def step(self) -> pl.Expr:
        """Identify the step number.

        Returns:
            pl.Expr: A polars expression for the step number.
        """
        return pl.col("Step")

    @property
    def event(self) -> pl.Expr:
        """Identify the event number.

        Events are defined by any change in the step number, increase or decrease.

        Returns:
            pl.Expr: A polars expression for the event number.
        """
        return (
            (pl.col("Step").cast(pl.Int64) - pl.col("Step").cast(pl.Int64).shift() != 0)
            .fill_null(strategy="zero")
            .cum_sum()
            .alias("Event")
            .cast(pl.Int64)
        )
