"""A module to load and process battery cycler data."""

import glob
import os
import warnings

# from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import polars as pl
from pydantic import BaseModel

from pyprobe.units import Units


class BaseCycler(BaseModel):
    """A class to load and process battery cycler data.

    Args:
        input_data_path (str): The path to the input data.
        column_dict (Dict[str, str]): A dictionary mapping the column name format of the
            cyler to the PyProBE format. Units are indicated by an asterisk (*).
    """

    input_data_path: str
    column_dict: Dict[str, str]

    def model_post_init(self, __context: Any) -> None:
        """Post initialization method for the BaseModel."""
        dataframe_list = self.get_dataframe_list(self.input_data_path)
        self._imported_dataframe = self.get_imported_dataframe(dataframe_list)
        self._column_map = self.map_columns(self.column_dict, self._dataframe_columns)

    @staticmethod
    def read_file(filepath: str) -> pl.DataFrame | pl.LazyFrame:
        """Read a battery cycler file into a DataFrame.

        Args:
            filepath (str): The path to the file.

        Returns:
            pl.DataFrame | pl.LazyFrame: The DataFrame.
        """
        file = os.path.basename(filepath)
        file_ext = os.path.splitext(file)[1]
        match file_ext:
            case ".xlsx":
                return pl.read_excel(filepath, engine="calamine", infer_schema_length=0)
            case ".csv":
                return pl.scan_csv(filepath, infer_schema=False)
            case _:
                raise ValueError(f"Unsupported file extension: {file_ext}")

    def get_dataframe_list(
        self, input_data_path: str
    ) -> list[pl.DataFrame | pl.LazyFrame]:
        """Return a list of all the imported dataframes.

        Args:
            input_data_path (str): The path to the input data.

        Returns:
            List[DataFrame]: A list of DataFrames.
        """
        files = glob.glob(input_data_path)
        files.sort()
        list = [self.read_file(file) for file in files]
        all_columns = set([col for df in list for col in df.collect_schema().names()])
        indices_to_remove = []
        for i in range(len(list)):
            if len(list[i].collect_schema().names()) < len(all_columns):
                indices_to_remove.append(i)
                warnings.warn(
                    f"File {os.path.basename(files[i])} has missing columns, "
                    "it has not been read."
                )
                continue
        return [df for i, df in enumerate(list) if i not in indices_to_remove]

    def get_imported_dataframe(
        self, dataframe_list: List[pl.DataFrame]
    ) -> pl.DataFrame:
        """Return a single DataFrame from a list of DataFrames.

        Args:
            dataframe_list: A list of DataFrames.

        Returns:
            DataFrame: A single DataFrame.
        """
        return pl.concat(dataframe_list, how="vertical", rechunk=True)

    @staticmethod
    def match_unit(column_name: str, pattern: str) -> Optional[str]:
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
    def map_columns(
        cls, column_dict: Dict[str, str], dataframe_columns: List[str]
    ) -> Dict[str, Dict[str, str | pl.DataType]]:
        """Map the columns of the imported dataframe to the PyProBE format."""
        column_map: Dict[str, Dict[str, str | pl.DataType]] = {}
        for cycler_format, pyprobe_format in column_dict.items():
            for cycler_column_name in dataframe_columns:
                unit = cls.match_unit(cycler_column_name, cycler_format)
                if unit is not None:
                    quantity = pyprobe_format.replace(" [*]", "")
                    if quantity == "Temperature":
                        if unit != "K":
                            unit = "C"
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

    @property
    def _dataframe_columns(self) -> List[str]:
        """The columns of the DataFrame.

        Returns:
            List[str]: The columns.
        """
        return self._imported_dataframe.collect_schema().names()

    @property
    def pyprobe_dataframe(self) -> pl.DataFrame:
        """The DataFrame containing the required columns.

        Returns:
            pl.DataFrame: The DataFrame.
        """
        required_columns = [
            self.date,
            self.time,
            self.cycle,
            self.step,
            self.event,
            self.current,
            self.voltage,
            self.capacity,
            self.temperature,
        ]
        name_converters = [
            self.convert_names(quantity) for quantity in self._column_map.keys()
        ]
        imported_dataframe = self._imported_dataframe.with_columns(name_converters)
        return imported_dataframe.select(required_columns)

    def convert_names(self, quantity: str) -> pl.Expr:
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
    def date(self) -> pl.Expr:
        """Identify and format the date column.

        Returns:
            pl.Expr: A polars expression for the date column.
        """
        if "Date" in self._column_map.keys():
            return pl.col("Date").str.to_datetime(time_unit="us")
        else:
            return pl.lit(None).alias("Date")

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
        if "Capacity" in self._column_map.keys():
            return Units(
                "Capacity", self._column_map["Capacity"]["Unit"]
            ).to_default_unit()
        else:
            return self.capacity_from_ch_dch

    @property
    def temperature(self) -> pl.Expr:
        """Identify and format the temperature column.

        An optional column, if not found, a column of None values is returned.

        Returns:
            pl.Expr: A polars expression for the temperature column.
        """
        if "Temperature" in self._column_map.keys():
            return Units(
                "Temperature", self._column_map["Temperature"]["Unit"]
            ).to_default_unit()
        else:
            return pl.lit(None).alias("Temperature [C]")

    @property
    def step(self) -> pl.Expr:
        """Identify the step number.

        Returns:
            pl.Expr: A polars expression for the step number.
        """
        return pl.col("Step")

    @property
    def cycle(self) -> pl.Expr:
        """Identify the cycle number.

        Cycles are defined by repetition of steps. They are identified by a decrease
        in the step number.

        Returns:
            pl.Expr: A polars expression for the cycle number.
        """
        return (
            (pl.col("Step").cast(pl.Int64) - pl.col("Step").cast(pl.Int64).shift() < 0)
            .fill_null(strategy="zero")
            .cum_sum()
            .alias("Cycle")
            .cast(pl.Int64)
        )

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
