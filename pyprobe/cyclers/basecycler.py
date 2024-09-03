"""A module to load and process battery cycler data."""

import glob
import os
import re
import warnings

# from abc import ABC, abstractmethod
from typing import Any, Dict, List

import polars as pl
from pydantic import BaseModel, Field

from pyprobe.units import Units


class BaseCycler(BaseModel):
    """A class to load and process battery cycler data.

    Args:
        input_data_path (str): The path to the input data.
        common_suffix (str): The part of the filename before an index number,
            when a single procedure is split into multiple files.
        column_name_pattern (str): The regular expression pattern to match the
            column names.
        column_dict (Dict[str, str]): A dictionary mapping the expected columns to
            the actual column names in the data.
    """

    input_data_path: str
    column_name_pattern: str
    column_dict: Dict[str, str]
    common_suffix: str = Field(default="")
    common_prefix: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Post initialization method for the BaseModel."""
        dataframe_list = self.get_dataframe_list(self.input_data_path)
        self._imported_dataframe = self.get_imported_dataframe(dataframe_list)

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
        files = self._sort_files(files)
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

    def _sort_files(self, file_list: List[str]) -> List[str]:
        """Sort a list of files by the integer in the filename.

        Args:
            file_list: The list of files.

        Returns:
            list: The sorted list of files.
        """
        # common first part of file names
        self.common_prefix = os.path.commonprefix(file_list)
        return sorted(file_list, key=self._sort_key)

    def _sort_key(self, filepath: str) -> int:
        """Sort key for the files.

        Args:
            filepath (str): The path to the file.

        Returns:
            int: The integer in the filename.
        """
        # replace common prefix
        stripped_filepath = filepath.replace(self.common_prefix, "")

        if self.common_suffix == "":
            self.common_suffix = os.path.splitext(stripped_filepath)[1]
        # find the index of the common suffix
        suffix_index = stripped_filepath.find(self.common_suffix)

        # if the suffix is found, strip it and everything after it
        if suffix_index != -1:
            stripped_filepath = stripped_filepath[:suffix_index]
        # extract the first number in the filename
        match = re.search(r"\d+", stripped_filepath)
        return int(match.group()) if match else 0

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

    @property
    def required_columns(self) -> Dict[str, pl.Expr]:
        """The required columns for the cycler data.

        Returns:
            Dict[str, pl.Expr]: A dictionary of the required columns.
        """
        return {
            "Date": self.date,
            "Time [s]": self.time,
            "Cycle": self.cycle,
            "Step": self.step,
            "Event": self.event,
            "Current [A]": self.current,
            "Voltage [V]": self.voltage,
            "Capacity [Ah]": self.capacity,
            "Temperature [C]": self.temperature,
        }

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
        return self._imported_dataframe.select(list(self.required_columns.values()))

    @property
    def date(self) -> pl.Expr:
        """Identify and format the date column.

        Returns:
            pl.Expr: A polars expression for the date column.
        """
        if self.column_dict["Date"] in self._dataframe_columns:
            return (
                pl.col(self.column_dict["Date"])
                .cast(str)
                .str.to_datetime(time_unit="us")
                .alias("Date")
            )
        else:
            return pl.lit(None).alias("Date")

    @property
    def time(self) -> pl.Expr:
        """Identify and format the time column.

        Returns:
            pl.Expr: A polars expression for the time column.
        """
        unit = self.search_columns(
            self._dataframe_columns,
            self.column_dict["Time"],
            self.column_name_pattern,
        )
        return pl.col(unit.name).cast(pl.Float64).alias("Time [s]")

    @property
    def step(self) -> pl.Expr:
        """Identify and format the step column."""
        return pl.col(self.column_dict["Step"]).cast(pl.Int64).alias("Step")

    @property
    def current(self) -> pl.Expr:
        """Identify and format the current column.

        Returns:
            pl.Expr: A polars expression for the current column.
        """
        return self.search_columns(
            self._dataframe_columns,
            self.column_dict["Current"],
            self.column_name_pattern,
        ).to_default_name_and_unit()

    @property
    def voltage(self) -> pl.Expr:
        """Identify and format the voltage column.

        Returns:
            pl.Expr: A polars expression for the voltage column.
        """
        return self.search_columns(
            self._dataframe_columns,
            self.column_dict["Voltage"],
            self.column_name_pattern,
        ).to_default_name_and_unit()

    @property
    def charge_capacity(self) -> pl.Expr:
        """Identify and format the charge capacity column.

        Returns:
            pl.Expr: A polars expression for the charge capacity column.
        """
        return self.search_columns(
            self._dataframe_columns,
            self.column_dict["Charge Capacity"],
            self.column_name_pattern,
        ).to_default_unit()

    @property
    def discharge_capacity(self) -> pl.Expr:
        """Identify and format the discharge capacity column.

        Returns:
            pl.Expr: A polars expression for the discharge capacity column.
        """
        return self.search_columns(
            self._dataframe_columns,
            self.column_dict["Discharge Capacity"],
            self.column_name_pattern,
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
        if "Capacity" in self.column_dict:
            return self.search_columns(
                self._dataframe_columns,
                self.column_dict["Capacity"],
                self.column_name_pattern,
            ).to_default_name_and_unit()
        else:
            return self.capacity_from_ch_dch

    @property
    def temperature(self) -> pl.Expr:
        """Identify and format the temperature column.

        An optional column, if not found, a column of None values is returned.

        Returns:
            pl.Expr: A polars expression for the temperature column.
        """
        try:
            return self.search_columns(
                self._dataframe_columns,
                self.column_dict["Temperature"],
                self.column_name_pattern,
            ).to_default_name_and_unit()
        except ValueError:
            return pl.lit(None).alias("Temperature [C]")

    @property
    def cycle(self) -> pl.Expr:
        """Identify the cycle number.

        Cycles are defined by repetition of steps. They are identified by a decrease
        in the step number.

        Returns:
            pl.Expr: A polars expression for the cycle number.
        """
        return (
            (
                pl.col(self.column_dict["Step"]).cast(pl.Int64)
                - pl.col(self.column_dict["Step"]).cast(pl.Int64).shift()
                < 0
            )
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
            (
                pl.col(self.column_dict["Step"]).cast(pl.Int64)
                - pl.col(self.column_dict["Step"]).cast(pl.Int64).shift()
                != 0
            )
            .fill_null(strategy="zero")
            .cum_sum()
            .alias("Event")
            .cast(pl.Int64)
        )

    @staticmethod
    def search_columns(
        columns: List[str],
        search_quantity: str,
        name_pattern: str,
    ) -> "Units":
        """Search for a quantity in the columns of the DataFrame.

        Args:
            columns: The columns to search.
            search_quantity: The quantity to search for.
            name_pattern: The pattern to match the column name.
            default_quantity: The default quantity name.
        """
        for column_name in columns:
            try:
                quantity, _ = Units.get_quantity_and_unit(column_name, name_pattern)
            except ValueError:
                continue

            if quantity == search_quantity:
                return Units(
                    column_name=column_name,
                    name_pattern=name_pattern,
                )
        raise ValueError(f"Quantity {search_quantity} not found in columns.")
