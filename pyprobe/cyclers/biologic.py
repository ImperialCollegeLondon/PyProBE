"""A module to load and process Biologic battery cycler data."""


import glob
import re
from datetime import datetime
from typing import List

import polars as pl

from pyprobe.cyclers.basecycler import BaseCycler
from pyprobe.unitconverter import UnitConverter


class Biologic(BaseCycler):
    """A class to load and process Biologic battery cycler data."""

    def __init__(self, input_data_path: str) -> None:
        """Create a Biologic cycler object.

        Args:
            input_data_path: The path to the input data.
        """
        self.input_data_path = input_data_path

    @property
    def imported_dataframe(self) -> pl.DataFrame:
        """The imported DataFrame."""
        imported_dataframe = self.get_cycle_and_event(self.processed_dataframe)
        return imported_dataframe.select(self.required_columns)

    @staticmethod
    def read_file(filepath: str) -> pl.DataFrame:
        """Read a battery cycler file into a DataFrame.

        Args:
            filepath: The path to the file.
        """
        with open(filepath, "r", encoding="iso-8859-1") as file:
            file.readline()  # Skip the first line
            second_line = file.readline().strip()  # Read the second line
        _, value = second_line.split(":")
        n_header_lines = int(value.strip())

        with open(filepath, "r", encoding="iso-8859-1") as file:
            for i in range(n_header_lines):
                line = file.readline()
                if "Acquisition started on" in line:
                    start_time_line = line
                    break
        _, value = start_time_line.split(" : ")
        start_time = datetime.strptime(value.strip(), "%m/%d/%Y %H:%M:%S.%f")

        columns_to_read = ["time", "Ns", "I", "Ecell", "Q charge", "Q discharge"]

        all_columns = pl.scan_csv(
            filepath, skip_rows=n_header_lines - 1, separator="\t"
        ).columns
        selected_columns = [
            col for col in all_columns if any(sub in col for sub in columns_to_read)
        ]
        dataframe = pl.read_csv(
            filepath,
            skip_rows=n_header_lines - 1,
            separator="\t",
            columns=selected_columns,
        )

        start = pl.DataFrame({"start": [start_time]})
        dataframe = dataframe.with_columns(
            (pl.col("time/s") * 1000000 + start).cast(pl.Datetime).alias("Date")
        )
        return dataframe

    @classmethod
    def sort_files(cls, file_list: List[str]) -> List[str]:
        """Sort a list of files by the integer in the filename.

        Args:
            file_list: The list of files.

        Returns:
            list: The sorted list of files.
        """
        return sorted(file_list, key=cls.sort_key)

    @staticmethod
    def sort_key(filepath: str) -> int:
        """Sort key for the files.

        Args:
            filepath (str): The path to the file.

        Returns:
            int: The integer in the filename.
        """
        match = re.search(r"\d+_MB", filepath)
        return int(match.group()[:-3]) if match else 0

    @property
    def raw_dataframe(self) -> pl.DataFrame:
        """Read a battery cycler file into a DataFrame.

        Args:
            filepath: The path to the file.
        """
        files = glob.glob(self.input_data_path)
        files = self.sort_files(files)
        dataframes = [self.read_file(file) for file in files]

        for i in range(1, len(dataframes)):
            dataframes[i] = dataframes[i].with_columns(
                pl.col("Ns") + dataframes[i - 1]["Ns"].max() + 1
            )
        return pl.concat(dataframes, how="vertical")

    @property
    def processed_dataframe(self) -> pl.DataFrame:
        """Process a DataFrame from battery cycler data.

        Args:
            dataframe: The DataFrame to process.

        Returns:
            pl.DataFrame: The dataframe in PyProBE format.
        """
        dataframe = self.raw_dataframe
        columns = dataframe.columns
        time = pl.col("time/s").alias("Time [s]")

        # Cycle and step
        step = (pl.col("Ns") + 1).alias("Step")

        # Measured data
        column_name_pattern = r"(.+)/(.+)"
        current = UnitConverter.search_columns(
            columns, "I", column_name_pattern, "Current"
        ).to_default()
        voltage = UnitConverter.search_columns(
            columns, "Ecell", column_name_pattern, "Voltage"
        ).to_default()

        make_charge_capacity = UnitConverter.search_columns(
            columns, "Q charge", column_name_pattern, "Capacity"
        ).to_default(keep_name=True)
        make_discharge_capacity = UnitConverter.search_columns(
            columns, "Q discharge", column_name_pattern, "Capacity"
        ).to_default(keep_name=True)

        dataframe = dataframe.with_columns(time, step, current, voltage)

        dataframe = dataframe.with_columns(
            make_charge_capacity, make_discharge_capacity
        )

        diff_charge_capacity = (
            pl.col("Q charge [Ah]")
            .diff()
            .clip(lower_bound=0)
            .fill_null(strategy="zero")
        )
        diff_discharge_capacity = (
            pl.col("Q discharge [Ah]")
            .diff()
            .clip(lower_bound=0)
            .fill_null(strategy="zero")
        )
        make_capacity = (
            (diff_charge_capacity - diff_discharge_capacity).cum_sum()
            + pl.col("Q charge [Ah]").max()
        ).alias("Capacity [Ah]")

        dataframe = dataframe.with_columns(make_capacity)
        return dataframe
