"""A module to load and process battery cycler data."""

import glob
import os
import re
import warnings
from typing import List

import polars as pl


class BaseCycler:
    """A class to load and process battery cycler data."""

    required_columns = [
        "Date",
        "Time [s]",
        "Cycle",
        "Step",
        "Event",
        "Current [A]",
        "Voltage [V]",
        "Capacity [Ah]",
    ]

    def __init__(self, input_data_path: str, common_suffix: str) -> None:
        """Create a cycler object.

        Args:
            input_data_path (str): The path to the input data.
            common_suffix (str): The part of the filename before an index number,
                when a single procedure is split into multiple files.
        """
        self.input_data_path = input_data_path
        self.common_suffix = common_suffix

    @property
    def processed_dataframe(self) -> pl.DataFrame:
        """Process a DataFrame from battery cycler data.

        Returns:
            pl.DataFrame: The dataframe in PyProBE format.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    def read_file(filepath: str) -> pl.DataFrame:
        """Read a battery cycler file into a DataFrame.

        Args:
            filepath (str): The path to the file.

        Returns:
            pl.DataFrame: The DataFrame.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    def get_cycle_and_event(dataframe: pl.DataFrame) -> pl.DataFrame:
        """Get the step and event columns from a DataFrame.

        Args:
            dataframe: The DataFrame to process.

        Returns:
            DataFrame: The DataFrame with the step and event columns.
        """
        cycle = (
            (pl.col("Step") - pl.col("Step").shift() < 0)
            .fill_null(strategy="zero")
            .cum_sum()
            .alias("Cycle")
            .cast(pl.Int64)
        )

        event = (
            (pl.col("Step") - pl.col("Step").shift() != 0)
            .fill_null(strategy="zero")
            .cum_sum()
            .alias("Event")
            .cast(pl.Int64)
        )
        return dataframe.with_columns(cycle, event)

    @property
    def imported_dataframe(self) -> pl.DataFrame:
        """The imported DataFrame."""
        imported_dataframe = self.get_cycle_and_event(self.processed_dataframe)
        return imported_dataframe.select(self.required_columns)

    @property
    def dataframe_list(self) -> list[pl.DataFrame]:
        """Return a list of all the imported dataframes.

        Returns:
            List[DataFrame]: A list of DataFrames.
        """
        files = glob.glob(self.input_data_path)
        files = self.sort_files(files)
        list = [self.read_file(file) for file in files]
        all_columns = set([col for df in list for col in df.columns])
        indices_to_remove = []
        for i in range(len(list)):
            if len(list[i].columns) < len(all_columns):
                indices_to_remove.append(i)
                warnings.warn(
                    f"File {os.path.basename(files[i])} has missing columns, "
                    "it has not been read."
                )
                continue
        return [df for i, df in enumerate(list) if i not in indices_to_remove]

    @property
    def raw_dataframe(self) -> pl.DataFrame:
        """Read a battery cycler file into a DataFrame.

        Args:
            filepath (str): The path to the file.
        """
        return pl.concat(self.dataframe_list, how="vertical", rechunk=True)

    def sort_files(self, file_list: List[str]) -> List[str]:
        """Sort a list of files by the integer in the filename.

        Args:
            file_list: The list of files.

        Returns:
            list: The sorted list of files.
        """
        # common first part of file names
        self.common_prefix = os.path.commonprefix(file_list)
        return sorted(file_list, key=self.sort_key)

    def sort_key(self, filepath: str) -> int:
        """Sort key for the files.

        Args:
            filepath (str): The path to the file.

        Returns:
            int: The integer in the filename.
        """
        # replace common prefix
        stripped_filepath = filepath.replace(self.common_prefix, "")

        # find the index of the common suffix
        suffix_index = stripped_filepath.find(self.common_suffix)

        # if the suffix is found, strip it and everything after it
        if suffix_index != -1:
            stripped_filepath = stripped_filepath[:suffix_index]
        # extract the first number in the filename
        match = re.search(r"\d+", stripped_filepath)
        return int(match.group()) if match else 0
