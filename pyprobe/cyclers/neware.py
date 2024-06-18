"""A module to load and process Neware battery cycler data."""


import os

import polars as pl

from pyprobe.cyclers.basecycler import BaseCycler
from pyprobe.unitconverter import UnitConverter


class Neware(BaseCycler):
    """A class to load and process Neware battery cycler data."""

    def __init__(self, input_data_path: str) -> None:
        """Create a Biologic cycler object.

        Args:
            input_data_path: The path to the input data.
        """
        file_ext = os.path.splitext(input_data_path)[1]
        super().__init__(input_data_path, common_suffix=file_ext)

    @staticmethod
    def read_file(filepath: str) -> pl.DataFrame:
        """Read a battery cycler file into a DataFrame.

        Args:
            filepath (str): The path to the file.

        Returns:
            pl.DataFrame: The DataFrame.
        """
        file = os.path.basename(filepath)
        file_ext = os.path.splitext(file)[1]
        match file_ext:
            case ".xlsx":
                return pl.read_excel(filepath, engine="calamine")
            case ".csv":
                return pl.read_csv(filepath)
            case _:
                raise ValueError(f"Unsupported file extension: {file_ext}")

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

        if dataframe.dtypes[dataframe.columns.index("Date")] != pl.Datetime:
            date = pl.col("Date").str.to_datetime().alias("Date")
            dataframe = dataframe.with_columns(date)

        # Time
        time = (
            (pl.col("Date").diff().dt.total_microseconds().cum_sum() / 1e6)
            .fill_null(strategy="zero")
            .alias("Time [s]")
        )

        # Cycle and step
        step = pl.col("Step Index").alias("Step")

        # Measured data
        column_name_pattern = r"(.+)\((.+)\)"
        current = UnitConverter.search_columns(
            columns, "Current", column_name_pattern, "Current"
        ).to_default()
        voltage = UnitConverter.search_columns(
            columns, "Voltage", column_name_pattern, "Voltage"
        ).to_default()

        dataframe = dataframe.with_columns(time, step, current, voltage)

        make_charge_capacity = UnitConverter.search_columns(
            columns, "Chg. Cap.", column_name_pattern, "Capacity"
        ).to_default(keep_name=True)
        make_discharge_capacity = UnitConverter.search_columns(
            columns, "DChg. Cap.", column_name_pattern, "Capacity"
        ).to_default(keep_name=True)

        dataframe = dataframe.with_columns(
            make_charge_capacity, make_discharge_capacity
        )

        diff_charge_capacity = (
            pl.col("Chg. Cap. [Ah]")
            .diff()
            .clip(lower_bound=0)
            .fill_null(strategy="zero")
        )
        diff_discharge_capacity = (
            pl.col("DChg. Cap. [Ah]")
            .diff()
            .clip(lower_bound=0)
            .fill_null(strategy="zero")
        )
        make_capacity = (
            (diff_charge_capacity - diff_discharge_capacity).cum_sum()
            + pl.col("Chg. Cap. [Ah]").max()
        ).alias("Capacity [Ah]")

        dataframe = dataframe.with_columns(make_capacity)
        return dataframe
