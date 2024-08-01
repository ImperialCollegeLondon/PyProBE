"""A module to load and process Neware battery cycler data."""


import os

import polars as pl

from pyprobe.cyclers.basecycler import BaseCycler


class Neware(BaseCycler):
    """A class to load and process Neware battery cycler data.

    Args:
        input_data_path: The path to the input data.
    """

    input_data_path: str
    common_suffix: str = ""
    column_name_pattern: str = r"(.+)\((.+)\)"
    column_dict: dict[str, str] = {
        "Date": "Date",
        "Step": "Step Index",
        "Current": "Current",
        "Voltage": "Voltage",
        "Charge Capacity": "Chg. Cap.",
        "Discharge Capacity": "DChg. Cap.",
    }

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
                return pl.read_excel(filepath, engine="calamine")
            case ".csv":
                return pl.read_csv(filepath)
            case _:
                raise ValueError(f"Unsupported file extension: {file_ext}")

    @property
    def time(self) -> pl.Expr:
        """Identify and format the time column.

        Returns:
            pl.Expr: A polars expression for the time column.
        """
        return (
            (
                pl.col(self.column_dict["Date"])
                .diff()
                .dt.total_microseconds()
                .cum_sum()
                / 1e6
            )
            .fill_null(strategy="zero")
            .alias("Time [s]")
        )
