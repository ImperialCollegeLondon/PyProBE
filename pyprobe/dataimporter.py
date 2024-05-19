"""A module to load and process battery cycler data."""
import os

import polars as pl

from pyprobe.cyclers import cycler_dict
from pyprobe.cyclers.neware import neware


class DataImporter:
    """A battery cycler object."""

    required_columns = [
        "Date",
        "Time [s]",
        "Cycle",
        "Step",
        "Current [A]",
        "Voltage [V]",
    ]
    optional_columns = ["Capacity [Ah]", "Temperature [K]"]

    def __init__(self, cycler: str):
        """Initialise a DataImporter object.

        Args:
            cycler: the cycler that the data is from.
        """
        self.cycler = cycler_dict[cycler]

    def read_file(self, filepath: str) -> pl.DataFrame:
        """Read a battery cycler file into a DataFrame.

        Args:
            filepath: The path to the file.
        """
        self.file = os.path.basename(filepath)
        self.path = os.path.dirname(filepath)
        self.file_name = os.path.splitext(self.file)[0]
        file_ext = os.path.splitext(self.file)[1]
        match file_ext:
            case ".xlsx":
                return pl.read_excel(filepath, engine="calamine")
            case ".csv":
                return pl.read_csv(filepath)
            case _:
                raise ValueError(f"Unsupported file extension: {file_ext}")

    def process_dataframe(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        """Process a DataFrame from battery cycler data.

        Args:
            dataframe: The DataFrame to process.

        Returns:
            pl.DataFrame: The dataframe in PyProBE format.
        """
        dataframe = neware(dataframe)
        return dataframe.select(
            [
                "Date",
                "Time [s]",
                "Cycle",
                "Step",
                "Current [A]",
                "Voltage [V]",
                "Capacity [Ah]",
            ]
        )
