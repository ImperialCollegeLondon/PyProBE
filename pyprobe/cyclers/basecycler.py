"""A module to load and process battery cycler data."""
from typing import Callable

import polars as pl


class BaseCycler:
    """A class to load and process battery cycler data."""

    required_columns = [
        "Date",
        "Time [s]",
        "Cycle",
        "Step",
        "Current [A]",
        "Voltage [V]",
        "Capacity [Ah]",
    ]

    def __init__(
        self,
        reader: Callable[[str], pl.DataFrame],
        processor: Callable[[pl.DataFrame], pl.DataFrame],
    ):
        """Create a BaseCycler object.

        Args:
            reader (Callable): A function to read the data file.
            processor (Callable): A function to process the data.
        """
        self.reader = reader
        self.processor = processor

    def read_file(self, filename: str) -> pl.DataFrame:
        """Read a battery cycler file into a DataFrame.

        Args:
            filename: The path to the file.

        Returns:
            DataFrame: The DataFrame containing the cycler data.
        """
        return self.reader(filename)

    def process_dataframe(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        """Process a DataFrame into PyProBE format.

        Args:
            dataframe: The DataFrame to process.

        Returns:
            DataFrame: The processed DataFrame.
        """
        return self.processor(dataframe).select(self.required_columns)
