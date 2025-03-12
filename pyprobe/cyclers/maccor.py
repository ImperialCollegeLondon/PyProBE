"""A module to load and process Maccor battery cycler data."""

import polars as pl

from pyprobe.cyclers import column_maps
from pyprobe.cyclers.basecycler import BaseCycler


class MaccorCapacityFromCurrentSign(column_maps.CapacityFromCurrentSignMap):
    """A class to calculate capacity from current and sign columns.

    Specific for maccor data where columns have no units.
    """

    def __init__(self, capacity_column: str, current_column: str) -> None:
        """Initialize the CapacityFromCurrentSignMap object."""
        super().__init__(capacity_column, current_column)

    @property
    def capacity(self) -> pl.Expr:
        """Get the capacity column."""
        return self.get(self.capacity_col).cast(pl.Float64)


class MaccorDateTime(column_maps.ColumnMap):
    """A class to convert a date and time column into a single datetime column.

    Specific for maccor data where the date column is not sampled at the same rate as
    the time column.
    """

    def __init__(
        self,
        date_column: str,
        time_column: str,
        datetime_format: str,
    ) -> None:
        """Initialize the DateTimeMap object."""
        self.pyprobe_name = "Date"
        super().__init__(self.pyprobe_name, [date_column, time_column])
        self.datetime_format = datetime_format
        self.date_column = date_column
        self.time_column = time_column

    @property
    def expr(self) -> pl.Expr:
        """Get the expression to convert the columns."""
        date_col = self.get(self.date_column).str.to_datetime(
            format=self.datetime_format,
            time_unit="us",
        )
        return (
            (self.get(self.time_column).cast(pl.Float64) * 1000000).cast(pl.Duration)
            + date_col.first()
        ).alias(self.pyprobe_name)


class Maccor(BaseCycler):
    """A class to load and process Neware battery cycler data."""

    column_importers: list[column_maps.ColumnMap] = [
        MaccorDateTime("DPT Time", "Test Time (sec)", "%d-%b-%y %I:%M:%S %p"),
        column_maps.CastAndRenameMap("Time [s]", "Test Time (sec)", pl.Float64),
        column_maps.CastAndRenameMap("Step", "Step", pl.UInt64),
        column_maps.CastAndRenameMap("Current [A]", "Current", pl.Float64),
        column_maps.CastAndRenameMap("Voltage [V]", "Voltage", pl.Float64),
        MaccorCapacityFromCurrentSign("Capacity", "Current"),
        column_maps.CastAndRenameMap("Temperature [C]", "Temp 1", pl.Float64),
    ]

    @staticmethod
    def read_file(
        filepath: str,
        header_row_index: int = 0,
    ) -> pl.DataFrame | pl.LazyFrame:
        """Read a battery cycler file into a DataFrame.

        Args:
            filepath: The path to the file.
            header_row_index: The index of the header row.

        Returns:
            pl.DataFrame | pl.LazyFrame: The DataFrame.
        """
        dataframe = pl.scan_csv(filepath, skip_rows=2, infer_schema=False)
        return dataframe
