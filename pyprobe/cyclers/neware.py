"""A module to load and process Neware battery cycler data."""


import polars as pl

from pyprobe.cyclers.basecycler import BaseCycler


class Neware(BaseCycler):
    """A class to load and process Neware battery cycler data."""

    input_data_path: str
    column_dict: dict[str, str] = {
        "Date": "Date",
        "Step Index": "Step",
        "Current(*)": "Current [*]",
        "Voltage(*)": "Voltage [*]",
        "Chg. Cap.(*)": "Charge Capacity [*]",
        "DChg. Cap.(*)": "Discharge Capacity [*]",
        "T1(*)": "Temperature [*]",
        "Total Time": "Time [*]",
    }

    @staticmethod
    def read_file(filepath: str) -> pl.DataFrame | pl.LazyFrame:
        """Read a battery cycler file into a DataFrame.

        Args:
            filepath (str): The path to the file.

        Returns:
            pl.DataFrame | pl.LazyFrame: The DataFrame.
        """
        dataframe = BaseCycler.read_file(filepath)
        print(dataframe)
        if "Time" in dataframe.collect_schema().names():
            dataframe = dataframe.with_columns(
                pl.col("Time")
                .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f")
                .cast(pl.Float64)
                .alias("Time")
            )
            dataframe = dataframe.with_columns(
                pl.col("Time") - pl.col("Time").first().alias("Time")
            )
            dataframe = dataframe.with_columns(pl.col("Time") / 1e6)
        if "Total Time" in dataframe.collect_schema().names():
            dataframe = dataframe.with_columns(
                pl.col("Total Time")
                .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f")
                .cast(pl.Float64)
                .alias("Total Time")
            )
            dataframe = dataframe.with_columns(
                pl.col("Total Time") - pl.col("Total Time").first().alias("Total Time")
            )
            dataframe = dataframe.with_columns(pl.col("Total Time") / 1e6)
        return dataframe

    @property
    def time(self) -> pl.Expr:
        """Identify and format the time column.

        For Neware data, by default the time column is calculated from the "Date"
        column if it exists.

        Returns:
            pl.Expr: A polars expression for the time column.
        """
        if self.date is not None:
            return (
                (self.date.diff().dt.total_microseconds().cum_sum() / 1e6)
                .fill_null(strategy="zero")
                .alias("Time [s]")
            )
        else:
            return pl.col("Time [s]")
            # return Units("Time", self._column_map["Time"]["Unit"]).to_default_unit()
