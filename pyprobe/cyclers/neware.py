"""A module to load and process Neware battery cycler data."""


import polars as pl

from pyprobe.cyclers.basecycler import BaseCycler
from pyprobe.units import Units


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
        "Capacity(*)": "Capacity [*]",
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
        if "Date" in self._column_map.keys():
            return (
                (self.date.diff().dt.total_microseconds().cum_sum() / 1e6)
                .fill_null(strategy="zero")
                .alias("Time [s]")
            )
        else:
            return pl.col("Time [s]")

    @property
    def charge_capacity(self) -> pl.Expr:
        """Identify and format the charge capacity column.

        For the Neware cycler, this is either the "Chg. Cap.(*)" column or the
        "Capacity(*)" column when the current is positive.

        Returns:
            pl.Expr: A polars expression for the charge capacity column.
        """
        if "Charge Capacity" in self._column_map.keys():
            return super().charge_capacity
        else:
            current_direction = self.current.sign()
            charge_capacity = (
                Units(
                    "Capacity", self._column_map["Capacity"]["Unit"]
                ).to_default_unit()
                * current_direction.replace(-1, 0).abs()
            )
            return charge_capacity.alias("Charge Capacity [Ah]")

    @property
    def discharge_capacity(self) -> pl.Expr:
        """Identify and format the discharge capacity column.

        For the Neware cycler, this is either the "DChg. Cap.(*)" column or the
        "Capacity(*)" column when the current is negative.

        Returns:
            pl.Expr: A polars expression for the discharge capacity column.
        """
        if "Discharge Capacity" in self._column_map.keys():
            return super().discharge_capacity
        else:
            current_direction = self.current.sign()
            discharge_capacity = (
                Units(
                    "Capacity", self._column_map["Capacity"]["Unit"]
                ).to_default_unit()
                * current_direction.replace(1, 0).abs()
            )
            return discharge_capacity.alias("Discharge Capacity [Ah]")

    @property
    def capacity(self) -> pl.Expr:
        """Identify and format the capacity column.

        For the Neware cycler remove the option to calculate the capacity from a single
        capacity column.

        Returns:
            pl.Expr: A polars expression for the capacity column.
        """
        return self.capacity_from_ch_dch
