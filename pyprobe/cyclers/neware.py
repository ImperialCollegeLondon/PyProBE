"""A module to load and process Neware battery cycler data."""

import logging
import os

import polars as pl

from pyprobe.cyclers.basecycler import BaseCycler
from pyprobe.units import Units

logger = logging.getLogger(__name__)


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
    def _convert_neware_time_format(
        data: pl.DataFrame | pl.LazyFrame, column: str
    ) -> pl.DataFrame | pl.LazyFrame:
        """Method to convert the Neware time columns to seconds.

        Neware time columns can be in the format "HH:MM:SS" or "YYYY-MM-DD HH:MM:SS".
        This method converts the time columns to seconds.

        Args:
            data: The DataFrame.
            column: The column name.
        """
        has_dates = data.select(
            pl.col(column).str.contains(r"\d{4}-\d{2}-\d{2}").any()
        ).item()
        if has_dates:
            data = data.with_columns(
                pl.col(column)
                .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f")
                .cast(pl.Float64)
                .alias(column)
            )
            data = data.with_columns(pl.col(column) / 1e6)
        else:
            data = data.with_columns(pl.col(column).str.split(":"))
            data = data.with_columns(
                [
                    (pl.col(column).list.get(0).cast(pl.Float64)).alias("hours"),
                    (pl.col(column).list.get(1).cast(pl.Float64)).alias("minutes"),
                    (pl.col(column).list.get(2).cast(pl.Float64)).alias("seconds"),
                ]
            )
            data = data.with_columns(
                (
                    pl.col("hours") * 3600 + pl.col("minutes") * 60 + pl.col("seconds")
                ).alias(column)
            )
            data = data.drop(["hours", "minutes", "seconds"])
        data = data.with_columns(pl.col(column) - pl.col(column).first().alias(column))
        return data

    @staticmethod
    def read_file(
        filepath: str, header_row_index: int = 0
    ) -> pl.DataFrame | pl.LazyFrame:
        """Read a battery cycler file into a DataFrame.

        Args:
            filepath: The path to the file.
            header_row_index: The index of the header row.

        Returns:
            pl.DataFrame | pl.LazyFrame: The DataFrame.
        """
        file = os.path.basename(filepath)
        file_ext = os.path.splitext(file)[1]
        match file_ext.lower():
            case ".xlsx":
                dataframe = pl.read_excel(
                    filepath,
                    engine="calamine",
                    infer_schema_length=0,
                    sheet_name="record",
                )
            case ".csv":
                dataframe = pl.scan_csv(filepath, infer_schema=False)
            case _:
                error_msg = f"Unsupported file extension: {file_ext}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        dataframe = BaseCycler.read_file(filepath)
        if "Total Time" in dataframe.collect_schema().names():
            dataframe = Neware._convert_neware_time_format(dataframe, "Total Time")
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
