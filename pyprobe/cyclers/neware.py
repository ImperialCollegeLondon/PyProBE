"""A module to load and process Neware battery cycler data."""

import os

import polars as pl
from loguru import logger

from pyprobe.cyclers import column_importers as ci
from pyprobe.cyclers.basecycler import BaseCycler


class Neware(BaseCycler):
    """A class to load and process Neware battery cycler data."""

    column_importers: list[ci.ColumnMap] = [
        ci.DateTime("Date", "%Y-%m-%d  %H:%M:%S%.f"),
        ci.TimeFromDate("Date", "%Y-%m-%d  %H:%M:%S%.f"),
        ci.CastAndRename("Step", "Step Index", pl.UInt64),
        ci.ConvertUnits("Current [A]", "Current(*)"),
        ci.ConvertUnits("Voltage [V]", "Voltage(*)"),
        ci.CapacityFromChDch("Chg. Cap.(*)", "DChg. Cap.(*)"),
        ci.ConvertTemperature("T1(*)"),
        ci.TimeFromDate("Total Time", "%Y-%m-%d  %H:%M:%S%.f"),
        ci.CapacityFromCurrentSign("Capacity(*)", "Current(*)"),
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
        file = os.path.basename(filepath)
        file_ext = os.path.splitext(file)[1]
        match file_ext.lower():
            case ".xlsx":
                dataframe = pl.read_excel(
                    filepath,
                    engine="calamine",
                    infer_schema_length=0,
                    sheet_name="record",
                ).lazy()
            case ".csv":
                dataframe = pl.scan_csv(filepath, infer_schema=False)
            case _:
                error_msg = f"Unsupported file extension: {file_ext}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        return dataframe
