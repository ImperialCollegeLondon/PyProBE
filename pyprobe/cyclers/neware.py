"""A module to load and process Neware battery cycler data."""

import logging
import os

import polars as pl

from pyprobe.cyclers import basecycler as bc

logger = logging.getLogger(__name__)


class Neware(bc.BaseCycler):
    """A class to load and process Neware battery cycler data."""

    column_importers: list[bc.ColumnMap] = [
        bc.DateTime("Date", "%Y-%m-%d  %H:%M:%S%.f"),
        bc.TimeFromDate("Date", "%Y-%m-%d  %H:%M:%S%.f"),
        bc.CastAndRename("Step", "Step Index", pl.Int64),
        bc.ConvertUnits("Current [A]", "Current(*)"),
        bc.ConvertUnits("Voltage [V]", "Voltage(*)"),
        bc.CapacityFromChDch("Chg. Cap.(*)", "DChg. Cap.(*)"),
        bc.ConvertTemperature("T1(*)"),
        bc.TimeFromDate("Total Time", "%Y-%m-%d  %H:%M:%S%.f"),
        bc.CapacityFromCurrentSign("Capacity(*)", "Current(*)"),
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
