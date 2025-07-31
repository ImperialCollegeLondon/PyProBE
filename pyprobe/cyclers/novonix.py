"""A module to load and process Novonix battery cycler data."""

import polars as pl

from pyprobe.cyclers import column_maps
from pyprobe.cyclers.basecycler import BaseCycler


class Novonix(BaseCycler):
    """A class to load and process Novonix battery cycler data."""

    input_data_path: str

    column_importers: list[column_maps.ColumnMap] = [
        column_maps.DateTimeMap(
            "Date and Time", "%Y-%m-%d %H:%M:%S"
        ),  ##swapped the date formatting around to match the Novonix file format
        column_maps.CastAndRenameMap("Step", "Step Number", pl.UInt64),
        column_maps.ConvertUnitsMap("Time [s]", "Run Time (*)"),
        column_maps.ConvertUnitsMap("Current [A]", "Current (*)"),
        column_maps.ConvertUnitsMap("Voltage [V]", "Potential (*)"),
        column_maps.ConvertUnitsMap("Capacity [Ah]", "Capacity (*)"),
        column_maps.ConvertTemperatureMap("Temperature (*)"),
    ]

    @staticmethod
    def read_file(
        filepath: str,
        header_row_index: int = 0,
    ) -> pl.DataFrame | pl.LazyFrame:
        """Read a Novonix file and return a DataFrame."""
        n_header_lines = 0
        with open(filepath, encoding="utf-8") as file:
            for line in file:
                if line.startswith("[Data]"):
                    n_header_lines += 1
                    break
                n_header_lines += 1
        return BaseCycler.read_file(
            filepath,
            header_row_index=n_header_lines,
        )
