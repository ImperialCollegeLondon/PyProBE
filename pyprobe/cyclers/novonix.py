"""A module to load and process Biologic battery cycler data."""

from datetime import datetime

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
        n_header_lines = 0
        start_time_line = None
        with open(filepath, encoding="utf-8") as file:
            for line in file:
                if line.startswith("Started:"):
                    start_time_line = line
                if line.startswith("Date and Time"):
                    break
                n_header_lines += 1

        if start_time_line is not None:
            _, value = start_time_line.split(": ")
            start_time = datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S")
            # Optionally use start_time here if needed

        dataframe = pl.scan_csv(
            filepath,
            skip_rows=max(n_header_lines, 0),
            separator=",",
            infer_schema_length=100,  # or infer_schema=True
        )
        return dataframe
