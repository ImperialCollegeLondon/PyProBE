"""A module to load and process Basytec battery cycler data."""

from datetime import datetime

import polars as pl

from pyprobe.cyclers import basecycler as bc


class Basytec(bc.BaseCycler):
    """A class to load and process Basytec battery cycler data."""

    column_importers: list[bc.ColumnMap] = [
        bc.DateTime("Date", "%Y-%m-%d %H:%M:%S%.f"),
        bc.CastAndRename("Step", "Line", pl.Int64),
        bc.ConvertUnits("Time [s]", "~Time[*]"),
        bc.ConvertUnits("Current [A]", "I[*]"),
        bc.ConvertUnits("Voltage [V]", "U[*]"),
        bc.ConvertUnits("Capacity [Ah]", "Ah[*]"),
        bc.ConvertTemperature("T1[*]"),
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
        n_header_lines = 0
        with open(filepath, encoding="utf-8") as file:
            for line in file:
                if line.startswith("~"):
                    n_header_lines += 1
                    if line.startswith("~Start of Test"):
                        start_time_line = line

        _, value = start_time_line.split(": ")
        start_time = datetime.strptime(value.strip(), "%d.%m.%Y %H:%M:%S")

        dataframe = pl.scan_csv(
            filepath,
            skip_rows=n_header_lines - 1,
            separator="\t",
            infer_schema=False,
        )

        dataframe = dataframe.with_columns(
            (
                (pl.col("~Time[s]").cast(pl.Float64) * 1000000).cast(pl.Duration)
                + pl.lit(start_time)
            )
            .cast(str)
            .alias("Date"),
        )
        return dataframe
