"""A module to load and process Maccor battery cycler data."""

import polars as pl

from pyprobe.cyclers.basecycler import BaseCycler


class Maccor(BaseCycler):
    """A class to load and process Neware battery cycler data."""

    input_data_path: str
    column_dict: dict[str, str] = {
        "DPT Time": "Date",
        "Test Time (sec)": "Time [*]",
        "Step": "Step",
        "Current": "Current [*]",
        "Voltage": "Voltage [*]",
        "Capacity": "Capacity [*]",
        "Temp 1": "Temperature [*]",
    }
    datetime_format: str = "%d-%b-%y %I:%M:%S %p"

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
        dataframe = pl.scan_csv(filepath, skip_rows=2, infer_schema=False)
        return dataframe

    @property
    def date(self) -> pl.Expr:
        """Identify and format the date column.

        For the Maccor cycler, this takes the first date in the file and adds the time
        column to it.

        Returns:
            pl.Expr: A polars expression for the date column.
        """
        if "Date" in self._column_map.keys():
            date_col = pl.col("Date").str.to_datetime(
                format=self.datetime_format, time_unit="us"
            )
            return (
                (pl.col("Time [s]").cast(pl.Float64) * 1000000).cast(pl.Duration)
                + date_col.first()
            ).alias("Date")
        else:
            return None

    @property
    def charge_capacity(self) -> pl.Expr:
        """Identify and format the charge capacity column.

        For the Maccor cycler, this is the capacity column when the current is positive.

        Returns:
            pl.Expr: A polars expression for the charge capacity column.
        """
        current_direction = self.current.sign()
        charge_capacity = pl.col("Capacity [Ah]") * current_direction.replace(-1, 0)
        return charge_capacity.alias("Charge Capacity [Ah]")

    @property
    def discharge_capacity(self) -> pl.Expr:
        """Identify and format the discharge capacity column.

        For the Maccor cycler, this is the capacity column when the current is negative.

        Returns:
            pl.Expr: A polars expression for the discharge capacity column.
        """
        current_direction = self.current.sign()
        charge_capacity = (
            pl.col("Capacity [Ah]") * current_direction.replace(1, 0).abs()
        )
        return charge_capacity.alias("Discharge Capacity [Ah]")

    @property
    def capacity(self) -> pl.Expr:
        """Identify and format the capacity column.

        For the Maccor cycler remove the option to calculate the capacity from a single
        capacity column.

        Returns:
            pl.Expr: A polars expression for the capacity column.
        """
        return self.capacity_from_ch_dch
