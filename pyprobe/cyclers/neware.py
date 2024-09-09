"""A module to load and process Neware battery cycler data."""


import polars as pl

from pyprobe.cyclers.basecycler import BaseCycler


class Neware(BaseCycler):
    """A class to load and process Neware battery cycler data.

    Args:
        input_data_path: The path to the input data.
    """

    input_data_path: str
    column_dict: dict[str, str] = {
        "Date": "Date",
        "Step Index": "Step",
        "Current(*)": "Current [*]",
        "Voltage(*)": "Voltage [*]",
        "Chg. Cap.(*)": "Charge Capacity [*]",
        "DChg. Cap.(*)": "Discharge Capacity [*]",
        "T1(*)": "Temperature [*]",
    }

    @property
    def time(self) -> pl.Expr:
        """Identify and format the time column.

        Returns:
            pl.Expr: A polars expression for the time column.
        """
        if (
            self._imported_dataframe.dtypes[
                self._imported_dataframe.columns.index("Date")
            ]
            != pl.Datetime
        ):
            date = pl.col("Date").str.to_datetime().alias("Date")
        else:
            date = pl.col("Date")

        return (
            (date.diff().dt.total_microseconds().cum_sum() / 1e6)
            .fill_null(strategy="zero")
            .alias("Time [s]")
        )
