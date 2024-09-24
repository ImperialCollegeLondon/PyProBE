"""A module to load and process Arbin battery cycler data."""


from typing import Optional

import polars as pl

from pyprobe.cyclers.basecycler import BaseCycler


class Arbin(BaseCycler):
    """A class to load and process Neware battery cycler data."""

    input_data_path: str
    column_dict: dict[str, str] = {
        "Date Time": "Date",
        "Test Time (*)": "Time [*]",
        "Step Index": "Step",
        "Current (*)": "Current [*]",
        "Voltage (*)": "Voltage [*]",
        "Charge Capacity (*)": "Charge Capacity [*]",
        "Discharge Capacity (*)": "Discharge Capacity [*]",
        "Aux_Temperature_1 (*)": "Temperature [*]",
    }

    @property
    def date(self) -> Optional[pl.Expr]:
        """Identify and format the date column.

        Returns:
            Optional[pl.Expr]: A polars expression for the date column.
        """
        if "Date" in self._column_map.keys():
            return pl.col("Date").str.to_datetime(
                format="%m/%d/%Y %H:%M:%S%.f", time_unit="us"
            )
        else:
            return None
