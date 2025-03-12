"""A module to load and process Arbin battery cycler data."""

import polars as pl

from pyprobe.cyclers import column_maps
from pyprobe.cyclers.basecycler import BaseCycler


class Arbin(BaseCycler):
    """A class to load and process Neware battery cycler data."""

    column_importers: list[column_maps.ColumnMap] = [
        column_maps.DateTimeMap("Date Time", "%m/%d/%Y %H:%M:%S%.f"),
        column_maps.CastAndRenameMap("Step", "Step Index", pl.UInt64),
        column_maps.ConvertUnitsMap("Time [s]", "Test Time (*)"),
        column_maps.ConvertUnitsMap("Current [A]", "Current (*)"),
        column_maps.ConvertUnitsMap("Voltage [V]", "Voltage (*)"),
        column_maps.CapacityFromChDchMap(
            "Charge Capacity (*)", "Discharge Capacity (*)"
        ),
        column_maps.ConvertTemperatureMap("Aux_Temperature_1 (*)"),
    ]
