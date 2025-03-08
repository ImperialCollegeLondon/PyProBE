"""A module to load and process Arbin battery cycler data."""

import polars as pl

from pyprobe.cyclers import column_maps
from pyprobe.cyclers.basecycler import BaseCycler


class Arbin(BaseCycler):
    """A class to load and process Neware battery cycler data."""

    column_importers: list[column_maps.ColumnMap] = [
        column_maps.DateTime("Date Time", "%m/%d/%Y %H:%M:%S%.f"),
        column_maps.CastAndRename("Step", "Step Index", pl.UInt64),
        column_maps.ConvertUnits("Time [s]", "Test Time (*)"),
        column_maps.ConvertUnits("Current [A]", "Current (*)"),
        column_maps.ConvertUnits("Voltage [V]", "Voltage (*)"),
        column_maps.CapacityFromChDch("Charge Capacity (*)", "Discharge Capacity (*)"),
        column_maps.ConvertTemperature("Aux_Temperature_1 (*)"),
    ]
