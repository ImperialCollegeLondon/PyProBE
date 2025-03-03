"""A module to load and process Arbin battery cycler data."""

import polars as pl

from pyprobe.cyclers import column_importers as ci
from pyprobe.cyclers.basecycler import BaseCycler


class Arbin(BaseCycler):
    """A class to load and process Neware battery cycler data."""

    column_importers: list[ci.ColumnMap] = [
        ci.DateTime("Date Time", "%m/%d/%Y %H:%M:%S%.f"),
        ci.CastAndRename("Step", "Step Index", pl.UInt64),
        ci.ConvertUnits("Time [s]", "Test Time (*)"),
        ci.ConvertUnits("Current [A]", "Current (*)"),
        ci.ConvertUnits("Voltage [V]", "Voltage (*)"),
        ci.CapacityFromChDch("Charge Capacity (*)", "Discharge Capacity (*)"),
        ci.ConvertTemperature("Aux_Temperature_1 (*)"),
    ]
