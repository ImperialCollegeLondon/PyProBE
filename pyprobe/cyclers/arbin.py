"""A module to load and process Arbin battery cycler data."""

import polars as pl

from pyprobe.cyclers import basecycler as bc


class Arbin(bc.BaseCycler):
    """A class to load and process Neware battery cycler data."""

    column_importers: list[bc.ColumnMap] = [
        bc.DateTime("Date Time", "%m/%d/%Y %H:%M:%S%.f"),
        bc.CastAndRename("Step", "Step Index", pl.Int64),
        bc.ConvertUnits("Time [s]", "Test Time (*)"),
        bc.ConvertUnits("Current [A]", "Current (*)"),
        bc.ConvertUnits("Voltage [V]", "Voltage (*)"),
        bc.CapacityFromChDch("Charge Capacity (*)", "Discharge Capacity (*)"),
        bc.ConvertTemperature("Aux_Temperature_1 (*)"),
    ]
