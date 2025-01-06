"""A module to load and process Arbin battery cycler data."""

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
    datetime_format: str = "%m/%d/%Y %H:%M:%S%.f"
