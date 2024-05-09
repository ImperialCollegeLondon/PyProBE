"""A module for unit conversion and zero referencing of data columns."""

import re

import polars as pl


class UnitInfo:
    """A class for storing unit conversion information for a quantity."""

    def __init__(
        self, units: list[str], scale_factor: list[float], zero_reference: bool
    ):
        """Initializes the UnitInfo class.

        Args:
            units (list[str]): A list of units for the quantity.
            scale_factor (list[float]): A list of scale factors for the units.
            zero_reference (bool):
                A boolean indicating if the quantity should be zero referenced.
        """
        self.units = units
        self.scale_factor = scale_factor
        self.zero_reference = zero_reference


class Units:
    """A class for unit conversion and zero referencing of data columns.

    Attributes:
        unit_dict (dict):
            A dictionary containing the unit conversion information for the
            different quantities.
    """

    unit_dict: dict[str, UnitInfo] = {
        "Current": UnitInfo(["A", "mA"], [1, 1000], False),
        "Voltage": UnitInfo(["V", "mV"], [1, 1000], False),
        "Capacity": UnitInfo(["Ah", "mAh"], [1, 1000], True),
        "Time": UnitInfo(["s", "min", "hr"], [1, 1 / 60, 1 / 3600], True),
    }

    @staticmethod
    def extract_quantity_and_unit(string: str) -> tuple[str, str | None]:
        """Extracts the quantity and unit from a string.

        Args:
            string (str): A string containing the quantity and unit.
        """
        match = re.search(r"\[(.*?)\]", string)
        if match:
            unit = match.group(1)
            quantity = string.replace(f"[{unit}]", "").strip()
        else:
            quantity = string
            unit = None
        return quantity, unit

    @classmethod
    def convert_units(cls, column: str) -> list[pl.Expr] | None:
        """Return a list of polars instructions to return multiple units.

        Args:
            column (str): The column to convert units of.

        Returns:
            list[pl.Expr]: A list of polars instructions to calculate
                the column in different units.
        """
        quantity, unit_from = cls.extract_quantity_and_unit(column)
        if quantity in cls.unit_dict.keys():
            polars_instruction_list = []
            units = cls.unit_dict[quantity].units
            for unit_to in units:
                if unit_to != unit_from:
                    scale_factor = cls.unit_dict[quantity].scale_factor[
                        cls.unit_dict[quantity].units.index(unit_to)
                    ]
                    polars_instruction_list.append(
                        (pl.col(column) * scale_factor).alias(f"{quantity} [{unit_to}]")
                    )
            return polars_instruction_list
        else:
            return None

    @classmethod
    def set_zero(cls, column: str) -> list[pl.Expr] | None:
        """Return a list of polars instructions to zero reference the column.

        Args:
            column (str): The column to zero reference.

        Returns:
            list[pl.Expr]: A list of polars instructions to zero reference the column.
        """
        quantity, _ = cls.extract_quantity_and_unit(column)
        if quantity in cls.unit_dict.keys():
            if cls.unit_dict[quantity].zero_reference is True:
                return [pl.col(column) - pl.col(column).first()]
            else:
                return None
        else:
            return None
