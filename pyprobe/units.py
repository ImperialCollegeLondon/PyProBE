"""A module for unit conversion and zero referencing of data columns."""

import re
from typing import Any, Optional, Tuple

import polars as pl


class Units:
    """A class to store unit conversion information about columns."""

    prefix_dict = {"m": 1e-3, "µ": 1e-6, "n": 1e-9, "p": 1e-12, "k": 1e3, "M": 1e6}
    time_unit_dict = {"s": 1.0, "min": 60.0, "hr": 3600.0}
    unit_dict = {
        "A": "Current",
        "V": "Voltage",
        "Ah": "Capacity",
        "A.h": "Capacity",
        "s": "Time",
    }

    def __init__(
        self,
        column_name: str,
        name_pattern: str = r"([\w\s]+?)\s*\[(\w+)\]",
    ) -> None:
        """Initialize the Units object."""
        self.name = column_name
        self.input_quantity, self.input_unit = self.get_quantity_and_unit(
            column_name, name_pattern
        )
        self.prefix, self.default_unit = self._get_default_unit(self.input_unit)
        self.default_quantity = self._get_default_quantity(self.default_unit)
        if self.default_quantity == "Time":
            self.factor = self.time_unit_dict[self.input_unit]
        else:
            self.factor = (
                self.prefix_dict[self.prefix] if self.prefix is not None else 1
            )

    def _get_default_unit(self, unit: str) -> Tuple[Optional[str], str]:
        """Return the default unit and prefix of a given unit.

        Args:
            unit (str): The unit to convert.

        Returns:
            Tuple[Optional[str], str]: The prefix and default unit.
        """
        if unit in self.time_unit_dict.keys():
            return None, "s"
        if unit[0] in self.prefix_dict:
            return unit[0], unit[1:]
        else:
            return None, unit

    def _get_default_quantity(self, unit: str) -> str:
        """Return the default quantity of a given unit.

        Args:
            unit (str): The unit to convert.
        """
        try:
            return self.unit_dict[unit]
        except KeyError:
            raise ValueError(f"Unit {unit} is not recognized.")

    @staticmethod
    def get_quantity_and_unit(
        name: str, name_pattern: str = r"([\w\s]+?)\s*\[(\w+)\]"
    ) -> Tuple[str | Any, ...]:
        """Return the quantity and unit of a column name.

        Args:
            name (str): The column name.
            name_pattern (str): The pattern to match the column name.
        """
        pattern = re.compile(name_pattern)
        match = pattern.match(name)
        if match is not None:
            return match.groups()
        else:
            raise ValueError(f"Name {name} does not match pattern.")

    def from_default_unit(self) -> pl.Expr:
        """Convert the column from the default unit.

        Returns:
            pl.Expr: The converted column expression.
        """
        return (
            pl.col(f"{self.input_quantity} [{self.default_unit}]") / self.factor
        ).alias(f"{self.input_quantity} [{self.input_unit}]")

    def to_default_name_and_unit(self) -> pl.Expr:
        """Convert the column to the default unit and change to the default name.

        Returns:
            pl.Expr: The converted column expression.
        """
        conversion = pl.col(self.name) * self.factor
        return conversion.alias(f"{self.default_quantity} [{self.default_unit}]")

    def to_default_unit(self) -> pl.Expr:
        """Convert the column to the default unit without changing the name.

        Returns:
            pl.Expr: The converted column expression.
        """
        return (pl.col(self.name) * self.factor).alias(
            f"{self.input_quantity} [{self.default_unit}]"
        )
