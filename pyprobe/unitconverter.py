"""A module for unit conversion and zero referencing of data columns."""

import re
from typing import Any, Optional, Tuple

import polars as pl


class UnitConverter:
    """A class to store unit conversion information about columns."""

    prefix_dict = {"m": 1e-3, "µ": 1e-6, "n": 1e-9, "p": 1e-12, "k": 1e3, "M": 1e6}
    time_unit_dict = {"min": 60, "hr": 3600}
    default_unit_dict = {"Current": "A", "Voltage": "V", "Capacity": "Ah", "Time": "s"}

    def __init__(
        self,
        name: str,
        name_pattern: str = r"(\w+)\s*\[(\w+)\]",
        default_quantity: Optional[str] = None,
    ) -> None:
        """Initialize the UnitConverter object.

        Args:
            name (str): The column name.
            name_pattern (str): The pattern to match the column name.
            default_quantity (Optional[str]): The PyProBE default name of the column.
        """
        self.name = name  # the column name

        self.input_quantity, self.input_unit = self.get_quantity_and_unit(
            name, name_pattern
        )  # the quantity and unit of the column

        # find the default quantity and unit
        if default_quantity is not None:
            self.default_quantity = default_quantity
        else:
            self.default_quantity = self.check_quantity()
        self.default_unit = self.default_unit_dict[self.default_quantity]
        self.default_name = f"{self.default_quantity} [{self.default_unit}]"

        # find the prefix
        if self.default_quantity == "Time":
            self.prefix = ""
        else:
            self.prefix = self.input_unit.rstrip(self.default_unit)

    @staticmethod
    def get_quantity_and_unit(name: str, name_pattern: str) -> Tuple[str | Any, ...]:
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

    def check_quantity(self) -> str:
        """Check the quantity being converted is recognised.

        Returns:
            str: The default quantity name.

        Raises:
            ValueError: If the quantity is not recognised.
        """
        if self.input_quantity in self.default_unit_dict.keys():
            return self.input_quantity
        else:
            raise ValueError(f"Quantity {self.input_quantity} not recognised.")

    def _convert(self, operation: str) -> pl.Expr:
        """Make a unit conversion from or to the default unit.

        Args:
            operation (str): The operation to perform.
        """
        if self.prefix in self.prefix_dict:
            factor = self.prefix_dict[self.prefix]
        elif self.default_quantity == "Time" and self.input_unit in self.time_unit_dict:
            factor = self.time_unit_dict[self.input_unit]
        elif self.input_unit == self.default_unit:
            factor = 1
        else:
            raise ValueError(
                f"Unit {self.input_unit} for {self.default_quantity} not recognised."
            )

        if operation == "from":
            return (pl.col(self.default_name) / factor).alias(self.name)
        elif operation == "to":
            return (pl.col(self.name) * factor).alias(self.default_name)

    def from_default(self) -> pl.Expr:
        """Convert the column from the default unit.

        Returns:
            pl.Expr: The converted column expression.
        """
        return self._convert("from")

    def to_default(self) -> pl.Expr:
        """Convert the column to the default unit.

        Returns:
            pl.Expr: The converted column expression.
        """
        return self._convert("to")
