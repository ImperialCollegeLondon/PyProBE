"""A module for unit conversion of PyProBE data."""

import logging
import re
from typing import Dict, Optional, Tuple

import polars as pl

logger = logging.getLogger(__name__)


class Units:
    """A class to store unit conversion information about columns.

    Args:
        input_quantity (str): The quantity of the column.
        input_unit (str): The unit of the column.
    """

    prefix_dict: Dict[str, float] = {
        "m": 1e-3,
        "µ": 1e-6,
        "n": 1e-9,
        "p": 1e-12,
        "k": 1e3,
        "M": 1e6,
    }
    """A dictionary of SI prefixes and their corresponding factors."""

    time_unit_dict: Dict[str, float] = {
        "s": 1.0,
        "min": 60.0,
        "hr": 3600.0,
        "Seconds": 1.0,
    }
    """A dictionary of time units and their corresponding factors."""

    unit_dict: Dict[str, str] = {
        "A": "Current",
        "V": "Voltage",
        "Ah": "Capacity",
        "A.h": "Capacity",
        "s": "Time",
        "C": "Temperature",
        "Ohms": "Resistance",
        "Seconds": "Time",
        "%": "Percentage",
    }
    """A dictionary of units and their corresponding quantities."""

    def __init__(
        self,
        input_quantity: str,
        input_unit: str,
    ) -> None:
        """Initialize the Units object."""
        self.input_quantity = input_quantity
        self.input_unit = input_unit
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
        unit = re.sub(r"[^a-zA-Z%]", "", unit)  # Remove non-alphabetic characters
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
            error_msg = f"Unit {unit} is not recognized."
            logger.error(error_msg)
            raise ValueError(error_msg)

    def from_default_unit(self) -> pl.Expr:
        """Convert the column from the default unit.

        Returns:
            pl.Expr: The converted column expression.
        """
        return (
            pl.col(f"{self.input_quantity} [{self.default_unit}]") / self.factor
        ).alias(f"{self.input_quantity} [{self.input_unit}]")

    def to_default_unit(self) -> pl.Expr:
        """Convert the column to the default unit.

        Returns:
            pl.Expr: The converted column expression.
        """
        return (
            pl.col(f"{self.input_quantity} [{self.input_unit}]").cast(pl.Float64)
            * self.factor
        ).alias(f"{self.input_quantity} [{self.default_unit}]")


def split_quantity_unit(
    name: str, regular_expression: str = r"^(.*?)(?:\s*\[([^\]]+)\])?$"
) -> Tuple[str, str]:
    """Split a column name into quantity and unit.

    Args:
        name: The column name (e.g. "Current [A]" or "Temperature")
        regular_expression: The pattern to match the column name.

    Returns:
        The quantity and unit.
    """
    pattern = re.compile(regular_expression)
    match = pattern.match(name)
    if match is not None:
        quantity = match.group(1).strip()
        unit = match.group(2) or ""  # Group 2 will be None if no brackets
        return quantity, unit
    else:
        error_msg = f"Name {name} does not match pattern."
        logger.error(error_msg)
        raise ValueError(error_msg)


def unit_from_regexp(
    name: str, regular_expression: str = r"^(.*?)(?:\s*\[([^\]]*)\])?$"
) -> "Units":
    """Create an instance of a units class from column name and regular expression.

    Args:
        name (str): The column name (e.g. "Current [A]" or "Temperature")
        regular_expression (str): The pattern to match the column name.

    Returns:
        Units: Instance with quantity and unit (empty string if no unit found)
    """
    quantity, unit = split_quantity_unit(name, regular_expression)
    return Units(quantity, unit)
