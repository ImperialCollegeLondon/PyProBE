"""A module for unit conversion of PyProBE data."""

import itertools
import re

import polars as pl
from loguru import logger

unit_dict: dict[str, str] = {
    "A": "Current",
    "V": "Voltage",
    "Ah": "Capacity",
    "A.h": "Capacity",
    "s": "Time",
    "C": "Temperature",
    "Ohms": "Resistance",
    "Seconds": "Time",
    "%": "Percentage",
    "K": "Temperature",
    "°C": "Temperature",
}
"""A dictionary of valid units and their corresponding quantities."""

prefix_dict: dict[str, float] = {
    "m": 1e-3,
    "µ": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "k": 1e3,
    "M": 1e6,
}
"""A dictionary of SI prefixes and their corresponding factors."""
time_unit_dict: dict[str, float] = {
    "s": 1,
    "min": 60,
    "hr": 3600,
    "Seconds": 1,
    "h": 3600,
}
"""A dictionary of valid time units and their corresponding factors."""
valid_units = set(
    # Generate prefixed units
    [f"{prefix}{unit}" for prefix, unit in itertools.product(prefix_dict, unit_dict)]
    +
    # Add base units
    list(unit_dict)
    +
    # Add time units
    list(time_unit_dict),
)
"""A set of all valid units, including prefixed combinations."""


def split_quantity_unit(
    name: str,
    regular_expression: str = r"^(.*?)(?:\s*\[([^\]]+)\])?$",
) -> tuple[str, str]:
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


def get_unit_scaling(unit: str) -> tuple[float, str]:
    """Return the default unit and prefix of a given unit.

    Args:
        unit (str): The unit to convert.

    Returns:
        Tuple[Optional[str], str]: The prefix and default unit.
    """
    if unit in time_unit_dict:
        return time_unit_dict[unit], "s"
    if len(unit) == 1:
        return 1, unit
    if unit[0] in prefix_dict:
        return prefix_dict[unit[0]], unit[1:]
    else:
        return 1, unit


@pl.api.register_expr_namespace("units")
class UnitsExpr:
    """A polars namespace for unit conversion of columns."""

    def __init__(self, expr: pl.Expr) -> None:
        """Initialize the UnitsExpr object."""
        self._expr = expr

    def to_base_unit(self, unit: str) -> pl.Expr:
        """Convert a from a given unit to its base unit.

        Args:
            unit (str): The unit to convert.
        """
        if unit not in self._expr.meta.output_name():
            raise ValueError(f"Unit {unit} is not in the column name.")
        scaling, _ = get_unit_scaling(unit)
        return self._expr.cast(pl.Float64) * scaling

    def from_base_unit(self, unit: str) -> pl.Expr:
        """Convert a column from its base unit to a given unit.

        Args:
            unit (str): The unit to convert.
        """
        scaling, _ = get_unit_scaling(unit)
        return self._expr.cast(pl.Float64) / scaling

    def to_unit(self, unit: str) -> pl.Expr:
        """Convert a column to a given unit.

        Args:
            unit (str): The unit to convert to.
        """
        current_col_name = self._expr.meta.output_name()
        quantity, base_unit = split_quantity_unit(current_col_name)
        base_unit_scaling, _ = get_unit_scaling(base_unit)
        target_unit_scaling, target_unit = get_unit_scaling(unit)
        scaling = base_unit_scaling / target_unit_scaling
        return (self._expr.cast(pl.Float64) * scaling).alias(f"{quantity} [{unit}]")
