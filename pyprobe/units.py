"""A module for unit conversion of PyProBE data."""

import itertools
import logging
import re

import polars as pl

logger = logging.getLogger(__name__)

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
    list(time_unit_dict)
)
"""A set of all valid units, including prefixed combinations."""


def split_quantity_unit(
    name: str, regular_expression: str = r"^(.*?)(?:\s*\[([^\]]+)\])?$"
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
    if unit in time_unit_dict.keys():
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

    def to_default(self, regexp: str = r"^(.*?)(?:\s*\[([^\]]*)\])?$") -> pl.Expr:
        """Convert a column to the default unit.

        Args:
            regexp (str): The pattern to match the column name.
        """
        column_name = self._expr.meta.output_name()
        quantity, unit = split_quantity_unit(column_name, regexp)
        if unit == "":
            return self._expr
        scaling, si_unit = get_unit_scaling(unit)
        default_name = f"{quantity} [{si_unit}]"
        return (self._expr.cast(pl.Float64) * scaling).alias(default_name)

    def to_si(self, unit: str) -> pl.Expr:
        """Convert a from a given unit to its SI equivalent.

        Args:
            unit (str): The unit to convert.
        """
        if unit not in self._expr.meta.output_name():
            raise ValueError(f"Unit {unit} is not in the column name.")
        scaling, _ = get_unit_scaling(unit)
        return self._expr.cast(pl.Float64) * scaling


class Units:
    """A class to store unit conversion information about columns.

    Args:
        input_quantity (str): The quantity of the column.
        input_unit (str): The unit of the column.
    """

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
        "s": 1.0,
        "min": 60.0,
        "hr": 3600.0,
        "Seconds": 1.0,
    }
    """A dictionary of time units and their corresponding factors."""

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

    def _get_default_unit(self, unit: str) -> tuple[str | None, str]:
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
