"""Parse and convert cycler column names with physical units.

This module provides :class:`ColumnName`, which parses cycler-specific column
name formats (Arbin, Biologic, Neware, etc.) into quantity and unit components,
and performs automatic unit conversions.

It is the core building block of the column resolution system:
:meth:`ColumnName.resolve` implements the full resolution chain used by
:meth:`~pyprobe.schema.bdf.BDFColumn.from_columns`:

1. Exact string match
2. Direct quantity match (with unit conversion if needed)
3. BDF alias match (requires :data:`~pyprobe.schema.bdf.ALL_COLUMNS`)
4. BDF recipe fallback (computed column, requires
   :data:`~pyprobe.schema.bdf.ALL_COLUMNS`)

All cycler format patterns are defined in :data:`FORMAT_REGISTRY`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pint
import polars as pl

if TYPE_CHECKING:
    from pyprobe.schema.bdf import BDFColumn

FORMAT_REGISTRY: dict[str, str] = {
    "bdf": r"^([^/]*?)(?:\s*/\s*(.+?))?\s*$",
    "square_bracket": r"^([^[\]]*?)(?:\s*\[([^\]]+)\])?\s*$",
    "parentheses": r"^([^()]*?)(?:\s*\(([^)]+)\))?\s*$",
    "neware": r"^([^()]*?)(?:\(([^)]+)\))?\s*$",
    "basytec": r"^~?([^[\]]*?)(?:\[([^\]]+)\])?\s*$",
    "biologic": r"^([^/]+?)(?:/(.+))?\s*$",
}
"""Regex patterns for all known cycler column name formats.

Each entry maps a human-readable format name to a regex pattern with exactly
two capture groups: ``(1)`` the quantity name and ``(2)`` the unit string
(which may be absent).

Format descriptions:

- ``"bdf"``: BDF slash format ``"Quantity / unit"``; quantity must not
  contain ``/``.
- ``"square_bracket"``: Bracket format ``"Quantity [unit]"``; quantity must
  not contain ``[`` or ``]``.
- ``"parentheses"``: Arbin/Novonix style ``"Quantity (unit)"`` with a
  mandatory space before the opening parenthesis.
- ``"neware"``: Neware style ``"Quantity(unit)"`` — no space before the
  parenthesis.
- ``"basytec"``: Basytec style ``"~Quantity[unit]"`` — optional leading
  tilde.
- ``"biologic"``: Biologic style ``"Quantity/unit"`` — slash with no
  surrounding spaces required.
"""

_ureg = pint.UnitRegistry()
"""Module-level shared pint unit registry."""

# Register non-standard unit spellings that pint does not know natively.
# Note: 'sec' and 'hr' are already recognised by pint and must NOT be
# redefined here — doing so would shadow the built-in and break equality
# checks.  Only spellings that are genuinely absent from pint's default
# registry need an explicit define() call.
for _alias, _canonical in [
    ("Ohms", "ohm"),
    ("Ohm", "ohm"),
    ("Seconds", "s"),
]:
    _ureg.define(f"{_alias} = {_canonical}")

_UNIT_ALIASES: dict[str, str] = {
    "°C": "degC",
}
"""Alias map for unit strings that pint cannot handle natively.

Only entries whose unit symbol contains characters that pint's
``define()`` cannot accept (e.g. the degree symbol ``°``) belong here.
All other non-standard spellings are registered directly on ``_ureg``.
"""


class ColumnName:
    """Parse a column name into a quantity and unit, and perform unit conversions.

    Supports any regex pattern that has exactly two capture groups: the first
    for the quantity name and the second for the unit string.  Patterns for
    all known cycler formats are defined centrally in
    :data:`FORMAT_REGISTRY`.

    Quantity names may contain any characters except the format's separator
    characters, which allows cycler column names that include characters such
    as ``~``, ``<>``, and ``.``.

    Examples:
        >>> from pyprobe.schema.column_name import FORMAT_REGISTRY
        >>> cn = ColumnName("Current / A", FORMAT_REGISTRY["bdf"])
        >>> cn.quantity
        'Current'
        >>> cn.unit
        <Unit('ampere')>
    """

    @staticmethod
    def _extract_quantity_and_unit(name: str, pattern: str) -> tuple[str, str | None]:
        """Extract the quantity name and raw unit string from a column name.

        Bare names (no unit separator) return ``None`` as the unit string.
        Names with a partial separator (e.g. ``"Step /"``) fail to match and
        raise ``ValueError``.

        Args:
            name: The column name string to parse.
            pattern: A regex pattern with two capture groups
                (quantity, unit).

        Returns:
            A ``(quantity, raw_unit)`` tuple where ``raw_unit`` is ``None`` for
            bare names.

        Raises:
            ValueError: If ``name`` does not match ``pattern``.

        Examples:
            >>> from pyprobe.schema.column_name import FORMAT_REGISTRY
            >>> ColumnName._extract_quantity_and_unit(
            ...     "Current [A]", FORMAT_REGISTRY["square_bracket"]
            ... )
            ('Current', 'A')
            >>> ColumnName._extract_quantity_and_unit(
            ...     "Step", FORMAT_REGISTRY["square_bracket"]
            ... )
            ('Step', None)
        """
        match = re.compile(pattern).match(name)
        if match is None:
            raise ValueError(
                f"Column name '{name}' does not match pattern '{pattern}'."
            )
        quantity = match.group(1).strip()
        raw_unit: str | None = (match.group(2) or "").strip() or None
        return quantity, raw_unit

    def __init__(self, name: str, pattern: str) -> None:
        """Parse a column name string into quantity and unit components.

        Bare names (no unit separator) are accepted and yield ``unit=None``.
        A name that contains a separator but no unit (e.g. ``"Step /"``) raises
        ``ValueError`` because the regex cannot match it.

        Args:
            name: The column name string to parse (e.g. ``"Current [A]"`` or
                ``"Step"``).
            pattern: A regex pattern with two capture groups
                (quantity, unit).  Use a pattern from
                :data:`FORMAT_REGISTRY`.

        Raises:
            ValueError: If the name contains a unit separator but no valid unit.
            ValueError: If the unit string cannot be parsed by pint.
        """
        self._name = name

        self._quantity, raw_unit = ColumnName._extract_quantity_and_unit(name, pattern)

        if raw_unit is None:
            self._unit: pint.Unit | None = None
        else:
            resolved = _UNIT_ALIASES.get(raw_unit, raw_unit)
            if self._quantity.lower() == "temperature" and resolved == "C":
                resolved = "degC"
            try:
                self._unit = _ureg.parse_units(resolved)
            except pint.errors.UndefinedUnitError as exc:
                raise ValueError(
                    f"Unit '{raw_unit}' in column '{name}' could not be parsed: {exc}"
                ) from exc

    @property
    def quantity(self) -> str:
        """The physical quantity name, with unit information removed.

        Returns:
            The quantity string (e.g. ``"Current"``).
        """
        return self._quantity

    @property
    def unit(self) -> pint.Unit | None:
        """The parsed pint unit, or ``None`` if the column has no unit.

        Returns:
            A :class:`pint.Unit` instance, or ``None``.
        """
        return self._unit

    def __str__(self) -> str:
        """Return the original column name string.

        Returns:
            The original name passed to the constructor.
        """
        return self._name

    def conversion_parameters(self, target_unit: str) -> tuple[float, float]:
        """Compute the factor and offset to convert this column's unit to another.

        The conversion is: ``target_value = source_value * factor + offset``.

        For purely multiplicative conversions (e.g. mA → A) the offset is
        ``0.0``.  For affine conversions (e.g. degC → K) the offset is
        non-zero.

        Args:
            target_unit: The target unit string (e.g. ``"mA"``, ``"K"``,
                ``"degC"``).

        Returns:
            A ``(factor, offset)`` tuple, both as :class:`float`.  For most
            unit pairs the offset is ``0.0``.

        Raises:
            ValueError: If this column has no unit (i.e. :attr:`unit` is
                ``None``).
            ValueError: If the units are dimensionally incompatible.

        Examples:
            >>> from pyprobe.schema.column_name import FORMAT_REGISTRY
            >>> cn = ColumnName("Current [A]", FORMAT_REGISTRY["square_bracket"])
            >>> cn.conversion_parameters("mA")
            (1000.0, 0.0)
            >>> pat = FORMAT_REGISTRY["square_bracket"]
            >>> cn_temp = ColumnName("Temperature [degC]", pat)
            >>> cn_temp.conversion_parameters("K")
            (1.0, 273.15)
        """
        if self._unit is None:
            raise ValueError(
                f"Column '{self._name}' has no unit; cannot compute conversion "
                "parameters."
            )
        resolved_target = _UNIT_ALIASES.get(target_unit, target_unit)
        try:
            target_pint = _ureg.parse_units(resolved_target)
            # Convert two reference points to derive factor and offset.
            zero = float(_ureg.Quantity(0, self._unit).to(target_pint).magnitude)
            one = float(_ureg.Quantity(1, self._unit).to(target_pint).magnitude)
        except pint.errors.DimensionalityError as exc:
            raise ValueError(
                f"Cannot convert '{self._unit}' to '{target_unit}': {exc}"
            ) from exc
        factor = one - zero
        offset = zero
        return factor, offset

    @staticmethod
    def find(
        quantity: str,
        available_columns: list[str],
        available_format: str,
    ) -> tuple[str, ColumnName] | None:
        """Find the first column whose parsed quantity matches (case-insensitive).

        Parses each column in available_columns using the regex pattern from
        FORMAT_REGISTRY[available_format]. Returns the first column whose
        quantity matches the given quantity string (case-insensitive, stripped).

        Args:
            quantity: The quantity name to search for (e.g. ``"Current"``).
            available_columns: Column name strings to search through.
            available_format: Key into FORMAT_REGISTRY for parsing columns.

        Returns:
            A ``(raw_column_string, parsed_ColumnName)`` tuple, or ``None``
            if no match is found.
        """
        pattern = FORMAT_REGISTRY[available_format]
        target = quantity.lower().strip()
        for col in available_columns:
            try:
                cn = ColumnName(col, pattern)
            except ValueError:
                continue
            if cn.quantity.lower().strip() == target:
                return col, cn
        return None

    def _to_expr(self, col_str: str, source_cn: ColumnName) -> pl.Expr:
        """Build a Polars expression with unit conversion aliased to str(self).

        Constructs a :class:`polars.Expr` that selects ``col_str``, applies a
        multiplicative unit conversion factor when necessary, and aliases the
        result to the original column name string (``str(self)``).

        When either the source or target unit is ``None`` (dimensionless column),
        no conversion is applied.  A conversion factor of exactly ``1.0`` is
        also skipped to avoid an unnecessary cast.

        Args:
            col_str: The raw column name string to select from the DataFrame.
            source_cn: The parsed :class:`ColumnName` of the source column,
                used to compute the conversion factor.

        Returns:
            A :class:`polars.Expr` selecting ``col_str``, optionally scaled,
            and aliased to ``str(self)``.
        """
        if self.unit is None or source_cn.unit is None:
            return pl.col(col_str).alias(str(self))
        factor, offset = source_cn.conversion_parameters(str(self.unit))
        if factor == 1.0 and offset == 0.0:
            return pl.col(col_str).alias(str(self))
        expr = pl.col(col_str).cast(pl.Float64)
        if factor != 1.0:
            expr = expr * factor
        if offset != 0.0:
            expr = expr + offset
        return expr.alias(str(self))

    def resolve(
        self,
        available_columns: list[str],
        available_format: str,
        bdf_columns: list[BDFColumn] | None = None,
    ) -> pl.Expr:
        """Resolve this column name against available columns.

        Full resolution chain (tried in order):

        1. **Exact match** — ``str(self)`` is already present in
           ``available_columns``; returns ``pl.col(str(self))`` with no alias.
        2. **Direct quantity match** — searches ``available_columns`` for a
           column whose parsed quantity equals ``self.quantity``; applies unit
           conversion if needed.
        3. **BDF alias match** (requires ``bdf_columns``) — finds the
           :class:`~pyprobe.schema.bdf.BDFColumn` whose :meth:`matches` returns
           ``True`` for ``self.quantity``, then searches all its aliases via
           :meth:`find`.
        4. **BDF recipe** (requires ``bdf_columns``) — calls
           ``bdf_col.try_recipes()`` to derive the column computationally.
        5. Raises :class:`ValueError` if all steps fail.

        ``column_name.py`` never imports from ``bdf.py``.  The objects in
        ``bdf_columns`` are accessed only via their public interface
        (``.matches()``, ``.aliases``, ``.name``, ``.try_recipes()``).

        Args:
            available_columns: Column name strings from the source DataFrame.
            available_format: Key into FORMAT_REGISTRY for parsing columns.
            bdf_columns: Optional list of BDF column objects enabling alias
                and recipe resolution (use cases 3-5).  When ``None``, only
                use cases 1-2 are attempted.

        Returns:
            A :class:`polars.Expr` that selects the matching column, applies
            unit conversion if needed, and aliases to ``str(self)``.

        Raises:
            ValueError: If no matching column is found after all resolution
                steps are exhausted.
        """
        # Step 1: exact string match — fastest path, no parsing required.
        if str(self) in available_columns:
            return pl.col(str(self))

        # Step 2: direct quantity match (use cases 1-3 in plan).
        result = ColumnName.find(self.quantity, available_columns, available_format)
        if result is not None:
            return self._to_expr(*result)

        # Steps 3-4: BDF alias + recipe (use cases 4-5 in plan).
        if bdf_columns is not None:
            bdf_col = None
            for col in bdf_columns:
                if col.matches(self.quantity):
                    bdf_col = col
                    break

            if bdf_col is not None:
                # Step 3: try canonical name + each alias.
                for alias in [bdf_col.name] + bdf_col.aliases:
                    result = ColumnName.find(alias, available_columns, available_format)
                    if result is not None:
                        return self._to_expr(*result)

                # Step 4: try recipes.
                expr = bdf_col.try_recipes(
                    available_columns, available_format, bdf_columns, str(self)
                )
                if expr is not None:
                    return expr

        raise ValueError(
            f"No column matching quantity '{self.quantity}' found in"
            f" {available_columns}"
        )
