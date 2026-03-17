"""A module for parsing and converting column names with physical units."""

import re

import pint

_ureg = pint.UnitRegistry()
"""Module-level shared pint unit registry."""

_UNIT_ALIASES: dict[str, str] = {
    "Ohms": "ohm",
    "Seconds": "s",
}
"""Alias map for non-standard unit strings used in the codebase."""


class ColumnName:
    """Parse a column name into a quantity and unit, and perform unit conversions.

    Supports two column name formats:

    - Bracket format: ``"Quantity [unit]"``
    - Slash format: ``"Quantity / unit"``

    Examples:
        >>> cn = ColumnName("Current [A]", ColumnName.BRACKET_FORMAT)
        >>> cn.quantity
        'Current'
        >>> cn = ColumnName("Current / A", ColumnName.SLASH_FORMAT)
        >>> cn.quantity
        'Current'
    """

    BRACKET_FORMAT: str = r"^([\w\s]*?)(?:\s*\[([^\]]+)\])?\s*$"
    """Regex pattern for bracket-style column names.

    Matches ``"Quantity [unit]"`` or a bare ``"Quantity"`` (no brackets).
    Group 1 is restricted to word characters and whitespace, so separator
    characters cause the match to fail for partial forms like ``"Quantity ["``.
    """

    SLASH_FORMAT: str = r"^([\w\s]*?)(?:\s*/\s*(.+?))?\s*$"
    """Regex pattern for slash-style column names.

    Matches ``"Quantity / unit"`` or a bare ``"Quantity"`` (no slash).
    Group 1 is restricted to word characters and whitespace, so
    ``"Quantity /"`` and bracket-format names like ``"Quantity [unit]"``
    fail to match, allowing :func:`_parse_column` to fall back to
    :attr:`BRACKET_FORMAT`.
    """

    @staticmethod
    def _extract_quantity_and_unit(name: str, pattern: str) -> tuple[str, str | None]:
        """Extract the quantity name and raw unit string from a column name.

        Bare names (no unit separator) return ``None`` as the unit string.
        Names with a partial separator (e.g. ``"Step /"``) fail to match and
        raise ``ValueError``.

        Args:
            name: The column name string to parse.
            pattern: The regex pattern to apply. Use :attr:`BRACKET_FORMAT` or
                :attr:`SLASH_FORMAT`.

        Returns:
            A ``(quantity, raw_unit)`` tuple where ``raw_unit`` is ``None`` for
            bare names.

        Raises:
            ValueError: If ``name`` does not match ``pattern``.

        Examples:
            >>> ColumnName._extract_quantity_and_unit(
            ...     "Current [A]", ColumnName.BRACKET_FORMAT
            ... )
            ('Current', 'A')
            >>> ColumnName._extract_quantity_and_unit(
            ...     "Step", ColumnName.BRACKET_FORMAT
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

    def __init__(self, name: str, pattern: str = SLASH_FORMAT) -> None:
        """Parse a column name string into quantity and unit components.

        Bare names (no unit separator) are accepted and yield ``unit=None``.
        A name that contains a separator but no unit (e.g. ``"Step /"``) raises
        ``ValueError`` because the regex cannot match it.

        Args:
            name: The column name string to parse (e.g. ``"Current [A]"`` or
                ``"Step"``).
            pattern: The regex pattern to apply. Use :attr:`BRACKET_FORMAT` or
                :attr:`SLASH_FORMAT`. Defaults to :attr:`SLASH_FORMAT`.

        Raises:
            ValueError: If the name contains a unit separator but no valid unit.
            ValueError: If the unit string cannot be parsed by pint.
        """
        self._name = name
        self._pattern = pattern

        self._quantity, raw_unit = ColumnName._extract_quantity_and_unit(name, pattern)

        if raw_unit is None:
            self._unit: pint.Unit | None = None
        else:
            resolved = _UNIT_ALIASES.get(raw_unit, raw_unit)
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

    def conversion_factor(self, target_unit: str) -> float:
        """Compute the multiplicative factor to convert this column's unit to another.

        Args:
            target_unit: The target unit string (e.g. ``"mA"``).

        Returns:
            The conversion factor as a float. Multiply a value in the current unit by
            this factor to obtain the equivalent value in ``target_unit``.

        Raises:
            ValueError: If this column has no unit (i.e. :attr:`unit` is ``None``).
            ValueError: If the units are dimensionally incompatible.

        Examples:
            >>> cn = ColumnName("Current [A]", ColumnName.BRACKET_FORMAT)
            >>> cn.conversion_factor("mA")
            1000.0
        """
        if self._unit is None:
            raise ValueError(
                f"Column '{self._name}' has no unit; cannot compute a conversion "
                "factor."
            )
        resolved_target = _UNIT_ALIASES.get(target_unit, target_unit)
        try:
            target_pint = _ureg.parse_units(resolved_target)
            magnitude: float = float((1.0 * self._unit).to(target_pint).magnitude)
        except pint.errors.DimensionalityError as exc:
            raise ValueError(
                f"Cannot convert '{self._unit}' to '{target_unit}': {exc}"
            ) from exc
        return magnitude

    def with_unit(self, target_unit: str) -> "ColumnName":
        """Return a new :class:`ColumnName` with a different unit.

        The quantity name is preserved; only the unit portion of the string changes.
        The output format mirrors the original (bracket or slash).

        Args:
            target_unit: The replacement unit string (e.g. ``"mA"``).

        Returns:
            A new :class:`ColumnName` instance using ``target_unit``.

        Raises:
            ValueError: If this column has no unit.

        Examples:
            >>> cn = ColumnName("Current [A]", ColumnName.BRACKET_FORMAT)
            >>> str(cn.with_unit("mA"))
            'Current [mA]'
            >>> cn2 = ColumnName("Current / A", ColumnName.SLASH_FORMAT)
            >>> str(cn2.with_unit("mA"))
            'Current / mA'
        """
        if self._unit is None:
            raise ValueError(
                f"Column '{self._name}' has no unit; cannot substitute a new unit."
            )
        if self._pattern == ColumnName.BRACKET_FORMAT:
            new_name = f"{self._quantity} [{target_unit}]"
        else:
            new_name = f"{self._quantity} / {target_unit}"
        return ColumnName(new_name, self._pattern)

    @classmethod
    def find_in_columns(
        cls,
        quantity: str,
        columns: list[str],
        pattern: str,
    ) -> "ColumnName | None":
        """Search a list of column names for one matching a given quantity.

        Args:
            quantity: The quantity name to search for (e.g. ``"Current"``).
            columns: The list of column name strings to search.
            pattern: The regex pattern to use when parsing each column.

        Returns:
            The first :class:`ColumnName` whose :attr:`quantity` equals ``quantity``,
            or ``None`` if no match is found.

        Examples:
            >>> cols = ["Time [s]", "Current [A]", "Voltage [V]"]
            >>> result = ColumnName.find_in_columns(
            ...     "Current", cols, ColumnName.BRACKET_FORMAT
            ... )
            >>> str(result)
            'Current [A]'
        """
        for col in columns:
            try:
                parsed = cls(col, pattern)
            except ValueError:
                continue
            if parsed.quantity == quantity:
                return parsed
        return None
