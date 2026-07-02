"""Column abstraction for BDF-standard battery data.

This module provides classes for working with BDF (Battery Data Format)
column names and Polars expressions:

- :class:`Column` — pure descriptor that parses a ``"Quantity / unit"``
  string and computes unit-conversion parameters. Owns resolution logic
  via :meth:`~Column.can_resolve` and :meth:`~Column.resolve`.
- :class:`BDFColumn` — subclass that adds recipe-based derivation metadata
  and a linked-data IRI. Extends resolution to cover recipe derivation via
  :meth:`~BDFColumn.can_resolve` and :meth:`~BDFColumn.resolve`.
- :class:`ColumnDict` — thin per-DataFrame wrapper that delegates resolution
  to :class:`Column` / :class:`BDFColumn` methods.

The :class:`BDF` enum mirrors the BDF-standard quantities defined in
:data:`bdf.spec.COLUMN_ONTOLOGY` as members (e.g. :attr:`BDF.CURRENT_AMPERE`,
:attr:`BDF.VOLTAGE_VOLT`). :data:`DEFAULT_COLUMNS` is the core subset that
PyProBE retains after ingestion.

Typical usage::

    from pyprobe.columns import BDF, DEFAULT_COLUMNS, ColumnDict

    cs = ColumnDict(DEFAULT_COLUMNS)
    # Select Current in milliamps from a DataFrame that has "Current / A".
    expr = cs.resolve("Current / mA")
"""

import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import cache
from types import MappingProxyType
from typing import Any, cast

import bdf.spec as _bdf_spec
import pint
import polars as pl
from loguru import logger

_ONTOLOGY = _bdf_spec.COLUMN_ONTOLOGY
"""The BDF column ontology that PyProBE's :class:`BDF` definitions track."""

BDF_PATTERN: str = r"^([^/]*?)(?:\s*/\s*(.+?))?\s*$"
"""Regex pattern for BDF ``"Quantity / unit"`` column names.

Two capture groups: ``(1)`` quantity name, ``(2)`` unit string (may be absent).
"""

BDF_IRI_PREFIX: str = (
    "https://w3id.org/battery-data-alliance/ontology/battery-data-format#"
)
"""Common prefix for all BDF ontology IRIs."""

_ureg: pint.UnitRegistry = pint.UnitRegistry()
"""Module-level shared pint unit registry."""

for _alias, _canonical in [
    ("Ohm", "ohm"),
]:
    _ureg.define(f"{_alias} = {_canonical}")

DEFAULT_COLUMNS: list[str] = [
    "Test Time / s",
    "Current / A",
    "Voltage / V",
    "Net Capacity / Ah",
    "Step Count / 1",
    "Step ID",
]
"""Core PyProBE column subset retained after BDF ingestion.

These are the column names (in BDF ``"Quantity / unit"`` format) that
PyProBE keeps after reducing raw cycler data to a minimal, analysis-ready
feature set.
"""


class UnitsError(ValueError):
    """Raised when unit conversion is invalid or impossible.

    This exception is raised when:
    - Attempting to convert a dimensionless column (unit == "1").
    - Units are dimensionally incompatible.
    - A unit string cannot be parsed.
    """


class ColumnResolutionError(ValueError):
    """Raised when a Column cannot be resolved from available columns.

    This exception is raised when :meth:`Column.can_resolve` fails to find a
    compatible column in the provided set.
    """


def _resolve_unit(raw_unit: str, quantity: str) -> str:
    """Return the pint-parseable unit string, resolving temperature ambiguity.

    ``"C"`` is ambiguous between coulombs and degrees Celsius.  When the
    quantity contains the word ``"temperature"`` (case-insensitive) the
    symbol is mapped to ``"degC"``; otherwise it is returned unchanged.

    Args:
        raw_unit: The unit string as stored in a column name (e.g. ``"C"``).
        quantity: The physical quantity name (e.g. ``"Ambient Temperature"``).

    Returns:
        The resolved unit string (e.g. ``"degC"`` or the original value).

    Examples:
        >>> _resolve_unit("C", "Ambient Temperature")
        'degC'
        >>> _resolve_unit("C", "Charge")
        'C'
        >>> _resolve_unit("mA", "Current")
        'mA'
    """
    if raw_unit == "C" and "temperature" in quantity.lower():
        return "degC"
    return raw_unit


def _apply_conversion(
    expr: pl.Expr,
    factor: float,
    offset: float,
    alias: str,
) -> pl.Expr:
    """Apply a linear unit conversion to a Polars expression.

    Computes ``target = source * factor + offset``, casting to ``Float64``
    only when a non-trivial conversion is needed.  A pure rename (factor
    ``1.0``, offset ``0.0``) returns the expression aliased without any
    arithmetic.

    Args:
        expr: The source Polars expression (any numeric dtype).
        factor: Multiplicative conversion factor.
        offset: Additive conversion offset (non-zero for affine conversions
            such as degC → K).
        alias: Alias string applied to the returned expression.

    Returns:
        A Polars expression aliased to ``alias``.

    Examples:
        >>> import polars as pl
        >>> e = _apply_conversion(pl.col("x"), 1.0, 0.0, "x / A")
        >>> type(e).__name__
        'Expr'
    """
    if factor == 1.0 and offset == 0.0:
        return expr.alias(alias)
    e = expr.cast(pl.Float64)
    if factor != 1.0:
        e = e * factor
    if offset != 0.0:
        e = e + offset
    return e.alias(alias)


def _split_quantity_unit(name: str, pattern: str) -> tuple[str, str | None]:
    """Extract quantity and raw unit string from a column name.

    Bare names (no unit separator) return ``None`` as the unit.

    Args:
        name: The column name string to parse.
        pattern: A regex pattern with two capture groups (quantity, unit).

    Returns:
        A ``(quantity, raw_unit)`` tuple where ``raw_unit`` is ``None`` for
        bare names.

    Raises:
        ValueError: If ``name`` does not match ``pattern``.

    Examples:
        >>> _split_quantity_unit("Current / A", BDF_PATTERN)
        ('Current', 'A')
        >>> _split_quantity_unit("Step", BDF_PATTERN)
        ('Step', None)
        >>> _split_quantity_unit("Step Count / 1", BDF_PATTERN)
        ('Step Count', '1')
    """
    match = re.compile(pattern).match(name)
    if match is None:
        raise ValueError(f"Column name '{name}' does not match pattern '{pattern}'.")
    quantity = match.group(1).strip()
    raw_unit: str | None = (match.group(2) or "").strip() or None
    return quantity, raw_unit


class _TrackingDict(dict[Any, Any]):
    """Dict subclass that records which keys are accessed via ``__getitem__``.

    Used by :meth:`Recipe.__post_init__` to validate that the compute function
    accesses exactly the columns declared in ``required``.

    Attributes:
        accessed: Set of BDFColumn keys that have been accessed.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.accessed: set[BDF] = set()

    def __getitem__(self, key: "BDF") -> pl.Expr:
        self.accessed.add(key)
        return super().__getitem__(key)


@dataclass
class Recipe:
    """A computation rule for deriving a :class:`BDF` from other columns.

    A recipe declares which BDF columns are needed (``required``) and
    provides a callable that maps :class:`BDFColumn` instances to resolved
    Polars expressions, returning a new Polars expression.

    The ``__post_init__`` method validates that the compute function accesses
    exactly the columns listed in ``required`` — no more, no fewer.

    Attributes:
        required: :class:`BDF` enum members that must be resolvable in the
            source DataFrame (e.g. ``[BDF.CHARGING_CAPACITY_AH,
            BDF.DISCHARGING_CAPACITY_AH]``).
        compute: A callable that receives a ``{BDF: pl.Expr}``
            mapping and returns a :class:`polars.Expr`.

    Examples:
        >>> import polars as pl
        >>> recipe = Recipe(
        ...     required=[BDF.CHARGING_CAPACITY_AH, BDF.DISCHARGING_CAPACITY_AH],
        ...     compute=lambda cols: (
        ...         cols[BDF.CHARGING_CAPACITY_AH] - cols[BDF.DISCHARGING_CAPACITY_AH]
        ...     ),
        ... )
        >>> len(recipe.required)
        2
    """

    required: list["BDF"]
    compute: Callable[[dict["BDF", pl.Expr]], pl.Expr]

    def __post_init__(self) -> None:
        """Validate that compute accesses exactly the required columns.

        Raises:
            ValueError: If the compute function accesses columns not in
                ``required``, or if any columns in ``required`` are unused.
        """
        dummy = _TrackingDict({col: pl.lit(0) for col in self.required})
        try:
            self.compute(dummy)
        except KeyError as exc:
            raise ValueError(
                f"Recipe compute accesses a column not in required: {exc}"
            ) from exc
        except Exception:
            return
        unused = set(self.required) - dummy.accessed
        if unused:
            raise ValueError(
                f"Recipe declares unused required columns: "
                f"{[c.quantity for c in unused]}"
            )


@dataclass(frozen=True)
class Column:
    """A BDF column descriptor: quantity name and unit string.

    Constructed directly with quantity and unit strings.  For parsing column
    names from strings, use :func:`column_factory_from_string`.
    Supports unit conversion through :meth:`conversion_parameters`.
    Resolution against a list of available columns is provided by
    :meth:`can_resolve` and :meth:`resolve`.

    Unit ``"1"`` denotes a dimensionless column.  All columns have a unit;
    use ``"1"`` rather than leaving it absent.

    Args:
        quantity: The physical quantity name (e.g. ``"Current"``).
        unit: The unit string (e.g. ``"A"``, ``"Ah"``, ``"1"``).
            Defaults to ``"1"`` for dimensionless columns.

    Attributes:
        quantity: The physical quantity name.
        unit: The unit string.

    Examples:
        >>> col = Column("Current", "A")
        >>> col.name
        'Current / A'
        >>> col_parsed = column_factory_from_string("Current / A")
        >>> col_parsed.quantity
        'Current'
        >>> col_parsed.name
        'Current / A'
        >>> Column("Step").name
        'Step / 1'
    """

    quantity: str
    unit: str | None = "1"

    @property
    def name(self) -> str:
        """BDF standard column name string.

        Columns with a unit are formatted as ``"Quantity / unit"``. Unitless
        columns (``unit`` is ``None``, e.g. ``"Step ID"``) are formatted as the
        bare quantity name, matching the BDF ontology label convention.

        Returns:
            The BDF column name string.

        Examples:
            >>> Column("Current", "A").name
            'Current / A'
            >>> Column("Net Capacity", "Ah").name
            'Net Capacity / Ah'
            >>> Column("Step Count", "1").name
            'Step Count / 1'
            >>> Column("Step").name
            'Step / 1'
            >>> Column("Step ID", None).name
            'Step ID'
        """
        if self.unit is None:
            return self.quantity
        return f"{self.quantity} / {self.unit}"

    def __str__(self) -> str:
        """Return the BDF column name string.

        Returns:
            The same value as :attr:`name`.
        """
        return self.name

    def conversion_parameters(self, target_unit: str) -> tuple[float, float]:
        """Compute the factor and offset to convert this column's unit.

        The conversion formula is:
        ``target_value = source_value * factor + offset``.

        For purely multiplicative conversions (e.g. mA → A) the offset is
        ``0.0``.  For affine conversions (e.g. degC → K) the offset is
        non-zero.

        Parses the stored unit string via pint on demand.

        Args:
            target_unit: The target unit string (e.g. ``"mA"``, ``"K"``).

        Returns:
            A ``(factor, offset)`` tuple, both as :class:`float`.

        Raises:
            UnitsError: If this column is dimensionless (``unit == "1"``).
            UnitsError: If the units are dimensionally incompatible.

        Examples:
            >>> col = Column("Current", "A")
            >>> col.conversion_parameters("mA")
            (1000.0, 0.0)
        """
        if self.unit in ("1", None):
            raise UnitsError(
                f"Column '{self.quantity}' is dimensionless; cannot convert."
            )
        source_unit_str = _resolve_unit(cast(str, self.unit), self.quantity)
        target_unit_str = _resolve_unit(target_unit, self.quantity)
        try:
            source_pint = _ureg.parse_units(source_unit_str)
        except pint.errors.UndefinedUnitError as exc:
            msg = (
                f"Unit '{self.unit}' for quantity '{self.quantity}' "
                f"could not be parsed: {exc}"
            )
            raise UnitsError(msg) from exc
        try:
            target_pint = _ureg.parse_units(target_unit_str)
            zero = float(_ureg.Quantity(0, source_pint).to(target_pint).magnitude)
            one = float(_ureg.Quantity(1, source_pint).to(target_pint).magnitude)
        except pint.errors.DimensionalityError as exc:
            raise UnitsError(
                f"Cannot convert '{self.unit}' to '{target_unit}': {exc}"
            ) from exc
        factor = one - zero
        offset = zero
        return factor, offset

    def can_resolve(self, available: "set[Column] | ColumnDict") -> bool:
        """Check whether this column can be resolved from available columns.

        Args:
            available: Set of available Column and/or BDFColumn objects,
                or a :class:`ColumnDict`.

        Returns:
            True if the column can be resolved, False otherwise.
        """
        try:
            self.resolve(available)
            return True
        except ColumnResolutionError:
            return False

    def _apply_unit_conversion(
        self, source_expr: pl.Expr, source_unit: str | None
    ) -> pl.Expr:
        """Convert resolved expression from source_unit to this column's unit."""
        if source_unit == self.unit:
            return source_expr.alias(self.name)
        source_col = Column(self.quantity, source_unit)
        factor, offset = source_col.conversion_parameters(cast(str, self.unit))
        return _apply_conversion(source_expr, factor, offset, self.name)

    def resolve(self, available: "set[Column] | ColumnDict") -> pl.Expr:
        """Resolve this column to a Polars expression from available columns.

        Resolution strategy:
        1. Exact match: return the column if it's in available.
        2. BDF recipe lookup: if this is not a BDFColumn, try to resolve via
           a BDF member's recipes (which may derive the quantity from others).
        3. Quantity scan: search available columns for matching quantity
           (case-insensitive), then apply unit conversion if needed.

        Args:
            available: Set of available :class:`Column` and/or
                :class:`BDFColumn` objects, or a :class:`ColumnDict`.

        Returns:
            A Polars expression that evaluates to this column's values,
            optionally with unit conversion applied.

        Raises:
            ColumnResolutionError: If no matching column or recipe is found,
                or if units are incompatible.

        Examples:
            >>> col = Column("Current", "mA")
            >>> expr = col.resolve({Column("Current", "A")})
            >>> type(expr).__name__
            'Expr'
        """
        q = self.quantity.lower()
        if isinstance(available, ColumnDict):
            if self.name in available:
                return pl.col(self.name)
            quantity_matches = available.columns_for_quantity(self.quantity)
        else:
            if self in available:
                return pl.col(self.name)
            quantity_matches = tuple(c for c in available if c.quantity.lower() == q)
        resolved_col: Column | BDF | None = None
        base_expr: pl.Expr | None = None
        if not isinstance(self, BDFColumn):
            try:
                resolved_col = BDF.lookup_by_quantity(self.quantity)
                base_expr = resolved_col.resolve(available)
            except (KeyError, ColumnResolutionError):
                pass
        for c in quantity_matches:
            resolved_col = c
            base_expr = pl.col(c.name)

        if resolved_col is not None and base_expr is not None:
            try:
                return self._apply_unit_conversion(base_expr, resolved_col.unit)
            except UnitsError as exc:
                raise ColumnResolutionError(
                    f"Found column '{resolved_col.name}' "
                    f"for quantity '{self.quantity}', "
                    f"but unit '{resolved_col.unit}' is incompatible with target unit "
                    f"'{self.unit}': {exc}"
                ) from exc

        msg = f"Cannot resolve '{self.name}' from available columns"
        raise ColumnResolutionError(msg)


def _resolves_directly(column: Column, available: "set[Column] | ColumnDict") -> bool:
    """Check whether a column resolves from recorded data without recipes.

    Uses :meth:`Column.resolve` explicitly so that :class:`BDFColumn` recipe
    fallback is bypassed: only an exact data-column match or a quantity match
    with unit conversion counts as direct.

    Args:
        column: The column descriptor to check.
        available: Set of available :class:`Column` objects, or a
            :class:`ColumnDict`.

    Returns:
        True if the column resolves directly from recorded columns.
    """
    try:
        Column.resolve(column, available)
        return True
    except ColumnResolutionError:
        return False


@dataclass(frozen=True)
class BDFColumn(Column):
    """A BDF-standard column descriptor with recipe-based derivation metadata.

    Extends :class:`Column` with:

    - Optional :class:`Recipe` list for deriving the quantity from other
      columns when no direct match exists.
    - :attr:`iri` computed from quantity and unit via pint long-form names.
    - :meth:`can_resolve` and :meth:`resolve` that implement the two-step
      resolution chain: exact data-column match first, recipe fallback second.

    Args:
        quantity: The BDF quantity name (e.g. ``"Current"``).
        unit: The unit string (e.g. ``"A"``, ``"Ah"``, ``"1"``).
            Defaults to ``"1"`` for dimensionless columns.
        recipes: Ordered list of :class:`Recipe` objects.

    Attributes:
        recipes: Fallback computation rules, tried in order.

    Examples:
        >>> col = BDFColumn("Current", "A")
        >>> col.name
        'Current / A'
        >>> col.iri
        'https://w3id.org/battery-data-alliance/ontology/battery-data-format#current_ampere'
        >>> col2 = BDFColumn("Step Count")
        >>> col2.name
        'Step Count / 1'
        >>> col2.iri
        'https://w3id.org/battery-data-alliance/ontology/battery-data-format#step_count'
    """

    @property
    def iri(self) -> str:
        """Full BDF ontology IRI for this column.

        The IRI is looked up directly from :data:`bdf.spec.COLUMN_ONTOLOGY`
        using this column's BDF label, keeping PyProBE in lock-step with the
        ontology. Columns that are not part of the ontology fall back to a
        computed IRI of :data:`BDF_IRI_PREFIX` + ``snake_case(quantity)``
        (plus a pint long-form unit suffix for dimensioned columns).

        Returns:
            The IRI string.

        Examples:
            >>> BDFColumn("Voltage", "V").iri
            'https://w3id.org/battery-data-alliance/ontology/battery-data-format#voltage_volt'
            >>> BDFColumn("Step Count").iri
            'https://w3id.org/battery-data-alliance/ontology/battery-data-format#step_count'
        """
        match = _ONTOLOGY.quantity_from_label(self.name)
        if match is not None:
            return match[0].iri
        slug = self.quantity.lower().replace(" ", "_")
        if self.unit in ("1", None):
            return f"{BDF_IRI_PREFIX}{slug}"
        unit_long = (
            str(_ureg.parse_units(_resolve_unit(cast(str, self.unit), self.quantity)))
            .lower()
            .replace(" ", "_")
        )
        return f"{BDF_IRI_PREFIX}{slug}_{unit_long}"

    def resolve(
        self,
        available: "set[Column] | ColumnDict",
        _resolving: "frozenset[BDFColumn] | None" = None,
    ) -> pl.Expr:
        """Resolve this BDF column to a Polars expression.

        Searches available data columns (skipping other :class:`BDFColumn`
        entries) for a matching quantity with compatible units. If no
        direct data match, falls back to recipes in two passes: first the
        recipes whose required columns all resolve directly from recorded
        data, then those needing recipe-derived inputs. Preference order is
        therefore: recorded column, recipe fed by recorded columns, recipe
        fed by derived columns; within each pass, registration order wins.

        Args:
            available: Set of available :class:`Column` and/or
                :class:`BDFColumn` objects, or a :class:`ColumnDict`.
            _resolving: Internal cycle-detection set; do not pass externally.

        Returns:
            A Polars expression that evaluates to this column's values.

        Examples:
            >>> BDF.CURRENT_AMPERE.can_resolve({Column("Current", "mA")})
            True
            >>> BDF.CURRENT_AMPERE.can_resolve({Column("Voltage", "V")})
            False
        """
        if _resolving is None:
            _resolving = frozenset()
        if self in _resolving:
            raise ColumnResolutionError(
                f"Circular recipe dependency detected for '{self.name}'."
            )
        try:
            return super().resolve(available)
        except ColumnResolutionError:
            try:
                recipes = BDF_RECIPES[cast(BDF, self)]
            except KeyError:
                raise ColumnResolutionError(
                    f"Cannot resolve '{self.quantity}' from available columns, "
                    f"and no recipes found."
                ) from None
            child_resolving = _resolving | {self}
            direct: list[Recipe] = []
            chained: list[Recipe] = []
            for recipe in recipes:
                if all(_resolves_directly(req, available) for req in recipe.required):
                    direct.append(recipe)
                else:
                    chained.append(recipe)
            for recipe in direct + chained:
                if all(
                    req.can_resolve_with_guard(available, child_resolving)
                    for req in recipe.required
                ):
                    expr_map: dict[BDF, pl.Expr] = {
                        req: req.resolve_with_guard(available, child_resolving)
                        for req in recipe.required
                    }
                    logger.debug(
                        f"Resolved '{self.quantity}' via recipe with dependencies "
                        f"{{c.quantity for c in expr_map}}."
                    )
                    return recipe.compute(expr_map).alias(self.name)
            raise ColumnResolutionError(
                f"Cannot resolve '{self.name}' from available columns, "
                f"even via recipes with dependencies "
                f"{[c.quantity for recipe in recipes for c in recipe.required]}."
            ) from None

    def resolve_with_guard(
        self,
        available: "set[Column] | ColumnDict",
        resolving: "frozenset[BDFColumn]",
    ) -> pl.Expr:
        """Resolve with an explicit cycle-detection set (internal use)."""
        return self.resolve(available, _resolving=resolving)

    def can_resolve_with_guard(
        self,
        available: "set[Column] | ColumnDict",
        resolving: "frozenset[BDFColumn]",
    ) -> bool:
        """Check resolvability with an explicit cycle-detection set (internal use)."""
        try:
            self.resolve_with_guard(available, resolving)
            return True
        except ColumnResolutionError:
            return False


class BDF(BDFColumn, Enum):
    """Enum of all BDF-standard columns as :class:`BDFColumn` instances."""

    TEST_TIME_SECOND = "Test Time", "s"
    VOLTAGE_VOLT = "Voltage", "V"
    CURRENT_AMPERE = "Current", "A"
    UNIX_TIME_SECOND = "Unix Time", "s"
    STEP_TIME_SECOND = "Step Time", "s"
    CYCLE_COUNT = "Cycle Count", "1"
    STEP_COUNT = "Step Count", "1"
    STEP_ID = "Step ID", None
    STEP_INDEX = "Step Index", "1"
    STEP_TYPE = "Step Type", None
    RECORD_INDEX = "Record Index", "1"
    POWER_WATT = "Power", "W"
    AMBIENT_TEMPERATURE_CELSIUS = "Ambient Temperature", "degC"
    SURFACE_TEMPERATURE_CELSIUS = "Surface Temperature", "degC"
    TEMPERATURE_T1_CELSIUS = "Temperature T1", "degC"
    TEMPERATURE_T2_CELSIUS = "Temperature T2", "degC"
    TEMPERATURE_T3_CELSIUS = "Temperature T3", "degC"
    TEMPERATURE_T4_CELSIUS = "Temperature T4", "degC"
    TEMPERATURE_T5_CELSIUS = "Temperature T5", "degC"
    AMBIENT_PRESSURE_PA = "Ambient Pressure", "Pa"
    APPLIED_PRESSURE_PA = "Applied Pressure", "Pa"
    SURFACE_PRESSURE_PA = "Surface Pressure", "Pa"
    CHARGING_CAPACITY_AH = "Charging Capacity", "Ah"
    DISCHARGING_CAPACITY_AH = "Discharging Capacity", "Ah"
    NET_CAPACITY_AH = "Net Capacity", "Ah"
    CUMULATIVE_CAPACITY_AH = "Cumulative Capacity", "Ah"
    STEP_CHARGING_CAPACITY_AH = "Step Charging Capacity", "Ah"
    STEP_DISCHARGING_CAPACITY_AH = "Step Discharging Capacity", "Ah"
    STEP_NET_CAPACITY_AH = "Step Net Capacity", "Ah"
    STEP_CUMULATIVE_CAPACITY_AH = "Step Cumulative Capacity", "Ah"
    CYCLE_CHARGING_CAPACITY_AH = "Cycle Charging Capacity", "Ah"
    CYCLE_DISCHARGING_CAPACITY_AH = "Cycle Discharging Capacity", "Ah"
    CYCLE_NET_CAPACITY_AH = "Cycle Net Capacity", "Ah"
    CYCLE_CUMULATIVE_CAPACITY_AH = "Cycle Cumulative Capacity", "Ah"
    CHARGING_ENERGY_WH = "Charging Energy", "Wh"
    DISCHARGING_ENERGY_WH = "Discharging Energy", "Wh"
    NET_ENERGY_WH = "Net Energy", "Wh"
    CUMULATIVE_ENERGY_WH = "Cumulative Energy", "Wh"
    STEP_CHARGING_ENERGY_WH = "Step Charging Energy", "Wh"
    STEP_DISCHARGING_ENERGY_WH = "Step Discharging Energy", "Wh"
    STEP_NET_ENERGY_WH = "Step Net Energy", "Wh"
    STEP_CUMULATIVE_ENERGY_WH = "Step Cumulative Energy", "Wh"
    CYCLE_CHARGING_ENERGY_WH = "Cycle Charging Energy", "Wh"
    CYCLE_DISCHARGING_ENERGY_WH = "Cycle Discharging Energy", "Wh"
    CYCLE_NET_ENERGY_WH = "Cycle Net Energy", "Wh"
    CYCLE_CUMULATIVE_ENERGY_WH = "Cycle Cumulative Energy", "Wh"
    INTERNAL_RESISTANCE_OHM = "Internal Resistance", "ohm"
    AC_INTERNAL_RESISTANCE_OHM = "AC Internal Resistance", "ohm"
    DC_INTERNAL_RESISTANCE_OHM = "DC Internal Resistance", "ohm"
    REAL_IMPEDANCE_OHM = "Real Impedance", "ohm"
    IMAGINARY_IMPEDANCE_OHM = "Imaginary Impedance", "ohm"
    ABSOLUTE_IMPEDANCE_OHM = "Absolute Impedance", "ohm"
    FREQUENCY_HERTZ = "Frequency", "Hz"
    PHASE_DEGREE = "Phase", "deg"

    def __str__(self) -> str:
        """Return the BDF column name string.

        Returns:
            The BDF ``"Quantity / unit"`` column name (e.g. ``'Current / A'``).

        Examples:
            >>> str(BDF.CURRENT_AMPERE)
            'Current / A'
            >>> print(BDF.CURRENT_AMPERE)
            Current / A
        """
        return self.name

    @classmethod
    @cache
    def _build_index(cls) -> dict[str, "BDF"]:
        """Builds a lookup dictionary exactly once and caches it in memory."""
        return {member.quantity: member for member in cls}

    @classmethod
    def get(cls, quantity: str, unit: str | None) -> "BDF":
        """Look up a BDF column by exact quantity and unit match.

        Args:
            quantity: The physical quantity name (e.g. ``"Current"``).
            unit: The unit string (e.g. ``"A"``, ``"Ah"``, ``"1"``), or
                ``None`` for unitless quantities (e.g. ``"Step ID"``).

        Returns:
            The matching :class:`BDF` enum member.

        Raises:
            KeyError: If no matching BDF column is found.
        """
        quantity_match = cls.lookup_by_quantity(quantity)
        if quantity_match.unit != unit:
            msg = f"No BDF column for quantity '{quantity}' with unit '{unit}'"
            raise KeyError(msg)
        return quantity_match

    @classmethod
    def lookup_by_quantity(cls, quantity: str) -> "BDF":
        """Look up a BDF column by quantity name, ignoring case and unit.

        Args:
            quantity: The physical quantity name (e.g. ``"Current"``).

        Returns:
            The matching :class:`BDF` enum member.

        Raises:
            KeyError: If no matching BDF column is found.
        """
        index = cls._build_index()

        # Look up the tuple in the dictionary
        match = index.get(quantity)
        if match is None:
            raise KeyError(f"No BDF column for quantity '{quantity}'")
        return match


def _seam_charge(current: pl.Expr, time: pl.Expr, key: pl.Expr) -> pl.Expr:
    """Trapezoidal seam contribution at each key-boundary row, zero elsewhere.

    A step-scoped charging/discharging column resets to 0 at every boundary,
    so the interval spanning the boundary — from the last row of the previous
    step to the first row of the new step — is dropped by a plain diff/clip
    reconstruction. This computes that dropped charge (or energy, if ``current``
    is power) via trapezoidal integration, so it can be added back in.

    Args:
        current: Current (or power) :class:`polars.Expr`, in amperes (or watts).
        time: Elapsed time :class:`polars.Expr`, in seconds.
        key: Group key :class:`polars.Expr` (e.g. step count) marking resets.

    Returns:
        A :class:`polars.Expr` with the seam term at boundary rows, zero
        elsewhere (including ``dt == 0`` duplicate-timestamp boundaries).
    """
    dt = time.diff().fill_null(0.0)
    seam = 0.5 * (current + current.shift(1)) * dt / 3600.0
    is_boundary = key.cast(pl.Int64).diff().fill_null(0) != 0
    return pl.when(is_boundary).then(seam).otherwise(0.0).fill_null(0.0)


def _step_diffs_and_seam(
    columns: dict["BDF", pl.Expr],
    ch_col: "BDF",
    dch_col: "BDF",
    current_col: "BDF",
    time_col: "BDF",
    key_col: "BDF",
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """Shared diff/clip and seam terms for step-level reconstructions.

    Removes step-reset artifacts from the charging/discharging columns via
    diff/clip and computes the trapezoidal seam charge dropped at each step
    boundary (see :func:`_seam_charge`).

    Args:
        columns: Mapping of resolved :class:`BDF` expressions.
        ch_col: Step-level charging :class:`BDF` column.
        dch_col: Step-level discharging :class:`BDF` column.
        current_col: Current (or power) :class:`BDF` column for the seam term.
        time_col: Elapsed-time :class:`BDF` column for the seam term.
        key_col: Group key :class:`BDF` column marking step boundaries.

    Returns:
        A ``(diff_charge, diff_discharge, seam)`` tuple of expressions.
    """
    charge = columns[ch_col].cast(pl.Float64)
    discharge = columns[dch_col].cast(pl.Float64)
    current = columns[current_col].cast(pl.Float64)
    time = columns[time_col].cast(pl.Float64)
    key = columns[key_col]
    diff_charge = charge.diff().clip(lower_bound=0).fill_null(strategy="zero")
    diff_discharge = discharge.diff().clip(lower_bound=0).fill_null(strategy="zero")
    seam = _seam_charge(current, time, key)
    return diff_charge, diff_discharge, seam


def _global_net_from_step_ch_dch(
    ch_col: "BDF",
    dch_col: "BDF",
    current_col: "BDF",
    time_col: "BDF",
    key_col: "BDF",
) -> Recipe:
    """Recipe for global net from step-level charging/discharging columns.

    Removes step-reset artifacts via diff/clip before accumulating the signed
    global integral, adding back the trapezoidal seam charge dropped at each
    step boundary (see :func:`_seam_charge`).

    Args:
        ch_col: Step-level charging :class:`BDF` column.
        dch_col: Step-level discharging :class:`BDF` column.
        current_col: Current (or power) :class:`BDF` column for the seam term.
        time_col: Elapsed-time :class:`BDF` column for the seam term.
        key_col: Group key :class:`BDF` column marking step boundaries.

    Returns:
        A :class:`Recipe` deriving the target column.
    """

    def _compute(columns: dict["BDF", pl.Expr]) -> pl.Expr:
        diff_charge, diff_discharge, seam = _step_diffs_and_seam(
            columns, ch_col, dch_col, current_col, time_col, key_col
        )
        return (diff_charge - diff_discharge + seam).cum_sum()

    return Recipe(
        required=[ch_col, dch_col, current_col, time_col, key_col],
        compute=_compute,
    )


def _global_cumulative_from_step_ch_dch(
    ch_col: "BDF",
    dch_col: "BDF",
    current_col: "BDF",
    time_col: "BDF",
    key_col: "BDF",
) -> Recipe:
    """Recipe for global cumulative throughput from step-level ch/dch columns.

    Adds back the trapezoidal seam charge dropped at each step boundary (see
    :func:`_seam_charge`).

    Args:
        ch_col: Step-level charging :class:`BDF` column.
        dch_col: Step-level discharging :class:`BDF` column.
        current_col: Current (or power) :class:`BDF` column for the seam term.
        time_col: Elapsed-time :class:`BDF` column for the seam term.
        key_col: Group key :class:`BDF` column marking step boundaries.

    Returns:
        A :class:`Recipe` deriving the target column.
    """

    def _compute(columns: dict["BDF", pl.Expr]) -> pl.Expr:
        diff_charge, diff_discharge, seam = _step_diffs_and_seam(
            columns, ch_col, dch_col, current_col, time_col, key_col
        )
        return diff_charge.cum_sum() + diff_discharge.cum_sum() + seam.abs().cum_sum()

    return Recipe(
        required=[ch_col, dch_col, current_col, time_col, key_col],
        compute=_compute,
    )


def _net_from_cumulative_current(cumul_col: "BDF", current_col: "BDF") -> Recipe:
    """Recipe for signed net from cumulative throughput and current direction.

    Args:
        cumul_col: Cumulative (unsigned throughput) :class:`BDF` column.
        current_col: Current :class:`BDF` column used for sign recovery.

    Returns:
        A :class:`Recipe` deriving the target column.
    """

    def _compute(columns: dict["BDF", pl.Expr]) -> pl.Expr:
        cumul = columns[cumul_col].cast(pl.Float64)
        current = columns[current_col].cast(pl.Float64)
        return (current.sign() * cumul.diff().fill_null(strategy="zero")).cum_sum()

    return Recipe(required=[cumul_col, current_col], compute=_compute)


def _trapz_integral_from_rate(rate_col: "BDF", time_col: "BDF") -> Recipe:
    """Recipe for net capacity/energy as the trapezoidal integral of current/power.

    Reconstructs the signed running integral directly from current (or power)
    and elapsed time, with no dependency on any recorded charging/discharging
    column. This is the least trustworthy reconstruction available -- prefer
    any recipe that uses a recorded charge/energy column when one resolves.

    Args:
        rate_col: Current (or power) :class:`BDF` column.
        time_col: Elapsed-time :class:`BDF` column.

    Returns:
        A :class:`Recipe` deriving the target column.
    """

    def _compute(columns: dict["BDF", pl.Expr]) -> pl.Expr:
        rate = columns[rate_col].cast(pl.Float64)
        time = columns[time_col].cast(pl.Float64)
        dt = time.diff().fill_null(0.0)
        return (0.5 * (rate + rate.shift(1)) * dt / 3600.0).fill_null(0.0).cum_sum()

    return Recipe(required=[rate_col, time_col], compute=_compute)


def _within_net(ch_col: "BDF", dch_col: "BDF") -> Recipe:
    """Recipe for within-scope signed net: ``ch - dch``.

    Valid when both inputs reset at the same step/cycle boundary.

    Args:
        ch_col: Within-scope charging :class:`BDF` column.
        dch_col: Within-scope discharging :class:`BDF` column.

    Returns:
        A :class:`Recipe` deriving the target column.
    """

    def _compute(columns: dict["BDF", pl.Expr]) -> pl.Expr:
        return columns[ch_col] - columns[dch_col]

    return Recipe(required=[ch_col, dch_col], compute=_compute)


def _within_cumulative(ch_col: "BDF", dch_col: "BDF") -> Recipe:
    """Recipe for within-scope cumulative throughput: ``ch + dch``.

    Valid when both inputs reset at the same step/cycle boundary.

    Args:
        ch_col: Within-scope charging :class:`BDF` column.
        dch_col: Within-scope discharging :class:`BDF` column.

    Returns:
        A :class:`Recipe` deriving the target column.
    """

    def _compute(columns: dict["BDF", pl.Expr]) -> pl.Expr:
        return columns[ch_col] + columns[dch_col]

    return Recipe(required=[ch_col, dch_col], compute=_compute)


def _component_from_net(net_col: "BDF", key_col: "BDF | None", sign: float) -> Recipe:
    """Recipe for one signed component of a net column.

    Accumulates the positive increments of ``sign * net``, i.e. the charging
    component for ``sign == 1.0`` and the discharging component for
    ``sign == -1.0``. Valid under the same assumption as
    :func:`_cumulative_from_net`: charge and discharge do not both flow within
    a single sampling interval.

    Args:
        net_col: Signed net :class:`BDF` column.
        key_col: Group key :class:`BDF` column for scoped (step/cycle) net
            columns, so the diff and running sum restart at each scope reset.
            ``None`` for global net columns.
        sign: ``1.0`` to extract the charging component, ``-1.0`` for the
            discharging component.

    Returns:
        A :class:`Recipe` deriving the target column.
    """

    def _compute(columns: dict["BDF", pl.Expr]) -> pl.Expr:
        net = columns[net_col].cast(pl.Float64)
        component = (
            (sign * net.diff()).clip(lower_bound=0).fill_null(strategy="zero").cum_sum()
        )
        if key_col is None:
            return component
        return component.over(columns[key_col])

    required = [net_col] if key_col is None else [net_col, key_col]
    return Recipe(required=required, compute=_compute)


def _charge_from_net(net_col: "BDF", key_col: "BDF | None" = None) -> Recipe:
    """Recipe for charging component: ``cumsum(clip(diff(net), 0))``.

    Args:
        net_col: Signed net :class:`BDF` column.
        key_col: Group key :class:`BDF` column for scoped net columns;
            ``None`` for global net columns.

    Returns:
        A :class:`Recipe` deriving the target column.
    """
    return _component_from_net(net_col, key_col, 1.0)


def _discharge_from_net(net_col: "BDF", key_col: "BDF | None" = None) -> Recipe:
    """Recipe for discharging component: ``cumsum(clip(-diff(net), 0))``.

    Args:
        net_col: Signed net :class:`BDF` column.
        key_col: Group key :class:`BDF` column for scoped net columns;
            ``None`` for global net columns.

    Returns:
        A :class:`Recipe` deriving the target column.
    """
    return _component_from_net(net_col, key_col, -1.0)


def _scope_reset(global_col: "BDF", key_col: "BDF") -> Recipe:
    """Recipe for cross-scope reset: ``val - val.first().over(key)``.

    Subtracts the per-group baseline value so the result starts at zero
    within each step or cycle.

    Args:
        global_col: Global (never-resetting) :class:`BDF` column.
        key_col: Group key :class:`BDF` column (e.g. ``BDF.STEP_COUNT``).

    Returns:
        A :class:`Recipe` deriving the target column.
    """

    def _compute(columns: dict["BDF", pl.Expr]) -> pl.Expr:
        val = columns[global_col]
        key = columns[key_col]
        return val - val.first().over(key)

    return Recipe(required=[global_col, key_col], compute=_compute)


def _time_from_unix_time(columns: dict[BDF, pl.Expr]) -> pl.Expr:
    """Derive elapsed test time from Unix epoch time in seconds.

    Computes successive differences and accumulates them so the result
    starts at zero.

    Args:
        columns: Mapping of ``{unix_time_second: expr}``.

    Returns:
        A :class:`polars.Expr` representing elapsed time in seconds.
    """
    t = columns[BDF.UNIX_TIME_SECOND].cast(pl.Float64)
    return (t - t.first()).alias(BDF.TEST_TIME_SECOND.name)


def _step_count_from_step_id(columns: dict[BDF, pl.Expr]) -> pl.Expr:
    """Derive step count from a Step ID column.

    Increments the step count whenever the step ID changes.

    Args:
        columns: Mapping of ``{step_id: expr}``.

    Returns:
        A :class:`polars.Expr` representing a monotonically increasing step
        count (``UInt64``).
    """
    return (
        columns[BDF.STEP_ID]
        .cast(pl.Int64)
        .diff()
        .fill_null(0)
        .ne(0)
        .cum_sum()
        .cast(pl.UInt64)
    ).alias(BDF.STEP_COUNT.name)


def _cumulative_from_net(net_col: "BDF") -> Recipe:
    """Recipe for cumulative-from-net derivation.

    Computes ``cumsum(|diff(net)|)`` null-filled to zero, giving a
    monotonically non-decreasing cumulative quantity.

    Args:
        net_col: The source net :class:`BDF` column (e.g.
            ``BDF.NET_CAPACITY_AH``).

    Returns:
        A :class:`Recipe` deriving the target column.
    """

    def _compute(columns: dict["BDF", pl.Expr]) -> pl.Expr:
        net = columns[net_col]
        return net.diff().abs().fill_null(strategy="zero").cum_sum()

    return Recipe(required=[net_col], compute=_compute)


BDF_RECIPES: dict[BDF, list[Recipe]] = {
    BDF.TEST_TIME_SECOND: [
        Recipe(required=[BDF.UNIX_TIME_SECOND], compute=_time_from_unix_time)
    ],
    BDF.NET_CAPACITY_AH: [
        _within_net(BDF.CHARGING_CAPACITY_AH, BDF.DISCHARGING_CAPACITY_AH),
        _global_net_from_step_ch_dch(
            BDF.STEP_CHARGING_CAPACITY_AH,
            BDF.STEP_DISCHARGING_CAPACITY_AH,
            BDF.CURRENT_AMPERE,
            BDF.TEST_TIME_SECOND,
            BDF.STEP_COUNT,
        ),
        _net_from_cumulative_current(BDF.CUMULATIVE_CAPACITY_AH, BDF.CURRENT_AMPERE),
        _trapz_integral_from_rate(BDF.CURRENT_AMPERE, BDF.TEST_TIME_SECOND),
    ],
    BDF.NET_ENERGY_WH: [
        _within_net(BDF.CHARGING_ENERGY_WH, BDF.DISCHARGING_ENERGY_WH),
        _global_net_from_step_ch_dch(
            BDF.STEP_CHARGING_ENERGY_WH,
            BDF.STEP_DISCHARGING_ENERGY_WH,
            BDF.POWER_WATT,
            BDF.TEST_TIME_SECOND,
            BDF.STEP_COUNT,
        ),
        _net_from_cumulative_current(BDF.CUMULATIVE_ENERGY_WH, BDF.CURRENT_AMPERE),
        _trapz_integral_from_rate(BDF.POWER_WATT, BDF.TEST_TIME_SECOND),
    ],
    BDF.STEP_COUNT: [Recipe(required=[BDF.STEP_ID], compute=_step_count_from_step_id)],
    BDF.CUMULATIVE_CAPACITY_AH: [
        _within_cumulative(BDF.CHARGING_CAPACITY_AH, BDF.DISCHARGING_CAPACITY_AH),
        _cumulative_from_net(BDF.NET_CAPACITY_AH),
        _global_cumulative_from_step_ch_dch(
            BDF.STEP_CHARGING_CAPACITY_AH,
            BDF.STEP_DISCHARGING_CAPACITY_AH,
            BDF.CURRENT_AMPERE,
            BDF.TEST_TIME_SECOND,
            BDF.STEP_COUNT,
        ),
    ],
    BDF.CUMULATIVE_ENERGY_WH: [
        _within_cumulative(BDF.CHARGING_ENERGY_WH, BDF.DISCHARGING_ENERGY_WH),
        _cumulative_from_net(BDF.NET_ENERGY_WH),
        _global_cumulative_from_step_ch_dch(
            BDF.STEP_CHARGING_ENERGY_WH,
            BDF.STEP_DISCHARGING_ENERGY_WH,
            BDF.POWER_WATT,
            BDF.TEST_TIME_SECOND,
            BDF.STEP_COUNT,
        ),
    ],
    # Global charging/discharging components
    BDF.CHARGING_CAPACITY_AH: [_charge_from_net(BDF.NET_CAPACITY_AH)],
    BDF.DISCHARGING_CAPACITY_AH: [_discharge_from_net(BDF.NET_CAPACITY_AH)],
    BDF.CHARGING_ENERGY_WH: [_charge_from_net(BDF.NET_ENERGY_WH)],
    BDF.DISCHARGING_ENERGY_WH: [_discharge_from_net(BDF.NET_ENERGY_WH)],
    # Step-scope capacity
    BDF.STEP_NET_CAPACITY_AH: [
        _within_net(BDF.STEP_CHARGING_CAPACITY_AH, BDF.STEP_DISCHARGING_CAPACITY_AH),
        _scope_reset(BDF.NET_CAPACITY_AH, BDF.STEP_COUNT),
    ],
    BDF.STEP_CUMULATIVE_CAPACITY_AH: [
        _within_cumulative(
            BDF.STEP_CHARGING_CAPACITY_AH, BDF.STEP_DISCHARGING_CAPACITY_AH
        ),
        _scope_reset(BDF.CUMULATIVE_CAPACITY_AH, BDF.STEP_COUNT),
    ],
    BDF.STEP_CHARGING_CAPACITY_AH: [
        _scope_reset(BDF.CHARGING_CAPACITY_AH, BDF.STEP_COUNT),
        _charge_from_net(BDF.STEP_NET_CAPACITY_AH, BDF.STEP_COUNT),
    ],
    BDF.STEP_DISCHARGING_CAPACITY_AH: [
        _scope_reset(BDF.DISCHARGING_CAPACITY_AH, BDF.STEP_COUNT),
        _discharge_from_net(BDF.STEP_NET_CAPACITY_AH, BDF.STEP_COUNT),
    ],
    # Cycle-scope capacity
    BDF.CYCLE_NET_CAPACITY_AH: [
        _within_net(BDF.CYCLE_CHARGING_CAPACITY_AH, BDF.CYCLE_DISCHARGING_CAPACITY_AH),
        _scope_reset(BDF.NET_CAPACITY_AH, BDF.CYCLE_COUNT),
    ],
    BDF.CYCLE_CUMULATIVE_CAPACITY_AH: [
        _within_cumulative(
            BDF.CYCLE_CHARGING_CAPACITY_AH, BDF.CYCLE_DISCHARGING_CAPACITY_AH
        ),
        _scope_reset(BDF.CUMULATIVE_CAPACITY_AH, BDF.CYCLE_COUNT),
    ],
    BDF.CYCLE_CHARGING_CAPACITY_AH: [
        _scope_reset(BDF.CHARGING_CAPACITY_AH, BDF.CYCLE_COUNT),
        _charge_from_net(BDF.CYCLE_NET_CAPACITY_AH, BDF.CYCLE_COUNT),
    ],
    BDF.CYCLE_DISCHARGING_CAPACITY_AH: [
        _scope_reset(BDF.DISCHARGING_CAPACITY_AH, BDF.CYCLE_COUNT),
        _discharge_from_net(BDF.CYCLE_NET_CAPACITY_AH, BDF.CYCLE_COUNT),
    ],
    # Step-scope energy
    BDF.STEP_NET_ENERGY_WH: [
        _within_net(BDF.STEP_CHARGING_ENERGY_WH, BDF.STEP_DISCHARGING_ENERGY_WH),
        _scope_reset(BDF.NET_ENERGY_WH, BDF.STEP_COUNT),
    ],
    BDF.STEP_CUMULATIVE_ENERGY_WH: [
        _within_cumulative(BDF.STEP_CHARGING_ENERGY_WH, BDF.STEP_DISCHARGING_ENERGY_WH),
        _scope_reset(BDF.CUMULATIVE_ENERGY_WH, BDF.STEP_COUNT),
    ],
    BDF.STEP_CHARGING_ENERGY_WH: [
        _scope_reset(BDF.CHARGING_ENERGY_WH, BDF.STEP_COUNT),
        _charge_from_net(BDF.STEP_NET_ENERGY_WH, BDF.STEP_COUNT),
    ],
    BDF.STEP_DISCHARGING_ENERGY_WH: [
        _scope_reset(BDF.DISCHARGING_ENERGY_WH, BDF.STEP_COUNT),
        _discharge_from_net(BDF.STEP_NET_ENERGY_WH, BDF.STEP_COUNT),
    ],
    # Cycle-scope energy
    BDF.CYCLE_NET_ENERGY_WH: [
        _within_net(BDF.CYCLE_CHARGING_ENERGY_WH, BDF.CYCLE_DISCHARGING_ENERGY_WH),
        _scope_reset(BDF.NET_ENERGY_WH, BDF.CYCLE_COUNT),
    ],
    BDF.CYCLE_CUMULATIVE_ENERGY_WH: [
        _within_cumulative(
            BDF.CYCLE_CHARGING_ENERGY_WH, BDF.CYCLE_DISCHARGING_ENERGY_WH
        ),
        _scope_reset(BDF.CUMULATIVE_ENERGY_WH, BDF.CYCLE_COUNT),
    ],
    BDF.CYCLE_CHARGING_ENERGY_WH: [
        _scope_reset(BDF.CHARGING_ENERGY_WH, BDF.CYCLE_COUNT),
        _charge_from_net(BDF.CYCLE_NET_ENERGY_WH, BDF.CYCLE_COUNT),
    ],
    BDF.CYCLE_DISCHARGING_ENERGY_WH: [
        _scope_reset(BDF.DISCHARGING_ENERGY_WH, BDF.CYCLE_COUNT),
        _discharge_from_net(BDF.CYCLE_NET_ENERGY_WH, BDF.CYCLE_COUNT),
    ],
}


def column_factory(quantity: str, unit: str | None = "1") -> "Column | BDF":
    """Create a Column or return a BDF enum member if available.

    Returns a BDF enum member if one exists for the given quantity and unit
    (including unitless BDF columns such as ``"Step ID"`` where ``unit`` is
    ``None``), otherwise creates a new :class:`Column`. A ``None`` unit that
    does not match a unitless BDF column falls back to the dimensionless
    unit ``"1"``.
    """
    try:
        return BDF.get(quantity, unit)
    except KeyError:
        return Column(quantity, "1" if unit is None else unit)


def column_factory_from_string(name: str, pattern: str = BDF_PATTERN) -> "Column | BDF":
    """Parse a column name string and return a Column or BDF member.

    Splits ``name`` into quantity and unit using the two capture groups in
    ``pattern``, then delegates to :func:`column_factory`.  The default
    ``pattern`` (:data:`BDF_PATTERN`) recognises ``"Quantity / unit"`` strings,
    but any two-group regex can be supplied for other naming conventions.

    Args:
        name: The column name string to parse.
        pattern: A regex with two capture groups ``(quantity, unit)``.
            Defaults to :data:`BDF_PATTERN`.

    Returns:
        The matching :class:`BDF` member when the parsed quantity and unit
        identify a BDF-standard column; otherwise a new :class:`Column`.
    """
    quantity, unit = _split_quantity_unit(name, pattern)
    return column_factory(quantity, unit)


class ColumnDict(Mapping[str, Column]):
    """Per-DataFrame resolved column context.

    Thin wrapper around a list of available column names. Resolution is
    delegated to :meth:`Column.can_resolve`, :meth:`Column.resolve`,
    :meth:`BDFColumn.can_resolve`, and :meth:`BDFColumn.resolve`.

    Implements :class:`collections.abc.Mapping` for direct lookup by
    raw column-name key.

    Provides:

    - :meth:`resolve` — select a Polars expression with optional unit conversion.
    - :meth:`can_resolve` — check whether a column can be resolved.
    - :attr:`names` — tuple of available column name strings.
    - :attr:`quantities` — tuple of available quantity strings.

    Args:
        available_columns: Column name strings present in the source DataFrame.

    Examples:
        >>> cs = ColumnDict(["Current / A", "Voltage / V"])
        >>> expr = cs.resolve("Current / A")
        >>> type(expr).__name__
        'Expr'
    """

    names: tuple[str, ...]
    """Tuple of available column name strings, in the same order as the source."""

    quantities: tuple[str, ...]
    """Tuple of available quantity strings, in the same order as the source."""

    def __init__(self, available_columns: list[str]) -> None:
        """Initialise a ColumnDict with the given available column names.

        Parses each column name string into a :class:`Column` or :class:`BDF`
        enum member (if a BDF-standard column).

        Args:
            available_columns: Column name strings present in the source
                DataFrame (in BDF format, e.g. "Current / A").
        """
        self.names = tuple(available_columns)
        parsed = [column_factory_from_string(name) for name in self.names]
        self.quantities: tuple[str, ...] = tuple(c.quantity for c in parsed)
        by_name: dict[str, Column] = dict(zip(self.names, parsed, strict=False))
        quantity_index: dict[str, list[Column]] = {}
        for col in parsed:
            quantity_index.setdefault(col.quantity.lower(), []).append(col)
        by_quantity: dict[str, tuple[Column, ...]] = {
            quantity: tuple(cols) for quantity, cols in quantity_index.items()
        }
        self._columns_by_name: Mapping[str, Column] = MappingProxyType(by_name)
        self._columns_by_quantity: Mapping[str, tuple[Column, ...]] = MappingProxyType(
            by_quantity
        )

    def columns_for_quantity(self, quantity: str) -> tuple[Column, ...]:
        """Return parsed columns that match quantity, ignoring case."""
        return self._columns_by_quantity.get(quantity.lower(), ())

    def __getitem__(self, key: str) -> Column:
        """Return the parsed Column descriptor for an exact column-name key."""
        return self._columns_by_name[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate available raw column names in insertion order."""
        return iter(self._columns_by_name)

    def __len__(self) -> int:
        """Return number of available raw column names."""
        return len(self._columns_by_name)

    def resolve(self, column: str | Column) -> pl.Expr:
        """Select a column expression, optionally converting units.

        String inputs are parsed via :func:`column_factory_from_string`.
        An exact raw-string match short-circuits to :func:`polars.col`
        directly (handling non-BDF column names like ``"Step"``). Otherwise
        resolution is delegated to :meth:`Column.resolve` or
        :meth:`BDFColumn.resolve`, which handle quantity matching, recipe
        derivation, and unit conversion.

        Args:
            column: A column name string or :class:`Column` /
                :class:`BDFColumn` descriptor. Strings are parsed via
                :func:`column_factory_from_string`.

        Returns:
            A Polars expression producing values in the requested unit.

        Raises:
            ColumnResolutionError: If no matching column can be resolved.
        """
        if isinstance(column, str):
            if column in self:
                return pl.col(column)
            column = column_factory_from_string(column)
        return column.resolve(self)

    def can_resolve(self, column: str | Column) -> bool:
        """Check whether a column can be resolved from available data.

        Delegates to :meth:`Column.can_resolve` or
        :meth:`BDFColumn.can_resolve`, which search the combined
        resolution context (data columns and derivable BDF columns).

        Args:
            column: A column name string or :class:`Column` /
                :class:`BDFColumn` descriptor. Strings are parsed via
                :meth:`Column.from_string`.

        Returns:
            True if :meth:`col` would succeed for this column.
        """
        if isinstance(column, str):
            if column in self:
                return True
            column = column_factory_from_string(column)
        return column.can_resolve(self)

    def __contains__(self, item: object) -> bool:
        """Check whether a column name is available.

        Args:
            item: The column name to check.

        Returns:
            True if the column name is present.

        Examples:
            >>> cs = ColumnDict(["Current / A", "Voltage / V"])
            >>> "Current / A" in cs
            True
            >>> "Step Count / 1" in cs
            False
        """
        return isinstance(item, str) and item in self._columns_by_name

    def __repr__(self) -> str:
        """Return a mapping-style representation of available columns.

        Keys are raw column-name strings and values are parsed descriptors.
        BDF values are shown as enum references (for example,
        ``BDF.CURRENT_AMPERE``) to make the structure explicit while staying
        compact.

        Returns:
            A string describing the column-name-to-descriptor mapping.

        Examples:
            >>> cs = ColumnDict(["Current / A", "Custom / 1"])
            >>> repr(cs)  # doctest: +ELLIPSIS
            "ColumnDict({'Current / A': BDF.CURRENT_AMPERE, ...})"
        """
        parts = []
        for name, col in self.items():
            value_repr = f"BDF.{col._name_}" if isinstance(col, BDF) else repr(col)
            parts.append(f"{name!r}: {value_repr}")
        return f"{self.__class__.__name__}({{{', '.join(parts)}}})"


class CurveColumns(ColumnDict):
    """A :class:`ColumnDict` variant exposing the ``x`` and ``y`` axis roles.

    Used by :class:`~pyprobe.result.Curve` to describe its two BDF quantities
    as resolvable columns. The variant carries exactly two columns — the curve's
    x and y quantities — and adds the :attr:`x` and :attr:`y` accessors that
    return the corresponding :class:`Column` / :class:`BDF` descriptors. The
    standard resolution machinery (:meth:`resolve`, :meth:`can_resolve`) works
    unchanged, since a curve's quantities are presented as ordinary columns.

    Args:
        x: The x-axis quantity as a column name string, :class:`Column`, or
            :class:`BDF` member.
        y: The y-axis quantity as a column name string, :class:`Column`, or
            :class:`BDF` member.

    Examples:
        >>> cc = CurveColumns("Stoichiometry / 1", "Voltage / V")
        >>> cc.x.quantity
        'Stoichiometry'
        >>> cc.y.quantity
        'Voltage'
        >>> cc.can_resolve("Voltage / V")
        True
    """

    def __init__(self, x: str | Column, y: str | Column) -> None:
        """Initialise a CurveColumns from the x and y axis quantities."""
        self._x: Column = column_factory_from_string(x) if isinstance(x, str) else x
        self._y: Column = column_factory_from_string(y) if isinstance(y, str) else y
        super().__init__([self._x.name, self._y.name])

    @property
    def x(self) -> Column:
        """The x-axis quantity descriptor."""
        return self._x

    @property
    def y(self) -> Column:
        """The y-axis quantity descriptor."""
        return self._y
