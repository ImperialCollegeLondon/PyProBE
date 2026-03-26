"""Column abstraction for BDF-standard battery data.

This module provides classes for working with BDF (Battery Data Format)
column names and Polars expressions:

- :class:`Column` — pure descriptor that parses a ``"Quantity / unit"``
  string and computes unit-conversion parameters. Owns resolution logic
  via :meth:`~Column.can_resolve` and :meth:`~Column.resolve`.
- :class:`BDFColumn` — subclass that adds recipe-based derivation metadata
  and a linked-data IRI. Extends resolution to cover recipe derivation via
  :meth:`~BDFColumn.can_resolve` and :meth:`~BDFColumn.resolve`.
- :class:`ColumnSet` — thin per-DataFrame wrapper that delegates resolution
  to :class:`Column` / :class:`BDFColumn` methods.

The :class:`BDF` enum provides all 27 BDF-standard quantities as members
(e.g. :attr:`BDF.CURRENT_AMPERE`, :attr:`BDF.VOLTAGE_VOLT`).
:data:`DEFAULT_COLUMNS` is the core subset that PyProBE retains after
ingestion.

Typical usage::

    from pyprobe.column import BDF, DEFAULT_COLUMNS, ColumnSet

    cs = ColumnSet(DEFAULT_COLUMNS)
    # Select Current in milliamps from a DataFrame that has "Current / A".
    expr = cs.resolve("Current / mA")
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import cache
from typing import Any, cast

import pint
import polars as pl
from loguru import logger

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
    "Step Index / 1",
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
    unit: str = "1"

    @property
    def name(self) -> str:
        """BDF standard column name string (``"Quantity / unit"``).

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
        """
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
        if self.unit == "1":
            raise UnitsError(
                f"Column '{self.quantity}' is dimensionless; cannot convert."
            )
        source_unit_str = _resolve_unit(self.unit, self.quantity)
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

    def can_resolve(self, available: "set[Column]") -> bool:
        """Check whether this column can be resolved from available columns.

        Args:
            available: Set of available Column and/or BDFColumn objects.

        Returns:
            True if the column can be resolved, False otherwise.
        """
        try:
            self.resolve(available)
            return True
        except ColumnResolutionError:
            return False

    def _apply_unit_conversion(self, source_expr: pl.Expr, source_unit: str) -> pl.Expr:
        """Convert resolved expression from source_unit to this column's unit."""
        if source_unit == self.unit:
            return source_expr.alias(self.name)
        source_col = Column(self.quantity, source_unit)
        factor, offset = source_col.conversion_parameters(self.unit)
        return _apply_conversion(source_expr, factor, offset, self.name)

    def resolve(self, available: "set[Column]") -> pl.Expr:
        """Resolve this column to a Polars expression from available columns.

        Resolution strategy:
        1. Exact match: return the column if it's in available.
        2. BDF recipe lookup: if this is not a BDFColumn, try to resolve via
           a BDF member's recipes (which may derive the quantity from others).
        3. Quantity scan: search available columns for matching quantity
           (case-insensitive), then apply unit conversion if needed.

        Args:
            available: Set of available :class:`Column` and/or
                :class:`BDFColumn` objects.

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
        if self in available:
            return pl.col(self.name)
        q = self.quantity.lower()
        col: Column | BDF | None = None
        base_expr = None
        if not isinstance(self, BDFColumn):
            try:
                col = BDF.lookup_by_quantity(self.quantity)
                base_expr = col.resolve(available)
            except (KeyError, ColumnResolutionError):
                pass
        for c in available:
            if c.quantity.lower() == q:
                col = c
                base_expr = pl.col(c.name)

        if col is not None and base_expr is not None:
            try:
                return self._apply_unit_conversion(base_expr, col.unit)
            except UnitsError as exc:
                raise ColumnResolutionError(
                    f"Found column '{c.name}' for quantity '{self.quantity}', "
                    f"but unit '{c.unit}' is incompatible with target unit "
                    f"'{self.unit}': {exc}"
                ) from exc

        msg = f"Cannot resolve '{self.quantity}' from available columns"
        raise ColumnResolutionError(msg)


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
        """Full BDF ontology IRI, computed from quantity and unit.

        The IRI is built as :data:`BDF_IRI_PREFIX` +
        ``snake_case(quantity)`` + ``_`` + ``pint_long_form(unit)``.
        Dimensionless columns (unit ``"1"``) omit the unit suffix.
        "Surface Temperature" quantities have the "Surface " prefix
        stripped to match the BDF ontology convention.

        Returns:
            The IRI string.

        Examples:
            >>> BDFColumn("Voltage", "V").iri
            'https://w3id.org/battery-data-alliance/ontology/battery-data-format#voltage_volt'
            >>> BDFColumn("Step Count").iri
            'https://w3id.org/battery-data-alliance/ontology/battery-data-format#step_count'
        """
        quantity = self.quantity
        if quantity.startswith("Surface "):
            quantity = quantity.removeprefix("Surface ")
        slug = quantity.lower().replace(" ", "_")
        if self.unit == "1":
            return f"{BDF_IRI_PREFIX}{slug}"
        unit_long = (
            str(_ureg.parse_units(_resolve_unit(self.unit, quantity)))
            .lower()
            .replace(" ", "_")
        )
        return f"{BDF_IRI_PREFIX}{slug}_{unit_long}"

    def resolve(self, available: "set[Column]") -> pl.Expr:
        """Resolve this BDF column to a Polars expression.

        Searches available data columns (skipping other :class:`BDFColumn`
        entries) for a matching quantity with compatible units. If no
        direct data match, checks whether at least one recipe has all its
        required columns resolvable.

        Args:
            available: List of available :class:`Column` and/or
                :class:`BDFColumn` objects.

        Returns:
            A Polars expression that evaluates to this column's values.

        Examples:
            >>> BDF.CURRENT_AMPERE.can_resolve({Column("Current", "mA")})
            True
            >>> BDF.CURRENT_AMPERE.can_resolve({Column("Voltage", "V")})
            False
        """
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
            for recipe in recipes:
                if all(req.can_resolve(available) for req in recipe.required):
                    expr_map: dict[BDF, pl.Expr] = {
                        req: req.resolve(available) for req in recipe.required
                    }
                    logger.debug(
                        "Resolved '%s' via recipe with dependencies %s.",
                        self.quantity,
                        [c.quantity for c in expr_map],
                    )
                    return recipe.compute(expr_map).alias(self.name)
            raise ColumnResolutionError(
                f"Cannot resolve '{self.quantity}' from available columns, "
                f"even via recipes with dependencies "
                f"{[c.quantity for recipe in recipes for c in recipe.required]}."
            ) from None


class BDF(BDFColumn, Enum):
    """Enum of all BDF-standard columns as :class:`BDFColumn` instances."""

    TEST_TIME_SECOND = "Test Time", "s"
    VOLTAGE_VOLT = "Voltage", "V"
    CURRENT_AMPERE = "Current", "A"
    UNIX_TIME_SECOND = "Unix Time", "s"
    CYCLE_COUNT = "Cycle Count", "1"
    STEP_COUNT = "Step Count", "1"
    STEP_INDEX = "Step Index", "1"
    AMBIENT_TEMPERATURE_CELSIUS = "Ambient Temperature", "degC"
    CHARGING_CAPACITY_AH = "Charging Capacity", "Ah"
    DISCHARGING_CAPACITY_AH = "Discharging Capacity", "Ah"
    STEP_CAPACITY_AH = "Step Capacity", "Ah"
    NET_CAPACITY_AH = "Net Capacity", "Ah"
    CUMULATIVE_CAPACITY_AH = "Cumulative Capacity", "Ah"
    CHARGING_ENERGY_WH = "Charging Energy", "Wh"
    DISCHARGING_ENERGY_WH = "Discharging Energy", "Wh"
    STEP_ENERGY_WH = "Step Energy", "Wh"
    NET_ENERGY_WH = "Net Energy", "Wh"
    CUMULATIVE_ENERGY_WH = "Cumulative Energy", "Wh"
    POWER_WATT = "Power", "W"
    INTERNAL_RESISTANCE_OHM = "Internal Resistance", "Ohm"
    AMBIENT_PRESSURE_PA = "Ambient Pressure", "Pa"
    APPLIED_PRESSURE_PA = "Applied Pressure", "Pa"
    TEMPERATURE_T1_CELCIUS = "Surface Temperature T1", "degC"
    TEMPERATURE_T2_CELCIUS = "Surface Temperature T2", "degC"
    TEMPERATURE_T3_CELCIUS = "Surface Temperature T3", "degC"
    TEMPERATURE_T4_CELCIUS = "Surface Temperature T4", "degC"
    TEMPERATURE_T5_CELCIUS = "Surface Temperature T5", "degC"

    @classmethod
    @cache
    def _build_index(cls) -> dict[str, "BDF"]:
        """Builds a lookup dictionary exactly once and caches it in memory."""
        return {member.quantity: member for member in cls}

    @classmethod
    def get(cls, quantity: str, unit: str) -> "BDF":
        """Look up a BDF column by exact quantity and unit match.

        Args:
            quantity: The physical quantity name (e.g. ``"Current"``).
            unit: The unit string (e.g. ``"A"``, ``"Ah"``, ``"1"``).

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


def _capacity_from_ch_dch(columns: dict[BDF, pl.Expr]) -> pl.Expr:
    """Derive net capacity from charging and discharging capacity columns.

    Computes incremental charge and discharge deltas, sums them, and offsets
    by the maximum observed charge capacity so that the result starts near
    zero.

    Args:
        columns: Mapping of ``{charging_capacity_ah: expr,
            discharging_capacity_ah: expr}``.

    Returns:
        A :class:`polars.Expr` representing net capacity in the same unit as
        the input columns.
    """
    charge = columns[BDF.CHARGING_CAPACITY_AH].cast(pl.Float64)
    discharge = columns[BDF.DISCHARGING_CAPACITY_AH].cast(pl.Float64)
    diff_charge = charge.diff().clip(lower_bound=0).fill_null(strategy="zero")
    diff_discharge = discharge.diff().clip(lower_bound=0).fill_null(strategy="zero")
    net_capacity = ((diff_charge - diff_discharge).cum_sum() + charge.max()).alias(
        BDF.NET_CAPACITY_AH.name
    )
    return net_capacity


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


def _step_count_from_step_index(columns: dict[BDF, pl.Expr]) -> pl.Expr:
    """Derive step count from a Step Index column.

    Increments the step count whenever the step index changes.

    Args:
        columns: Mapping of ``{step_index: expr}``.

    Returns:
        A :class:`polars.Expr` representing a monotonically increasing step
        count (``UInt64``).
    """
    return (
        columns[BDF.STEP_INDEX]
        .cast(pl.Int64)
        .diff()
        .fill_null(0)
        .ne(0)
        .cum_sum()
        .cast(pl.UInt64)
    ).alias(BDF.STEP_COUNT.name)


BDF_RECIPES: dict[BDF, list[Recipe]] = {
    BDF.TEST_TIME_SECOND: [
        Recipe(required=[BDF.UNIX_TIME_SECOND], compute=_time_from_unix_time)
    ],
    BDF.NET_CAPACITY_AH: [
        Recipe(
            required=[
                BDF.CHARGING_CAPACITY_AH,
                BDF.DISCHARGING_CAPACITY_AH,
            ],
            compute=_capacity_from_ch_dch,
        )
    ],
    BDF.STEP_COUNT: [
        Recipe(required=[BDF.STEP_INDEX], compute=_step_count_from_step_index)
    ],
}


def column_factory(quantity: str, unit: str = "1") -> "Column | BDF":
    """Create a Column or return a BDF enum member if available.

    Returns a BDF enum member if one exists for the given quantity and unit,
    otherwise creates a new Column.
    """
    try:
        return BDF.get(quantity, unit)
    except KeyError:
        return Column(quantity, unit)


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
    return column_factory(quantity, unit or "1")


class ColumnSet:
    """Per-DataFrame resolved column context.

    Thin wrapper around a list of available column names. Resolution is
    delegated to :meth:`Column.can_resolve`, :meth:`Column.resolve`,
    :meth:`BDFColumn.can_resolve`, and :meth:`BDFColumn.resolve`.

    Provides:

    - :meth:`resolve` — select a Polars expression with optional unit conversion.
    - :meth:`can_resolve` — check whether a column can be resolved.
    - :attr:`names` — list of available column name strings.
    - :attr:`quantities` — list of available quantity strings.

    Args:
        available_columns: Column name strings present in the source DataFrame.

    Examples:
        >>> cs = ColumnSet(["Current / A", "Voltage / V"])
        >>> expr = cs.resolve("Current / A")
        >>> type(expr).__name__
        'Expr'
    """

    def __init__(self, available_columns: list[str]) -> None:
        """Initialise a ColumnSet with the given available column names.

        Parses each column name string into a :class:`Column` or :class:`BDF`
        enum member (if a BDF-standard column). The ``_columns`` list contains
        the parsed descriptors used for resolution and unit conversion.

        Args:
            available_columns: Column name strings present in the source
                DataFrame (in BDF format, e.g. "Current / A").
        """
        self._columns: list[Column] = [
            column_factory_from_string(name) for name in available_columns
        ]

    @property
    def names(self) -> list[str]:
        """Return the column names as a list of strings.

        Returns:
            List of column name strings.

        Examples:
            >>> cs = ColumnSet(["Current / A", "Voltage / V"])
            >>> cs.names
            ['Current / A', 'Voltage / V']
        """
        return [c.name for c in self._columns]

    @property
    def quantities(self) -> list[str]:
        """Return the column quantities as a list of strings.

        Returns:
            List of column quantity strings.

        Examples:
            >>> cs = ColumnSet(["Current / A", "Voltage / V"])
            >>> cs.quantities
            ['Current', 'Voltage']
        """
        return [c.quantity for c in self._columns]

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
            column = column_factory_from_string(column)
        return column.resolve(set(self._columns))

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
            column = column_factory_from_string(column)
        return column.can_resolve(set(self._columns))

    def __contains__(self, item: object) -> bool:
        """Check whether a column name is available.

        Args:
            item: The column name to check.

        Returns:
            True if the column name is present.

        Examples:
            >>> cs = ColumnSet(["Current / A", "Voltage / V"])
            >>> "Current / A" in cs
            True
            >>> "Step Count / 1" in cs
            False
        """
        return item in self.names
