"""Column abstraction for BDF-standard battery data.

This module provides classes for working with BDF (Battery Data Format)
column names and Polars expressions:

- :class:`Column` — pure descriptor that parses a ``"Quantity / unit"``
  string and computes unit-conversion parameters.
- :class:`BDFColumn` — subclass that adds recipe-based derivation metadata
  and a linked-data IRI.
- :class:`ColumnSet` — per-DataFrame resolution context that selects and
  optionally converts columns, falling back to recipe derivation for
  :class:`BDFColumn` descriptors.

Module-level instances cover 27 BDF-standard quantities (e.g.
:data:`current_ampere`, :data:`voltage_volt`) and are collected in
:data:`ALL_COLUMNS`.  :data:`DEFAULT_COLUMNS` is the core subset that
PyProBE retains after ingestion.

Typical usage::

    from pyprobe.column import current_ampere, DEFAULT_COLUMNS, ColumnSet

    cs = ColumnSet(DEFAULT_COLUMNS)
    # Select Current in milliamps from a DataFrame that has "Current / A".
    expr = cs.col(current_ampere, unit="mA")
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

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
        >>> str(e)  # doctest: +ELLIPSIS
        '...'
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
        self.accessed: set[BDFColumn] = set()

    def __getitem__(self, key: "BDFColumn") -> pl.Expr:
        self.accessed.add(key)
        return super().__getitem__(key)


@dataclass
class Recipe:
    """A computation rule for deriving a :class:`BDFColumn` from other columns.

    A recipe declares which BDF columns are needed (``required``) and
    provides a callable that maps :class:`BDFColumn` instances to resolved
    Polars expressions, returning a new Polars expression.

    The ``__post_init__`` method validates that the compute function accesses
    exactly the columns listed in ``required`` — no more, no fewer.

    Attributes:
        required: :class:`BDFColumn` instances that must be resolvable in the
            source DataFrame (e.g. ``[charging_capacity_ah,
            discharging_capacity_ah]``).
        compute: A callable that receives a ``{BDFColumn: pl.Expr}``
            mapping and returns a :class:`polars.Expr`.

    Examples:
        >>> import polars as pl
        >>> recipe = Recipe(
        ...     required=[charging_capacity_ah, discharging_capacity_ah],
        ...     compute=lambda cols: (
        ...         cols[charging_capacity_ah] - cols[discharging_capacity_ah]
        ...     ),
        ... )
        >>> len(recipe.required)
        2
    """

    required: list["BDFColumn"]
    compute: Callable[[dict["BDFColumn", pl.Expr]], pl.Expr]

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


@dataclass(eq=False)
class Column:
    """A BDF column descriptor: quantity name and unit string.

    Constructed directly or parsed from a string via :meth:`from_string`.
    Supports unit conversion through :meth:`conversion_parameters`.

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
        >>> col.column_name
        'Current / A'
        >>> col = Column.from_string("Current / A")
        >>> col.quantity
        'Current'
        >>> col.column_name
        'Current / A'
        >>> Column("Step").column_name
        'Step / 1'
    """

    quantity: str
    unit: str = "1"

    @classmethod
    def from_string(cls, name: str, pattern: str = BDF_PATTERN) -> "Column":
        """Parse a ``"Quantity / unit"`` string into a :class:`Column`.

        Bare names (no separator) are accepted and yield ``unit="1"``.
        Named columns with an explicit unit round-trip back to their original
        string via :attr:`column_name`.

        Args:
            name: The column name string to parse (e.g. ``"Current / A"`` or
                ``"Step Count / 1"``).
            pattern: A regex pattern with two capture groups (quantity, unit).
                Defaults to :data:`BDF_PATTERN`.

        Returns:
            A new :class:`Column` instance.

        Raises:
            ValueError: If ``name`` does not match ``pattern``.

        Examples:
            >>> col = Column.from_string("Current / A")
            >>> col.quantity
            'Current'
            >>> col.column_name
            'Current / A'
            >>> col2 = Column.from_string("Step Count / 1")
            >>> col2.column_name
            'Step Count / 1'
            >>> col3 = Column.from_string("Step")
            >>> col3.unit
            '1'
            >>> col3.column_name
            'Step / 1'
        """
        quantity, raw_unit = _split_quantity_unit(name, pattern)
        return cls(quantity, raw_unit or "1")

    @property
    def column_name(self) -> str:
        """BDF standard column name string (``"Quantity / unit"``).

        Returns:
            The BDF column name string.

        Examples:
            >>> Column("Current", "A").column_name
            'Current / A'
            >>> Column("Net Capacity", "Ah").column_name
            'Net Capacity / Ah'
            >>> Column("Step Count", "1").column_name
            'Step Count / 1'
            >>> Column("Step").column_name
            'Step / 1'
        """
        return f"{self.quantity} / {self.unit}"

    def __str__(self) -> str:
        """Return the BDF column name string.

        Returns:
            The same value as :attr:`column_name`.
        """
        return self.column_name

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
            ValueError: If this column is dimensionless (``unit == "1"``).
            ValueError: If the units are dimensionally incompatible.

        Examples:
            >>> col = Column.from_string("Current / A")
            >>> col.conversion_parameters("mA")
            (1000.0, 0.0)
        """
        if self.unit == "1":
            raise ValueError(
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
            raise ValueError(msg) from exc
        try:
            target_pint = _ureg.parse_units(target_unit_str)
            zero = float(_ureg.Quantity(0, source_pint).to(target_pint).magnitude)
            one = float(_ureg.Quantity(1, source_pint).to(target_pint).magnitude)
        except pint.errors.DimensionalityError as exc:
            raise ValueError(
                f"Cannot convert '{self.unit}' to '{target_unit}': {exc}"
            ) from exc
        factor = one - zero
        offset = zero
        return factor, offset


@dataclass(eq=False)
class BDFColumn(Column):
    """A BDF-standard column descriptor with recipe-based derivation metadata.

    Extends :class:`Column` with:

    - Optional :class:`Recipe` list for deriving the quantity from other
      columns when no direct match exists.
    - :attr:`iri` computed from quantity and unit via pint long-form names.

    Resolution of BDFColumn descriptors against actual DataFrames is handled
    by :class:`ColumnSet`, which implements the two-step chain:

    1. **Exact match** — column name already present in available columns.
    2. **Recipe fallback** — derive from dependency columns via a
       :class:`Recipe`.

    Args:
        quantity: The BDF quantity name (e.g. ``"Current"``).
        unit: The unit string (e.g. ``"A"``, ``"Ah"``, ``"1"``).
            Defaults to ``"1"`` for dimensionless columns.
        recipes: Ordered list of :class:`Recipe` objects.

    Attributes:
        recipes: Fallback computation rules, tried in order.

    Examples:
        >>> col = BDFColumn("Current", "A")
        >>> col.column_name
        'Current / A'
        >>> col.iri
        'https://w3id.org/battery-data-alliance/ontology/battery-data-format#current_ampere'
        >>> col2 = BDFColumn("Step Count")
        >>> col2.column_name
        'Step Count / 1'
        >>> col2.iri
        'https://w3id.org/battery-data-alliance/ontology/battery-data-format#step_count'
    """

    recipes: list[Recipe] = field(default_factory=list)

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


class ColumnSet:
    """Per-DataFrame resolved column context.

    Created with the list of column names available in a DataFrame.
    Provides a single :meth:`col` method for selecting and optionally
    converting columns.

    Args:
        available_columns: Column name strings present in the source DataFrame.

    Examples:
        >>> cs = ColumnSet(["Current / A", "Voltage / V"])
        >>> cs.col("Current / A")  # doctest: +ELLIPSIS
        <Expr ['col("Current / A")'] at ...>
    """

    def __init__(self, available_columns: list[str]) -> None:
        """Initialise a ColumnSet with the given available column names.

        Args:
            available_columns: Column name strings present in the source
                DataFrame.
        """
        self._available: set[str] = set(available_columns)

    def col(
        self,
        column: str | Column,
        unit: str | None = None,
    ) -> pl.Expr:
        """Select a column expression, optionally converting units.

        Args:
            column: A column name string, :class:`Column`, or
                :class:`BDFColumn`. Strings are parsed via
                :meth:`Column.from_string`.
            unit: Target unit for conversion (e.g. ``"mA"``). When ``None``,
                returns the expression in the column's native unit.

        Returns:
            A Polars expression, aliased to ``"Quantity / unit"`` when
            unit conversion is applied.

        Raises:
            ValueError: If the column cannot be resolved from available
                columns or recipes.

        Examples:
            >>> cs = ColumnSet(["Current / A", "Voltage / V"])
            >>> cs.col("Current / A")  # doctest: +ELLIPSIS
            <Expr ['col("Current / A")'] at ...>
        """
        if isinstance(column, str):
            column = Column.from_string(column)

        if isinstance(column, BDFColumn):
            base_expr = self._resolve_bdf(column)
        else:
            base_expr = pl.col(column.column_name)

        if unit is None:
            return base_expr

        factor, offset = column.conversion_parameters(unit)
        target_name = f"{column.quantity} / {unit}"
        return _apply_conversion(base_expr, factor, offset, target_name)

    def _resolve_bdf(self, col: BDFColumn) -> pl.Expr:
        """Resolve a BDFColumn via exact match or recursive recipe.

        Args:
            col: The BDFColumn descriptor to resolve.

        Returns:
            A Polars expression for the resolved column.

        Raises:
            ValueError: If the column cannot be resolved.
        """
        if col.column_name in self._available:
            return pl.col(col.column_name)

        for recipe in col.recipes:
            expr_map: dict[BDFColumn, pl.Expr] = {}
            all_found = True
            for req_col in recipe.required:
                try:
                    expr_map[req_col] = self._resolve_bdf(req_col)
                except ValueError:
                    all_found = False
                    break
            if all_found:
                logger.debug(
                    "Resolved '%s' via recipe with dependencies %s.",
                    col.quantity,
                    [c.quantity for c in expr_map],
                )
                return recipe.compute(expr_map).alias(col.column_name)

        raise ValueError(f"Cannot resolve '{col.quantity}' from available columns")


test_time_second = BDFColumn(
    quantity="Test Time",
    unit="s",
)
"""BDF Test Time column (base unit: seconds)."""

voltage_volt = BDFColumn(
    quantity="Voltage",
    unit="V",
)
"""BDF Voltage column (base unit: volts)."""

current_ampere = BDFColumn(
    quantity="Current",
    unit="A",
)
"""BDF Current column (base unit: amperes)."""

unix_time_second = BDFColumn(
    quantity="Unix Time",
    unit="s",
)
"""BDF Unix Time column (base unit: seconds)."""

cycle_count = BDFColumn(
    quantity="Cycle Count",
    unit="1",
)
"""BDF Cycle Count column (dimensionless cycle index)."""

step_count = BDFColumn(
    quantity="Step Count",
    unit="1",
)
"""BDF Step Count column (dimensionless integer step index)."""

ambient_temperature_celsius = BDFColumn(
    quantity="Ambient Temperature",
    unit="degC",
)
"""BDF Ambient Temperature column (base unit: degrees Celsius)."""

step_index = BDFColumn(
    quantity="Step Index",
    unit="1",
)
"""BDF Step Index column (dimensionless)."""

charging_capacity_ah = BDFColumn(
    quantity="Charging Capacity",
    unit="Ah",
)
"""BDF Charging Capacity column (base unit: ampere-hours)."""

discharging_capacity_ah = BDFColumn(
    quantity="Discharging Capacity",
    unit="Ah",
)
"""BDF Discharging Capacity column (base unit: ampere-hours)."""

step_capacity_ah = BDFColumn(
    quantity="Step Capacity",
    unit="Ah",
)
"""BDF Step Capacity column (base unit: ampere-hours)."""

net_capacity_ah = BDFColumn(
    quantity="Net Capacity",
    unit="Ah",
)
"""BDF Net Capacity column (base unit: ampere-hours).

Falls back to computing net capacity from charging and discharging sub-columns
when no direct ``Net Capacity`` column is available.
"""

cumulative_capacity_ah = BDFColumn(
    quantity="Cumulative Capacity",
    unit="Ah",
)
"""BDF Cumulative Capacity column (base unit: ampere-hours)."""

charging_energy_wh = BDFColumn(
    quantity="Charging Energy",
    unit="Wh",
)
"""BDF Charging Energy column (base unit: watt-hours)."""

discharging_energy_wh = BDFColumn(
    quantity="Discharging Energy",
    unit="Wh",
)
"""BDF Discharging Energy column (base unit: watt-hours)."""

step_energy_wh = BDFColumn(
    quantity="Step Energy",
    unit="Wh",
)
"""BDF Step Energy column (base unit: watt-hours)."""

net_energy_wh = BDFColumn(
    quantity="Net Energy",
    unit="Wh",
)
"""BDF Net Energy column (base unit: watt-hours)."""

cumulative_energy_wh = BDFColumn(
    quantity="Cumulative Energy",
    unit="Wh",
)
"""BDF Cumulative Energy column (base unit: watt-hours)."""

power_watt = BDFColumn(
    quantity="Power",
    unit="W",
)
"""BDF Power column (base unit: watts)."""

internal_resistance_ohm = BDFColumn(
    quantity="Internal Resistance",
    unit="Ohm",
)
"""BDF Internal Resistance column (base unit: ohms)."""

ambient_pressure_pa = BDFColumn(
    quantity="Ambient Pressure",
    unit="Pa",
)
"""BDF Ambient Pressure column (base unit: pascals)."""

applied_pressure_pa = BDFColumn(
    quantity="Applied Pressure",
    unit="Pa",
)
"""BDF Applied Pressure column (base unit: pascals)."""

temperature_t1_celsius = BDFColumn(
    quantity="Surface Temperature T1",
    unit="degC",
)
"""BDF Surface Temperature T1 column (base unit: degrees Celsius)."""

temperature_t2_celsius = BDFColumn(
    quantity="Surface Temperature T2",
    unit="degC",
)
"""BDF Surface Temperature T2 column (base unit: degrees Celsius)."""

temperature_t3_celsius = BDFColumn(
    quantity="Surface Temperature T3",
    unit="degC",
)
"""BDF Surface Temperature T3 column (base unit: degrees Celsius)."""

temperature_t4_celsius = BDFColumn(
    quantity="Surface Temperature T4",
    unit="degC",
)
"""BDF Surface Temperature T4 column (base unit: degrees Celsius)."""

temperature_t5_celsius = BDFColumn(
    quantity="Surface Temperature T5",
    unit="degC",
)
"""BDF Surface Temperature T5 column (base unit: degrees Celsius)."""


def _capacity_from_ch_dch(columns: dict[BDFColumn, pl.Expr]) -> pl.Expr:
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
    charge = columns[charging_capacity_ah].cast(pl.Float64)
    discharge = columns[discharging_capacity_ah].cast(pl.Float64)
    diff_charge = charge.diff().clip(lower_bound=0).fill_null(strategy="zero")
    diff_discharge = discharge.diff().clip(lower_bound=0).fill_null(strategy="zero")
    return (diff_charge - diff_discharge).cum_sum() + charge.max()


def _time_from_unix_time(columns: dict[BDFColumn, pl.Expr]) -> pl.Expr:
    """Derive elapsed test time from Unix epoch time in seconds.

    Computes successive differences and accumulates them so the result
    starts at zero.

    Args:
        columns: Mapping of ``{unix_time_second: expr}``.

    Returns:
        A :class:`polars.Expr` representing elapsed time in seconds.
    """
    t = columns[unix_time_second].cast(pl.Float64)
    return t - t.first()


def _step_count_from_step_index(columns: dict[BDFColumn, pl.Expr]) -> pl.Expr:
    """Derive step count from a Step Index column.

    Increments the step count whenever the step index changes.

    Args:
        columns: Mapping of ``{step_index: expr}``.

    Returns:
        A :class:`polars.Expr` representing a monotonically increasing step
        count (``UInt64``).
    """
    return columns[step_index].diff().fill_null(0).ne(0).cum_sum().cast(pl.UInt64)


test_time_second.recipes = [
    Recipe(required=[unix_time_second], compute=_time_from_unix_time)
]

net_capacity_ah.recipes = [
    Recipe(
        required=[charging_capacity_ah, discharging_capacity_ah],
        compute=_capacity_from_ch_dch,
    )
]

step_count.recipes = [
    Recipe(required=[step_index], compute=_step_count_from_step_index)
]

ALL_COLUMNS: list[BDFColumn] = [
    test_time_second,
    voltage_volt,
    current_ampere,
    unix_time_second,
    cycle_count,
    step_count,
    ambient_temperature_celsius,
    step_index,
    charging_capacity_ah,
    discharging_capacity_ah,
    step_capacity_ah,
    net_capacity_ah,
    cumulative_capacity_ah,
    charging_energy_wh,
    discharging_energy_wh,
    step_energy_wh,
    net_energy_wh,
    cumulative_energy_wh,
    power_watt,
    internal_resistance_ohm,
    ambient_pressure_pa,
    applied_pressure_pa,
    temperature_t1_celsius,
    temperature_t2_celsius,
    temperature_t3_celsius,
    temperature_t4_celsius,
    temperature_t5_celsius,
]
"""All 27 BDF-standard BDFColumn instances in canonical order."""
