"""A module for the filtering classes."""

import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import polars as pl

from pyprobe import utils
from pyprobe.columns import BDF, Column
from pyprobe.rawdata import CyclingData

if TYPE_CHECKING:
    from pyprobe.pyprobe_types import (
        FilterToCycleType,
    )


from loguru import logger

IndexType = int | Sequence[int] | slice


def _include_preceding_row(mask: pl.Expr) -> pl.Expr:
    """Extend a boolean mask to include the row preceding each contiguous True run.

    Args:
        mask: A boolean polars expression.

    Returns:
        pl.Expr: A boolean expression that is True for every originally selected
            row plus the row immediately before each run of selected rows.
    """
    return mask | mask.shift(-1).fill_null(False)


def _group_start_expr(col_expr: pl.Expr, condition: pl.Expr) -> pl.Expr:
    """Return a boolean expression that is True on the first row of each matching group.

    Args:
        col_expr: Resolved polars expression for the column to rank groups on.
        condition: Expression selecting rows that belong to matching groups.

    Returns:
        pl.Expr: Boolean expression True at the start of each matching group.
    """
    event_rank = col_expr.rank("dense")
    prev_matching_rank = (
        pl.when(condition).then(event_rank).otherwise(None).forward_fill().shift()
    )
    return condition & (event_rank != prev_matching_rank.fill_null(-1))


def _is_bounded(index: "Sequence[int] | slice") -> bool:
    """Return True when the index is a bounded slice (step=1, finite non-zero stop).

    Args:
        index: A sequence or slice to test.

    Returns:
        bool: True when index is a slice with step=1 and a finite, non-zero stop.
    """
    if not isinstance(index, slice):
        return False
    s = index
    return (s.step is None or s.step == 1) and (s.stop is not None and s.stop != 0)


def _needs_collect(index: "IndexType | None") -> bool:
    """Return True when rank values must be collected before iteration.

    Args:
        index: The index argument passed to a filter method.

    Returns:
        bool: True when the group count or positions cannot be determined from
            the index alone without querying the data.
    """
    if index is None:
        return True
    if isinstance(index, int):
        return False
    if isinstance(index, slice):
        if not _is_bounded(index):
            return True
        s = index
        start = s.start or 0
        stop = s.stop
        return (start >= 0) != (stop > 0)
    return False


def _span_mask(s: slice, asc: pl.Expr, desc: pl.Expr) -> pl.Expr:
    """Return a range expression for a bounded slice, routing each bound by sign.

    Positive start → ``asc >= start + 1``; negative start → ``desc <= -start``;
    None/zero start → no lower-bound predicate. Positive stop → ``asc <= stop``;
    negative stop → ``desc > -stop``.

    Args:
        s: A bounded slice. Must satisfy ``_is_bounded(s)``.
        asc: Ascending rank expression (1 = first group).
        desc: Descending rank expression (1 = last group).

    Returns:
        pl.Expr: Boolean expression selecting rows within the slice bounds.
    """
    parts: list[pl.Expr] = []
    if s.start is not None and s.start > 0:
        parts.append(asc >= s.start + 1)
    elif s.start is not None and s.start < 0:
        parts.append(desc <= -s.start)

    if s.stop is not None:
        if s.stop > 0:
            parts.append(asc <= s.stop)
        else:
            parts.append(desc > -s.stop)

    assert parts, "violates _is_bounded contract: unbounded slice passed to _span_mask"
    result: pl.Expr = parts[0]
    for p in parts[1:]:
        result = result & p
    return result


def _point_mask(i: int, asc: pl.Expr, desc: "pl.Expr | None") -> pl.Expr:
    """Return an equality expression selecting a single group by position.

    Args:
        i: Zero-based group index. Negative values count from the last group.
        asc: Ascending rank expression (1 = first group).
        desc: Descending rank expression (1 = last group). Required when ``i < 0``.

    Returns:
        pl.Expr: Boolean expression True for rows in the selected group.
    """
    if i >= 0:
        return asc == i + 1
    assert desc is not None
    return desc == -i


@dataclass
class _RankExprs:
    """Lazy rank expressions for a filtered column."""

    col_expr: pl.Expr
    condition: "pl.Expr | None"

    @cached_property
    def asc(self) -> pl.Expr:
        """Ascending group rank expression (1 = first group)."""
        if self.condition is not None:
            is_new = _group_start_expr(self.col_expr, self.condition)
            return is_new.cast(pl.Int32).cum_sum()
        return self.col_expr.rank("dense")

    @cached_property
    def desc(self) -> pl.Expr:
        """Descending group rank expression (1 = last group)."""
        if self.condition is not None:
            is_new = _group_start_expr(self.col_expr, self.condition)
            asc = is_new.cast(pl.Int32).cum_sum()
            return asc.max() - asc + 1
        return self.col_expr.rank("dense", descending=True)


def _iter_group_masks(index: "IndexType", exprs: _RankExprs) -> "Iterator[pl.Expr]":
    """Yield one equality mask per selected group position.

    Args:
        index: A single int, a sequence of ints, or a bounded slice.
        exprs: Rank expressions for the target column.

    Yields:
        pl.Expr: One ``_point_mask`` per selected group, in order.
    """
    if isinstance(index, int):
        yield _point_mask(index, exprs.asc, exprs.desc)
    elif isinstance(index, slice):
        start = index.start or 0
        stop = index.stop
        for i in range(start, stop):
            yield _point_mask(i, exprs.asc, exprs.desc)
    else:
        for i in index:
            yield _point_mask(i, exprs.asc, exprs.desc)


def _nonbounded_slice_mask(s: slice, asc: pl.Expr, desc: pl.Expr) -> pl.Expr:
    """Build a range expression for a non-bounded slice (negative or open-ended bounds).

    Args:
        s: A slice with negative or open-ended bounds. Non-zero, positive step only.
        asc: Ascending rank expression.
        desc: Descending rank expression.

    Returns:
        pl.Expr: Boolean expression selecting rows matched by the slice.

    Raises:
        ValueError: If ``s.step`` is zero or negative.
    """
    if s.step is not None and s.step <= 0:
        error_msg = (
            "slice step cannot be zero"
            if s.step == 0
            else "Negative step is not supported in a slice index."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    parts: list[pl.Expr] = []

    if s.start is not None:
        if s.start >= 0:
            parts.append(asc >= s.start + 1)
        else:
            parts.append(desc <= -s.start)

    if s.stop is not None:
        if s.stop > 0:
            parts.append(asc <= s.stop)
        elif s.stop < 0:
            parts.append(desc > -s.stop)
        else:
            return pl.lit(False)

    step_val = s.step
    if step_val is not None and step_val > 1:
        effective_start = s.start if s.start is not None else 0
        if effective_start >= 0:
            anchor = effective_start + 1
            parts.append((asc - anchor) % step_val == 0)
        else:
            anchor = -effective_start
            parts.append((anchor - desc) % step_val == 0)

    if not parts:
        return pl.lit(True)

    result: pl.Expr = parts[0]
    for p in parts[1:]:
        result = result & p
    return result


def _combined_mask(
    index: "IndexType | None",
    exprs: _RankExprs,
    condition: "pl.Expr | None",
) -> pl.Expr:
    """Return a single combined boolean mask for the given index.

    Args:
        index: The index argument. ``None`` selects all matching rows.
        exprs: Rank expressions for the target column.
        condition: Optional row condition. Applied after the rank mask.

    Returns:
        pl.Expr: Combined boolean expression selecting the target rows.
    """
    if index is None:
        return condition if condition is not None else pl.lit(True)

    if isinstance(index, slice):
        if _is_bounded(index):
            mask = _span_mask(index, exprs.asc, exprs.desc)
        else:
            mask = _nonbounded_slice_mask(index, exprs.asc, exprs.desc)
    else:
        masks = list(_iter_group_masks(index, exprs))
        mask = masks[0]
        for m in masks[1:]:
            mask = mask | m

    return (condition & mask) if condition is not None else mask


def _build_result(
    obj: "FilterToCycleType",
    mask: pl.Expr,
    column: "BDF | Column",
    include_preceding_row: bool,
) -> "Cycle | Step":
    """Create a filtered Cycle or Step result object.

    Args:
        obj: The source object to filter.
        mask: Boolean expression selecting rows to include.
        column: The filter column; ``BDF.CYCLE_COUNT`` produces a Cycle, else a Step.
        include_preceding_row: When ``True``, include the row immediately
            before each contiguous block of selected rows.

    Returns:
        Cycle | Step: A new result object containing the filtered data.
    """
    if include_preceding_row:
        mask = _include_preceding_row(mask)
    filtered_lf = obj.lf.filter(mask)
    path = getattr(obj, "_path", None)
    if column == BDF.CYCLE_COUNT:
        cycle_info = obj.cycle_info[1:] if len(obj.cycle_info) > 1 else []
        return Cycle(
            lf=filtered_lf,
            metadata=obj.info,
            column_definitions=obj.column_definitions,
            step_descriptions=obj.step_descriptions,
            cycle_info=cycle_info,
            _path=path,
        )
    return Step(
        lf=filtered_lf,
        metadata=obj.info,
        column_definitions=obj.column_definitions,
        step_descriptions=obj.step_descriptions,
        _path=path,
    )


class _Filter:
    """Encapsulates group-filtering logic for a single column and optional condition."""

    def __init__(
        self,
        column: "BDF | Column" = BDF.STEP_COUNT,
        condition: "pl.Expr | None" = None,
    ) -> None:
        """Initialize a _Filter instance.

        Args:
            column: The column to perform filtering on. Defaults to BDF.STEP_COUNT.
            condition: Optional polars expression that defines which rows
                qualify as part of a group. When ``None``, all rows in each
                group are selected.
        """
        self.column = column
        self.condition = condition

    def singular(
        self,
        obj: "FilterToCycleType",
        index: "IndexType | None" = None,
        include_preceding_row: bool = False,
    ) -> "Cycle | Step":
        """Return a single filtered result for the given index.

        Args:
            obj: The source object to filter.
            index: Positional selector. ``None`` returns all matching rows.
            include_preceding_row: When ``True``, include the row immediately
                before each contiguous block of selected rows.

        Returns:
            Cycle | Step: A single filtered result object.
        """
        obj.lf = get_cycle_column(obj) if self.column == BDF.CYCLE_COUNT else obj.lf
        col_expr = cast("FilterToCycleType", obj).columns.resolve(self.column)
        exprs = _RankExprs(col_expr, self.condition)
        mask = _combined_mask(index, exprs, self.condition)
        return _build_result(obj, mask, self.column, include_preceding_row)

    def plural(
        self,
        obj: "FilterToCycleType",
        index: "IndexType | None" = None,
        include_preceding_row: bool = False,
    ) -> "Iterator[Cycle | Step]":
        """Iterate over filtered results for each selected group.

        Uses a zero-collect static path when the index is statically resolvable
        (positive/negative ints, sequences, bounded slices); falls back to a
        single collect for open-ended or negative-start slices and ``None``.

        Args:
            obj: The source object to filter.
            index: Positional selector. ``None`` yields one result per group.
            include_preceding_row: When ``True``, include the row immediately
                before each contiguous block of selected rows.

        Yields:
            Cycle | Step: Filtered result objects, one per selected group.
        """
        obj.lf = get_cycle_column(obj) if self.column == BDF.CYCLE_COUNT else obj.lf
        col_expr = cast("FilterToCycleType", obj).columns.resolve(self.column)
        exprs = _RankExprs(col_expr, self.condition)

        if not _needs_collect(index):
            assert index is not None
            for rank_mask in _iter_group_masks(index, exprs):
                cond_mask = (
                    (self.condition & rank_mask)
                    if self.condition is not None
                    else rank_mask
                )
                yield _build_result(obj, cond_mask, self.column, include_preceding_row)
        else:
            full_mask = _combined_mask(index, exprs, self.condition)
            lf_lazy = obj.lf.lazy() if isinstance(obj.lf, pl.DataFrame) else obj.lf
            rank_values = (
                lf_lazy.with_columns(exprs.asc.alias("__rank__"))
                .filter(full_mask)
                .select("__rank__")
                .unique()
                .sort("__rank__")
                .collect()["__rank__"]
                .to_list()
            )
            for rank_val in rank_values:
                rank_mask = exprs.asc == rank_val
                cond_mask = (
                    (self.condition & rank_mask)
                    if self.condition is not None
                    else rank_mask
                )
                yield _build_result(obj, cond_mask, self.column, include_preceding_row)


def get_cycle_column(
    filtered_object: "FilterToCycleType",
) -> "pl.DataFrame | pl.LazyFrame":
    """Adds a cycle column to the data.

    If cycle details have been provided in the README, the cycle column will be
    created by checking for the last step of the cycle. For nested cycles, the
    "outer" cycle will be created first; subsequent filtering with the cycle method
    allows for filtering on the "inner" cycles.

    If no cycle details have been provided, the cycle column will be inferred from
    a decrease in the step count.

    Args:
        filtered_object: The experiment or cycle object.

    Returns:
        pl.DataFrame | pl.LazyFrame: The data with a cycle count column.
    """
    step_expr = filtered_object.columns.resolve(BDF.STEP_ID)
    cycle_col_name = BDF.CYCLE_COUNT.name
    if len(filtered_object.cycle_info) > 0:
        cycle_ends = (
            (
                (step_expr.shift() == filtered_object.cycle_info[0][1])
                & (step_expr != filtered_object.cycle_info[0][1])
            )
            .fill_null(strategy="zero")
            .cast(pl.Int16)
        )
        cycle_column = (
            cycle_ends.cum_sum().fill_null(strategy="zero").alias(cycle_col_name)
        )
    else:
        warnings.warn(
            "No cycle information provided. Cycles will be inferred from the step "
            "numbers.",
        )
        cycle_column = (
            (step_expr.cast(pl.Int64) - step_expr.cast(pl.Int64).shift() < 0)
            .fill_null(strategy="zero")
            .cum_sum()
            .alias(cycle_col_name)
        )
    return filtered_object.lf.with_columns(cycle_column)


def _make_constant_condition(
    col_expr: pl.Expr,
    target: float | None = None,
    rtol: float = 0.001,
    mask: pl.Expr | None = None,
) -> pl.Expr:
    """Return a polars expression selecting rows in a constant-value section.

    Args:
        col_expr: Resolved column expression to evaluate.
        target: When supplied, selects rows where ``col_expr`` lies within
            ``target ± |target| * rtol``. Sign is preserved: a positive target
            only matches positive values; a negative target only matches negative
            values. When ``None``, the signed mode of ``col_expr`` (filtered by
            ``mask`` if given) is used as the target. This is sign-sensitive:
            the implicit target reflects the most frequent signed value, which
            may match only one polarity (e.g. charge-only or discharge-only).
        rtol: Relative tolerance (dimensionless). The acceptance band around
            the target is ``|target| * rtol``. Defaults to ``0.001`` (0.1%).
        mask: Optional polars expression used to filter ``col_expr`` before
            computing the global mode (only used when ``target`` is ``None``).

    Returns:
        pl.Expr: Boolean expression that is True for rows in constant sections.
    """
    if target is not None:
        return (col_expr - target).abs() <= abs(target) * rtol
    mode_expr = col_expr.filter(mask) if mask is not None else col_expr
    t = mode_expr.mode().first()
    return (col_expr - t).abs() <= t.abs() * rtol


class StepFiltersMixin:
    """Mixin providing step-specific filter methods."""

    def _constant_current_filter(
        self, target: "float | None", rtol: float
    ) -> "_Filter":
        """Create a filter for constant-current steps."""
        current_expr = cast("FilterToCycleType", self).columns.resolve(
            BDF.CURRENT_AMPERE
        )
        col_mask = current_expr != 0 if target is None else None
        return _Filter(
            BDF.STEP_COUNT,
            _make_constant_condition(current_expr, target, rtol, mask=col_mask),
        )

    def _constant_voltage_filter(
        self, target: "float | None", rtol: float
    ) -> "_Filter":
        """Create a filter for constant-voltage steps."""
        voltage_expr = cast("FilterToCycleType", self).columns.resolve(BDF.VOLTAGE_VOLT)
        return _Filter(
            BDF.STEP_COUNT,
            _make_constant_condition(voltage_expr, target, rtol),
        )

    def constant_current(
        self,
        index: "IndexType | None" = None,
        *,
        target: float | None = None,
        rtol: float = 0.001,
        include_preceding_row: bool = False,
    ) -> "Step":
        """Filter constant-current events selected by index.

        Args:
            index: Positional selector. Supports zero-based integers, sequences
                of integers, and slices, including negative indexing relative to
                the end. ``None`` returns all matching rows as a single result.
            target: When supplied, select only rows where ``Current / A`` lies
                within ``target ± |target| * rtol``. Sign is preserved:
                ``target=1.0`` matches only positive (charge) values;
                ``target=-1.0`` matches only negative (discharge) values. When
                ``None``, the signed mode of non-zero ``Current / A`` values is
                used as the target.
            rtol: Relative tolerance (dimensionless) controlling the acceptance
                band as a fraction of ``|target|``. Defaults to ``0.001`` (0.1%).
            include_preceding_row: When ``True``, include the data row
                immediately before each contiguous block of selected rows.

        Returns:
            Step: Filtered result for the selected groups.
        """
        return cast(
            "Step",
            self._constant_current_filter(target, rtol).singular(
                cast("FilterToCycleType", self),
                index,
                include_preceding_row=include_preceding_row,
            ),
        )

    def iter_constant_current(
        self,
        index: "IndexType | None" = None,
        *,
        target: float | None = None,
        rtol: float = 0.001,
        include_preceding_row: bool = False,
    ) -> "Iterator[Step]":
        """Iterate over constant-current events selected by index.

        Args:
            index: Positional selector. Supports zero-based integers, sequences
                of integers, and slices, including negative indexing relative to
                the end. ``None`` yields one result per group.
            target: When supplied, select only rows where current lies within
                ``target ± |target| * rtol``.
            rtol: Relative tolerance (dimensionless). Defaults to ``0.001``
                (0.1%).
            include_preceding_row: When ``True``, include the data row
                immediately before each contiguous block of selected rows.

        Returns:
            Iterator[Step]: Filtered result for the selected groups.
        """
        return cast(
            "Iterator[Step]",
            self._constant_current_filter(target, rtol).plural(
                cast("FilterToCycleType", self),
                index,
                include_preceding_row=include_preceding_row,
            ),
        )

    def constant_voltage(
        self,
        index: "IndexType | None" = None,
        *,
        target: float | None = None,
        rtol: float = 0.001,
        include_preceding_row: bool = False,
    ) -> "Step":
        """Filter constant-voltage events selected by index.

        Args:
            index: Positional selector. Supports zero-based integers, sequences
                of integers, and slices, including negative indexing relative to
                the end. ``None`` returns all matching rows as a single result.
            target: When supplied, select only rows where voltage lies within
                ``target ± |target| * rtol``. When ``None``, the signed mode of
                voltage is used.
            rtol: Relative tolerance (dimensionless). Defaults to ``0.001``
                (0.1%).
            include_preceding_row: When ``True``, include the data row
                immediately before each contiguous block of selected rows.

        Returns:
            Step: Filtered result for the selected groups.
        """
        return cast(
            "Step",
            self._constant_voltage_filter(target, rtol).singular(
                cast("FilterToCycleType", self),
                index,
                include_preceding_row=include_preceding_row,
            ),
        )

    def iter_constant_voltage(
        self,
        index: "IndexType | None" = None,
        *,
        target: float | None = None,
        rtol: float = 0.001,
        include_preceding_row: bool = False,
    ) -> "Iterator[Step]":
        """Iterate over constant-voltage events selected by index.

        Args:
            index: Positional selector. Supports zero-based integers, sequences
                of integers, and slices, including negative indexing relative to
                the end. ``None`` yields one result per group.
            target: When supplied, select only rows where voltage lies within
                ``target ± |target| * rtol``. When ``None``, the signed mode of
                voltage is used.
            rtol: Relative tolerance (dimensionless). Defaults to ``0.001``
                (0.1%).
            include_preceding_row: When ``True``, include the data row
                immediately before each contiguous block of selected rows.

        Returns:
            Iterator[Step]: Filtered result for the selected groups.
        """
        return cast(
            "Iterator[Step]",
            self._constant_voltage_filter(target, rtol).plural(
                cast("FilterToCycleType", self),
                index,
                include_preceding_row=include_preceding_row,
            ),
        )


class CycleFiltersMixin:
    """Mixin providing cycle and charge/discharge filter methods."""

    def _charge_filter(self) -> "_Filter":
        """Create a filter for charge events."""
        current_expr = cast("FilterToCycleType", self).columns.resolve(
            BDF.CURRENT_AMPERE
        )
        return _Filter(BDF.STEP_COUNT, current_expr > current_expr.abs().max() / 10e4)

    def _discharge_filter(self) -> "_Filter":
        """Create a filter for discharge events."""
        current_expr = cast("FilterToCycleType", self).columns.resolve(
            BDF.CURRENT_AMPERE
        )
        return _Filter(BDF.STEP_COUNT, current_expr < -current_expr.abs().max() / 10e4)

    def _chargeordischarge_filter(self) -> "_Filter":
        """Create a filter for non-rest events."""
        current_expr = cast("FilterToCycleType", self).columns.resolve(
            BDF.CURRENT_AMPERE
        )
        return _Filter(
            BDF.STEP_COUNT, current_expr.abs() > current_expr.abs().max() / 10e4
        )

    def _rest_filter(self) -> "_Filter":
        """Create a filter for rest events."""
        current_expr = cast("FilterToCycleType", self).columns.resolve(
            BDF.CURRENT_AMPERE
        )
        return _Filter(BDF.STEP_COUNT, current_expr == 0)

    def _step_filter(self) -> "_Filter":
        """Create a filter for step events."""
        return _Filter(BDF.STEP_COUNT)

    def _cycle_filter(self) -> "_Filter":
        """Create a filter for cycle events."""
        return _Filter(BDF.CYCLE_COUNT)

    def step(
        self,
        index: "IndexType | None" = None,
        *,
        include_preceding_row: bool = False,
    ) -> "Step":
        """Filter step events selected by index.

        Args:
            index: Positional selector. Supports zero-based integers, sequences
                of integers, and slices, including negative indexing relative to
                the end. ``None`` returns all matching rows as a single result.
            include_preceding_row: When ``True``, include the data row
                immediately before each contiguous block of selected rows.

        Returns:
            Step: Filtered result for the selected groups.
        """
        return cast(
            "Step",
            self._step_filter().singular(
                cast("FilterToCycleType", self),
                index,
                include_preceding_row=include_preceding_row,
            ),
        )

    def iter_step(
        self,
        index: "IndexType | None" = None,
        *,
        include_preceding_row: bool = False,
    ) -> "Iterator[Step]":
        """Iterate over step events selected by index.

        Args:
            index: Positional selector. Supports zero-based integers, sequences
                of integers, and slices, including negative indexing relative to
                the end. ``None`` yields one result per group.
            include_preceding_row: When ``True``, include the data row
                immediately before each contiguous block of selected rows.

        Returns:
            Iterator[Step]: Filtered result for the selected groups.
        """
        return cast(
            "Iterator[Step]",
            self._step_filter().plural(
                cast("FilterToCycleType", self),
                index,
                include_preceding_row=include_preceding_row,
            ),
        )

    def cycle(
        self,
        index: "IndexType | None" = None,
        *,
        include_preceding_row: bool = False,
    ) -> "Cycle":
        """Filter cycles selected by index.

        Args:
            index: Positional selector. Supports zero-based integers, sequences
                of integers, and slices, including negative indexing relative to
                the end. ``None`` returns all matching rows as a single result.
            include_preceding_row: When ``True``, include the data row
                immediately before each contiguous block of selected rows.

        Returns:
            Cycle: Filtered result for the selected groups.
        """
        return cast(
            "Cycle",
            self._cycle_filter().singular(
                cast("FilterToCycleType", self),
                index,
                include_preceding_row=include_preceding_row,
            ),
        )

    def iter_cycle(
        self,
        index: "IndexType | None" = None,
        *,
        include_preceding_row: bool = False,
    ) -> "Iterator[Cycle]":
        """Iterate over cycles selected by index.

        Args:
            index: Positional selector. Supports zero-based integers, sequences
                of integers, and slices, including negative indexing relative to
                the end. ``None`` yields one result per group.
            include_preceding_row: When ``True``, include the data row
                immediately before each contiguous block of selected rows.

        Returns:
            Iterator[Cycle]: Filtered result for the selected groups.
        """
        return cast(
            "Iterator[Cycle]",
            self._cycle_filter().plural(
                cast("FilterToCycleType", self),
                index,
                include_preceding_row=include_preceding_row,
            ),
        )

    def charge(
        self,
        index: "IndexType | None" = None,
        *,
        include_preceding_row: bool = False,
    ) -> "Step":
        """Filter charge events selected by index.

        Args:
            index: Positional selector. Supports zero-based integers, sequences
                of integers, and slices, including negative indexing relative to
                the end. ``None`` returns all matching rows as a single result.
            include_preceding_row: When ``True``, include the data row
                immediately before each contiguous block of selected rows.

        Returns:
            Step: Filtered result for the selected groups.
        """
        return cast(
            "Step",
            self._charge_filter().singular(
                cast("FilterToCycleType", self),
                index,
                include_preceding_row=include_preceding_row,
            ),
        )

    def iter_charge(
        self,
        index: "IndexType | None" = None,
        *,
        include_preceding_row: bool = False,
    ) -> "Iterator[Step]":
        """Iterate over charge events selected by index.

        Args:
            index: Positional selector. Supports zero-based integers, sequences
                of integers, and slices, including negative indexing relative to
                the end. ``None`` yields one result per group.
            include_preceding_row: When ``True``, include the data row
                immediately before each contiguous block of selected rows.

        Returns:
            Iterator[Step]: Filtered result for the selected groups.
        """
        return cast(
            "Iterator[Step]",
            self._charge_filter().plural(
                cast("FilterToCycleType", self),
                index,
                include_preceding_row=include_preceding_row,
            ),
        )

    def discharge(
        self,
        index: "IndexType | None" = None,
        *,
        include_preceding_row: bool = False,
    ) -> "Step":
        """Filter discharge events selected by index.

        Args:
            index: Positional selector. Supports zero-based integers, sequences
                of integers, and slices, including negative indexing relative to
                the end. ``None`` returns all matching rows as a single result.
            include_preceding_row: When ``True``, include the data row
                immediately before each contiguous block of selected rows.

        Returns:
            Step: Filtered result for the selected groups.
        """
        return cast(
            "Step",
            self._discharge_filter().singular(
                cast("FilterToCycleType", self),
                index,
                include_preceding_row=include_preceding_row,
            ),
        )

    def iter_discharge(
        self,
        index: "IndexType | None" = None,
        *,
        include_preceding_row: bool = False,
    ) -> "Iterator[Step]":
        """Iterate over discharge events selected by index.

        Args:
            index: Positional selector. Supports zero-based integers, sequences
                of integers, and slices, including negative indexing relative to
                the end. ``None`` yields one result per group.
            include_preceding_row: When ``True``, include the data row
                immediately before each contiguous block of selected rows.

        Returns:
            Iterator[Step]: Filtered result for the selected groups.
        """
        return cast(
            "Iterator[Step]",
            self._discharge_filter().plural(
                cast("FilterToCycleType", self),
                index,
                include_preceding_row=include_preceding_row,
            ),
        )

    def chargeordischarge(
        self,
        index: "IndexType | None" = None,
        *,
        include_preceding_row: bool = False,
    ) -> "Step":
        """Filter non-rest events selected by index.

        Args:
            index: Positional selector. Supports zero-based integers, sequences
                of integers, and slices, including negative indexing relative to
                the end. ``None`` returns all matching rows as a single result.
            include_preceding_row: When ``True``, include the data row
                immediately before each contiguous block of selected rows.

        Returns:
            Step: Filtered result for the selected groups.
        """
        return cast(
            "Step",
            self._chargeordischarge_filter().singular(
                cast("FilterToCycleType", self),
                index,
                include_preceding_row=include_preceding_row,
            ),
        )

    def iter_chargeordischarge(
        self,
        index: "IndexType | None" = None,
        *,
        include_preceding_row: bool = False,
    ) -> "Iterator[Step]":
        """Iterate over non-rest events selected by index.

        Args:
            index: Positional selector. Supports zero-based integers, sequences
                of integers, and slices, including negative indexing relative to
                the end. ``None`` yields one result per group.
            include_preceding_row: When ``True``, include the data row
                immediately before each contiguous block of selected rows.

        Returns:
            Iterator[Step]: Filtered result for the selected groups.
        """
        return cast(
            "Iterator[Step]",
            self._chargeordischarge_filter().plural(
                cast("FilterToCycleType", self),
                index,
                include_preceding_row=include_preceding_row,
            ),
        )

    def rest(
        self,
        index: "IndexType | None" = None,
        *,
        include_preceding_row: bool = False,
    ) -> "Step":
        """Filter rest events selected by index.

        Args:
            index: Positional selector. Supports zero-based integers, sequences
                of integers, and slices, including negative indexing relative to
                the end. ``None`` returns all matching rows as a single result.
            include_preceding_row: When ``True``, include the data row
                immediately before each contiguous block of selected rows.

        Returns:
            Step: Filtered result for the selected groups.
        """
        return cast(
            "Step",
            self._rest_filter().singular(
                cast("FilterToCycleType", self),
                index,
                include_preceding_row=include_preceding_row,
            ),
        )

    def iter_rest(
        self,
        index: "IndexType | None" = None,
        *,
        include_preceding_row: bool = False,
    ) -> "Iterator[Step]":
        """Iterate over rest events selected by index.

        Args:
            index: Positional selector. Supports zero-based integers, sequences
                of integers, and slices, including negative indexing relative to
                the end. ``None`` yields one result per group.
            include_preceding_row: When ``True``, include the data row
                immediately before each contiguous block of selected rows.

        Returns:
            Iterator[Step]: Filtered result for the selected groups.
        """
        return cast(
            "Iterator[Step]",
            self._rest_filter().plural(
                cast("FilterToCycleType", self),
                index,
                include_preceding_row=include_preceding_row,
            ),
        )


class ExperimentFiltersMixin:
    """Mixin providing experiment-level filter methods."""

    def experiment(
        self,
        *experiment_names: str,
        include_preceding_row: bool = False,
    ) -> "Experiment":
        """Return an experiment object from the object.

        The name resolves against the groups at the current level of the
        protocol tree alone, so an experiment can hold a further experiment.

        Args:
            experiment_names: Variable-length argument list of experiment names.
            include_preceding_row: When ``True``, prepend the data point
                immediately before the experiment's first row.

        Returns:
            Experiment: An experiment object holding the data and the protocol
                node of the named experiments.

        Raises:
            ValueError: If the current level holds no group with that
                description, or if every leaf below the resolved group carries
                no step identifier. The message names the experiment.
        """
        raise NotImplementedError

    @property
    def experiment_names(self) -> list[str]:
        """The descriptions of the groups at the current protocol tree level.

        Returns:
            list[str]: The names of the experiments below this object.
        """
        raise NotImplementedError


class Procedure(CycleFiltersMixin, StepFiltersMixin, CyclingData):
    """A class for a procedure in a battery experiment."""

    def __init__(
        self,
        lf: pl.LazyFrame | pl.DataFrame | str,
        metadata: dict[str, Any | None],
        readme_dict: dict[str, dict[str, list[str | int | tuple[int, int, int]]]],
        column_definitions: dict[str, str] | None = None,
        step_descriptions: dict[str, list[str | int | None]] | None = None,
        cycle_info: list[tuple[int, int, int]] | None = None,
    ) -> None:
        """Initialize a procedure with README-derived experiment metadata.

        Args:
            lf: A LazyFrame, DataFrame, or a path to a parquet file.
            metadata: Dictionary containing metadata about the procedure and
                data source.
            readme_dict: Experiment definitions from README.
            column_definitions: Column descriptions.
            step_descriptions: Step-by-step descriptions.
            cycle_info: Cycle boundary information.
        """
        super().__init__(
            lf=lf,
            metadata=metadata,
            column_definitions=column_definitions,
            step_descriptions=step_descriptions,
        )
        self.readme_dict = readme_dict
        self.cycle_info = cycle_info.copy() if cycle_info is not None else []
        self._populate_step_descriptions()

    def _populate_step_descriptions(self) -> None:
        """Populate step_descriptions from readme_dict."""
        self.step_descriptions = {"Step": [], "Description": []}
        for experiment in self.readme_dict:
            steps = cast(list[int], self.readme_dict[experiment]["Steps"])
            descriptions: list[str | None] = [None] * len(steps)
            if "Step Descriptions" in self.readme_dict[experiment]:
                descriptions = cast(
                    list[str | None],
                    self.readme_dict[experiment]["Step Descriptions"],
                )
            self.step_descriptions["Step"].extend(steps)
            self.step_descriptions["Description"].extend(descriptions)

    @classmethod
    def load(
        cls,
        source: str | Path | pl.LazyFrame | pl.DataFrame,
        readme_path: str | Path | None = None,
        metadata_prefer: Literal["parquet", "json"] = "parquet",
    ) -> "Procedure":
        """Load a Procedure from a file, LazyFrame, or DataFrame.

        Accepts a path to a ``.parquet`` or ``.csv`` file, or directly a Polars
        :class:`~polars.LazyFrame` or :class:`~polars.DataFrame`. The source
        data can come from **any origin** — battery cycler software, simulation
        tools, or manual construction — as long as it contains BDF-compatible
        columns:

        - A time column: ``"Test Time / s"`` or ``"Unix Time / s"``
        - ``"Current / A"``
        - ``"Voltage / V"``

        Use :func:`~pyprobe.io.process_cycler` to convert raw cycler files to
        BDF format, or :func:`~pyprobe.io.process_generic` with a column map to
        normalise arbitrary DataFrames before passing them here.

        For ``.parquet`` sources, metadata is read from the Parquet footer (or a
        JSON sidecar, depending on *metadata_prefer*). For ``.csv`` and
        frame sources, metadata is empty. README auto-discovery only applies when
        *source* is a file path.

        Args:
            source: A path to a ``.parquet`` or ``.csv`` file, a
                :class:`~polars.LazyFrame`, or a :class:`~polars.DataFrame`.
                Must contain BDF-compatible columns (see above).
            readme_path: Explicit path to a README.yaml for experiment definitions.
                When ``None`` (default), the parent directory of a file *source*
                is checked automatically.
            metadata_prefer: Whether to prefer the Parquet footer (``"parquet"``,
                default) or a JSON sidecar (``"json"``) when both sources exist.
                Only used when *source* is a ``.parquet`` file.

        Returns:
            Procedure with BDF-format columns, optional metadata, and optional
            experiment definitions from README.yaml.

        Raises:
            FileNotFoundError: If *source* is a path that does not exist.
            ValueError: If *source* is a path with an unsupported suffix.
            ValueError: If required BDF columns are missing from the data.

        Example:
            Load from a processed parquet file::

                from pyprobe.filters import Procedure

                procedure = Procedure.load("data.bdf.parquet")
                procedure = Procedure.load(
                    "data.bdf.parquet", readme_path="README.yaml"
                )

            Load from a LazyFrame::

                procedure = Procedure.load(my_lf)
        """
        from pyprobe.io import read_metadata
        from pyprobe.readme_processor import process_readme

        resolved_path: Path | None = None
        lf: pl.LazyFrame
        parquet_metadata: dict[str, Any] = {}

        if isinstance(source, pl.DataFrame):
            lf = source.lazy()
        elif isinstance(source, pl.LazyFrame):
            lf = source
        else:
            file_path = Path(source)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            suffix = file_path.suffix.lower()
            if suffix == ".parquet":
                resolved_path = file_path
                lf = pl.scan_parquet(file_path)
                parquet_metadata = read_metadata(file_path, prefer=metadata_prefer)
            elif suffix == ".csv":
                lf = pl.scan_csv(file_path)
            else:
                raise ValueError(
                    f"Unsupported file format '{suffix}'. "
                    "Use '.parquet' or '.csv', or pass a LazyFrame/DataFrame directly."
                )
            if readme_path is None:
                candidate = file_path.parent / "README.yaml"
                if candidate.exists():
                    readme_path = candidate
                else:
                    logger.info(
                        "No README.yaml found in '{}'; proceeding without "
                        "experiment definitions.",
                        file_path.parent,
                    )

        readme_dict: dict[str, dict[str, Any]] = {}
        if readme_path is not None:
            rp = Path(readme_path)
            if rp.exists():
                readme_dict = process_readme(str(rp)).experiment_dict
            else:
                logger.warning("README path provided but not found: {}", readme_path)

        procedure = cls(lf=lf, metadata=parquet_metadata, readme_dict=readme_dict)
        procedure.lf = procedure.lf.with_columns(
            procedure.columns.resolve(BDF.TEST_TIME_SECOND)
        )
        procedure._path = resolved_path
        return procedure

    def attach_legacy_readme(self, readme_path: str | Path) -> None:
        """Convert a legacy README.yaml file and attach it as the protocol.

        The converted tree replaces the current
        ``metadata.battinfo_test_protocol.method``.

        Args:
            readme_path: The path to the README.yaml file.

        Raises:
            FileNotFoundError: If the README file does not exist.
            ValueError: If a cycle does not bound a contiguous group of steps.
                The message names the experiment and the cycle key.
        """
        raise NotImplementedError

    def sync_metadata(self, *, protect_existing: bool = True) -> None:
        """Write ``self.metadata`` back to the backing Parquet file.

        Args:
            protect_existing: When ``True`` (default), raises ``ValueError`` if
                any key present in the file metadata is absent from
                ``self.metadata`` or has a changed value. Set to ``False`` to
                allow removing or changing existing keys.

        Raises:
            RuntimeError: If ``self._path`` is ``None`` (no backing file).
            ValueError: If *protect_existing* is ``True`` and a destructive
                change would be made.
        """
        if self._path is None:
            raise RuntimeError(
                "sync_metadata requires a backing Parquet file but _path is None. "
                "Load the Procedure from a file path to enable sync_metadata."
            )

        from pyprobe.io import MetadataManager

        manager = MetadataManager(self._path)

        if protect_existing:
            file_meta = manager.read_parquet()
            conflicts: list[str] = []
            for key, value in file_meta.items():
                if key not in self.metadata:
                    conflicts.append(
                        f"key '{key}' is present in file but absent from metadata"
                    )
                elif self.metadata[key] != value:
                    conflicts.append(
                        f"key '{key}' has a changed value "
                        f"(file={value!r}, memory={self.metadata[key]!r})"
                    )
            if conflicts:
                raise ValueError(
                    "sync_metadata(protect_existing=True) would overwrite existing "
                    "file metadata. Conflicts:\n"
                    + "\n".join(f"  - {c}" for c in conflicts)
                )

        manager.write(self.metadata)
        logger.info("Synced metadata to '{}'.", self._path)

    def experiment(
        self,
        *experiment_names: str,
        include_preceding_row: bool = False,
    ) -> "Experiment":
        """Return an experiment object from the procedure.

        Args:
            experiment_names: Variable-length argument list of experiment names.
            include_preceding_row: When ``True``, prepend the data point
                immediately before the experiment's first row.

        Returns:
            Experiment: An experiment object from the procedure.
        """
        steps_idx = []
        for experiment_name in experiment_names:
            if experiment_name not in self.experiment_names:
                error_msg = f"{experiment_name} not in procedure."
                logger.error(error_msg)
                raise ValueError(error_msg)
            steps_idx.append(self.readme_dict[experiment_name]["Steps"])
        flattened_steps = utils.flatten_list(steps_idx)
        mask = self.columns.resolve(BDF.STEP_ID).is_in(flattened_steps)
        if include_preceding_row:
            mask = _include_preceding_row(mask)
        lf_filtered = self.lf.filter(mask)
        cycles_list: list[tuple[int, int, int]] = []
        if len(experiment_names) > 1:
            warnings.warn(
                "Multiple experiments selected. Cycles will be inferred from "
                "the step numbers.",
            )
        elif "Cycles" in self.readme_dict[experiment_names[0]]:
            cycles_list = self.readme_dict[experiment_names[0]]["Cycles"]  # type: ignore[assignment]

        return Experiment(
            lf=lf_filtered,
            metadata=self.metadata,
            column_definitions=self.column_definitions,
            step_descriptions=self.step_descriptions,
            cycle_info=cycles_list,
            _path=self._path,
        )

    def remove_experiment(self, *experiment_names: str) -> None:
        """Remove an experiment from the procedure.

        Args:
            experiment_names: Variable-length argument list of experiment names.
        """
        steps_idx = []
        for experiment_name in experiment_names:
            if experiment_name not in self.experiment_names:
                error_msg = f"{experiment_name} not in procedure."
                logger.error(error_msg)
                raise ValueError(error_msg)
            steps_idx.append(self.readme_dict[experiment_name]["Steps"])
        flattened_steps = utils.flatten_list(steps_idx)
        conditions = [
            self.columns.resolve(BDF.STEP_ID).is_in(flattened_steps).not_(),
        ]
        for experiment_name in experiment_names:
            self.readme_dict.pop(experiment_name)
        self._populate_step_descriptions()
        self.lf = self.lf.filter(conditions)

    @property
    def experiment_names(self) -> list[str]:
        """Return the names of the experiments in the procedure.

        Returns:
            List[str]: The names of the experiments in the procedure.
        """
        return list(self.readme_dict.keys())

    @utils.deprecated(
        reason="Use add_data instead.",
        version="2.3.1",
    )
    def add_external_data(
        self,
        filepath: str,
        importing_columns: list[str] | dict[str, str],
        date_column_name: str = "Date",
    ) -> None:
        """Add data from another source to the procedure.

        Args:
            filepath: The path to the external file.
            importing_columns: The columns to import from the external file.
            date_column_name: The name of the date column in the external data.
        """
        raise NotImplementedError(
            "add_external_data is deprecated. Use add_data instead."
        )


class Experiment(CycleFiltersMixin, StepFiltersMixin, CyclingData):
    """A class for an experiment in a battery experimental procedure."""

    cycle_info: list[tuple[int, int, int]] = []
    """A list of tuples representing the cycle information from the README yaml file.

    The tuple format is
    :code:`(start step (inclusive), end step (inclusive), cycle count)`.
    """

    def __init__(
        self,
        lf: pl.LazyFrame | pl.DataFrame | str,
        metadata: dict[str, Any | None],
        column_definitions: dict[str, str] | None = None,
        step_descriptions: dict[str, list[str | int | None]] | None = None,
        cycle_info: list[tuple[int, int, int]] | None = None,
        _path: Path | None = None,
    ) -> None:
        """Initialize an experiment view with optional cycle metadata.

        Args:
            lf: A LazyFrame, DataFrame, or a path to a parquet file.
            metadata: Dictionary containing metadata about the experiment and
                data source.
            column_definitions: Column descriptions.
            step_descriptions: Step-by-step descriptions.
            cycle_info: Cycle boundary information.
            _path: Optional path to the backing Parquet file.
        """
        super().__init__(
            lf=lf,
            metadata=metadata,
            column_definitions=column_definitions,
            step_descriptions=step_descriptions,
            _path=_path,
        )
        self.cycle_info = cycle_info.copy() if cycle_info is not None else []


class Cycle(CycleFiltersMixin, StepFiltersMixin, CyclingData):
    """A class for a cycle in a battery experimental procedure."""

    cycle_info: list[tuple[int, int, int]] = []
    """A list of tuples representing the cycle information from the README yaml file.

    The tuple format is
    :code:`(start step (inclusive), end step (inclusive), cycle count)`.
    """

    def __init__(
        self,
        lf: pl.LazyFrame | pl.DataFrame | str,
        metadata: dict[str, Any | None],
        column_definitions: dict[str, str] | None = None,
        step_descriptions: dict[str, list[str | int | None]] | None = None,
        cycle_info: list[tuple[int, int, int]] | None = None,
        _path: Path | None = None,
    ) -> None:
        """Initialize a cycle view with optional nested cycle metadata.

        Args:
            lf: A LazyFrame, DataFrame, or a path to a parquet file.
            metadata: Dictionary containing metadata about the cycle and data source.
            column_definitions: Column descriptions.
            step_descriptions: Step-by-step descriptions.
            cycle_info: Cycle boundary information.
            _path: Optional path to the backing Parquet file.
        """
        super().__init__(
            lf=lf,
            metadata=metadata,
            column_definitions=column_definitions,
            step_descriptions=step_descriptions,
            _path=_path,
        )
        self.cycle_info = cycle_info.copy() if cycle_info is not None else []


class Step(StepFiltersMixin, CyclingData):
    """A class for a step in a battery experimental procedure."""

    def __init__(
        self,
        lf: pl.LazyFrame | pl.DataFrame | str,
        metadata: dict[str, Any | None],
        column_definitions: dict[str, str] | None = None,
        step_descriptions: dict[str, list[str | int | None]] | None = None,
        _path: Path | None = None,
    ) -> None:
        """Initialize a step view.

        Args:
            lf: A LazyFrame, DataFrame, or a path to a parquet file.
            metadata: Dictionary containing metadata about the step and data source.
            column_definitions: Column descriptions.
            step_descriptions: Step-by-step descriptions.
            _path: Optional path to the backing Parquet file.
        """
        super().__init__(
            lf=lf,
            metadata=metadata,
            column_definitions=column_definitions,
            step_descriptions=step_descriptions,
            _path=_path,
        )
