"""A module for the filtering classes."""

import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import polars as pl

from pyprobe import utils
from pyprobe.rawdata import RawData

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


def _group_start_expr(column: str, condition: pl.Expr) -> pl.Expr:
    """Return a boolean expression that is True on the first row of each matching group.

    Args:
        column: The column to compute rank groups on.
        condition: Expression selecting rows that belong to matching groups.

    Returns:
        pl.Expr: Boolean expression True at the start of each matching group.
    """
    event_rank = pl.col(column).rank("dense")
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

    column: str
    condition: "pl.Expr | None"

    @cached_property
    def asc(self) -> pl.Expr:
        """Ascending group rank expression (1 = first group)."""
        if self.condition is not None:
            is_new = _group_start_expr(self.column, self.condition)
            return is_new.cast(pl.Int32).cum_sum()
        return pl.col(self.column).rank("dense")

    @cached_property
    def desc(self) -> pl.Expr:
        """Descending group rank expression (1 = last group)."""
        if self.condition is not None:
            is_new = _group_start_expr(self.column, self.condition)
            asc = is_new.cast(pl.Int32).cum_sum()
            return asc.max() - asc + 1
        return pl.col(self.column).rank("dense", descending=True)


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
    lf: "pl.LazyFrame | pl.DataFrame",
    mask: pl.Expr,
    column: str,
    include_preceding_point: bool,
) -> "Cycle | Step":
    """Create a filtered Cycle or Step result object.

    Args:
        obj: The source object to filter.
        lf: The polars LazyFrame or DataFrame to filter.
        mask: Boolean expression selecting rows to include.
        column: The filter column name; ``"Cycle"`` produces a Cycle, else a Step.
        include_preceding_point: When ``True``, include the row immediately
            before each contiguous block of selected rows.

    Returns:
        Cycle | Step: A new result object containing the filtered data.
    """
    if include_preceding_point:
        mask = _include_preceding_row(mask)
    filtered_lf = lf.filter(mask)
    if column == "Cycle":
        cycle_info = obj.cycle_info[1:] if len(obj.cycle_info) > 1 else []
        return Cycle(
            lf=filtered_lf,
            info=obj.info,
            column_definitions=obj.column_definitions,
            step_descriptions=obj.step_descriptions,
            cycle_info=cycle_info,
        )
    return Step(
        lf=filtered_lf,
        info=obj.info,
        column_definitions=obj.column_definitions,
        step_descriptions=obj.step_descriptions,
    )


class _Filter:
    """Encapsulates group-filtering logic for a single column and optional condition."""

    def __init__(
        self, column: str = "Event", condition: "pl.Expr | None" = None
    ) -> None:
        """Initialize a _Filter instance.

        Args:
            column: The column name to perform filtering on. Defaults to "Event".
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
        include_preceding_point: bool = False,
    ) -> "Cycle | Step":
        """Return a single filtered result for the given index.

        Args:
            obj: The source object to filter.
            index: Positional selector. ``None`` returns all matching rows.
            include_preceding_point: When ``True``, include the row immediately
                before each contiguous block of selected rows.

        Returns:
            Cycle | Step: A single filtered result object.
        """
        lf = get_cycle_column(obj) if self.column == "Cycle" else obj.lf
        exprs = _RankExprs(self.column, self.condition)
        mask = _combined_mask(index, exprs, self.condition)
        return _build_result(obj, lf, mask, self.column, include_preceding_point)

    def plural(
        self,
        obj: "FilterToCycleType",
        index: "IndexType | None" = None,
        include_preceding_point: bool = False,
    ) -> "Iterator[Cycle | Step]":
        """Iterate over filtered results for each selected group.

        Uses a zero-collect static path when the index is statically resolvable
        (positive/negative ints, sequences, bounded slices); falls back to a
        single collect for open-ended or negative-start slices and ``None``.

        Args:
            obj: The source object to filter.
            index: Positional selector. ``None`` yields one result per group.
            include_preceding_point: When ``True``, include the row immediately
                before each contiguous block of selected rows.

        Yields:
            Cycle | Step: Filtered result objects, one per selected group.
        """
        lf = get_cycle_column(obj) if self.column == "Cycle" else obj.lf
        exprs = _RankExprs(self.column, self.condition)

        if not _needs_collect(index):
            assert index is not None
            for rank_mask in _iter_group_masks(index, exprs):
                cond_mask = (
                    (self.condition & rank_mask)
                    if self.condition is not None
                    else rank_mask
                )
                yield _build_result(
                    obj, lf, cond_mask, self.column, include_preceding_point
                )
        else:
            full_mask = _combined_mask(index, exprs, self.condition)
            lf_lazy = lf.lazy() if isinstance(lf, pl.DataFrame) else lf
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
                yield _build_result(
                    obj, lf, cond_mask, self.column, include_preceding_point
                )


def get_cycle_column(
    filtered_object: "FilterToCycleType",
) -> "pl.DataFrame | pl.LazyFrame":
    """Adds a cycle column to the data.

    If cycle details have been provided in the README, the cycle column will be created
    by checking for the last step of the cycle. For nested cycles, the "outer" cycle
    will be created first. Subsequent filtering with the cycle method will then allow
    for filtering on the "inner" cycles.

    If no cycle details have been provided, the cycle column will be created by
    identifying the last step of the cycle by checking for a decrease in the step
    number.

    Args:
        filtered_object: The experiment or cycle object.

    Returns:
        pl.DataFrame | pl.LazyFrame: The data with a cycle column.
    """
    if len(filtered_object.cycle_info) > 0:
        cycle_ends = (pl.col("Step").shift() == filtered_object.cycle_info[0][1]) & (
            pl.col("Step") != filtered_object.cycle_info[0][1]
        ).fill_null(strategy="zero").cast(pl.Int16)
        cycle_column = cycle_ends.cum_sum().fill_null(strategy="zero").alias("Cycle")
    else:
        warnings.warn(
            "No cycle information provided. Cycles will be inferred from the step "
            "numbers.",
        )
        cycle_column = (
            (pl.col("Step").cast(pl.Int64) - pl.col("Step").cast(pl.Int64).shift() < 0)
            .fill_null(strategy="zero")
            .cum_sum()
            .alias("Cycle")
        )
    return filtered_object.lf.with_columns(cycle_column)


_step_f = _Filter("Event")
_cycle_f = _Filter("Cycle")
_charge_f = _Filter(
    "Event", pl.col("Current [A]") > pl.col("Current [A]").abs().max() / 10e4
)
_discharge_f = _Filter(
    "Event", pl.col("Current [A]") < -pl.col("Current [A]").abs().max() / 10e4
)
_chargeordischarge_f = _Filter(
    "Event", pl.col("Current [A]").abs() > pl.col("Current [A]").abs().max() / 10e4
)
_rest_f = _Filter("Event", pl.col("Current [A]") == 0)


def _make_constant_condition(
    col: str,
    target: float | None = None,
    rtol: float = 0.001,
    mask: pl.Expr | None = None,
) -> pl.Expr:
    """Return a polars expression selecting rows in a constant-value section.

    Args:
        col: Column name to evaluate (e.g. ``"Current [A]"`` or ``"Voltage [V]"``).
        target: When supplied, selects rows where ``col`` lies within
            ``target ± |target| * rtol``. Sign is preserved: a positive target
            only matches positive values; a negative target only matches negative
            values. When ``None``, the signed mode of ``col`` (filtered by
            ``mask`` if given) is used as the target. This is sign-sensitive:
            the implicit target reflects the most frequent signed value, which
            may match only one polarity (e.g. charge-only or discharge-only).
        rtol: Relative tolerance (dimensionless). The acceptance band around
            the target is ``|target| * rtol``. Defaults to ``0.001`` (0.1%).
        mask: Optional polars expression used to filter ``col`` before computing
            the global mode (only used when ``target`` is ``None``).

    Returns:
        pl.Expr: Boolean expression that is True for rows in constant sections.
    """
    if target is not None:
        return (pl.col(col) - target).abs() <= abs(target) * rtol
    mode_expr = pl.col(col).filter(mask) if mask is not None else pl.col(col)
    t = mode_expr.mode().first()
    return (pl.col(col) - t).abs() <= t.abs() * rtol


class StepFiltersMixin:
    """Mixin providing step-specific filter methods."""

    def step(
        self,
        index: "IndexType | None" = None,
        include_preceding_point: bool = False,
    ) -> "Step":
        """Filter step events selected by index.

        Args:
            index (int | Sequence[int] | slice | None): Positional selector.
                Supports zero-based integers, sequences of integers, and slices,
                including negative indexing relative to the end. ``None`` returns
                all matching rows as a single result.
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Step: Filtered result for the selected groups.
        """
        return cast(
            "Step",
            _step_f.singular(
                cast("FilterToCycleType", self),
                index,
                include_preceding_point=include_preceding_point,
            ),
        )

    def iter_step(
        self,
        index: "IndexType | None" = None,
        include_preceding_point: bool = False,
    ) -> "Iterator[Step]":
        """Iterate over step events selected by index.

        Args:
            index (int | Sequence[int] | slice | None): Positional selector.
                Supports zero-based integers, sequences of integers, and slices,
                including negative indexing relative to the end. ``None`` yields
                one result per group.
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Iterator[Step]: Filtered result for the selected groups.
        """
        return cast(
            "Iterator[Step]",
            _step_f.plural(
                cast("FilterToCycleType", self),
                index,
                include_preceding_point=include_preceding_point,
            ),
        )

    def constant_current(
        self,
        index: "IndexType | None" = None,
        *,
        target: float | None = None,
        rtol: float = 0.001,
        include_preceding_point: bool = False,
    ) -> "Step":
        """Filter constant-current events selected by index.

        Args:
            index (int | Sequence[int] | slice | None): Positional selector.
                Supports zero-based integers, sequences of integers, and slices,
                including negative indexing relative to the end. ``None`` returns
                all matching rows as a single result.
            target (float | None): When supplied, select only rows where
                ``Current [A]`` lies within ``target ± |target| * rtol``. Sign
                is preserved: ``target=1.0`` matches only positive (charge)
                values; ``target=-1.0`` matches only negative (discharge)
                values. When ``None``, the signed mode of non-zero
                ``Current [A]`` values is used as the target.
            rtol (float): Relative tolerance (dimensionless) controlling the
                acceptance band as a fraction of ``|target|``. Defaults to
                ``0.001`` (0.1%).
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Step: Filtered result for the selected groups.
        """
        mask = pl.col("Current [A]") != 0 if target is None else None
        condition = _make_constant_condition("Current [A]", target, rtol, mask=mask)
        f = _Filter("Event", condition)
        return cast(
            "Step",
            f.singular(
                cast("FilterToCycleType", self),
                index,
                include_preceding_point=include_preceding_point,
            ),
        )

    def iter_constant_current(
        self,
        index: "IndexType | None" = None,
        *,
        target: float | None = None,
        rtol: float = 0.001,
        include_preceding_point: bool = False,
    ) -> "Iterator[Step]":
        """Iterate over constant-current events selected by index.

        Args:
            index (int | Sequence[int] | slice | None): Positional selector.
                Supports zero-based integers, sequences of integers, and slices,
                including negative indexing relative to the end. ``None`` yields
                one result per group.
            target (float | None): When supplied, select only rows where
                ``Current [A]`` lies within ``target ± |target| * rtol``.
            rtol (float): Relative tolerance (dimensionless). Defaults to
                ``0.001`` (0.1%).
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Iterator[Step]: Filtered result for the selected groups.
        """
        mask = pl.col("Current [A]") != 0 if target is None else None
        condition = _make_constant_condition("Current [A]", target, rtol, mask=mask)
        f = _Filter("Event", condition)
        return cast(
            "Iterator[Step]",
            f.plural(
                cast("FilterToCycleType", self),
                index,
                include_preceding_point=include_preceding_point,
            ),
        )

    def constant_voltage(
        self,
        index: "IndexType | None" = None,
        *,
        target: float | None = None,
        rtol: float = 0.001,
        include_preceding_point: bool = False,
    ) -> "Step":
        """Filter constant-voltage events selected by index.

        Args:
            index (int | Sequence[int] | slice | None): Positional selector.
                Supports zero-based integers, sequences of integers, and slices,
                including negative indexing relative to the end. ``None`` returns
                all matching rows as a single result.
            target (float | None): When supplied, select only rows where
                ``Voltage [V]`` lies within ``target ± |target| * rtol``.
                When ``None``, the signed mode of ``Voltage [V]`` is used.
            rtol (float): Relative tolerance (dimensionless). Defaults to
                ``0.001`` (0.1%).
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Step: Filtered result for the selected groups.
        """
        condition = _make_constant_condition("Voltage [V]", target, rtol)
        f = _Filter("Event", condition)
        return cast(
            "Step",
            f.singular(
                cast("FilterToCycleType", self),
                index,
                include_preceding_point=include_preceding_point,
            ),
        )

    def iter_constant_voltage(
        self,
        index: "IndexType | None" = None,
        *,
        target: float | None = None,
        rtol: float = 0.001,
        include_preceding_point: bool = False,
    ) -> "Iterator[Step]":
        """Iterate over constant-voltage events selected by index.

        Args:
            index (int | Sequence[int] | slice | None): Positional selector.
                Supports zero-based integers, sequences of integers, and slices,
                including negative indexing relative to the end. ``None`` yields
                one result per group.
            target (float | None): When supplied, select only rows where
                ``Voltage [V]`` lies within ``target ± |target| * rtol``.
                When ``None``, the signed mode of ``Voltage [V]`` is used.
            rtol (float): Relative tolerance (dimensionless). Defaults to
                ``0.001`` (0.1%).
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Iterator[Step]: Filtered result for the selected groups.
        """
        condition = _make_constant_condition("Voltage [V]", target, rtol)
        f = _Filter("Event", condition)
        return cast(
            "Iterator[Step]",
            f.plural(
                cast("FilterToCycleType", self),
                index,
                include_preceding_point=include_preceding_point,
            ),
        )


class CycleFiltersMixin:
    """Mixin providing cycle and charge/discharge filter methods."""

    def cycle(
        self,
        index: "IndexType | None" = None,
        include_preceding_point: bool = False,
    ) -> "Cycle":
        """Filter cycles selected by index.

        Args:
            index (int | Sequence[int] | slice | None): Positional selector.
                Supports zero-based integers, sequences of integers, and slices,
                including negative indexing relative to the end. ``None`` returns
                all matching rows as a single result.
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Cycle: Filtered result for the selected groups.
        """
        return cast(
            "Cycle",
            _cycle_f.singular(
                cast("FilterToCycleType", self),
                index,
                include_preceding_point=include_preceding_point,
            ),
        )

    def iter_cycle(
        self,
        index: "IndexType | None" = None,
        include_preceding_point: bool = False,
    ) -> "Iterator[Cycle]":
        """Iterate over cycles selected by index.

        Args:
            index (int | Sequence[int] | slice | None): Positional selector.
                Supports zero-based integers, sequences of integers, and slices,
                including negative indexing relative to the end. ``None`` yields
                one result per group.
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Iterator[Cycle]: Filtered result for the selected groups.
        """
        return cast(
            "Iterator[Cycle]",
            _cycle_f.plural(
                cast("FilterToCycleType", self),
                index,
                include_preceding_point=include_preceding_point,
            ),
        )

    def charge(
        self,
        index: "IndexType | None" = None,
        include_preceding_point: bool = False,
    ) -> "Step":
        """Filter charge events selected by index.

        Args:
            index (int | Sequence[int] | slice | None): Positional selector.
                Supports zero-based integers, sequences of integers, and slices,
                including negative indexing relative to the end. ``None`` returns
                all matching rows as a single result.
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Step: Filtered result for the selected groups.
        """
        return cast(
            "Step",
            _charge_f.singular(
                cast("FilterToCycleType", self),
                index,
                include_preceding_point=include_preceding_point,
            ),
        )

    def iter_charge(
        self,
        index: "IndexType | None" = None,
        include_preceding_point: bool = False,
    ) -> "Iterator[Step]":
        """Iterate over charge events selected by index.

        Args:
            index (int | Sequence[int] | slice | None): Positional selector.
                Supports zero-based integers, sequences of integers, and slices,
                including negative indexing relative to the end. ``None`` yields
                one result per group.
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Iterator[Step]: Filtered result for the selected groups.
        """
        return cast(
            "Iterator[Step]",
            _charge_f.plural(
                cast("FilterToCycleType", self),
                index,
                include_preceding_point=include_preceding_point,
            ),
        )

    def discharge(
        self,
        index: "IndexType | None" = None,
        include_preceding_point: bool = False,
    ) -> "Step":
        """Filter discharge events selected by index.

        Args:
            index (int | Sequence[int] | slice | None): Positional selector.
                Supports zero-based integers, sequences of integers, and slices,
                including negative indexing relative to the end. ``None`` returns
                all matching rows as a single result.
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Step: Filtered result for the selected groups.
        """
        return cast(
            "Step",
            _discharge_f.singular(
                cast("FilterToCycleType", self),
                index,
                include_preceding_point=include_preceding_point,
            ),
        )

    def iter_discharge(
        self,
        index: "IndexType | None" = None,
        include_preceding_point: bool = False,
    ) -> "Iterator[Step]":
        """Iterate over discharge events selected by index.

        Args:
            index (int | Sequence[int] | slice | None): Positional selector.
                Supports zero-based integers, sequences of integers, and slices,
                including negative indexing relative to the end. ``None`` yields
                one result per group.
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Iterator[Step]: Filtered result for the selected groups.
        """
        return cast(
            "Iterator[Step]",
            _discharge_f.plural(
                cast("FilterToCycleType", self),
                index,
                include_preceding_point=include_preceding_point,
            ),
        )

    def chargeordischarge(
        self,
        index: "IndexType | None" = None,
        include_preceding_point: bool = False,
    ) -> "Step":
        """Filter non-rest events selected by index.

        Args:
            index (int | Sequence[int] | slice | None): Positional selector.
                Supports zero-based integers, sequences of integers, and slices,
                including negative indexing relative to the end. ``None`` returns
                all matching rows as a single result.
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Step: Filtered result for the selected groups.
        """
        return cast(
            "Step",
            _chargeordischarge_f.singular(
                cast("FilterToCycleType", self),
                index,
                include_preceding_point=include_preceding_point,
            ),
        )

    def iter_chargeordischarge(
        self,
        index: "IndexType | None" = None,
        include_preceding_point: bool = False,
    ) -> "Iterator[Step]":
        """Iterate over non-rest events selected by index.

        Args:
            index (int | Sequence[int] | slice | None): Positional selector.
                Supports zero-based integers, sequences of integers, and slices,
                including negative indexing relative to the end. ``None`` yields
                one result per group.
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Iterator[Step]: Filtered result for the selected groups.
        """
        return cast(
            "Iterator[Step]",
            _chargeordischarge_f.plural(
                cast("FilterToCycleType", self),
                index,
                include_preceding_point=include_preceding_point,
            ),
        )

    def rest(
        self,
        index: "IndexType | None" = None,
        include_preceding_point: bool = False,
    ) -> "Step":
        """Filter rest events selected by index.

        Args:
            index (int | Sequence[int] | slice | None): Positional selector.
                Supports zero-based integers, sequences of integers, and slices,
                including negative indexing relative to the end. ``None`` returns
                all matching rows as a single result.
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Step: Filtered result for the selected groups.
        """
        return cast(
            "Step",
            _rest_f.singular(
                cast("FilterToCycleType", self),
                index,
                include_preceding_point=include_preceding_point,
            ),
        )

    def iter_rest(
        self,
        index: "IndexType | None" = None,
        include_preceding_point: bool = False,
    ) -> "Iterator[Step]":
        """Iterate over rest events selected by index.

        Args:
            index (int | Sequence[int] | slice | None): Positional selector.
                Supports zero-based integers, sequences of integers, and slices,
                including negative indexing relative to the end. ``None`` yields
                one result per group.
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Iterator[Step]: Filtered result for the selected groups.
        """
        return cast(
            "Iterator[Step]",
            _rest_f.plural(
                cast("FilterToCycleType", self),
                index,
                include_preceding_point=include_preceding_point,
            ),
        )


class Procedure(CycleFiltersMixin, StepFiltersMixin, RawData):
    """A class for a procedure in a battery experiment."""

    readme_dict: dict[str, dict[str, list[str | int | tuple[int, int, int]]]]
    """A dictionary representing the data contained in the README yaml file."""

    cycle_info: list[tuple[int, int, int]] = []
    """A list of tuples representing the cycle information from the README yaml file.

    The tuple format is
    :code:`(start step (inclusive), end step (inclusive), cycle count)`.
    """

    def model_post_init(self, __context: Any) -> None:
        """Create a procedure class."""
        super().model_post_init(self)
        self.zero_column(
            "Time [s]",
            "Procedure Time [s]",
            "Time elapsed since beginning of procedure.",
        )

        self.zero_column(
            "Capacity [Ah]",
            "Procedure Capacity [Ah]",
            "The net charge passed since beginning of procedure.",
        )
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

    def experiment(
        self,
        *experiment_names: str,
        include_preceding_point: bool = False,
    ) -> "Experiment":
        """Return an experiment object from the procedure.

        Args:
            experiment_names: Variable-length argument list of experiment names.
            include_preceding_point: When ``True``, prepend the data point
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
        mask = pl.col("Step").is_in(flattened_steps)
        if include_preceding_point:
            mask = _include_preceding_row(mask)
        lf_filtered = self.lf.filter(mask)
        cycles_list: list[tuple[int, int, int]] = []
        if len(experiment_names) > 1:
            warnings.warn(
                "Multiple experiments selected. Cycles will be inferred from "
                "the step numbers.",
            )
        elif "Cycles" in self.readme_dict[experiment_names[0]]:
            # ignore type on below line due to persistent mypy warnings about
            # incompatible types
            cycles_list = self.readme_dict[experiment_names[0]]["Cycles"]  # type: ignore

        return Experiment(
            lf=lf_filtered,
            info=self.info,
            column_definitions=self.column_definitions,
            step_descriptions=self.step_descriptions,
            cycle_info=cycles_list,
        )

    def remove_experiment(self, *experiment_names: str) -> None:
        """Remove an experiment from the procedure.

        Args:
            experiment_names (str):
                Variable-length argument list of experiment names.
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
            pl.col("Step").is_in(flattened_steps).not_(),
        ]
        for experiment_name in experiment_names:
            self.readme_dict.pop(experiment_name)
        self.model_post_init(self)
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

        The data must be timestamped, with a column that can be interpreted in
        DateTime format. The data will be interpolated to the procedure's time.

        Args:
            filepath (str): The path to the external file.
            importing_columns (List[str] | dict[str, str]):
                The columns to import from the external file. If a list, the columns
                will be imported as is. If a dict, the keys are the columns in the data
                you want to import and the values are the columns you want to rename
                them to.
            date_column_name (str, optional):
                The name of the date column in the external data. Defaults to "Date".
        """
        external_data = self.load_external_file(filepath)
        if isinstance(importing_columns, dict):
            external_data = external_data.select(
                [date_column_name] + list(importing_columns.keys()),
            )
            external_data = external_data.rename(importing_columns)
        elif isinstance(importing_columns, list):
            external_data = external_data.select([date_column_name] + importing_columns)
        self.add_new_data_columns(external_data, date_column_name)


class Experiment(CycleFiltersMixin, StepFiltersMixin, RawData):
    """A class for an experiment in a battery experimental procedure."""

    cycle_info: list[tuple[int, int, int]] = []
    """A list of tuples representing the cycle information from the README yaml file.

    The tuple format is
    :code:`(start step (inclusive), end step (inclusive), cycle count)`.
    """

    def model_post_init(self, __context: Any) -> None:
        """Create an experiment class."""
        super().model_post_init(self)
        self.zero_column(
            "Time [s]",
            "Experiment Time [s]",
            "Time elapsed since beginning of experiment.",
        )

        self.zero_column(
            "Capacity [Ah]",
            "Experiment Capacity [Ah]",
            "The net charge passed since beginning of experiment.",
        )


class Cycle(CycleFiltersMixin, StepFiltersMixin, RawData):
    """A class for a cycle in a battery experimental procedure."""

    cycle_info: list[tuple[int, int, int]] = []
    """A list of tuples representing the cycle information from the README yaml file.

    The tuple format is
    :code:`(start step (inclusive), end step (inclusive), cycle count)`.
    """

    def model_post_init(self, __context: Any) -> None:
        """Create a cycle class."""
        super().model_post_init(self)
        self.zero_column(
            "Time [s]",
            "Cycle Time [s]",
            "Time elapsed since beginning of cycle.",
        )

        self.zero_column(
            "Capacity [Ah]",
            "Cycle Capacity [Ah]",
            "The net charge passed since beginning of cycle.",
        )


class Step(StepFiltersMixin, RawData):
    """A class for a step in a battery experimental procedure."""

    def model_post_init(self, __context: Any) -> None:
        """Create a step class."""
        super().model_post_init(self)
        self.zero_column(
            "Time [s]",
            "Step Time [s]",
            "Time elapsed since beginning of step.",
        )

        self.zero_column(
            "Capacity [Ah]",
            "Step Capacity [Ah]",
            "The net charge passed since beginning of step.",
        )
