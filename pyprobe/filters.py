"""A module for the filtering classes."""

import warnings
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast

import polars as pl

from pyprobe import utils
from pyprobe.rawdata import RawData

if TYPE_CHECKING:
    from pyprobe.pyprobe_types import (
        FilterToCycleType,
    )


from loguru import logger


def _extend_mask_with_preceding_point(mask: pl.Expr) -> pl.Expr:
    """Extend a boolean mask to include the row preceding each contiguous True run.

    Args:
        mask: A boolean polars expression.

    Returns:
        pl.Expr: A boolean expression that is True for every originally selected
            row plus the row immediately before each run of selected rows.
    """
    return mask | mask.shift(-1).fill_null(False)


def _make_group_marker_expr(column: str, condition: pl.Expr) -> pl.Expr:
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


def _count_condition_groups(
    lf: pl.LazyFrame | pl.DataFrame,
    column: str,
    condition: pl.Expr | None = None,
) -> int:
    """Count the number of distinct groups in lf, optionally within a condition.

    Performs a single lightweight ``.collect()`` — collects only an integer,
    not row data.

    Args:
        lf: The LazyFrame or DataFrame to inspect.
        column: Column whose rank groups are counted.
        condition: When supplied, only groups where this expression is True
            are counted.

    Returns:
        int: Number of distinct groups.
    """
    lazy = lf.lazy() if isinstance(lf, pl.DataFrame) else lf
    count_expr = (
        pl.col(column).n_unique()
        if condition is None
        else _make_group_marker_expr(column, condition).cast(pl.Int32).sum()
    )
    return lazy.select(count_expr).collect().item()


class _Filter:
    """Encapsulates group-filtering logic for a single column and optional condition."""

    def __init__(self, column: str = "Event", condition: pl.Expr | None = None) -> None:
        """Initialize a _Filter instance.

        Args:
            column: The column name to perform filtering on. Defaults to "Event".
            condition: Optional polars expression that defines which rows
                qualify as part of a group. When ``None``, all rows in each
                group are selected.
        """
        self.column = column
        self.condition = condition

    def _build_mask(self, indices: tuple[int | range | slice, ...]) -> pl.Expr:
        """Build a boolean polars expression selecting rows by positional indices.

        Supports positive and negative indexing, ranges, and slices.
        Handles both absolute group ranks and relative indexing from the end.

        Args:
            indices: Tuple of int, range, or slice objects selecting groups
                by position (zero-based).

        Returns:
            pl.Expr: A boolean polars expression where ``True`` indicates
                rows that should be included in the result.
        """
        if len(indices) == 0:
            return self.condition if self.condition is not None else pl.lit(True)

        normalized_indices: list[int | slice] = []
        has_negative = False
        for idx in indices:
            if isinstance(idx, int):
                normalized_indices.append(idx)
                if idx < 0:
                    has_negative = True
            elif isinstance(idx, range):
                normalized_indices.append(slice(idx.start, idx.stop, idx.step))
                if idx.start < 0 or idx.stop < 0:
                    has_negative = True
            elif isinstance(idx, slice):
                normalized_indices.append(idx)
                if (idx.start is not None and idx.start < 0) or (
                    idx.stop is not None and idx.stop < 0
                ):
                    has_negative = True

        if self.condition is not None:
            is_new_matching_event = _make_group_marker_expr(self.column, self.condition)
            asc_rank = is_new_matching_event.cast(pl.Int32).cum_sum()
            if has_negative:
                total_groups = is_new_matching_event.cast(pl.Int32).sum()
                cond_desc_rank = total_groups - asc_rank + 1
            else:
                cond_desc_rank = None
        else:
            asc_rank = pl.col(self.column).rank("dense")
            cond_desc_rank = (
                pl.col(self.column).rank("dense", descending=True)
                if has_negative
                else None
            )

        sub_masks: list[pl.Expr] = []
        for idx in normalized_indices:
            if isinstance(idx, int):
                if idx >= 0:
                    sub_masks.append(asc_rank == idx + 1)
                else:
                    sub_masks.append(cond_desc_rank == -idx)
            else:
                sub_masks.append(_slice_to_mask_expr(idx, asc_rank, cond_desc_rank))

        if not sub_masks:
            return self.condition if self.condition is not None else pl.lit(True)

        combined: pl.Expr = sub_masks[0]
        for m in sub_masks[1:]:
            combined = combined | m

        return (self.condition & combined) if self.condition is not None else combined

    def _expand_positions(
        self,
        lf: pl.LazyFrame,
        indices: tuple[int | range | slice, ...],
    ) -> list[int]:
        """Expand positional indices (int, range, slice) to a flat list of integers.

        Resolves negative indices and slices relative to the total number
        of groups in the condition.

        Args:
            lf: The polars LazyFrame or DataFrame to inspect for group count.
            indices: Tuple of int, range, or slice objects.

        Returns:
            list[int]: Flattened list of integer positions (zero-based).

        Raises:
            ValueError: If a slice has a negative step.
            TypeError: If an unsupported index type is encountered.
        """
        if not indices:
            return list(range(_count_condition_groups(lf, self.column, self.condition)))

        positions: list[int] = []
        total: int | None = None

        def get_total() -> int:
            nonlocal total
            if total is None:
                total = _count_condition_groups(lf, self.column, self.condition)
            return total

        for idx in indices:
            if isinstance(idx, int):
                positions.append(idx)
            elif isinstance(idx, range):
                positions.extend(idx)
            elif isinstance(idx, slice):
                if idx.step is not None and idx.step < 0:
                    raise ValueError("Negative step is not supported in a slice index.")
                step_val = idx.step if idx.step is not None else 1
                start = idx.start if idx.start is not None else 0
                if start >= 0 and idx.stop is not None and idx.stop >= 0:
                    positions.extend(range(start, idx.stop, step_val))
                else:
                    positions.extend(range(*idx.indices(get_total())))
            else:
                raise TypeError(f"Unsupported index type: {type(idx).__name__}")
        return positions

    def _create_result(
        self,
        obj: "FilterToCycleType",
        lf: pl.LazyFrame | pl.DataFrame,
        mask: pl.Expr,
        include_preceding_point: bool,
    ) -> "Cycle | Step":
        """Create a filtered result object (Cycle or Step).

        Applies the mask to the data and returns the appropriate result type
        based on the filter's column.

        Args:
            obj: The source object (Procedure, Experiment, Cycle, or Step).
            lf: The polars LazyFrame or DataFrame to filter.
            mask: Boolean expression selecting rows to include.
            include_preceding_point: When ``True``, include the row immediately
                before each contiguous block of selected rows.

        Returns:
            Cycle | Step: A new Cycle or Step object containing the filtered data.
        """
        if include_preceding_point:
            mask = _extend_mask_with_preceding_point(mask)

        filtered_lf = lf.filter(mask)
        if self.column == "Cycle":
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

    def singular(
        self,
        obj: "FilterToCycleType",
        *indices: int | range | slice,
        include_preceding_point: bool = False,
    ) -> "Cycle | Step":
        """Return a single filtered result for the given indices.

        Args:
            obj: The source object to filter.
            *indices: Positional selectors (int, range, or slice).
            include_preceding_point: When ``True``, include the row immediately
                before each contiguous block of selected rows.

        Returns:
            Cycle | Step: A single filtered result object.
        """
        lf = get_cycle_column(obj) if self.column == "Cycle" else obj.lf
        mask = self._build_mask(indices)
        return self._create_result(obj, lf, mask, include_preceding_point)

    def plural(
        self,
        obj: "FilterToCycleType",
        *indices: int | range | slice,
        include_preceding_point: bool = False,
    ) -> Iterator["Cycle | Step"]:
        """Iterate over filtered results for each selected index.

        Expands indices to individual positions and yields a result object
        for each position.

        Args:
            obj: The source object to filter.
            *indices: Positional selectors (int, range, or slice).
            include_preceding_point: When ``True``, include the row immediately
                before the selected row in each result.

        Yields:
            Cycle | Step: Filtered result objects, one per selected position.
        """
        lf = get_cycle_column(obj) if self.column == "Cycle" else obj.lf
        for i in self._expand_positions(lf, indices):
            mask = self._build_mask((i,))
            yield self._create_result(obj, lf, mask, include_preceding_point)


def _slice_to_mask_expr(
    s: slice,
    asc_rank: pl.Expr,
    desc_rank: pl.Expr | None,
) -> pl.Expr:
    """Convert a slice to a polars boolean expression using rank expressions.

    Args:
        s: The slice to convert. Negative step raises ``ValueError``.
        asc_rank: Ascending rank expression for the target column.
        desc_rank: Descending rank expression for the target column.
            Required if ``s.start`` or ``s.stop`` are negative.

    Returns:
        pl.Expr: Boolean expression selecting rows matched by the slice.

    Raises:
        ValueError: If ``s.step`` is negative or if negative bounds are used
            without a ``desc_rank``.
    """
    if s.step is not None and s.step < 0:
        error_msg = "Negative step is not supported in a slice index."
        logger.error(error_msg)
        raise ValueError(error_msg)

    parts: list[pl.Expr] = []

    if s.start is not None:
        if s.start >= 0:
            parts.append(asc_rank >= s.start + 1)
        else:
            if desc_rank is None:
                error_msg = "Negative slice start requires a descending rank."
                logger.error(error_msg)
                raise ValueError(error_msg)
            parts.append(desc_rank <= -s.start)

    if s.stop is not None:
        if s.stop > 0:
            parts.append(asc_rank <= s.stop)
        elif s.stop < 0:
            if desc_rank is None:
                error_msg = "Negative slice stop requires a descending rank."
                logger.error(error_msg)
                raise ValueError(error_msg)
            parts.append(desc_rank > -s.stop)
        else:  # s.stop == 0
            if s.start is not None and s.start < 0:
                # slice(-n, 0) is treated as slice(-n, None): no upper bound,
                # matching Python's convention that stop=0 with a negative start
                # is open-ended when the caller intends "from -n to end".
                pass
            else:
                # asc_rank starts at 1, so <= 0 is always False → empty result,
                # matching slice(0, 0) / slice(k, 0) for k >= 0.
                parts.append(asc_rank <= 0)

    step_val = s.step
    if step_val is not None and step_val > 1:
        effective_start = s.start if s.start is not None else 0
        if effective_start >= 0:
            anchor = effective_start + 1
            parts.append((asc_rank - anchor) % step_val == 0)
        else:
            assert desc_rank is not None, "desc_rank must be provided when start < 0"
            anchor = -effective_start
            parts.append((anchor - desc_rank) % step_val == 0)

    if not parts:
        return pl.lit(True)

    result: pl.Expr = parts[0]
    for p in parts[1:]:
        result = result & p
    return result


def get_cycle_column(
    filtered_object: "FilterToCycleType",
) -> pl.DataFrame | pl.LazyFrame:
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
            values. When ``None``, the global mode of ``col`` (filtered by
            ``mask`` if given) is used as the target.
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
        *indices: int | range | slice,
        include_preceding_point: bool = False,
    ) -> "Step":
        """Filter step events selected by positional indices.

        Args:
            *indices (int | range | slice): Positional selectors for groups.
                Supports zero-based integers, ranges, and slices, including
                negative indexing relative to the end.
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Step: Filtered result for the selected groups.
        """
        return cast(
            "Step",
            _step_f.singular(
                cast("FilterToCycleType", self),
                *indices,
                include_preceding_point=include_preceding_point,
            ),
        )

    def iter_step(
        self,
        *indices: int | range | slice,
        include_preceding_point: bool = False,
    ) -> Iterator["Step"]:
        """Iterate over step events selected by positional indices.

        Args:
            *indices (int | range | slice): Positional selectors for groups.
                Supports zero-based integers, ranges, and slices, including
                negative indexing relative to the end.
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Iterator[Step]: Filtered result for the selected groups.
        """
        return cast(
            Iterator["Step"],
            _step_f.plural(
                cast("FilterToCycleType", self),
                *indices,
                include_preceding_point=include_preceding_point,
            ),
        )

    def constant_current(
        self,
        *indices: int | range | slice,
        target: float | None = None,
        rtol: float = 0.001,
        include_preceding_point: bool = False,
    ) -> "Step":
        """Filter constant-current events selected by positional indices.

        Args:
            *indices (int | range | slice): Positional selectors for groups.
                Supports zero-based integers, ranges, and slices, including
                negative indexing relative to the end.
            target (float | None): When supplied, select only rows where
                ``Current [A]`` lies within ``target ± |target| * rtol``. Sign
                is preserved: ``target=1.0`` matches only positive (charge)
                values; ``target=-1.0`` matches only negative (discharge)
                values. When ``None``, the global mode of non-zero
                ``Current [A]`` values is used as the target (backward-
                compatible behaviour).
            rtol (float): Relative tolerance (dimensionless) controlling the
                acceptance band as a fraction of ``|target|``. Defaults to
                ``0.001`` (0.1%). Near-zero targets collapse the band; use a
                dedicated filter for rest steps instead.
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
                *indices,
                include_preceding_point=include_preceding_point,
            ),
        )

    def iter_constant_current(
        self,
        *indices: int | range | slice,
        target: float | None = None,
        rtol: float = 0.001,
        include_preceding_point: bool = False,
    ) -> Iterator["Step"]:
        """Iterate over constant-current events selected by positional indices.

        Args:
            *indices (int | range | slice): Positional selectors for groups.
                Supports zero-based integers, ranges, and slices, including
                negative indexing relative to the end.
            target (float | None): When supplied, select only rows where
                ``Current [A]`` lies within ``target ± |target| * rtol``. Sign
                is preserved: ``target=1.0`` matches only positive (charge)
                values; ``target=-1.0`` matches only negative (discharge)
                values. When ``None``, the global mode of non-zero
                ``Current [A]`` values is used as the target (backward-
                compatible behaviour).
            rtol (float): Relative tolerance (dimensionless) controlling the
                acceptance band as a fraction of ``|target|``. Defaults to
                ``0.001`` (0.1%). Near-zero targets collapse the band; use a
                dedicated filter for rest steps instead.
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Iterator[Step]: Filtered result for the selected groups.
        """
        mask = pl.col("Current [A]") != 0 if target is None else None
        condition = _make_constant_condition("Current [A]", target, rtol, mask=mask)
        f = _Filter("Event", condition)
        return cast(
            Iterator["Step"],
            f.plural(
                cast("FilterToCycleType", self),
                *indices,
                include_preceding_point=include_preceding_point,
            ),
        )

    def constant_voltage(
        self,
        *indices: int | range | slice,
        target: float | None = None,
        rtol: float = 0.001,
        include_preceding_point: bool = False,
    ) -> "Step":
        """Filter constant-voltage events selected by positional indices.

        Args:
            *indices (int | range | slice): Positional selectors for groups.
                Supports zero-based integers, ranges, and slices, including
                negative indexing relative to the end.
            target (float | None): When supplied, select only rows where
                ``Voltage [V]`` lies within ``target ± |target| * rtol``. Sign
                is preserved: a positive target only matches positive voltages.
                When ``None``, the global mode of ``Voltage [V]`` is used as
                the target (backward-compatible behaviour). Note: 0 V is a
                valid CV target; the band collapses if ``target`` is exactly
                zero — use a rest filter instead.
            rtol (float): Relative tolerance (dimensionless) controlling the
                acceptance band as a fraction of ``|target|``. Defaults to
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
                *indices,
                include_preceding_point=include_preceding_point,
            ),
        )

    def iter_constant_voltage(
        self,
        *indices: int | range | slice,
        target: float | None = None,
        rtol: float = 0.001,
        include_preceding_point: bool = False,
    ) -> Iterator["Step"]:
        """Iterate over constant-voltage events selected by positional indices.

        Args:
            *indices (int | range | slice): Positional selectors for groups.
                Supports zero-based integers, ranges, and slices, including
                negative indexing relative to the end.
            target (float | None): When supplied, select only rows where
                ``Voltage [V]`` lies within ``target ± |target| * rtol``. Sign
                is preserved: a positive target only matches positive voltages.
                When ``None``, the global mode of ``Voltage [V]`` is used as
                the target (backward-compatible behaviour). Note: 0 V is a
                valid CV target; the band collapses if ``target`` is exactly
                zero — use a rest filter instead.
            rtol (float): Relative tolerance (dimensionless) controlling the
                acceptance band as a fraction of ``|target|``. Defaults to
                ``0.001`` (0.1%).
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Iterator[Step]: Filtered result for the selected groups.
        """
        condition = _make_constant_condition("Voltage [V]", target, rtol)
        f = _Filter("Event", condition)
        return cast(
            Iterator["Step"],
            f.plural(
                cast("FilterToCycleType", self),
                *indices,
                include_preceding_point=include_preceding_point,
            ),
        )


class CycleFiltersMixin:
    """Mixin providing cycle and charge/discharge filter methods."""

    def cycle(
        self,
        *indices: int | range | slice,
        include_preceding_point: bool = False,
    ) -> "Cycle":
        """Filter cycles selected by positional indices.

        Args:
            *indices (int | range | slice): Positional selectors for groups.
                Supports zero-based integers, ranges, and slices, including
                negative indexing relative to the end.
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Cycle: Filtered result for the selected groups.
        """
        return cast(
            "Cycle",
            _cycle_f.singular(
                cast("FilterToCycleType", self),
                *indices,
                include_preceding_point=include_preceding_point,
            ),
        )

    def iter_cycle(
        self,
        *indices: int | range | slice,
        include_preceding_point: bool = False,
    ) -> Iterator["Cycle"]:
        """Iterate over cycles selected by positional indices.

        Args:
            *indices (int | range | slice): Positional selectors for groups.
                Supports zero-based integers, ranges, and slices, including
                negative indexing relative to the end.
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Iterator[Cycle]: Filtered result for the selected groups.
        """
        return cast(
            Iterator["Cycle"],
            _cycle_f.plural(
                cast("FilterToCycleType", self),
                *indices,
                include_preceding_point=include_preceding_point,
            ),
        )

    def charge(
        self,
        *indices: int | range | slice,
        include_preceding_point: bool = False,
    ) -> "Step":
        """Filter charge events selected by positional indices.

        Args:
            *indices (int | range | slice): Positional selectors for groups.
                Supports zero-based integers, ranges, and slices, including
                negative indexing relative to the end.
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Step: Filtered result for the selected groups.
        """
        return cast(
            "Step",
            _charge_f.singular(
                cast("FilterToCycleType", self),
                *indices,
                include_preceding_point=include_preceding_point,
            ),
        )

    def iter_charge(
        self,
        *indices: int | range | slice,
        include_preceding_point: bool = False,
    ) -> Iterator["Step"]:
        """Iterate over charge events selected by positional indices.

        Args:
            *indices (int | range | slice): Positional selectors for groups.
                Supports zero-based integers, ranges, and slices, including
                negative indexing relative to the end.
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Iterator[Step]: Filtered result for the selected groups.
        """
        return cast(
            Iterator["Step"],
            _charge_f.plural(
                cast("FilterToCycleType", self),
                *indices,
                include_preceding_point=include_preceding_point,
            ),
        )

    def discharge(
        self,
        *indices: int | range | slice,
        include_preceding_point: bool = False,
    ) -> "Step":
        """Filter discharge events selected by positional indices.

        Args:
            *indices (int | range | slice): Positional selectors for groups.
                Supports zero-based integers, ranges, and slices, including
                negative indexing relative to the end.
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Step: Filtered result for the selected groups.
        """
        return cast(
            "Step",
            _discharge_f.singular(
                cast("FilterToCycleType", self),
                *indices,
                include_preceding_point=include_preceding_point,
            ),
        )

    def iter_discharge(
        self,
        *indices: int | range | slice,
        include_preceding_point: bool = False,
    ) -> Iterator["Step"]:
        """Iterate over discharge events selected by positional indices.

        Args:
            *indices (int | range | slice): Positional selectors for groups.
                Supports zero-based integers, ranges, and slices, including
                negative indexing relative to the end.
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Iterator[Step]: Filtered result for the selected groups.
        """
        return cast(
            Iterator["Step"],
            _discharge_f.plural(
                cast("FilterToCycleType", self),
                *indices,
                include_preceding_point=include_preceding_point,
            ),
        )

    def chargeordischarge(
        self,
        *indices: int | range | slice,
        include_preceding_point: bool = False,
    ) -> "Step":
        """Filter non-rest events selected by positional indices.

        Args:
            *indices (int | range | slice): Positional selectors for groups.
                Supports zero-based integers, ranges, and slices, including
                negative indexing relative to the end.
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Step: Filtered result for the selected groups.
        """
        return cast(
            "Step",
            _chargeordischarge_f.singular(
                cast("FilterToCycleType", self),
                *indices,
                include_preceding_point=include_preceding_point,
            ),
        )

    def iter_chargeordischarge(
        self,
        *indices: int | range | slice,
        include_preceding_point: bool = False,
    ) -> Iterator["Step"]:
        """Iterate over non-rest events selected by positional indices.

        Args:
            *indices (int | range | slice): Positional selectors for groups.
                Supports zero-based integers, ranges, and slices, including
                negative indexing relative to the end.
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Iterator[Step]: Filtered result for the selected groups.
        """
        return cast(
            Iterator["Step"],
            _chargeordischarge_f.plural(
                cast("FilterToCycleType", self),
                *indices,
                include_preceding_point=include_preceding_point,
            ),
        )

    def rest(
        self,
        *indices: int | range | slice,
        include_preceding_point: bool = False,
    ) -> "Step":
        """Filter rest events selected by positional indices.

        Args:
            *indices (int | range | slice): Positional selectors for groups.
                Supports zero-based integers, ranges, and slices, including
                negative indexing relative to the end.
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Step: Filtered result for the selected groups.
        """
        return cast(
            "Step",
            _rest_f.singular(
                cast("FilterToCycleType", self),
                *indices,
                include_preceding_point=include_preceding_point,
            ),
        )

    def iter_rest(
        self,
        *indices: int | range | slice,
        include_preceding_point: bool = False,
    ) -> Iterator["Step"]:
        """Iterate over rest events selected by positional indices.

        Args:
            *indices (int | range | slice): Positional selectors for groups.
                Supports zero-based integers, ranges, and slices, including
                negative indexing relative to the end.
            include_preceding_point (bool): When ``True``, include the data
                row immediately before each contiguous block of selected rows.

        Returns:
            Iterator[Step]: Filtered result for the selected groups.
        """
        return cast(
            Iterator["Step"],
            _rest_f.plural(
                cast("FilterToCycleType", self),
                *indices,
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
            mask = _extend_mask_with_preceding_point(mask)
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
