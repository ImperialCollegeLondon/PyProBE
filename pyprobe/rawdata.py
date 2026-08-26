"""A module for the CyclingData class."""

import warnings
from pathlib import Path
from typing import Any, Literal, Optional, Union

import bdf
import polars as pl
from loguru import logger

from pyprobe.columns import BDF, CORE_COLUMN_GROUPS, CORE_COLUMNS, Column
from pyprobe.result import Table
from pyprobe.utils import deprecated


class CyclingData(Table):
    """A class for holding battery cycler data in BDF-standard column format.

    This is the default object returned when data is loaded into PyProBE with the
    standard methods of the :class:`~pyprobe.cell.Cell` class. It is a subclass of
    :class:`~pyprobe.result.Table` and can be used in the same way.

    .. note::
        ``RawData`` is a deprecated alias of ``CyclingData``. Existing code using
        ``RawData`` keeps working but emits a deprecation warning on construction.

    The CyclingData object validates that the required BDF columns are resolvable
    from the data via :class:`~pyprobe.columns.ColumnDict`:

    - At least one time column: ``Unix Time / s`` (preferred) or ``Test Time / s``
    - ``Current / A``
    - ``Voltage / V``

    The following BDF columns are optional but emit a warning if absent:

    - ``Net Capacity / Ah``
    - ``Step Count / 1``
    - ``Step ID``
    """

    step_descriptions: dict[str, list[str | int | None]]
    """A dictionary containing the fields 'Step' and 'Description'.

    - 'Step' is a list of step numbers (from the README).
    - 'Description' is a list of corresponding descriptions in PyBaMM Experiment format.
    """

    def __init__(
        self,
        lf: pl.LazyFrame | pl.DataFrame | str,
        metadata: bdf.Metadata,
        column_definitions: dict[str, str] | None = None,
        step_descriptions: dict[str, list[str | int | None]] | None = None,
        _path: Path | None = None,
    ) -> None:
        """Create a CyclingData object with BDF-column validation."""
        super().__init__(
            lf=lf, metadata=metadata, column_definitions=column_definitions, _path=_path
        )

        if step_descriptions is None:
            self.step_descriptions = {}
        else:
            self.step_descriptions = {
                key: value.copy() for key, value in step_descriptions.items()
            }

        self._check_required_columns()

    def _check_required_columns(self) -> None:
        """Validate that required and optional BDF columns are resolvable.

        Required columns must be resolvable from the data (either as a direct
        data column or via a recipe derivation). Optional columns emit a warning
        if unavailable but do not raise an error.

        Time column validation: at least one of Unix Time or Test Time must be
        resolvable (Unix Time is preferred).

        Raises:
            ValueError: If neither Unix Time nor Test Time can be resolved.
            ValueError: If any required BDF column (Current, Voltage) cannot be
                resolved from available data.
        """
        col_set = self.columns

        # Validate required column groups (at least one member must resolve)
        for group, status in CORE_COLUMN_GROUPS.items():
            if status != "required":
                continue
            if not any(col_set.can_resolve(bdf_col) for bdf_col in group):
                names = " or ".join(
                    f"'{bdf_col.quantity} / {bdf_col.unit}'" for bdf_col in group
                )
                error_msg = (
                    f"Required time column: either {names} must be resolvable "
                    "from available columns."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Validate required and optional columns
        for bdf_col, status in CORE_COLUMNS.items():
            if status == "silent":
                continue
            if col_set.can_resolve(bdf_col):
                continue
            if status == "required":
                error_msg = (
                    f"Required BDF column '{bdf_col.name}' is not resolvable "
                    f"from available columns."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            logger.warning(
                f"Optional BDF column '{bdf_col.name}' is not resolvable; some "
                "features may be unavailable."
            )

    def extend(  # type: ignore[override]
        self,
        other: Union["Table", list["Table"]],  # noqa: UP007
        *,
        order: Literal["start_time", "given"] = "start_time",
        time: Literal["continue", "elapsed", "keep"] = "continue",
        step_id: Literal["offset", "keep"] = "offset",
        concat_method: str = "diagonal",
    ) -> None:
        """Extend this cycling data with the rows of one or more other objects.

        The sources are ordered, then concatenated.

        Args:
            other: The other cycling data object(s) to extend with.
            order: ``"start_time"`` orders the sources by the first
                ``Unix Time / s`` value of each one, falling back to the given
                order where an object holds no such column. ``"given"`` keeps
                the order the caller gave.
            time: How the ``Test Time / s`` column crosses a source boundary.
                ``"continue"`` adds the last value of one source to the next.
                ``"elapsed"`` derives the test time from ``Unix Time / s``, so
                the real gap between sources survives. ``"keep"`` stacks the
                recorded values verbatim.
            step_id: Reserved for a future rule on how the ``Step ID`` column
                crosses a source boundary. It has no effect yet.
            concat_method: The method to use for concatenation. See the
                :func:`polars.concat` documentation for the available values.

        Raises:
            ValueError: ``time`` is ``"elapsed"`` and a source holds no
                ``Unix Time / s`` column.
        """
        if not isinstance(other, list):
            other = [other]
        sources: list[Table] = [self, *other]
        if order == "start_time":
            sources = self._ordered_by_start_time(sources)
        frames = [source.lf for source in sources]
        base_frame, other_frames = self._verify_compatible_frames(
            frames[0], frames[1:], mode="collect all"
        )
        frames = [base_frame, *other_frames]
        frames = self._with_time_rule(frames, time)
        self.lf = pl.concat(frames, how=concat_method)
        original_column_definitions = self.column_definitions.copy()
        for source in other:
            self.column_definitions.update(source.column_definitions)
        self.column_definitions.update(original_column_definitions)

    @staticmethod
    def _with_time_rule(
        frames: list[pl.LazyFrame],
        time: Literal["continue", "elapsed", "keep"],
    ) -> list[pl.LazyFrame]:
        """Adjust the ``Test Time / s`` column of each frame across a boundary.

        Args:
            frames: The frames to adjust, in the order they will be
                concatenated.
            time: ``"continue"`` adds the last ``Test Time / s`` value of one
                frame to the next. ``"elapsed"`` derives the test time from
                ``Unix Time / s``, relative to the first value of the first
                frame. ``"keep"`` returns the frames unchanged.

        Returns:
            The frames with an adjusted ``Test Time / s`` column, where
            ``time`` calls for one.

        Raises:
            ValueError: ``time`` is ``"elapsed"`` and a frame holds no
                ``Unix Time / s`` column.
        """
        if time == "keep":
            return frames
        test_time_col = BDF.TEST_TIME_SECOND.name
        unix_time_col = BDF.UNIX_TIME_SECOND.name
        if time == "elapsed":
            for frame in frames:
                if unix_time_col not in frame.collect_schema().names():
                    raise ValueError(
                        f"An elapsed test time needs a '{unix_time_col}' "
                        "column on every source."
                    )
            start = frames[0].select(pl.col(unix_time_col).first()).collect().item()
            return [
                frame.with_columns((pl.col(unix_time_col) - start).alias(test_time_col))
                for frame in frames
            ]
        # The remaining case is time == "continue".
        return CyclingData._with_running_offset(frames, test_time_col, "last")

    @staticmethod
    def _with_running_offset(
        frames: list[pl.LazyFrame],
        column: str,
        aggregate: Literal["max", "last"],
    ) -> list[pl.LazyFrame]:
        """Add a running offset to a column across a frame boundary.

        Args:
            frames: The frames to adjust, in the order they will be
                concatenated.
            column: The column to offset.
            aggregate: How the next offset is read from a frame once it is
                shifted. ``"max"`` reads the maximum value. ``"last"`` reads
                the last row.

        Returns:
            The frames with the column offset, where a frame holds it. A
            frame without the column is returned unchanged, and the running
            offset carries over unchanged to the next frame.
        """
        adjusted_frames = []
        offset = 0.0
        for frame in frames:
            if column not in frame.collect_schema().names():
                adjusted_frames.append(frame)
                continue
            shifted = frame.with_columns((pl.col(column) + offset).alias(column))
            if aggregate == "max":
                aggregated = pl.col(column).max()
            else:
                aggregated = pl.col(column).last()
            offset = shifted.select(aggregated).collect().item()
            adjusted_frames.append(shifted)
        return adjusted_frames

    @staticmethod
    def _ordered_by_start_time(sources: list["Table"]) -> list["Table"]:
        """Order the sources by the first ``Unix Time / s`` value of each one.

        Args:
            sources: The sources to order.

        Returns:
            The sources sorted by the first ``Unix Time / s`` value of each
            one, or unchanged where an object holds no such column.
        """
        unix_time_col = BDF.UNIX_TIME_SECOND.name
        starts: list[float] = []
        for source in sources:
            if unix_time_col not in source.lf.collect_schema().names():
                return sources
            starts.append(
                source.lf.select(pl.col(unix_time_col).first()).collect().item()
            )
        return [
            source
            for _, source in sorted(
                zip(starts, sources, strict=True), key=lambda pair: pair[0]
            )
        ]

    def zero_column(
        self,
        column: str | Column,
    ) -> "CyclingData":
        """Zero a column relative to the start of this data slice.

        Returns a new CyclingData object with *column* shifted so its first row
        is zero. The original object is not modified.

        Args:
            column: A BDF column string or :class:`~pyprobe.columns.Column`
                instance resolvable via
                :meth:`~pyprobe.columns.ColumnDict.resolve` (e.g.
                ``"Net Capacity / Ah"`` or ``BDF.NET_CAPACITY_AH``).

        Returns:
            A new CyclingData with the zeroed column.
        """
        column_str = str(column)
        expr = self.columns.resolve(column)
        new_lf = self.lf.with_columns(
            (expr - expr.first()).alias(column_str),
        )
        return CyclingData(
            lf=new_lf,
            metadata=self.metadata,
            column_definitions=self.column_definitions,
            step_descriptions=self.step_descriptions,
        )

    @property
    @deprecated(
        reason='Use ``range("Net Capacity / Ah").item()`` instead.',
        version="3.0.0",
    )
    def capacity(self) -> float:
        """Calculate the net capacity passed.

        Returns:
            float: The net capacity passed.
        """
        return self.range(BDF.NET_CAPACITY_AH).item()

    def set_soc(
        self,
        reference_capacity: float | None = None,
        reference_charge: Optional["CyclingData"] = None,
    ) -> None:
        """Add an SOC column to the data.

        Apply this method on a filtered data object to add an ``SOC`` column.
        This column remains with the data if the object is filtered further.

        The SOC column is calculated either relative to a provided reference capacity
        value, a reference charge (provided as a CyclingData object), or the
        maximum capacity delta across the data in the CyclingData object upon
        which this method is called.

        Args:
            reference_capacity: The reference capacity value.
            reference_charge: A CyclingData object containing a charge to use as
                a reference.
        """
        cap_col = BDF.NET_CAPACITY_AH.name
        if reference_capacity is None:
            reference_capacity = float(
                self.lf.select(
                    (pl.col(cap_col).max() - pl.col(cap_col).min()).alias("_ref")
                )
                .collect()
                .item()
            )
        if reference_charge is None:
            self.lf = self.lf.with_columns(
                (
                    (pl.col(cap_col) - pl.col(cap_col).max() + reference_capacity)
                    / reference_capacity
                    * 100
                ).alias("SOC / %"),
            )
        else:
            unix_col = BDF.UNIX_TIME_SECOND.name
            reference_charge_data = reference_charge.lf.select(unix_col, cap_col)
            self.lf = self.lf.join(
                reference_charge_data,
                on=unix_col,
                how="left",
            )
            right_col = cap_col + "_right"
            full_ref = float(
                self.lf.select(pl.col(right_col).max().alias("_fc")).collect().item()
            )
            self.lf = self.lf.drop(right_col)
            self.lf = self.lf.with_columns(
                (
                    (pl.col(cap_col) - full_ref + reference_capacity)
                    / reference_capacity
                    * 100
                ).alias("SOC / %"),
            )
        self.define_column("SOC / %", "The full cell State-of-Charge.")

    @deprecated(
        reason="Use set_soc instead.",
        version="2.0.1",
    )
    def set_SOC(  # noqa: N802
        self,
        reference_capacity: float | None = None,
        reference_charge: Optional["CyclingData"] = None,
    ) -> None:
        """Add an SOC column to the data.

        Args:
            reference_capacity: The reference capacity value.
            reference_charge: A CyclingData object containing a charge to use as
                a reference.
        """
        self.set_soc(reference_capacity, reference_charge)

    def set_reference_capacity(self, reference_capacity: float | None = None) -> None:
        """Fix the capacity to a reference value.

        Apply this method on a filtered data object to fix the capacity to a
        reference. This calculates a permanent column named
        ``Capacity - Referenced / Ah`` in the data.

        Args:
            reference_capacity: The reference capacity value.
        """
        cap_col = BDF.NET_CAPACITY_AH.name
        if reference_capacity is None:
            reference_capacity = float(
                self.lf.select(
                    (pl.col(cap_col).max() - pl.col(cap_col).min()).alias("_ref")
                )
                .collect()
                .item()
            )
        self.lf = self.lf.with_columns(
            (pl.col(cap_col) - pl.col(cap_col).max() + reference_capacity).alias(
                "Capacity - Referenced / Ah"
            ),
        )

    @property
    def pybamm_experiment(self) -> list[str | tuple[str]]:
        """Return a list of operating conditions for a PyBaMM experiment object.

        These can be passed directly to ``pybamm.Experiment()`` to create an
        experiment for use with PyBaMM.

        Returns:
            The PyBaMM operating conditions.
        """
        step_index_col = BDF.STEP_ID.name
        step_count_col = BDF.STEP_COUNT.name
        only_steps: pl.DataFrame = (
            self.lf.with_row_index()
            .group_by(step_count_col, maintain_order=True)
            .agg(pl.col(step_index_col).first())
            .collect()
        )

        step_description_df = pl.DataFrame(
            {
                step_index_col: self.step_descriptions.get("Step", []),
                "Description": self.step_descriptions.get("Description", []),
            }
        )
        no_step_descriptions = step_description_df.filter(
            pl.col("Description").is_null(),
        )
        missing_steps = no_step_descriptions.select(step_index_col).to_numpy().flatten()
        if len(missing_steps) > 0:
            error_msg = (
                f"Descriptions for steps {str(missing_steps)} are missing."
                f" Unable to create a PyBaMM experiment object. Please "
                f"filter the data to a section with descriptions for all "
                f"steps to create an experiment."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        all_steps_with_descriptions = (
            only_steps.join(
                step_description_df,
                on=step_index_col,
                how="left",
            )
            .select("Description")
            .to_numpy()
            .flatten()
        )
        description_list = []
        for description in all_steps_with_descriptions:
            line = description.split(",")
            for item in line:
                description_list.append(item.strip())
        return description_list


class _CyclingDataMeta(type):
    """Metaclass making ``isinstance(obj, RawData)`` true for any ``CyclingData``.

    The :class:`RawData` alias is a deprecated subclass of :class:`CyclingData`.
    This metaclass keeps ``isinstance(obj, RawData)`` working for *all*
    ``CyclingData`` instances (including filtered slices such as ``Step`` and
    ``Cycle``), preserving the pre-rename behaviour while still warning on direct
    construction.
    """

    def __instancecheck__(cls, instance: object) -> bool:
        """Return ``True`` for any :class:`CyclingData` instance."""
        return isinstance(instance, CyclingData)


class RawData(CyclingData, metaclass=_CyclingDataMeta):
    """Deprecated alias of :class:`CyclingData`.

    ``RawData`` was renamed to :class:`CyclingData`. This subclass keeps existing
    code and notebooks working: it constructs a fully functional ``CyclingData``
    while emitting a :class:`DeprecationWarning`. ``isinstance(obj, RawData)``
    remains ``True`` for any ``CyclingData`` (or subclass) instance.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Warn that ``RawData`` is deprecated, then construct a ``CyclingData``."""
        warnings.warn(
            "RawData has been renamed to CyclingData. Use 'from pyprobe.rawdata "
            "import CyclingData'. The RawData alias will be removed in a future "
            "release.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
