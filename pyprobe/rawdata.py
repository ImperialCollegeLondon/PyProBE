"""A module for the CyclingData class."""

import warnings
from pathlib import Path
from typing import Any, Optional

import polars as pl
from loguru import logger

from pyprobe.columns import BDF, Column
from pyprobe.result import Table
from pyprobe.utils import deprecated

_REQUIRED_BDF_TIME: list[BDF] = [BDF.UNIX_TIME_SECOND, BDF.TEST_TIME_SECOND]
"""Time columns (at least one must be resolvable); Unix Time is preferred."""

_REQUIRED_BDF: list[BDF] = [BDF.CURRENT_AMPERE, BDF.VOLTAGE_VOLT]
"""BDF columns that must be resolvable; CyclingData raises ValueError if not."""

_OPTIONAL_BDF: list[BDF] = [BDF.NET_CAPACITY_AH, BDF.STEP_COUNT, BDF.STEP_ID]
"""BDF columns included when available; warnings emitted on failure."""


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
        metadata: dict[str, Any | None],
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

        # Validate time column (either Unix Time or Test Time must be resolvable)
        if not any(col_set.can_resolve(time_col) for time_col in _REQUIRED_BDF_TIME):
            error_msg = (
                "Required time column: either 'Unix Time / s' or 'Test Time / s' "
                "must be resolvable from available columns."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate other required columns
        for bdf_col in _REQUIRED_BDF:
            if not col_set.can_resolve(bdf_col):
                error_msg = (
                    f"Required BDF column '{bdf_col.name}' is not resolvable "
                    f"from available columns."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Validate optional columns
        for bdf_col in _OPTIONAL_BDF:
            if not col_set.can_resolve(bdf_col):
                logger.warning(
                    f"Optional BDF column '{bdf_col.name}' is not resolvable; some "
                    "features may be unavailable."
                )

    def _net(self, quantity: Column) -> float:
        """Return the signed extent (max − min) of a resolved quantity."""
        from pyprobe.analysis.utils import validate_quantity

        validate_quantity(self, quantity)
        expr = self.columns.resolve(quantity)
        value = self.lf.select((expr.max() - expr.min()).alias("_v")).collect()
        return float(value["_v"][0])

    def _throughput(self, quantity: Column) -> float:
        """Return the cumulative absolute change of a resolved quantity."""
        from pyprobe.analysis.utils import validate_quantity

        validate_quantity(self, quantity)
        expr = self.columns.resolve(quantity)
        value = self.lf.select(expr.diff().abs().sum().alias("_v")).collect()
        return float(value["_v"][0])

    def net_capacity(self) -> float:
        """The signed extent (max − min) of the net capacity, as a ``float``.

        Returns:
            The net capacity passed.
        """
        return self._net(BDF.NET_CAPACITY_AH)

    def net_energy(self) -> float:
        """The signed extent (max − min) of the net energy, as a ``float``.

        Returns:
            The net energy passed.
        """
        return self._net(BDF.NET_ENERGY_WH)

    def capacity_throughput(self) -> float:
        """The cumulative absolute change of net capacity, as a ``float``.

        Returns:
            The cumulative absolute capacity throughput.
        """
        return self._throughput(BDF.NET_CAPACITY_AH)

    def energy_throughput(self) -> float:
        """The cumulative absolute change of net energy, as a ``float``.

        Returns:
            The cumulative absolute energy throughput.
        """
        return self._throughput(BDF.NET_ENERGY_WH)

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
        reason="Use the ``net_capacity`` method instead.",
        version="3.0.0",
    )
    def capacity(self) -> float:
        """Calculate the net capacity passed.

        Returns:
            float: The net capacity passed.
        """
        return self.net_capacity()

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
