"""A module for the Table data object and the Curve continuous data object."""

import os
import re
import warnings
from collections.abc import Callable, Iterable
from functools import wraps
from pathlib import Path
from pprint import pprint
from typing import Any, Literal, Protocol, Union, cast, runtime_checkable

import bdf
import bdf.io
import bdf.spec
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.interpolate import BSpline, PchipInterpolator, PPoly
from scipy.io import savemat

from pyprobe.columns import (
    BDF,
    Column,
    ColumnDict,
    CurveColumns,
    column_factory_from_string,
)
from pyprobe.protocol import Step, leaves
from pyprobe.utils import deprecated, validate_timezone

try:
    import hvplot.polars  # noqa: F401

    hvplot_exists = True
except ImportError:
    hvplot_exists = False


@runtime_checkable
class Quantified(Protocol):
    """The shared contract for every PyProBE data object.

    Any object that carries BDF quantities plus metadata satisfies ``Quantified``.
    A ``Quantified`` object exposes a :attr:`columns` accessor (a
    :class:`~pyprobe.columns.ColumnDict` or one of its variants such as
    :class:`~pyprobe.columns.CurveColumns`), a :attr:`metadata` record, and a
    :attr:`column_definitions` mapping. Both :class:`Table` (discrete) and
    :class:`Curve` (continuous) satisfy it, so a consumer can validate the
    quantities it received against a single surface regardless of storage.
    Every object holds its metadata as a :class:`bdf.Metadata` record.

    This is a structural :class:`typing.Protocol`: any object exposing the three
    members is recognised by ``isinstance(obj, Quantified)`` without explicit
    inheritance.
    """

    metadata: bdf.Metadata

    @property
    def column_definitions(self) -> dict[str, str]:
        """The definition of each of the object's columns, keyed by its label."""
        ...

    @property
    def columns(self) -> ColumnDict:
        """The object's quantities as a ColumnDict (or variant)."""
        ...


def _derived_quantity(y_quantity: Column, x_quantity: Column) -> Column:
    """Return a parser-safe quantity representing ``d(y)/d(x)``.

    Args:
        y_quantity: The y-axis quantity descriptor.
        x_quantity: The x-axis quantity descriptor.

    Returns:
        A :class:`~pyprobe.columns.Column` labelled
        ``d(<y>)_d(<x>) / <derived unit>`` with a unit derived from the ratio
        of the input units.
    """
    y_unit = y_quantity.unit or "1"
    x_unit = x_quantity.unit or "1"

    if y_unit == "1":
        derived_unit = "1" if x_unit == "1" else f"{x_unit}^-1"
    elif x_unit == "1":
        derived_unit = y_unit
    else:
        derived_unit = f"{y_unit} {x_unit}^-1"

    return Column(
        f"d({y_quantity.quantity})_d({x_quantity.quantity})",
        derived_unit,
    )


def _coerce_metadata(metadata: bdf.Metadata | None) -> bdf.Metadata:
    """Return a valid metadata record, defaulting a ``None`` to an empty one.

    Args:
        metadata: A metadata record, or ``None``.

    Returns:
        ``metadata`` unchanged, or an empty :class:`bdf.Metadata` where
        ``metadata`` is ``None``.

    Raises:
        TypeError: If ``metadata`` is neither ``None`` nor a ``bdf.Metadata``.
    """
    if metadata is None:
        return bdf.Metadata()
    if not isinstance(metadata, bdf.Metadata):
        raise TypeError("metadata must be a bdf.Metadata instance.")
    return metadata


class Curve(Quantified, PPoly):
    """A quantity-labelled continuous data object that *is* a scipy ``PPoly``.

    ``Curve`` subclasses :class:`scipy.interpolate.PPoly`, so
    ``isinstance(curve, PPoly)`` is ``True`` and a ``Curve`` drops straight into
    scipy and matplotlib. It additionally carries ``x`` and ``y`` BDF quantities
    (:attr:`x_quantity`, :attr:`y_quantity`) plus :attr:`metadata`, satisfying
    the :class:`Quantified` contract.

    Construct a ``Curve`` from any scipy ``PPoly`` (e.g. a ``PchipInterpolator``)
    or ``BSpline`` via :meth:`from_poly`.

    Attributes:
        x_quantity: The x-axis BDF quantity descriptor.
        y_quantity: The y-axis BDF quantity descriptor.
        metadata: Metadata carried with the curve.
        column_definitions: Definitions of the curve's quantities.
    """

    def __init__(
        self,
        c: NDArray[np.float64],
        x: NDArray[np.float64],
        *,
        x_quantity: str | Column,
        y_quantity: str | Column,
        metadata: bdf.Metadata | None = None,
        extrapolate: bool | None = None,
        axis: int = 0,
    ) -> None:
        """Create a Curve from piecewise-polynomial coefficients.

        Args:
            c: Polynomial coefficients (as for :class:`scipy.interpolate.PPoly`).
            x: Breakpoints (as for :class:`scipy.interpolate.PPoly`).
            x_quantity: The x-axis quantity (string, ``Column``, or ``BDF``).
            y_quantity: The y-axis quantity (string, ``Column``, or ``BDF``).
            metadata: The metadata record for the curve. An empty
                ``bdf.Metadata`` is used where ``None``.
            extrapolate: Passed through to :class:`scipy.interpolate.PPoly`.
            axis: Passed through to :class:`scipy.interpolate.PPoly`.

        Raises:
            TypeError: If ``metadata`` is neither ``None`` nor a ``bdf.Metadata``.
        """
        # Call PPoly.__init__ directly: the Quantified Protocol sits ahead of
        # PPoly in the MRO and would otherwise swallow the constructor call.
        PPoly.__init__(self, c, x, extrapolate=extrapolate, axis=axis)
        self.x_quantity: Column = (
            column_factory_from_string(x_quantity)
            if isinstance(x_quantity, str)
            else x_quantity
        )
        self.y_quantity: Column = (
            column_factory_from_string(y_quantity)
            if isinstance(y_quantity, str)
            else y_quantity
        )
        metadata = _coerce_metadata(metadata)
        self.metadata: bdf.Metadata = metadata.model_copy(deep=True)

    @property
    def column_definitions(self) -> dict[str, str]:
        """The definition of each quantity of the curve, keyed by its label.

        A quantity that the BDF ontology defines takes its definition from the
        ontology. Any other label takes it from
        ``metadata.extras["column_definitions"]``. Where both hold one label,
        the extras win.

        Returns:
            dict[str, str]: The definition of each quantity, keyed by its label.
        """
        return _definitions_of(
            (self.x_quantity.name, self.y_quantity.name),
            self.metadata,
        )

    @classmethod
    def from_poly(
        cls,
        poly: PPoly | BSpline,
        *,
        x_quantity: str | Column,
        y_quantity: str | Column,
        metadata: bdf.Metadata | None = None,
    ) -> "Curve":
        """Build a Curve from a scipy ``PPoly`` or ``BSpline``.

        A ``BSpline`` is not a ``PPoly`` (a sibling representation), so it is
        normalised once at construction via
        :meth:`scipy.interpolate.PPoly.from_spline`. The original construction
        method is recorded in ``metadata.extras["curve_method"]``.

        Args:
            poly: A scipy ``PPoly`` (e.g. ``PchipInterpolator``) or ``BSpline``.
            x_quantity: The x-axis quantity (string, ``Column``, or ``BDF``).
            y_quantity: The y-axis quantity (string, ``Column``, or ``BDF``).
            metadata: The metadata record for the curve. ``curve_method`` is
                added under ``metadata.extras`` if not already present. An
                empty ``bdf.Metadata`` is used where ``None``.

        Returns:
            A new :class:`Curve` wrapping the (normalised) piecewise polynomial.

        Raises:
            TypeError: If ``poly`` is neither a ``PPoly`` nor a ``BSpline``, or
                if ``metadata`` is neither ``None`` nor a ``bdf.Metadata``.
        """
        if isinstance(poly, BSpline):
            method = "smoothing_spline"
            poly = PPoly.from_spline(poly)
        elif isinstance(poly, PPoly):
            method = type(poly).__name__
        else:
            raise TypeError(
                "Curve.from_poly expects a scipy PPoly or BSpline, "
                f"got {type(poly).__name__}."
            )
        metadata = _coerce_metadata(metadata)
        extras = dict(metadata.extras or {})
        extras.setdefault("curve_method", method)
        metadata = metadata.model_copy(update={"extras": extras})
        return cls(
            poly.c,
            poly.x,
            x_quantity=x_quantity,
            y_quantity=y_quantity,
            metadata=metadata,
            extrapolate=poly.extrapolate,
            axis=poly.axis,
        )

    @property
    def columns(self) -> CurveColumns:
        """The curve's quantities as a :class:`~pyprobe.columns.CurveColumns`.

        Returns:
            A :class:`~pyprobe.columns.CurveColumns` exposing ``.x`` and ``.y``
            axis roles that resolve to the curve's quantities.
        """
        return CurveColumns(self.x_quantity, self.y_quantity)

    def derivative(self, n: int = 1) -> "Curve":
        """Return the ``n``-th derivative as a new ``Curve``.

        The returned curve carries the derived ``d(y)/d(x)`` quantity, while its
        x quantity and metadata are preserved.

        Args:
            n: The order of derivative to compute. Default is 1.

        Returns:
            A new :class:`Curve` representing the derivative.
        """
        d = PPoly.derivative(self, n)
        y_quantity = self.y_quantity
        for _ in range(n):
            y_quantity = _derived_quantity(y_quantity, self.x_quantity)
        return Curve(
            d.c,
            d.x,
            x_quantity=self.x_quantity,
            y_quantity=y_quantity,
            metadata=self.metadata,
            extrapolate=d.extrapolate,
            axis=d.axis,
        )

    def to_table(self, x: NDArray[np.float64]) -> "Table":
        """Sample the curve onto a grid and return a discrete :class:`Table`.

        Args:
            x: The x grid to evaluate the curve on.

        Returns:
            A :class:`Table` with the x quantity and the sampled y quantity.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(self(x_arr), dtype=np.float64)
        frame = pl.DataFrame({self.x_quantity.name: x_arr, self.y_quantity.name: y_arr})
        return Table(
            lf=frame.lazy(),
            metadata=self.metadata,
        )


def _node_repeats(node: "Step") -> bool:
    """Report whether a protocol node is a group that repeats.

    Args:
        node: The protocol node to read.

    Returns:
        bool: True where the node is a group that carries a count.
    """
    return node.mode == "group" and node.count is not None


def _leaf_repeats(node: "Step", *, repeats_above: bool) -> list[bool]:
    """Report, for each leaf below a node, whether a group above it repeats.

    Args:
        node: The protocol node to walk.
        repeats_above: Whether a group above the node repeats.

    Returns:
        list[bool]: One flag per leaf below the node, in tree order.
    """
    if not node.steps:
        return [repeats_above]
    found: list[bool] = []
    for child in node.steps:
        found.extend(
            _leaf_repeats(child, repeats_above=repeats_above or _node_repeats(child)),
        )
    return found


_DEFINITIONS_KEY = "column_definitions"
"""The key of the extras mapping that holds the stored column definitions."""


def _definitions_of(
    labels: Iterable[str],
    metadata: bdf.Metadata,
) -> dict[str, str]:
    """Return the definition of each label, keyed by the label.

    A label that the BDF ontology defines takes its definition from the
    ontology. Any label takes it from ``metadata.extras["column_definitions"]``
    as well, and that mapping wins where both hold one label.

    Args:
        labels: The column labels to define, in the order they are read.
        metadata: The record that holds the stored definitions.

    Returns:
        dict[str, str]: The definition of each label that either source
            defines.
    """
    definitions: dict[str, str] = {}
    for label in labels:
        definition = _ontology_definition(label)
        if definition is not None:
            definitions[label] = definition
    extras = metadata.extras or {}
    definitions.update(dict(extras.get(_DEFINITIONS_KEY, {})))
    return definitions


def _ontology_definition(label: str) -> str | None:
    """Return the BDF ontology definition of a column label.

    Args:
        label: The column label to look up, such as "Current / A".

    Returns:
        str | None: The definition field of the matching
            :class:`bdf.spec.Quantity`, or ``None`` where the ontology defines
            no quantity for the label.
    """
    match = bdf.spec.COLUMN_ONTOLOGY.quantity_from_label(label)
    if match is None or not match[0].definition:
        return None
    return match[0].definition


_STAT_SUITE: dict[str, Callable[[pl.Expr], pl.Expr]] = {
    "delta": lambda e: e.last() - e.first(),
    "range": lambda e: e.max() - e.min(),
    "mean": lambda e: e.mean(),
    "max": lambda e: e.max(),
    "min": lambda e: e.min(),
    "first": lambda e: e.first(),
    "last": lambda e: e.last(),
}
"""Shared aggregation registry used by reduction methods and :meth:`~Table.summary`."""


class Table:
    """A class for holding any tabular data in PyProBE.

    A Table object is the base type for every discrete data object in PyProBE. It
    composes a polars :class:`~polars.LazyFrame` and carries metadata and column
    definitions, satisfying the :class:`Quantified` contract. This class includes
    all of the main methods for returning and describing tabular data in PyProBE.

    Continuous fits are produced with :meth:`to_curve`; discrete operations are
    exposed as the flat methods :meth:`savgol`, :meth:`downsample`, and
    :meth:`gradient`.

    .. note::
        ``Result`` is a deprecated alias of ``Table``. Existing code using
        ``Result`` keeps working but emits a deprecation warning on construction.

    Key attributes for returning data:
        - :attr:`data`: The data as a Polars DataFrame.
        - :meth:`get`: Get a column from the data as a NumPy array.

    Key attributes for describing the data:
        - :attr:`metadata`: A :class:`bdf.Metadata` record describing the cell
          and data source. Free-form keys live under ``metadata.extras``.
        - :attr:`column_definitions`: A read-only mapping of a column label to
          its definition, read from the BDF ontology and from the definitions
          that :meth:`define_column` writes.
        - :meth:`print_definitions`: Print the column definitions.
        - :attr:`columns`: A :class:`~pyprobe.columns.ColumnDict` object providing
          column name access (via ``.names``) and BDF-aware resolution (via
          ``.resolve()`` and ``.can_resolve()``).
    """

    _protocol_node: "Step | None" = None
    """The test protocol node that this object was filtered to.

    A ``Procedure`` holds a synthetic root node over the whole method. A
    structural filter reduces this node, and a condition filter keeps it.
    """

    def __init__(
        self,
        lf: pl.LazyFrame | pl.DataFrame | str,
        metadata: bdf.Metadata | None = None,
        _path: Path | None = None,
    ) -> None:
        """Create a Table with explicit constructor validation.

        Args:
            lf: A LazyFrame, DataFrame, or a path to a parquet file.
            metadata: The metadata record for the result. An empty
                ``bdf.Metadata`` is used where ``None``.
            _path: Optional path to the backing Parquet file.

        Raises:
            ValueError: If constructor inputs do not match expected types.
            TypeError: If ``metadata`` is neither ``None`` nor a ``bdf.Metadata``.
        """
        if isinstance(lf, str):
            lf = pl.scan_parquet(lf)
        if not isinstance(lf, pl.LazyFrame):
            if isinstance(lf, pl.DataFrame):
                lf = lf.lazy()
            elif isinstance(lf, str):
                lf = pl.scan_parquet(lf)
            else:
                raise ValueError(
                    "lf must be a polars DataFrame, LazyFrame, or a parquet file path."
                )
        metadata = _coerce_metadata(metadata)

        self.lf: pl.LazyFrame = lf
        self.metadata = metadata.model_copy(deep=True)
        self._path: Path | None = _path

    @property
    def column_definitions(self) -> dict[str, str]:
        """The definition of each column, keyed by its label.

        A column that the BDF ontology defines takes its definition from the
        ontology. A column that the ontology does not define takes it from
        ``metadata.extras["column_definitions"]``, which :meth:`define_column`
        writes. Where both hold one label, the extras win.

        Returns:
            dict[str, str]: The definition of each column, keyed by its label.
        """
        return _definitions_of(self.lf.collect_schema().names(), self.metadata)

    def _stored_definitions(self) -> dict[str, str]:
        """Return the definitions that the metadata record holds.

        Returns:
            dict[str, str]: A copy of ``extras["column_definitions"]``, empty
                where the record holds none.
        """
        extras = self.metadata.extras or {}
        return dict(extras.get(_DEFINITIONS_KEY, {}))

    def _write_definitions(self, definitions: dict[str, str]) -> None:
        """Write the definitions into the metadata record.

        A definition of a column of the frame is not stored where the BDF
        ontology already gives that exact text, because the property reads it
        from the ontology. A definition of any other label is stored, because
        the property derives an ontology definition for a column of the frame
        alone.

        Args:
            definitions: The definitions to store, keyed by column label.
        """
        labels = set(self.lf.collect_schema().names())
        stored = {
            label: definition
            for label, definition in definitions.items()
            if label not in labels or definition != _ontology_definition(label)
        }
        extras = dict(self.metadata.extras or {})
        if stored == dict(extras.get(_DEFINITIONS_KEY, {})):
            return
        if stored:
            extras[_DEFINITIONS_KEY] = stored
        else:
            extras.pop(_DEFINITIONS_KEY, None)
        self.metadata.extras = extras

    def _protocol_nodes(self) -> list["Step"]:
        """Return the protocol nodes at the level of this object.

        An object that holds a node reports that node alone. An object that
        holds none reports the whole method of its metadata record.

        Returns:
            list[Step]: The nodes at the level of this object, in tree order.
        """
        if self._protocol_node is not None:
            return [self._protocol_node]
        protocol = self.metadata.battinfo_test_protocol
        if protocol is None:
            return []
        return list(protocol.method or [])

    def _protocol_children(self) -> list["Step"]:
        """Return the protocol nodes one level below this object.

        An object that holds no node stands over the whole method, so its
        children are the top level nodes of the method.

        Returns:
            list[Step]: The nodes below this object, in tree order.
        """
        if self._protocol_node is not None:
            return list(self._protocol_node.steps or [])
        protocol = self.metadata.battinfo_test_protocol
        if protocol is None:
            return []
        return list(protocol.method or [])

    def _protocol_leaves(self) -> list["Step"]:
        """Return the protocol leaves at the level of this object.

        Returns:
            list[Step]: The leaves below the nodes of this object, in tree
                order.
        """
        found: list[Step] = []
        for node in self._protocol_nodes():
            found.extend(leaves(node))
        return found

    def _protocol_leaf_repeats(self, *, include_nodes: bool = True) -> list[bool]:
        """Report, for each leaf below this object, whether a group repeats above it.

        A repeat above a leaf gives many step events for that one leaf, so a
        step index does not address the leaf at the same position.

        Args:
            include_nodes: When ``True``, a count on a node of this object
                counts as a repeat above every leaf below that node.

        Returns:
            list[bool]: One flag per leaf below this object, in tree order.
        """
        found: list[bool] = []
        for node in self._protocol_nodes():
            found.extend(
                _leaf_repeats(
                    node,
                    repeats_above=include_nodes and _node_repeats(node),
                ),
            )
        return found

    def collect(self) -> pl.DataFrame:
        """Collect the lazy dataframe into a polars DataFrame.

        Use this method to resolve the lazy computations in the Result object. This can
        improve performance if you are reading a large amount of data from disk, and
        will be performing multiple calls to access the data.

        Returns:
            pl.DataFrame: The collected dataframe.
        """
        lf = self.lf.collect()
        self.lf = lf.lazy()
        return lf

    def to_curve(
        self,
        y: str | Column,
        x: str | Column = "Test Time / s",
        fit: Callable[..., PPoly | BSpline] = PchipInterpolator,
        **kwargs: Any,
    ) -> "Curve":
        """Fit a continuous :class:`Curve` to a column of this table.

        Resolves ``x`` and ``y`` to numpy arrays, fits them with ``fit``, and
        wraps the result as a quantity-labelled :class:`Curve`. ``fit`` may be
        any scipy 1-D interpolator/smoother or a user callable sharing the
        informal protocol ``callable(x, y, **kwargs) -> PPoly | BSpline``.

        Args:
            y: The y-axis column (name, ``Column``, or ``BDF``).
            x: The x-axis column (name, ``Column``, or ``BDF``). Defaults to
                ``"Test Time / s"``.
            fit: A callable ``(x, y, **kwargs) -> PPoly | BSpline``. Defaults to
                :class:`scipy.interpolate.PchipInterpolator`. Any scipy 1-D
                interpolator (e.g. ``CubicSpline``, ``Akima1DInterpolator``) or
                smoother (e.g. ``make_smoothing_spline``) is accepted.
            **kwargs: Extra keyword arguments forwarded to ``fit`` (e.g.
                ``bc_type``, ``lam``, ``k``).

        Returns:
            A :class:`Curve` labelled with the ``x`` and ``y`` quantities.

        Raises:
            ColumnResolutionError: If ``x`` or ``y`` cannot be resolved.
            TypeError: If ``fit`` returns neither a ``PPoly`` nor a ``BSpline``.
        """
        from pyprobe.analysis.utils import get_columns

        x_data, y_data = get_columns(self, x, y)
        obj = fit(x_data, y_data, **kwargs)
        return Curve.from_poly(
            obj,
            x_quantity=x,
            y_quantity=y,
            metadata=self.metadata,
        )

    def savgol(
        self,
        target_column: str,
        window_length: int,
        polyorder: int,
        derivative: int = 0,
    ) -> "Table":
        """Savitzky-Golay denoise ``target_column``, returning a new ``Table``.

        Args:
            target_column: The name of the column to smooth.
            window_length: The length of the filter window (positive odd int).
            polyorder: The order of the polynomial used to fit the samples.
            derivative: The order of the derivative to compute. Default is 0.

        Returns:
            A new :class:`Table` with ``target_column`` smoothed.

        Raises:
            ColumnResolutionError: If ``target_column`` cannot be resolved.
        """
        from pyprobe.analysis.smoothing import savgol_smoothing

        return savgol_smoothing(
            self,
            target_column,
            window_length=window_length,
            polyorder=polyorder,
            derivative=derivative,
        )

    def downsample(
        self,
        target_column: str,
        sampling_interval: float,
        monotonic: bool = True,
        occurrence: Literal["first", "last", "middle"] = "first",
    ) -> "Table":
        """Downsample on ``target_column`` to ``sampling_interval``.

        Args:
            target_column: The column to downsample on.
            sampling_interval: The desired minimum interval between points.
            monotonic: If True, ``target_column`` is assumed monotonic. Default
                is True.
            occurrence: The occurrence to take within each bin (only used when
                ``monotonic``). Default is ``"first"``.

        Returns:
            A new :class:`Table` containing the downsampled data.

        Raises:
            ColumnResolutionError: If ``target_column`` cannot be resolved.
        """
        from pyprobe.analysis.smoothing import downsample

        return downsample(
            self,
            target_column,
            sampling_interval=sampling_interval,
            monotonic=monotonic,
            occurrence=occurrence,
        )

    def gradient(self, y: str, x: str) -> "Table":
        """Finite-difference derivative of ``y`` with respect to ``x``.

        Args:
            y: The name of the y variable.
            x: The name of the x variable.

        Returns:
            A new :class:`Table` with the ``x``, ``y`` and gradient columns.

        Raises:
            ColumnResolutionError: If ``x`` or ``y`` cannot be resolved.
        """
        from pyprobe.analysis.differentiation import gradient

        return gradient(self, x=x, y=y)

    def _reducible_columns(self) -> list[str]:
        """Return all numeric column names from the frame schema, excluding Step ID.

        Float and integer columns are included; ``Step ID`` and non-numeric columns
        are excluded.

        Returns:
            List of column name strings eligible for reduction.
        """
        schema = self.lf.collect_schema()
        excluded = {BDF.STEP_ID.name}
        return [
            name
            for name, dtype in schema.items()
            if name not in excluded and (dtype.is_float() or dtype.is_integer())
        ]

    def _aggregate(
        self,
        agg: Callable[[pl.Expr], pl.Expr],
        *columns: str | Column,
        by: str | None = None,
    ) -> "Table":
        """Aggregate columns using ``agg``, optionally grouped by a column.

        Pre-computes each named (or default reducible) column via
        ``with_columns`` before aggregating, so windowed recipes such as
        ``cumsum``/``diff`` are evaluated over the full slice rather than
        per group.

        Args:
            agg: Aggregation function mapping a ``pl.Expr`` to a ``pl.Expr``
                (e.g. ``lambda e: e.last() - e.first()``).
            *columns: Column names to aggregate. Defaults to all numeric
                columns from :meth:`_reducible_columns` when omitted.
            by: Column name to group by. If ``None``, a single-row result is
                returned via ``select``.

        Returns:
            A new :class:`Table` carrying the original metadata.

        Raises:
            ColumnResolutionError: If any column in ``columns`` cannot be
                resolved.
        """
        cols: list[str | Column] = (
            list(columns)
            if columns
            else cast("list[str | Column]", self._reducible_columns())
        )
        if by is not None and by in cols:
            cols.remove(by)
        pre = [self.columns.resolve(c) for c in cols]
        lf = self.lf.with_columns(pre)
        exprs = [agg(pl.col(str(c))).alias(str(c)) for c in cols]
        if by is not None:
            lf = lf.group_by(by, maintain_order=True).agg(exprs)
        else:
            lf = lf.select(exprs)
        return Table(lf, metadata=self.metadata)

    def delta(self, *columns: str | Column) -> "Table":
        """Collapse the frame to a single row by computing ``last − first``.

        Args:
            *columns: Column names to reduce. Defaults to all numeric
                columns from :meth:`_reducible_columns` when omitted.

        Returns:
            A single-row :class:`Table` with the signed delta for each column.

        Raises:
            ColumnResolutionError: If any named column cannot be resolved.

        Examples:
            >>> import polars as pl
            >>> from pyprobe import Table
            >>> t = Table(lf=pl.LazyFrame({"Net Capacity / Ah": [0.0, 0.5, 1.0]}))
            >>> t.delta().get("Net Capacity / Ah")
            array([1.])
        """
        return self._aggregate(_STAT_SUITE["delta"], *columns)

    def range(self, *columns: str | Column) -> "Table":
        """Collapse the frame to a single row by computing ``max - min``.

        Args:
            *columns: Column names to reduce. Defaults to all numeric
                columns from :meth:`_reducible_columns` when omitted.

        Returns:
            A single-row :class:`Table` with the non-negative range for each
            column.

        Raises:
            ColumnResolutionError: If any named column cannot be resolved.
        """
        return self._aggregate(_STAT_SUITE["range"], *columns)

    def mean(self, *columns: str | Column) -> "Table":
        """Collapse the frame to a single row by computing the column-wise mean.

        Args:
            *columns: Column names to reduce. Defaults to all numeric
                columns from :meth:`_reducible_columns` when omitted.

        Returns:
            A single-row :class:`Table` with the mean for each column.

        Raises:
            ColumnResolutionError: If any named column cannot be resolved.
        """
        return self._aggregate(_STAT_SUITE["mean"], *columns)

    def maximum(self, *columns: str | Column) -> "Table":
        """Collapse the frame to a single row by computing the column-wise maximum.

        Args:
            *columns: Column names to reduce. Defaults to all numeric
                columns from :meth:`_reducible_columns` when omitted.

        Returns:
            A single-row :class:`Table` with the maximum for each column.

        Raises:
            ColumnResolutionError: If any named column cannot be resolved.
        """
        return self._aggregate(_STAT_SUITE["max"], *columns)

    def minimum(self, *columns: str | Column) -> "Table":
        """Collapse the frame to a single row by computing the column-wise minimum.

        Args:
            *columns: Column names to reduce. Defaults to all numeric
                columns from :meth:`_reducible_columns` when omitted.

        Returns:
            A single-row :class:`Table` with the minimum for each column.

        Raises:
            ColumnResolutionError: If any named column cannot be resolved.
        """
        return self._aggregate(_STAT_SUITE["min"], *columns)

    def first(self, *columns: str | Column) -> "Table":
        """Collapse the frame to a single row by taking the first value.

        Args:
            *columns: Column names to reduce. Defaults to all numeric
                columns from :meth:`_reducible_columns` when omitted.

        Returns:
            A single-row :class:`Table` with the first value for each column.

        Raises:
            ColumnResolutionError: If any named column cannot be resolved.
        """
        return self._aggregate(_STAT_SUITE["first"], *columns)

    def last(self, *columns: str | Column) -> "Table":
        """Collapse the frame to a single row by taking the last value.

        Args:
            *columns: Column names to reduce. Defaults to all numeric
                columns from :meth:`_reducible_columns` when omitted.

        Returns:
            A single-row :class:`Table` with the last value for each column.

        Raises:
            ColumnResolutionError: If any named column cannot be resolved.
        """
        return self._aggregate(_STAT_SUITE["last"], *columns)

    def item(self, column: str | None = None) -> float:
        """Return a scalar from a single-row table.

        This method triggers :meth:`collect`.

        Args:
            column: Exact schema column name to extract. When omitted, the
                table must contain exactly one column.

        Returns:
            The selected scalar value as a ``float``.

        Raises:
            ValueError: If the table does not have exactly one row, or if
                ``column`` is omitted and the table has multiple columns.
            KeyError: If ``column`` is provided but does not exist.
            TypeError: If the selected value is not numeric.
        """
        data = self.collect()
        if data.height != 1:
            raise ValueError(
                f"item() requires exactly one row, found {data.height} rows."
            )

        if column is None:
            if data.width != 1:
                columns = ", ".join(data.columns)
                raise ValueError(
                    "item() requires exactly one column when column is omitted. "
                    f"Available columns: {columns}"
                )
            column = data.columns[0]
        elif column not in data.columns:
            raise KeyError(column)

        value = data[column][0]
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError(
                f"item() requires a numeric value, got {type(value).__name__}."
            )
        return float(value)

    def summary(
        self,
        *columns: str | Column,
        by: str = BDF.STEP_COUNT.name,
    ) -> "Table":
        """Grouped multi-statistic reduction of the frame.

        Applies the full aggregation suite (``delta``, ``range``, ``mean``,
        ``max``, ``min``, ``first``, ``last``) to each column, grouped by ``by``.
        Output columns are named ``"{stat} {column}"``, e.g.
        ``"delta Net Capacity / Ah"``. When ``Step ID`` is present in the
        frame it is retained as a per-group descriptor (``first()``), not
        reduced.

        Pre-computes named columns via ``with_columns`` before grouping so
        windowed recipes (e.g. ``Cumulative Capacity / Ah``) are materialised
        over the full slice.

        Args:
            *columns: Column names to summarise. Defaults to all numeric
                columns from :meth:`_reducible_columns` when omitted.
            by: Column name to group by. Defaults to
                ``"Step Count / 1"`` (:attr:`~pyprobe.columns.BDF.STEP_COUNT`).

        Returns:
            A :class:`Table` with one row per ``by`` group and one output
            column per (statistic, input column) combination.

        Raises:
            ColumnResolutionError: If any named column cannot be resolved.
        """
        cols: list[str | Column] = (
            list(columns)
            if columns
            else cast("list[str | Column]", self._reducible_columns())
        )
        if by in cols:
            cols.remove(by)
        pre = [self.columns.resolve(c) for c in cols]
        lf = self.lf.with_columns(pre)
        agg_exprs: list[pl.Expr] = [
            fn(pl.col(str(c))).alias(f"{stat} {c}")
            for c in cols
            for stat, fn in _STAT_SUITE.items()
        ]
        step_id_name = BDF.STEP_ID.name
        if step_id_name in self.lf.collect_schema().names():
            agg_exprs.append(pl.col(step_id_name).first())
        lf = lf.group_by(by, maintain_order=True).agg(agg_exprs)
        return Table(lf, metadata=self.metadata)

    @property
    def columns(self) -> ColumnDict:
        """The columns in the data as a ColumnDict.

        Returns a :class:`~pyprobe.columns.ColumnDict` object that provides
        both simple column name access and BDF-aware resolution:

        - :attr:`~pyprobe.columns.ColumnDict.names`: tuple of column name strings.
        - :attr:`~pyprobe.columns.ColumnDict.quantities`: tuple of quantity strings.
        - :meth:`~pyprobe.columns.ColumnDict.resolve`: resolve a column by name
          or quantity, with optional unit conversion.
        - :meth:`~pyprobe.columns.ColumnDict.can_resolve`: check if a column
          or BDF quantity is available.

        Returns:
            ColumnDict: A column introspection and resolution object.

        Examples:
            >>> import polars as pl
            >>> from pyprobe import Table
            >>> r = Table(lf=pl.LazyFrame({"Current / A": [1.0]}))
            >>> r.columns.names
            ('Current / A',)
            >>> r.columns.quantities
            ('Current',)
        """
        return ColumnDict(self.lf.collect_schema().names())

    @property
    def info(self) -> dict[str, Any | None]:
        """The extras mapping of the metadata record, without the definitions.

        The column definitions live under a reserved key of the extras, and
        they are a mapping rather than a value. This property drops that key,
        so every value it returns is a value a caller can write to a column or
        to a MAT file.

        Returns:
            dict: The extras mapping without its column definitions, or an
                empty mapping where the record holds no extras.
        """
        extras = self.metadata.extras or {}
        return {key: value for key, value in extras.items() if key != _DEFINITIONS_KEY}

    @property
    def df(self) -> pl.DataFrame:
        """Return the data as a Polars DataFrame.

        Returns:
            pl.DataFrame: The data as a Polars DataFrame.
        """
        return self.collect()

    @df.setter
    def df(self, dataframe: pl.DataFrame) -> None:
        """Set the data as a Polars DataFrame.

        Args:
            dataframe (pl.DataFrame): The data as a Polars DataFrame.
        """
        self.lf = dataframe.lazy()

    @property
    def data(self) -> pl.DataFrame:
        """Return the data as a polars DataFrame.

        Returns:
            pl.DataFrame: The data as a polars DataFrame.

        Raises:
            ValueError: If no data exists for this filter.
        """
        df = self.collect()
        if df.is_empty():
            raise ValueError("No data exists for this filter.")
        return df

    @wraps(pd.DataFrame.plot)
    def plot(self, *args: Any, **kwargs: Any) -> Axes | NDArray[Axes]:
        """Wrapper for plotting using the pandas library."""
        data_to_plot = self.get_plotting_data(args, kwargs)
        return data_to_plot.to_pandas().plot(*args, **kwargs)

    plot.__doc__ = """Plot the data using the pandas plot method.

    Call this method on a Result object in the same way you would call the pandas plot
    method on a DataFrame. For example:

    .. code-block:: python

        result.plot(x="Test Time / s", y="Current / A")

    Refer to the `pandas documentation \
    <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html>`_
    for detailed information and examples.
    """

    if hvplot_exists is True:

        @wraps(hvplot.hvPlot)
        def hvplot(self, *args: Any, **kwargs: Any) -> Any:
            """Wrapper for plotting using the hvplot library."""
            data_to_plot = self.get_plotting_data(args, kwargs)
            return data_to_plot.hvplot(*args, **kwargs)

    else:

        def hvplot(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore
            """Wrapper for plotting using the hvplot library."""
            raise ImportError(
                "Optional dependency hvplot is not installed. Please install it via "
                "'pip install hvplot' or by installing PyProBE with hvplot as an "
                "optional dependency: pip install 'PyProBE-Data[hvplot]'.",
            )

    hvplot.__doc__ = """HvPlot is a library for creating fast and interactive plots.
        This method requires the hvplot library to be installed as an optional
        dependency. You can install it with PyProBE by running
        :code:`pip install 'PyProBE-Data[hvplot]'`, or install it seperately with
        :code:`pip install hvplot`.

        The default backend is bokeh, which can be changed by setting the backend
        with :code:`hvplot.extension('matplotlib')` or
        :code:`hvplot.extension('plotly')`.

        Example usage:

        .. code-block:: python

            result.hvplot(x="Test Time / s", y="Current / A", kind="scatter")

        This method is not compatible with the inline syntax for hvplot:
        :code:`result.hvplot.scatter(...)`.

        See the `hvplot documentation
        <https://hvplot.holoviz.org/user_guide/Plotting.html>`_ for information
        and examples.
        """

    def __getitem__(self, *column_names: str | Column) -> "Table":
        """Return a new result object with the specified columns.

        Args:
            *column_names (str | Column):
                The columns to include in the new result object.

        Returns:
            Table: A new Table object with the specified columns.
        """
        col_set = self.columns
        exprs = [col_set.resolve(name) for name in column_names]
        return Table(
            lf=self.lf.select(*exprs),
            metadata=self.metadata,
        )

    def get(
        self,
        *column_names: str | Column,
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], ...]:
        """Return one or more columns of the data as separate 1D numpy arrays.

        Args:
            column_names (str | Column): The column name(s) to return.

        Returns:
            Union[NDArray[np.float64], Tuple[NDArray[np.float64], ...]]:
                The column(s) as numpy array(s).

        Raises:
            ValueError: If no column names are provided.
            ValueError: If a column name is not in the data.
        """
        if len(column_names) == 0:
            error_msg = "At least one column name must be provided."
            logger.error(error_msg)
            raise ValueError(error_msg)
        col_set = self.columns
        exprs = [col_set.resolve(name) for name in column_names]
        array = self.lf.select(*exprs).collect().to_numpy()
        if len(column_names) == 1:
            return array.T[0]
        else:
            return tuple(array.T)

    @deprecated(
        reason="The get_only method is deprecated. Use the get method instead.",
        version="1.2.0",
    )
    def get_only(self, column_name: str | Column) -> NDArray[np.float64]:
        """Return a single column of the data as a numpy array.

        Args:
            column_name (str | Column): The column name to return.

        Returns:
            NDArray[np.float64]: The column as a numpy array.

        Raises:
            ValueError: If the column name is not in the data.
            ValueError: If no column name is provided.
        """
        column = self.get(column_name)
        if not isinstance(column, np.ndarray):
            error_msg = "More than one column returned."
            logger.error(error_msg)
            raise ValueError(error_msg)
        return column

    def get_plotting_data(
        self,
        args: tuple[Any, ...],
        kwargs: dict[Any, Any],
    ) -> pl.DataFrame:
        """Extract and resolve columns for plotting from function arguments.

        This method analyzes the arguments passed to a plotting function and
        retrieves the used columns as a DataFrame. It extracts column names from
        positional and keyword arguments, resolves them using the ColumnDict
        (which handles unit conversions and BDF-aware resolution), and returns
        a collected DataFrame suitable for passing to plotting libraries.

        Args:
            args: Positional arguments from the plotting function.
            kwargs: Keyword arguments from the plotting function.

        Returns:
            pl.DataFrame: A collected DataFrame containing the requested columns.

        Raises:
            ValueError: If none of the requested columns are present in the data.

        Examples:
            >>> result = Table(lf=pl.LazyFrame({"Current / A": [1.0, 2.0]}))
            >>> df = result.get_plotting_data(["Current / mA"], {})
            >>> df.shape
            (2, 1)
        """
        kwargs_values = [
            v
            for k, v in kwargs.items()
            if isinstance(v, (str, Column)) and k != "label"
        ]
        args_values = [v for v in args if isinstance(v, (str, Column))]
        all_args = set(kwargs_values + args_values)
        relevant_columns = []
        col_set = self.columns

        for arg in all_args:
            if col_set.can_resolve(arg):
                relevant_columns.append(arg)

        if len(relevant_columns) == 0:
            raise ValueError(
                f"None of the columns in {all_args} are present in the Result object.",
            )

        # Resolve columns using ColumnDict to handle unit conversions
        exprs = [col_set.resolve(col) for col in relevant_columns]
        return self.lf.select(*exprs).collect()

    def define_column(self, column_name: str, definition: str) -> None:
        """Define a new column when it is added to the dataframe.

        Args:
            column_name (str): The name of the column.
            definition (str): The definition of the quantity stored in the column
        """
        definitions = self._stored_definitions()
        definitions[column_name] = definition
        self._write_definitions(definitions)

    def print_definitions(self) -> None:
        """Print the definitions of the columns stored in this result object."""
        pprint(self.column_definitions)  # noqa: T203

    def clean_copy(
        self,
        dataframe: pl.DataFrame | pl.LazyFrame | None = None,
        column_definitions: dict[str, str] | None = None,
    ) -> "Table":
        """Create a copy of the Table object with info dictionary but without data.

        Args:
            dataframe (Optional[Union[pl.DataFrame, pl.LazyFrame]):
                The data to include in the new Table object.
            column_definitions (Optional[dict[str, str]]):
                The definitions of the columns in the new Table object.

        Returns:
            Table: A new Table object with the specified data.
        """
        if dataframe is None:
            dataframe = pl.LazyFrame({})
        elif isinstance(dataframe, pl.DataFrame):
            dataframe = dataframe.lazy()
        extras = dict(self.metadata.extras or {})
        extras.pop(_DEFINITIONS_KEY, None)
        copy = Table(
            lf=dataframe,
            metadata=self.metadata.model_copy(update={"extras": extras}),
        )
        if column_definitions is not None:
            copy._write_definitions(dict(column_definitions))
        return copy

    @staticmethod
    def _verify_compatible_frames(
        base_frame: pl.DataFrame | pl.LazyFrame,
        frames: list[pl.DataFrame | pl.LazyFrame],
        mode: Literal["match 1", "collect all"] = "collect all",
    ) -> tuple[pl.DataFrame | pl.LazyFrame, list[pl.DataFrame | pl.LazyFrame]]:
        """Verify that frames are compatible and return them as DataFrames.

        Args:
            base_frame (pl.DataFrame | pl.LazyFrame): The first frame to verify.
            frames (List[pl.DataFrame | pl.LazyFrame]): The list of frames to verify.
            mode:
                The mode to use for verification. Either 'match 1' or 'collect all'.
                'match 1' will convert the frames to match the base frame. 'collect all'
                will collect all frames to DataFrames.

        Returns:
            Tuple[pl.DataFrame | pl.LazyFrame, List[pl.DataFrame | pl.LazyFrame]]:
                The first frame and the list of verified frames as DataFrames.
        """
        verified_frames = []
        for frame in frames:
            if isinstance(base_frame, pl.LazyFrame) and isinstance(frame, pl.DataFrame):
                if mode == "match 1":
                    frame = frame.lazy()
                elif mode == "collect all":
                    base_frame = base_frame.collect()
            elif isinstance(base_frame, pl.DataFrame) and isinstance(
                frame,
                pl.LazyFrame,
            ):
                frame = frame.collect()
            verified_frames.append(frame)

        return base_frame, verified_frames

    def load_external_file(self, filepath: str) -> pl.LazyFrame:
        """Load an external file into a LazyFrame.

        Supported file types are CSV, Parquet, and Excel. For maximum performance,
        consider using Parquet files. If you have an Excel file, consider converting
        it to CSV before loading.

        Args:
            filepath (str): The path to the external file.
        """
        file = os.path.basename(filepath)
        file_ext = os.path.splitext(file)[1]
        match file_ext:
            case ".csv":
                return pl.scan_csv(filepath)
            case ".parquet":
                return pl.scan_parquet(filepath)
            case ".xlsx":
                warnings.warn("Excel reading is slow. Consider converting to CSV.")
                return pl.read_excel(filepath).lazy()
            case _:
                error_msg = f"Unsupported file type: {file_ext}"
                logger.error(error_msg)
                raise ValueError(error_msg)

    def add_data(
        self,
        new_data: pl.DataFrame | pl.LazyFrame | str,
        time_column_name: str,
        column_map: dict[str, str] | None = None,
        datetime_format: str | None = None,
        timezone: str = "UTC",
        align_on: tuple[str, str] | None = None,
        join_strategy: Literal[
            "keep_existing", "keep_new", "keep_both"
        ] = "keep_existing",
        fill_strategy: Literal["interpolate", "forward_fill", "backward_fill"]
        | None = "interpolate",
    ) -> None:
        """Add new data columns to the result object using Unix Time as the join key.

        The data must be time series data with a time column. The new data is joined to
        the base dataframe on the "Unix Time / s" column. Choose which dates to keep
        with the join strategy, and how to fill missing values with the fill strategy.

        Args:
            new_data:
                The new data to add to the result object. Can be a DataFrame, LazyFrame,
                or a path to a file (CSV, Parquet, Excel).
            time_column_name:
                The name of the column in the new data containing the time. Can be a
                datetime column (which will be auto-converted to UTC unix seconds), a
                numeric column (assumed to be UTC unix seconds), or a string column
                (which will be parsed then converted).
            column_map:
                Mapping from output names to source column names:
                {output_name: source_name}.
                Only the columns in this dict will be imported. If None, all columns
                (except time_column_name) will be imported. Output names do not need to
                follow "Quantity / unit" format.
            datetime_format:
                The format string for parsing the time column if it is a string.
                Defaults to None (auto-detect).
            timezone:
                The timezone of the new data's time column, as an IANA string
                (e.g. ``"UTC"``, ``"Europe/Berlin"``).  Applied only to tz-naive
                datetime columns; tz-aware columns are converted to UTC directly.
                Defaults to ``"UTC"``.
            align_on:
                A tuple of column names to use for aligning the new data with the
                existing data. The first element is the column name in the existing
                data, and the second element is the column name in the new data.
                The new data will be shifted in time to maximize the cross-correlation
                between the two columns. Defaults to None.
            join_strategy:
                The strategy for which times to keep in the result:
                - "keep_existing": Keep only times from existing data
                - "keep_new": Keep only times from new data
                - "keep_both": Keep all times from both datasets
                Defaults to "keep_existing".
            fill_strategy:
                The strategy for filling missing values in the merged dataset columns
                after applying the join strategy (this may affect both existing and
                new columns):
                - "interpolate": Interpolate missing values by unix time
                - "forward_fill": Forward fill missing values
                - "backward_fill": Backward fill missing values
                - None: Don't fill missing values
                Defaults to "interpolate".

        Raises:
            ValueError: If the base dataframe has no "Unix Time / s" column.
            ValueError: If an invalid timezone string is provided.
        """
        # Load external file if needed
        if isinstance(new_data, str):
            new_data = self.load_external_file(new_data)

        # Apply column_map (select and rename columns)
        if column_map is not None:
            cols_to_select = [time_column_name] + list(column_map.values())
            new_data = new_data.select(cols_to_select)
            rename_map = {src: dest for dest, src in column_map.items()}
            new_data = new_data.rename(rename_map)

        # Validate base dataframe has Unix Time column
        if "Unix Time / s" not in self.lf.collect_schema().names():
            error_msg = "No 'Unix Time / s' column in the base dataframe."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Convert new_data to match the type of lf
        _, new_data = self._verify_compatible_frames(
            self.lf,
            [new_data],
            mode="match 1",
        )
        new_data = new_data[0]

        # Convert time column to "Unix Time / s" Float64
        schema = new_data.collect_schema()
        time_dtype = schema[time_column_name]

        # Handle String dtype: parse to datetime first
        if isinstance(time_dtype, pl.String):
            new_data = new_data.with_columns(
                pl.col(time_column_name).str.to_datetime(format=datetime_format)
            )
            time_dtype = pl.Datetime(time_unit="us")  # Update dtype after conversion

        # Handle Datetime dtype: convert to UTC unix seconds
        if isinstance(time_dtype, pl.Datetime):
            col_tz = time_dtype.time_zone
            if col_tz is None:
                # Tz-naive: interpret as the specified timezone (default "UTC")
                validate_timezone(timezone)
                col = pl.col(time_column_name).dt.replace_time_zone(timezone)
            else:
                # Tz-aware: convert to UTC directly
                col = pl.col(time_column_name).dt.convert_time_zone("UTC")

            new_data = new_data.with_columns(
                col.dt.epoch(time_unit="s").cast(pl.Float64).alias(time_column_name)
            )
        # Handle numeric dtype: cast to Float64 (assumed UTC unix seconds)
        elif isinstance(time_dtype, (pl.Float32, pl.Float64, pl.Int32, pl.Int64)):
            new_data = new_data.with_columns(pl.col(time_column_name).cast(pl.Float64))
        else:
            error_msg = (
                f"Unsupported dtype for time column: {time_dtype}. "
                "Must be String, Datetime, or numeric."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Rename time column to "Unix Time / s"
        new_data = new_data.rename({time_column_name: "Unix Time / s"})
        if isinstance(new_data, pl.DataFrame):
            new_data = new_data.lazy()
        new_result = Table(lf=new_data)

        # Collect new data column names (excluding unix time)
        new_data_cols = [
            col for col in new_data.collect_schema().names() if col != "Unix Time / s"
        ]

        # Optionally align the new data with existing data
        if align_on is not None:
            from pyprobe.analysis.time_series import align_data

            col_existing, col_new = align_on
            _, new_result = align_data(self, new_result, col_existing, col_new)

        new_data = new_result.lf

        # Join all data to prepare for filling
        all_data = (
            self.lf.clone()
            .join(
                new_data,
                on="Unix Time / s",
                how="full",
                coalesce=True,
            )
            .sort("Unix Time / s")
        )

        # Get all non-Unix Time columns for filling
        all_cols_except_time = [
            col for col in all_data.collect_schema().names() if col != "Unix Time / s"
        ]
        # Restrict interpolation to numeric columns only, since interpolate_by
        # is not supported for non-numeric dtypes.
        schema = all_data.collect_schema()
        numeric_cols_except_time = [
            name
            for name, dtype in zip(schema.names(), schema.dtypes())
            if name != "Unix Time / s" and dtype in pl.NUMERIC_DTYPES
        ]

        # Apply fill strategy to all columns (both existing and new)
        valid_fill_strategies = {None, "interpolate", "forward_fill", "backward_fill"}
        if fill_strategy not in valid_fill_strategies:
            raise ValueError(
                f"Unsupported fill_strategy: {fill_strategy!r}. "
                "Valid options are None, 'interpolate', 'forward_fill', "
                "'backward_fill'."
            )
        if fill_strategy == "interpolate":
            if numeric_cols_except_time:
                filled = all_data.with_columns(
                    pl.col(numeric_cols_except_time).interpolate_by("Unix Time / s"),
                )
            else:
                # No numeric columns to interpolate; leave data unchanged.
                filled = all_data
        elif fill_strategy == "forward_fill":
            filled = all_data.with_columns(
                pl.col(all_cols_except_time).forward_fill(),
            )
        elif fill_strategy == "backward_fill":
            filled = all_data.with_columns(
                pl.col(all_cols_except_time).backward_fill(),
            )
        else:  # fill_strategy is None
            filled = all_data

        # Apply join strategy
        if join_strategy == "keep_existing":
            # Keep only existing times
            filled_new_cols = filled.select(pl.col(["Unix Time / s"] + new_data_cols))
            self.lf = self.lf.join(
                filled_new_cols,
                on="Unix Time / s",
                how="left",
                coalesce=True,
            )
        elif join_strategy == "keep_new":
            # Keep only new times
            # Filter filled to only times that exist in new_data
            self.lf = filled.join(
                new_data.select(["Unix Time / s"]),
                on="Unix Time / s",
                how="inner",
            )
        elif join_strategy == "keep_both":
            # Keep all times from both datasets
            self.lf = filled
        else:
            raise ValueError(
                f"Unsupported join_strategy: {join_strategy!r}. "
                "Expected one of: 'keep_existing', 'keep_new', 'keep_both'."
            )

    @deprecated(
        reason="Use add_data instead.",
        version="2.3.1",
    )
    def add_new_data_columns(
        self,
        new_data: pl.DataFrame | pl.LazyFrame,
        date_column_name: str,
    ) -> None:
        """Add new data columns to the result object.

        The data must be time series data with a date column. The new data is joined to
        the base dataframe on the date column, and the new data columns are interpolated
        to fill in missing values.

        Args:
            new_data (pl.DataFrame | pl.LazyFrame):
                The new data to add to the result object.
            date_column_name (str):
                The name of the column in the new data containing the date.

        Raises:
            ValueError: If the base dataframe has no date column.
        """
        raise NotImplementedError("This method is deprecated. Use add_data instead.")

    def join(
        self,
        other: "Table",
        on: str | list[str],
        how: str = "inner",
        coalesce: bool = True,
    ) -> None:
        """Join two Result objects on a column. A wrapper around the polars join method.

        This will extend the data in the Result object horizontally. Each object
        keeps its own metadata record, so the column definitions of the other object
        do not travel with its columns.

        Args:
            other (Result): The other Result object to join with.
            on (Union[str, List[str]]): The column(s) to join on.
            how (str): The type of join to perform. Default is 'inner'.
            coalesce (bool): Whether to coalesce the columns. Default is True.
        """
        _, other_frame = self._verify_compatible_frames(
            self.lf,
            [other.lf],
            mode="match 1",
        )
        if isinstance(on, str):
            on = [on]
        self.lf = self.lf.join(
            other_frame[0],
            on=on,
            how=how,
            coalesce=coalesce,
        )

    def extend(
        self,
        other: Union["Table", list["Table"]],  # noqa: UP007
        concat_method: str = "diagonal",
    ) -> None:
        """Extend the data in this Result object with the data in another Result object.

        This method will concatenate the data in the two Result objects, with the Result
        object calling the method above the other Result object.

        This object keeps its own metadata record. Where another record differs, one
        warning names every top level field that differs.

        Args:
            other (Result | List[Result]): The other Result object(s) to extend with.
            concat_method (str):
                The method to use for concatenation. Default is 'diagonal'. See the
                polars.concat method documentation for more information.
        """
        other = self._as_list(other)
        self._warn_on_differing_metadata(other)
        other_frame_list = [other_result.lf for other_result in other]
        self.lf, other_frame_list = self._verify_compatible_frames(
            self.lf,
            other_frame_list,
            mode="collect all",
        )
        self.lf = pl.concat(
            [self.lf] + other_frame_list,
            how=concat_method,
        )

    @staticmethod
    def _as_list(other: Union["Table", list["Table"]]) -> list["Table"]:  # noqa: UP007
        """Return the other object(s) an extend combines with, as a list.

        Args:
            other: One Table object, or a list of them.

        Returns:
            ``other`` unchanged where it is already a list, otherwise a list
            holding the one object.
        """
        if isinstance(other, list):
            return other
        return [other]

    def _merge_column_definitions(self, other: list["Table"]) -> None:
        """Merge the column definitions of other objects into this one's.

        Where a definition conflicts, this object's own definition takes
        precedence.

        Args:
            other: The other Table object(s) being extended with.
        """
        merged: dict[str, str] = {}
        for other_result in other:
            merged.update(other_result.column_definitions)
        merged.update(self._stored_definitions())
        self._write_definitions(merged)

    def _warn_on_differing_metadata(self, other: list["Table"]) -> None:
        """Log one warning naming every top level field that differs.

        Args:
            other: The other Table object(s) being extended with.
        """
        differing_fields: set[str] = set()
        for other_result in other:
            for field in type(self.metadata).model_fields:
                if getattr(self.metadata, field) != getattr(
                    other_result.metadata, field
                ):
                    differing_fields.add(field)
        if differing_fields:
            logger.warning(
                "This object keeps its own metadata record; the following "
                "top level fields differ from another record being extended "
                "with: {}",
                ", ".join(sorted(differing_fields)),
            )

    @classmethod
    def build(
        cls,
        data_list: list[
            pl.LazyFrame
            | pl.DataFrame
            | dict[str, NDArray[np.float64] | list[float]]
            | list[
                pl.LazyFrame
                | pl.DataFrame
                | dict[str, NDArray[np.float64] | list[float]]
            ]
        ],
        info: dict[str, Any | None],
    ) -> "Table":
        """Build a Table object from a list of dataframes.

        Args:
            data_list (List[List[pl.LazyFrame | pl.DataFrame | dict]]):
                The data to include in the new result object.
                The first index indicates the cycle and the second index indicates the
                step.
            info (dict[str, Optional[str | int | float]]): A dict containing test info.

        Returns:
            Result: A new result object with the specified data.
        """
        cycles_and_steps_given = all(isinstance(item, list) for item in data_list)
        if not cycles_and_steps_given:
            data_list = [data_list]
        data = []
        for cycle, cycle_data in enumerate(data_list):
            for step, step_data in enumerate(cycle_data):
                if isinstance(step_data, dict):
                    step_data = pl.DataFrame(step_data)
                step_data = step_data.with_columns(
                    pl.lit(cycle).alias("Cycle"),
                    pl.lit(step).alias("Step"),
                )
                data.append(step_data)
        data = pl.concat(data)
        if isinstance(data, pl.DataFrame):
            data = data.lazy()
        return cls(lf=data, metadata=bdf.Metadata(extras=info))

    def save(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
        labels: Literal["preferred", "machine", "unchanged"] = "unchanged",
        compression_priority: Literal[
            "performance",
            "file size",
            "uncompressed",
        ] = "performance",
    ) -> Path:
        """Write the object to a BDF artifact.

        The write produces the Parquet data file and the
        ``<stem>.metadata.json`` sidecar together. It records the PyProBE
        version and the write time under ``metadata.extras["pyprobe"]``.

        Args:
            path: The path of the Parquet data file to write.
            overwrite: When ``True``, replace an existing data file.
            labels: The column label form that the data file holds.
            compression_priority: The trade-off between write speed and file
                size that selects the Parquet compression algorithm.

        Returns:
            Path: The path of the data file that was written.

        Raises:
            ValueError: If *path* does not end with ``.parquet``. The message
                names the suffix.
            FileExistsError: If the data file exists and *overwrite* is
                ``False``.
            bdf.BDFValidationError: If a required BDF column is absent.
        """
        import datetime as dt

        from pyprobe._version import __version__ as _pyprobe_version
        from pyprobe.io import _COMPRESSION_MAP

        resolved_path = Path(path)
        if resolved_path.suffix != ".parquet":
            raise ValueError(f"path must end with '.parquet', got: '{resolved_path}'")
        if not overwrite and resolved_path.exists():
            raise FileExistsError(
                f"'{resolved_path}' already exists. Pass overwrite=True to replace it."
            )

        extras = dict(self.metadata.extras or {})
        extras["pyprobe"] = {
            "version": _pyprobe_version,
            "written_at": dt.datetime.now(dt.UTC).isoformat(),
        }
        metadata = self.metadata.model_copy(update={"extras": extras})

        bdf.io.save(
            self.lf,
            resolved_path,
            metadata=metadata,
            labels=labels,
            compression=_COMPRESSION_MAP[compression_priority],
        )
        return resolved_path

    def export_to_mat(self, filename: str) -> None:
        """Export the data to a .mat file.

        This method will export the data and metadata dictionary to a .mat file. The
        variables in the .mat file will be named 'data' and 'metadata'. Column names and
        dictionary keys will have any non-alphanumeric characters replaced with an
        underscore, to comply with MATLAB variable naming rules.

        Args:
            filename: The name of the file to export to.
        """
        # Replace any non-alphanumeric character with an underscore in the DataFrame
        # columns
        renamed_data = self.data.rename(
            {col: re.sub(r"\W", "_", col) for col in self.data.columns},
        )

        # Replace any non-alphanumeric character with an underscore in the metadata
        # dictionary keys
        renamed_metadata = {
            re.sub(r"\W", "_", key): value for key, value in self.info.items()
        }

        variable_dict = {
            "data": renamed_data.to_dict(),
            "metadata": renamed_metadata,
        }
        savemat(filename, variable_dict, oned_as="column")

    @staticmethod
    def from_polars_io(
        polars_io_func: Callable[..., pl.DataFrame | pl.LazyFrame],
        metadata: bdf.Metadata | None = None,
        **kwargs: Any,
    ) -> "Table":
        """Create a new Table object with data from a Polars IO function.

        Refer to the Polars documentation for a list of available IO functions:

        - `External file import functions \
            <https://docs.pola.rs/api/python/stable/reference/io.html>`_
        - `Python object conversion functions \
            <https://docs.pola.rs/api/python/stable/reference/functions.html>`_

        Args:
            polars_io_func (Callable[..., pl.DataFrame | pl.LazyFrame]):
                The Polars IO function to use to create the data.
            metadata (bdf.Metadata | None):
                The metadata record for the new Table object. An empty
                ``bdf.Metadata`` is used where ``None``.
            **kwargs: The keyword arguments to pass to the Polars IO function.

        Returns:
            Result: A new Result object with the specified data and info.

        Example:
            From a saved .csv file:

            .. code-block:: python

            result = Table.from_polars_io(
                pl.scan_csv,
                metadata=bdf.Metadata(extras={"test": "test"}),
                source="data.csv",
            )

            From a pandas DataFrame:

            .. code-block:: python

            result = Table.from_polars_io(
                pl.from_pandas,
                metadata=bdf.Metadata(extras={"test": "test"}),
                data=pd.DataFrame({"a": [1, 2, 3]}),
            )

            From a numpy array:

            .. code-block:: python

            result = Table.from_polars_io(
                pl.from_numpy,
                metadata=bdf.Metadata(extras={"test": "test"}),
                data=np.array([[1, 2, 3], [4, 5, 6]]),
                schema=["a", "b"]
            )

        """
        lf = polars_io_func(**kwargs)
        if isinstance(lf, pl.DataFrame):
            lf = lf.lazy()
        return Table(lf=lf, metadata=metadata)

    @property
    @deprecated(
        reason=(
            "The live_dataframe property is deprecated. Use the lf property instead."
        ),
        version="2.4.0",
    )
    def live_dataframe(self) -> pl.LazyFrame:
        """The base dataframe as a LazyFrame.

        Returns:
            pl.LazyFrame: The base dataframe as a LazyFrame.
        """
        return self.lf

    @live_dataframe.setter
    @deprecated(
        reason=(
            "The live_dataframe property is deprecated. Use the lf property instead."
        ),
        version="2.4.0",
    )
    def live_dataframe(self, value: pl.LazyFrame) -> None:
        self.lf = value

    @property
    @deprecated(
        reason=(
            "The base_dataframe property is deprecated. Use the lf property instead."
        ),
        version="2.4.0",
    )
    def base_dataframe(self) -> pl.LazyFrame:
        """The base dataframe as a LazyFrame.

        Returns:
            pl.LazyFrame: The base dataframe as a LazyFrame.
        """
        return self.lf

    @base_dataframe.setter
    @deprecated(
        reason=(
            "The base_dataframe property is deprecated. Use the lf property instead."
        ),
        version="2.4.0",
    )
    def base_dataframe(self, value: pl.LazyFrame) -> None:
        self.lf = value


def combine_results(
    results: list[Table],
    concat_method: str = "diagonal",
) -> Table:
    """Combine multiple Table objects into a single Table object.

    This method should be used to combine multiple Table objects that have different
    entries in their info dictionaries. The info dictionaries of the Table objects will
    be integrated into the dataframe of the new Table object

    Args:
        results (List[Table]): The Table objects to combine.
        concat_method (str):
            The method to use for concatenation. Default is 'diagonal'. See the
            polars.concat method documentation for more information.

    Returns:
        Table: A new Table object with the combined data.
    """
    for result in results:
        instructions = [pl.lit(result.info[key]).alias(key) for key in result.info]
        result.lf = result.lf.with_columns(instructions)
    results[0].extend(results[1:], concat_method=concat_method)
    return results[0]


class _ResultMeta(type):
    """Metaclass making ``isinstance(obj, Result)`` true for any ``Table``.

    The :class:`Result` alias is a deprecated subclass of :class:`Table`. This
    metaclass keeps ``isinstance(table, Result)`` working for *all* ``Table``
    instances (not just those constructed via ``Result``), preserving the
    pre-rename behaviour while still warning on direct construction.
    """

    def __instancecheck__(cls, instance: object) -> bool:
        """Return ``True`` for any :class:`Table` instance."""
        return isinstance(instance, Table)


class Result(Table, metaclass=_ResultMeta):
    """Deprecated alias of :class:`Table`.

    ``Result`` was renamed to :class:`Table`. This subclass keeps existing code
    and notebooks working: it constructs a fully functional ``Table`` while
    emitting a :class:`DeprecationWarning`. ``isinstance(obj, Result)`` remains
    ``True`` for any ``Table`` (or subclass) instance.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Warn that ``Result`` is deprecated, then construct a ``Table``."""
        warnings.warn(
            "Result has been renamed to Table. Use 'from pyprobe.result import "
            "Table'. The Result alias will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
