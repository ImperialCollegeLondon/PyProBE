"""Module for utilities for analysis classes."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from pyprobe.columns import Column, ColumnResolutionError
from pyprobe.pyprobe_types import PyProBEDataType
from pyprobe.result import Result


class ColumnCollisionError(ValueError):
    """Raised when a new column would overwrite an existing column.

    Pass ``overwrite=True`` to ``append_columns`` to suppress this error.
    """


def validate_columns(
    input_data: PyProBEDataType,
    *columns: str | Column,
) -> None:
    """Confirm each column reference can be resolved against input_data.

    Args:
        input_data: The input data object.
        *columns: Column references — str, Column, or BDF enum members.

    Raises:
        ColumnResolutionError: On the first unresolvable column reference.
    """
    col_dict = input_data.columns
    for col in columns:
        if not col_dict.can_resolve(col):
            name = col.name if isinstance(col, Column) else str(col)
            raise ColumnResolutionError(
                f"Cannot resolve '{name}' from available columns: {col_dict.names}"
            )


def get_columns(
    input_data: PyProBEDataType,
    *columns: str | Column,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], ...]:
    """Validate and return one or more columns as numpy arrays.

    Returns a single NDArray for one column, or a tuple of NDArrays for multiple.
    Unit conversion is automatic when a Column with a non-default unit is supplied.

    Args:
        input_data: The input data object.
        *columns: Column references — str, Column, or BDF enum members.

    Returns:
        A single NDArray for one column, or a tuple of NDArrays for multiple.

    Raises:
        ColumnResolutionError: On the first unresolvable column reference.
    """
    validate_columns(input_data, *columns)
    return input_data.get(*columns)


def resolve_exprs(
    input_data: PyProBEDataType,
    *columns: str | Column,
) -> tuple[pl.Expr, ...]:
    """Validate and return polars Expr objects for column references.

    Args:
        input_data: The input data object.
        *columns: Column references — str, Column, or BDF enum members.

    Returns:
        A tuple of polars Expr objects in the order supplied.

    Raises:
        ColumnResolutionError: On the first unresolvable column reference.
    """
    validate_columns(input_data, *columns)
    col_dict = input_data.columns
    return tuple(col_dict.resolve(col) for col in columns)


def build_result(
    source: PyProBEDataType,
    data: pl.LazyFrame | pl.DataFrame,
    column_definitions: dict[str, str] | None = None,
) -> Result:
    """Construct a Result inheriting source metadata.

    When column_definitions is None, source.column_definitions is inherited.
    When provided, it replaces source.column_definitions entirely on the new Result.
    To inherit and extend, call with ``column_definitions={**source.column_definitions,
    "New": "new def"}``.

    Args:
        source: The source data object providing metadata and column_definitions.
        data: The new data as a LazyFrame or DataFrame.
        column_definitions: When provided, replaces source.column_definitions entirely.

    Returns:
        A new Result with the given data and inherited or replaced column_definitions.
    """
    defs = (
        source.column_definitions if column_definitions is None else column_definitions
    )
    if isinstance(data, pl.DataFrame):
        data = data.lazy()
    return Result(lf=data, metadata=source.metadata, column_definitions=defs)


def append_columns(
    source: PyProBEDataType,
    new_columns: Mapping[str, pl.Expr | np.ndarray | pl.Series],
    *,
    overwrite: bool = False,
    column_definitions: dict[str, str] | None = None,
) -> Result:
    """Append new columns to source and return a new Result.

    Args:
        source: The source data object.
        new_columns: Mapping of column name to value (Expr, ndarray, or Series).
        overwrite: If False, raise ColumnCollisionError when a key already exists.
        column_definitions: Passed through to build_result.

    Returns:
        A new Result whose LazyFrame is source.lf extended with new_columns.

    Raises:
        ColumnCollisionError: If overwrite=False and a key collides with an existing
            column.
    """
    existing = set(source.columns.names)
    if not overwrite:
        for name in new_columns:
            if name in existing:
                raise ColumnCollisionError(
                    f"Column '{name}' already exists in source. "
                    "Use overwrite=True to replace."
                )
    exprs: list[pl.Expr | pl.Series] = []
    for name, value in new_columns.items():
        if isinstance(value, np.ndarray):
            exprs.append(pl.Series(name, value))
        elif isinstance(value, pl.Series):
            exprs.append(value.alias(name))
        else:
            exprs.append(value.alias(name))
    lf = source.lf.with_columns(exprs)
    return build_result(source, lf, column_definitions)


def assemble_array(
    input_data: list[Result],
    column: str | Column,
) -> NDArray[Any]:
    """Assemble an array from a list of results by stacking a column across all of them.

    Args:
        input_data: A list of Result objects.
        column: The column name or Column reference.

    Returns:
        NDArray: The assembled array via numpy.vstack.
    """
    return np.vstack([item.get(column) for item in input_data])
