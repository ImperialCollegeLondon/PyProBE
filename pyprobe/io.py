"""BDF-based cycler data import utilities for PyProBE.

Provides :func:`process_cycler` as the primary entry point for reading raw
cycler files via the ``batterydf`` package, normalising them to BDF-standard
column names, and persisting to a BDF artifact.

Also provides :func:`process_generic` for normalising arbitrary battery data to
BDF format under a caller-supplied column map, :func:`read_sidecar` for
reading the metadata that sits beside a data file, and :func:`is_pyprobe_file`
for detecting a file that PyProBE wrote.

Typical usage::

    from pyprobe.io import process_cycler

    path = process_cycler("path/to/data.xlsx")
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import bdf.io
import bdf.metadata_parsers
import polars as pl
from loguru import logger

from pyprobe.columns import (
    BDF,
    CORE_COLUMN_GROUPS,
    CORE_COLUMNS,
    ColumnDict,
    column_factory_from_string,
)

if TYPE_CHECKING:
    import pandas as pd

_ParquetCompression = Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"]

_COMPRESSION_MAP: dict[str, _ParquetCompression] = {
    "performance": "lz4",
    "file size": "zstd",
    "uncompressed": "uncompressed",
}
"""Maps compression_priority literals to Parquet compression algorithm names."""


def _resolve_glob(source: str | Path) -> list[Path]:
    """Expand a glob pattern or return a single path as a list.

    Args:
        source: A file path or a glob pattern containing ``"*"``.

    Returns:
        Sorted list of resolved paths.

    Raises:
        FileNotFoundError: If *source* is a glob pattern that matches no files.
    """
    source_str = str(source)
    if "*" in source_str:
        matches = sorted(glob.glob(source_str))
        if not matches:
            raise FileNotFoundError(f"No files found matching pattern: {source}")
        return [Path(m) for m in matches]
    return [Path(source)]


def _handle_existing_cached_file(output_path: Path) -> Path | None:
    """Check if a cached output file exists and should be reused.

    Args:
        output_path: Path to the expected cached Parquet file.

    Returns:
        The cached file path if it exists, otherwise ``None``.
    """
    if not output_path.exists():
        return None
    logger.info("Skipping processing; using cached file '{}'.", output_path)
    return output_path


def is_pyprobe_file(path: Path | str) -> bool:
    """Return True if path is a PyProBE-written Parquet file.

    Args:
        path: Path to the Parquet file.

    Returns:
        True if the BDF metadata sidecar beside *path* holds a ``"pyprobe"``
        key with a dict value under its ``extras``.

    Raises:
        FileNotFoundError: If path does not exist.
        bdf.BDFMetadataError: If a sidecar exists beside path and does not
            parse.

    Example::

        from pyprobe.io import is_pyprobe_file

        if is_pyprobe_file("data.bdf.parquet"):
            procedure = Procedure.load("data.bdf.parquet")
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    metadata = read_sidecar(file_path)
    extras = metadata.extras or {}
    return isinstance(extras.get("pyprobe"), dict)


def _build_column_map_exprs(
    columns: list[str],
    column_map: dict[str | BDF, str],
) -> list[pl.Expr]:
    """Validate a column map and build the corresponding Polars select expressions.

    Args:
        columns: Column names available in the source frame.
        column_map: Mapping from BDF-format output names (e.g. ``"Current / A"``
            or :attr:`BDF.CURRENT_AMPERE`) to source column names.

    Returns:
        A list of ``pl.col(src).alias(output)`` expressions ready for
        ``.select()`` or ``.sink_parquet()``.

    Raises:
        ValueError: If an output name is not a valid BDF-format string, or if a
            source column name is not present in *columns*.
    """
    strict_pattern = r"^(.+?)\s*/\s*([^/]+(?:/[^/]+)*)$"
    exprs: list[pl.Expr] = []
    for key, src_name in column_map.items():
        if isinstance(key, BDF):
            output_name = key.name
        else:
            column_factory_from_string(key, pattern=strict_pattern)
            output_name = key
        if src_name not in columns:
            raise ValueError(
                f"column_map source '{src_name}' not found in data. "
                f"Available: {columns}"
            )
        exprs.append(pl.col(src_name).alias(output_name))
    return exprs


def _resolve_time_column(column_set: ColumnDict) -> pl.Expr:
    """Resolve a time column, preferring Unix Time but falling back to Test Time.

    Attempts to resolve Unix Time first; if unavailable, falls back to Test Time.
    At least one of these must be resolvable.

    Args:
        column_set: ColumnDict with available columns.

    Returns:
        A Polars expression for the resolved time column.

    Raises:
        ValueError: If neither Unix Time nor Test Time can be resolved.
    """
    # Try Unix Time first (preferred)
    try:
        return column_set.resolve(BDF.UNIX_TIME_SECOND)
    except ValueError:
        pass

    # Fall back to Test Time
    try:
        return column_set.resolve(BDF.TEST_TIME_SECOND)
    except ValueError as exc:
        raise ValueError(
            "Required time column: either 'Unix Time / s' or 'Test Time / s' "
            "must be available in the source data."
        ) from exc


def _core_time_group_name(group: tuple[BDF, ...]) -> str:
    """Return a human-readable name for a required BDF column group.

    Args:
        group: The columns of a :data:`~pyprobe.columns.CORE_COLUMN_GROUPS` entry.

    Returns:
        A description naming every column of *group*, e.g.
        ``"'Unix Time / s' or 'Test Time / s'"``.
    """
    return " or ".join(f"'{bdf_col.quantity} / {bdf_col.unit}'" for bdf_col in group)


def _normalised_column_expressions(
    column_set: ColumnDict, *, warn: bool = True
) -> list[pl.Expr]:
    """Resolve every core BDF column against *column_set*.

    Iterates the required column groups of
    :data:`~pyprobe.columns.CORE_COLUMN_GROUPS` and the single columns of
    :data:`~pyprobe.columns.CORE_COLUMNS`, keeping every column that resolves.
    A required item that resolves from no column raises. A silent item is
    skipped without comment.

    Args:
        column_set: The available columns to resolve against.
        warn: Where ``True`` (default), log one warning for an optional item
            that resolves from no column. A caller that separately reports a
            missing optional column passes ``False`` to avoid a second report.

    Returns:
        Expressions for every core column, or group member, that resolves.

    Raises:
        ValueError: If a required column, or every column of a required group,
            cannot be resolved.
    """
    expressions: list[pl.Expr] = []
    for group, status in CORE_COLUMN_GROUPS.items():
        resolved_any = False
        for bdf_col in group:
            try:
                expressions.append(column_set.resolve(bdf_col))
                resolved_any = True
            except ValueError:
                continue
        if not resolved_any and status == "required":
            raise ValueError(
                f"Required time column: either {_core_time_group_name(group)} "
                "must be available in the source data."
            )

    for bdf_col, status in CORE_COLUMNS.items():
        try:
            expressions.append(column_set.resolve(bdf_col))
        except ValueError as exc:
            if status == "required":
                raise ValueError(
                    f"Required BDF column '{bdf_col.quantity}' could not be resolved "
                    f"from the source data: {exc}"
                ) from exc
            if status == "optional" and warn:
                logger.warning(
                    "Optional BDF column '{}' could not be resolved; skipping.",
                    bdf_col.quantity,
                )
    return expressions


def process_cycler(
    source: str | Path,
    output_path: str | Path | None = None,
    *,
    overwrite_data: bool = False,
    compression_priority: Literal[
        "performance", "file size", "uncompressed"
    ] = "performance",
    **load_kwargs: Any,
) -> Path:
    """Read cycler file(s), normalise to BDF columns, and write a BDF artifact.

    Expands *source* to one or more raw cycler files, loads each through
    :meth:`~pyprobe.filters.Procedure.load`, extends the first with the rest,
    and saves the result through :meth:`~pyprobe.result.Table.save`.

    Args:
        source: Path to the raw cycler file, or a glob pattern matching multiple
            files (e.g. ``"data/session_*.csv"``).
        output_path: Full destination path for the output Parquet file (must end
            with ``.parquet``). When ``None``, defaults to
            ``<source_parent>/<stem>.bdf.parquet`` where *stem* comes from
            *source* (or the first sorted glob match for glob patterns).
        overwrite_data: When ``False`` (default), return the cached Parquet path
            immediately if it already exists without reprocessing raw data.
            When ``True``, reprocess and overwrite the existing file.
        compression_priority: Controls the Parquet compression algorithm:

            - ``"performance"`` (default) — uses ``lz4`` for fast read/write.
            - ``"file size"`` — uses ``zstd`` for smaller files.
            - ``"uncompressed"`` — no compression.

        load_kwargs: Forwarded to :meth:`~pyprobe.filters.Procedure.load` for
            every file, e.g. ``plugin`` or ``extra_columns``.

    Returns:
        Path to the written ``.bdf.parquet`` file.

    Raises:
        FileNotFoundError: If *source* is a glob pattern that matches no files.
        ValueError: If *source* is a PyProBE-written file (use
            :func:`~pyprobe.filters.Procedure.load` instead).
        ValueError: If *output_path* is provided but does not end with ``.parquet``.

    Example:
        Basic usage (writes ``data.bdf.parquet`` next to source)::

            path = process_cycler("data.xlsx")

        Output to a specific path::

            path = process_cycler("data.xlsx", output_path="cache/data.bdf.parquet")

        Add a column not auto-resolved by ``bdf``::

            path = process_cycler(
                "data.xlsx",
                extra_columns={"Pressure(kPa)": "Ambient Pressure / kPa"},
            )
    """
    from pyprobe.filters import Procedure

    first_file = _resolve_glob(source)[0]

    if (
        first_file.suffix == ".parquet"
        and first_file.exists()
        and is_pyprobe_file(first_file)
    ):
        raise ValueError(
            f"'{first_file}' is a PyProBE-written file. "
            "Use Procedure.load() to load already-processed files "
            "instead of process_cycler()."
        )

    if output_path is not None:
        candidate = Path(output_path)
        if candidate.suffix == "":
            # Treat as a directory; auto-generate the filename within it.
            resolved_output_path = candidate / (first_file.stem + ".bdf.parquet")
        elif candidate.suffix != ".parquet":
            raise ValueError(
                f"output_path must end with '.parquet', got: '{output_path}'"
            )
        else:
            resolved_output_path = candidate
    else:
        resolved_output_path = first_file.parent / (first_file.stem + ".bdf.parquet")

    if not overwrite_data:
        cached = _handle_existing_cached_file(resolved_output_path)
        if cached is not None:
            return cached

    files = _resolve_glob(source)
    procedure = Procedure.load(files[0], **load_kwargs)
    if len(files) > 1:
        procedure.extend([Procedure.load(f, **load_kwargs) for f in files[1:]])

    procedure.save(
        resolved_output_path,
        overwrite=True,
        compression_priority=compression_priority,
    )
    logger.info("Wrote normalised data to '{}'.", resolved_output_path)
    return resolved_output_path


def read_sidecar(path: str | Path) -> bdf.Metadata:
    """Read the BDF metadata sidecar that sits beside a data file.

    The sidecar is ``<stem>.metadata.json``, which ``bdf.io.save`` writes with
    the data file.

    Args:
        path: Path to the data file.

    Returns:
        bdf.Metadata: The record the sidecar holds, or an empty record where
            the data file has no sidecar.

    Raises:
        FileNotFoundError: If the data file does not exist.
        bdf.BDFMetadataError: If the sidecar does not parse.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    parser = bdf.metadata_parsers.BdfSidecarParser()
    return parser.parse(path)


def process_generic(
    source: str | Path | pl.LazyFrame | pl.DataFrame | pd.DataFrame,
    column_map: dict[str | BDF, str],
    output_path: str | Path | None = None,
    *,
    overwrite_data: bool = False,
    compression_priority: Literal[
        "performance", "file size", "uncompressed"
    ] = "performance",
    **load_kwargs: Any,
) -> Path:
    """Normalise arbitrary battery data to BDF format and write a BDF artifact.

    Loads *source* through :meth:`~pyprobe.filters.Procedure.load`, passing
    *column_map* to name the source column of each BDF column, and saves the
    result through :meth:`~pyprobe.result.Table.save`.

    Args:
        source: Raw battery data: a path to a file, a polars DataFrame, a
            polars LazyFrame, or a pandas DataFrame.
        column_map: Mapping from BDF-format output name (e.g. ``"Current / A"``)
            to the column name in *source*. Applies where *source* is a frame;
            ignored where *source* is a path.
        output_path: Destination path for the output Parquet file. When
            ``None`` and *source* is a path, defaults to
            ``<source_parent>/<stem>.bdf.parquet``.
        overwrite_data: When ``False`` (default), return the existing output path
            immediately if it already exists without reprocessing. When ``True``,
            reprocess and overwrite the existing file.
        compression_priority: Compression algorithm selection.
        load_kwargs: Forwarded to :meth:`~pyprobe.filters.Procedure.load`.

    Returns:
        The resolved path of the written Parquet file.

    Raises:
        ValueError: If *output_path* is ``None`` and *source* is not a path.
        ValueError: If the resolved output path does not end with ``.parquet``.
        ValueError: If any required BDF column cannot be resolved after
            applying *column_map*.
    """
    from pyprobe.filters import Procedure

    if output_path is not None:
        resolved_output_path = Path(output_path)
    elif isinstance(source, (str, Path)):
        source_path = Path(source)
        resolved_output_path = source_path.parent / (source_path.stem + ".bdf.parquet")
    else:
        raise ValueError("output_path is required unless source is a file path.")

    if resolved_output_path.suffix != ".parquet":
        raise ValueError(f"output_path must end with '.parquet', got: '{output_path}'")

    if not overwrite_data:
        cached = _handle_existing_cached_file(resolved_output_path)
        if cached is not None:
            return cached

    procedure = Procedure.load(source, column_map=column_map, **load_kwargs)
    procedure.save(
        resolved_output_path,
        overwrite=True,
        compression_priority=compression_priority,
    )
    logger.info("Wrote generic data to '{}'.", resolved_output_path)
    return resolved_output_path
