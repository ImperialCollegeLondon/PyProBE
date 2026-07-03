"""BDF-based cycler data import utilities for PyProBE.

Provides :func:`process_cycler` as the primary entry point for reading raw
cycler files via the ``batterydf`` package, normalising them to BDF-standard
column names, and persisting to Parquet with attached metadata.

Also provides :func:`attach_metadata` for updating metadata on existing Parquet
files, and :func:`process_generic` for normalising arbitrary DataFrames to BDF
format without going through the cycler pipeline.

Typical usage::

    from pyprobe.io import process_cycler

    path = process_cycler("path/to/data.xlsx")
"""

from __future__ import annotations

import contextlib
import glob
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import bdf.io
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from pyprobe.columns import (
    BDF,
    ColumnDict,
    column_factory_from_string,
)

if TYPE_CHECKING:
    import pandas as pd

_PARQUET_METADATA_KEY: bytes = b"bdf_metadata"
"""Key used to store user metadata in Parquet footer."""

_REQUIRED_BDF_COLUMNS: list[BDF] = [
    BDF.TEST_TIME_SECOND,
    BDF.CURRENT_AMPERE,
    BDF.VOLTAGE_VOLT,
]
"""BDF columns that must be resolvable; :func:`process_cycler` raises if not."""

_OPTIONAL_BDF_COLUMNS: list[BDF] = [
    BDF.UNIX_TIME_SECOND,
    BDF.NET_CAPACITY_AH,
    BDF.STEP_COUNT,
    BDF.STEP_ID,
]
"""BDF columns included when available; warnings are emitted on failure."""

_SILENT_OPTIONAL_BDF_COLUMNS: list[BDF] = [
    BDF.AMBIENT_TEMPERATURE_CELSIUS,
    BDF.SURFACE_TEMPERATURE_CELSIUS,
    BDF.TEMPERATURE_T1_CELSIUS,
    BDF.TEMPERATURE_T2_CELSIUS,
    BDF.TEMPERATURE_T3_CELSIUS,
    BDF.TEMPERATURE_T4_CELSIUS,
    BDF.TEMPERATURE_T5_CELSIUS,
]
"""BDF columns included when available; no warning if missing."""

_ParquetCompression = Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"]

_COMPRESSION_MAP: dict[str, _ParquetCompression] = {
    "performance": "lz4",
    "file size": "zstd",
    "uncompressed": "uncompressed",
}
"""Maps compression_priority literals to Parquet compression algorithm names."""


class MetadataManager:
    """Encapsulates all metadata operations for Parquet files.

    Handles reading from and writing to both Parquet footers and JSON sidecars,
    with preference logic for choosing between sources and updating existing files.

    Example::

        manager = MetadataManager(output_path, metadata_format="parquet")
        existing = manager.read(metadata_format="parquet")
        manager.write({"cell_id": "C001"})
        manager.update({"new_key": "new_value"})
    """

    def __init__(self, path: Path) -> None:
        """Initialize MetadataManager for a Parquet file.

        Args:
            path: Path to the Parquet file.
        """
        self.path = Path(path)
        self.json_path = self.path.with_suffix(".json")

    def read_parquet(self) -> dict[str, Any]:
        """Read metadata from the Parquet file footer.

        Returns:
            Dictionary of metadata, or empty dict if missing.

        Raises:
            ValueError: If metadata exists but is corrupted (invalid JSON or encoding).
        """
        pf = pq.ParquetFile(self.path)
        raw: dict[bytes, bytes] = pf.schema_arrow.metadata or {}
        if _PARQUET_METADATA_KEY not in raw:
            return {}
        try:
            return json.loads(raw[_PARQUET_METADATA_KEY].decode())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Parquet metadata is corrupted (invalid JSON): {exc}"
            ) from exc
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Parquet metadata is corrupted (invalid UTF-8 encoding): {exc}"
            ) from exc

    def read_json(self) -> dict[str, Any]:
        """Read metadata from the JSON sidecar file.

        Returns:
            Dictionary of metadata, or empty dict if missing or not a dict.
        """
        if not self.json_path.exists():
            return {}
        try:
            raw: Any = json.loads(self.json_path.read_text())
            if isinstance(raw, dict):
                return raw
        except json.JSONDecodeError as exc:
            logger.warning(
                "Failed to decode JSON metadata from '{}': {}. "
                "Returning empty metadata.",
                self.json_path,
                exc,
            )
        return {}

    def read(
        self, metadata_format: Literal["parquet", "json"] = "parquet"
    ) -> dict[str, Any]:
        """Read metadata for a specific storage format.

        Args:
            metadata_format: Which format to read from. ``"parquet"`` reads from
                the Parquet footer; ``"json"`` reads from the sidecar.

        Returns:
            Dictionary of metadata.
        """
        if metadata_format == "parquet":
            return self.read_parquet()
        return self.read_json()

    def read_both(
        self, prefer: Literal["parquet", "json"] = "parquet"
    ) -> dict[str, Any]:
        """Read metadata from both sources with preference and fallback logic.

        Tries to read the preferred source first. If the preferred source is
        corrupted (raises ValueError), falls back to the alternative source.
        If the alternative source is also unavailable, the error is re-raised.
        If both sources are missing or empty, returns an empty dict.

        Args:
            prefer: Which source to prefer when both exist or when only one
                has valid (non-corrupted) metadata.

        Returns:
            Dictionary of metadata from the preferred source, or the alternative
            source if the preferred source is corrupted, or an empty dict if
            both are missing.

        Raises:
            ValueError: If the preferred source is corrupted and the alternative
                source is also unavailable.
        """
        prefer_primary = prefer == "parquet"
        primary_reader = self.read_parquet if prefer_primary else self.read_json
        secondary_reader = self.read_json if prefer_primary else self.read_parquet

        # Try preferred source first
        try:
            primary_meta = primary_reader()
            if primary_meta:
                return primary_meta
        except ValueError:
            # Preferred source is corrupted; try the alternative
            secondary_meta = secondary_reader()
            if secondary_meta:
                return secondary_meta
            # Both sources corrupted or missing; re-raise the original error
            raise

        # Preferred source is empty; try secondary
        secondary_meta = secondary_reader()
        return secondary_meta if secondary_meta else {}

    def write(
        self,
        metadata: dict[str, Any],
        metadata_format: Literal["parquet", "json"] = "parquet",
    ) -> None:
        """Write metadata to a Parquet file in the specified format.

        Reads the existing Parquet file, embeds or sidecars the metadata, and
        writes back. If *metadata_format* is ``"parquet"``, metadata is stored
        in the Parquet footer. If ``"json"``, a sidecar file is written instead.

        Args:
            metadata: Dictionary of metadata to write.
            metadata_format: Where to store metadata.

        Raises:
            ValueError: If the Parquet file is corrupted.
        """
        if metadata_format == "parquet":
            pf = pq.ParquetFile(self.path)
            original_compression = (
                pf.metadata.row_group(0).column(0).compression.lower()
                if pf.metadata.num_row_groups > 0
                and pf.metadata.row_group(0).num_columns > 0
                else "snappy"
            )
            if original_compression == "uncompressed":
                original_compression = "none"
            table = pf.read()
            existing: dict[bytes, bytes] = table.schema.metadata or {}
            combined_meta = {
                **existing,
                _PARQUET_METADATA_KEY: json.dumps(metadata).encode(),
            }
            table = table.replace_schema_metadata(combined_meta)
            pq.write_table(table, self.path, compression=original_compression)
        else:
            self.json_path.write_text(json.dumps(metadata, indent=2))

    def update(
        self,
        metadata: dict[str, Any],
        metadata_format: Literal["parquet", "json"] = "parquet",
    ) -> None:
        """Update metadata on an existing cached file without reprocessing.

        Merges *metadata* with existing metadata (new values override old ones),
        then writes back in the specified format.

        Args:
            metadata: Dictionary of metadata to merge in.
            metadata_format: Which format to update.

        Raises:
            ValueError: If the Parquet file or JSON sidecar is corrupted.
        """
        existing_meta = self.read(metadata_format=metadata_format)
        merged_metadata = {**existing_meta, **metadata}
        if merged_metadata == existing_meta:
            return
        self.write(merged_metadata, metadata_format=metadata_format)

    @classmethod
    def create(
        cls,
        table: pa.Table,
        path: Path,
        metadata: dict[str, Any] | None = None,
        metadata_format: Literal["parquet", "json"] = "parquet",
    ) -> None:
        """Write a new Parquet file with optional metadata.

        Embeds or sidecars metadata as specified, then writes the Arrow table
        to the Parquet file. This method is for creating new files; use
        :meth:`write` or :meth:`update` for existing files.

        Args:
            table: Arrow table to persist.
            path: Destination file path.
            metadata: Optional metadata dictionary to attach.
            metadata_format: Where to store metadata ("parquet" or "json").
        """
        if metadata:
            if metadata_format == "parquet":
                existing: dict[bytes, bytes] = table.schema.metadata or {}
                combined_meta = {
                    **existing,
                    _PARQUET_METADATA_KEY: json.dumps(metadata).encode(),
                }
                table = table.replace_schema_metadata(combined_meta)
            else:
                json_path = path.with_suffix(".json")
                json_path.write_text(json.dumps(metadata, indent=2))
        pq.write_table(table, path)


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


def _load_raw_dataframes(
    source: str | Path,
    plugin: str | None,
    normalize: bool = True,
    extra_columns: dict[str, str] | None = None,
) -> list[pl.LazyFrame]:
    """Load raw cycler files into Polars LazyFrames.

    Expands *source* via :func:`_resolve_glob`, then reads each file using
    :func:`bdf.io.read`, optionally normalising to BDF column names.
    :func:`bdf.io.read` returns a ``(LazyFrame, metadata)`` tuple; only the
    LazyFrame is retained here.

    Args:
        source: A file path or glob pattern.
        plugin: BatteryDF plugin name. ``None`` triggers auto-detection.
        normalize: When ``True`` (default), normalise to BDF column names.
            When ``False``, preserve original source column names.
        extra_columns: Passed straight through to :func:`bdf.io.read` as its
            own ``extra_columns`` argument (mapping of source column name to
            output alias). ``bdf`` handles all validation.

    Returns:
        One LazyFrame per resolved file, in sorted order.
    """
    files = _resolve_glob(source)
    frames: list[pl.LazyFrame] = []
    for f in files:
        df, _meta = bdf.io.read(
            str(f),
            plugin=plugin,
            normalize=normalize,
            extra_columns=extra_columns,
            lazy=True,
        )
        frames.append(df if isinstance(df, pl.LazyFrame) else df.lazy())
    return frames


def _concat_dataframes(dfs: list[pl.LazyFrame]) -> pl.LazyFrame:
    """Concatenate a list of LazyFrames using diagonal (schema-union) mode.

    Args:
        dfs: LazyFrames to concatenate. Columns need not be identical; missing
            columns are filled with ``null``.

    Returns:
        Single concatenated LazyFrame.
    """
    return pl.concat(dfs, how="diagonal")


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


def _embed_provenance(path: Path) -> None:
    """Embed PyProBE provenance metadata into a Parquet file footer."""
    import datetime as _dt

    from pyprobe._version import __version__ as _version

    MetadataManager(path).update(
        {
            "pyprobe": {
                "version": _version,
                "written_at": _dt.datetime.now(_dt.UTC).isoformat(),
            }
        }
    )


def is_pyprobe_file(path: Path | str) -> bool:
    """Return True if path is a PyProBE-written Parquet file.

    Args:
        path: Path to the Parquet file.

    Returns:
        True if the file contains a ``"pyprobe"`` key with a dict value
        in its footer metadata.

    Raises:
        FileNotFoundError: If path does not exist.

    Example::

        from pyprobe.io import is_pyprobe_file

        if is_pyprobe_file("data.bdf.parquet"):
            procedure = Procedure.load("data.bdf.parquet")
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    meta = MetadataManager(file_path).read_parquet()
    return isinstance(meta.get("pyprobe"), dict)


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


def process_cycler(
    source: str | Path,
    output_path: str | Path | None = None,
    *,
    plugin: str | None = None,
    overwrite_data: bool = False,
    compression_priority: Literal[
        "performance", "file size", "uncompressed"
    ] = "performance",
    extra_columns: dict[str, str] | None = None,
) -> Path:
    """Read cycler file(s), normalise to BDF columns, and write to Parquet.

    Reads one or more raw cycler files (via a file path or glob pattern),
    normalises columns to BDF standard using ``batterydf``, and writes the
    result to a ``.bdf.parquet`` file.

    Args:
        source: Path to the raw cycler file, or a glob pattern matching multiple
            files (e.g. ``"data/session_*.csv"``).
        output_path: Full destination path for the output Parquet file (must end
            with ``.parquet``). When ``None``, defaults to
            ``<source_parent>/<stem>.bdf.parquet`` where *stem* comes from
            *source* (or the first sorted glob match for glob patterns).
        plugin: BatteryDF plugin name for reading. ``None`` triggers auto-detection.
        overwrite_data: When ``False`` (default), return the cached Parquet path
            immediately if it already exists without reprocessing raw data.
            When ``True``, reprocess and overwrite the existing file.
        compression_priority: Controls the Parquet compression algorithm:

            - ``"performance"`` (default) — uses ``lz4`` for fast read/write.
            - ``"file size"`` — uses ``zstd`` for smaller files.
            - ``"uncompressed"`` — no compression.

        extra_columns: Mapping from source column name to output alias.
            Passed straight through to :func:`bdf.io.read`'s own
            ``extra_columns`` argument; ``bdf`` performs the aliasing and all
            validation. Can only add columns not already auto-resolved by
            ``bdf`` (an alias colliding with an auto-resolved BDF column
            raises inside ``bdf``).

    Returns:
        Path to the written ``.bdf.parquet`` file.

    Raises:
        FileNotFoundError: If *source* is a glob pattern that matches no files.
        ValueError: If *source* is a PyProBE-written file (use
            :func:`~pyprobe.filters.Procedure.load` instead).
        ValueError: If *output_path* is provided but does not end with ``.parquet``.
        ValueError: If any time column (Unix Time or Test Time) cannot be resolved
            from the source data.
        ValueError: If any required BDF column (current, voltage) cannot be resolved
            from the source data.

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

    dfs = _load_raw_dataframes(source, plugin, extra_columns=extra_columns)
    df = _concat_dataframes(dfs)

    # if Unix Time / s in data already, drop Test Time / s
    # means Test Time / s is calculated from Unix Time / s where possible
    column_names = set(df.collect_schema().names())
    if {"Unix Time / s", "Test Time / s"}.issubset(column_names):
        df = df.drop("Test Time / s")

    column_set = ColumnDict(df.collect_schema().names())
    expressions: list[pl.Expr] = []
    for bdf_col in _REQUIRED_BDF_COLUMNS:
        try:
            expressions.append(column_set.resolve(bdf_col))
        except ValueError as exc:
            if bdf_col == BDF.TEST_TIME_SECOND:
                raise ValueError(
                    "Required time column: either 'Unix Time / s' or 'Test Time / s' "
                    "must be available in the source data."
                ) from exc
            raise ValueError(
                f"Required BDF column '{bdf_col.quantity}' could not be resolved "
                f"from the source data: {exc}"
            ) from exc

    for bdf_col in _OPTIONAL_BDF_COLUMNS:
        try:
            expressions.append(column_set.resolve(bdf_col))
        except ValueError:
            logger.warning(
                "Optional BDF column '{}' could not be resolved; skipping.",
                bdf_col.quantity,
            )

    for bdf_col in _SILENT_OPTIONAL_BDF_COLUMNS:
        with contextlib.suppress(ValueError):
            expressions.append(column_set.resolve(bdf_col))

    if extra_columns is not None:
        expressions.extend(pl.col(alias) for alias in extra_columns.values())

    normalised: pl.LazyFrame = df.select(expressions)

    normalised.sink_parquet(
        str(resolved_output_path),
        compression=_COMPRESSION_MAP[compression_priority],
    )
    _embed_provenance(resolved_output_path)
    logger.info("Wrote normalised data to '{}'.", resolved_output_path)
    return resolved_output_path


def read_metadata(
    path: str | Path,
    prefer: Literal["parquet", "json"] = "parquet",
) -> dict[str, Any]:
    r"""Read metadata from a Parquet file's footer or a ``.json`` sidecar.

    Checks both the Parquet footer (stored under \"bdf_metadata\") and a ``.json``
    sidecar (derived from *path* by replacing the ``.parquet`` suffix with
    ``.json``). When both sources contain metadata, *prefer* controls which is
    returned. When only one source has metadata, that source is returned
    regardless of *prefer*. When neither has metadata, an empty dict is returned.

    Args:
        path: Path to the Parquet file.
        prefer: Which source to return when both exist. ``\"parquet\"`` (default)
            returns the Parquet footer metadata; ``\"json\"`` returns the sidecar
            metadata.

    Returns:
        A dictionary of metadata key-value pairs with their original types
        preserved (via JSON round-tripping).

    Raises:
        ValueError: If *prefer* is not ``\"parquet\"`` or ``\"json\"``.

    Example:
        Load metadata from a processed battery file, choosing between Parquet
        footer and JSON sidecar::

            from pyprobe.io import read_metadata

            # Prefer Parquet footer metadata (default)
            meta = read_metadata("data.bdf.parquet")
            print(meta["cell_id"])  # 'C001'

            # Or prefer JSON sidecar if both exist
            meta = read_metadata("data.bdf.parquet", prefer="json")
    """
    if prefer not in ("parquet", "json"):
        raise ValueError(f"prefer must be 'parquet' or 'json', got '{prefer}'.")

    manager = MetadataManager(Path(path))
    return manager.read_both(prefer=prefer)


def attach_metadata(
    path: str | Path,
    metadata: dict[str, Any],
    metadata_format: Literal["parquet", "json"] = "parquet",
) -> None:
    """Attach or update metadata on an existing Parquet file.

    Merges *metadata* with any existing metadata stored in the file, with
    new values taking precedence.

    Args:
        path: Path to the existing Parquet file.
        metadata: JSON-serializable key-value pairs to attach.
        metadata_format: Where to store metadata. ``"parquet"`` (default) embeds
            in the Parquet footer. ``"json"`` writes a ``.json`` sidecar file.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    MetadataManager(path).update(metadata, metadata_format=metadata_format)


def process_generic(
    source: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
    column_map: dict[str | BDF, str],
    output_path: str | Path,
    compression_priority: Literal[
        "performance", "file size", "uncompressed"
    ] = "performance",
    *,
    overwrite_data: bool = False,
) -> Path:
    """Normalise an arbitrary DataFrame to BDF format and write to Parquet.

    Accepts a polars DataFrame, polars LazyFrame, or pandas DataFrame, renames
    columns per *column_map* (mapping BDF output name to source column name),
    validates that required BDF columns are resolvable, and writes all mapped
    columns to *output_path*.

    Args:
        source: Raw battery data. Accepts a polars DataFrame, polars LazyFrame,
            or pandas DataFrame.
        column_map: Mapping from BDF-format output name (e.g. ``"Current / A"``)
            to the column name in *source*.
        output_path: Destination path for the output Parquet file.
        compression_priority: Compression algorithm selection.
        overwrite_data: When ``False`` (default), return the existing output path
            immediately if it already exists without reprocessing. When ``True``,
            reprocess and overwrite the existing file.

    Returns:
        The resolved path of the written Parquet file.

    Raises:
        TypeError: If *source* cannot be converted to a Polars DataFrame.
        ValueError: If any required BDF column cannot be resolved after
            applying *column_map*.
    """
    output = Path(output_path)
    compression = _COMPRESSION_MAP[compression_priority]

    if not overwrite_data:
        cached = _handle_existing_cached_file(output)
        if cached is not None:
            return cached

    # Normalize input: convert to LazyFrame, tracking original type for output method
    is_lazy = isinstance(source, pl.LazyFrame)
    if not is_lazy:
        if not isinstance(source, pl.DataFrame):
            try:
                source = pl.from_pandas(source)
            except Exception as exc:
                raise TypeError(
                    f"Could not convert source to a Polars DataFrame: {exc}"
                ) from exc
        source = source.lazy()

    # Build and apply column map expressions
    exprs = _build_column_map_exprs(source.collect_schema().names(), column_map)
    output_columns = [str(e.meta.output_name()) for e in exprs]
    column_set = ColumnDict(output_columns)

    # Validate required BDF columns
    for bdf_col in _REQUIRED_BDF_COLUMNS:
        try:
            column_set.resolve(bdf_col)
        except ValueError as exc:
            raise ValueError(
                f"Required BDF column '{bdf_col.quantity}' could not be resolved "
                f"from the source: {exc}"
            ) from exc

    # Select mapped columns and write (method depends on original type)
    selected = source.select(exprs)
    if is_lazy:
        selected.sink_parquet(str(output), compression=compression)
    else:
        selected.collect().write_parquet(str(output), compression=compression)

    _embed_provenance(output)
    logger.info("Wrote generic data to '{}'.", output)
    return output
