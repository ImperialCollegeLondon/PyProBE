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

import bdf
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from pyprobe.column import (
    BDF,
    ColumnSet,
    column_factory_from_string,
)

if TYPE_CHECKING:
    import pandas as pd

_PARQUET_METADATA_KEY: bytes = b"bdx_metadata"
"""Key used to store user metadata in Parquet footer."""

_REQUIRED_BDF_TIME: list[BDF] = [BDF.UNIX_TIME_SECOND, BDF.TEST_TIME_SECOND]
"""Time columns (at least one must be resolvable); Unix Time is preferred."""

_REQUIRED_BDF_COLUMNS: list[BDF] = [
    BDF.CURRENT_AMPERE,
    BDF.VOLTAGE_VOLT,
]
"""BDF columns that must be resolvable; :func:`process_cycler` raises if not."""

_OPTIONAL_BDF_COLUMNS: list[BDF] = [
    BDF.NET_CAPACITY_AH,
    BDF.STEP_COUNT,
    BDF.STEP_INDEX,
]
"""BDF columns included when available; warnings are emitted on failure."""

_SILENT_OPTIONAL_BDF_COLUMNS: list[BDF] = [
    BDF.AMBIENT_TEMPERATURE_CELSIUS,
    BDF.TEMPERATURE_T1_CELCIUS,
    BDF.TEMPERATURE_T2_CELCIUS,
    BDF.TEMPERATURE_T3_CELCIUS,
    BDF.TEMPERATURE_T4_CELCIUS,
    BDF.TEMPERATURE_T5_CELCIUS,
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
        table = pq.read_table(self.path)

        if metadata_format == "parquet":
            existing: dict[bytes, bytes] = table.schema.metadata or {}
            combined_meta = {
                **existing,
                _PARQUET_METADATA_KEY: json.dumps(metadata).encode(),
            }
            table = table.replace_schema_metadata(combined_meta)
            pq.write_table(table, self.path)
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
    timezone: str | None = None,
) -> list[pl.DataFrame]:
    """Load raw cycler files into Polars DataFrames.

    Expands *source* via :func:`_resolve_glob`, then reads each file using
    ``batterydf``, optionally normalising to BDF column names.

    Args:
        source: A file path or glob pattern.
        plugin: BatteryDF plugin name. ``None`` triggers auto-detection.
        normalize: When ``True`` (default), normalise to BDF column names.
            When ``False``, preserve original source column names.
        timezone: Optional timezone (IANA string) to apply to tz-naive datetime
            columns in the raw data. Tz-aware columns are converted to UTC directly.
            Defaults to None (assumes UTC for tz-naive columns).

    Returns:
        One DataFrame per resolved file, in sorted order.
    """
    files = _resolve_glob(source)
    return [
        pl.from_pandas(
            bdf.read(str(f), plugin=plugin, normalize=normalize, timezone=timezone)
        )
        for f in files
    ]


def _concat_dataframes(dfs: list[pl.DataFrame]) -> pl.DataFrame:
    """Concatenate a list of DataFrames using diagonal (schema-union) mode.

    Args:
        dfs: DataFrames to concatenate. Columns need not be identical; missing
            columns are filled with ``null``.

    Returns:
        Single concatenated DataFrame.
    """
    return pl.concat(dfs, how="diagonal", rechunk=True)


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


def _extract_column_map_columns(
    df: pl.DataFrame,
    column_map: dict[str | BDF, str],
) -> pl.DataFrame:
    """Extract and rename columns from a DataFrame using a BDF column map.

    Args:
        df: Source DataFrame to extract columns from.
        column_map: Mapping from BDF-format output names (e.g. ``"Current / A"``
            or :attr:`BDF.CURRENT_AMPERE`) to source column names in *df*.

    Returns:
        A new DataFrame with columns renamed per *column_map*, containing only
        the mapped columns.

    Raises:
        ValueError: If an output name is not a valid BDF-format string, or if a
            source column name is not found in *df*.
    """
    return df.select(_build_column_map_exprs(df.columns, column_map))


def _resolve_time_column(column_set: ColumnSet) -> pl.Expr:
    """Resolve a time column, preferring Unix Time but falling back to Test Time.

    Attempts to resolve Unix Time first; if unavailable, falls back to Test Time.
    At least one of these must be resolvable.

    Args:
        column_set: ColumnSet with available columns.

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
    skip_if_exists: bool = True,
    compression_priority: Literal[
        "performance", "file size", "uncompressed"
    ] = "performance",
    column_map: dict[str | BDF, str] | None = None,
    timezone: str | None = None,
) -> Path:
    """Read cycler file(s), normalise to BDF columns, and write to Parquet.

    Reads one or more raw cycler files (via a file path or glob pattern),
    normalises columns to BDF standard using ``batterydf``, and writes the
    result to a ``.bdx.parquet`` file.

    Args:
        source: Path to the raw cycler file, or a glob pattern matching multiple
            files (e.g. ``"data/session_*.csv"``).
        output_path: Full destination path for the output Parquet file (must end
            with ``.parquet``). When ``None``, defaults to
            ``<source_parent>/<stem>.bdx.parquet`` where *stem* comes from
            *source* (or the first sorted glob match for glob patterns).
        plugin: BatteryDF plugin name for reading. ``None`` triggers auto-detection.
        skip_if_exists: When ``True`` (default), return the cached Parquet path
            immediately if it already exists without reprocessing raw data.
        compression_priority: Controls the Parquet compression algorithm:

            - ``"performance"`` (default) — uses ``lz4`` for fast read/write.
            - ``"file size"`` — uses ``zstd`` for smaller files.
            - ``"uncompressed"`` — no compression.

        column_map: Mapping from BDF-format output names (e.g. ``"Pressure / kPa"``)
            to source column names in the raw data. Keys must follow the
            ``"Quantity / unit"`` format. Where a key matches an already-resolved
            BDF column, the *column_map* entry overrides it.
        timezone: Optional timezone (IANA string) to apply to tz-naive datetime
            columns in the raw data. Tz-aware columns are converted to UTC directly.
            Defaults to None (assumes UTC for tz-naive columns).

    Returns:
        Path to the written ``.bdx.parquet`` file.

    Raises:
        FileNotFoundError: If *source* is a glob pattern that matches no files.
        ValueError: If *output_path* is provided but does not end with ``.parquet``.
        ValueError: If any time column (Unix Time or Test Time) cannot be resolved
            from the source data.
        ValueError: If any required BDF column (current, voltage) cannot be resolved
            from the source data.
        ValueError: If a *column_map* key does not follow the ``"Quantity / unit"``
            format.
        ValueError: If a *column_map* source column name is not present in the raw data.

    Example:
        Basic usage (writes ``data.bdx.parquet`` next to source)::

            path = process_cycler("data.xlsx")

        Output to a specific path::

            path = process_cycler("data.xlsx", output_path="cache/data.bdx.parquet")

        Override a resolved BDF column with a custom source column::

            path = process_cycler(
                "data.xlsx",
                column_map={"Ambient Pressure / kPa": "Pressure(kPa)"},
            )
    """
    first_file = _resolve_glob(source)[0]
    if output_path is not None:
        candidate = Path(output_path)
        if candidate.suffix == "":
            # Treat as a directory; auto-generate the filename within it.
            resolved_output_path = candidate / (first_file.stem + ".bdx.parquet")
        elif candidate.suffix != ".parquet":
            raise ValueError(
                f"output_path must end with '.parquet', got: '{output_path}'"
            )
        else:
            resolved_output_path = candidate
    else:
        resolved_output_path = first_file.parent / (first_file.stem + ".bdx.parquet")

    if skip_if_exists:
        cached = _handle_existing_cached_file(resolved_output_path)
        if cached is not None:
            return cached

    dfs = _load_raw_dataframes(source, plugin, timezone=timezone)
    df = _concat_dataframes(dfs)
    column_set = ColumnSet(df.columns)
    expressions: list[pl.Expr] = []

    # Resolve time column (Unix Time preferred, Test Time fallback)
    expressions.append(_resolve_time_column(column_set))

    for bdf_col in _REQUIRED_BDF_COLUMNS:
        try:
            expressions.append(column_set.resolve(bdf_col))
        except ValueError as exc:
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

    normalised: pl.DataFrame = df.select(expressions)

    if column_map is not None:
        raw_dfs = _load_raw_dataframes(
            source, plugin, normalize=False, timezone=timezone
        )
        raw_df = _concat_dataframes(raw_dfs)
        mapped = _extract_column_map_columns(raw_df, column_map)
        for col_name in mapped.columns:
            if col_name in normalised.columns:
                normalised = normalised.with_columns(mapped[col_name])
            else:
                normalised = normalised.hstack([mapped[col_name]])

    normalised.write_parquet(
        str(resolved_output_path),
        compression=_COMPRESSION_MAP[compression_priority],
    )
    logger.info("Wrote normalised data to '{}'.", resolved_output_path)
    return resolved_output_path


def read_metadata(
    path: str | Path,
    prefer: Literal["parquet", "json"] = "parquet",
) -> dict[str, Any]:
    r"""Read metadata from a Parquet file's footer or a ``.json`` sidecar.

    Checks both the Parquet footer (stored under \"bdx_metadata\") and a ``.json``
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
            meta = read_metadata("data.bdx.parquet")
            print(meta["cell_id"])  # 'C001'

            # Or prefer JSON sidecar if both exist
            meta = read_metadata("data.bdx.parquet", prefer="json")
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
    data: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
    column_map: dict[str | BDF, str],
    output_path: str | Path,
    compression_priority: Literal[
        "performance", "file size", "uncompressed"
    ] = "performance",
) -> Path:
    """Normalise an arbitrary DataFrame to BDF format and write to Parquet.

    Accepts a polars DataFrame, polars LazyFrame, or pandas DataFrame, renames
    columns per *column_map* (mapping BDF output name to source column name),
    validates that required BDF columns are resolvable, and writes all mapped
    columns to *output_path*.

    Args:
        data: Raw battery data. Accepts a polars DataFrame, polars LazyFrame,
            or pandas DataFrame.
        column_map: Mapping from BDF-format output name (e.g. ``"Current / A"``)
            to the source column name in *data*.
        output_path: Destination path for the output Parquet file.
        compression_priority: Compression algorithm selection.

    Returns:
        The resolved path of the written Parquet file.

    Raises:
        TypeError: If *data* cannot be converted to a Polars DataFrame.
        ValueError: If any required BDF column cannot be resolved after
            applying *column_map*.
    """
    output = Path(output_path)
    compression = _COMPRESSION_MAP[compression_priority]

    # Normalize input: convert to LazyFrame, tracking original type for output method
    is_lazy = isinstance(data, pl.LazyFrame)
    if not is_lazy:
        if not isinstance(data, pl.DataFrame):
            try:
                data = pl.from_pandas(data)
            except Exception as exc:
                raise TypeError(
                    f"Could not convert data to a Polars DataFrame: {exc}"
                ) from exc
        data = data.lazy()

    # Build and apply column map expressions
    exprs = _build_column_map_exprs(data.collect_schema().names(), column_map)
    output_columns = [str(e.meta.output_name()) for e in exprs]
    column_set = ColumnSet(output_columns)

    # Validate required BDF columns
    for bdf_col in _REQUIRED_BDF_COLUMNS:
        try:
            column_set.resolve(bdf_col)
        except ValueError as exc:
            raise ValueError(
                f"Required BDF column '{bdf_col.quantity}' could not be resolved "
                f"from the data: {exc}"
            ) from exc

    # Select mapped columns and write (method depends on original type)
    selected = data.select(exprs)
    if is_lazy:
        selected.sink_parquet(str(output), compression=compression)
    else:
        selected.collect().write_parquet(str(output), compression=compression)

    logger.info("Wrote generic data to '{}'.", output)
    return output
