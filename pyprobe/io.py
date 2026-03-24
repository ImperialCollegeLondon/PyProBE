"""BDF-based cycler data import utilities for PyProBE.

Provides :func:`process_cycler` as the primary entry point for reading raw
cycler files via the ``batterydf`` package, normalising them to BDF-standard
column names, and persisting to Parquet with attached metadata.

Typical usage::

    from pyprobe.io import process_cycler

    lf = process_cycler("path/to/data.xlsx")
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import bdf
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

if TYPE_CHECKING:
    from pyprobe.filters import Procedure

from pyprobe.column import (
    BDFColumn,
    Column,
    ColumnSet,
    current_ampere,
    net_capacity_ah,
    step_count,
    step_index,
    test_time_second,
    voltage_volt,
)

_PARQUET_METADATA_KEY: bytes = b"bdx_metadata"
"""Key used to store user metadata in Parquet footer."""

_REQUIRED_BDF_COLUMNS: list[BDFColumn] = [
    test_time_second,
    current_ampere,
    voltage_volt,
]
"""BDF columns that must be resolvable; :func:`process_cycler` raises if not."""

_OPTIONAL_BDF_COLUMNS: list[BDFColumn] = [
    net_capacity_ah,
    step_count,
    step_index,
]
"""BDF columns included when available; warnings are emitted on failure."""


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
            Dictionary of metadata, or empty dict if missing or unreadable.
        """
        try:
            pf = pq.ParquetFile(self.path)
            raw: dict[bytes, bytes] = pf.schema_arrow.metadata or {}
            if _PARQUET_METADATA_KEY not in raw:
                return {}
            return json.loads(raw[_PARQUET_METADATA_KEY].decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning(
                "Failed to decode metadata from '{}': {}. Returning empty metadata.",
                self.path,
                exc,
            )
            return {}

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
        """Read metadata from both sources with preference logic.

        When both sources have metadata, *prefer* controls which is returned.
        When only one source has metadata, that source is returned regardless
        of *prefer*. When neither has metadata, an empty dict is returned.

        Args:
            prefer: Which source to prefer when both exist.

        Returns:
            Dictionary of metadata from the preferred source, or the only
            source that has metadata, or an empty dict.
        """
        parquet_meta = self.read_parquet()
        json_meta = self.read_json()

        has_parquet = bool(parquet_meta)
        has_json = bool(json_meta)

        if has_parquet and has_json:
            return parquet_meta if prefer == "parquet" else json_meta
        if has_parquet:
            return parquet_meta
        if has_json:
            return json_meta
        return {}

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


def _handle_existing_cached_file(
    output_path: Path,
    metadata: dict[str, Any] | None,
    metadata_format: Literal["parquet", "json"],
) -> pl.LazyFrame | None:
    """Handle skip_if_exists logic for cached cycler output files.

    Checks if a cached file exists and determines whether to use it or
    reprocess. If the file exists and no metadata update is needed, returns
    a lazy scan of the cached file. If metadata needs updating, updates it
    in-place without reprocessing raw data.

    Args:
        output_path: Path to the cached Parquet file.
        metadata: Optional metadata to apply to the cached file. If provided
            and differs from existing metadata, the cached file is updated
            (without reprocessing raw data).
        metadata_format: Format for storing/reading metadata.

    Returns:
        A LazyFrame scanning the cached file if it exists and no reprocessing
        is needed, or None if the file does not exist or reprocessing is required.
    """
    if not output_path.exists():
        return None

    if metadata:
        manager = MetadataManager(output_path)
        existing_metadata = manager.read(metadata_format=metadata_format)
        needs_update = any(
            existing_metadata.get(str(k)) != v for k, v in metadata.items()
        )
        if needs_update:
            logger.info(
                "Updating metadata on cached file '{}' without reprocessing raw data.",
                output_path,
            )
            manager.update(
                metadata,
                metadata_format=metadata_format,
            )

    logger.info("Skipping processing; using cached file '{}'.", output_path)
    return pl.scan_parquet(output_path)


def process_cycler(
    source: str | Path,
    output_dir: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
    *,
    plugin: str | None = None,
    write_parquet: bool = True,
    skip_if_exists: bool = True,
    metadata_format: Literal["json", "parquet"] = "parquet",
    extra_columns: dict[str, str] | None = None,
) -> pl.LazyFrame:
    """Read a cycler file, normalise to BDF columns, and optionally cache.

    By default the normalised data is written as ``{source_stem}.bdx.parquet``
    in *output_dir* (defaulting to the same directory as *source*). Set
    *write_parquet* to ``False`` to skip file writing and return an in-memory
    LazyFrame instead.

    Args:
        source: Path to the raw cycler file (any format supported by
            ``batterydf``).
        output_dir: Directory for the output Parquet file. Defaults to the
            parent directory of *source*. Ignored when *write_parquet* is
            ``False``.
        metadata: Optional JSON-serializable key-value pairs to attach to the
            output. Ignored when *write_parquet* is ``False``.
        plugin: Optional BatteryDF plugin name to use for reading the file.
            If ``None`` (default), BatteryDF auto-detects the format.
        write_parquet: When ``True`` (default), write the normalised data to
            a Parquet file and return a lazy scan. When ``False``, return an
            in-memory LazyFrame without writing.
        skip_if_exists: When ``True`` (default) and the output file already
            exists, skip processing and return the cached file immediately.
            If *metadata* is provided, requested keys are still written to the
            cached output (without re-reading raw cycler data) when values are
            missing or stale. Only applies when *write_parquet* is ``True``.
        metadata_format: Controls how *metadata* is stored. ``"parquet"``
            (default) embeds metadata in the Parquet footer. ``"json"`` writes
            a ``.json`` sidecar file instead and does **not** embed metadata in
            the Parquet footer. Ignored when *write_parquet* is ``False`` or
            *metadata* is ``None``.
        extra_columns: Optional mapping of BDF-format output names to source
            column names. These columns are read from the raw source file
            (before BDF normalisation) and appended to the output. Keys must
            follow the ``"Quantity / unit"`` format. Example::

                {"Pressure / kPa": "Pressure(kPa)", "Aux Temp / degC": "T_aux[C]"}

    Returns:
        A :class:`polars.LazyFrame` over the normalised BDF columns.

    Raises:
        ValueError: If any required BDF column (test time, current, voltage)
            cannot be resolved from the source data.
        ValueError: If *metadata_format* is not ``"parquet"`` or ``"json"``.

    Examples:
        Basic usage (writes ``data.bdx.parquet`` next to source)::

            lf = process_cycler("data.xlsx")

        Without writing to disk::

            lf = process_cycler("data.xlsx", write_parquet=False)

        Output to a different directory with metadata::

            lf = process_cycler(
                "data.xlsx",
                output_dir="cache/",
                metadata={"cell_id": "C001"},
            )

        With extra columns from the raw file::

            lf = process_cycler(
                "data.xlsx",
                extra_columns={"Pressure / kPa": "Pressure(kPa)"},
            )
    """
    if metadata_format not in ("parquet", "json"):
        raise ValueError(
            f"metadata_format must be 'parquet' or 'json', got '{metadata_format}'."
        )

    source_path = Path(source)
    output_path: Path | None = None
    if write_parquet:
        if output_dir is None:
            output_dir = source_path.parent
        output_path = Path(output_dir) / (source_path.stem + ".bdx.parquet")
        if skip_if_exists:
            cached = _handle_existing_cached_file(
                output_path, metadata, metadata_format
            )
            if cached is not None:
                return cached

    logger.info("Reading cycler file '{}'.", source)
    pandas_df = bdf.read(source, plugin=plugin)
    df: pl.DataFrame = pl.from_pandas(pandas_df)

    column_set = ColumnSet(df.columns)
    expressions: list[pl.Expr] = []

    for bdf_col in _REQUIRED_BDF_COLUMNS:
        try:
            expressions.append(column_set.col(bdf_col))
        except ValueError as exc:
            raise ValueError(
                f"Required BDF column '{bdf_col.quantity}' could not be resolved "
                f"from the source data: {exc}"
            ) from exc

    for bdf_col in _OPTIONAL_BDF_COLUMNS:
        try:
            expressions.append(column_set.col(bdf_col))
        except ValueError:
            logger.warning(
                "Optional BDF column '{}' could not be resolved; skipping.",
                bdf_col.quantity,
            )

    normalised: pl.DataFrame = df.select(expressions)

    if extra_columns:
        # Validate all output_name formats upfront before reading raw data.
        # Valid: "Channel", "Pressure / kPa", "Flow Rate / mL/min"
        # Invalid: "InvalidNoUnit", "Pressure kPa", "/ kPa", "", "Quantity //"
        strict_pattern = r"^(.+?)\s*/\s*([^/]+(?:/[^/]+)*)$"
        for output_name in extra_columns:
            Column.from_string(output_name, pattern=strict_pattern)

        # Dual bdf.read() calls are necessary: the initial read (above) normalizes
        # column names to BDF standard, while this read with normalize=False
        # accesses original source column names for extra_columns mapping.
        # This should be improved in future
        # versions to provide a single-pass read supporting both operations.
        raw_df = pl.from_pandas(bdf.read(source, plugin=plugin, normalize=False))
        for output_name, source_name in extra_columns.items():
            if source_name not in raw_df.columns:
                raise ValueError(
                    f"Extra column source '{source_name}' not found in data. "
                    f"Available: {raw_df.columns}"
                )
        normalised = normalised.hstack(
            [
                raw_df[source_name].alias(output_name)
                for output_name, source_name in extra_columns.items()
            ]
        )

    if output_path is not None:
        _write_parquet(
            normalised, output_path, metadata, metadata_format=metadata_format
        )
        logger.info("Wrote normalised data to '{}'.", output_path)
        return pl.scan_parquet(output_path)

    return normalised.lazy()


def _write_parquet(
    df: pl.DataFrame,
    path: Path,
    metadata: dict[str, Any] | None = None,
    *,
    metadata_format: Literal["json", "parquet"] = "parquet",
) -> None:
    """Write a Polars DataFrame to Parquet, embedding optional metadata.

    Converts *df* to an Arrow table and delegates to MetadataManager
    for consistent metadata handling across parquet footer and JSON sidecar
    formats.

    Args:
        df: The DataFrame to persist.
        path: Destination file path. Parent directories must already exist.
        metadata: Optional JSON-serializable key-value pairs to attach.
        metadata_format: ``"parquet"`` (default) embeds metadata in the
            Parquet footer. ``"json"`` writes a ``.json`` sidecar instead.
    """
    table = df.to_arrow()
    MetadataManager.create(table, path, metadata, metadata_format)


def read_parquet_metadata(path: str | Path) -> dict[str, Any]:
    """Read key-value metadata from a Parquet file's footer.

    Reads metadata from the "bdx_metadata" key in the Parquet footer, which
    stores a JSON-encoded object of all user metadata.

    Args:
        path: Path to the Parquet file.

    Returns:
        A dictionary of metadata key-value pairs. Returns an empty dict if
        the file has no metadata or the "bdx_metadata" key is missing.

    Example:
        Retrieve metadata from a cached battery parquet file::

            from pyprobe.io import read_parquet_metadata

            meta = read_parquet_metadata("data.bdx.parquet")
            print(meta["cell_id"])       # 'C001'
            print(meta["cycler"])        # 'neware'
    """
    manager = MetadataManager(Path(path))
    return manager.read_parquet()


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


def create_procedure_from_parquet(
    parquet_path: str | Path,
    readme_path: str | Path | None = None,
    metadata: dict[str, Any | None] | None = None,
    metadata_prefer: Literal["parquet", "json"] = "parquet",
) -> "Procedure":
    """Create a Procedure from a processed cycler parquet file with metadata.

    Loads metadata from the parquet footer or sidecar JSON, and optionally reads
    experiment definitions from a README.yaml file.

    Args:
        parquet_path: Path to the output parquet file (e.g., from process_cycler).
        readme_path: Optional path to README.yaml for experiment definitions.
            When None, an empty experiment dict is used.
        metadata: Optional metadata dictionary to include. Merged with metadata
            from parquet source. Defaults to empty dict.
        metadata_prefer: Whether to prefer parquet footer or JSON sidecar metadata
            when both exist. Defaults to "parquet".

    Returns:
        A Procedure object with loaded data, metadata, and experiment definitions.

    Raises:
        FileNotFoundError: If parquet file does not exist.
        ValueError: If README exists but fails to parse.

    Example:
        Load a processed battery parquet file and optionally attach experiment
        definitions from a README::

            from pyprobe.io import create_procedure_from_parquet

            # Load parquet with metadata from footer
            procedure = create_procedure_from_parquet("data.bdx.parquet")

            # Include experiment definitions from README.yaml
            procedure = create_procedure_from_parquet(
                "data.bdx.parquet",
                readme_path="experiments.yaml",
                metadata={"cell_id": "Cell1"},
            )
    """
    from pyprobe.filters import Procedure
    from pyprobe.readme_processor import process_readme

    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    lf = pl.scan_parquet(parquet_path)
    parquet_metadata = read_metadata(parquet_path, prefer=metadata_prefer)

    # Merge provided metadata with parquet metadata (provided takes precedence)
    merged_metadata = {**parquet_metadata, **(metadata or {})}

    readme_dict: dict[str, dict[str, Any]] = {}
    if readme_path is not None:
        readme_path = Path(readme_path)
        if readme_path.exists():
            readme_obj = process_readme(str(readme_path))
            readme_dict = readme_obj.experiment_dict
        else:
            logger.warning("README path provided but not found: {}", readme_path)

    return Procedure(
        lf=lf,
        metadata=merged_metadata,
        readme_dict=readme_dict,
    )
