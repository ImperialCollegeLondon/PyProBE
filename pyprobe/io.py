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
from typing import Any, Literal

import bdf
import polars as pl
import pyarrow.parquet as pq
from loguru import logger

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


def process_cycler(
    source: str | Path,
    output_dir: str | Path | None = None,
    metadata: dict[str, str | int | float | bool] | None = None,
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
        metadata: Optional key-value pairs to attach to the output. Values may
            be strings, ints, floats, or bools. Ignored when *write_parquet*
            is ``False``.
        plugin: Optional ``batterydf`` plugin name passed to ``bdf.read()``.
            When ``None`` the plugin is auto-detected.
        write_parquet: When ``True`` (default), write the normalised data to
            a Parquet file and return a lazy scan. When ``False``, return an
            in-memory LazyFrame without writing.
        skip_if_exists: When ``True`` (default) and the output file already
            exists, skip processing and return the cached file immediately.
            Only applies when *write_parquet* is ``True``.
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
        if skip_if_exists and output_path.exists():
            logger.info("Skipping processing; using cached file '{}'.", output_path)
            return pl.scan_parquet(output_path)

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
        raw_df = pl.from_pandas(bdf.read(source, plugin=plugin, normalize=False))
        for output_name, source_name in extra_columns.items():
            Column.from_string(output_name)  # validate BDF format
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
    metadata: dict[str, str | int | float | bool] | None = None,
    *,
    metadata_format: Literal["json", "parquet"] = "parquet",
) -> None:
    """Write a Polars DataFrame to Parquet, embedding optional metadata.

    Converts *df* to an Arrow table and writes via
    :func:`pyarrow.parquet.write_table`. When *metadata* is provided, it is
    stored according to *metadata_format*: embedded in the Parquet footer
    (``"parquet"``) or written to a ``.json`` sidecar file (``"json"``).

    Args:
        df: The DataFrame to persist.
        path: Destination file path. Parent directories must already exist.
        metadata: Optional key-value pairs to attach. Values may be strings,
            ints, floats, or bools. When *metadata_format* is ``"parquet"``,
            all values are converted to strings before embedding.
        metadata_format: ``"parquet"`` (default) embeds metadata in the Parquet
            footer. ``"json"`` writes metadata to a ``.json`` sidecar and does
            not embed anything in the footer.
    """
    table = df.to_arrow()
    if metadata:
        if metadata_format == "parquet":
            existing: dict[bytes, bytes] = table.schema.metadata or {}
            encoded: dict[bytes, bytes] = {
                k.encode(): str(v).encode() for k, v in metadata.items()
            }
            table = table.replace_schema_metadata({**existing, **encoded})
        else:
            sidecar_path = path.with_suffix(".json")
            sidecar_path.write_text(json.dumps(metadata, indent=2))
    pq.write_table(table, path)


def read_parquet_metadata(path: str | Path) -> dict[str, str]:
    """Read key-value metadata from a Parquet file's footer.

    Args:
        path: Path to the Parquet file.

    Returns:
        A dictionary of metadata key-value pairs decoded from UTF-8.
        Returns an empty dict if the file has no metadata.

    Examples:
        >>> import tempfile, pathlib, polars as pl
        >>> from pyprobe.io import _write_parquet, read_parquet_metadata
        >>> df = pl.DataFrame({"x": [1, 2, 3]})
        >>> with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        ...     tmp = pathlib.Path(f.name)
        >>> _write_parquet(df, tmp, {"cell_id": "C001", "cycler": "neware"})
        >>> meta = read_parquet_metadata(tmp)
        >>> meta["cell_id"]
        'C001'
        >>> meta["cycler"]
        'neware'
        >>> tmp.unlink()
    """
    pf = pq.ParquetFile(path)
    raw: dict[bytes, bytes] = pf.schema_arrow.metadata or {}
    return {k.decode(): v.decode() for k, v in raw.items()}


def read_metadata(
    path: str | Path,
    prefer: Literal["parquet", "json"] = "parquet",
) -> dict[str, str]:
    """Read metadata from a Parquet file's footer or a ``.json`` sidecar.

    Checks both the Parquet footer and a ``.json`` sidecar (derived from
    *path* by replacing the ``.parquet`` suffix with ``.json``). When both
    sources contain metadata, *prefer* controls which is returned. When only
    one source has metadata, that source is returned regardless of *prefer*.
    When neither has metadata, an empty dict is returned.

    Args:
        path: Path to the Parquet file.
        prefer: Which source to return when both exist. ``"parquet"`` (default)
            returns the Parquet footer metadata; ``"json"`` returns the sidecar
            metadata.

    Returns:
        A dictionary of metadata key-value pairs. Values from the Parquet
        footer are always strings (decoded UTF-8). Values from the JSON sidecar
        are returned as strings via JSON decoding.

    Raises:
        ValueError: If *prefer* is not ``"parquet"`` or ``"json"``.

    Examples:
        >>> import tempfile, pathlib, polars as pl
        >>> from pyprobe.io import _write_parquet, read_metadata
        >>> df = pl.DataFrame({"x": [1, 2, 3]})
        >>> with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        ...     tmp = pathlib.Path(f.name)
        >>> _write_parquet(df, tmp, {"cell_id": "C001"}, metadata_format="parquet")
        >>> read_metadata(tmp)
        {'cell_id': 'C001'}
        >>> tmp.unlink()
    """
    if prefer not in ("parquet", "json"):
        raise ValueError(f"prefer must be 'parquet' or 'json', got '{prefer}'.")

    parquet_path = Path(path)
    json_path = parquet_path.with_suffix(".json")

    parquet_meta: dict[str, str] = read_parquet_metadata(parquet_path)
    # Strip Arrow/Polars internal keys so only user metadata remains.
    parquet_meta = {k: v for k, v in parquet_meta.items() if not k.startswith("pandas")}

    json_meta: dict[str, str] = {}
    if json_path.exists():
        raw: Any = json.loads(json_path.read_text())
        if isinstance(raw, dict):
            json_meta = {str(k): str(v) for k, v in raw.items()}

    has_parquet = bool(parquet_meta)
    has_json = bool(json_meta)

    if has_parquet and has_json:
        return parquet_meta if prefer == "parquet" else json_meta
    if has_parquet:
        return parquet_meta
    if has_json:
        return json_meta
    return {}
