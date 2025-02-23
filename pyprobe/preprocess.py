"""Module for preprocessing battery cycler data into PyProBE format."""

import logging
from typing import Literal

from pyprobe.cyclers import (
    arbin,
    basecycler,
    basytec,
    biologic,
    maccor,
    neware,
)
from pyprobe.utils import catch_pydantic_validation

logger = logging.getLogger(__name__)


@catch_pydantic_validation
def process_cycler_data(
    cycler_type: Literal[
        "neware", "biologic", "biologic_MB", "arbin", "basytec", "maccor", "generic"
    ],
    input_data_path: str,
    output_data_path: str | None = None,
    column_importers: list[basecycler.ColumnMap] = [],
    compression_priority: Literal[
        "performance", "file size", "uncompressed"
    ] = "performance",
    overwrite_existing: bool = False,
) -> None:
    """Process battery cycler data into PyProBE format.

    Args:
        cycler_type: Type of battery cycler used.
        input_data_path: Path to input data file(s). Supports glob patterns.
        output_data_path: Path for output parquet file. If None, the output file will
            have the same name as the input file with a .parquet extension.
        column_importers:
            List of column importers to apply to the input data. Required for generic
            cycler type. Overrides default column importers for other cycler types.
        compression_priority: Compression method for output file.
        overwrite_existing: Whether to overwrite existing output file.
    """
    cycler_map = {
        "neware": neware.Neware,
        "biologic": biologic.Biologic,
        "biologic_MB": biologic.BiologicMB,
        "arbin": arbin.Arbin,
        "basytec": basytec.Basytec,
        "maccor": maccor.Maccor,
        "generic": basecycler.BaseCycler,
    }

    cycler_class = cycler_map.get(cycler_type)
    if not cycler_class:
        msg = f"Unsupported cycler type: {cycler_type}"
        logger.error(msg)
        raise ValueError(msg)

    if cycler_type == "generic" and column_importers == []:
        msg = "Column importers must be provided for generic cycler type."
        logger.error(msg)
        raise ValueError(msg)

    if column_importers != []:
        processor = cycler_class(
            input_data_path=input_data_path,
            output_data_path=output_data_path,
            compression=compression_priority,
            overwrite_existing=overwrite_existing,
            column_importers=column_importers,
        )
    else:
        processor = cycler_class(
            input_data_path=input_data_path,
            output_data_path=output_data_path,
            compression=compression_priority,
            overwrite_existing=overwrite_existing,
        )
    processor.process()
