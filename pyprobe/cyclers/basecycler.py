"""A module to load and process battery cycler data."""

import glob
import os
import time
from pathlib import Path
from typing import Literal

import polars as pl
from loguru import logger
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

from pyprobe.cyclers.column_maps import ColumnMap


class BaseCycler(BaseModel):
    """A class to load and process battery cycler data."""

    input_data_path: str = Field(
        description="Path to input data file(s). Supports glob patterns."
    )
    output_data_path: str | None = Field(
        default=None,
        description="Path for output parquet file. Defaults to the input path with a"
        " .parquet suffix.",
    )
    compression_priority: Literal["performance", "file size", "uncompressed"] = Field(
        default="performance", description="Compression algorithm for output file"
    )
    overwrite_existing: bool = Field(
        default=False, description="Whether to overwrite existing output file"
    )
    header_row_index: int = Field(
        default=0, description="Index of header row in input file"
    )
    column_importers: list[ColumnMap]

    extra_column_importers: list[ColumnMap] = Field(
        default_factory=list,
        description="Additional column importers to be added to the cycler",
    )

    @field_validator("input_data_path", mode="after")
    @classmethod
    def validate_input_path(cls, v: str) -> str:
        """Validate that the input path exists."""
        if "*" in v:
            files = glob.glob(v)
            if not files:
                raise ValueError(f"No files found matching pattern: {v}")
            return v
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Input file not found: {path}")
        return str(path)

    @model_validator(mode="after")
    def validate_output_path(self) -> "BaseCycler":
        """Set the default output path if not provided."""
        if self.output_data_path is not None:
            path = Path(self.output_data_path)
            if path.suffix != ".parquet":
                if path.suffix:
                    logger.warning(
                        f"Output file extension {path.suffix} will be replaced with"
                        " .parquet"
                    )
                else:
                    logger.info("Output file has no extension, will be given .parquet")
                self.output_data_path = str(path.with_suffix(".parquet"))
            if "*" in self.output_data_path:
                raise ValueError("Output path cannot contain wildcards.")
            if not path.parent.exists():
                raise ValueError(f"Output directory does not exist: {path.parent}")
        else:
            input_path = Path(self.input_data_path)
            self.output_data_path = str(input_path.with_suffix(".parquet")).replace(
                "*", "x"
            )
            logger.info(
                f"Output path not provided, defaulting to: {self.output_data_path}"
            )
        return self

    @model_validator(mode="after")
    def import_and_validate_data(self) -> "BaseCycler":
        """Import the data and validate the column mapping."""
        if not os.path.exists(str(self.output_data_path)) or self.overwrite_existing:
            dataframe_list = self._get_dataframe_list()
            self._imported_dataframe = self.get_imported_dataframe(dataframe_list)
            if len(self.extra_column_importers) > 0:
                self.column_importers += self.extra_column_importers
            with logger.contextualize(
                input_data_path=self.input_data_path,
            ):
                for column_importer in self.column_importers:
                    column_importer.validate(
                        self._imported_dataframe.collect_schema().names()
                    )
        return self

    def _get_dataframe_list(self) -> list[pl.DataFrame | pl.LazyFrame]:
        """Return a list of all the imported dataframes.

        Args:
            input_data_path (str): The path to the input data.

        Returns:
            List[DataFrame]: A list of DataFrames.
        """
        files = glob.glob(self.input_data_path)
        files.sort()
        df_list = [self.read_file(file, self.header_row_index) for file in files]
        all_columns = {col for df in df_list for col in df.collect_schema().names()}
        for i in range(len(df_list)):
            if len(df_list[i].collect_schema().names()) < len(all_columns):
                logger.warning(
                    f"File {os.path.basename(files[i])} has missing columns, "
                    "these have been filled with null values.",
                )
        return df_list

    def get_imported_dataframe(
        self,
        dataframe_list: list[pl.DataFrame],
    ) -> pl.DataFrame:
        """Return a single DataFrame from a list of DataFrames.

        Args:
            dataframe_list: A list of DataFrames.

        Returns:
            DataFrame: A single DataFrame.
        """
        return pl.concat(dataframe_list, how="diagonal", rechunk=True)

    @staticmethod
    def read_file(
        filepath: str,
        header_row_index: int = 0,
    ) -> pl.DataFrame | pl.LazyFrame:
        """Read a battery cycler file into a DataFrame.

        Args:
            filepath: The path to the file.
            header_row_index: The index of the header row.
            header_row_index: The index of the header row.

        Returns:
            pl.DataFrame | pl.LazyFrame: The DataFrame.
        """
        file = os.path.basename(filepath)
        file_ext = os.path.splitext(file)[1]
        match file_ext.lower():
            case ".xlsx":
                return pl.read_excel(
                    filepath,
                    engine="calamine",
                    infer_schema_length=0,
                    read_options={"header_row": header_row_index},
                )
            case ".csv":
                return pl.scan_csv(
                    filepath,
                    infer_schema=False,
                    skip_rows=header_row_index,
                )
            case _:
                error_msg = f"Unsupported file extension: {file_ext}"
                logger.error(error_msg)
                raise ValueError(error_msg)

    @staticmethod
    def event_expr() -> pl.Expr:
        """Return the event expression."""
        return (
            pl.col("Step")
            .diff()
            .fill_null(0)
            .ne(0)
            .cum_sum()
            .cast(pl.UInt64)
            .alias("Event")
        )

    def get_pyprobe_dataframe(self) -> pl.DataFrame:
        """Return the PyProBE DataFrame."""
        imported_columns = set()
        importers: list[ColumnMap] = []
        for importer in self.column_importers:
            if (
                importer.columns_validated
                and importer.pyprobe_name not in imported_columns
            ):
                importers.append(importer.expr)
                imported_columns.add(importer.pyprobe_name)
        return (
            self._imported_dataframe.select(importers)
            .with_columns(self.event_expr())
            .collect()
        )

    def process(self) -> None:
        """Process the battery cycler data."""
        compression_dict = {
            "uncompressed": "uncompressed",
            "performance": "lz4",
            "file size": "zstd",
        }
        if not os.path.exists(str(self.output_data_path)) or self.overwrite_existing:
            t1 = time.time()
            pyprobe_dataframe = self.get_pyprobe_dataframe()
            pyprobe_dataframe.write_parquet(
                self.output_data_path,
                compression=compression_dict[self.compression_priority],
            )
            logger.info(f"parquet written in{time.time() - t1: .2f} seconds.")
        else:
            logger.info(f"File {self.output_data_path} already exists. Skipping.")
