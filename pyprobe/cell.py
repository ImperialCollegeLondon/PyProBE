"""Module for the Cell class."""

import json
import os
import shutil
import warnings
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import bdf
import polars as pl
from loguru import logger

from pyprobe._version import __version__
from pyprobe.filters import Procedure
from pyprobe.utils import PyBaMMSolution, deprecated


@dataclass
class Cell:
    """A class for a cell in a battery experiment."""

    procedure: dict[str, Procedure] = field(default_factory=dict)
    """Dictionary containing the procedures that have been run on the cell."""

    def add_procedure(
        self,
        procedure_name: str,
        source: str | Path | pl.LazyFrame | pl.DataFrame | Procedure,
    ) -> None:
        """Add a procedure to the cell.

        Loads *source* as a :class:`~pyprobe.filters.Procedure` and stores it
        under *procedure_name*. When *source* is already a
        :class:`~pyprobe.filters.Procedure` it is stored directly; otherwise
        it is passed to :meth:`~pyprobe.filters.Procedure.load`.

        *source* must contain BDF-compatible columns — at minimum a time column
        (``"Test Time / s"`` or ``"Unix Time / s"``), ``"Current / A"``, and
        ``"Voltage / V"``. Any file or DataFrame with these columns is accepted,
        regardless of origin. Use :func:`~pyprobe.io.process_cycler` to convert
        raw cycler files to BDF format, or :func:`~pyprobe.io.process_generic`
        with a column map to convert arbitrary DataFrames.

        Args:
            procedure_name: Key under which the procedure is stored in
                ``self.procedure``.
            source: A :class:`~pyprobe.filters.Procedure`, a path to a
                ``.parquet`` or ``.csv`` file, a :class:`~polars.LazyFrame`,
                or a :class:`~polars.DataFrame`. Must have BDF-compatible columns.

        Raises:
            ValueError: If *source* lacks required BDF columns (time, current,
                voltage).
        """
        if isinstance(source, Procedure):
            self.procedure[procedure_name] = source
        else:
            self.procedure[procedure_name] = Procedure.load(source)

    @deprecated(
        reason="Use :func:`~pyprobe.io.process_generic` with a column map to convert "
        "PyBaMM outputs to BDF format, then load via "
        ":meth:`~pyprobe.filters.Procedure.load`.",
        version="2.5.0",
        plain_reason="Cell.import_pybamm_solution() is deprecated. "
        "Use pyprobe.io.process_generic() with a column map to convert PyBaMM "
        "outputs to BDF format, then load via Procedure.load().",
    )
    def import_pybamm_solution(
        self,
        procedure_name: str,
        experiment_names: list[str] | str,
        pybamm_solutions: list[PyBaMMSolution] | PyBaMMSolution,
        output_data_path: str | None = None,
        optional_variables: list[str] | None = None,
    ) -> None:
        """Import a PyBaMM solution object into a procedure of the cell.

        .. deprecated::
            Use :func:`~pyprobe.io.process_generic` with a column map to convert
            PyBaMM outputs to BDF format, then load via
            :meth:`~pyprobe.filters.Procedure.load`.

        Args:
            procedure_name (str):
                A name to give the procedure. This will be used when calling
                :code:`cell.procedure[procedure_name]`.
            pybamm_solutions (list or pybamm_solution):
                A list of PyBaMM solution objects or a single PyBaMM solution object.
            experiment_names (list or str):
                A list of experiment names or a single experiment name to assign to the
                PyBaMM solution object.
            output_data_path (str, optional):
                The path to write the parquet file. Defaults to None.
            optional_variables (list, optional):
                A list of variables to import from the PyBaMM solution object in
                addition to the PyProBE required variables. Defaults to None.
        """
        # the minimum required variables to import from the PyBaMM solution object
        required_variables = [
            "Time [s]",
            "Current [A]",
            "Terminal voltage [V]",
            "Discharge capacity [A.h]",
        ]

        # get the list of variables to import from the PyBaMM solution object
        if optional_variables is not None:
            import_variables = required_variables + optional_variables
        else:
            import_variables = required_variables

        # Ensure pybamm_solutions is a list
        if not isinstance(pybamm_solutions, list):
            pybamm_solutions = [pybamm_solutions]

        # Ensure experiment_names is a list
        if not isinstance(experiment_names, list):
            experiment_names = [experiment_names]

        # Check if the lengths of experiment_names and pybamm_solutions match
        if len(experiment_names) != len(pybamm_solutions):
            error_msg = (
                "The number of experiment names and PyBaMM solutions must be equal."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        lazyframe_created = False
        for experiment_name, pybamm_solution in zip(
            experiment_names,
            pybamm_solutions,
            strict=False,
        ):
            # get the data from the PyBaMM solution object
            pybamm_data = pybamm_solution.get_data_dict(import_variables)
            # convert the PyBaMM data to a polars dataframe and add the experiment name
            # as a column
            solution_data = pl.LazyFrame(pybamm_data).with_columns(
                pl.lit(experiment_name).alias("Experiment"),
            )
            if lazyframe_created is False:
                all_solution_data = solution_data
                lazyframe_created = True
            else:
                # join the new solution data with the existing solution data, a right
                # join is used to keep all the data
                all_solution_data = all_solution_data.join(
                    solution_data,
                    on=import_variables + ["Step"],
                    how="right",
                )
                # fill null values where the experiment has been extended with the newly
                #  joined experiment name
                all_solution_data = all_solution_data.with_columns(
                    pl.col("Experiment").fill_null(pl.col("Experiment_right")),
                )
        # get the maximum step number for each experiment
        max_steps = (
            all_solution_data.group_by("Experiment")
            .agg(pl.max("Step").alias("Max Step"))
            .sort("Experiment")
            .with_columns(pl.col("Max Step").cum_sum().shift())
        )
        # add the maximum step number from the previous experiment to the step number
        all_solution_data = all_solution_data.join(
            max_steps,
            on="Experiment",
            how="left",
        ).with_columns(
            (pl.col("Step") + pl.col("Max Step").fill_null(-1) + 1).alias("Step"),
        )
        # get the range of step values for each experiment
        step_ranges = all_solution_data.group_by("Experiment").agg(
            pl.arange(pl.col("Step").min(), pl.col("Step").max() + 1).alias(
                "Step Range",
            ),
        )

        # create a dictionary of the experiment names and the step ranges
        experiment_dict = {}
        for row in step_ranges.collect().iter_rows():
            experiment = row[0]
            experiment_dict[experiment] = {"Steps": row[1]}
            experiment_dict[experiment]["Step Descriptions"] = []

        # reformat the data to the PyProBE format
        lf = all_solution_data.select(
            [
                pl.col("Time [s]").alias("Test Time / s"),
                (pl.col("Current [A]") * -1).alias("Current / A"),
                pl.col("Terminal voltage [V]").alias("Voltage / V"),
                (pl.col("Discharge capacity [A.h]") * -1).alias("Net Capacity / Ah"),
                pl.col("Step").alias("Step ID"),
                (
                    (
                        pl.col("Step").cast(pl.Int64)
                        - pl.col("Step").cast(pl.Int64).shift()
                        != 0
                    )
                    .fill_null(strategy="zero")
                    .cum_sum()
                    .alias("Step Count / 1")
                ),
            ],
        )
        self.procedure[procedure_name] = Procedure(
            lf=lf,
            metadata=bdf.Metadata(),
            readme_dict=experiment_dict,
        )

        # write the data to a parquet file if a path is provided
        if output_data_path is not None:
            if not output_data_path.endswith(".parquet"):
                output_data_path += ".parquet"
            lf.collect().write_parquet(output_data_path)

    @deprecated(
        reason="Use :func:`~pyprobe.io.process_cycler` and "
        ":meth:`~pyprobe.filters.Procedure.load` to manage data persistence.",
        version="2.5.0",
        plain_reason="Cell.archive() is deprecated. "
        "Use pyprobe.io.process_cycler() and Procedure.load() "
        "to manage data persistence.",
    )
    def archive(self, path: str) -> None:
        """Archive the cell object.

        .. deprecated::
            Use :func:`~pyprobe.io.process_cycler` and
            :meth:`~pyprobe.filters.Procedure.load` to manage data persistence.

        Args:
            path (str): The path to the archive directory or zip file.
        """
        if path.endswith(".zip"):
            zip_file = True
            path = path[:-4]
        else:
            zip_file = False
        if not os.path.exists(path):
            os.makedirs(path)
        metadata: dict[str, Any] = {
            "info": {},
            "procedure": {},
            "PyProBE Version": __version__,
        }
        for procedure_name, procedure in self.procedure.items():
            if isinstance(procedure.lf, pl.LazyFrame):
                df = procedure.lf.collect()
            else:
                df = procedure.lf
            # write the dataframe to a parquet file
            filename = procedure_name + ".parquet"
            filepath = os.path.join(path, filename)
            df.write_parquet(filepath)
            metadata["procedure"][procedure_name] = {
                "lf": filename,
                "info": procedure.info,
                "column_definitions": procedure.column_definitions,
                "step_descriptions": procedure.step_descriptions,
                "readme_dict": procedure.readme_dict,
                "cycle_info": procedure.cycle_info,
            }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        if zip_file:
            with zipfile.ZipFile(path + ".zip", "w") as zipf:
                for root, _, files in os.walk(path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, path)
                        zipf.write(file_path, arcname)
            # Delete the original directory
            shutil.rmtree(path)

    @deprecated(
        reason="Use :meth:`add_procedure` instead, which now handles all standard "
        "data input types (files and DataFrames).",
        version="2.0.1",
        plain_reason="process_cycler_file() is deprecated. Use add_procedure() "
        "instead.",
    )
    def process_cycler_file(
        self,
        cycler: str,
        folder_path: str,
        filename: str | Callable[[Any], str],
        output_name: str | None = None,
        filename_inputs: list[str] | None = None,
        compression_priority: Literal[
            "performance", "file size", "uncompressed"
        ] = "performance",
        overwrite_existing: bool = False,
    ) -> None:
        """Deprecated: Use add_procedure() instead.

        This method is deprecated and will be removed in a future version.
        Use :meth:`add_procedure` with a file path instead.
        """
        raise NotImplementedError(
            "process_cycler_file() has been removed. "
            "Use cell.add_procedure(procedure_name, source_path) instead, "
            "where source_path is the path to your cycler file."
        )

    @deprecated(
        reason="Use :meth:`add_procedure` instead, which now handles all standard "
        "data input types (files and DataFrames).",
        version="2.0.1",
        plain_reason="process_generic_file() is deprecated. Use add_procedure() "
        "instead.",
    )
    def process_generic_file(
        self,
        folder_path: str,
        input_filename: str,
        output_filename: str,
        column_importers: list[Any] | None = None,
    ) -> None:
        """Deprecated: Use add_procedure() instead.

        This method is deprecated and will be removed in a future version.
        Use :meth:`add_procedure` with a DataFrame and column_map instead.
        """
        raise NotImplementedError(
            "process_generic_file() has been removed. "
            "Use cell.add_procedure(procedure_name, dataframe, "
            "column_map=..., output_path=...) instead."
        )

    @deprecated(
        reason="Use :meth:`add_procedure` instead, which now handles all standard "
        "data input types (files and DataFrames).",
        version="2.5.0",
        plain_reason="import_data() is deprecated. Use add_procedure() instead.",
    )
    def import_data(
        self,
        procedure_name: str,
        data_path: str,
        readme_path: str | None = None,
    ) -> None:
        """Deprecated: Use add_procedure() instead.

        This method is deprecated and will be removed in a future version.
        Use :meth:`add_procedure` with a parquet file path instead.
        """
        raise NotImplementedError(
            "import_data() has been removed. "
            "Use cell.add_procedure(procedure_name, data_path, "
            "readme_path=...) instead, where data_path is a path to a "
            "parquet file."
        )

    @deprecated(
        reason="Use :meth:`add_procedure` instead, which now handles all standard "
        "data input types (files and DataFrames).",
        version="2.5.0",
        plain_reason="import_from_cycler() is deprecated. Use add_procedure() instead.",
    )
    def import_from_cycler(
        self,
        procedure_name: str,
        cycler: str,
        input_data_path: str,
        output_data_path: str | None = None,
        readme_path: str | None = None,
        column_importers: list[Any] | None = None,
        extra_column_importers: list[Any] | None = None,
        compression_priority: Literal[
            "performance", "file size", "uncompressed"
        ] = "performance",
        overwrite_existing: bool = False,
    ) -> None:
        """Deprecated: Use add_procedure() instead.

        This method is deprecated and will be removed in a future version.
        Use :meth:`add_procedure` with a file path instead.
        """
        raise NotImplementedError(
            "import_from_cycler() has been removed. "
            "Use cell.add_procedure(procedure_name, input_data_path, "
            "output_path=..., readme_path=...) instead."
        )


@deprecated(
    reason="Use :meth:`~pyprobe.filters.Procedure.load` to load data directly from "
    "Parquet files written by :func:`~pyprobe.io.process_cycler`.",
    version="2.5.0",
    plain_reason="load_archive() is deprecated. "
    "Use Procedure.load() to load data from Parquet files written by "
    "process_cycler().",
)
def load_archive(path: str) -> Cell:
    """Load a cell object from an archive.

    .. deprecated::
        Use :meth:`~pyprobe.filters.Procedure.load` to load data directly from
        Parquet files written by :func:`~pyprobe.io.process_cycler`.

    Args:
        path (str): The path to the archive directory.

    Returns:
        Cell: The cell object.
    """
    if path.endswith(".zip"):
        extract_path = path[:-4]
        with zipfile.ZipFile(path, "r") as zipf:
            zipf.extractall(extract_path)
        os.remove(path)
        archive_path = extract_path
    else:
        archive_path = path

    with open(os.path.join(archive_path, "metadata.json")) as f:
        metadata = json.load(f)
    if metadata["PyProBE Version"] != __version__:
        warnings.warn(
            f"The PyProBE version used to archive the cell was "
            f"{metadata['PyProBE Version']}, the current version is "
            f"{__version__}. There may be compatibility"
            f" issues.",
        )
    metadata.pop("PyProBE Version")
    legacy_info: dict[str, Any] = metadata.get("info", {})
    cell = Cell()
    for procedure_name, procedure in metadata["procedure"].items():
        readme_dict = procedure.get("readme_dict", {})
        for experiment_data in readme_dict.values():
            if "Cycles" in experiment_data:
                experiment_data["Cycles"] = [
                    tuple(cycle) for cycle in experiment_data["Cycles"]
                ]
        cell.procedure[procedure_name] = Procedure(
            lf=os.path.join(archive_path, procedure["lf"]),
            metadata=bdf.Metadata(extras=procedure.get("metadata", legacy_info)),
            readme_dict=readme_dict,
            column_definitions=procedure.get("column_definitions"),
            step_descriptions=procedure.get("step_descriptions"),
            cycle_info=procedure.get("cycle_info"),
        )

    return cell


@deprecated(
    reason="Replaced by :func:`pyprobe.io.process_cycler` and "
    ":meth:`Cell.add_procedure`, which provide a more flexible API.",
    version="2.5.0",
    plain_reason="process_cycler_data() is deprecated. Use Cell.add_procedure() "
    "instead.",
)
def process_cycler_data(
    cycler: str,
    input_data_path: str,
    output_data_path: str | None = None,
    column_importers: list[Any] | None = None,
    extra_column_importers: list[Any] | None = None,
    compression_priority: Literal[
        "performance", "file size", "uncompressed"
    ] = "performance",
    overwrite_existing: bool = False,
) -> str | None:
    """Deprecated: Use Cell.add_procedure() instead.

    This module-level function is deprecated and will be removed in a future version.
    Create a Cell instance and use its add_procedure() method instead.
    """
    raise NotImplementedError(
        "process_cycler_data() has been removed. "
        "Use cell.add_procedure(procedure_name, input_data_path, "
        "output_path=...) instead, where cell is a Cell instance."
    )


@deprecated(
    reason="Use :class:`~pyprobe.cell.Cell` directly and load procedures via "
    ":meth:`~pyprobe.cell.Cell.add_procedure`.",
    version="2.5.0",
    plain_reason="make_cell_list() is deprecated. "
    "Use Cell() directly and load procedures via Cell.add_procedure().",
)
def make_cell_list(
    record_filepath: str,
    worksheet_name: str,
    header_row: int = 0,
) -> list[Cell]:
    """Function to make a list of cell objects from a record of tests in Excel format.

    .. deprecated::
        Use :class:`~pyprobe.cell.Cell` directly and load procedures via
        :meth:`~pyprobe.cell.Cell.add_procedure`.

    Args:
        record_filepath (str): The path to the experiment record .xlsx file.
        worksheet_name (str): The worksheet name to read from the record.
        header_row (int, optional):
            The row number containing the column headers. Defaults to 0.

    Returns:
        list: The list of cell objects.
    """
    record = pl.read_excel(
        record_filepath,
        sheet_name=worksheet_name,
        read_options={"header_row": header_row},
    )

    n_cells = len(record)
    cell_list = []
    for _ in range(n_cells):
        cell_list.append(Cell())
    return cell_list
