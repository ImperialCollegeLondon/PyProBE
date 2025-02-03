"""Module for the Cell class."""

import json
import logging
import os
import shutil
import time
import warnings
import zipfile
from typing import Any, Callable, Dict, List, Literal, Optional

import polars as pl
from pydantic import BaseModel, Field, validate_call

from pyprobe._version import __version__
from pyprobe.cyclers import arbin, basecycler, basytec, biologic, maccor, neware
from pyprobe.filters import Procedure
from pyprobe.readme_processor import process_readme
from pyprobe.utils import PyBaMMSolution

logger = logging.getLogger(__name__)


class Cell(BaseModel):
    """A class for a cell in a battery experiment."""

    info: dict[str, Optional[Any]]
    """Dictionary containing information about the cell.
    The dictionary must contain a 'Name' field, other information may include
    channel number or other rig information.
    """
    procedure: Dict[str, Procedure] = Field(default_factory=dict)
    """Dictionary containing the procedures that have been run on the cell."""

    def _convert_to_parquet(
        self,
        importer: basecycler.BaseCycler,
        folder_path: str,
        output_filename: str | Callable[[str], str],
        filename_inputs: Optional[List[str]] = None,
        compression_priority: Literal[
            "performance", "file size", "uncompressed"
        ] = "performance",
        overwrite_existing: bool = False,
    ) -> None:
        """Convert a file into PyProBE format.

        Args:
            importer:
                The cycler object to import the data.
            folder_path:
                The path to the folder containing the data file.
            output_filename:
                A filename string or a function to generate the file name for PyProBE
                data.
            filename_inputs:
                The list of inputs to input_filename and output_filename, if they are
                functions. These must be keys of the cell info.
            compression_priority:
                The priority of the compression algorithm to use on the resulting
                parquet file. Available options are:
                - 'performance': Use the 'lz4' compression algorithm (default).
                - 'file size': Use the 'zstd' compression algorithm.
                - 'uncompressed': Do not use compression.
            overwrite_existing:
                If True, any existing parquet file with the output_filename will be
                overwritten. If False, the function will skip the conversion if the
                parquet file already exists.
        """
        output_data_path = self._get_data_paths(
            folder_path, output_filename, filename_inputs
        )
        output_data_path = self._verify_parquet(output_data_path)

        if not os.path.exists(output_data_path) or overwrite_existing:
            t1 = time.time()
            compression_dict = {
                "uncompressed": "uncompressed",
                "performance": "lz4",
                "file size": "zstd",
            }
            self._write_parquet(
                importer,
                output_data_path,
                compression=compression_dict[compression_priority],
            )
            logger.info(f"\tparquet written in {time.time()-t1: .2f} seconds.")
        else:
            logger.info(f"File {output_data_path} already exists. Skipping.")

    @validate_call
    def process_cycler_file(
        self,
        cycler: Literal[
            "neware", "biologic", "biologic_MB", "arbin", "basytec", "maccor", "generic"
        ],
        folder_path: str,
        input_filename: str | Callable[[str], str],
        output_filename: str | Callable[[str], str],
        filename_inputs: Optional[List[str]] = None,
        compression_priority: Literal[
            "performance", "file size", "uncompressed"
        ] = "performance",
        overwrite_existing: bool = False,
    ) -> None:
        """Convert a file into PyProBE format.

        Args:
            cycler:
                The cycler used to produce the data.
            folder_path:
                The path to the folder containing the data file.
            input_filename:
                A filename string or a function to generate the file name for cycler
                data.
            output_filename:
                A filename string or a function to generate the file name for PyProBE
                data.
            filename_inputs:
                The list of inputs to input_filename and output_filename, if they are
                functions. These must be keys of the cell info.
            compression_priority:
                The priority of the compression algorithm to use on the resulting
                parquet file. Available options are:
                - 'performance': Use the 'lz4' compression algorithm (default).
                - 'file size': Use the 'zstd' compression algorithm.
                - 'uncompressed': Do not use compression.
            overwrite_existing:
                If True, any existing parquet file with the output_filename will be
                overwritten. If False, the function will skip the conversion if the
                parquet file already exists.
        """
        cycler_dict = {
            "neware": neware.Neware,
            "biologic": biologic.Biologic,
            "biologic_MB": biologic.BiologicMB,
            "arbin": arbin.Arbin,
            "basytec": basytec.Basytec,
            "maccor": maccor.Maccor,
        }
        input_data_path = self._get_data_paths(
            folder_path, input_filename, filename_inputs
        )
        importer = cycler_dict[cycler](input_data_path=input_data_path)
        self._convert_to_parquet(
            importer,
            folder_path,
            output_filename,
            filename_inputs,
            compression_priority,
            overwrite_existing,
        )

    @validate_call
    def process_generic_file(
        self,
        folder_path: str,
        input_filename: str | Callable[[str], str],
        output_filename: str | Callable[[str], str],
        column_dict: Dict[str, str],
        header_row_index: int = 0,
        date_column_format: Optional[str] = None,
        filename_inputs: Optional[List[str]] = None,
        compression_priority: Literal[
            "performance", "file size", "uncompressed"
        ] = "performance",
    ) -> None:
        """Convert generic file into PyProBE format.

        Args:
            folder_path (str):
                The path to the folder containing the data file.
            input_filename (str | function):
                A filename string or a function to generate the file name for the
                generic data.
            output_filename (str | function):
                A filename string or a function to generate the file name for PyProBE
                data.
            column_dict (dict):
                A dictionary mapping the column names in the generic file to the PyProBE
                column names. The keys of the dictionary are the cycler column names and
                the values are the PyProBE column names. You must use asterisks to
                indicate the units of the columns.
                E.g. :code:`{"V (*)": "Voltage [*]"}`.
            header_row_index (int, optional):
                The index of the header row in the file. Defaults to 0.
            date_column_format (str, optional):
                The format of the date column in the generic file. Defaults to None.
            filename_inputs (list):
                The list of inputs to input_filename and output_filename.
                These must be keys of the cell info.
            compression_priority:
                The priority of the compression algorithm to use on the resulting
                parquet file. Available options are:
                - 'performance': Use the 'lz4' compression algorithm (default).
                - 'file size': Use the 'zstd' compression algorithm.
                - 'uncompressed': Do not use compression.
        """
        input_data_path = self._get_data_paths(
            folder_path, input_filename, filename_inputs
        )
        importer = basecycler.BaseCycler(
            input_data_path=input_data_path,
            column_dict=column_dict,
            header_row_index=header_row_index,
            datetime_format=date_column_format,
        )
        self._convert_to_parquet(
            importer,
            folder_path,
            output_filename,
            filename_inputs,
            compression_priority,
        )

    @validate_call
    def add_procedure(
        self,
        procedure_name: str,
        folder_path: str,
        filename: str | Callable[[str], str],
        filename_inputs: Optional[List[str]] = None,
        readme_name: str = "README.yaml",
    ) -> None:
        """Add data in a PyProBE-format parquet file to the procedure dict of the cell.

        Args:
            procedure_name (str):
                A name to give the procedure. This will be used when calling
                :code:`cell.procedure[procedure_name]`.
            folder_path (str):
                The path to the folder containing the data file.
            filename (str | function):
                A filename string or a function to generate the file name for PyProBE
                data.
            filename_inputs (Optional[list]):
                The list of inputs to filename_function. These must be keys of the cell
                info.
            readme_name (str, optional):
                The name of the readme file. Defaults to "README.yaml". It is assumed
                that the readme file is in the same folder as the data file.
        """
        output_data_path = self._get_data_paths(folder_path, filename, filename_inputs)
        output_data_path = self._verify_parquet(output_data_path)
        base_dataframe = pl.scan_parquet(output_data_path)
        data_folder = os.path.dirname(output_data_path)
        readme_path = os.path.join(data_folder, readme_name)
        readme = process_readme(readme_path)

        self.procedure[procedure_name] = Procedure(
            readme_dict=readme.experiment_dict,
            base_dataframe=base_dataframe,
            info=self.info,
        )

    @validate_call
    def quick_add_procedure(
        self,
        procedure_name: str,
        folder_path: str,
        filename: str | Callable[[str], str],
        filename_inputs: Optional[List[str]] = None,
    ) -> None:
        """Add data in a PyProBE-format parquet file to the procedure dict of the cell.

        This method does not require a README file. It is useful for quickly adding data
        but filtering by experiment on the resulting object will not be possible.

        Args:
            procedure_name (str):
                A name to give the procedure. This will be used when calling
                :code:`cell.procedure[procedure_name]`.
            folder_path (str):
                The path to the folder containing the data file.
            filename (str | function):
                A filename string or a function to generate the file name for PyProBE
                data.
            filename_inputs (Optional[list]):
                The list of inputs to filename_function. These must be keys of the cell
                info.
        """
        output_data_path = self._get_data_paths(folder_path, filename, filename_inputs)
        output_data_path = self._verify_parquet(output_data_path)
        base_dataframe = pl.scan_parquet(output_data_path)
        self.procedure[procedure_name] = Procedure(
            base_dataframe=base_dataframe, info=self.info, readme_dict={}
        )

    @staticmethod
    def _verify_parquet(filename: str) -> str:
        """Function to verify the filename is in the correct parquet format.

        Args:
            filename (str): The filename to verify.

        Returns:
            str: The filename.
        """
        # Get the file extension of output_filename
        _, ext = os.path.splitext(filename)

        # If the file extension is not .parquet, replace it with .parquet
        if ext != ".parquet":
            filename = os.path.splitext(filename)[0] + ".parquet"
        if "*" in filename:
            error_msg = "* characters are not allowed for output filename."
            logger.error(error_msg)
            raise ValueError(error_msg)
        return filename

    def _write_parquet(
        self,
        importer: basecycler.BaseCycler,
        output_data_path: str,
        compression: str,
    ) -> None:
        """Import data from a cycler file and write to a PyProBE parquet file.

        Args:
            importer (BaseCycler): The cycler object to import the data.
            output_data_path (str): The path to write the parquet file.
            compression (str): The compression algorithm to use.
        """
        dataframe = importer.pyprobe_dataframe
        if isinstance(dataframe, pl.LazyFrame):
            dataframe = dataframe.collect()
        dataframe.write_parquet(output_data_path, compression=compression)

    @staticmethod
    def _get_filename(
        info: Dict[str, Optional[Any]],
        filename_function: Callable[[str], str],
        filename_inputs: List[str],
    ) -> str:
        """Function to generate the filename for the data, if provided as a function.

        Args:
            info (dict): The info entry for the data file.
            filename_function (function): The function to generate the input name.
            filename_inputs (list):
                The list of inputs to filename_function. These must be keys of the cell
                info.

        Returns:
            str: The input name for the data file.
        """
        return filename_function(
            *(str(info[filename_inputs[i]]) for i in range(len(filename_inputs)))
        )

    def _get_data_paths(
        self,
        folder_path: str,
        filename: str | Callable[[str], str],
        filename_inputs: Optional[List[str]] = None,
    ) -> str:
        """Function to generate the input and output paths for the data file.

        Args:
            folder_path (str): The path to the folder containing the data file.
            filename (str | function): A filename string or a function to generate
                the file name.
            filename_inputs (Optional[list]): The list of inputs to filename_function.
                These must be keys of the cell info.

        Returns:
            str: The full path for the data file.
        """
        if isinstance(filename, str):
            filename_str = filename
        else:
            if filename_inputs is None:
                error_msg = (
                    "filename_inputs must be provided when filename is a function."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            filename_str = self._get_filename(self.info, filename, filename_inputs)

        data_path = os.path.join(folder_path, filename_str)
        return data_path

    def import_pybamm_solution(
        self,
        procedure_name: str,
        experiment_names: list[str] | str,
        pybamm_solutions: list[PyBaMMSolution] | PyBaMMSolution,
        output_data_path: Optional[str] = None,
        optional_variables: Optional[list[str]] = None,
    ) -> None:
        """Import a PyBaMM solution object into a procedure of the cell.

        Filtering a PyBaMM solution object by cycle and step reflects the behaviour of
        the :code:`cycles` and :code:`steps` dictionaries of the PyBaMM solution object.

        Multiple experiments can be imported into the same procedure. This is achieved
        by providing multiple solution objects and experiment names.

        This method optionally writes the data to a parquet file, if a data path is
        provided.

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
        for experiment_name, pybamm_solution in zip(experiment_names, pybamm_solutions):
            # get the data from the PyBaMM solution object
            pybamm_data = pybamm_solution.get_data_dict(import_variables)
            # convert the PyBaMM data to a polars dataframe and add the experiment name
            # as a column
            solution_data = pl.LazyFrame(pybamm_data).with_columns(
                pl.lit(experiment_name).alias("Experiment")
            )
            if lazyframe_created is False:
                all_solution_data = solution_data
                lazyframe_created = True
            else:
                # join the new solution data with the existing solution data, a right
                # join is used to keep all the data
                all_solution_data = all_solution_data.join(
                    solution_data, on=import_variables + ["Step"], how="right"
                )
                # fill null values where the experiment has been extended with the newly
                #  joined experiment name
                all_solution_data = all_solution_data.with_columns(
                    pl.col("Experiment").fill_null(pl.col("Experiment_right"))
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
            max_steps, on="Experiment", how="left"
        ).with_columns(
            (pl.col("Step") + pl.col("Max Step").fill_null(-1) + 1).alias("Step")
        )
        # get the range of step values for each experiment
        step_ranges = all_solution_data.group_by("Experiment").agg(
            pl.arange(pl.col("Step").min(), pl.col("Step").max() + 1).alias(
                "Step Range"
            )
        )

        # create a dictionary of the experiment names and the step ranges
        experiment_dict = {}
        for row in step_ranges.collect().iter_rows():
            experiment = row[0]
            experiment_dict[experiment] = {"Steps": row[1]}
            experiment_dict[experiment]["Step Descriptions"] = []

        # reformat the data to the PyProBE format
        base_dataframe = all_solution_data.select(
            [
                pl.col("Time [s]"),
                pl.col("Current [A]") * -1,
                pl.col("Terminal voltage [V]").alias("Voltage [V]"),
                (pl.col("Discharge capacity [A.h]") * -1).alias("Capacity [Ah]"),
                pl.col("Step"),
                (
                    (
                        pl.col("Step").cast(pl.Int64)
                        - pl.col("Step").cast(pl.Int64).shift()
                        != 0
                    )
                    .fill_null(strategy="zero")
                    .cum_sum()
                    .alias("Event")
                ),
            ]
        )
        # create the procedure object
        self.procedure[procedure_name] = Procedure(
            base_dataframe=base_dataframe, info=self.info, readme_dict=experiment_dict
        )

        # write the data to a parquet file if a path is provided
        if output_data_path is not None:
            if not output_data_path.endswith(".parquet"):
                output_data_path += ".parquet"
            base_dataframe.collect().write_parquet(output_data_path)

    def archive(self, path: str) -> None:
        """Archive the cell object.

        Args:
            path (str): The path to the archive directory or zip file.
        """
        if path.endswith(".zip"):
            zip = True
            path = path[:-4]
        else:
            zip = False
        if not os.path.exists(path):
            os.makedirs(path)
        metadata = self.dict()
        metadata["PyProBE Version"] = __version__
        for procedure_name, procedure in self.procedure.items():
            if isinstance(procedure.live_dataframe, pl.LazyFrame):
                df = procedure.live_dataframe.collect()
            else:
                df = procedure.live_dataframe
            # write the dataframe to a parquet file
            filename = procedure_name + ".parquet"
            filepath = os.path.join(path, filename)
            df.write_parquet(filepath)
            # update the metadata with the filename
            metadata["procedure"][procedure_name]["base_dataframe"] = filename
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        if zip:
            with zipfile.ZipFile(path + ".zip", "w") as zipf:
                for root, _, files in os.walk(path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, path)
                        zipf.write(file_path, arcname)
            # Delete the original directory
            shutil.rmtree(path)


def load_archive(path: str) -> Cell:
    """Load a cell object from an archive.

    Args:
        path (str): The path to the archive directory.

    Returns:
        Cell: The cell object.
    """
    if path.endswith(".zip"):
        extract_path = path[:-4]
        with zipfile.ZipFile(path, "r") as zipf:
            with zipfile.ZipFile(path, "r") as zipf:
                zipf.extractall(extract_path)
            # Delete the original zip file
            os.remove(path)
            archive_path = extract_path
    else:
        archive_path = path

    with open(os.path.join(archive_path, "metadata.json"), "r") as f:
        metadata = json.load(f)
    if metadata["PyProBE Version"] != __version__:
        warnings.warn(
            f"The PyProBE version used to archive the cell was "
            f"{metadata['PyProBE Version']}, the current version is "
            f"{__version__}. There may be compatibility"
            f" issues."
        )
    metadata.pop("PyProBE Version")
    for procedure in metadata["procedure"].values():
        procedure["base_dataframe"] = os.path.join(
            archive_path, procedure["base_dataframe"]
        )
    cell = Cell(**metadata)

    return cell


def make_cell_list(
    record_filepath: str,
    worksheet_name: str,
    header_row: int = 0,
) -> List[Cell]:
    """Function to make a list of cell objects from a record of tests in Excel format.

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
    for i in range(n_cells):
        info = record.row(i, named=True)
        cell_list.append(Cell(info=info))
    return cell_list
