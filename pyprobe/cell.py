"""Module for the Cell class."""
import os
import time
import warnings
from typing import Callable, Dict, List, Optional

import distinctipy
import polars as pl
import pybamm.solvers.solution
from pydantic import BaseModel, Field, field_validator, validate_call

from pyprobe.cyclers import arbin, basecycler, basytec, biologic, maccor, neware
from pyprobe.filters import Procedure
from pyprobe.readme_processor import process_readme


class Cell(BaseModel):
    """A class for a cell in a battery experiment."""

    info: Dict[str, Optional[str | int | float]]
    """Dictionary containing information about the cell.
    The dictionary must contain a 'Name' field, other information may include
    channel number or other rig information.
    """
    procedure: Dict[str, Procedure] = Field(default_factory=dict)
    """Dictionary containing the procedures that have been run on the cell."""

    @field_validator("info")
    def check_and_set_name(
        cls, info: Dict[str, Optional[str | int | float]]
    ) -> Dict[str, Optional[str | int | float]]:
        """Validate the `info` field.

        Checks that a `Name` field is present in the `info` dictionary, if not it is
        set to 'Default Name'. If the `color` field is not present, a color is
        generated.
        """
        if "Name" not in info.keys():
            info["Name"] = "Default Name"
            warnings.warn(
                "The 'Name' field was not in info. It has been set to 'Default Name'."
            )

        if "color" not in info.keys():
            info["color"] = distinctipy.get_hex(
                distinctipy.get_colors(
                    1,
                    rng=1,  # Set the random seed
                    exclude_colors=[
                        (0, 0, 0),
                        (1, 1, 1),
                        (1, 1, 0),
                    ],
                )[0]
            )
        values = info
        return values

    @validate_call
    def process_cycler_file(
        self,
        cycler: str,
        folder_path: str,
        input_filename: str | Callable[[str], str],
        output_filename: str | Callable[[str], str],
        filename_inputs: Optional[List[str]] = None,
    ) -> None:
        """Convert cycler file into PyProBE format.

        Args:
            cycler (str):
                The cycler used to produce the data. Available cyclers are:
                - 'neware'
                - 'biologic'
                - 'biologic_MB' (for Modulo Bat Biologic data)
            folder_path (str):
                The path to the folder containing the data file.
            input_filename (str | function):
                A filename string or a function to generate the file name for cycler
                data.
            output_filename (str | function):
                A filename string or a function to generate the file name for PyProBE
                data.
            filename_inputs (list):
                The list of inputs to input_filename and output_filename, if they are
                functions. These must be keys of the cell info.
        """
        input_data_path = self._get_data_paths(
            folder_path, input_filename, filename_inputs
        )
        output_data_path = self._get_data_paths(
            folder_path, output_filename, filename_inputs
        )
        output_data_path = self._verify_parquet(output_data_path)
        if "*" in output_data_path:
            raise ValueError("* characters are not allowed for a complete data path.")

        cycler_dict = {
            "neware": neware.Neware,
            "biologic": biologic.Biologic,
            "biologic_MB": biologic.BiologicMB,
            "arbin": arbin.Arbin,
            "maccor": maccor.Maccor,
            "basytec": basytec.Basytec,
        }
        t1 = time.time()
        importer = cycler_dict[cycler](input_data_path=input_data_path)
        self._write_parquet(importer, output_data_path)
        print(f"\tparquet written in {time.time()-t1: .2f} seconds.")

    @validate_call
    def process_generic_file(
        self,
        folder_path: str,
        input_filename: str | Callable[[str], str],
        output_filename: str | Callable[[str], str],
        column_dict: Dict[str, str],
        filename_inputs: Optional[List[str]] = None,
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
            filename_inputs (list):
                The list of inputs to input_filename and output_filename.
                These must be keys of the cell info.
        """
        input_data_path = self._get_data_paths(
            folder_path, input_filename, filename_inputs
        )
        output_data_path = self._get_data_paths(
            folder_path, output_filename, filename_inputs
        )
        output_data_path = self._verify_parquet(output_data_path)
        if "*" in output_data_path:
            raise ValueError("* characters are not allowed for a complete data path")

        t1 = time.time()
        importer = basecycler.BaseCycler(
            input_data_path=input_data_path,
            column_dict=column_dict,
        )
        self._write_parquet(importer, output_data_path)
        print(f"\tparquet written in {time.time()-t1: .2f} seconds.")

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
        if "*" in output_data_path:
            raise ValueError("* characters are not allowed for a complete data path.")

        base_dataframe = pl.scan_parquet(output_data_path)
        data_folder = os.path.dirname(output_data_path)
        readme_path = os.path.join(data_folder, readme_name)
        readme = process_readme(readme_path)

        self.procedure[procedure_name] = Procedure(
            readme_dict=readme.experiment_dict,
            base_dataframe=base_dataframe,
            info=self.info,
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
        return filename

    def _write_parquet(
        self,
        importer: basecycler.BaseCycler,
        output_data_path: str,
    ) -> None:
        """Import data from a cycler file and write to a PyProBE parquet file.

        Args:
            importer (BaseCycler): The cycler object to import the data.
            output_data_path (str): The path to write the parquet file.
        """
        dataframe = importer.pyprobe_dataframe
        if isinstance(dataframe, pl.LazyFrame):
            dataframe = dataframe.collect()
        dataframe.write_parquet(output_data_path)

    @staticmethod
    def _get_filename(
        info: Dict[str, Optional[str | int | float]],
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
                raise ValueError(
                    "filename_inputs must be provided when filename is a function."
                )
            filename_str = self._get_filename(self.info, filename, filename_inputs)

        data_path = os.path.join(folder_path, filename_str)
        return data_path

    def import_pybamm_solution(
        self,
        procedure_name: str,
        experiment_names: List[str] | str,
        pybamm_solutions: List[pybamm.solvers.solution] | pybamm.solvers.solution,
        output_data_path: Optional[str] = None,
        optional_variables: Optional[List[str]] = None,
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

        # check if the experiment names and PyBaMM solutions are lists
        if isinstance(experiment_names, list) and isinstance(pybamm_solutions, list):
            if len(experiment_names) != len(pybamm_solutions):
                raise ValueError(
                    "The number of experiment names and PyBaMM solutions must be equal."
                )
        elif isinstance(experiment_names, list) != isinstance(pybamm_solutions, list):
            if isinstance(experiment_names, list):
                raise ValueError(
                    "A list of experiment names must be provided with a list of PyBaMM"
                    " solutions."
                )
            else:
                raise ValueError(
                    "A single experiment name must be provided with a single PyBaMM"
                    " solution."
                )
        else:
            experiment_names = [str(experiment_names)]
            pybamm_solutions = [pybamm_solutions]

        all_solution_data = pl.LazyFrame({})
        for experiment_name, pybamm_solution in zip(experiment_names, pybamm_solutions):
            # get the data from the PyBaMM solution object
            pybamm_data = pybamm_solution.get_data_dict(import_variables)
            # convert the PyBaMM data to a polars dataframe and add the experiment name
            # as a column
            solution_data = pl.LazyFrame(pybamm_data).with_columns(
                pl.lit(experiment_name).alias("Experiment")
            )
            if all_solution_data == pl.LazyFrame({}):
                all_solution_data = solution_data
            else:
                # join the new solution data with the existing solution data, a right
                # join is used to keep all the data
                all_solution_data = all_solution_data.join(
                    solution_data, on=import_variables + ["Cycle", "Step"], how="right"
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
                pl.col("Current [A]"),
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


def make_cell_list(
    record_filepath: str,
    worksheet_name: str,
) -> List[Cell]:
    """Function to make a list of cell objects from a record of tests in Excel format.

    Args:
        record_filepath (str): The path to the experiment record .xlsx file.
        worksheet_name (str): The worksheet name to read from the record.

    Returns:
        list: The list of cell objects.
    """
    record = pl.read_excel(record_filepath, sheet_name=worksheet_name)

    n_cells = len(record)
    cell_list = []
    rgb = distinctipy.get_colors(
        n_cells,
        exclude_colors=[
            (0, 0, 0),
            (1, 1, 1),
            (1, 1, 0),
        ],  # Exclude black, white, and yellow
        rng=1,  # Set the random seed
        n_attempts=5000,
    )
    for i in range(n_cells):
        info = record.row(i, named=True)
        info["color"] = distinctipy.get_hex(rgb[i])
        cell_list.append(Cell(info=info))
    return cell_list
