"""Module for the Cell class."""
import os
import pickle
import platform
import subprocess
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import distinctipy
import polars as pl
import pybamm
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator, validate_call

from pyprobe.cyclers import basecycler, biologic, neware
from pyprobe.filters import Procedure


class Cell(BaseModel):
    """A class for a cell in a battery experiment.

    Args:
        info (dict): Dictionary containing information about the cell.
            The dictionary must contain a 'Name' field, other information may include
            channel number or other rig information.
        procedure (dict, optional): Dictionary containing the procedures that have been
            run on the cell. Defaults to an empty dictionary.
    """

    info: Dict[str, Optional[str | int | float]]
    """Dictionary containing information about the cell.
    The dictionary must contain a 'Name' field, other information may include
    channel number or other rig information.
    """
    procedure: Dict[str, Procedure] = Field(default_factory=dict)
    """Dictionary containing the procedures that have been run on the cell."""

    @field_validator("info")
    def _check_and_set_name(
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

    @classmethod
    def make_cell_list(
        cls,
        record_filepath: str,
        worksheet_name: str,
    ) -> List["Cell"]:
        """Function to make a list of cell objects from a record of tests.

        Args:
            record_filepath (str): The path to the experiment record .xlsx file.
            worksheet_name (str): The worksheet name to read from the record.

        Returns:
            list: The list of cell objects
        """
        record = pl.read_excel(record_filepath, sheet_name=worksheet_name)

        n_cells = len(record)
        cell_list = []
        colors = cls.set_color_scheme(n_cells, scheme="distinctipy")
        for i in range(n_cells):
            info = record.row(i, named=True)
            info["color"] = colors[i]
            cell_list.append(cls(info=info))
        return cell_list

    @staticmethod
    def _verify_parquet(filename: str) -> str:
        """Function to verify the filename is in the correct format.

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
        """Import data from a cycler file.

        Args:
            importer (BaseCycler): The cycler object to import the data.
            output_data_path (str): The path to write the parquet file.
        """
        dataframe = importer.pyprobe_dataframe
        if isinstance(dataframe, pl.LazyFrame):
            dataframe = dataframe.collect()
        dataframe.write_parquet(output_data_path)

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
                The cycler used to produce the data. E.g. 'neware' or 'biologic'.
            folder_path (str):
                The path to the folder containing the data file.
            input_filename (str | function):
                A filename string or a function to generate the file name for cycler
                data.
            output_filename (str | function):
                A filename string or a function to generate the file name for PyProBE
                data.
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
            raise ValueError("* characters are not allowed for a complete data path.")

        cycler_dict = {
            "neware": neware.Neware,
            "biologic": biologic.Biologic,
            "biologic_MB": biologic.BiologicMB,
        }
        t1 = time.time()
        importer = cycler_dict[cycler](input_data_path=input_data_path)
        self._write_parquet(importer, output_data_path)
        print(f"\tparquet written in {time.time()-t1:.2f} seconds.")

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
                column names.
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
        print(f"\tparquet written in {time.time()-t1:.2f} seconds.")

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
                A filename string or a function to generate the file name for PyProBE d
                ata.
            filename_inputs (Optional[list]):
                The list of inputs to filename_function. These must be keys of the cell
                info.
            readme_name (str, optional):
                The name of the readme file. Defaults to "README.yaml".
        """
        output_data_path = self._get_data_paths(folder_path, filename, filename_inputs)
        output_data_path = self._verify_parquet(output_data_path)
        if "*" in output_data_path:
            raise ValueError("* characters are not allowed for a complete data path.")

        base_dataframe = pl.scan_parquet(output_data_path)
        data_folder = os.path.dirname(output_data_path)
        readme_path = os.path.join(data_folder, readme_name)
        readme = self._process_readme(readme_path)

        self.procedure[procedure_name] = Procedure(
            titles=readme.titles,
            steps_idx=readme.step_numbers,
            base_dataframe=base_dataframe,
            info=self.info,
            pybamm_experiment=readme.pybamm_experiment,
            pybamm_experiment_list=readme.pybamm_experiment_list,
        )

    @staticmethod
    def _get_filename(
        info: Dict[str, Optional[str | int | float]],
        filename_function: Callable[[str], str],
        filename_inputs: List[str],
    ) -> str:
        """Function to generate the input name for the data file.

        Args:
            info (dict): The info entry for the data file.
            filename_function (function): The function to generate the input name.
            filename_inputs (list): The list of inputs to filename_function.
                These must be keys of the cell info.

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

    @staticmethod
    def set_color_scheme(
        n_colors: int, scheme: str = "distinctipy", **kwargs: Any
    ) -> List[str]:
        """Function to set the colour scheme for plotting.

        Args:
            n_colors (int): The number of colors to produce.
            scheme (str): The colour scheme to use.
            **kwargs: Additional keyword arguments for the colour scheme.

        Returns:
            list: The list of colours in hex format.
        """
        if scheme == "distinctipy":
            rgb = distinctipy.get_colors(
                n_colors,
                exclude_colors=[
                    (0, 0, 0),
                    (1, 1, 1),
                    (1, 1, 0),
                ],  # Exclude black, white, and yellow
                rng=1,  # Set the random seed
                n_attempts=5000,
                **kwargs,
            )
            hex = []
            for i in range(len(rgb)):
                hex.append(distinctipy.get_hex(rgb[i]))
            return hex
        else:
            raise NotImplementedError

    @staticmethod
    def launch_dashboard(cell_list: List["Cell"]) -> None:
        """Function to launch the dashboard for the preprocessed data.

        Args:
            cell_list (list): The list of cell objects to display in the dashboard.
        """
        with open("dashboard_data.pkl", "wb") as f:
            pickle.dump(cell_list, f)

        if platform.system() == "Windows":
            subprocess.Popen(
                [
                    "cmd",
                    "/c",
                    "start",
                    "/B",
                    "streamlit",
                    "run",
                    os.path.join(os.path.dirname(__file__), "dashboard.py"),
                    ">",
                    "nul",
                    "2>&1",
                ],
                shell=True,
            )
        elif platform.system() == "Darwin":
            subprocess.Popen(
                [
                    "nohup",
                    "streamlit",
                    "run",
                    os.path.join(os.path.dirname(__file__), "dashboard.py"),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )

    def _process_readme(
        cls,
        readme_path: str,
    ) -> "ReadmeModel":
        """Function to process the README.yaml file.

        Args:
            readme_path (str): The path to the README.yaml file.

        Returns:
            Tuple[List[str], List[List[int]], Optional[pybamm.Experiment]]
                - List[str]: The list of titles from the README.yaml file.
                - List[List[int]]: The list of steps from the README.yaml file.
                - Optional[pybamm.Experiment]: The PyBaMM experiment object.
        """
        with open(readme_path, "r") as file:
            readme_dict = yaml.safe_load(file)
        return ReadmeModel(readme_dict=readme_dict)


class ReadmeModel(BaseModel):
    """A class for processing the README.yaml file."""

    readme_dict: Dict[str, Any]
    readme_type: List[str] = Field(default_factory=list)
    number_of_experiments: int = Field(default_factory=int)
    titles: List[str] = Field(default_factory=list)

    step_numbers: List[List[int]] = Field(default_factory=list)
    step_indices: List[List[int]] = Field(default_factory=list)
    step_descriptions: List[List[str]] = Field(default_factory=list)
    cycle_details: List[List[Tuple[int, int, int]]] = Field(default_factory=list)
    pybamm_experiment_descriptions: List[Tuple[str, ...]] = Field(default_factory=list)
    pybamm_experiment_list: List[pybamm.Experiment] = Field(default_factory=list)
    pybamm_experiment: Optional[pybamm.Experiment] = Field(default=None)

    class Config:
        """Pydantic configuration settings."""

        arbitrary_types_allowed = True

    @model_validator(mode="before")
    @classmethod
    def _check_readme_dict(cls, data: Any) -> "ReadmeModel":
        """Validate the structure of the README.yaml file."""
        readme_dict = data["readme_dict"]
        data["readme_type"] = []
        cls.number_of_experiments = len(readme_dict)
        for experiment in readme_dict:
            if "Steps" in readme_dict[experiment]:
                steps = readme_dict[experiment]["Steps"]
                if isinstance(steps, dict):
                    if not all(
                        isinstance(k, int) and isinstance(v, str)
                        for k, v in steps.items()
                    ):
                        raise TypeError(
                            "The 'Steps' field must be a dictionary with keys of type"
                            " int and values of type str"
                        )
                    cycle_keys = [
                        key
                        for key in readme_dict[experiment].keys()
                        if "cycle" in key.lower()
                    ]
                    for cycle in cycle_keys:
                        cycle_dict = readme_dict[experiment][cycle]
                        if not all(
                            isinstance(k, str) and isinstance(v, int)
                            for k, v in cycle_dict.items()
                        ):
                            raise TypeError(
                                f"{cycle} must be a dictionary with keys of type str"
                                " and values of type int"
                            )
                    data["readme_type"].append("explicit")
                elif isinstance(steps, list):
                    if not all(isinstance(step, str) for step in steps):
                        raise TypeError("The 'Steps' field must be a list of strings")
                    data["readme_type"].append("implicit")
            elif "Total Steps" in readme_dict[experiment]:
                data["readme_type"].append("total")
        return data

    def model_post_init(self, __context: Any) -> None:
        """Get all the attributes of the class."""
        self.titles = list(self.readme_dict.keys())
        self.step_numbers = self.get_step_numbers()
        self.step_indices = self.get_step_indices()
        self.step_descriptions = self.get_step_descriptions()
        self.cycle_details = self.get_cycle_details()
        self.pybamm_experiment_descriptions = self.get_pybamm_experiment_descriptions()
        self.pybamm_experiment_list = self.get_pybamm_experiment_list()
        self.pybamm_experiment = self.get_pybamm_experiment()

    def get_step_numbers(self) -> List[List[int]]:
        """Get the step numbers from the README.yaml file."""
        max_step = 0
        all_steps = []
        for experiment, readme_format in zip(self.readme_dict, self.readme_type):
            if readme_format == "explicit":
                exp_steps = list(self.readme_dict[experiment]["Steps"].keys())
            elif readme_format == "total":
                exp_steps = list(range(self.readme_dict[experiment]["Total Steps"]))
                exp_steps = [x + max_step + 1 for x in exp_steps]
            else:
                exp_steps = list(range(len(self.readme_dict[experiment]["Steps"])))
                exp_steps = [x + max_step + 1 for x in exp_steps]
            max_step = exp_steps[-1]
            all_steps.append(exp_steps)
        return all_steps

    def get_step_indices(self) -> List[List[int]]:
        """Get the step indices from the README.yaml file."""
        step_indices = []
        for exp_step_numbers in self.step_numbers:
            step_indices.append(list(range(len(exp_step_numbers))))
        return step_indices

    def get_step_descriptions(self) -> List[List[str]]:
        """Get the step descriptions from the README.yaml file."""
        all_descriptions = []
        for experiment, readme_format in zip(self.readme_dict, self.readme_type):
            if readme_format == "explicit":
                exp_step_descriptions = list(
                    self.readme_dict[experiment]["Steps"].values()
                )
            elif readme_format == "implicit":
                exp_step_descriptions = self.readme_dict[experiment]["Steps"]
            else:
                exp_step_descriptions = []
            all_descriptions.append(exp_step_descriptions)
        return all_descriptions

    def get_cycle_details(self) -> List[List[Tuple[int, int, int]]]:
        """Get the cycle details from the README.yaml file."""
        cycles = []
        for experiment, readme_format, step_numbers in zip(
            self.readme_dict, self.readme_type, self.step_numbers
        ):
            exp_cycles = []
            if readme_format == "explicit":
                cycle_keys = [
                    key
                    for key in self.readme_dict[experiment].keys()
                    if "cycle" in key.lower()
                ]
                for cycle in cycle_keys:
                    cycle_dict = self.readme_dict[experiment][cycle]
                    start = cycle_dict["Start"]
                    end = cycle_dict["End"]
                    count = cycle_dict["Count"]
                    exp_cycles.append(
                        (step_numbers.index(start), step_numbers.index(end), count)
                    )
            cycles.append(exp_cycles)
        return cycles

    def get_pybamm_experiment_descriptions(self) -> List[Tuple[str, ...]]:
        """Get the PyBaMM experiment objects from the README.yaml file."""
        all_descriptions = []
        for step_descriptions, step_indices, step_numbers, cycle_details in zip(
            self.step_descriptions,
            self.step_indices,
            self.step_numbers,
            self.cycle_details,
        ):
            final_descriptions = []
            if len(step_descriptions) > 0:
                expanded_indices = self._expand_cycles(step_indices, cycle_details)
                expanded_descriptions = [step_descriptions[i] for i in expanded_indices]
                # split any descriptions seperated by commas
                for desciption in expanded_descriptions:
                    line = desciption.split(",")
                    for item in line:
                        final_descriptions.append(item.strip())
            all_descriptions.append(tuple(final_descriptions))
        return all_descriptions

    def get_pybamm_experiment_list(self) -> List[pybamm.Experiment]:
        """Get the PyBaMM experiment objects from the README.yaml file."""
        pybamm_experiments = []
        for experiment, descriptions in zip(
            self.readme_dict, self.pybamm_experiment_descriptions
        ):
            if len(descriptions) > 0:
                try:
                    pybamm_experiments.append(pybamm.Experiment(descriptions))
                except Exception as e:
                    warnings.warn(
                        f"PyBaMM experiment could not be created for experiment:"
                        f" {experiment}. {e}"
                    )
                    pybamm_experiments.append(None)
            else:
                pybamm_experiments.append(None)
                warnings.warn(
                    f"PyBaMM experiment could not be created for experiment:"
                    f" {experiment} as there are no step descriptions."
                )
        return pybamm_experiments

    def get_pybamm_experiment(self) -> Optional[pybamm.Experiment]:
        """Get the PyBaMM experiment object from the README.yaml file."""
        if any(exp is None for exp in self.pybamm_experiment_list):
            warnings.warn(
                "Some experiments do not have valid step descriptions."
                " Unable to create PyBaMM experiment."
            )
            return None
        else:
            all_descriptions = [exp for exp in self.pybamm_experiment_descriptions]
            return pybamm.Experiment(all_descriptions)

    @staticmethod
    def _expand_cycles(
        indices: List[int], cycles: List[Tuple[int, int, int]]
    ) -> List[int]:
        if len(cycles) == 0:
            return indices
        repeated_list = indices
        for cycle in cycles:
            # cycle = (start, end, repeats)
            start = cycle[0]
            end = cycle[1] + 1
            repeats = cycle[2]

            # get sublist
            sublist = indices[start:end]

            # repeat sublist
            repeated_sublist = sublist * repeats

            # insert repeated sublist into repeated_list
            result: List[int] = []
            i = 0
            while i < len(repeated_list):
                if repeated_list[i : i + len(sublist)] == sublist:
                    result.extend(repeated_sublist)
                    i += len(sublist)
                else:
                    result.append(repeated_list[i])
                    i += 1
            repeated_list = result
        return result
