"""Module for the Cell class."""
import os
import pickle
import platform
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional

import distinctipy
import polars as pl

from pyprobe.cyclers import biologic, neware
from pyprobe.procedure import Procedure


class Cell:
    """A class for a cell in a battery experiment.

    Attributes:
        info (dict): Rig and setup information on the cell
            e.g. cycler number, thermocouple channel.
        procedure (dict): The raw data from each procedure conducted on the cell.
        processed_data (dict): A place to store processed data for each procedure
            for use later.
    """

    def __init__(
        self,
        info: Dict[str, str | int | float],
    ):
        """Create a cell object.

        Args:
            info (dict): Rig and setup information on the cell
                e.g. cycler number, thermocouple channel.
        """
        self.info = info
        self.info["color"] = distinctipy.get_hex(
            distinctipy.get_colors(
                1,
                exclude_colors=[
                    (0, 0, 0),
                    (1, 1, 1),
                    (1, 1, 0),
                ],
            )[0]
        )
        self.procedure: Dict[str, Procedure] = {}
        self.processed_data: Dict[str, pl.DataFrame] = {}

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
        """
        record = pl.read_excel(record_filepath, sheet_name=worksheet_name)

        n_cells = len(record)
        cell_list = []
        colors = cls.set_color_scheme(n_cells, scheme="distinctipy")
        for i in range(n_cells):
            cell_list.append(cls(record.row(i, named=True)))
            cell_list[i].info["color"] = colors[i]
        return cell_list

    @staticmethod
    def verify_parquet(filename: str) -> str:
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

    def process_cycler_file(
        self,
        cycler: str,
        folder_path: str,
        input_filename: str | Callable[[str], str],
        output_filename: str | Callable[[str], str],
        filename_args: Optional[List[str]] = None,
    ) -> None:
        """Convert cycler file into PyProBE format.

        Args:
            cycler (str): The cycler used to produce the data.
            folder_path (str): The path to the folder containing the data file.
            input_filename (str | function): A filename string or a function to
                generate the file name for cycler data.
            output_filename (str | function): A filename string or a function to
                generate the file name for PyProBE data.
            filename_args (list): The list of inputs to filename_function.
                These must be keys of the cell info.
        """
        input_data_path = self.get_data_paths(
            folder_path, input_filename, filename_args
        )
        output_data_path = self.get_data_paths(
            folder_path, output_filename, filename_args
        )
        output_data_path = self.verify_parquet(output_data_path)
        if "*" in output_data_path:
            raise ValueError("* characters are not allowed for a complete data path.")
        cycler_dict = {"neware": neware.Neware, "biologic": biologic.Biologic}
        t1 = time.time()
        importer = cycler_dict[cycler](input_data_path)
        dataframe = importer.pyprobe_dataframe
        dataframe.write_parquet(output_data_path)
        print(f"\tparquet written in {time.time()-t1:.2f} seconds.")

    def add_procedure(
        self,
        procedure_name: str,
        folder_path: str,
        filename: str | Callable[[str], str],
        filename_inputs: Optional[List[str]] = None,
        custom_readme_name: Optional[str] = None,
    ) -> None:
        """Function to add data to the cell object.

        Args:
            procedure_name (str): A name to give the procedure. This will be used
                when calling cell.procedure[procedure_name].
            folder_path (str): The path to the folder containing the data file.
            filename (str | function): A filename string or a function to generate
                the file name for PyProBE data.
            filename_inputs (Optional[list]): The list of inputs to filename_function.
                These must be keys of the cell info.
            custom_readme_name (str, optional): The name of the custom README file.
        """
        output_data_path = self.get_data_paths(folder_path, filename, filename_inputs)
        output_data_path = self.verify_parquet(output_data_path)
        if "*" in output_data_path:
            raise ValueError("* characters are not allowed for a complete data path.")
        self.procedure[procedure_name] = Procedure(
            output_data_path, self.info, custom_readme_name=custom_readme_name
        )
        self.processed_data[procedure_name] = {}

    @staticmethod
    def get_filename(
        info: Dict[str, str | int | float],
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

    def get_data_paths(
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
            filename_str = self.get_filename(self.info, filename, filename_inputs)

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
