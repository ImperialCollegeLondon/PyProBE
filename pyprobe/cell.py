"""Module for the Cell class."""
import os
import pickle
import platform
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import distinctipy
import polars as pl

from pyprobe.dataimporter import DataImporter
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
        root_directory: str,
        record_name: str,
    ) -> List["Cell"]:
        """Function to make a list of cell objects from a record of tests.

        Args:
            root_directory (str): The root directory containing the
                Experiment_Record.xlsx file.
            record_name (str): The name of the record (worksheet name)
                in the Experiment_Record.xlsx file.
        """
        record = cls.read_record(root_directory, record_name)

        n_cells = len(record)
        cell_list = []
        colors = cls.set_color_scheme(n_cells, scheme="distinctipy")
        for i in range(n_cells):
            cell_list.append(cls(record.row(i, named=True)))
            cell_list[i].info["color"] = colors[i]
        return cell_list

    def process_cycler_file(
        self,
        cycler: str,
        folder_path: str,
        filename: str | Callable[[str], str],
        filename_inputs: Optional[List[str]] = None,
    ) -> None:
        """Convert cycler file into PyProBE format.

        Args:
            cycler (DataImporter): The cycler used to produce the data.
            folder_path (str): The path to the folder containing the data file.
            filename (str | function): A filename string or a function to
                generate the file name.
            filename_inputs (list): The list of inputs to filename_function.
                These must be keys of the cell info.
            root_directory (str): The root directory containing the subfolder.
        """
        input_data_path, output_data_path = self.get_data_paths(
            folder_path, filename, filename_inputs
        )
        t1 = time.time()
        importer = DataImporter(cycler)
        dataframe = importer.read_file(input_data_path)
        dataframe = importer.process_dataframe(dataframe)
        dataframe.write_parquet(output_data_path)
        print(f"\tparquet written in {time.time()-t1:.2f} seconds.")

    def add_procedure(
        self,
        procedure_name: str,
        folder_path: str,
        filename: str | Callable[[str], str],
        filename_inputs: Optional[List[str]] = None,
    ) -> None:
        """Function to add data to the cell object.

        Args:
            procedure_name (str): A name to give the procedure. This will be used
                when calling cell.procedure[procedure_name].
            folder_path (str): The path to the folder containing the data file.
            filename (str | function): A filename string or a function to generate
                the file name.
            filename_inputs (Optional[list]): The list of inputs to filename_function.
                These must be keys of the cell info.
        """
        _, output_data_path = self.get_data_paths(
            folder_path, filename, filename_inputs
        )
        self.procedure[procedure_name] = Procedure(output_data_path, self.info)
        self.processed_data[procedure_name] = {}

    @staticmethod
    def read_record(root_directory: str, record_name: str) -> pl.DataFrame:
        """Function to read the record of tests from the Experiment_Record.xlsx file.

        Args:
            root_directory (str): The root directory containing the
                Experiment_Record.xlsx file.
            record_name (str): The name of the record (worksheet name)
                in the Experiment_Record.xlsx file.

        Returns:
            pl.DataFrame: The record of tests run with this procedure.
        """
        record_xlsx = os.path.join(root_directory, "Experiment_Record.xlsx")
        return pl.read_excel(record_xlsx, sheet_name=record_name)

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
    ) -> Tuple[str, str]:
        """Function to generate the input and output paths for the data file.

        Args:
            folder_path (str): The path to the folder containing the data file.
            filename (str | function): A filename string or a function to generate
                the file name.
            filename_inputs (Optional[list]): The list of inputs to filename_function.
                These must be keys of the cell info.

        Returns:
            Tuple[str, str]:
                - str: The input path for the data file.
                - str: The output path for the parquet file.
        """
        if isinstance(filename, str):
            filename_str = filename
        else:
            if filename_inputs is None:
                raise ValueError(
                    "filename_inputs must be provided when filename is a function."
                )
            filename_str = self.get_filename(self.info, filename, filename_inputs)

        input_data_path = os.path.join(folder_path, filename_str)
        output_data_path = os.path.splitext(input_data_path)[0] + ".parquet"
        return input_data_path, output_data_path

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
