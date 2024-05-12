"""Module for the Cell class."""
import os
import pickle
import platform
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional

import distinctipy
import polars as pl
from polars.testing import assert_frame_equal

from pybatdata.batterycycler import BatteryCycler
from pybatdata.procedure import Procedure


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
        cycler: BatteryCycler,
        folder_path: str,
        filename_function: Callable[[str], str],
        filename_inputs: List[str],
    ) -> None:
        """Convert cycler file into PyBatData format.

        Args:
            cycler (BatteryCycler): The cycler used to produce the data.
            folder_path (str): The path to the folder containing the data file.
            filename_function (function): The function to generate the file name.
            filename_inputs (list): The list of inputs to filename_function.
                These must be keys of the cell info.
            root_directory (str): The root directory containing the subfolder.
        """
        filename = self.get_filename(self.info, filename_function, filename_inputs)
        input_data_path = os.path.join(folder_path, filename)
        output_data_path = os.path.splitext(input_data_path)[0] + ".parquet"
        self.write_parquet(input_data_path, output_data_path, cycler)

    def add_data(
        self,
        title: str,
        folder_path: str,
        filename: str | Callable[[str], str],
        filename_inputs: Optional[List[str]] = None,
    ) -> None:
        """Function to add data to the cell object.

        Args:
            title (str): The title of the procedure.
            folder_path (str): The path to the folder containing the data file.
            filename (str | function): The function to generate the file name.
            filename_inputs (Optional[list]): The list of inputs to filename_function.
                These must be keys of the cell info.
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
        self.procedure[title] = Procedure(output_data_path, self.info)
        self.processed_data[title] = {}

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

    @staticmethod
    def verify_parquet(input_path: str, cycler: BatteryCycler) -> bool:
        """Function to verify that the data in a parquet file is correct.

        Args:
            input_path (str): The path to the input data file.
            cycler (BatteryCycler): The cycler used to produce the data.

        Returns:
            bool: True if the data is correct, False otherwise.
        """
        output_path = os.path.splitext(input_path)[0] + ".parquet"
        if os.path.exists(output_path) is False:
            return False
        test_data = cycler.load_file(input_path).head()
        parquet_data = pl.scan_parquet(output_path).head()
        try:
            assert_frame_equal(test_data, parquet_data)
            return True
        except AssertionError:
            return False

    @staticmethod
    def write_parquet(input_path: str, output_path: str, cycler: BatteryCycler) -> None:
        """Function to write the data to a parquet file.

        Args:
            input_path (str): The path to the input data file.
            output_path (str): The path to the output parquet file.
            cycler (BatteryCycler): The cycler used to produce the data.
        """
        print(f"Processing file: {os.path.basename(input_path)}")
        output_path = os.path.splitext(input_path)[0] + ".parquet"
        filepath = os.path.join(input_path)
        t1 = time.time()
        test_data = cycler.load_file(filepath)
        if isinstance(test_data, pl.LazyFrame):
            test_data = test_data.collect()
        test_data.write_parquet(output_path)
        print(f"\tparquet written in {time.time()-t1:.2f} seconds.")

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
