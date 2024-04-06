"""Module for the Cell class."""
from typing import Callable
from pybatdata.batterycycler import BatteryCycler
import pandas as pd
import polars as pl
from pybatdata.procedure import Procedure
import os
import time
import distinctipy
import pickle
import subprocess

class Cell:
    """A class for a cell in a battery experiment.
    
    Attributes:
        metadata (dict): Rig and setup information on the cell e.g. cycler number, thermocouple channel.
        raw_data (dict): The raw data from each procedure conducted on the cell.
        processed_data (dict): A place to store processed data for each procedure for use later.    
    """

    def __init__(self, 
                 metadata: dict,
                 ):
        """Create a cell object.
        
        Args:
            metadata (dict): Rig and setup information on the cell e.g. cycler number, thermocouple channel.
        """
        self.metadata = metadata
        self.raw_data = {}
        self.processed_data = {}

    @classmethod
    def batch_process(cls,
                      root_directory: str,
                      record_name: str,
                      cycler: BatteryCycler,
                      filename_function: Callable,
                      filename_inputs: list,
                      cell_list: list=None,
                      title: str = None,
                      fast_mode: bool = False) -> list['Cell']:
        """Function to batch process all data files in a record.
        
        Args:
            root_directory (str): The root directory containing the Experiment_Record.xlsx file.
            record_name (str): The name of the record (worksheet name) in the Experiment_Record.xlsx file.
            cycler (BatteryCycler): The cycler used to produce the data.
            filename_function (function): The function to generate the data file name.
            filename_inputs (list): The list of inputs to filename_function. These must be keys of the cell metadata.
            cell_list (list, optional): An existing list of cell objects to append data to.
            title (str, optional): A custom title for the data in the Cell.raw_data dictionary.
            fast_mode (bool): Whether to skip rewriting the parquet files if it already exists.
        
        Returns:
            list: A list of cell objects with the data added.
        """
        print(f"Batch pre-processing running...")
        record = cls.read_record(root_directory, record_name)
        n_cells = len(record)
        if cell_list == None:
            cell_list = []
            colors = cls.set_color_scheme(n_cells, scheme='distinctipy')
            for i in range(n_cells):
                cell_list.append(cls(record.iloc[i].to_dict()))
                cell_list[i].color = colors[i]
        parquet_verified = False
        if title is None:
            title = record_name
        for i in range(n_cells):
            filename = cls.get_filename(cell_list[i].metadata, filename_function, filename_inputs)
            data_path = os.path.join(root_directory, record_name, filename)
            if fast_mode is True:
                if i == 0:
                    parquet_verified = cls.verify_parquet(data_path, cycler)
                if parquet_verified is True:
                    cell_list[i].add_data(data_path, 
                                        title,
                                        cycler,
                                        skip_writing=True)
                else:
                    cell_list[i].add_data(data_path, 
                                        title,
                                        cycler,
                                        skip_writing=False)
            else:
                cell_list[i].add_data(data_path, 
                                    title,
                                    cycler,
                                    skip_writing=False)
        return cell_list

    @staticmethod
    def read_record(root_directory, record_name)->pd.DataFrame:
        """Function to read the record of tests from the Experiment_Record.xlsx file.

        Args:
            root_directory (str): The root directory containing the Experiment_Record.xlsx file.
            record_name (str): The name of the record (worksheet name) in the Experiment_Record.xlsx file.
        
        Returns:
            pd.DataFrame: The record of tests run with this procedure.
        """
        record_xlsx = os.path.join(root_directory, "Experiment_Record.xlsx")
        return pd.read_excel(record_xlsx, sheet_name = record_name)

    @staticmethod
    def get_filename(metadata: dict, filename_function: Callable, filename_inputs: list)->str:
        """Function to generate the input name for the data file.
        
        Args:
            metadata (dict): The metadata entry for the data file.
            filename_function (function): The function to generate the input name.
            filename_inputs (list): The list of inputs to filename_function. These must be keys of the cell metadata.
            
        Returns:
            str: The input name for the data file.
        """
        return filename_function(*(metadata[filename_inputs[i]] for i in range(len(filename_inputs))))
    
    def add_data(self, 
                 input_path: str, 
                 title: str, 
                 cycler: BatteryCycler, 
                 skip_writing=False) -> None:
        """Function to add data to the cell object.
        
        Args:
            input_path (str): The path to the data file to add.
            title (str): The title to give the data in the raw_data dictionary.
            cycler (BatteryCycler): The cycler used to produce the data.
            skip_writing (bool): Whether to skip rewriting the parquet file if it already exists.
        """
        output_path = os.path.splitext(input_path)[0]+'.parquet'
        if (os.path.exists(output_path) is False or 
            skip_writing is False):
            self.write_parquet(input_path, output_path, cycler)
        self.raw_data[title] = Procedure(output_path)
        self.processed_data[title] = {}

    @staticmethod
    def verify_parquet(input_path: str, cycler: BatteryCycler) -> bool:
        """Function to verify that the data in a parquet file is correct.
        
        Args:
            input_path (str): The path to the input data file.
            cycler (BatteryCycler): The cycler used to produce the data.
        
        Returns:
            bool: True if the data is correct, False otherwise.
        """
        output_path = os.path.splitext(input_path)[0]+'.parquet'
        test_data = cycler.load_file(input_path).head()
        parquet_data = pl.scan_parquet(output_path).head().collect().to_pandas()
        try:
            pd.testing.assert_frame_equal(test_data, parquet_data)
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
        filepath = os.path.join(input_path)
        t1 = time.time()
        test_data = cycler.load_file(filepath)
        test_data.to_parquet(output_path, engine='pyarrow')
        print(f"\tparquet written in {time.time()-t1:.2f} seconds.")

    @staticmethod
    def set_color_scheme(n_colors, scheme='distinctipy', **kwargs) -> list[str]:
        """Function to set the colour scheme for plotting.
        
        Args:
            n_colors (int): The number of colors to produce.
            scheme (str): The colour scheme to use.
            **kwargs: Additional keyword arguments for the colour scheme.
            
        Returns:
            list: The list of colours in hex format."""
        if scheme == 'distinctipy':
            rgb = distinctipy.get_colors(n_colors, 
                                         exclude_colors=[(0,0,0), (1,1,1),(1,1,0)], # Exclude black, white, and yellow
                                         rng=0, # Set the random seed
                                         **kwargs,
                                         )
            hex = []
            for i in range(len(rgb)):
                hex.append(distinctipy.get_hex(rgb[i]))
            return hex

    @staticmethod
    def launch_dashboard(cell_list) -> None:
        """Function to launch the dashboard for the preprocessed data.
        
        Args:
            cell_list (list): The list of cell objects to display in the dashboard.
        """
        with open('dashboard_data.pkl', 'wb') as f:
            pickle.dump(cell_list, f)
        subprocess.Popen(["nohup", "streamlit", "run", os.path.join(os.path.dirname(__file__), "dashboard.py")], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
