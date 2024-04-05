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
    def __init__(self, 
                 metadata: dict,
                 ):
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
                     fast_mode: bool = False):
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
        for i in range(n_cells):
            filename = cls.get_filename(cell_list[i].metadata, filename_function, filename_inputs)
            data_path = os.path.join(root_directory, record_name, filename)
            if fast_mode is True:
                if i == 0:
                    parquet_verified = cls.verify_parquet(data_path, cycler)
                if parquet_verified is True:
                    cell_list[i].add_data(data_path, 
                                        title if title is not None else record_name,
                                        cycler,
                                        skip_writing=True)
                else:
                    cell_list[i].add_data(data_path, 
                                        title if title is not None else record_name,
                                        cycler,
                                        skip_writing=False)
            else:
                cell_list[i].add_data(data_path, 
                                    title if title is not None else record_name,
                                    cycler,
                                    skip_writing=False)
        return cell_list

    @staticmethod
    def read_record(root_directory, record_name)->pd.DataFrame:
        """Function to read the record of tests run with this procedure from the Experiment_Record.xlsx file.
        
        Returns:
            pd.DataFrame: The record of tests run with this procedure.
        """
        record_xlsx = os.path.join(root_directory, "Experiment_Record.xlsx")
        return pd.read_excel(record_xlsx, sheet_name = record_name)

    @staticmethod
    def get_filename(metadata: dict, filename_function: Callable, filename_inputs: list)->str:
        """Function to generate the input name for the data file.
        
        Args:
            procedure_dict_entry (dict): The metadata entry for the data file.
            filename_function (function): The function to generate the input name.
            filename_inputs (list): The list of inputs to filename_function. These must be keys of procedure_dict_entry.
            
        Returns:
            str: The input name for the data file.
        """
        return filename_function(*(metadata[filename_inputs[i]] for i in range(len(filename_inputs))))
    
    def add_data(self, input_path: str, title: str, cycler: BatteryCycler, skip_writing=False):
        output_path = os.path.splitext(input_path)[0]+'.parquet'
        if (os.path.exists(output_path) is False or 
            skip_writing is False):
            self.write_parquet(input_path, output_path, cycler)
        self.raw_data[title] = Procedure(output_path)
        self.processed_data[title] = {}

    @staticmethod
    def verify_parquet(input_path: str, cycler: BatteryCycler)->bool:
        """Function to verify that the data in the parquet file is correct.
        
        Args:
            input_path (str): The path to the input data file.
            output_path (str): The path to the output parquet file.
        
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
    
    def write_parquet(self, input_path: str, output_path: str, cycler: BatteryCycler)->None:
        """Function to write the data to a parquet file.
        
        Args:
            input_path (str): The path to the input data file.
            output_path (str): The path to the output parquet file.
        """  
        print(f"Processing file: {os.path.basename(input_path)}")
        filepath = os.path.join(input_path)
        t1 = time.time()
        test_data = cycler.load_file(filepath)
        test_data.to_parquet(output_path, engine='pyarrow')
        print(f"\tparquet written in {time.time()-t1:.2f} seconds.")

    @staticmethod
    def set_color_scheme(n_cells, scheme='distinctipy', **kwargs):
        """Function to set the colour scheme for plotting."""
        if scheme == 'distinctipy':
            rgb = distinctipy.get_colors(n_cells, 
                                         exclude_colors=[(0,0,0), (1,1,1),(1,1,0)], # Exclude black, white, and yellow
                                         rng=0, # Set the random seed
                                         **kwargs,
                                         )
            hex = []
            for i in range(len(rgb)):
                hex.append(distinctipy.get_hex(rgb[i]))
            return hex

    def launch_dashboard(cell_list):
        """Function to launch the dashboard for the preprocessed data."""
        # Serialize procedure_dict to a file
        with open('dashboard_data.pkl', 'wb') as f:
            pickle.dump(cell_list, f)
        subprocess.run(["streamlit", "run", os.path.join(os.path.dirname(__file__), "dashboard.py")])