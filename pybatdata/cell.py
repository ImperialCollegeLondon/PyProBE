from typing import Callable
from pybatdata.cycler import Cycler
import pandas as pd
import polars as pl
from pybatdata.procedure import Procedure
import os
import time

class Cell:
    def __init__(self, 
                 record_entry: pd.DataFrame, 
                 cycler: Cycler,
                 ):
        self.metadata = record_entry.to_dict()
        self.data = {}
        self.cycler = cycler

    def read_record(root_directory, record_name)->pd.DataFrame:
        """Function to read the record of tests run with this procedure from the Experiment_Record.xlsx file.
        
        Returns:
            pd.DataFrame: The record of tests run with this procedure.
        """
        record_xlsx = os.path.join(root_directory, "Experiment_Record.xlsx")
        return pd.read_excel(record_xlsx, sheet_name = record_name)

    @staticmethod
    def get_filename(procedure_dict_entry: dict, filename_function: Callable, filename_inputs: list)->str:
        """Function to generate the input name for the data file.
        
        Args:
            procedure_dict_entry (dict): The metadata entry for the data file.
            filename_function (function): The function to generate the input name.
            filename_inputs (list): The list of inputs to filename_function. These must be keys of procedure_dict_entry.
            
        Returns:
            str: The input name for the data file.
        """
        return filename_function(*(procedure_dict_entry[filename_inputs[i]] for i in range(len(filename_inputs))))
    
    def add_data(self, input_path: str, title: str, readme_info: tuple = None, verification: bool=True):
        output_path = os.path.splitext(input_path)[0]+'.parquet'
        if (os.path.exists(output_path) is False or 
            verification is True and self.verify_parquet(input_path, output_path) is False):
            self.write_parquet(input_path, output_path)
        self.data[title] = Procedure(output_path, readme_info)

    def verify_parquet(self, input_path: str, output_path: str)->bool:
        """Function to verify that the data in the parquet file is correct.
        
        Args:
            input_path (str): The path to the input data file.
            output_path (str): The path to the output parquet file.
        
        Returns:
            bool: True if the data is correct, False otherwise.
        """
        test_data = self.cycler.load_file(input_path).head()
        parquet_data = pl.scan_parquet(output_path).head().collect().to_pandas()
        try:
            pd.testing.assert_frame_equal(test_data, parquet_data)
            return True
        except AssertionError:
            return False
    
    def write_parquet(self, input_path: str, output_path: str)->None:
        """Function to write the data to a parquet file.
        
        Args:
            input_path (str): The path to the input data file.
            output_path (str): The path to the output parquet file.
        """  
        print(f"Processing file: {os.path.basename(input_path)}")
        filepath = os.path.join(input_path)
        t1 = time.time()
        test_data = self.cycler.load_file(filepath)
        test_data.to_parquet(output_path, engine='pyarrow')
        print(f"\tparquet written in {time.time()-t1:.2f} seconds.")

