"""Module for the Preprocessor class."""

import pandas as pd
import os
import re
import time
import polars as pl
from pybatdata.procedure import Procedure
from pybatdata.cyclers import Neware
from typing import Callable
import pickle
import subprocess

class Preprocessor:
    """A preprocessor class to write to and read from parquet data files.
    
    Attributes:
        folderpath (str): The path to the folder containing the data.
        procedure_name (str): The name of the procedure being processed.
        record (pd.DataFrame): The metadata record of the procedure.
        procedure_dict (list): The list of dictionaries containing the metadata and data for each test run with a procedure.
        titles (dict): The titles of the experiments inside a procddure. Fomat {title: experiment type}.
        steps (list): The step numbers inside the procedure.
        cycles (list): The cycle numbers inside the procedure.
        step_names (list): The names of the steps inside the procedure.
        cycler_dict (dict): The dictionary of cyclers that can be used.
        cycler (str): The cycler used for this procedure.
        """
    def __init__(self, folderpath: str, procedure_name: str, cycler: str):
        """Create a preprocessor object.
        
        Args:
            folderpath (str): The path to the folder containing the data.
            procedure_name (str): The name of the procedure being processed.
            cycler (str): The cycler used for this procedure.
        """
        self.folderpath = folderpath
        self.procedure_name = procedure_name
        self.record = self.read_record()
        self.procedure_dict = [None]*len(self.record)
        readme_path = os.path.join(self.folderpath, self.procedure_name, 'README.txt')
        self.titles, self.steps, self.cycles, self.step_names = self.process_readme(readme_path)
        self.cycer_dict = {'Neware': Neware}
        self.cycler = self.cycer_dict[cycler]
        
    def run(self, filename_function: Callable, filename_inputs: list)->list[dict]:
        """Run the preprocessor.
        
        Args:
            filename_function (function): The function to generate the input name.
            filename_inputs (list): The list of inputs to filename_function.

        Returns:
            list[dict]: The list of dictionaries containing the metadata and data for each test run with a procedure.
        """
        print("Preprocessor running...")
        parquets_verified = False
        for record_entry in range(len(self.record)):
            self.procedure_dict[record_entry] = self.record.iloc[record_entry].to_dict()
            filename = self.get_filename(self.procedure_dict[record_entry], filename_function, filename_inputs)
            filepath = self.get_filepath(filename)
            output_path = os.path.splitext(filepath)[0]+'.parquet'
            if record_entry == 0 and self.verify_parquet(filepath, output_path) == True:
                parquets_verified = True
            if parquets_verified == False:
                self.write_parquet(filepath, output_path)
            self.procedure_dict[record_entry]['Data'] = self.read_parquet(output_path)
        print("Preprocessor complete.")
        return self.procedure_dict
    
    def launch_dashboard(self):
        """Function to launch the dashboard for the preprocessed data."""
        # Serialize procedure_dict to a file
        with open('procedure_dict.pkl', 'wb') as f:
            pickle.dump(self.procedure_dict, f)
        subprocess.run(["streamlit", "run", os.path.join(os.path.dirname(__file__), "dashboard.py")])
    
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
    
    def get_filepath(self, input_name: str)->str:
        """Function to generate the path to the data file.
        
        Args:
            input_name (str): The name of the data file.

        Returns:
            str: The path to the data file.
        """
        return os.path.join(self.folderpath, self.procedure_name, input_name)
    
    def read_record(self)->pd.DataFrame:
        """Function to read the record of tests run with this procedure from the Experiment_Record.xlsx file.
        
        Returns:
            pd.DataFrame: The record of tests run with this procedure.
        """
        record_xlsx = os.path.join(self.folderpath, "Experiment_Record.xlsx")
        return pd.read_excel(record_xlsx, sheet_name = self.procedure_name)
    
    def verify_parquet(self, input_path: str, output_path: str)->bool:
        """Function to verify that the data in the parquet file is correct.
        
        Args:
            input_path (str): The path to the input data file.
            output_path (str): The path to the output parquet file.
        
        Returns:
            bool: True if the data is correct, False otherwise.
        """
        if not os.path.exists(output_path):
            return False
        else:
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
            
    def read_parquet(self, data_path: str)->Procedure:
        """Function to read the data from a parquet file and create a Procedure object.
        
        Args:
            data_path (str): The path to the parquet file.
        
        Returns:
            Procedure: The Procedure object created from the parquet data.
        """
        data = pl.scan_parquet(data_path)
        return Procedure(data, self.titles, self.cycles, self.steps, self.step_names)
    
    @staticmethod
    def process_readme(readme_path):
            """Function to process the README.txt file and extract the relevant information.
            
            Args:
                readme_path (str): The path to the README.txt file.
                
            Returns:
                dict: The titles of the experiments inside a procddure. Fomat {title: experiment type}.
                list: The step numbers inside the procedure.
                list: The cycle numbers inside the procedure.
                list: The names of the steps inside the procedure.
            """
            with open(readme_path, 'r') as file:
                lines = file.readlines()

            titles = {}
            title_index = 0
            for line in lines:
                if line.startswith('##'):    
                    splitted_line = line[3:].split(":")
                    titles[splitted_line[0].strip()] = splitted_line[1].strip()

            steps = [[[]] for _ in range(len(titles))]
            cycles = [[] for _ in range(len(titles))]
            line_index = 0
            title_index = -1
            cycle_index = 0
            while line_index < len(lines):
                if lines[line_index].startswith('##'):    
                    title_index += 1
                    cycle_index = 0
                if lines[line_index].startswith('#-'):
                    match = re.search(r'Step (\d+)', lines[line_index])
                    if match:
                        steps[title_index][cycle_index].append(int(match.group(1)))  # Append step number to the corresponding title's list
                    latest_step = int(match.group(1))
                if lines[line_index].startswith('#x'):
                    line_index += 1
                    match = re.search(r'Starting step: (\d+)', lines[line_index])
                    if match:
                        starting_step = int(match.group(1))
                    line_index += 1
                    match = re.search(r'Cycle count: (\d+)', lines[line_index])
                    if match:
                        cycle_count = int(match.group(1))
                    for i in range(cycle_count-1):
                        steps[title_index].append(list(range(starting_step, latest_step+1)))
                        cycle_index += 1
                line_index += 1

            cycles = [list(range(len(sublist))) for sublist in steps]
            for i in range(len(cycles)-1):
                cycles[i+1] = [item+cycles[i][-1] for item in cycles[i+1]]
            for i in range(len(cycles)): 
                cycles[i] = [item+1 for item in cycles[i]]
            
            step_names = [None for _ in range(steps[-1][-1][-1]+1)]
            line_index = 0
            while line_index < len(lines):
                if lines[line_index].startswith('#-'):    
                    match = re.search(r'Step (\d+)', lines[line_index])
                    if match: 
                        step_names[int(match.group(1))] = lines[line_index].split(': ')[1].strip()
                line_index += 1
                
            return titles, steps, cycles, step_names

    # def launch_dashboard(self):
    #     """Function to launch the dashboard for the preprocessed data."""
    #     import streamlit as st
    #     #from pybatdata.dashboard import launch
    #     dash = Dashboard(self.procedure_dict)
    #     dash.launch()