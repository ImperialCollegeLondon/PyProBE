"""Module to process the README.md file and extract the relevant information."""

import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from procedure import Procedure
import json
import os
import time
  
class PostProcessor:
    def __init__(self, preprocessor, input):
        print("Postprocessor running...")
        data = []
        metadata = []
        self.folderpath = preprocessor.folderpath
        self.test_name = preprocessor.test_name
        self.record = preprocessor.record
        self.data_path = preprocessor.data_path
        self.results = pd.DataFrame(index = range(len(self.record)), columns = input.keys())
        for record_entry in range(len(self.record)):
            cell_results = pd.DataFrame()
            print(f"Processing record {record_entry+1} of {len(self.record)}:")
            test_details = self.record.iloc[record_entry].to_dict()
            if test_details['Notes'] == 'Internal Short' or test_details['Notes']  == 'Internal short':
                continue
            cell_num = int(test_details['Cell'])
            cell_name = f"Cell_{cell_num}"
            if os.path.exists(self.results_path(cell_name)): # if results file exists
                cell_results = pd.read_parquet(self.results_path(cell_name))
                if set(cell_results.keys()) == set(self.results.keys()): # if results file has the same columns as the input
                    print("\tResults file up to date.")
                    self.results.loc[record_entry] = cell_results.loc[0]
                else: # if results file has different columns to the input
                    t1 = time.time()
                    data, metadata = self.read_rawdata(cell_name)
                    results_keys = set(self.results.keys())
                    cell_results_keys = set(cell_results.keys())
                    # Keys in results that are not in cell_results
                    missing_in_cell_results = results_keys - cell_results_keys
                    for column in missing_in_cell_results:
                        cell_results.loc[0,column] = eval(input[column], {'data': data, 'metadata': metadata})
                    # Keys in cell_results that are not in results
                    missing_in_results = cell_results_keys - results_keys
                    for key in missing_in_results:
                        del cell_results[key]
                    cell_results.to_parquet(self.results_path(cell_name), engine='pyarrow')
                    print(f"\tResults file updated in {time.time()-t1:.2f}s.")
                    self.results.loc[record_entry] = cell_results.loc[0]
            else: 
                t1 = time.time()
                data, metadata = self.read_rawdata(cell_name)
                cell_results = pd.DataFrame(columns = input.keys())
                for column in input.keys():
                    cell_results.loc[0,column] = eval(input[column], {'data': data, 'metadata': metadata})
                cell_results.to_parquet(self.results_path(cell_name), engine='pyarrow')
                self.results.loc[record_entry] = cell_results.loc[0]
                
                print(f"\tResults file created in {time.time()-t1:.2f}s.")
    
    def results_path(self, cell_name):
        return os.path.join(self.folderpath, f"{cell_name}", f'{self.test_name}_results.parquet')
    
    @property
    def readme_path(self):
        return os.path.join(self.folderpath, self.test_name, 'README.txt')
    
    def read_rawdata(self, cell_name):
        data = pd.read_parquet(self.data_path(cell_name), engine='pyarrow')
        titles, steps, cycles, step_names = process_readme(self.readme_path)
        metadata_filename = f'{self.test_name}_details.json'
        with open(os.path.join(self.folderpath, cell_name, metadata_filename)) as f:
            metadata = json.load(f)
        return Procedure(data, titles, cycles, steps, step_names), metadata

class DataLoader:
    @classmethod
    def from_csv(cls, filepath):
        data = cls.import_csv(filepath)
        titles, steps, cycles, step_names = process_readme(os.path.dirname(filepath))
        return Procedure(data, titles, cycles, steps, step_names)
    
    @classmethod
    def from_parquet(cls, directory, test_name, cell_name):
        data = pd.read_parquet(os.path.join(directory, cell_name, f'{test_name}.parquet'), engine='pyarrow')
        titles, steps, cycles, step_names = process_readme(os.path.join(directory, test_name))
        metadata_filename = f'{test_name}_details.json'
        with open(os.path.join(directory, cell_name, metadata_filename)) as f:
            metadata = json.load(f)
        return Procedure(data, titles, cycles, steps, step_names), metadata



def process_readme(loc):
        """Function to process the README.md file and extract the relevant information."""
        with open(loc, 'r') as file:
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
    
def capacity_ref(df):
        return df.loc[(df['Current (A)'] == 0) & (df['Voltage (V)'] == df[df['Current (A)'] == 0]['Voltage (V)'].max()), 'Capacity (Ah)'].values[0]