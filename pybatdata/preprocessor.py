import numpy as np
import pandas as pd
import os
import re
import time
import polars as pl
from procedure import Procedure
from cyclers import Neware

class Preprocessor:
    def __init__(self, folderpath, test_name, cycler):
        self.folderpath = folderpath
        self.test_name = test_name
        self.record = self.read_record()
        self.column_headings = ['Date', 
                                'Time', 
                                'Cycle', 
                                'Step', 
                                'Current (A)',  
                                'Voltage (V)', 
                                'Capacity (Ah)', 
                                'Discharge Capacity (Ah)', 
                                'Charge Capacity (Ah)',]

        self.test_dict = [None]*len(self.record)
        readme_path = os.path.join(self.folderpath, self.test_name, 'README.txt')
        self.titles, self.steps, self.cycles, self.step_names = self.process_readme(readme_path)
        self.cycer_dict = {'Neware': Neware}
        self.cycler = self.cycer_dict[cycler]
        
    def run(self, filename, filename_inputs):
        print("Preprocessor running...")
        for record_entry in range(len(self.record)):
            self.test_dict[record_entry] = self.record.iloc[record_entry].to_dict()
            input_name = self.input_name(self.test_dict[record_entry], filename, filename_inputs)
            input_path = self.input_path(input_name)
            output_path = os.path.splitext(input_path)[0]+'.parquet'
            self.write_parquet(input_path, output_path)
            self.test_dict[record_entry]['Data'] = self.read_parquet(output_path)
        print("Preprocessor complete.")
        return self.test_dict
    
    @staticmethod
    def input_name(test_dict_entry, filename, filename_inputs):
        return filename(*(test_dict_entry[filename_inputs[i]] for i in range(len(filename_inputs))))
    
    def input_path(self, input_name):
        return os.path.join(self.folderpath, self.test_name, input_name)
    
    def read_record(self):
        record_xlsx = os.path.join(self.folderpath, "Experiment_Record.xlsx")
        return pd.read_excel(record_xlsx, sheet_name = self.test_name)
    
    def write_parquet(self, input_path, output_path):  
        if not os.path.exists(output_path):
            print(f"Processing file: {os.path.basename(input_path)}")
            filepath = os.path.join(input_path)
            t1 = time.time()
            test_data = self.cycler.load_file(filepath)
            test_data.to_parquet(output_path, engine='pyarrow')
            print(f"\tparquet written in {time.time()-t1:.2f} seconds.")
            
    def read_parquet(self, data_path):
        data = pl.scan_parquet(data_path)
        return Procedure(data, self.titles, self.cycles, self.steps, self.step_names)
    
    @staticmethod
    def process_readme(readme_path):
            """Function to process the README.md file and extract the relevant information."""
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

