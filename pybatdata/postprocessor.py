"""Module to process the README.md file and extract the relevant information."""

import re
import os
from procedure import Procedure
import preprocessor
import os
import polars as pl
  
class PostProcessor:
    def __init__(self, folderpath, test_name):
        self.PreProc = preprocessor.Preprocessor(folderpath, test_name)
        self.test_dict = self.PreProc.test_dict
        readme_path = os.path.join(folderpath, test_name, 'README.txt')
        self.titles, self.steps, self.cycles, self.step_names = self.process_readme(readme_path)
        for record_entry in range(len(self.test_dict)):
            data_path = self.PreProc.parquet_path(self.test_dict[record_entry]['Cycler'], self.test_dict[record_entry]['Channel'])
            self.test_dict[record_entry]['Data'] = self.from_parquet(data_path)
        
    def from_parquet(self, data_path):
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
    
def capacity_ref(df):
        return df.loc[(df['Current (A)'] == 0) & (df['Voltage (V)'] == df[df['Current (A)'] == 0]['Voltage (V)'].max()), 'Capacity (Ah)'].values[0]