"""Module to process the README.md file and extract the relevant information."""

import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from procedure import Procedure
import json
  
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

class Neware: 
    @classmethod
    def import_csv(cls,filepath):
        df = pd.read_csv(filepath)
        column_dict = {'Date': 'Date', 'Time': 'Time', 'Cycle Index': 'Cycle', 'Step Index': 'Step', 'Current(A)': 'Current (A)', 'Voltage(V)': 'Voltage (V)', 'Capacity(Ah)': 'Capacity (Ah)', 'DChg. Cap.(Ah)': 'Discharge Capacity (Ah)', 'Chg. Cap.(Ah)': 'Charge Capacity (Ah)','dQ/dV(Ah/V)': 'dQ/dV (Ah/V)'}
        df = cls.convert_units(df)
        df = df[list(column_dict.keys())].rename(columns=column_dict)
        dQ_charge = np.diff(df['Charge Capacity (Ah)'])
        dQ_discharge = np.diff(df['Discharge Capacity (Ah)'])
        dQ_charge[dQ_charge < 0] = 0
        dQ_discharge[dQ_discharge < 0] = 0
        dQ_charge = np.append(dQ_charge, 0)
        dQ_discharge = np.append(dQ_discharge, 0)
        df['Capacity (Ah)'] = np.cumsum(dQ_charge - dQ_discharge)
        df['Capacity (Ah)'] = df['Capacity (Ah)'] + df['Charge Capacity (Ah)'].max()
        df['Exp Capacity (Ah)'] = np.zeros(len(df))
        df['Cycle Capacity (Ah)'] = np.zeros(len(df))
        df['Date'] = pd.to_datetime(df['Date'])
        df['Time'] = pd.to_timedelta(df['Time']).dt.total_seconds()
        return df
    
    @staticmethod
    def convert_units(df):
        conversion_dict = {'m': 1e-3, 'µ': 1e-6, 'n': 1e-9, 'p': 1e-12}
        for column in df.columns:
            match = re.search(r'\((.*?)\)', column)  # Update the regular expression pattern to extract the unit from the column name
            if match:
                unit = match.group(1)
                prefix = next((x for x in unit if not x.isupper()), None)
                if prefix in conversion_dict:
                    df[column] = df[column] * conversion_dict[prefix]
                    df.rename(columns={column: column.replace('('+unit+')', '('+unit.replace(prefix, '')+')')}, inplace=True)  # Update the logic for replacing the unit in the column name

        return df

def process_readme(loc):
        """Function to process the README.md file and extract the relevant information."""
        with open(loc + '/README.txt', 'r') as file:
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