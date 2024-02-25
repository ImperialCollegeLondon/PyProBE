#%%
import pandas as pd
import re

loc = "/Users/tom/Library/CloudStorage/OneDrive-ImperialCollegeLondon/PhD Workspace/05 - Parallel SLP/DATA"

# Load the data


def convert_units(df):
    conversion_dict = {'m': 1e-3, 'µ': 1e-6, 'n': 1e-9, 'p': 1e-12}

    for column in df.columns:
        match = re.search(r'\((.*?)\)', column)
        if match:
            unit = match.group(1)
            prefix = next((x for x in unit if not x.isupper()), None)
            if prefix in conversion_dict:
                df[column] = df[column] * conversion_dict[prefix]
                df.rename(columns={column: column.replace('('+unit+')', '('+unit.replace(prefix, '')+')')}, inplace=True)

    return df

def read_data(filepath):
    df = pd.read_csv(filepath)
    column_dict = {'Date': 'Date', 'Time': 'Time', 'Cycle Index': 'Cycle', 'Step Index': 'Step', 'Current(A)': 'Current (A)', 'Voltage(V)': 'Voltage (V)', 'Capacity(Ah)': 'Capacity (Ah)', 'dQ/dV(Ah/V)': 'dQ/dV (Ah/V)'}
    df = convert_units(df)
    df = df[list(column_dict.keys())].rename(columns=column_dict)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Time'] = pd.to_timedelta(df['Time']).dt.total_seconds()
    return df

# %%

import pandas as pd
import re

# Read README.txt and extract titles and step numbers
with open(loc + '/README.txt', 'r') as file:
    lines = file.readlines()

titles = []


title_index = 0
for line in lines:
    if line.startswith('##'):    
        titles.append(line[3:].strip())

steps = [[[]] for _ in range(len(titles))]
cycles = [[] for _ in range(len(titles))]
#%%
line_index = 0
title_index = -1
cycle_index = 0
cycle_numbers = []
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

#cycles = [[len(sublist) for sublist in group] for group in steps]
cycles = [list(range(len(sublist))) for sublist in steps]
for i in range(len(cycles)-1):
    cycles[i+1] = [item+cycles[i][-1] for item in cycles[i+1]]
    cycles[i] = [item+1 for item in cycles[i]]

step_names = [None for _ in range(steps[-1][-1][-1]+1)]
line_index = 0
while line_index < len(lines):
    if lines[line_index].startswith('#-'):    
        match = re.search(r'Step (\d+)', lines[line_index])
        if match: 
            step_names[int(match.group(1))] = lines[line_index].split(': ')[1].strip()
    line_index += 1

#%%
def flatten(lst):
    if not isinstance(lst, list):
        return [lst]
    if lst == []:
        return lst
    if isinstance(lst[0], list):
        return flatten(lst[0]) + flatten(lst[1:])
    return lst[:1] + flatten(lst[1:])


class Experiment:
    def __init__(self, data, cycles_idx, steps_idx, step_names):
        self.data = data
        self.cycles_idx = cycles_idx
        self.steps_idx = steps_idx
        self.step_names = step_names
    
    @property
    def RawData(self):
        return self.data[(self.data['Cycle'].isin(flatten(self.cycles_idx))) & (self.data['Step'].isin(flatten(self.steps_idx)))]

    def cycle(self,cycle_number):
        return Cycle(self.data, self.cycles_idx[cycle_number-1], self.steps_idx[cycle_number-1], self.step_names)
    
class Cycle:
    def __init__(self, data, cycles_idx, steps_idx, step_names):
        self.data = data
        self.cycles_idx = cycles_idx
        self.steps_idx = steps_idx
        self.step_names = step_names

    @property
    def RawData(self):
        return self.data[(self.data['Cycle'].isin(flatten(self.cycles_idx))) & (self.data['Step'].isin(flatten(self.steps_idx)))]

    def step(self, step_number):
        return Step(self.data, self.cycles_idx, self.steps_idx[step_number-1], self.step_names)

class Step:
    def __init__(self, data, cycles_idx, steps_idx, step_names):
        self.data = data
        self.cycles_idx = cycles_idx
        self.steps_idx = steps_idx
        self.step_names = step_names

    @property
    def RawData(self):
        return self.data[(self.data['Cycle'].isin(flatten(self.cycles_idx))) & (self.data['Step'].isin(flatten(self.steps_idx)))]

    @property
    def capacity(self):
        return self.RawData['Capacity (Ah)'].max()

cells = []
for cycler in range(4,10):
    for channel in range(1,8):
        filename = f'SLP_Cell-Cell_Variation_R1_{cycler}_{channel}.csv'
        df = read_data(loc + '/' + filename)
        experiments = []
        for i in range(len(titles)):
            experiments.append(Experiment(df, cycles[i], steps[i], step_names))
        cells.append(experiments)

import matplotlib.pyplot as plt

capacities = []
for i in range(len(cells)):
    capacities.append([cells[i][1].cycle(5).step(1).capacity])

#%%
plt.boxplot(flatten(capacities))

#%%
class Step:
    def __init__(self, data):
        self.RawData = data

    @property
    def capacity(self):
        return self.RawData['Capacity (Ah)'].max()
    
section = df[(df['Cycle'] == 1) & (df['Step'] == 4)]
step = Step(section)

step_objects = []
for title_steps in steps:
    title_objects = []
    for cycle_steps in title_steps:
        cycle_objects = []
        for step_number in cycle_steps:
            section = df[(df['Cycle'] == cycle_number) & (df['Step'] == step_number)]
            step = Step(section)
            cycle_objects.append(step)
        title_objects.append(cycle_objects)
    step_objects.append(title_objects)




# %%
