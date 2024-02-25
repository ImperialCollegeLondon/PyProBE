#%%
import pandas as pd
import re

loc = "/Users/tom/Library/CloudStorage/OneDrive-ImperialCollegeLondon/PhD Workspace/05 - Parallel SLP/DATA"

# Load the data
df = pd.read_csv(loc + "/SLP_Cell-Cell_Variation_R1_4_1.csv")

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

column_dict = {'Date': 'Date', 'Time': 'Time', 'Cycle Index': 'Cycle', 'Step Index': 'Step', 'Current(A)': 'Current (A)', 'Voltage(V)': 'Voltage (V)', 'Capacity(Ah)': 'Capacity (Ah)', 'dQ/dV(Ah/V)': 'dQ/dV (Ah/V)'}
df = convert_units(df)
df = df[list(column_dict.keys())].rename(columns=column_dict)
df['Date'] = pd.to_datetime(df['Date'])
df['Time'] = pd.to_timedelta(df['Time']).dt.total_seconds()
print(df.head())
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
print(steps)
#%%
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
        for i in range(cycle_count):
            steps[title_index].append(list(range(starting_step, latest_step+1)))
            cycle_index += 1
    line_index += 1
        
step_names = [None for _ in range(steps[-1][-1][-1]+1)]
line_index = 0
while line_index < len(lines):
    if lines[line_index].startswith('#-'):    
        match = re.search(r'Step (\d+)', lines[line_index])
        if match: 
            step_names[int(match.group(1))] = lines[line_index].split(': ')[1].strip()
    line_index += 1



# %%
