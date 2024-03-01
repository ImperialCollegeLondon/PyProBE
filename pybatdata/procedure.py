import numpy as np
import pandas as pd
from experiment import Experiment, Pulsing

class Procedure:
    def __init__(self, data, titles, cycles_idx, steps_idx, step_names):
        self.RawData = data.set_index(['Cycle', 'Step'])
        self.titles = titles
        self.cycles_idx = cycles_idx
        self.steps_idx = steps_idx
        self.step_names = step_names
        
    def experiment(self, experiment_name):
        experiment_number = list(self.titles.keys()).index(experiment_name)
        cycles_idx = self.cycles_idx[experiment_number]
        steps_idx = self.steps_idx[experiment_number]
        data = self.RawData.loc[self.RawData.index.isin(flatten(cycles_idx), level='Cycle') & self.RawData.index.isin(flatten(steps_idx), level='Step')]
        experiment_types = {'Constant Current': Experiment, 
                            'Pulsing': Pulsing, 
                            'Cycling': Experiment, 
                            'SOC Reset': Experiment}
        return experiment_types[self.titles[experiment_name]](data, cycles_idx, steps_idx, self.step_names)
   
def flatten(lst):
    if not isinstance(lst, list):
        return [lst]
    if lst == []:
        return lst
    if isinstance(lst[0], list):
        return flatten(lst[0]) + flatten(lst[1:])
    return lst[:1] + flatten(lst[1:])

def capacity_ref(df):
        return df.loc[(df['Current (A)'] == 0) & (df['Voltage (V)'] == df[df['Current (A)'] == 0]['Voltage (V)'].max()), 'Capacity (Ah)'].values[0]