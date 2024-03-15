import numpy as np
import pandas as pd
from experiment import Experiment, Pulsing
import polars as pl

class Procedure:
    def __init__(self, lazyframe, titles, cycles_idx, steps_idx, step_names):
        self.lf = lazyframe
        self.titles = titles
        self.cycles_idx = cycles_idx
        self.steps_idx = steps_idx
        self.step_names = step_names
        
    def experiment(self, experiment_name):
        experiment_number = list(self.titles.keys()).index(experiment_name)
        cycles_idx = self.cycles_idx[experiment_number]
        steps_idx = self.steps_idx[experiment_number]
        conditions = [
                        (pl.col('Cycle').apply(lambda group: group in flatten(cycles_idx), return_dtype=pl.Boolean)).alias('Cycle'),
                        (pl.col('Step').apply(lambda group: group in flatten(steps_idx), return_dtype=pl.Boolean)).alias('Step')
                    ]
        lf_filtered = self.lf.filter(conditions)
        experiment_types = {'Constant Current': Experiment, 
                            'Pulsing': Experiment, 
                            'Cycling': Experiment, 
                            'SOC Reset': Experiment}
        return experiment_types[self.titles[experiment_name]](lf_filtered, cycles_idx, steps_idx, self.step_names)
   
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