import numpy as np
import pandas as pd
from cycle import Cycle

class Experiment:
    def __init__(self, data, cycles_idx, steps_idx, step_names):
        self.cycles_idx = cycles_idx
        self.steps_idx = steps_idx
        self.step_names = step_names
        self.RawData = data
        
        if not self.RawData.empty:
            self.RawData['Exp Capacity (Ah)'] = self.RawData['Capacity (Ah)'] - self.RawData['Capacity (Ah)'].iloc[0]
        else:
            print("The DataFrame is empty.")
    
    def cycle(self,cycle_number):
        cycles_idx = self.cycles_idx[cycle_number-1]
        steps_idx = self.steps_idx[cycle_number-1]
        data = self.RawData.loc[self.RawData.index.isin(flatten(cycles_idx), level='Cycle') & self.RawData.index.isin(flatten(steps_idx), level='Step')]
        return Cycle(data, cycles_idx, steps_idx, self.step_names)
    
class Pulsing(Experiment):
    def __init__(self, data, cycles_idx, steps_idx, step_names):
        super().__init__(data, cycles_idx, steps_idx, step_names)
    
    def pulse(self, pulse_number):
        if (self.cycle(pulse_number).RawData['Current (A)']>= 0).all():
            return self.cycle(pulse_number).charge(1)
        elif (self.cycle(pulse_number).RawData['Current (A)']<= 0).all():
            return self.cycle(pulse_number).discharge(1)
          
    def calc_resistances(self):
        _R0 = np.zeros(len(self.cycles_idx))
        _R_10s = np.zeros(len(self.cycles_idx))
        _start_capacity = np.zeros(len(self.cycles_idx))
        for i in range(1, len(self.cycles_idx)+1):
            _R0[i-1] = self.pulse(i).R0
            _start_capacity[i-1] = self.pulse(i).start_capacity
            _R_10s[i-1] = self.pulse(i).R_time
            
        return pd.DataFrame({'R0': _R0, 'Capacity Throughput (Ah)': _start_capacity, 'R 10s': _R_10s})

def flatten(lst):
    if not isinstance(lst, list):
        return [lst]
    if lst == []:
        return lst
    if isinstance(lst[0], list):
        return flatten(lst[0]) + flatten(lst[1:])
    return lst[:1] + flatten(lst[1:])