import numpy as np
import pandas as pd
from cycle import Cycle
import polars as pl

class Experiment:
    def __init__(self, lazyframe, cycles_idx, steps_idx, step_names):
        self.cycles_idx = cycles_idx
        self.steps_idx = steps_idx
        self.step_names = step_names
        self.lf = lazyframe
        self._raw_data = None
    
    @property
    def RawData(self):
        if self._raw_data is None:
            self._raw_data = self.lf.collect()
        return self._raw_data

        # if not self.RawData.empty:
        #     self.RawData.loc[:, 'Exp Capacity (Ah)'] = self.RawData['Capacity (Ah)'] - self.RawData['Capacity (Ah)'].iloc[0]
        # else:
        #     print("The DataFrame is empty.")
    
    def cycle(self,cycle_number):
        cycles_idx = self.cycles_idx[cycle_number-1]
        steps_idx = self.steps_idx[cycle_number-1]
        conditions = ([
                                (pl.col('Cycle').apply(lambda group: group in flatten(cycles_idx), return_dtype=pl.Boolean)).alias('Cycle'),
                                (pl.col('Step').apply(lambda group: group in flatten(steps_idx), return_dtype=pl.Boolean)).alias('Step')
                                ])
        lf_filtered = self.lf.filter(conditions)
    #     #data = self.RawData.loc[self.RawData.index.isin(flatten(cycles_idx), level='Cycle') & self.RawData.index.isin(flatten(steps_idx), level='Step')]
        return Cycle(lf_filtered, cycles_idx, steps_idx, self.step_names)
    
class Pulsing(Experiment):
    def __init__(self, data, cycles_idx, steps_idx, step_names):
        super().__init__(data, cycles_idx, steps_idx, step_names)
    
    def pulse(self, pulse_number):
        if (self.cycle(pulse_number).RawData['Current (A)']>= 0).all():
            return self.cycle(pulse_number).charge(1)
        elif (self.cycle(pulse_number).RawData['Current (A)']<= 0).all():
            return self.cycle(pulse_number).discharge(1)
        
    def R0(self, pulse_number):
        if (self.cycle(pulse_number).RawData['Current (A)']>= 0).all():
            V0 = self.cycle(pulse_number).rest(1).RawData['Voltage (V)'].iloc[0]
            V1 = self.cycle(pulse_number).charge(1).RawData['Voltage (V)'].iloc[0]
            I = self.cycle(pulse_number).charge(1).RawData['Current (A)'].iloc[0]
            return (V1-V0)/I
        elif (self.cycle(pulse_number).RawData['Current (A)']<= 0).all():
            V0 = self.cycle(pulse_number).rest(1).RawData['Voltage (V)'].iloc[0]
            V1 = self.cycle(pulse_number).discharge(1).RawData['Voltage (V)'].iloc[0]
            I = self.cycle(pulse_number).discharge(1).RawData['Current (A)'].iloc[0]
            return (V1-V0)/I
        
    def R_time(self, pulse_number):
        if (self.cycle(pulse_number).RawData['Current (A)']>= 0).all():
            V0 = self.cycle(pulse_number).rest(1).RawData['Voltage (V)'].iloc[0]
            V1 = self.cycle(pulse_number).charge(1).RawData['Voltage (V)'].loc[self.cycle(pulse_number).charge(1).RawData['Time'] >= 10].iloc[0]
            I = self.cycle(pulse_number).charge(1).RawData['Current (A)'].iloc[0]
            return (V1-V0)/I
        elif (self.cycle(pulse_number).RawData['Current (A)']<= 0).all():
            V0 = self.cycle(pulse_number).rest(1).RawData['Voltage (V)'].iloc[0]
            V1 = self.cycle(pulse_number).discharge(1).RawData['Voltage (V)'].loc[self.cycle(pulse_number).discharge(1).RawData['Time'] >= 10].iloc[0]
            I = self.cycle(pulse_number).discharge(1).RawData['Current (A)'].iloc[0]
            return (V1-V0)/I
    

    def start_capacity(self):
        return self.RawData['Exp Capacity (Ah)'].iloc[0]

    def end_capacity(self):
        return self.RawData['Exp Capacity (Ah)'].iloc[-1]  
        
    def calc_resistances(self):
        _R0 = np.zeros(len(self.cycles_idx))
        _R_10s = np.zeros(len(self.cycles_idx))
        _start_capacity = np.zeros(len(self.cycles_idx))
        for i in range(1, len(self.cycles_idx)+1):
            _R0[i-1] = self.R0(i)
            # _start_capacity[i-1] = self.pulse(i).start_capacity
            _R_10s[i-1] = self.R_time(i)
            
        return pd.DataFrame({'R0': _R0, 'Capacity Throughput (Ah)': _start_capacity, 'R 10s': _R_10s})

def flatten(lst):
    if not isinstance(lst, list):
        return [lst]
    if lst == []:
        return lst
    if isinstance(lst[0], list):
        return flatten(lst[0]) + flatten(lst[1:])
    return lst[:1] + flatten(lst[1:])