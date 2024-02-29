import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
class Experiment:
    def __init__(self, data, cycles_idx, steps_idx, step_names):
        self.cycles_idx = cycles_idx
        self.steps_idx = steps_idx
        self.step_names = step_names
        
        self.RawData = data[(data['Cycle'].isin(flatten(self.cycles_idx))) & (data['Step'].isin(flatten(self.steps_idx)))]
        if not self.RawData.empty:
            self.RawData['Exp Capacity (Ah)'] = self.RawData['Capacity (Ah)'] - self.RawData['Capacity (Ah)'].iloc[0]
        else:
            print("The DataFrame is empty.")
    
    def cycle(self,cycle_number):
        return Cycle(self.RawData, self.cycles_idx[cycle_number-1], self.steps_idx[cycle_number-1], self.step_names)
    
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
    
class Cycle:
    def __init__(self, data, cycles_idx, steps_idx, step_names):
        self.cycles_idx = cycles_idx
        self.steps_idx = steps_idx
        self.step_names = step_names
        self.RawData = data[(data['Cycle'].isin(flatten(cycles_idx))) & (data['Step'].isin(flatten(self.steps_idx)))]

    def step(self, step_number):
        return Step(self.RawData, self.cycles_idx, self.steps_idx[step_number-1], self.step_names)

    def charge(self, charge_number):
        charge_steps = [index for index, item in enumerate(self.step_names) if (item == 'CC Chg' or item == 'CCCV Chg')]
        charge_steps = list(set(charge_steps) & set(flatten(self.steps_idx)))
        return Charge(self.RawData, self.cycles_idx, charge_steps[charge_number-1], self.step_names)
 
    def discharge(self, discharge_number):
        discharge_steps = [index for index, item in enumerate(self.step_names) if item == 'CC DChg']
        discharge_steps = list(set(discharge_steps) & set(flatten(self.steps_idx)))
        return Discharge(self.RawData, self.cycles_idx, discharge_steps[discharge_number-1], self.step_names)
    
    def rest(self, rest_number):
        rest_steps = [index for index, item in enumerate(self.step_names) if item == 'Rest']
        rest_steps = list(set(rest_steps) & set(flatten(self.steps_idx)))
        return Rest(self.RawData, self.cycles_idx, rest_steps[rest_number-1], self.step_names)
    
class Step:
    def __init__(self, data, cycles_idx, steps_idx, step_names):
        self.cycles_idx = cycles_idx
        self.steps_idx = steps_idx
        self.step_names = step_names
        self.RawData = data[(data['Cycle'].isin(flatten(self.cycles_idx))) & (data['Step'].isin(flatten(self.steps_idx)))]

    @property
    def R0(self):
        V1 = self.RawData['Voltage (V)'].iloc[0]
        V2 = self.RawData['Voltage (V)'].loc[self.RawData['Voltage (V)'] != self.RawData['Voltage (V)'].iloc[0]].iloc[0]
        I = self.RawData['Current (A)'].iloc[0]
        return (V2-V1)/I
    
    @property
    def R_time(self):
        V1 = self.RawData['Voltage (V)'].iloc[0]
        V2 = self.RawData['Voltage (V)'].loc[self.RawData['Time'] >= 10].iloc[0]
        I = self.RawData['Current (A)'].iloc[0]
        return (V2-V1)/I
    
    @property
    def start_capacity(self):
        return self.RawData['Exp Capacity (Ah)'].iloc[0]
    
    @property
    def end_capacity(self):
        return self.RawData['Exp Capacity (Ah)'].iloc[-1]
    
class Charge(Step):
    def __init__(self, data, cycles_idx, steps_idx, step_names):
        super().__init__(data, cycles_idx, steps_idx, step_names)
        
    @property
    def capacity(self):
        return self.RawData['Charge Capacity (Ah)'].max()
    
class Discharge(Step):
    def __init__(self, data, cycles_idx, steps_idx, step_names):
        super().__init__(data, cycles_idx, steps_idx, step_names)
        
    @property
    def capacity(self):
        return self.RawData['Discharge Capacity (Ah)'].max()
        
class Rest(Step):
    def __init__(self, data, cycles_idx, steps_idx, step_names):
        super().__init__(data, cycles_idx, steps_idx, step_names)
    
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