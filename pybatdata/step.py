import numpy as np
import pandas as pd
import polars as pl

class Step:
    def __init__(self, lf, steps_idx, step_names):
        self.steps_idx = steps_idx
        self.step_names = step_names
        self.lf = lf
        self._raw_data = None

    @property
    def RawData(self):
        if self._raw_data is None:
            self._raw_data = self.lf.collect()
        return self._raw_data
    
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
    def __init__(self, lf, steps_idx, step_names):
        super().__init__(lf, steps_idx, step_names)
        
    @property
    def capacity(self):
        return self.RawData['Charge Capacity (Ah)'].max()
    
class Discharge(Step):
    def __init__(self, lf, steps_idx, step_names):
        super().__init__(lf, steps_idx, step_names)
        
    @property
    def capacity(self):
        return self.RawData['Discharge Capacity (Ah)'].max()
        
class Rest(Step):
    def __init__(self, lf, steps_idx, step_names):
        super().__init__(lf, steps_idx, step_names)