import numpy as np
import pandas as pd
from cycle import Cycle
import polars as pl
from base import Base

class Experiment(Base):
    def __init__(self, lf, cycles_idx, steps_idx, step_names):
        super().__init__(lf, cycles_idx, steps_idx, step_names)

    def cycle(self,cycle_number):
        cycles_idx = self.cycles_idx[cycle_number]
        steps_idx = self.steps_idx[cycle_number]
        conditions = [self.get_conditions('Cycle', cycles_idx),
                      self.get_conditions('Step', steps_idx)]
        lf_filtered = self.lf.filter(conditions)
        return Cycle(lf_filtered, cycles_idx, steps_idx, self.step_names)
    
class Pulsing(Experiment):
    def __init__(self, lf, cycles_idx, steps_idx, step_names):
        super().__init__(lf, cycles_idx, steps_idx, step_names)
        self.charge_status = (self.RawData['Current (A)']>= 0).all()
        self.rests = [None]*len(cycles_idx)
        self.pulses = [None]*len(cycles_idx)

    @property
    def pulse_starts(self):
        df = self.RawData.with_columns(pl.col('Current (A)').shift().alias('Prev Current'))
        df = df.with_columns(pl.col('Voltage (V)').shift().alias('Prev Voltage'))
        return  df.filter((df['Current (A)'].shift() == 0) & (df['Current (A)'] != 0))

    @property
    def V0(self):
        return self.pulse_starts['Prev Voltage'].to_numpy()
    
    @property
    def V1(self):
        return self.pulse_starts['Voltage (V)'].to_numpy()
    
    @property
    def I1(self):
        return self.pulse_starts['Current (A)'].to_numpy()

    @property
    def R0(self):
        return (self.V1-self.V0)/self.I1
    
    def Rt(self, t):
        t_point = self.pulse_starts['Time']+t
        Vt = np.zeros(len(t_point))
        for i in range(len(Vt)):
            condition = self.RawData['Time'] >= t_point[i]
            first_row = self.RawData.filter(condition).sort('Time').head(1)
            Vt[i] = first_row['Voltage (V)'].to_numpy()
        return (Vt-self.V0)/self.I1    
        
    def pulse(self, pulse_number):
        if self.pulses[pulse_number] is None:
            self.pulses[pulse_number] = self.cycle(pulse_number).chargeordischarge(0)
        return self.pulses[pulse_number]
    
    def rest(self, rest_number):
        if self.rests[rest_number] is None:
            self.rests[rest_number] = self.cycle(rest_number).rest(0)
        return self.rests[rest_number]
    

    def start_capacity(self):
        return self.RawData['Exp Capacity (Ah)'].iloc[0]

    def end_capacity(self):
        return self.RawData['Exp Capacity (Ah)'].iloc[-1]  