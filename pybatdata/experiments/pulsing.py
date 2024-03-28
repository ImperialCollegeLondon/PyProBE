from pybatdata.experiment import Experiment
import polars as pl
import numpy as np

class Pulsing(Experiment):
    def __init__(self, lazyframe, cycles_idx, steps_idx, step_names):
        super().__init__(lazyframe, cycles_idx, steps_idx, step_names)
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
            Vt[i] = first_row['Voltage (V)'].to_numpy()[0]
        return (Vt-self.V0)/self.I1    
        
    def pulse(self, pulse_number):
        if self.pulses[pulse_number] is None:
            self.pulses[pulse_number] = self.cycle(pulse_number).chargeordischarge(0)
        return self.pulses[pulse_number]
    
    def rest(self, rest_number):
        if self.rests[rest_number] is None:
            self.rests[rest_number] = self.cycle(rest_number).rest(0)
        return self.rests[rest_number]