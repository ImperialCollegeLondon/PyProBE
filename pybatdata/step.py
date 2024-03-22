import numpy as np
import pandas as pd
from base import Base
import polars as pl

class Step(Base):
    def __init__(self, lf, steps_idx, step_names):
        super().__init__(lf, None, steps_idx, step_names)
    
    def IC(self, deltaV):
        V = self.RawData['Voltage (V)']
        
        n = len(V)
        V_range = V.max() - V.min()
        v = np.linspace(V.min(), V.max(), int(V_range/deltaV))
        deltaV = v[1]-v[0]
        
        N, _ = np.histogram(V, bins=v)
        IC = N/n * 1/deltaV
        v_midpoints = v[:-1] + np.diff(v)/2
        
        IC = self.smooth_IC(IC, [0.0668, 0.2417, 0.3830, 0.2417, 0.0668])
        return v_midpoints, IC

    @staticmethod
    def smooth_IC(IC, alpha):
        A = np.zeros((len(IC), len(IC)))
        w = np.floor(len(alpha)/2)
        for n in range(len(alpha)):
            k = n - w
            vector = np.ones(int(len(IC) - abs(k)))
            diag = np.diag(vector, int(k))
            A += alpha[n] * diag
        return A @ IC
            
    
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
    
class ChargeOrDischarge(Step):
    def __init__(self, lf, steps_idx, step_names):
        super().__init__(lf, steps_idx, step_names)
        
    @property
    def capacity(self):
        return self.RawData['Discharge Capacity (Ah)'].max()
        
class Rest(Step):
    def __init__(self, lf, steps_idx, step_names):
        super().__init__(lf, steps_idx, step_names)