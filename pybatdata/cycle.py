from step import *
import polars as pl
from base import Base

class Cycle(Base):
    def __init__(self, lf, cycles_idx, steps_idx, step_names):
        super().__init__(lf, cycles_idx, steps_idx, step_names)
    
    def step(self, step_number):
        steps_idx = self.steps_idx[step_number-1]
        conditions = [self.get_conditions('Step', steps_idx)]
        lf_filtered = self.lf.filter(conditions)
        return Step(lf_filtered, steps_idx, self.step_names)

    def charge(self, charge_number):
        charge_steps = [index for index, item in enumerate(self.step_names) if (item == 'CC Chg' or item == 'CCCV Chg')]
        charge_steps = list(set(charge_steps) & set(self.flatten(self.steps_idx)))
        steps_idx = charge_steps[charge_number-1]
        conditions = [self.get_conditions('Step', steps_idx)]
        lf_filtered = self.lf.filter(conditions)
        return Charge(lf_filtered, charge_steps[charge_number-1], self.step_names)
 
    def discharge(self, discharge_number):
        discharge_steps = [index for index, item in enumerate(self.step_names) if item == 'CC DChg']
        discharge_steps = list(set(discharge_steps) & set(self.flatten(self.steps_idx)))
        steps_idx = discharge_steps[discharge_number-1]
        conditions = [self.get_conditions('Step', steps_idx)]
        lf_filtered = self.lf.filter(conditions)
        return Discharge(lf_filtered, discharge_steps[discharge_number-1], self.step_names)
    
    def chargeordischarge(self, chargeordischarge_number):
        chargeordischarge_steps = [index for index, item in enumerate(self.step_names) if (item == 'CC Chg' or item == 'CCCV Chg' or item == 'CC DChg')]
        chargeordischarge_steps = list(set(chargeordischarge_steps) & set(self.self.flatten(self.steps_idx)))
        steps_idx = chargeordischarge_steps[chargeordischarge_number-1]
        conditions = [self.get_conditions('Step', steps_idx)]
        lf_filtered = self.lf.filter(conditions)
        return ChargeOrDischarge(lf_filtered, chargeordischarge_steps[chargeordischarge_number-1], self.step_names)

    def rest(self, rest_number):
        rest_steps = [index for index, item in enumerate(self.step_names) if item == 'Rest']
        rest_steps = list(set(rest_steps) & set(self.flatten(self.steps_idx)))
        steps_idx = rest_steps[rest_number-1]
        conditions = [self.get_conditions('Step', steps_idx)]
        lf_filtered = self.lf.filter(conditions)
        return Rest(lf_filtered, rest_steps[rest_number-1], self.step_names)
