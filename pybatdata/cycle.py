from pybatdata.step import *
import polars as pl
from pybatdata.base import Base

class Cycle(Base):
    def __init__(self, lf, cycles_idx, steps_idx, step_names):
        super().__init__(lf, cycles_idx, steps_idx, step_names)
    
    def step(self, step_number):
        steps_idx = self.steps_idx[step_number]
        conditions = [self.get_conditions('Step', steps_idx)]
        lf_filtered = self.lf.filter(conditions)
        return Step(lf_filtered, steps_idx, self.step_names)

    def charge(self, charge_number):
        def criteria(step_name):
            return step_name == 'CC Chg' or step_name == 'CCCV Chg'
        lf_filtered, charge_steps = self.get_steps(criteria, charge_number)
        return Charge(lf_filtered, charge_steps, self.step_names)
 
    def discharge(self, discharge_number):
        def criteria(step_name):
            return step_name == 'CC DChg'
        lf_filtered, discharge_steps = self.get_steps(criteria, discharge_number)
        return Discharge(lf_filtered, discharge_steps, self.step_names)
    
    def chargeordischarge(self, chargeordischarge_number):
        def criteria(step_name):
            return step_name == 'CC DChg' or step_name == 'CCCV Chg' or step_name == 'CC DChg'
        lf_filtered, chargeordischarge_steps = self.get_steps(criteria, chargeordischarge_number)
        return ChargeOrDischarge(lf_filtered, chargeordischarge_steps, self.step_names)

    def rest(self, rest_number):
        def criteria(step_name):
            return step_name == 'Rest'
        lf_filtered, rest_steps = self.get_steps(criteria, rest_number)
        
        return Rest(lf_filtered, rest_steps, self.step_names)

    def get_steps(self, criteria, step_number):
        steps = [index for index, item in enumerate(self.step_names) if criteria(item)]
        flattened_steps = self.flatten(self.steps_idx)
        steps = [step for step in steps if step in flattened_steps]
        steps_idx = steps[step_number]
        conditions = [self.get_conditions('Step', steps_idx)]
        return self.lf.filter(conditions), steps[step_number]
