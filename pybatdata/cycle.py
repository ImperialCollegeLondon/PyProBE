from step import Step, Charge, Discharge, Rest
import polars as pl
class Cycle:
    def __init__(self, lazyframe, conditions,  cycles_idx, steps_idx, step_names):
        self.cycles_idx = cycles_idx
        self.steps_idx = steps_idx
        self.step_names = step_names
        self.lf = lazyframe
    
    def RawData(self):
        return self.lf.collect()
        
    def step(self, step_number):
        steps_idx = self.steps_idx[step_number-1]
        self.conditions = ([(pl.col('Step').apply(lambda group: group in flatten(steps_idx), return_dtype=pl.Boolean)).alias('Step')
                                ])
        lf_filtered = self.lf.filter(self.conditions)
        return Step(lf_filtered, steps_idx, self.step_names)

    def charge(self, charge_number):
        charge_steps = [index for index, item in enumerate(self.step_names) if (item == 'CC Chg' or item == 'CCCV Chg')]
        charge_steps = list(set(charge_steps) & set(flatten(self.steps_idx)))
        data = self.RawData.loc[self.RawData.index.isin(flatten(charge_steps[charge_number-1]), level='Step')]
        return Charge(data, charge_steps[charge_number-1], self.step_names)
 
    def discharge(self, discharge_number):
        discharge_steps = [index for index, item in enumerate(self.step_names) if item == 'CC DChg']
        discharge_steps = list(set(discharge_steps) & set(flatten(self.steps_idx)))
        data = self.RawData.loc[self.RawData.index.isin(flatten(discharge_steps[discharge_number-1]), level='Step')]
        return Discharge(data, discharge_steps[discharge_number-1], self.step_names)
    
    def rest(self, rest_number):
        rest_steps = [index for index, item in enumerate(self.step_names) if item == 'Rest']
        rest_steps = list(set(rest_steps) & set(flatten(self.steps_idx)))
        data = self.RawData.loc[self.RawData.index.isin(flatten(rest_steps[rest_number-1]), level='Step')]
        return Rest(data, rest_steps[rest_number-1], self.step_names)
    
def flatten(lst):
    if not isinstance(lst, list):
        return [lst]
    if lst == []:
        return lst
    if isinstance(lst[0], list):
        return flatten(lst[0]) + flatten(lst[1:])
    return lst[:1] + flatten(lst[1:])