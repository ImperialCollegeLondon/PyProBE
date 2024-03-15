from step import Step, Charge, Discharge, Rest
import polars as pl
class Cycle:
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
        
    def step(self, step_number):
        steps_idx = self.steps_idx[step_number-1]
        conditions = ([(pl.col('Step').apply(lambda group: group in flatten(steps_idx), return_dtype=pl.Boolean)).alias('Step')
                                ])
        lf_filtered = self.lf.filter(conditions)
        return Step(lf_filtered, steps_idx, self.step_names)

    def charge(self, charge_number):
        charge_steps = [index for index, item in enumerate(self.step_names) if (item == 'CC Chg' or item == 'CCCV Chg')]
        charge_steps = list(set(charge_steps) & set(flatten(self.steps_idx)))
        conditions = ([(pl.col('Step').apply(lambda group: group in flatten(charge_steps[charge_number-1]), return_dtype=pl.Boolean)).alias('Step')])
        lf_filtered = self.lf.filter(conditions)
        return Charge(lf_filtered, charge_steps[charge_number-1], self.step_names)
 
    def discharge(self, discharge_number):
        discharge_steps = [index for index, item in enumerate(self.step_names) if item == 'CC DChg']
        discharge_steps = list(set(discharge_steps) & set(flatten(self.steps_idx)))
        conditions = ([(pl.col('Step').apply(lambda group: group in flatten(discharge_steps[discharge_number-1]), return_dtype=pl.Boolean)).alias('Step')])
        lf_filtered = self.lf.filter(conditions)
        return Discharge(lf_filtered, discharge_steps[discharge_number-1], self.step_names)
    
    def rest(self, rest_number):
        rest_steps = [index for index, item in enumerate(self.step_names) if item == 'Rest']
        rest_steps = list(set(rest_steps) & set(flatten(self.steps_idx)))
        conditions = ([(pl.col('Step').apply(lambda group: group in flatten(rest_steps[rest_number-1]), return_dtype=pl.Boolean)).alias('Step')])
        lf_filtered = self.lf.filter(conditions)
        return Rest(lf_filtered, rest_steps[rest_number-1], self.step_names)
    
def flatten(lst):
    if not isinstance(lst, list):
        return [lst]
    if lst == []:
        return lst
    if isinstance(lst[0], list):
        return flatten(lst[0]) + flatten(lst[1:])
    return lst[:1] + flatten(lst[1:])