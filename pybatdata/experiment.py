from pybatdata.cycle import Cycle
import polars as pl
from pybatdata.base import Base

class Experiment(Base):
    def __init__(self, lazyframe, cycles_idx, steps_idx, step_names):
        super().__init__(lazyframe, cycles_idx, steps_idx, step_names)

    def cycle(self,cycle_number):
        cycles_idx = self.cycles_idx[cycle_number]
        steps_idx = self.steps_idx[cycle_number]
        conditions = [self.get_conditions('Cycle', cycles_idx),
                      self.get_conditions('Step', steps_idx)]
        lf_filtered = self.lazyframe.filter(conditions)
        return Cycle(lf_filtered, cycles_idx, steps_idx, self.step_names)