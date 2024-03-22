import numpy as np
import pandas as pd
from experiment import Experiment, Pulsing
import polars as pl
from base import Base

class Procedure(Base):
    def __init__(self, lf, titles, cycles_idx, steps_idx, step_names):
        super().__init__(lf, cycles_idx, steps_idx, step_names)
        self.titles = titles
        
    def experiment(self, experiment_name):
        experiment_number = list(self.titles.keys()).index(experiment_name)
        cycles_idx = self.cycles_idx[experiment_number]
        steps_idx = self.steps_idx[experiment_number]
        conditions = [self.get_conditions('Cycle', cycles_idx),
                      self.get_conditions('Step', steps_idx)]
        lf_filtered = self.lf.filter(conditions)
        experiment_types = {'Constant Current': Experiment, 
                            'Pulsing': Pulsing, 
                            'Cycling': Experiment, 
                            'SOC Reset': Experiment}
        return experiment_types[self.titles[experiment_name]](lf_filtered, cycles_idx, steps_idx, self.step_names)