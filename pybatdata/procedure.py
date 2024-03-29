"""A module for the Procedure class."""
from pybatdata.experiment import Experiment
from pybatdata.experiments.pulsing import Pulsing
import polars as pl
from pybatdata.base import Base

class Procedure(Base):
    """A class for a procedure in a battery experiment."""
    def __init__(self, 
                 lazyframe: pl.LazyFrame, 
                 titles: dict, 
                 cycles_idx: list, 
                 steps_idx: list, 
                 step_names: list):
        """ Create a procedure class.
        
        Args:
            lazyframe (polars.LazyFrame): The lazyframe of data being filtered.
            titles (dict): The titles of the experiments inside a procedure. Fomat {title: experiment type}.
            cycles_idx (list): The indices of the cycles in the current selection.
            steps_idx (list): The indices of the steps in the current selection.
            step_names (list): The names of all of the steps in the procedure.
        """
        super().__init__(lazyframe, cycles_idx, steps_idx, step_names)
        self.titles = titles
        
    def experiment(self, experiment_name: str)->Experiment:
        """Return an experiment object from the procedure.
        
        Args:
            experiment_name (str): The name of the experiment.
            
        Returns:
            Experiment: An experiment object from the procedure.
        """
        experiment_number = list(self.titles.keys()).index(experiment_name)
        cycles_idx = self.cycles_idx[experiment_number]
        steps_idx = self.steps_idx[experiment_number]
        conditions = [self.get_conditions('Cycle', cycles_idx),
                      self.get_conditions('Step', steps_idx)]
        lf_filtered = self.lazyframe.filter(conditions)
        experiment_types = {'Constant Current': Experiment, 
                            'Pulsing': Pulsing, 
                            'Cycling': Experiment, 
                            'SOC Reset': Experiment}
        return experiment_types[self.titles[experiment_name]](lf_filtered, cycles_idx, steps_idx, self.step_names)