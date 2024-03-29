"""A module for the Experiment class."""

from pybatdata.cycle import Cycle
import polars as pl
from pybatdata.base import Base

class Experiment(Base):
    """ An experiment in a battery procedure."""
    def __init__(self, 
                 lazyframe: pl.LazyFrame, 
                 cycles_idx: list, 
                 steps_idx: list, 
                 step_names: list):
        """Create an experiment.

            Args:
                lazyframe (polars.LazyFrame): The lazyframe of data being filtered.
                cycles_idx (list): The indices of the cycles of the experiment.
                steps_idx (list): The indices of the steps of the experiment.
                step_names (list): The names of all of the steps in the procedure.
        """
        super().__init__(lazyframe, cycles_idx, steps_idx, step_names)

    def cycle(self,cycle_number: int) -> Cycle:
        """Return a cycle object from the experiment.
        
        Args:
            cycle_number (int): The cycle number to return.
            
        Returns:
            Cycle: A cycle object from the experiment.
        """
        cycles_idx = self.cycles_idx[cycle_number]
        steps_idx = self.steps_idx[cycle_number]
        conditions = [self.get_conditions('Cycle', cycles_idx),
                      self.get_conditions('Step', steps_idx)]
        lf_filtered = self.lazyframe.filter(conditions)
        return Cycle(lf_filtered, cycles_idx, steps_idx, self.step_names)