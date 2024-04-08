"""A module for the Cycle class."""

from pybatdata.step import Step
import polars as pl
from pybatdata.base import Base
from typing import Callable

class Cycle(Base):
    """A cycle in a battery procedure."""

    def __init__(self, 
                 lazyframe: pl.LazyFrame, 
                 cycles_idx: int, 
                 steps_idx: list, 
                 step_names: list):
        """Create a cycle.

            Args:
                lazyframe (polars.LazyFrame): The lazyframe of data being filtered.
                cycles_idx (int): The index of the cycle in the procedure.
                steps_idx (list): The indices of the steps of the experiment.
                step_names (list): The names of all of the steps in the procedure.
        """
        super().__init__(lazyframe, cycles_idx, steps_idx, step_names)
        
    
    def step(self, step_number: int) -> Step:
        """Return a step object from the cycle.
        
        Args:
            step_number (int): The step number to return.
            
        Returns:
            Step: A step object from the cycle.
        """
        lf_filtered = self.filter_numerical(self.lazyframe, 'Step', step_number)
        return Step(lf_filtered, 0, self.step_names)

    def charge(self, charge_number: int) -> Step:
        """Return a charge step object from the cycle.
        
        Args:
            charge_number (int): The charge number to return.
            
        Returns:
            Step: A charge step object from the cycle.
        """
        lf_filtered = self.filter_numerical(self.lazyframe.filter(pl.col('Current (A)') > 0), 'Step', charge_number)
        return Step(lf_filtered, 0, self.step_names)
 
    def discharge(self, discharge_number: int) -> Step:
        """Return a discharge step object from the cycle.
        
        Args:
            discharge_number (int): The discharge number to return.
            
        Returns:
            Step: A discharge step object from the cycle.
        """
        lf_filtered = self.filter_numerical(self.lazyframe.filter(pl.col('Current (A)') < 0), 'Step', discharge_number)
        return Step(lf_filtered, 4, self.step_names)
    
    def chargeordischarge(self, chargeordischarge_number: int) -> Step:
        """Return a charge or discharge step object from the cycle.
        
        Args:
            chargeordischarge_number (int): The charge or discharge number to return.
            
        Returns:
            Step: A charge or discharge step object from the cycle.
        # """
        lf_filtered = self.filter_numerical(self.lazyframe.filter(pl.col('Current (A)') != 0), 'Step', chargeordischarge_number)
        return Step(lf_filtered, 0, self.step_names)

    def rest(self, rest_number: int) -> Step:
        """Return a rest step object from the cycle.
        
        Args:
            rest_number (int): The rest number to return.
            
        Returns:
            Step: A rest step object from the cycle.
        """
        lf_filtered = self.filter_numerical(self.lazyframe.filter(pl.col('Current (A)') == 0), 'Step', rest_number)
        return Step(lf_filtered, 0, self.step_names)
