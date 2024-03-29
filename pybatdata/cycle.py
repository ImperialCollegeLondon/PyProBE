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
        steps_idx = self.steps_idx[step_number]
        conditions = [self.get_conditions('Step', steps_idx)]
        lf_filtered = self.lazyframe.filter(conditions)
        return Step(lf_filtered, steps_idx, self.step_names)

    def charge(self, charge_number: int) -> Step:
        """Return a charge step object from the cycle.
        
        Args:
            charge_number (int): The charge number to return.
            
        Returns:
            Step: A charge step object from the cycle.
        """
        def criteria(step_name):
            return step_name == 'CC Chg' or step_name == 'CCCV Chg'
        lf_filtered, charge_steps = self.get_steps(criteria, charge_number)
        return Step(lf_filtered, charge_steps, self.step_names)
 
    def discharge(self, discharge_number: int) -> Step:
        """Return a discharge step object from the cycle.
        
        Args:
            discharge_number (int): The discharge number to return.
            
        Returns:
            Step: A discharge step object from the cycle.
        """
        def criteria(step_name):
            return step_name == 'CC DChg'
        lf_filtered, discharge_steps = self.get_steps(criteria, discharge_number)
        return Step(lf_filtered, discharge_steps, self.step_names)
    
    def chargeordischarge(self, chargeordischarge_number: int) -> Step:
        """Return a charge or discharge step object from the cycle.
        
        Args:
            chargeordischarge_number (int): The charge or discharge number to return.
            
        Returns:
            Step: A charge or discharge step object from the cycle.
        """
        def criteria(step_name):
            return step_name == 'CC DChg' or step_name == 'CCCV Chg' or step_name == 'CC DChg'
        lf_filtered, chargeordischarge_steps = self.get_steps(criteria, chargeordischarge_number)
        return Step(lf_filtered, chargeordischarge_steps, self.step_names)

    def rest(self, rest_number: int) -> Step:
        """Return a rest step object from the cycle.
        
        Args:
            rest_number (int): The rest number to return.
            
        Returns:
            Step: A rest step object from the cycle.
        """
        def criteria(step_name):
            return step_name == 'Rest'
        lf_filtered, rest_steps = self.get_steps(criteria, rest_number)
        
        return Step(lf_filtered, rest_steps, self.step_names)

    def get_steps(self, criteria: Callable, step_number: int) -> tuple[pl.LazyFrame, int]:
        """Filter the lazyframe to get the steps of a certain type and number.
        
        Args:
            criteria (Callable): The criteria to filter the steps.
            step_number (int): The step number to return.
            
        Returns:
            tuple[pl.LazyFrame, int]: The filtered lazyframe and the step number.
        """
        steps = [index for index, item in enumerate(self.step_names) if criteria(item)]
        flattened_steps = self.flatten(self.steps_idx)
        steps = [step for step in steps if step in flattened_steps]
        steps_idx = steps[step_number]
        conditions = [self.get_conditions('Step', steps_idx)]
        return self.lazyframe.filter(conditions), steps[step_number]
