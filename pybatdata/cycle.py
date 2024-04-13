"""A module for the Cycle class."""

from pybatdata.step import Step
import polars as pl
from typing import Callable
from pybatdata.filter import Filter
from pybatdata.result import Result

class Cycle(Result):
    """A cycle in a battery procedure."""

    def __init__(self, 
                 lazyframe: pl.LazyFrame,
                 info: dict):
        """Create a cycle.

            Args:
                lazyframe (polars.LazyFrame): The lazyframe of data being filtered.
        """
        super().__init__(lazyframe, info)
        
    
    def step(self, step_number: int|list[int] = None, condition: pl.Expr = None) -> Step:
        """Return a step object from the cycle.
        
        Args:
            step_number (int): The step number to return.
            
        Returns:
            Step: A step object from the cycle.
        """
        if condition is not None:
            lazyframe= Filter.filter_numerical(self.lazyframe.filter(condition), '_step', step_number)
        else:
            lazyframe = Filter.filter_numerical(self.lazyframe, '_step', step_number)
        return Step(lazyframe, self.info)

    def charge(self, charge_number: int=None) -> Step:
        """Return a charge step object from the cycle.
        
        Args:
            charge_number (int): The charge number to return.
            
        Returns:
            Step: A charge step object from the cycle.
        """
        # lf_filtered = Filter.filter_numerical(self.lazyframe.filter(pl.col('Current [A]') > 0), '_step', charge_number)
        condition = pl.col('Current [A]') > 0
        return self.step(charge_number, condition)
 
    def discharge(self, discharge_number: int=None) -> Step:
        """Return a discharge step object from the cycle.
        
        Args:
            discharge_number (int): The discharge number to return.
            
        Returns:
            Step: A discharge step object from the cycle.
        """
        condition = pl.col('Current [A]') < 0
        return self.step(discharge_number, condition)
    
    def chargeordischarge(self, chargeordischarge_number: int=None) -> Step:
        """Return a charge or discharge step object from the cycle.
        
        Args:
            chargeordischarge_number (int): The charge or discharge number to return.
            
        Returns:
            Step: A charge or discharge step object from the cycle.
        # """
        condition = pl.col('Current [A]') != 0
        return self.step(chargeordischarge_number, condition)

    def rest(self, rest_number: int=None) -> Step:
        """Return a rest step object from the cycle.
        
        Args:
            rest_number (int): The rest number to return.
            
        Returns:
            Step: A rest step object from the cycle.
        """
        condition = pl.col('Current [A]') == 0
        return self.step(rest_number, condition)
