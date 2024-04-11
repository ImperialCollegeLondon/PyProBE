"""A module for the Experiment class."""

from pybatdata.cycle import Cycle
import polars as pl
from pybatdata.base import Base

class Experiment(Cycle):
    """ An experiment in a battery procedure."""
    def __init__(self, 
                 lazyframe: pl.LazyFrame,
                 info: dict):
        """Create an experiment.

            Args:
                lazyframe (polars.LazyFrame): The lazyframe of data being filtered.
        """
        super().__init__(lazyframe, info)

    def cycle(self,cycle_number: int) -> Cycle:
        """Return a cycle object from the experiment.
        
        Args:
            cycle_number (int): The cycle number to return.
            
        Returns:
            Cycle: A cycle object from the experiment.
        """
        lf_filtered = self.filter_numerical(self.lazyframe, '_cycle', cycle_number)
        return Cycle(lf_filtered, self.info)