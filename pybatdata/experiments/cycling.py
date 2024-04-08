"""A module for the Cycing class."""

from pybatdata.experiment import Experiment
from pybatdata.step import Step
import polars as pl
import numpy as np

class Cycling(Experiment):
    """A cycling experiment in a battery procedure."""

    def __init__(self, lazyframe):
        """Create a cycling experiment.

            Args:
                lazyframe (polars.LazyFrame): The lazyframe of data being filtered.
        """
        super().__init__(lazyframe)

    def SOH_capacity(self, step) -> np.ndarray:
        """Calculate the state of health of the battery.

        Returns:
            np.ndarray: The state of health of the battery.
        """
        print(self.cycle(0).discharge(1).RawData)
        reference = eval(f"self.cycle(0).{step}.capacity")
        SOH = np.zeros(self.n_cycles)
        for i in range(self.n_cycles):
            SOH[i] = eval(f"self.cycle({i}).{step}.capacity") / reference
        return SOH