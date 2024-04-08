"""A module for the Cycing class."""

from pybatdata.experiment import Experiment
from pybatdata.step import Step
import polars as pl
import numpy as np

class Cycling(Experiment):
    """A cycling experiment in a battery procedure."""

    def __init__(self, lazyframe, cycles_idx, steps_idx, step_names):
        """Create a cycling experiment.

            Args:
                lazyframe (polars.LazyFrame): The lazyframe of data being filtered.
                cycles_idx (list): The indices of the cycles of the cycling experiment.
                steps_idx (list): The indices of the steps in the cycling experiment.
                step_names (list): The names of all of the steps in the procedure.
        """
        super().__init__(lazyframe, cycles_idx, steps_idx, step_names)
        self.n_cycles = len(cycles_idx)

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