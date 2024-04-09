"""A module for the Cycing class."""

from pybatdata.experiment import Experiment
from pybatdata.step import Step
import polars as pl
import numpy as np
from pybatdata.viewer import Viewer

class Cycling(Experiment):
    """A cycling experiment in a battery procedure."""

    def __init__(self, lazyframe, info):
        """Create a cycling experiment.

            Args:
                lazyframe (polars.LazyFrame): The lazyframe of data being filtered.
        """
        super().__init__(lazyframe, info)

    def SOH_capacity(self) -> np.ndarray:
        """Calculate the state of health of the battery.

        Returns:
            np.ndarray: The state of health of the battery.
        """
        print(self.charge().lazyframe.collect())
        lf_charge = self.charge().lazyframe.groupby('_cycle', maintain_order = True).agg([pl.col('Capacity (Ah)').max()-pl.col('Capacity (Ah)').min()
                                                                                          ,pl.col('Capacity Throughput (Ah)').last()])
        lf_discharge = self.discharge().lazyframe.groupby('_cycle', maintain_order = True).agg([pl.col('Capacity (Ah)').max()-pl.col('Capacity (Ah)').min()
                                                                                                ,pl.col('Capacity Throughput (Ah)').last()])
        print(lf_charge.collect())
        print(lf_discharge.collect())
        return Viewer(lf_discharge.collect(), self.info)