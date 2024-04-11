"""A module for the Cycing class."""

from pybatdata.experiment import Experiment
from pybatdata.step import Step
import polars as pl
import numpy as np
from pybatdata.result import Result

class Cycling(Experiment):
    """A cycling experiment in a battery procedure."""

    def __init__(self, lazyframe, info):
        """Create a cycling experiment.

            Args:
                lazyframe (polars.LazyFrame): The lazyframe of data being filtered.
        """
        super().__init__(lazyframe, info)

    @property
    def SOH_capacity(self) -> np.ndarray:
        """Calculate the state of health of the battery.

        Returns:
            np.ndarray: The state of health of the battery.
        """
        
        lf_capacity_throughput = self.lazyframe.groupby('_cycle', maintain_order = True).agg(pl.col('Capacity Throughput [Ah]').first())
        lf_time = self.lazyframe.groupby('_cycle', maintain_order = True).agg(pl.col('Time (s)').first())
        
        lf_charge = self.charge().lazyframe.groupby('_cycle', maintain_order = True).agg(pl.col('Capacity [Ah]').max()-pl.col('Capacity [Ah]').min()).rename({'Capacity [Ah]': 'Charge Capacity [Ah]'})
        lf_discharge = self.discharge().lazyframe.groupby('_cycle', maintain_order = True).agg(pl.col('Capacity [Ah]').max()-pl.col('Capacity [Ah]').min()).rename({'Capacity [Ah]': 'Discharge Capacity [Ah]'})
        
        lf = lf_capacity_throughput.join(lf_time, on='_cycle', how = 'outer_coalesce').join(lf_charge, on='_cycle', how='outer_coalesce').join(lf_discharge, on='_cycle', how='outer_coalesce')
        
        lf = lf.with_columns((pl.col('Charge Capacity [Ah]')/pl.first('Charge Capacity [Ah]')*100).alias('SOH Charge (%)'))
        lf = lf.with_columns((pl.col('Discharge Capacity [Ah]')/pl.first('Discharge Capacity [Ah]')*100).alias('SOH Discharge (%)'))
        print(lf.collect())
        return Result(lf, self.info)