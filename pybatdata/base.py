"""Module for the Base class."""

import polars as pl
import matplotlib.pyplot as plt
from pybatdata.result import Result
from pybatdata.filter import Filter
# from pybatdata.result import Plot

class Base(Result):
    """Base class for all filtering classes in PyBatData.

    Attributes:
            lazyframe (polars.LazyFrame): The lazyframe of data being filtered.
            _raw_data (polars.DataFrame): The collected dataframe of the current filter.
    """
    def __init__(self, lazyframe: pl.LazyFrame, info):
        """ Create a filtering class.
        
        Args:
            lazyframe (polars.LazyFrame): The lazyframe of data being filtered.
        """
        self.lazyframe = lazyframe
        self.info = info
        # self._set_zero_capacity()
        # self._set_zero_time()
        self._create_capacity_throughput()
        self.lazyframe = Filter._get_events(self.lazyframe)
        super().__init__(self.lazyframe, self.info)
    
    def _set_zero_capacity(self) -> None:
        """Recalculate the capacity column to start from zero at beginning of current selection."""
        self.lazyframe = self.lazyframe.with_columns([
            (pl.col("Capacity [Ah]") - pl.col("Capacity [Ah]").first()).alias("Capacity [Ah]")
        ]) 

    def _set_zero_time(self) -> None:
        """Recalculate the time column to start from zero at beginning of current selection."""
        self.lazyframe = self.lazyframe.with_columns([
            (pl.col("Time [s]") - pl.col("Time [s]").first()).alias("Time [s]")
        ])

    
    def _create_capacity_throughput(self)->None:
        """Recalculate the capacity column to show the total capacity passed at each point."""
        self.lazyframe = self.lazyframe.with_columns([
            (pl.col("Capacity [Ah]").diff().abs().cum_sum()).alias("Capacity Throughput [Ah]")
        ])

    
    
    # def plot(self, x, y, **kwargs):
    #     plt.plot(self.data[x], self.data[y], **kwargs)
    #     plt.xlabel(x)
    #     plt.ylabel(y)
    #     plt.legend()
        
    # def plot_any(df, x, y):
    #     plt.plot(df[x], df[y])
    #     plt.xlabel(x)
    #     plt.ylabel(y)
    #     plt.legend()

# class DataHolder:
#     """A class to hold data to return to a user."""
#     def __init__(self, data):
#         self.data = data
#         self._plot = Plot(self.data)
    
#     def plot(self, x, y):
#         self._plot(x, y)