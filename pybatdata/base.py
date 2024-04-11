"""Module for the Base class."""

import polars as pl
import matplotlib.pyplot as plt
from pybatdata.result import Result
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
        self._set_zero_capacity()
        self._set_zero_time()
        self._create_capacity_throughput()
        self.lazyframe = self._get_events(self.lazyframe)
        super().__init__(self.lazyframe, self.info)
    
    def _set_zero_capacity(self) -> None:
        """Recalculate the capacity column to start from zero at beginning of current selection."""
        self.lazyframe = self.lazyframe.with_columns([
            (pl.col("Capacity [Ah]") - pl.col("Capacity [Ah]").first()).alias("Capacity [Ah]")
        ]) 

    def _set_zero_time(self) -> None:
        """Recalculate the time column to start from zero at beginning of current selection."""
        self.lazyframe = self.lazyframe.with_columns([
            (pl.col("Time (s)") - pl.col("Time (s)").first()).alias("Time (s)")
        ])

    
    def _create_capacity_throughput(self)->None:
        """Recalculate the capacity column to show the total capacity passed at each point."""
        self.lazyframe = self.lazyframe.with_columns([
            (pl.col("Capacity [Ah]").diff().abs().cum_sum()).alias("Capacity Throughput [Ah]")
        ])

    @staticmethod
    def _get_events(lazyframe: pl.LazyFrame):
        lazyframe = lazyframe.with_columns(((pl.col('Cycle') - pl.col('Cycle').shift() != 0)
                                    .fill_null(strategy='zero').cum_sum()
                                    .alias('_cycle').cast(pl.Int32)))
        lazyframe = lazyframe.with_columns((((pl.col('Cycle') - pl.col('Cycle').shift() != 0) | (pl.col('Step') - pl.col('Step').shift() != 0))
                                    .fill_null(strategy='zero').cum_sum()
                                    .alias('_step').cast(pl.Int32)))
        lazyframe = lazyframe.with_columns([
            (pl.col('_cycle') - pl.col('_cycle').max() - 1).alias('_cycle_reversed'),
            (pl.col('_step') - pl.col('_step').max() - 1).alias('_step_reversed')
        ])
        return lazyframe
    
    def filter_numerical(self, lazyframe: pl.LazyFrame, column: str, condition_number: int|list[int]) -> pl.Expr:
        if isinstance(condition_number, int):
            condition_number = [condition_number]
        elif isinstance(condition_number, list):
            condition_number = list(range(condition_number[0], condition_number[1] + 1))
        lazyframe = self._get_events(lazyframe)
        if condition_number is not None:
            return lazyframe.filter(pl.col(column).is_in(condition_number) | pl.col(column + '_reversed').is_in(condition_number))
        else: 
            return lazyframe
    
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