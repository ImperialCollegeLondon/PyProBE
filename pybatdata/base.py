"""Module for the Base class."""

import polars as pl
import matplotlib.pyplot as plt

class Base:
    """Base class for all filtering classes in PyBatData.

    Attributes:
            lazyframe (polars.LazyFrame): The lazyframe of data being filtered.
            _raw_data (polars.DataFrame): The collected dataframe of the current filter.
    """
    def __init__(self, lazyframe: pl.LazyFrame):
        """ Create a filtering class.
        
        Args:
            lazyframe (polars.LazyFrame): The lazyframe of data being filtered.
        """
        self.lazyframe = lazyframe
        self._set_zero_capacity()
        self._set_zero_time()
        self._create_mA_units()
        self._create_capacity_throughput()
        self._raw_data = None

    def _set_zero_capacity(self) -> None:
        """Recalculate the capacity column to start from zero at beginning of current selection."""
        self.lazyframe = self.lazyframe.with_columns([
            (pl.col("Capacity (Ah)") - pl.col("Capacity (Ah)").first()).alias("Capacity (Ah)")
        ]) 

    def _set_zero_time(self) -> None:
        """Recalculate the time column to start from zero at beginning of current selection."""
        self.lazyframe = self.lazyframe.with_columns([
            (pl.col("Time (s)") - pl.col("Time (s)").first()).alias("Time (s)")
        ])

    def _create_mA_units(self) -> None:
        """Convert the current and capacity columns to mA units."""
        self.lazyframe = self.lazyframe.with_columns([
            (pl.col("Current (A)") * 1000).alias("Current (mA)")
        ])
        self.lazyframe = self.lazyframe.with_columns([
            (pl.col("Capacity (Ah)") * 1000).alias("Capacity (mAh)")
        ])
    
    def _create_capacity_throughput(self)->None:
        """Recalculate the capacity column to show the total capacity passed at each point."""
        self.lazyframe = self.lazyframe.with_columns([
            (pl.col("Capacity (Ah)").diff().abs().cum_sum()).alias("Capacity Throughput (Ah)")
        ])
    
    @property
    def raw_data(self) -> pl.DataFrame:
        """Collect the LazyFrame of the current selection, if not already collected.
        
        Returns:
            raw_data (polars.DataFrame): The collected data of the current selection.
        """
        if self._raw_data is None:
            self._raw_data = self.lazyframe.collect()
            if self._raw_data.shape[0] == 0:
                raise ValueError("No data exists for this filter.")
        return self._raw_data
    
    @staticmethod
    def _get_events(lazyframe: pl.LazyFrame):
        lazyframe = lazyframe.with_columns(((pl.col('Cycle') - pl.col('Cycle').shift() != 0)
                                    .fill_null(False).cum_sum()
                                    .alias('_cycle')))
        lazyframe = lazyframe.with_columns((((pl.col('Cycle') - pl.col('Cycle').shift() != 0) | (pl.col('Step') - pl.col('Step').shift() != 0))
                                    .fill_null(False).cum_sum()
                                    .alias('_step')))
        return lazyframe
    
    def filter_numerical(self, lazyframe: pl.LazyFrame, column: str, condition_number: int|list[int]) -> pl.Expr:
        if isinstance(condition_number, int):
            condition_number = [condition_number]
        elif isinstance(condition_number, list):
            condition_number = list(range(condition_number[0], condition_number[1] + 1))
        lazyframe = self._get_events(lazyframe)
        if condition_number is not None:
            return lazyframe.filter(pl.col(column).is_in(condition_number))
        else: 
            return lazyframe
    
    def plot(self, x, y, **kwargs):
        plt.plot(self.raw_data[x], self.raw_data[y], **kwargs)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend()
        

