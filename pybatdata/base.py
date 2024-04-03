"""Module for the Base class."""

import polars as pl
import matplotlib.pyplot as plt

class Base:
    """Base class for all filtering classes in PyBatData.

    Attributes:
            lazyframe (polars.LazyFrame): The lazyframe of data being filtered.
            cycles_idx (list): The indices of the cycles in the current filter.
            steps_idx (list): The indices of the steps in the current filter.
            step_names (list): The names of all of the steps in the procedure.
            _raw_data (polars.DataFrame): The collected dataframe of the current filter.
    """
    def __init__(self, lazyframe: pl.LazyFrame, 
                 cycles_idx: list | int, 
                 steps_idx: list | int, 
                 step_names: list):
        """ Create a filtering class.
        
        Args:
            lazyframe (polars.LazyFrame): The lazyframe of data being filtered.
            cycles_idx (list): The indices of the cycles in the current selection.
            steps_idx (list): The indices of the steps in the current selection.
            step_names (list): The names of all of the steps in the procedure.
        """
        self.cycles_idx = cycles_idx
        self.steps_idx = steps_idx
        self.step_names = step_names
        self.lazyframe = lazyframe
        self.zero_capacity()
        self.zero_time()
        self._raw_data = None

    def zero_capacity(self) -> None:
        """Recalculate the capacity column to start from zero at beginning of current selection."""
        self.lazyframe = self.lazyframe.with_columns([
            (pl.col("Capacity (Ah)") - pl.col("Capacity (Ah)").first()).alias("Capacity (Ah)")
        ]) 

    def zero_time(self) -> None:
        """Recalculate the time column to start from zero at beginning of current selection."""
        self.lazyframe = self.lazyframe.with_columns([
            (pl.col("Time (s)") - pl.col("Time (s)").first()).alias("Time (s)")
        ])
    
    @property
    def RawData(self) -> pl.DataFrame:
        """Collect the LazyFrame of the current selection, if not already collected.
        
        Returns:
            RawData (polars.DataFrame): The collected data of the current selection.
        """
        if self._raw_data is None:
            self._raw_data = self.lazyframe.collect()
        return self._raw_data
    
    @classmethod
    def get_conditions(cls, column: str, indices: list) -> pl.Expr:
        """Convert a list of indices for a column into a polars expression for filtering.
        
        Args:
            column (str): The column to filter.
            indices (list): The indices to filter.
            
        Returns:
            pl.Expr: The polars expression for filtering the column."""
        return pl.col(column).is_in(cls.flatten(indices)).alias(column)
    
    def plot(self, x, y, **kwargs):
        plt.plot(self.RawData[x], self.RawData[y], **kwargs)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend()
        

    @classmethod
    def flatten(cls, lst: list) -> list:
        """Flatten a list of lists into a single list.
        
        Args:
            lst (list): The list of lists to flatten.
            
        Returns:
            list: The flattened list."""
        if not isinstance(lst, list):
            return [lst]
        if lst == []:
            return lst
        if isinstance(lst[0], list):
            return cls.flatten(lst[0]) + cls.flatten(lst[1:])
        return lst[:1] + cls.flatten(lst[1:])