import polars as pl
import matplotlib.pyplot as plt
class Base:
    def __init__(self, lf, cycles_idx, steps_idx, step_names):
        self.cycles_idx = cycles_idx
        self.steps_idx = steps_idx
        self.step_names = step_names
        self.lf = lf
        self.set_capacity()
        self._raw_data = None

    def set_capacity(self):
        self.lf = self.lf.with_columns([
            (pl.col("Capacity (Ah)") - pl.col("Capacity (Ah)").first()).alias("Capacity (Ah)")
        ]) 
    
    @property
    def RawData(self):
        if self._raw_data is None:
            self._raw_data = self.lf.collect()
        return self._raw_data
    
    @classmethod
    def get_conditions(cls,column, indices):
        return pl.col(column).is_in(cls.flatten(indices)).alias(column)
    
    def plot(self, x, y, **kwargs):
        plt.plot(self.RawData[x], self.RawData[y], **kwargs)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend()
        

    @classmethod
    def flatten(cls, lst):
        if not isinstance(lst, list):
            return [lst]
        if lst == []:
            return lst
        if isinstance(lst[0], list):
            return cls.flatten(lst[0]) + cls.flatten(lst[1:])
        return lst[:1] + cls.flatten(lst[1:])