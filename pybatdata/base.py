import polars as pl
class Base:
    @property
    def RawData(self):
        if self._raw_data is None:
            self._raw_data = self.lf.collect()
        return self._raw_data
    
    @classmethod
    def get_conditions(cls,column, indices):
        return pl.col(column).is_in(cls.flatten(indices)).alias(column)

    @classmethod
    def flatten(cls, lst):
        if not isinstance(lst, list):
            return [lst]
        if lst == []:
            return lst
        if isinstance(lst[0], list):
            return cls.flatten(lst[0]) + cls.flatten(lst[1:])
        return lst[:1] + cls.flatten(lst[1:])