import polars as pl

class Filter:
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
    
    @classmethod
    def filter_numerical(cls, lazyframe: pl.LazyFrame, column: str, condition_number: int|list[int]) -> pl.Expr:
        if isinstance(condition_number, int):
            condition_number = [condition_number]
        elif isinstance(condition_number, list):
            condition_number = list(range(condition_number[0], condition_number[1] + 1))
        lazyframe = cls._get_events(lazyframe)
        if condition_number is not None:
            return lazyframe.filter(pl.col(column).is_in(condition_number) | pl.col(column + '_reversed').is_in(condition_number))
        else: 
            return lazyframe