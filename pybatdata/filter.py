"""A module for filtering data."""
import polars as pl


class Filter:
    """A class for filtering data."""

    @staticmethod
    def _get_events(lazyframe: pl.LazyFrame) -> pl.LazyFrame:
        """Get the events from cycle and step columns.

        Args:
            lazyframe: A LazyFrame object.

        Returns:
            lazyframe: A LazyFrame object with added _cycle and _step columns.
        """
        lazyframe = lazyframe.with_columns(
            (
                (pl.col("Cycle") - pl.col("Cycle").shift() != 0)
                .fill_null(strategy="zero")
                .cum_sum()
                .alias("_cycle")
                .cast(pl.Int32)
            )
        )
        lazyframe = lazyframe.with_columns(
            (
                (
                    (pl.col("Cycle") - pl.col("Cycle").shift() != 0)
                    | (pl.col("Step") - pl.col("Step").shift() != 0)
                )
                .fill_null(strategy="zero")
                .cum_sum()
                .alias("_step")
                .cast(pl.Int32)
            )
        )
        lazyframe = lazyframe.with_columns(
            [
                (pl.col("_cycle") - pl.col("_cycle").max() - 1).alias(
                    "_cycle_reversed"
                ),
                (pl.col("_step") - pl.col("_step").max() - 1).alias("_step_reversed"),
            ]
        )
        return lazyframe

    @classmethod
    def filter_numerical(
        cls,
        lazyframe: pl.LazyFrame,
        column: str,
        condition_number: int | list[int] | None,
    ) -> pl.LazyFrame:
        """Filter a LazyFrame by a numerical condition.

        Args:
            lazyframe (pl.LazyFrame): A LazyFrame object.
            column (str): The column to filter on.
            condition_number (int, list): A number or a list of numbers.
        """
        if isinstance(condition_number, int):
            condition_number = [condition_number]
        elif isinstance(condition_number, list):
            condition_number = list(range(condition_number[0], condition_number[1] + 1))
        lazyframe = cls._get_events(lazyframe)
        if condition_number is not None:
            return lazyframe.filter(
                pl.col(column).is_in(condition_number)
                | pl.col(column + "_reversed").is_in(condition_number)
            )
        else:
            return lazyframe
