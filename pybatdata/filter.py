"""A module for filtering data."""
from typing import List, Optional

import polars as pl

from pybatdata.rawdata import RawData


class Filter(RawData):
    """A class for filtering data."""

    def __init__(
        self, _data: pl.LazyFrame | pl.DataFrame, info: dict[str, str | int | float]
    ):
        """Create a filter object.

        Args:
            _data (pl.LazyFrame | pl.DataFrame): A LazyFrame object.
            info (dict): A dict containing test info.
        """
        super().__init__(_data, info)

    @staticmethod
    def _get_events(_data: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
        """Get the events from cycle and step columns.

        Args:
            _data: A LazyFrame object.

        Returns:
            _data: A LazyFrame object with added _cycle and _step columns.
        """
        _data = _data.with_columns(
            (
                (pl.col("Cycle") - pl.col("Cycle").shift() != 0)
                .fill_null(strategy="zero")
                .cum_sum()
                .alias("_cycle")
                .cast(pl.Int32)
            )
        )
        _data = _data.with_columns(
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
        _data = _data.with_columns(
            [
                (pl.col("_cycle") - pl.col("_cycle").max() - 1).alias(
                    "_cycle_reversed"
                ),
                (pl.col("_step") - pl.col("_step").max() - 1).alias("_step_reversed"),
            ]
        )
        return _data

    @classmethod
    def filter_numerical(
        cls,
        _data: pl.LazyFrame | pl.DataFrame,
        column: str,
        condition_number: int | list[int] | None,
    ) -> pl.LazyFrame:
        """Filter a LazyFrame by a numerical condition.

        Args:
            _data (pl.LazyFrame | pl.DataFrame): A LazyFrame object.
            column (str): The column to filter on.
            condition_number (int, list): A number or a list of numbers.
        """
        if isinstance(condition_number, int):
            condition_number = [condition_number]
        elif isinstance(condition_number, list):
            condition_number = list(range(condition_number[0], condition_number[1] + 1))
        _data = cls._get_events(_data)
        if condition_number is not None:
            return _data.filter(
                pl.col(column).is_in(condition_number)
                | pl.col(column + "_reversed").is_in(condition_number)
            )
        else:
            return _data

    def step(
        self,
        step_number: Optional[int | List[int]] = None,
        condition: Optional[pl.Expr] = None,
    ) -> RawData:
        """Return a step object from the cycle.

        Args:
            step_number (int): The step number to return.

        Returns:
            Result: A step object from the cycle.
        """
        if condition is not None:
            _data = self.filter_numerical(
                self._data.filter(condition), "_step", step_number
            )
        else:
            _data = self.filter_numerical(self._data, "_step", step_number)
        return RawData(_data, self.info)

    def cycle(self, cycle_number: int) -> "Filter":
        """Return a cycle object from the experiment.

        Args:
            cycle_number (int): The cycle number to return.

        Returns:
            Cycle: A cycle object from the experiment.
        """
        lf_filtered = self.filter_numerical(self._data, "_cycle", cycle_number)
        return Filter(lf_filtered, self.info)

    def charge(self, charge_number: Optional[int] = None) -> RawData:
        """Return a charge step object from the cycle.

        Args:
            charge_number (int): The charge number to return.

        Returns:
            RawData: A charge step object from the cycle.
        """
        condition = pl.col("Current [A]") > 0
        return self.step(charge_number, condition)

    def discharge(self, discharge_number: Optional[int] = None) -> RawData:
        """Return a discharge step object from the cycle.

        Args:
            discharge_number (int): The discharge number to return.

        Returns:
            RawData: A discharge step object from the cycle.
        """
        condition = pl.col("Current [A]") < 0
        return self.step(discharge_number, condition)

    def chargeordischarge(
        self, chargeordischarge_number: Optional[int] = None
    ) -> RawData:
        """Return a charge or discharge step object from the cycle.

        Args:
            chargeordischarge_number (int): The charge or discharge number to return.

        Returns:
            RawData: A charge or discharge step object from the cycle.
        """
        condition = pl.col("Current [A]") != 0
        return self.step(chargeordischarge_number, condition)

    def rest(self, rest_number: Optional[int] = None) -> RawData:
        """Return a rest step object from the cycle.

        Args:
            rest_number (int): The rest number to return.

        Returns:
            RawData: A rest step object from the cycle.
        """
        condition = pl.col("Current [A]") == 0
        return self.step(rest_number, condition)
