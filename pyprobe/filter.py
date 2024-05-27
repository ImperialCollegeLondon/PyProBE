"""A module for filtering data."""
from typing import Optional, Tuple, Union

import polars as pl

from pyprobe.rawdata import RawData


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
        indices: Tuple[Union[int, range], ...],
    ) -> pl.LazyFrame:
        """Filter a LazyFrame by a numerical condition.

        Args:
            _data (pl.LazyFrame | pl.DataFrame): A LazyFrame object.
            column (str): The column to filter on.
            indices (Tuple[Union[int, range], ...]): A tuple of index
                values to filter by.
        """
        index_list = []
        for index in indices:
            if isinstance(index, range):
                index_list.extend(list(index))
            else:
                index_list.extend([index])
        index_list = [item + 1 for item in index_list]
        if len(indices) > 0:
            return _data.filter(pl.col(column).rank("dense").is_in(index_list))
        else:
            return _data

    def step(
        self,
        *step_numbers: Union[int, range],
        condition: Optional[pl.Expr] = None,
    ) -> RawData:
        """Return a step object from the cycle.

        Args:
            step_number (int | range): Variable-length argument list of
                step numbers or a range object.

        Returns:
            RawData: A step object from the cycle.
        """
        if condition is not None:
            _data = self.filter_numerical(
                self._data.filter(condition), "Event", step_numbers
            )
        else:
            _data = self.filter_numerical(self._data, "Event", step_numbers)
        return RawData(_data, self.info)

    def cycle(self, *cycle_numbers: Union[int]) -> "Filter":
        """Return a cycle object from the experiment.

        Args:
            cycle_number (int | range): Variable-length argument list of
                cycle numbers or a range object.

        Returns:
            Filter: A filter object for the specified cycles.
        """
        lf_filtered = self.filter_numerical(self._data, "Cycle", cycle_numbers)
        return Filter(lf_filtered, self.info)

    def charge(self, *charge_numbers: Union[int, range]) -> RawData:
        """Return a charge step object from the cycle.

        Args:
            charge_number (int | range): Variable-length argument list of
                charge numbers or a range object.

        Returns:
            RawData: A charge step object from the cycle.
        """
        condition = pl.col("Current [A]") > 0
        return self.step(*charge_numbers, condition=condition)

    def discharge(self, *discharge_numbers: Union[int, range]) -> RawData:
        """Return a discharge step object from the cycle.

        Args:
            discharge_number (int | range): Variable-length argument list of
                discharge numbers or a range object.

        Returns:
            RawData: A discharge step object from the cycle.
        """
        condition = pl.col("Current [A]") < 0
        return self.step(*discharge_numbers, condition=condition)

    def chargeordischarge(
        self, *chargeordischarge_numbers: Union[int, range]
    ) -> RawData:
        """Return a charge or discharge step object from the cycle.

        Args:
            chargeordischarge_number (int | range): Variable-length argument list of
                charge or discharge numbers or a range object.

        Returns:
            RawData: A charge or discharge step object from the cycle.
        """
        condition = pl.col("Current [A]") != 0
        return self.step(*chargeordischarge_numbers, condition=condition)

    def rest(self, *rest_numbers: Union[int, range]) -> RawData:
        """Return a rest step object from the cycle.

        Args:
            rest_number (int | range): Variable-length argument list of rest
                numbers or a range object.

        Returns:
            RawData: A rest step object from the cycle.
        """
        condition = pl.col("Current [A]") == 0
        return self.step(*rest_numbers, condition=condition)
