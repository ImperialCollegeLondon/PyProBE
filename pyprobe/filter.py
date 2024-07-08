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

        if len(index_list) > 0:
            if all(item >= 0 for item in index_list):
                index_list = [item + 1 for item in index_list]
                return _data.filter(pl.col(column).rank("dense").is_in(index_list))
            elif all(item < 0 for item in index_list):
                index_list = [item * -1 for item in index_list]
                return _data.filter(
                    pl.col(column).rank("dense", descending=True).is_in(index_list)
                )
            else:
                raise ValueError("Indices must be all positive or all negative.")
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

    def constant_current(self, *constant_current_numbers: Union[int, range]) -> RawData:
        """Return a constant current step object.

        Args:
            constant_current_numbers (int | range): Variable-length argument list of
                constant current numbers or a range object.

        Returns:
            RawData: A constant current step object.
        """
        condition = (
            (pl.col("Current [A]") != 0)
            & (
                pl.col("Current [A]").abs()
                > 0.999 * pl.col("Current [A]").abs().round_sig_figs(4).mode()
            )
            & (
                pl.col("Current [A]").abs()
                < 1.001 * pl.col("Current [A]").abs().round_sig_figs(4).mode()
            )
        )
        return self.step(*constant_current_numbers, condition=condition)

    def constant_voltage(self, *constant_voltage_numbers: Union[int, range]) -> RawData:
        """Return a constant voltage step object.

        Args:
            constant_current_numbers (int | range): Variable-length argument list of
                constant voltage numbers or a range object.

        Returns:
            RawData: A constant voltage step object.
        """
        condition = (
            pl.col("Voltage [V]").abs()
            > 0.999 * pl.col("Voltage [V]").abs().round_sig_figs(4).mode()
        ) & (
            pl.col("Voltage [V]").abs()
            < 1.001 * pl.col("Voltage [V]").abs().round_sig_figs(4).mode()
        )
        return self.step(*constant_voltage_numbers, condition=condition)
