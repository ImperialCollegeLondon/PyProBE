"""A module for the Cycle class."""

from typing import Dict, List, Optional

import polars as pl

from pybatdata.filter import Filter
from pybatdata.result import Result
from pybatdata.step import Step


class Cycle(Result):
    """A cycle in a battery procedure."""

    def __init__(
        self, _data: pl.LazyFrame | pl.DataFrame, info: Dict[str, str | int | float]
    ):
        """Create a cycle.

        Args:
            _data (polars.LazyFrame): The _data of data being filtered.
            info (Dict[str, str | int | float]): A dict containing test info.
        """
        super().__init__(_data, info)

    def step(
        self,
        step_number: Optional[int | List[int]] = None,
        condition: Optional[pl.Expr] = None,
    ) -> Step:
        """Return a step object from the cycle.

        Args:
            step_number (int): The step number to return.

        Returns:
            Step: A step object from the cycle.
        """
        if condition is not None:
            _data = Filter.filter_numerical(
                self._data.filter(condition), "_step", step_number
            )
        else:
            _data = Filter.filter_numerical(self._data, "_step", step_number)
        return Step(_data, self.info)

    def charge(self, charge_number: Optional[int] = None) -> Step:
        """Return a charge step object from the cycle.

        Args:
            charge_number (int): The charge number to return.

        Returns:
            Step: A charge step object from the cycle.
        """
        condition = pl.col("Current [A]") > 0
        return self.step(charge_number, condition)

    def discharge(self, discharge_number: Optional[int] = None) -> Step:
        """Return a discharge step object from the cycle.

        Args:
            discharge_number (int): The discharge number to return.

        Returns:
            Step: A discharge step object from the cycle.
        """
        condition = pl.col("Current [A]") < 0
        return self.step(discharge_number, condition)

    def chargeordischarge(self, chargeordischarge_number: Optional[int] = None) -> Step:
        """Return a charge or discharge step object from the cycle.

        Args:
            chargeordischarge_number (int): The charge or discharge number to return.

        Returns:
            Step: A charge or discharge step object from the cycle.
        """
        condition = pl.col("Current [A]") != 0
        return self.step(chargeordischarge_number, condition)

    def rest(self, rest_number: Optional[int] = None) -> Step:
        """Return a rest step object from the cycle.

        Args:
            rest_number (int): The rest number to return.

        Returns:
            Step: A rest step object from the cycle.
        """
        condition = pl.col("Current [A]") == 0
        return self.step(rest_number, condition)
