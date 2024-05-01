"""A module for the Step class."""

from typing import Dict

import polars as pl

from pybatdata.methods.differentiation import feng_2020
from pybatdata.result import Result


class Step(Result):
    """A step in a battery test procedure."""

    def __init__(
        self, _data: pl.LazyFrame | pl.DataFrame, info: Dict[str, str | int | float]
    ):
        """Create a step.

        Args:
            _data (polars.LazyFrame): The _data of data being filtered.
            info (Dict[str, str | int | float]): A dict containing test info.
        """
        super().__init__(_data, info)

    @property
    def capacity(self) -> float:
        """Calculate the capacity passed during the step.

        Returns:
            float: The capacity passed during the step.
        """
        return abs(self.data["Capacity [Ah]"].max() - self.data["Capacity [Ah]"].min())

    def dQdV(self, method: str, parameter_dict: Dict[str, float]) -> Result:
        """Calculate the normalised incremental capacity of the step.

        Args:
            method (str): The method to use to calculate the incremental capacity.
            parameter_dict (Dict[str, float]): A dictionary containing
                the parameters for the method

        Returns:
            Result: a result object containing the normalised incremental capacity
        """
        method_dict = {"feng_2020": feng_2020.IC}
        return method_dict[method](self, parameter_dict)
