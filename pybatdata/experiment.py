"""A module for the Experiment class."""

from typing import Dict

import polars as pl

from pybatdata.cycle import Cycle
from pybatdata.filter import Filter


class Experiment(Cycle):
    """An experiment in a battery procedure."""

    def __init__(
        self, _data: pl.LazyFrame | pl.DataFrame, info: Dict[str, str | int | float]
    ):
        """Create an experiment.

        Args:
            _data (polars.LazyFrame): The _data of data being filtered.
            info (Dict[str, str | int | float]): A dict containing test info.
        """
        super().__init__(_data, info)

    def cycle(self, cycle_number: int) -> Cycle:
        """Return a cycle object from the experiment.

        Args:
            cycle_number (int): The cycle number to return.

        Returns:
            Cycle: A cycle object from the experiment.
        """
        lf_filtered = Filter.filter_numerical(self._data, "_cycle", cycle_number)
        return Cycle(lf_filtered, self.info)
