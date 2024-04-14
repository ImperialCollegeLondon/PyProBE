"""A module for the Experiment class."""

from typing import Dict

import polars as pl

from pybatdata.cycle import Cycle
from pybatdata.filter import Filter


class Experiment(Cycle):
    """An experiment in a battery procedure."""

    def __init__(self, lazyframe: pl.LazyFrame, info: Dict[str, str | int | float]):
        """Create an experiment.

        Args:
            lazyframe (polars.LazyFrame): The lazyframe of data being filtered.
            info (Dict[str, str | int | float]): A dict containing test info.
        """
        super().__init__(lazyframe, info)

    def cycle(self, cycle_number: int) -> Cycle:
        """Return a cycle object from the experiment.

        Args:
            cycle_number (int): The cycle number to return.

        Returns:
            Cycle: A cycle object from the experiment.
        """
        lf_filtered = Filter.filter_numerical(self.lazyframe, "_cycle", cycle_number)
        return Cycle(lf_filtered, self.info)
