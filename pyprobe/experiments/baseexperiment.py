"""A module for the BaseExperiment class."""

from typing import Dict

import polars as pl

from pyprobe.filter import Filter


class BaseExperiment(Filter):
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