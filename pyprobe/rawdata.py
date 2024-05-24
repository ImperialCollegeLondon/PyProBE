"""A module for the RawData class."""
from typing import Any, Dict, Optional

import polars as pl

from pyprobe.methods.differentiation.feng_2020 import Feng2020
from pyprobe.result import Result
from pyprobe.units import Units


class RawData(Result):
    """A RawData object for returning data and plotting.

    Attributes:
        _data (pl.LazyFrame | pl.DataFrame): The filtered _data.
        dataframe (Optional[pl.DataFrame]): The data as a polars DataFrame.
        info (Dict[str, str | int | float]): A dictionary containing test info.
    """

    def __init__(
        self,
        _data: pl.LazyFrame | pl.DataFrame,
        info: Dict[str, str | int | float],
    ) -> None:
        """Initialize the RawData object.

        Args:
            _data (pl.LazyFrame | pl.DataFrame): The filtered _data.
            info (Dict[str, str | int | float]): A dict containing test info.
        """
        super().__init__(_data, info)
        self.data_property_called = False

    @property
    def data(self) -> pl.DataFrame:
        """Return the data as a polars DataFrame.

        Returns:
            pl.DataFrame: The data as a polars DataFrame.
        """
        if isinstance(self._data, pl.LazyFrame):
            instruction_list = []
            for column in self._data.columns:
                new_instruction = Units.convert_units(column)
                if new_instruction is not None:
                    instruction_list.extend(new_instruction)
            self._data = self._data.with_columns(instruction_list).collect()
        if self.data_property_called is False:
            instruction_list = []
            for column in self._data.columns:
                new_instruction = Units.set_zero(column)
                if new_instruction is not None:
                    instruction_list.extend(new_instruction)
                self._data = self._data.with_columns(instruction_list)
        if self._data.shape[0] == 0:
            raise ValueError("No data exists for this filter.")
        self.data_property_called = True
        return self._data

    @property
    def capacity(self) -> float:
        """Calculate the capacity passed during the step.

        Returns:
            float: The capacity passed during the step.
        """
        return abs(self.data["Capacity [Ah]"].max() - self.data["Capacity [Ah]"].min())

    def dQdV(self, method: str, *parameters: Any) -> Result:
        """Calculate the dQdV curves for the experiment.

        Args:
            method (str): The method to use for the calculation.
            parameters (dict): A dictionary of parameters for the method.

        Returns:
            Result: A Result object containing the dQdV curves.
        """
        method_dict = dict(
            {
                "feng_2020": Feng2020,
            }
        )
        return method_dict[method](self, *parameters).dQdV

    def set_SOC(
        self,
        reference_capacity: Optional[float] = None,
        reference_charge: Optional["RawData"] = None,
    ) -> None:
        """Add an SOC column to the data."""
        if reference_capacity is None:
            reference_capacity = (
                pl.col("Capacity [Ah]").max() - pl.col("Capacity [Ah]").min()
            )
        if reference_charge is None:
            # capacity_reference = pl.select(pl.col("Capacity [Ah]").max())
            self._data = self._data.with_columns(
                (
                    (
                        pl.col("Capacity [Ah]")
                        - pl.col("Capacity [Ah]").max()
                        + reference_capacity
                    )
                    / reference_capacity
                ).alias("SOC")
            )
        else:
            self.data
            fully_charged_reference_point = reference_charge.data.select(
                pl.col("Date").max()
            )[0][0]
            capacity_reference = (
                self._data.filter(pl.col("Date") == fully_charged_reference_point)
                .select("Capacity [Ah]")
                .head(1)
            )
            self._data = self._data.with_columns(
                (
                    (pl.col("Capacity [Ah]") - capacity_reference + reference_capacity)
                    / reference_capacity
                ).alias("SOC")
            )
