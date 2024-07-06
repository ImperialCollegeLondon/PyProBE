"""A module for the RawData class."""
from typing import Any, Dict, Optional

import polars as pl

from pyprobe.methods import differentiation
from pyprobe.result import Result


class RawData(Result):
    """A RawData object for returning data.

    Attributes:
        _data (pl.LazyFrame | pl.DataFrame): The filtered _data with the following
            columns:

            - 'Date' (pl.Datetime): the timestamp of the measurement
            - 'Time [s]' (pl.Float64): the measurement time from the start of the
              filtered section in seconds
            - 'Step' (pl.Int64): the unique step number corresponding to a single
              instruction in the cycling program
            - 'Cycle' (pl.Int64): the cycle number, automatically identified when Step
              decreases
            - 'Event' (pl.Int64): the event number, automatically identified when Step
              changes
            - 'Current [A]' (pl.Float64): the current in Amperes
            - 'Voltage [V]' (pl.Float64): the voltage in Volts
            - 'Capacity [Ah]' (pl.Float64): the capacity relative to the start of the
              filtered section in Ampere-hours. Its value increases when charge
              current is passed and decreases when discharge current is passed.

        info (Dict[str, str | int | float]): A dictionary containing test info.
    """

    def __init__(
        self,
        _data: pl.LazyFrame | pl.DataFrame,
        info: Dict[str, str | int | float],
    ) -> None:
        """Initialize the RawData object."""
        super().__init__(_data, info)
        self.data_property_called = False

    @property
    def data(self) -> pl.DataFrame:
        """Return the data as a polars DataFrame.

        Returns:
            pl.DataFrame: The data as a polars DataFrame.
        """
        instruction_list = []
        zero_reference_list = ["Capacity [Ah]", "Time [s]"]
        for column in zero_reference_list:
            instruction_list.extend([pl.col(column) - pl.col(column).first()])
        self._data = self._data.with_columns(instruction_list)
        if isinstance(self._data, pl.LazyFrame):
            self._data = self._data.collect()
        if self._data.shape[0] == 0:
            raise ValueError("No data exists for this filter.")
        return self._data

    @property
    def capacity(self) -> float:
        """Calculate the capacity passed during the step.

        Returns:
            float: The capacity passed during the step.
        """
        return abs(self.data["Capacity [Ah]"].max() - self.data["Capacity [Ah]"].min())

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

    def gradient(
        self, method: str, x: str, y: str, *args: Any, **kwargs: Any
    ) -> Result:
        """Calculate the gradient of the data from a variety of methods.

        Args:
            method (str): The differentiation method.
            x (str): The x data column.
            y (str): The y data column.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Result: The result object from the gradient method.
        """
        return differentiation.method_dict[method](
            self, x, y, *args, **kwargs
        ).output_data
