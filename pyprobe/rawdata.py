"""A module for the RawData class."""
from typing import Any, Dict, Optional

import polars as pl

from pyprobe.analysis.differentiation import Differentiation
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

        self.define_column("Date", "The timestamp of the data point. Type: datetime.")
        self.define_column(
            "Time [s]",
            "The time in seconds passed from the start of the current filter.",
        )
        self.define_column("Current [A]", "The current in Amperes.")
        self.define_column("Voltage [V]", "The terminal voltage in Volts.")
        self.define_column(
            "Capacity [Ah]",
            "The net charge passed since the start of the current filter.",
        )

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

    def zero_column(
        self,
        column: str,
        new_column_name: str,
        new_column_definition: Optional[str] = None,
    ) -> None:
        """Add a new column to a dataframe or lazyframe, zeroed to the first value.

        Args:
            column (str): The column to zero.
            new_column_name (str): The new column name.
            new_column_definition (Optional[str]): The new column definition.

        Returns:
            pl.DataFrame | pl.LazyFrame: The dataframe or lazyframe with the new column.
        """
        self._data = self._data.with_columns(
            (pl.col(column) - pl.col(column).first()).alias(new_column_name)
        )
        if new_column_definition is not None:
            self.define_column(new_column_name, new_column_definition)

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
        """Add an SOC column to the data.

        Apply this method on a filtered data object to add an `SOC` column to the data.
        This column remains with the data if the object is filtered further.


        The SOC column is calculated either relative to a provided reference capacity
        value, a reference charge (provided as a RawData object), or the maximum
        capacity delta across the data in the RawData object upon which this method
        is called.

        Args:
            reference_capacity (Optional[float]): The reference capacity value.
            reference_charge (Optional[RawData]): The reference charge data.
        """
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

    def set_reference_capacity(
        self, reference_capacity: Optional[float] = None
    ) -> None:
        """Fix the capacity to a reference value.

        Apply this method on a filtered data object to fix the capacity to a reference.
        This calculates a permanent column named `Capacity - Referenced [Ah]` in the
        data, which remains if this object is filtered further.

        The reference value is either the maximum capacity delta across the data in the
        RawData object upon which this method is called or a user-specified value.

        Args:
            reference_capacity (Optional[float]): The reference capacity value.
        """
        if reference_capacity is None:
            reference_capacity = (
                pl.col("Capacity [Ah]").max() - pl.col("Capacity [Ah]").min()
            )
        self._data = self._data.with_columns(
            (
                pl.col("Capacity [Ah]")
                - pl.col("Capacity [Ah]").max()
                + reference_capacity
            ).alias("Capacity - Referenced [Ah]")
        )

    def gradient(
        self, x: str, y: str, method: str, *args: Any, **kwargs: Any
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
        differentiation = Differentiation(self)
        if method == "LEAN":
            return differentiation.differentiate_LEAN(x, y, *args, **kwargs)
        elif method == "Finite Difference":
            return differentiation.differentiate_FD(x, y, *args, **kwargs)
        else:
            raise ValueError("Invalid differentiation method.")
