"""A module for the RawData class."""
from typing import Dict, Optional

import polars as pl
from pydantic import Field, field_validator

# from pyprobe.analysis.differentiation import Differentiation
from pyprobe.result import Result

required_columns = ["Date", "Time [s]", "Current [A]", "Voltage [V]", "Capacity [Ah]"]

default_column_definitions = {
    "Date": "The timestamp of the data point. Type: datetime.",
    "Time [s]": "The time passed from the start of the procedure.",
    "Current [A]": "The current through the cell.",
    "Voltage [V]": "The terminal voltage.",
    "Capacity [Ah]": "The net charge passed since the start of the procedure.",
}


class RawData(Result):
    """A RawData object for returning data.

    Args:
        base_dataframe (Union[pl.LazyFrame, pl.DataFrame]):
            The data as a polars DataFrame or LazyFrame.
        info (Dict[str, Union[str, int, float]]):
            A dictionary containing test info.
        column_definitions (Dict[str, str], optional):
            A dictionary containing the definitions of the columns in the data.
    """

    base_dataframe: pl.LazyFrame | pl.DataFrame
    info: Dict[str, Optional[str | int | float]]
    column_definitions: Dict[str, str] = Field(
        default_factory=lambda: default_column_definitions.copy()
    )
    """A dictionary containing the definitions of the columns in the data."""

    @field_validator("base_dataframe")
    @classmethod
    def _check_required_columns(
        cls, dataframe: pl.LazyFrame | pl.DataFrame
    ) -> "RawData":
        """Check if the required columns are present in the input_data."""
        missing_columns = [
            col
            for col in required_columns
            if col not in dataframe.collect_schema().names()
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        return dataframe

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
        self.base_dataframe = self.base_dataframe.with_columns(
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
            reference_charge (Optional[RawData]): A rawdata object containing a charge
                cycle to use as a reference.
        """
        if reference_capacity is None:
            reference_capacity = (
                pl.col("Capacity [Ah]").max() - pl.col("Capacity [Ah]").min()
            )
        if reference_charge is None:
            self.base_dataframe = self.base_dataframe.with_columns(
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
            if self.contains_lazyframe:
                reference_charge_data = reference_charge.base_dataframe.select(
                    "Time [s]", "Capacity [Ah]"
                )
                self.base_dataframe = self.base_dataframe.join(
                    reference_charge_data, on="Time [s]", how="left"
                )
                self.base_dataframe = self.base_dataframe.with_columns(
                    pl.col("Capacity [Ah]_right")
                    .max()
                    .alias("Full charge reference capacity"),
                ).drop("Capacity [Ah]_right")
            else:
                full_charge_reference_capacity = (
                    reference_charge.data.select("Capacity [Ah]").max().item()
                )
                self.base_dataframe = self.base_dataframe.with_columns(
                    pl.lit(full_charge_reference_capacity).alias(
                        "Full charge reference capacity"
                    ),
                )

            self.base_dataframe = self.base_dataframe.with_columns(
                (
                    (
                        pl.col("Capacity [Ah]")
                        - pl.col("Full charge reference capacity")
                        + reference_capacity
                    )
                    / reference_capacity
                ).alias("SOC")
            )
        self.define_column("SOC", "The full cell State-of-Charge.")

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
        self.base_dataframe = self.base_dataframe.with_columns(
            (
                pl.col("Capacity [Ah]")
                - pl.col("Capacity [Ah]").max()
                + reference_capacity
            ).alias("Capacity - Referenced [Ah]")
        )

    # def gradient(
    #     self, x: str, y: str, method: str, *args: Any, **kwargs: Any
    # ) -> Result:
    #     """Calculate the gradient of the data from a variety of methods.

    #     Args:
    #         method (str): The differentiation method.
    #         x (str): The x data column.
    #         y (str): The y data column.
    #         *args: Additional arguments.
    #         **kwargs: Additional keyword arguments.

    #     Returns:
    #         Result: The result object from the gradient method.
    #     """
    #     differentiation = Differentiation(rawdata=self)
    #     if method == "LEAN":
    #         return differentiation.differentiate_LEAN(x, y, *args, **kwargs)
    #     elif method == "Finite Difference":
    #         return differentiation.differentiate_FD(x, y, *args, **kwargs)
    #     else:
    #         raise ValueError("Invalid differentiation method.")
