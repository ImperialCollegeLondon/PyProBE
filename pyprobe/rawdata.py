"""A module for the RawData class."""

from typing import Optional

import polars as pl
from loguru import logger
from pydantic import Field, field_validator

from pyprobe.result import Result
from pyprobe.units import split_quantity_unit
from pyprobe.utils import deprecated

required_columns = [
    "Time [s]",
    "Step",
    "Event",
    "Current [A]",
    "Voltage [V]",
    "Capacity [Ah]",
]

default_column_definitions = {
    "Date": "The timestamp of the data point. Type: datetime.",
    "Time": "The time passed from the start of the procedure.",
    "Step": "The step number.",
    "Cycle": "The cycle number.",
    "Event": "The event number. Counts the changes in cycles and steps.",
    "Current": "The current through the cell.",
    "Voltage": "The terminal voltage.",
    "Capacity": "The net charge passed since the start of the procedure.",
    "Temperature": "The temperature of the cell.",
}


class RawData(Result):
    """A class for holding data in the PyProBE format.

    This is the default object returned when data is loaded into PyProBE with the
    standard methods of the `pyprobe.cell.Cell` class. It is a subclass of the
    `pyprobe.result.Result` class so can be used in the same way as other result
    objects.

    The RawData object is stricter than the `pyprobe.result.Result` object in that it
    requires the presence of specific columns in the data. These columns are:
        - `Time [s]`
        - `Step`
        - `Cycle`
        - `Event`
        - `Current [A]`
        - `Voltage [V]`
        - `Capacity [Ah]`

    This defines the PyProBE format.
    """

    column_definitions: dict[str, str] = Field(
        default_factory=lambda: default_column_definitions.copy(),
    )
    step_descriptions: dict[str, list[str | int | None]] = {}
    """A dictionary containing the fields 'Step' and 'Description'.

    - 'Step' is a list of step numbers.
    - 'Description' is a list of corresponding descriptions in PyBaMM Experiment format.
    """

    @field_validator("lf", mode="after")
    @classmethod
    def check_required_columns(
        cls,
        dataframe: pl.LazyFrame,
    ) -> "RawData":
        """Check if the required columns are present in the input_data."""
        columns = dataframe.collect_schema().names()
        missing_columns = [col for col in required_columns if col not in columns]
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        return dataframe

    @property
    def data(self) -> pl.DataFrame:
        """Return the data as a polars DataFrame.

        Returns:
            pl.DataFrame: The data as a polars DataFrame.

        Raises:
            ValueError: If no data exists for this filter.
        """
        dataframe = super().data
        unsorted_columns = set(dataframe.collect_schema().names()) - set(
            required_columns,
        )
        sorted_columns = list(required_columns) + list(unsorted_columns)
        return dataframe.select(sorted_columns)

    def zero_column(
        self,
        column: str,
        new_column_name: str,
        new_column_definition: str | None = None,
    ) -> None:
        """Set the first value of a column to zero.

        Args:
            column (str): The column to zero.
            new_column_name (str): The new column name.
            new_column_definition (Optional[str]): The new column definition.
        """
        self.lf = self.lf.with_columns(
            (pl.col(column) - pl.col(column).first()).alias(new_column_name),
        )
        new_column_quantity, _ = split_quantity_unit(new_column_name)
        if new_column_definition is not None:
            self.define_column(new_column_quantity, new_column_definition)
        else:
            self.define_column(
                new_column_quantity,
                f"{column} with first value zeroed.",
            )

    @property
    def capacity(self) -> float:
        """Calculate the net capacity passed.

        Returns:
            float: The net capacity passed.
        """
        return abs(self.data["Capacity [Ah]"].max() - self.data["Capacity [Ah]"].min())

    def set_soc(
        self,
        reference_capacity: float | None = None,
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
            reference_charge (Optional[RawData]):
                A RawData object containing a charge to use as a reference.
        """
        if reference_capacity is None:
            reference_capacity = (
                pl.col("Capacity [Ah]").max() - pl.col("Capacity [Ah]").min()
            )
        if reference_charge is None:
            self.lf = self.lf.with_columns(
                (
                    (
                        pl.col("Capacity [Ah]")
                        - pl.col("Capacity [Ah]").max()
                        + reference_capacity
                    )
                    / reference_capacity
                ).alias("SOC"),
            )
        else:
            reference_charge_data = reference_charge.lf.select(
                "Time [s]",
                "Capacity [Ah]",
            )
            self.lf = self.lf.join(
                reference_charge_data,
                on="Time [s]",
                how="left",
            )
            self.lf = self.lf.with_columns(
                pl.col("Capacity [Ah]_right")
                .max()
                .alias("Full charge reference capacity"),
            ).drop("Capacity [Ah]_right")

            self.lf = self.lf.with_columns(
                (
                    (
                        pl.col("Capacity [Ah]")
                        - pl.col("Full charge reference capacity")
                        + reference_capacity
                    )
                    / reference_capacity
                ).alias("SOC"),
            )
        self.define_column("SOC", "The full cell State-of-Charge.")

    @deprecated(
        reason="Use set_soc instead.",
        version="2.0.1",
    )
    def set_SOC(  # noqa: N802
        self,
        reference_capacity: float | None = None,
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
            reference_charge (Optional[RawData]):
                A RawData object containing a charge to use as a reference.
        """
        self.set_soc(reference_capacity, reference_charge)

    def set_reference_capacity(self, reference_capacity: float | None = None) -> None:
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
        self.lf = self.lf.with_columns(
            (
                pl.col("Capacity [Ah]")
                - pl.col("Capacity [Ah]").max()
                + reference_capacity
            ).alias("Capacity - Referenced [Ah]"),
        )

    @property
    def pybamm_experiment(self) -> list[str | tuple[str]]:
        """Return a list of operating conditions for a PyBaMM experiment object.

        These can be passed directly to pybamm.Experiment() to create an experiment
        for use with PyBaMM.

        PyProBE does not check the validity of the operating condition strings. When
        creating the Experiment object, PyBaMM will raise an error if the operating
        conditions are not valid. The user should then modify the step descriptions
        in the readme file accordingly.

        Returns:
            The PyBaMM operating conditions.
        """
        # reduce the full dataframe to only the steps as they appear in order in
        # the data
        only_steps = (
            self.lf.with_row_index()
            .group_by("Event", maintain_order=True)
            .agg(pl.col("Step").first())
        )
        if isinstance(only_steps, pl.LazyFrame):
            only_steps = only_steps.collect()

        step_description_df = pl.DataFrame(self.step_descriptions)
        no_step_descriptions = step_description_df.filter(
            pl.col("Description").is_null(),
        )
        missing_steps = no_step_descriptions.select("Step").to_numpy().flatten()
        if len(missing_steps) > 0:
            error_msg = (
                f"Descriptions for steps {str(missing_steps)} are missing."
                f" Unable to create a PyBaMM experiment object. Please "
                f"filter the data to a section with descriptions for all "
                f"steps to create an experiment."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # match the step with its description
        all_steps_with_descriptions = only_steps.join(
            step_description_df,
            on="Step",
            how="left",
        ).select("Description")
        # form a list of all the descriptions
        all_steps_with_descriptions = all_steps_with_descriptions.to_numpy().flatten()
        description_list = []
        for description in all_steps_with_descriptions:
            line = description.split(",")
            for item in line:
                description_list.append(item.strip())
        return description_list
