"""A module for the Cycling class."""


import polars as pl
from pydantic import BaseModel

from pyprobe.analysis.utils import AnalysisValidator
from pyprobe.filters import Experiment, get_cycle_column
from pyprobe.result import Result


class Cycling(BaseModel):
    """A cycling experiment in a battery procedure."""

    input_data: Experiment
    """The input data for the cycling experiment."""

    def _create_capacity_throughput(self) -> None:
        """Calculate the capcity throughput of the input_data."""
        self.input_data.base_dataframe = self.input_data.base_dataframe.with_columns(
            [
                (
                    pl.col("Capacity [Ah]")
                    .diff()
                    .fill_null(strategy="zero")
                    .abs()
                    .cum_sum()
                ).alias("Capacity Throughput [Ah]")
            ]
        )

    def summary(self, dchg_before_chg: bool = True) -> Result:
        """Calculate the state of health of the battery.

        Args:
            dchg_before_chg (bool): Whether the discharge comes before the
                charge in the cycle loop. Default is True.

        Returns:
            Result: A result object for the capacity SOH of the cell.
        """
        AnalysisValidator(
            input_data=self.input_data, required_columns=["Capacity [Ah]", "Time [s]"]
        )
        self.input_data.base_dataframe = get_cycle_column(self.input_data)

        self._create_capacity_throughput()
        lf_capacity_throughput = self.input_data.base_dataframe.group_by(
            "Cycle", maintain_order=True
        ).agg(pl.col("Capacity Throughput [Ah]").first())
        lf_time = self.input_data.base_dataframe.group_by(
            "Cycle", maintain_order=True
        ).agg(pl.col("Time [s]").first())

        lf_charge = (
            self.input_data.charge()
            .base_dataframe.group_by("Cycle", maintain_order=True)
            .agg(pl.col("Capacity [Ah]").max() - pl.col("Capacity [Ah]").min())
            .rename({"Capacity [Ah]": "Charge Capacity [Ah]"})
        )
        lf_discharge = (
            self.input_data.discharge()
            .base_dataframe.group_by("Cycle", maintain_order=True)
            .agg(pl.col("Capacity [Ah]").max() - pl.col("Capacity [Ah]").min())
            .rename({"Capacity [Ah]": "Discharge Capacity [Ah]"})
        )

        lf = (
            lf_capacity_throughput.join(lf_time, on="Cycle", how="outer_coalesce")
            .join(lf_charge, on="Cycle", how="outer_coalesce")
            .join(lf_discharge, on="Cycle", how="outer_coalesce")
        )

        lf = lf.with_columns(
            (
                pl.col("Charge Capacity [Ah]") / pl.first("Charge Capacity [Ah]") * 100
            ).alias("SOH Charge [%]")
        )
        lf = lf.with_columns(
            (
                pl.col("Discharge Capacity [Ah]")
                / pl.first("Discharge Capacity [Ah]")
                * 100
            ).alias("SOH Discharge [%]")
        )

        if dchg_before_chg:
            lf = lf.with_columns(
                (
                    pl.col("Discharge Capacity [Ah]")
                    / pl.col("Charge Capacity [Ah]").shift()
                ).alias("Coulombic Efficiency")
            )
        else:
            (
                pl.col("Discharge Capacity [Ah]").shift()
                / pl.col("Charge Capacity [Ah]")
            ).alias("Coulombic Efficiency")
        column_definitions = {
            "Cycle": "The cycle number.",
            "Capacity Throughput [Ah]": "The cumulative capacity throughput.",
            "Time [s]": "The time since the beginning of the input_data.",
            "Charge Capacity [Ah]": "The capacity passed during charge in a cycle.",
            "Discharge Capacity [Ah]": (
                "The capacity passed during discharge in a cycle."
            ),
            "SOH Charge [%]": (
                "The charge passed during charge normalized to the first charge."
            ),
            "SOH Discharge [%]": (
                "The charge passed during discharge normalised to the first discharge."
            ),
            "Coulombic Efficiency": (
                "The ratio between a discharge and its preceding charge."
            ),
        }
        return Result(
            base_dataframe=lf,
            info=self.input_data.info,
            column_definitions=column_definitions,
        )
