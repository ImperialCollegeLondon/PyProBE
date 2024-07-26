"""A module for the Cycing class."""

import polars as pl

from pyprobe.analysis.utils import BaseAnalysis, analysismethod
from pyprobe.filters import Experiment
from pyprobe.result import Result


class Cycling(Experiment, BaseAnalysis):
    """A cycling experiment in a battery procedure."""

    def __init__(
        self,
        experiment: Experiment,
    ):
        """Create a cycling experiment.

        Args:
            experiment (Experiment): The cycling experiment to be analysed.
        """
        super().__init__(experiment._data, experiment.info)
        self._create_capacity_throughput()

    def _create_capacity_throughput(self) -> None:
        """Calculate the capcity throughput of the experiment."""
        self._data = self._data.with_columns(
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

    @analysismethod
    def summary(self, dchg_before_chg: bool = True) -> Result:
        """Calculate the state of health of the battery.

        Args:
            dchg_before_chg (bool): Whether the discharge comes before the
                charge in the cycle loop. Default is True.

        Returns:
            Result: A result object for the capacity SOH of the cell.
        """
        lf_capacity_throughput = self._data.groupby("Cycle", maintain_order=True).agg(
            pl.col("Capacity Throughput [Ah]").first()
        )
        lf_time = self._data.groupby("Cycle", maintain_order=True).agg(
            pl.col("Time [s]").first()
        )

        lf_charge = (
            self.charge()
            ._data.groupby("Cycle", maintain_order=True)
            .agg(pl.col("Capacity [Ah]").max() - pl.col("Capacity [Ah]").min())
            .rename({"Capacity [Ah]": "Charge Capacity [Ah]"})
        )
        lf_discharge = (
            self.discharge()
            ._data.groupby("Cycle", maintain_order=True)
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
            "Time [s]": "The time since the beginning of the experiment.",
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
        return Result(lf, self.info, column_definitions=column_definitions)
