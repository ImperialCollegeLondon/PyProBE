"""A module for the Cycing class."""

from typing import Dict

import polars as pl

from pyprobe.experiment import Experiment
from pyprobe.filter import Filter
from pyprobe.result import Result


class Cycling(Experiment):
    """A cycling experiment in a battery procedure."""

    def __init__(
        self, _data: pl.LazyFrame | pl.DataFrame, info: Dict[str, str | int | float]
    ):
        """Create a cycling experiment.

        Args:
            _data (polars.LazyFrame): The _data of data being filtered.
            info (Dict[str, str | int | float]): A dict containing test info.
        """
        super().__init__(_data, info)
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

    def summary(self, dchg_before_chg: bool = True) -> Result:
        """Calculate the state of health of the battery.

        Args:
            dchg_before_chg (bool): Whether the discharge comes before the
                charge in the cycle loop. Default is True.

        Returns:
            Result: A result object for the capacity SOH of the cell.
        """
        self._data = Filter._get_events(self._data)
        lf_capacity_throughput = self._data.groupby("_cycle", maintain_order=True).agg(
            pl.col("Capacity Throughput [Ah]").first()
        )
        lf_time = self._data.groupby("_cycle", maintain_order=True).agg(
            pl.col("Time [s]").first()
        )

        lf_charge = (
            self.charge()
            ._data.groupby("_cycle", maintain_order=True)
            .agg(pl.col("Capacity [Ah]").max() - pl.col("Capacity [Ah]").min())
            .rename({"Capacity [Ah]": "Charge Capacity [Ah]"})
        )
        lf_discharge = (
            self.discharge()
            ._data.groupby("_cycle", maintain_order=True)
            .agg(pl.col("Capacity [Ah]").max() - pl.col("Capacity [Ah]").min())
            .rename({"Capacity [Ah]": "Discharge Capacity [Ah]"})
        )

        lf = (
            lf_capacity_throughput.join(lf_time, on="_cycle", how="outer_coalesce")
            .join(lf_charge, on="_cycle", how="outer_coalesce")
            .join(lf_discharge, on="_cycle", how="outer_coalesce")
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
        return Result(lf, self.info)
