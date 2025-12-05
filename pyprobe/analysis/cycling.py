"""A module for the Cycling class."""

import polars as pl
from pydantic import validate_call

from pyprobe.analysis.utils import AnalysisValidator
from pyprobe.filters import get_cycle_column
from pyprobe.pyprobe_types import FilterToCycleType
from pyprobe.result import Result


def _create_capacity_throughput(
    data: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame | pl.LazyFrame:
    """Add a column to the input data with the cumulative capacity throughput.

    Args:
        data: The input data.

    Returns:
        pl.DataFrame: The input data with the cumulative capacity throughput.
    """
    return data.with_columns(
        [
            (
                pl.col("Capacity [Ah]")
                .diff()
                .fill_null(strategy="zero")
                .abs()
                .cum_sum()
            ).alias("Capacity Throughput [Ah]"),
        ],
    )


@validate_call
def summary(input_data: FilterToCycleType, dchg_before_chg: bool = True) -> Result:
    """Calculate the state of health of the battery.

    Args:
        input_data: A PyProBE object containing cycling data.
        dchg_before_chg (bool): Whether the discharge comes before the
            charge in the cycle loop. Default is True.

    Returns:
        Result: A result object for the capacity SOH of the cell.
    """
    AnalysisValidator(
        input_data=input_data,
        required_columns=["Capacity [Ah]", "Time [s]"],
    )
    input_data.lf = get_cycle_column(input_data)

    input_data.lf = _create_capacity_throughput(input_data.lf)
    lf_capacity_throughput = input_data.lf.group_by(
        "Cycle",
        maintain_order=True,
    ).agg(pl.col("Capacity Throughput [Ah]").first())
    lf_time = input_data.lf.group_by("Cycle", maintain_order=True).agg(
        pl.col("Time [s]").first(),
    )

    lf_charge = (
        input_data.charge()
        .lf.group_by("Cycle", maintain_order=True)
        .agg(pl.col("Capacity [Ah]").max() - pl.col("Capacity [Ah]").min())
        .rename({"Capacity [Ah]": "Charge Capacity [Ah]"})
    )
    lf_discharge = (
        input_data.discharge()
        .lf.group_by("Cycle", maintain_order=True)
        .agg(pl.col("Capacity [Ah]").max() - pl.col("Capacity [Ah]").min())
        .rename({"Capacity [Ah]": "Discharge Capacity [Ah]"})
    )

    lf = (
        lf_capacity_throughput.join(lf_time, on="Cycle", how="outer_coalesce")
        .join(lf_charge, on="Cycle", how="outer_coalesce")
        .join(lf_discharge, on="Cycle", how="outer_coalesce")
    )

    lf = lf.with_columns(
        (pl.col("Charge Capacity [Ah]") / pl.first("Charge Capacity [Ah]") * 100).alias(
            "SOH Charge [%]",
        ),
    )
    lf = lf.with_columns(
        (
            pl.col("Discharge Capacity [Ah]")
            / pl.first("Discharge Capacity [Ah]")
            * 100
        ).alias("SOH Discharge [%]"),
    )

    if dchg_before_chg:
        lf = lf.with_columns(
            (
                pl.col("Discharge Capacity [Ah]")
                / pl.col("Charge Capacity [Ah]").shift()
            ).alias("Coulombic Efficiency"),
        )
    else:
        (
            pl.col("Discharge Capacity [Ah]").shift() / pl.col("Charge Capacity [Ah]")
        ).alias("Coulombic Efficiency")
    column_definitions = {
        "Cycle": "The cycle number.",
        "Capacity Throughput": "The cumulative capacity throughput.",
        "Time": "The time since the beginning of the input_data.",
        "Charge Capacity": "The capacity passed during charge in a cycle.",
        "Discharge Capacity": ("The capacity passed during discharge in a cycle."),
        "SOH Charge": (
            "The charge passed during charge normalized to the first charge."
        ),
        "SOH Discharge": (
            "The charge passed during discharge normalised to the first discharge."
        ),
        "Coulombic Efficiency": (
            "The ratio between a discharge and its preceding charge."
        ),
    }
    return Result(
        lf=lf,
        info=input_data.info,
        column_definitions=column_definitions,
    )
