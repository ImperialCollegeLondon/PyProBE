"""A module for the Cycling class."""

import polars as pl

from pyprobe.analysis.utils import build_result, validate_columns
from pyprobe.columns import BDF
from pyprobe.filters import get_cycle_column
from pyprobe.pyprobe_types import FilterToCycleType
from pyprobe.result import Table


def summary(input_data: FilterToCycleType, dchg_before_chg: bool = True) -> Table:
    """Calculate the state of health of the battery.

    Args:
        input_data: A PyProBE object containing cycling data.
        dchg_before_chg (bool): Whether the discharge comes before the
            charge in the cycle loop. Default is True.

    Returns:
        Table: A result object for the capacity SOH of the cell.

    Raises:
        ColumnResolutionError: If required columns cannot be resolved from `input_data`.
    """
    validate_columns(input_data, BDF.NET_CAPACITY_AH, BDF.TEST_TIME_SECOND)
    input_data.lf = get_cycle_column(input_data)

    cumulative_expr = input_data.columns.resolve(BDF.CUMULATIVE_CAPACITY_AH)
    input_data.lf = input_data.lf.with_columns(cumulative_expr)
    lf_capacity_throughput = (
        input_data.lf.group_by(BDF.CYCLE_COUNT.name, maintain_order=True)
        .agg(pl.col(BDF.CUMULATIVE_CAPACITY_AH.name).first())
        .rename({BDF.CUMULATIVE_CAPACITY_AH.name: "Capacity Throughput / Ah"})
    )
    time_expr = input_data.columns.resolve(BDF.TEST_TIME_SECOND)
    lf_time = (
        input_data.lf.with_columns(time_expr)
        .group_by(BDF.CYCLE_COUNT.name, maintain_order=True)
        .agg(pl.col(BDF.TEST_TIME_SECOND.name).first().alias(BDF.TEST_TIME_SECOND.name))
    )

    lf_charge = (
        input_data.charge()
        .lf.group_by(BDF.CYCLE_COUNT.name, maintain_order=True)
        .agg(
            pl.col(BDF.NET_CAPACITY_AH.name).max()
            - pl.col(BDF.NET_CAPACITY_AH.name).min()
        )
        .rename({BDF.NET_CAPACITY_AH.name: "Charge Capacity / Ah"})
    )
    lf_discharge = (
        input_data.discharge()
        .lf.group_by(BDF.CYCLE_COUNT.name, maintain_order=True)
        .agg(
            pl.col(BDF.NET_CAPACITY_AH.name).max()
            - pl.col(BDF.NET_CAPACITY_AH.name).min()
        )
        .rename({BDF.NET_CAPACITY_AH.name: "Discharge Capacity / Ah"})
    )

    lf = (
        lf_capacity_throughput.join(
            lf_time, on=BDF.CYCLE_COUNT.name, how="outer_coalesce"
        )
        .join(lf_charge, on=BDF.CYCLE_COUNT.name, how="outer_coalesce")
        .join(lf_discharge, on=BDF.CYCLE_COUNT.name, how="outer_coalesce")
    ).sort(BDF.CYCLE_COUNT.name)

    lf = lf.with_columns(
        (pl.col("Charge Capacity / Ah") / pl.first("Charge Capacity / Ah") * 100).alias(
            "SOH Charge / %",
        ),
    )
    lf = lf.with_columns(
        (
            pl.col("Discharge Capacity / Ah")
            / pl.first("Discharge Capacity / Ah")
            * 100
        ).alias("SOH Discharge / %"),
    )

    if dchg_before_chg:
        lf = lf.with_columns(
            (
                pl.col("Discharge Capacity / Ah")
                / pl.col("Charge Capacity / Ah").shift()
            ).alias("Coulombic Efficiency"),
        )
    else:
        (
            pl.col("Discharge Capacity / Ah").shift() / pl.col("Charge Capacity / Ah")
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
    return build_result(input_data, lf, column_definitions=column_definitions)
