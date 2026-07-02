"""A module for the Cycling class."""

import polars as pl

from pyprobe.analysis.utils import build_result, validate_columns
from pyprobe.columns import BDF
from pyprobe.filters import get_cycle_column
from pyprobe.pyprobe_types import FilterToCycleType
from pyprobe.result import _STAT_SUITE, Table


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
    validate_columns(input_data, BDF.NET_CAPACITY_AH, BDF.TEST_TIME_SECOND, BDF.STEP_ID)

    cycle_lf = Table(
        lf=get_cycle_column(input_data),
        metadata=input_data.metadata,
        column_definitions=input_data.column_definitions,
    )
    resolved_exprs = [
        cycle_lf.columns.resolve(BDF.CUMULATIVE_CAPACITY_AH),
        cycle_lf.columns.resolve(BDF.TEST_TIME_SECOND),
        cycle_lf.columns.resolve(BDF.CYCLE_CHARGING_CAPACITY_AH),
        cycle_lf.columns.resolve(BDF.CYCLE_DISCHARGING_CAPACITY_AH),
    ]
    lf = (
        cycle_lf.lf.with_columns(resolved_exprs)
        .group_by(BDF.CYCLE_COUNT.name, maintain_order=True)
        .agg(
            pl.col(BDF.CUMULATIVE_CAPACITY_AH.name)
            .first()
            .alias("Capacity Throughput / Ah"),
            pl.col(BDF.TEST_TIME_SECOND.name).first().alias(BDF.TEST_TIME_SECOND.name),
            _STAT_SUITE["delta"](pl.col(BDF.CYCLE_CHARGING_CAPACITY_AH.name)).alias(
                "Charge Capacity / Ah"
            ),
            _STAT_SUITE["delta"](pl.col(BDF.CYCLE_DISCHARGING_CAPACITY_AH.name)).alias(
                "Discharge Capacity / Ah"
            ),
        )
        .sort(BDF.CYCLE_COUNT.name)
    )

    coulombic_efficiency_expr = (
        pl.col("Discharge Capacity / Ah") / pl.col("Charge Capacity / Ah").shift()
        if dchg_before_chg
        else pl.col("Discharge Capacity / Ah").shift() / pl.col("Charge Capacity / Ah")
    )
    lf = lf.with_columns(
        (pl.col("Charge Capacity / Ah") / pl.first("Charge Capacity / Ah") * 100).alias(
            "SOH Charge / %",
        ),
        (
            pl.col("Discharge Capacity / Ah")
            / pl.first("Discharge Capacity / Ah")
            * 100
        ).alias("SOH Discharge / %"),
        coulombic_efficiency_expr.alias("Coulombic Efficiency"),
    )
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
