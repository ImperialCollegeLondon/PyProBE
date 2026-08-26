"""A module for the Pulsing class."""

import polars as pl

from pyprobe.analysis.utils import build_result, validate_columns
from pyprobe.columns import BDF
from pyprobe.filters import Experiment, Step
from pyprobe.pyprobe_types import PyProBEDataType
from pyprobe.result import Table


def _get_pulse_number(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """Return the pulse number for each row in the input data.

    Args:
        data: The input data.

    Returns:
        The input data with a new column "Pulse Number".
    """
    return data.with_columns(
        (
            (pl.col(BDF.CURRENT_AMPERE.name).shift() == 0)
            & (pl.col(BDF.CURRENT_AMPERE.name) != 0)
        )
        .cum_sum()
        .alias("Pulse Number"),
    )


def _get_end_of_rest_points(
    data: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame | pl.LazyFrame:
    """Return the last OCV point and timestamp before each pulse.

    Args:
        data: The input data.

    Returns:
        The input data with new columns "OCV / V" and "Start Time / s".
    """
    if "Pulse Number" not in data.columns:
        data = _get_pulse_number(data)
    return (
        data.filter(pl.col(BDF.CURRENT_AMPERE.name) == 0)
        .group_by("Pulse Number")
        .last()
        .with_columns(pl.col("Pulse Number") + 1)
        .sort("Pulse Number")
    )


def get_ocv_curve(input_data: PyProBEDataType) -> Table:
    """Filter down a pulsing experiment to the points representing the cell OCV.

    Args:
        input_data: The input data for the pulsing experiment.

    Returns:
        A new Table object containing the OCV curve.

    Raises:
        ColumnResolutionError: If required columns cannot be resolved from `input_data`.
    """
    validate_columns(
        input_data,
        BDF.CURRENT_AMPERE,
        BDF.VOLTAGE_VOLT,
        BDF.TEST_TIME_SECOND,
        "SOC / %",
    )

    all_data_df = input_data.lf
    ocv_df = _get_end_of_rest_points(all_data_df).drop("Pulse Number")
    return build_result(input_data, ocv_df)


def get_resistances(
    input_data: PyProBEDataType,
    r_times: list[float | int] = [],
) -> Table:
    """Returns a result object summarising the pulsing experiment.

    Args:
        input_data:
            The input data for the pulsing experiment. Must contain the columns:
            - Current / A
            - Voltage / V
            - Time / s
            - Event
            - SOC
        r_times:
            A list of times (in seconds) after each pulse at which to evaluate
            the cell resistance.

    Returns:
        Table:
            A result object containing key summary statistics for a pulsing
            experiment. Includes:
            - Experiment Capacity [Ah]
            - SOC
            - OCV / V
            - R0 / Ohm, calculated from the OCV and the first data point in the
            pulse where the current is within 1% of the median pulse current
            - Resistance calculated at each time provided in seconds in the r_times
            argument

    Raises:
        ColumnResolutionError: If required columns cannot be resolved from `input_data`.
    """
    validate_columns(
        input_data,
        BDF.CURRENT_AMPERE,
        BDF.VOLTAGE_VOLT,
        BDF.TEST_TIME_SECOND,
        "SOC / %",
    )
    time_expr = input_data.columns.resolve(BDF.TEST_TIME_SECOND)
    all_data_df = input_data.lf.with_columns(time_expr)

    # get the pulse number for each row
    all_data_df = _get_pulse_number(all_data_df)
    # get the last OCV point and timestamp before each pulse
    ocv = (
        all_data_df.filter(pl.col(BDF.CURRENT_AMPERE.name) == 0)
        .group_by("Pulse Number")
        .agg(
            pl.col(BDF.VOLTAGE_VOLT.name).last().alias("OCV / V"),
            pl.col(BDF.TEST_TIME_SECOND.name).last().alias("Start Time / s"),
        )
        .with_columns(pl.col("Pulse Number") + 1)
    )
    # get the median current for each pulse
    pulse_current = (
        all_data_df.filter(pl.col(BDF.CURRENT_AMPERE.name) != 0)
        .group_by("Pulse Number")
        .agg(pl.col(BDF.CURRENT_AMPERE.name).median().alias("Pulse Current"))
    )
    # recombine the dataframes
    all_data_df = (
        all_data_df.join(ocv, on="Pulse Number", how="left")
        .join(pulse_current, on="Pulse Number", how="left")
        .sort(BDF.TEST_TIME_SECOND.name)
    )
    # get the first point in each pulse where the current is within 1% of the pulse
    # current
    pulse_df = (
        all_data_df.filter(
            (
                pl.col(BDF.CURRENT_AMPERE.name).abs()
                > 0.99 * pl.col("Pulse Current").abs()
            )
            & (
                pl.col(BDF.CURRENT_AMPERE.name).abs()
                < 1.01 * pl.col("Pulse Current").abs()
            ),
        )
        .group_by("Pulse Number")
        .first()
        .sort("Pulse Number")
    )

    # calculate the resistance at the start of the pulse
    r0 = (
        (pl.col(BDF.VOLTAGE_VOLT.name) - pl.col("OCV / V"))
        / pl.col(BDF.CURRENT_AMPERE.name)
    ).alias("R0 / Ohm")
    pulse_df = pulse_df.with_columns(r0)

    t_col_names = [f"t_{time}s / s" for time in r_times]
    r_t_col_names = [f"R_{time}s / Ohm" for time in r_times]
    if t_col_names != []:
        # add columns for the timestamps requested after each pulse
        pulse_df = pulse_df.with_columns(
            [
                (pl.col("Start Time / s") + time).alias(t_col_names[idx])
                for idx, time in enumerate(r_times)
            ],
        )

        # reformat df into two rows, r_time and the corresponding timestamp
        t_after_pulse_df = pulse_df.unpivot(t_col_names).rename(
            {"variable": "r_time", "value": BDF.TEST_TIME_SECOND.name},
        )

        # merge this dataframe into the full dataframe and sort
        t_after_pulse_df = all_data_df.join(
            t_after_pulse_df,
            on=BDF.TEST_TIME_SECOND.name,
            how="full",
            coalesce=True,
        ).sort(BDF.TEST_TIME_SECOND.name)

        # after merging, where the requested time doesn't match with an existing
        # timestamp, null values will be inserted in the Voltage column.
        # Use linear interpolation for voltage.
        t_after_pulse_df = t_after_pulse_df.with_columns(
            [
                pl.col(BDF.VOLTAGE_VOLT.name).interpolate(),
            ],
        )
        # filter the array to return only the inserted rows
        t_after_pulse_df = t_after_pulse_df.filter(
            pl.col("r_time").is_not_null(),
        ).select(BDF.VOLTAGE_VOLT.name, BDF.TEST_TIME_SECOND.name)

        for time in r_times:
            pulse_df = pulse_df.join(
                t_after_pulse_df,
                left_on=f"t_{time}s / s",
                right_on=BDF.TEST_TIME_SECOND.name,
                how="left",
            ).rename({f"{BDF.VOLTAGE_VOLT.name}_right": f"V_{time}s / V"})
            pulse_df = pulse_df.with_columns(
                (pl.col(f"V_{time}s / V") - pl.col("OCV / V"))
                / pl.col(BDF.CURRENT_AMPERE.name),
            ).rename({f"V_{time}s / V": f"R_{time}s / Ohm"})

        # filter the dataframe to the final selection
        pulse_df = pulse_df.select(
            [
                "Pulse Number",
                BDF.NET_CAPACITY_AH.name,
                "SOC / %",
                "OCV / V",
                "R0 / Ohm",
            ]
            + r_t_col_names,
        )
    else:
        pulse_df = pulse_df.select(
            [
                "Pulse Number",
                BDF.NET_CAPACITY_AH.name,
                "SOC / %",
                "OCV / V",
                "R0 / Ohm",
            ],
        )

    column_definitions = {
        "Pulse Number": "An index for each pulse.",
        "Net Capacity": input_data.column_definitions.get(
            "Net Capacity", "The net capacity passed."
        ),
        "SOC / %": input_data.column_definitions.get("SOC / %", "The state of charge."),
        "OCV": "The voltage value at the final data point in the rest before a pulse.",
        "R0": "The instantaneous resistance measured between the final rest "
        "point and the first data point in the pulse.",
    }
    result = build_result(input_data, pulse_df, column_definitions=column_definitions)
    for time in r_times:
        result.define_column(
            f"R_{time}s",
            f"The resistance measured between the OCV and the voltage t = {time}s "
            f"after the pulse.",
        )
    return result


class Pulsing:
    """A pulsing experiment in a battery procedure."""

    def __init__(self, input_data: Experiment) -> None:
        """Initialise the Pulsing experiment.

        Args:
            input_data: The input data for the pulsing experiment.
        """
        self.input_data = input_data

    def pulse(self, pulse_number: int) -> Step:
        """Return a step object for a pulse in the pulsing experiment.

        Args:
            pulse_number (int): The Pulse Number to return.

        Returns:
            pyprobe.filters.Step: A step object for a pulse in the pulsing experiment.
        """
        return self.input_data.cycle(pulse_number).chargeordischarge(0)

    def pulse_rest(self, rest_number: int) -> Step:
        """Return a step object for a rest in the pulsing experiment.

        Args:
            rest_number (int): The rest number to return.

        Returns:
            pyprobe.filters.Step: A step object for a rest in the pulsing experiment.
        """
        return self.input_data.cycle(rest_number).rest(0)
