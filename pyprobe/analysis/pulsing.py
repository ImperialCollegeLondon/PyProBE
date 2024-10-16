"""A module for the Pulsing class."""

from typing import List

import polars as pl
from pydantic import BaseModel

from pyprobe.analysis.utils import AnalysisValidator
from pyprobe.filters import Experiment, Step
from pyprobe.result import Result


class Pulsing(BaseModel):
    """A pulsing experiment in a battery procedure."""

    input_data: Experiment
    """The input data for the pulsing experiment."""

    def pulse(self, pulse_number: int) -> Step:
        """Return a step object for a pulse in the pulsing experiment.

        Args:
            pulse_number (int): The Pulse Number to return.

        Returns:
            Step: A step object for a pulse in the pulsing experiment.
        """
        return self.input_data.cycle(pulse_number).chargeordischarge(0)

    def pulse_rest(self, rest_number: int) -> Step:
        """Return a step object for a rest in the pulsing experiment.

        Args:
            rest_number (int): The rest number to return.

        Returns:
            Step: A step object for a rest in the pulsing experiment.
        """
        return self.input_data.cycle(rest_number).rest(0)

    def pulse_summary(self, r_times: List[float] = []) -> Result:
        """Returns a result object summarising the pulsing experiment.

        Args:
            r_times:
                A list of times (in seconds) after each pulse at which to evaluate
                the cell resistance.

        Returns:
            Result:
                A result object containing key summary statistics for a pulsing
                experiment. Includes:
                - Experiment Capacity [Ah]
                - SOC
                - OCV [V]
                - R0 [Ohms], calculated from the OCV and the first data point in the
                pulse where the current is within 1% of the median pulse current
                - Resistance calculated at each time provided in seconds in the r_times
                argument
        """
        AnalysisValidator(
            input_data=self.input_data,
            required_columns=["Current [A]", "Voltage [V]", "Time [s]", "Event", "SOC"],
        )
        all_data_df = self.input_data.base_dataframe

        # get the pulse number for each row
        all_data_df = all_data_df.with_columns(
            ((pl.col("Current [A]").shift() == 0) & (pl.col("Current [A]") != 0))
            .cum_sum()
            .alias("Pulse Number")
        )
        # get the last OCV point and timestamp before each pulse
        ocv = (
            all_data_df.filter(pl.col("Current [A]") == 0)
            .group_by("Pulse Number")
            .agg(
                pl.col("Voltage [V]").last().alias("OCV [V]"),
                pl.col("Time [s]").last().alias("Start Time [s]"),
            )
            .with_columns(pl.col("Pulse Number") + 1)
        )
        # get the median current for each pulse
        pulse_current = (
            all_data_df.filter(pl.col("Current [A]") != 0)
            .group_by("Pulse Number")
            .agg(pl.col("Current [A]").median().alias("Pulse Current"))
        )
        # recombine the dataframes
        all_data_df = (
            all_data_df.join(ocv, on="Pulse Number", how="left")
            .join(pulse_current, on="Pulse Number", how="left")
            .sort("Time [s]")
        )
        # get the first point in each pulse where the current is within 1% of the pulse
        # current
        pulse_df = (
            all_data_df.filter(
                (pl.col("Current [A]").abs() > 0.99 * pl.col("Pulse Current").abs())
                & (pl.col("Current [A]").abs() < 1.01 * pl.col("Pulse Current").abs())
            )
            .group_by("Pulse Number")
            .first()
            .sort("Pulse Number")
        )

        # calculate the resistance at the start of the pulse
        R0 = (
            (pl.col("Voltage [V]") - pl.col("OCV [V]")) / pl.col("Current [A]")
        ).alias("R0 [Ohms]")
        pulse_df = pulse_df.with_columns(R0)

        t_col_names = [f"t_{time}s [s]" for time in r_times]
        r_t_col_names = [f"R_{time}s [Ohms]" for time in r_times]
        if t_col_names != []:
            # add columns for the timestamps requested after each pulse
            pulse_df = pulse_df.with_columns(
                [
                    (pl.col("Start Time [s]") + time).alias(t_col_names[idx])
                    for idx, time in enumerate(r_times)
                ]
            )

            # reformat df into two rows, r_time and the corresponding timestamp
            t_after_pulse_df = pulse_df.unpivot(t_col_names).rename(
                {"variable": "r_time", "value": "Time [s]"}
            )

            # merge this dataframe into the full dataframe and sort
            t_after_pulse_df = all_data_df.join(
                t_after_pulse_df, on="Time [s]", how="full", coalesce=True
            ).sort("Time [s]")

            # after merging, where the requested time doesn't match with an existing
            # timestamp, null values will be inserted in the Voltage and Event columns.
            # Use linear interpolation for voltage and just look backward for the event
            # number
            t_after_pulse_df = t_after_pulse_df.with_columns(
                [
                    pl.col("Voltage [V]").interpolate(),
                ]
            )
            # filter the array to return only the inserted rows
            t_after_pulse_df = t_after_pulse_df.filter(
                pl.col("r_time").is_not_null()
            ).select("Voltage [V]", "Time [s]")

            for time in r_times:
                pulse_df = pulse_df.join(
                    t_after_pulse_df,
                    left_on=f"t_{time}s [s]",
                    right_on="Time [s]",
                    how="left",
                ).rename({"Voltage [V]_right": f"V_{time}s [V]"})
                pulse_df = pulse_df.with_columns(
                    (pl.col(f"V_{time}s [V]") - pl.col("OCV [V]"))
                    / pl.col("Current [A]")
                ).rename({f"V_{time}s [V]": f"R_{time}s [Ohms]"})

            # filter the dataframe to the final selection
            pulse_df = pulse_df.select(
                [
                    "Pulse Number",
                    "Experiment Capacity [Ah]",
                    "SOC",
                    "OCV [V]",
                    "R0 [Ohms]",
                ]
                + r_t_col_names
            )
        else:
            pulse_df = pulse_df.select(
                [
                    "Pulse Number",
                    "Experiment Capacity [Ah]",
                    "SOC",
                    "OCV [V]",
                    "R0 [Ohms]",
                ]
            )

        column_definitions = {
            "Pulse Number": "An index for each pulse.",
            "Experiment Capacity [Ah]": self.input_data.column_definitions[
                "Experiment Capacity [Ah]"
            ],
            "SOC": self.input_data.column_definitions["SOC"],
            "OCV [V]": "The voltage value at the final data point in the rest before a "
            "pulse.",
            "R0 [Ohms]": "The instantaneous resistance measured between the final rest "
            "point and the first data point in the pulse.",
        }
        result = self.input_data.clean_copy(pulse_df, column_definitions)
        for time in r_times:
            result.define_column(
                f"R_{time}s",
                f"The resistance measured between the OCV and the voltage t = {time}s "
                f"after the pulse.",
            )
        return result
