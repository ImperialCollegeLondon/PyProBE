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
            pulse_number (int): The pulse number to return.

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
                - R0 [Ohms]
                - Resistance calculated at each time provided in seconds in the r_times
                argument
        """
        AnalysisValidator(
            input_data=self.input_data,
            required_columns=["Current [A]", "Voltage [V]", "Time [s]", "Event", "SOC"],
        )

        all_data_df = self.input_data.data.with_columns(
            [
                pl.col("Voltage [V]").shift().alias("Prev Voltage"),
                pl.col("Voltage [V]").shift(-1).alias("Next Voltage"),
            ]
        )

        starts = (pl.col("Current [A]").shift() == 0) & (pl.col("Current [A]") != 0)
        pulse_df = all_data_df.filter(starts)
        pulse_df = pulse_df.rename({"Prev Voltage": "OCV [V]"})

        R0 = (
            (pl.col("Voltage [V]") - pl.col("OCV [V]")) / pl.col("Current [A]")
        ).alias("R0 [Ohms]")
        pulse_df = pulse_df.with_columns(R0)

        r_t_col_names = [f"R_{time}s [Ohms]" for time in r_times]
        if r_t_col_names != []:
            # add columns for the timestamps requested after each pulse
            t_after_pulse_df = pulse_df.with_columns(
                [
                    (pl.col("Time [s]") + time).alias(r_t_col_names[idx])
                    for idx, time in enumerate(r_times)
                ]
            )

            # reformat df into two rows, r_time and the corresponding timestamp
            t_after_pulse_df = t_after_pulse_df.unpivot(r_t_col_names).rename(
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
                    pl.col("Event").fill_null(strategy="backward"),
                ]
            )

            # filter the array to return only the inserted rows
            t_after_pulse_df = t_after_pulse_df.filter(
                pl.col("r_time").is_not_null()
            ).select("Voltage [V]", "Event", "r_time")

            # pivot the dataframe to obtain seperate rows for the requested time points
            # against Event
            t_after_pulse_df = t_after_pulse_df.pivot("r_time", values="Voltage [V]")

            # merge back into the main pulse dataframe on Event
            pulse_df = pulse_df.join(t_after_pulse_df, on="Event")

            # calculate the resistance
            pulse_df = pulse_df.with_columns(
                [
                    (pl.col(r_time_column) - pl.col("OCV [V]")) / pl.col("Current [A]")
                    for r_time_column in r_t_col_names
                ]
            )

            pulse_df = pulse_df.with_columns(
                pl.col("Event").rank(method="dense").alias("Pulse number")
            )
            # filter the dataframe to the final selection
            pulse_df = pulse_df.select(
                [
                    "Pulse number",
                    "Experiment Capacity [Ah]",
                    "SOC",
                    "OCV [V]",
                    "R0 [Ohms]",
                ]
                + r_t_col_names
            )
        else:
            pulse_df = pulse_df.with_columns(
                pl.col("Event").rank(method="dense").alias("Pulse number")
            )
            pulse_df = pulse_df.select(
                [
                    "Pulse number",
                    "Experiment Capacity [Ah]",
                    "SOC",
                    "OCV [V]",
                    "R0 [Ohms]",
                ]
            )

        column_definitions = {
            "Pulse number": "An index for each pulse.",
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
        for r_t in r_t_col_names:
            time = r_t.split("_")[1][-9]
            result.define_column(
                r_t,
                f"The resistance measured between the OCV and the voltage t = {time}s "
                f"after the pulse.",
            )
        return result
