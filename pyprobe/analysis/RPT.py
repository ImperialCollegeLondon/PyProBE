"""A module to streamline processing of identical reference performance tests."""

from typing import Any, List, Optional

import numpy as np
import polars as pl
from pydantic import BaseModel

from pyprobe.analysis import pulsing
from pyprobe.analysis.utils import AnalysisValidator
from pyprobe.filters import Procedure
from pyprobe.result import Result


class RPT(BaseModel):
    """A class for processing RPTs."""

    input_data: List[Procedure]
    """A list of RPT Experiment objects."""

    def model_post_init(self, __context: Any) -> None:
        """Initialize the model."""
        self._base_summary_df = pl.DataFrame(
            {
                "RPT Number": list(range(len(self.input_data))),
            }
        )
        self._rpt_summary = self.input_data[0].clean_copy(self._base_summary_df)
        self._rpt_summary.column_definitions = {
            "RPT Number": "The RPT number.",
        }
        self._process_RPT_start_date()

    @property
    def rpt_summary(self) -> Result:
        """Summarize the RPTs.

        Returns:
            Result: A result object for the RPT summary.
        """
        return self._rpt_summary

    def _process_RPT_start_date(self) -> None:
        """Add the start date of the RPT to the summary."""
        start_dates = []
        for rpt in self.input_data:
            date = rpt.base_dataframe.select("Date").head(1)
            start_dates.append(date)
        start_date_df = pl.concat(start_dates)
        if isinstance(start_date_df, pl.LazyFrame):
            start_date_df = start_date_df.collect()
        self._rpt_summary.base_dataframe = self._rpt_summary.base_dataframe.hstack(
            start_date_df.rename({"Date": "RPT Start Date"})
        )
        print(self._rpt_summary.base_dataframe)
        column_definition = {"RPT Start Date": "The RPT start date."}
        self._rpt_summary.column_definitions.update(column_definition)

    def process_cell_capacity(self, filter: str, name: str = "Capacity [Ah]") -> None:
        """Calculate the capacity for a particular experiment step across the RPTs.

        Results are stored in the :property:`rpt_summary` attribute.

        Args:
            filter (str): The filter to apply to the data.
            name (str): The name of the column to store the capacity.
        """
        all_capacities = np.zeros(len(self.input_data))
        for rpt_number, experiment in enumerate(self.input_data):
            AnalysisValidator(input_data=experiment, required_columns=["Capacity [Ah]"])
            reference_step = eval(f"experiment.{filter}")
            all_capacities[rpt_number] = reference_step.capacity

        capacity_df = pl.DataFrame(
            {
                name: all_capacities,
            }
        )
        self._rpt_summary.base_dataframe = self._rpt_summary.base_dataframe.hstack(
            capacity_df
        )
        column_definition = {name: "The cell capacity."}
        self._rpt_summary.column_definitions.update(column_definition)

    def process_soh(self, filter: str, name: str = "SOH") -> None:
        """Calculate the SOH for a particular experiment step across the RPTs.

        Results are stored in the :property:`rpt_summary` attribute.

        Args:
            filter (str): The filter to apply to the data.
            name (str): The name of the column to store the SOH.
        """
        all_soh = np.zeros(len(self.input_data))
        for rpt_number, experiment in enumerate(self.input_data):
            AnalysisValidator(input_data=experiment, required_columns=["Capacity [Ah]"])
            reference_step = eval(f"experiment.{filter}")
            all_soh[rpt_number] = reference_step.capacity
        all_soh = all_soh / all_soh[0]

        soh_df = pl.DataFrame(
            {
                name: all_soh,
            }
        )
        self._rpt_summary.base_dataframe = self._rpt_summary.base_dataframe.hstack(
            soh_df
        )
        column_definition = {name: "The cell SOH."}
        self._rpt_summary.column_definitions.update(column_definition)

    def process_pulse_resistance(
        self,
        filter: str,
        eval_time: float = 0.0,
        pulse_number: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        """Calculate the pulse resistance from a pulsing experiment across the RPTs.

        Results are stored in the :property:`rpt_summary` attribute.

        Args:
            filter: The filter to apply to the data to extract the pulse data.
            eval_time:
                The time at which to evaluate the resistance. Default is 0.0. If 0.0,
                the resistance using the first pulse datapoint is calculated (R0).
            pulse_number:
                The pulse number to evaluate the resistance. Default is None. If None,
                the resistance for all pulses is calculated.
            name: The name of the column to store the pulse resistance.
        """
        if eval_time == 0.0:
            resistance_col_name = "R0 [Ohms]"
        else:
            resistance_col_name = f"R_{eval_time}s [Ohms]"
        if name is None:
            name = resistance_col_name

        all_resistances = []
        for experiment in self.input_data:
            pulse_data = eval(f"experiment.{filter}")
            if eval_time == 0.0:
                resistance_result = pulsing.get_resistances(pulse_data)
            else:
                resistance_result = pulsing.get_resistances(pulse_data, [eval_time])
            if pulse_number is not None:
                resistance_df = resistance_result.data.filter(
                    pl.col("Pulse Number") == pulse_number
                )
                resistance_value = resistance_df[resistance_col_name].to_numpy()[0]
            else:
                resistance_value = resistance_result.data[
                    resistance_col_name
                ].to_numpy()
            all_resistances.append(resistance_value)

        resistance_df = pl.DataFrame(
            {
                name: all_resistances,
            }
        )
        self._rpt_summary.base_dataframe = self._rpt_summary.base_dataframe.hstack(
            resistance_df
        )
        column_definition = {name: "The cell pulse resistance."}
        self._rpt_summary.column_definitions.update(column_definition)
