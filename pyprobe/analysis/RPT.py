"""A module to streamline processing of identical reference performance tests."""

from typing import Any, List

import numpy as np
import polars as pl
from pydantic import BaseModel

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

    @property
    def rpt_summary(self) -> Result:
        """Summarize the RPTs.

        Returns:
            Result: A result object for the RPT summary.
        """
        return self._rpt_summary

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
