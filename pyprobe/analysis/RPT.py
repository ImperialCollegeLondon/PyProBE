"""A module for processing separate RPTs."""


import copy
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
        self._summary_df = copy.deepcopy(self._base_summary_df)

    def process_cell_capacity(self, filter: str, name: str = "Capacity [Ah]") -> Result:
        """Calculate the capacity for a particular experiment step across the RPTs.

        Args:
            filter (str): The filter to apply to the data.
            name (str): The name of the column to store the capacity.

        Returns:
            Result: A result object for the cell capacity in each RPT.
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
        capacities_result = self.input_data[0].clean_copy(
            self._base_summary_df.hstack(capacity_df)
        )
        self._summary_df = self._summary_df.hstack(capacity_df)
        capacities_result.column_definitions = {
            "RPT Number": "The RPT number.",
            name: "The cell capacity.",
        }
        return capacities_result
