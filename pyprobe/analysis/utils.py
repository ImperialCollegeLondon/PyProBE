"""Module for utilities for analysis classes."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, model_validator

from pyprobe.pyprobe_types import PyProBEDataType
from pyprobe.result import Result


def assemble_array(input_data: list[Result], name: str) -> NDArray[Any]:
    """Assemble an array from a list of results.

    Args:
        input_data (List[Result]): A list of results.
        name (str): The name of the variable.

    Returns:
        NDArray: The assembled array.
    """
    return np.vstack([input_item.get(name) for input_item in input_data])


class AnalysisValidator(BaseModel):
    """A base class for analysis classes."""

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    input_data: PyProBEDataType
    """The input data to an analysis class."""
    required_columns: list[str]
    """The columns required to conduct the analysis."""

    @model_validator(mode="after")
    def validate_required_columns(self) -> "AnalysisValidator":
        """Check if the required columns are present in the input_data.

        Returns:
            AnalysisValidator: The validated instance.

        Raises:
            ValueError: If any of the required columns are missing.
        """
        self.input_data.check_columns(list(self.required_columns))
        return self

    @property
    def variables(self) -> tuple[NDArray[np.float64], ...]:
        """Return the required columns in the input data as NDArrays.

        Returns:
            Tuple[NDArray[np.float64], ...]: The required columns as NDArrays.
        """
        return self.input_data.get(*self.required_columns)
