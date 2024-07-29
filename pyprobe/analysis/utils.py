"""Module for utilities for analysis classes."""
from typing import Any, List

import cerberus
import numpy as np
from numpy.typing import NDArray

from pyprobe.filters import Cycle, Experiment, Procedure, RawData, Step
from pyprobe.result import Result
from pyprobe.typing import (
    FilterToCycleType,
    FilterToExperimentType,
    FilterToStepType,
    PyProBEDataType,
    PyProBERawDataType,
)


def assemble_array(input_data: List[Result], name: str) -> NDArray[Any]:
    """Assemble an array from a list of results.

    Args:
        input_data (List[Result]): A list of results.
        name (str): The name of the variable.

    Returns:
        NDArray: The assembled array.
    """
    return np.vstack([input.get_only(name) for input in input_data])


class PyProBEValidator(cerberus.Validator):
    """A custom validator for PyProBE data types."""

    types_mapping = cerberus.Validator.types_mapping.copy()
    types_mapping["Experiment"] = cerberus.TypeDefinition(
        "Experiment", (Experiment,), ()
    )
    types_mapping["Procedure"] = cerberus.TypeDefinition("Procedure", (Procedure,), ())
    types_mapping["Cycle"] = cerberus.TypeDefinition("Cycle", (Cycle,), ())
    types_mapping["Step"] = cerberus.TypeDefinition("Step", (Step,), ())
    types_mapping["RawData"] = cerberus.TypeDefinition("RawData", (RawData,), ())
    types_mapping["Result"] = cerberus.TypeDefinition("Result", (Result,), ())
    types_mapping["FilterToExperimentType"] = cerberus.TypeDefinition(
        "FilterToExperimentType", (FilterToExperimentType,), ()
    )
    types_mapping["FilterToCycleType"] = cerberus.TypeDefinition(
        "FilterToCycleType", (FilterToCycleType,), ()
    )
    types_mapping["FilterToStepType"] = cerberus.TypeDefinition(
        "FilterToStepType", (FilterToStepType,), ()
    )
    types_mapping["PyProBERawDataType"] = cerberus.TypeDefinition(
        "PyProBERawDataType", (PyProBERawDataType,), ()
    )
    types_mapping["PyProBEDataType"] = cerberus.TypeDefinition(
        "PyProBEDataType", (PyProBEDataType,), ()
    )

    def _validate_contains_columns(self, columns: Any, field: Any, value: Any) -> None:
        """Validate that the value contains the specified column."""
        if isinstance(columns, str):
            column = [columns]
        for column in columns:
            if column not in value.column_list:
                self._error(field, f"Column '{column}' not in data.")
