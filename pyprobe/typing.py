"""Module for type hints and type aliases."""
from typing import Any, Union

import cerberus

# if TYPE_CHECKING:
from pyprobe.filters import Cycle, Experiment, Procedure, Step
from pyprobe.rawdata import RawData
from pyprobe.result import Result

FilterToExperimentType = Union[Procedure, Experiment]
"""Type alias for filtering to an experiment."""
FilterToCycleType = Union[Procedure, Experiment, Cycle]
"""Type alias for filtering to a cycle."""
FilterToStepType = Union[Procedure, Experiment, Cycle, Step]
"""Type alias for filtering to a step."""
PyProBERawDataType = Union[RawData, FilterToStepType]
"""Type alias for raw data in PyProbe."""
PyProBEDataType = Union[PyProBERawDataType, Result]
"""Type alias for data in PyProbe."""


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
