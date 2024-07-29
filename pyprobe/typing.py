"""Module for type hints and type aliases."""
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from pyprobe.filters import Cycle, Experiment, Procedure, Step
    from pyprobe.rawdata import RawData
    from pyprobe.result import Result

FilterToExperimentType = Union["Procedure", "Experiment"]
"""Type alias for filtering to an experiment."""
FilterToCycleType = Union["Procedure", "Experiment", "Cycle"]
"""Type alias for filtering to a cycle."""
FilterToStepType = Union["Procedure", "Experiment", "Cycle", "Step"]
"""Type alias for filtering to a step."""
PyProBERawDataType = Union["RawData", FilterToStepType]
"""Type alias for raw data in PyProbe."""
PyProBEDataType = Union[PyProBERawDataType, "Result"]
"""Type alias for data in PyProbe."""
