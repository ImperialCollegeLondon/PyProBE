"""Module for type hints and type aliases."""
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from pyprobe.filters import Cycle, Experiment, Procedure, Step
    from pyprobe.rawdata import RawData
    from pyprobe.result import Result

FilterToExperimentType = Union["Procedure", "Experiment"]
FilterToCycleType = Union["Procedure", "Experiment", "Cycle"]
FilterToStepType = Union["Procedure", "Experiment", "Cycle", "Step"]

PyProBERawDataType = Union["RawData", FilterToStepType]

PyProBEDataType = Union[PyProBERawDataType, "Result"]
