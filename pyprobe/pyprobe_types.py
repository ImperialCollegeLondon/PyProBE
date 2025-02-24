"""Module for type hints and type aliases."""

from pyprobe.filters import Cycle, Experiment, Procedure, Step
from pyprobe.rawdata import RawData
from pyprobe.result import Result

FilterToExperimentType = Procedure | Experiment
"""Type alias for filtering to an experiment."""
FilterToCycleType = Procedure | Experiment | Cycle
"""Type alias for filtering to a cycle."""
FilterToStepType = Procedure | Experiment | Cycle | Step
"""Type alias for filtering to a step."""
PyProBERawDataType = RawData | FilterToStepType
"""Type alias for raw data in PyProbe."""
PyProBEDataType = PyProBERawDataType | Result
"""Type alias for data in PyProbe."""
ExperimentOrCycleType = Experiment | Cycle
"""Type alias for an experiment or cycle."""
