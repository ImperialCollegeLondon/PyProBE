"""Module for type hints and type aliases."""

from pyprobe.filters import Cycle, Experiment, Procedure, Step
from pyprobe.rawdata import CyclingData
from pyprobe.result import Table

FilterToExperimentType = Procedure | Experiment
"""Type alias for filtering to an experiment."""
FilterToCycleType = Procedure | Experiment | Cycle
"""Type alias for filtering to a cycle."""
FilterToStepType = Procedure | Experiment | Cycle | Step
"""Type alias for filtering to a step."""
PyProBERawDataType = CyclingData | FilterToStepType
"""Type alias for cycling data in PyProbe."""
PyProBEDataType = PyProBERawDataType | Table
"""Type alias for data in PyProbe; aligns with the :class:`Quantified` contract."""
ExperimentOrCycleType = Experiment | Cycle
"""Type alias for an experiment or cycle."""
