"""Module for type hints and type aliases."""
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from pyprobe.procedure import Cycle, Experiment, Procedure, RawData

PyProBEFilterType = Union["Procedure", "Experiment", "Cycle"]

PyProBERawDataType = Union["RawData", PyProBEFilterType]
