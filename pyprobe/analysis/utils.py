"""Module for the BaseMethod class."""
from typing import Any, Callable, List

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from pyprobe.result import Result


def analysismethod(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to mark a method as an analysis method."""
    setattr(func, "is_analysis_method", True)
    return func


class BaseAnalysis(BaseModel):
    """Base class for analysis methods."""

    @staticmethod
    def assemble_array(input_data: List[Result], name: str) -> NDArray[Any]:
        """Assemble an array from a list of results.

        Args:
            input_data (List[Result]): A list of results.
            name (str): The name of the variable.

        Returns:
            NDArray: The assembled array.
        """
        return np.vstack([input.get_only(name) for input in input_data])
