"""Module for the BaseMethod class."""
from typing import Any, Callable, List

import numpy as np
from numpy.typing import NDArray

from pyprobe.result import Result


def analysismethod(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to mark a method as an analysis method."""
    setattr(func, "is_analysis_method", True)
    return func


class BaseAnalysis:
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
        return np.vstack([input.get(name) for input in input_data])

    @property
    def analysis_methods(self) -> tuple[str, ...]:
        """Return a tuple of the analysis methods in the class.

        Returns:
            Tuple[str]: The analysis methods in the class.
        """
        methods = []
        for method in dir(self):
            if method != "analysis_methods":
                attr = getattr(self, method)
                if getattr(attr, "is_analysis_method", False):
                    methods.append(method)
        return tuple(methods)
