"""Module for utilities for analysis classes."""
from typing import Any, List

import numpy as np
from numpy.typing import NDArray

from pyprobe.result import Result


def assemble_array(input_data: List[Result], name: str) -> NDArray[Any]:
    """Assemble an array from a list of results.

    Args:
        input_data (List[Result]): A list of results.
        name (str): The name of the variable.

    Returns:
        NDArray: The assembled array.
    """
    return np.vstack([input.get_only(name) for input in input_data])
