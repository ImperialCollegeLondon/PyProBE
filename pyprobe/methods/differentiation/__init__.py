"""A module for differentiation methods."""
from typing import Any, cast

from pyprobe.methods.differentiation import LEAN
from pyprobe.result import Result

method_dict = {"LEAN": LEAN.DifferentiateLEAN}


def gradient(
    method: str, input_data: Result, x: str, y: str, *args: Any, **kwargs: Any
) -> Result:
    """Calculate the gradient of the data from a variety of methods.

    Args:
        method (str): The differentiation method.
        input_data (Result): The input data as a Result object.
        x (str): The x data column.
        y (str): The y data column.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Result: The result object from the gradient method.
    """
    result = method_dict[method](input_data, x, y, *args, **kwargs).output_data
    return cast(Result, result)
