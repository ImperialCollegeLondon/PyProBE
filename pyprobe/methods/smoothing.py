"""Module containing methods for smoothing noisy experimental data."""

from typing import Optional

import numpy as np
from scipy.interpolate import make_smoothing_spline

from pyprobe.methods.basemethod import BaseMethod
from pyprobe.result import Result


def spline_smoothing(
    rawdata: Result, x: str, y: str, smoothing_lambda: Optional[float] = None
) -> Result:
    """A method for smoothing noisy data using a spline.

    Args:
        rawdata (Result):
            The input data to the method.
        x (str):
            The name of the x variable.
        y (str):
            The name of the y variable.
        smoothing_lambda (float, optional):
            The smoothing parameter. Default is None.

    Returns:
        Result:
            A result object containing the columns, `x` and the smoothed `y`.
    """
    method = BaseMethod(rawdata)
    x_data = method.variable(x)
    y_data = method.variable(y)
    smoothing_lambda = smoothing_lambda
    data_flipped = False
    if x_data[0] > x_data[-1]:  # flip the data if it is not in ascending order
        x_data = np.flip(x_data)
        y_data = np.flip(y_data)
        data_flipped = True

    y_spline = make_smoothing_spline(x_data, y_data, lam=smoothing_lambda)

    smoothed_y = y_spline(x_data)
    if data_flipped:
        x_data = np.flip(x_data)
        smoothed_y = np.flip(smoothed_y)

    return method.make_result({x: x_data, y: smoothed_y})
