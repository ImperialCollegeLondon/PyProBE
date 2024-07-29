"""Module containing methods for smoothing noisy experimental data."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.interpolate import make_smoothing_spline

from pyprobe.result import Result


@dataclass(kw_only=True)
class Smoothing:
    """A class for smoothing noisy experimental data.

    Args:
        rawdata (Result): The raw data to analyse.
    """

    rawdata: Result

    def spline_smoothing(
        self, x: str, y: str, smoothing_lambda: Optional[float] = None
    ) -> Result:
        """A method for smoothing noisy data using a spline.

        Args:
            x (str):
                The name of the x variable.
            y (str):
                The name of the y variable.
            smoothing_lambda (float, optional):
                The smoothing parameter. Default is None.

        Returns:
            Result:
                A result object containing the columns, `x`, the smoothed `y` and the
                gradient of the smoothed `y` with respect to `x`.
        """
        x_data = self.rawdata.get_only(x)
        y_data = self.rawdata.get_only(y)
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

        derivative = y_spline.derivative()
        smoothed_dydx = derivative(x_data)

        smoothing_result = self.rawdata.clean_copy(
            {x: x_data, y: smoothed_y, f"d({y})/d({x})": smoothed_dydx}
        )
        smoothing_result.column_definitions = {
            x: self.rawdata.column_definitions[x],
            y: self.rawdata.column_definitions[y],
            f"d({y})/d({x})": "The gradient of the smoothed data.",
        }
        return smoothing_result
