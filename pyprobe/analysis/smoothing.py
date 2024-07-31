"""Module containing methods for smoothing noisy experimental data."""

from typing import Optional

import numpy as np
from pydantic import BaseModel
from scipy.interpolate import make_smoothing_spline

from pyprobe.analysis.utils import AnalysisValidator
from pyprobe.result import Result


class Smoothing(BaseModel):
    """A class for smoothing noisy experimental data.

    Args:
        input_data (Result): The raw data to analyse.
    """

    input_data: Result

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
        # validate and identify variables
        validator = AnalysisValidator(
            input_data=self.input_data, required_columns=[x, y]
        )
        x_data, y_data = validator.variables

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

        smoothing_result = self.input_data.clean_copy(
            {x: x_data, y: smoothed_y, f"d({y})/d({x})": smoothed_dydx}
        )
        smoothing_result.column_definitions = {
            x: self.input_data.column_definitions[x],
            y: self.input_data.column_definitions[y],
            f"d({y})/d({x})": "The gradient of the smoothed data.",
        }
        return smoothing_result
