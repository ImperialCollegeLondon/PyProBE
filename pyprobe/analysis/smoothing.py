"""Module containing methods for smoothing noisy experimental data."""

from typing import Optional

import numpy as np
import polars as pl
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

    def level_smoothing(
        self,
        target_column: str,
        interval: float,
    ) -> Result:
        """Smooth noisy data by resampling to a specified interval.

        Args:
            target_column (str):
                The name of the target variable to smooth.
            interval (float):
                The desired minimum interval between points.

        Returns:
            Result:
                A result object containing all of the columns of input_data resampled
                to the specified interval for the target column.
        """
        # validate and identify variables
        validator = AnalysisValidator(
            input_data=self.input_data, required_columns=[target_column]
        )
        x = validator.variables

        last_x = x[0]  # Start with the first point
        x_resampled = [last_x]  # Add the first point to the resampled list

        # Use a while loop to find the next point that meets the interval condition
        i = 1
        while i < len(x):
            # Find the next index where the difference meets the interval condition
            next_indices = np.where(abs(x[i:] - last_x) >= interval)[0]
            if len(next_indices) == 0:
                break
            next_index = next_indices[0] + i
            x_resampled.append(x[next_index])
            last_x = x[next_index]
            i = next_index + 1

        # Filter the dataframe to only include the resampled points
        dataframe = self.input_data.base_dataframe.filter(
            pl.col(target_column).is_in(x_resampled)
        )

        # Create a new result object with the resampled data
        result = self.input_data
        result.base_dataframe = dataframe
        return result
