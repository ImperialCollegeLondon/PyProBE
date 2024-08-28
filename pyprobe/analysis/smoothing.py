"""Module containing methods for smoothing noisy experimental data."""

import copy
from typing import Optional

import numpy as np
import polars as pl
from pydantic import BaseModel
from scipy.interpolate import make_smoothing_spline
from scipy.signal import savgol_filter

from pyprobe.analysis.utils import AnalysisValidator
from pyprobe.result import Result


class Smoothing(BaseModel):
    """A class for smoothing noisy experimental data.

    Smoothing methods return result objects containing the same columns as the input
    data, but with the target column smoothed using the specified method.

    Args:
        input_data (Result): The raw data to analyse.
    """

    input_data: Result

    def spline_smoothing(
        self,
        target_column: str,
        smoothing_lambda: Optional[float] = None,
        x: str = "Time [s]",
    ) -> Result:
        """A method for smoothing noisy data using a spline.

        Args:
            target_column (str):
                The name of the target variable to smooth.
            smoothing_lambda (float, optional):
                The smoothing parameter. Default is None.
            x (str, optional):
                The name of the x variable for the spline curve fit.
                Default is "Time [s]".

        Returns:
            Result:
                A result object containing the data from input data with the target
                column smoothed using a spline, and the gradient of the smoothed data
                with respect to the x variable.
        """
        # validate and identify variables
        validator = AnalysisValidator(
            input_data=self.input_data, required_columns=[x, target_column]
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

        result = copy.deepcopy(self.input_data)
        smoothed_data_column = pl.Series(target_column, smoothed_y)
        result.base_dataframe = result.base_dataframe.with_columns(
            smoothed_data_column.alias(target_column)
        )

        gradient_column_name = f"d({target_column})/d({x})"
        dydx_column = pl.Series(gradient_column_name, smoothed_dydx)
        result.base_dataframe = result.base_dataframe.with_columns(
            dydx_column.alias(gradient_column_name)
        )
        result.define_column(
            f"d({target_column})/d({x})",
            "The gradient of the smoothed data.",
        )
        return result

    def level_smoothing(
        self,
        target_column: str,
        interval: float,
        monotonic: Optional[bool] = False,
    ) -> Result:
        """Smooth noisy data by resampling to a specified interval.

        Args:
            target_column (str):
                The name of the target variable to smooth.
            interval (float):
                The desired minimum interval between points.
            monotonic (bool, optional):
                If True, the target_column is assumed to be monotonic. Default is False.

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

        if monotonic:
            if x[0] > x[-1]:
                x = np.flip(x)

        last_x = x[0]  # Start with the first point
        x_resampled = [last_x]  # Add the first point to the resampled list

        # Use a while loop to find the next point that meets the interval condition
        i = 1
        while i < len(x):
            # Find the next index where the difference meets the interval condition
            if monotonic:  # if monotinic, only consider positive differences
                comparison = x[i:] - last_x
            else:  # if not monotonic, consider all differences
                comparison = abs(x[i:] - last_x)

            next_indices = np.where(comparison >= interval)[0]
            if len(next_indices) == 0:
                break
            next_index = next_indices[0] + i
            x_resampled.append(x[next_index])
            last_x = x[next_index]
            i = next_index + 1

        # Filter the dataframe to only include the resampled points
        dataframe = self.input_data.base_dataframe.filter(
            [
                pl.col(target_column).is_in(x_resampled),
                pl.col(target_column).is_first_distinct(),
            ]
        )

        # Create a new result object with the resampled data
        result = self.input_data
        result.base_dataframe = dataframe
        return result

    def savgol_smoothing(
        self,
        target_column: str,
        window_length: int,
        polyorder: int,
        derivative: int = 0,
    ) -> Result:
        """Smooth noisy data using a Savitzky-Golay filter.

        Args:
            target_column (str):
                The name of the target variable to smooth.
            window_length (int):
                The length of the filter window. Must be a positive odd integer.
            polynomial_order (int):
                The order of the polynomial used to fit the samples.
            derivative (int, optional):
                The order of the derivative to compute. Default is 0.

        Returns:
            Result:
                A result object containing all of the columns of input_data smoothed
                using the Savitzky-Golay filter.
        """
        # validate and identify variables
        validator = AnalysisValidator(
            input_data=self.input_data, required_columns=[target_column]
        )
        x = validator.variables
        smoothed_y = savgol_filter(
            x=x, window_length=window_length, polyorder=polyorder, deriv=derivative
        )

        smoothed_data_column = pl.Series(target_column, smoothed_y)
        result = copy.deepcopy(self.input_data)
        result.base_dataframe = result.base_dataframe.with_columns(
            smoothed_data_column.alias(target_column)
        )
        return result
