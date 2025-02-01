"""A module for differentiating experimental data."""

from typing import List

import numpy as np
import polars as pl
from deprecated import deprecated
from pydantic import BaseModel, validate_call

import pyprobe.analysis.base.differentiation_functions as diff_functions
from pyprobe.analysis.utils import AnalysisValidator
from pyprobe.pyprobe_types import PyProBEDataType
from pyprobe.result import Result


@validate_call
def gradient(  # 1. Define the method
    input_data: PyProBEDataType,
    x: str,
    y: str,
) -> Result:
    """Differentiate smooth data with a finite difference method.

    A wrapper of the numpy.gradient function. This method calculates the gradient
    of the data in the y column with respect to the data in the x column.

    Args:
        input_data:
            The input data PyProBE object for the differentiation
        x: The name of the x variable.
        y: The name of the y variable.

    Returns:
        A result object containing the columns, `x`, `y` and the
        calculated gradient.
    """
    # 2. Validate the inputs to the method
    validator = AnalysisValidator(
        input_data=input_data,
        required_columns=[x, y],
        # required_type not neccessary here as type specified when declaring
        # input_data attribute is strict enough
    )
    # 3. Retrieve the validated columns as numpy arrays
    x_data, y_data = validator.variables

    # 4. Perform the computation
    gradient_title = f"d({y})/d({x})"
    gradient_data = np.gradient(y_data, x_data)

    # 5. Create a Result object to store the results
    gradient_result = input_data.clean_copy(
        pl.DataFrame({x: x_data, y: y_data, gradient_title: gradient_data})
    )
    # 6. Define the column definitions for the Result object
    gradient_result.column_definitions = input_data.column_definitions
    gradient_result.column_definitions[gradient_title] = "The calculated gradient."
    # 7. Return the Result object
    return gradient_result


@validate_call
def differentiate_LEAN(
    input_data: PyProBEDataType,
    x: str,
    y: str,
    k: int = 1,
    gradient: str = "dydx",
    smoothing_filter: List[float] = [0.0668, 0.2417, 0.3830, 0.2417, 0.0668],
    section: str = "longest",
) -> Result:
    r"""A method for differentiating noisy data.

    Uses 'Level Evaluation ANalysis' (LEAN) method described in the paper of
    :footcite:t:`Feng2020`.

    This method assumes :math:`x` datapoints to be evenly spaced, it can return
    either :math:`\frac{dy}{dx}` or :math:`\frac{dx}{dy}` depending on the argument
    provided to the `gradient` parameter.

    Args:
        input_data:
            The input data PyProBE object for the differentiation.
        x:
            The name of the x variable.
        y:
            The name of the y variable.
        k:
            The integer multiple to apply to the sampling interval for the bin size
            (:math:`\delta R` in paper). Default is 1.
        gradient:
            The gradient to calculate, either 'dydx' or 'dxdy'. Default is 'dydx'.
        smoothing_filter:
            The coefficients of the smoothing matrix.

            Examples provided by :footcite:t:`Feng2020` include:

                - [0.25, 0.5, 0.25] for a 3-point smoothing filter.
                - [0.0668, 0.2417, 0.3830, 0.2417, 0.0668] (default) for a 5-point
                    smoothing filter.
                - [0.1059, 0.121, 0.1745, 0.1972, 0.1745, 0.121, 0.1059] for a
                    7-point smoothing filter.

        section:
            The section of the data with constant sample rate in x to be considered.
            Default is 'longest', which just returns the longest unifomly sampled
            section. Alternative is 'all', which returns all sections.

    Returns:
        Result:
            A result object containing the columns, `x`, `y` and the calculated
            gradient.
    """
    # validate and identify variables
    validator = AnalysisValidator(input_data=input_data, required_columns=[x, y])
    x_data, y_data = validator.variables

    # split input data into uniformly sampled sections
    x_sections = diff_functions.get_x_sections(x_data)
    if section == "longest":
        x_sections = [max(x_sections, key=lambda x: x.stop - x.start)]
    x_all = np.array([])
    y_all = np.array([])
    calc_gradient_all = np.array([])

    # over each uniformly sampled section, calculate the gradient
    for i in range(len(x_sections)):
        x_data = x_data[x_sections[i]]
        y_data = y_data[x_sections[i]]
        x_pts, y_pts, calculated_gradient = diff_functions.calc_gradient_with_LEAN(
            x_data, y_data, k, gradient
        )
        x_all = np.append(x_all, x_pts)
        y_all = np.append(y_all, y_pts)
        calc_gradient_all = np.append(calc_gradient_all, calculated_gradient)

    # smooth the calculated gradient
    smoothed_gradient = diff_functions.smooth_gradient(
        calc_gradient_all, smoothing_filter
    )

    # output the results
    gradient_title = f"d({y})/d({x})" if gradient == "dydx" else f"d({x})/d({y})"
    gradient_result = input_data.clean_copy(
        pl.DataFrame({x: x_all, y: y_all, gradient_title: smoothed_gradient})
    )
    gradient_result.column_definitions = input_data.column_definitions
    gradient_result.column_definitions[gradient_title] = "The calculated gradient."
    return gradient_result
