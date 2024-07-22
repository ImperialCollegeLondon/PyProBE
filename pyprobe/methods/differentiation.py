"""A module for differentiating experimental data."""

from typing import Any, List, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from pyprobe.methods.basemethod import BaseMethod
from pyprobe.result import Result


def differentiate_FD(
    rawdata: Result,
    x: str,
    y: str,
    gradient: str = "dydx",
) -> Result:
    """Differentiate smooth data with the finite difference method.

    A light wrapper of the numpy.gradient function.

    Args:
        rawdata (Result): The input data to the method
        x (str): The name of the x variable.
        y (str): The name of the y variable.
        gradient (str, optional):
            The gradient to calculate, either 'dydx' or 'dxdy'.
            Defaults to "dydx".

    Returns:
        Result:
        A result object containing the columns, `x`, `y` and the calculated gradient.
    """
    method = BaseMethod(rawdata)
    x_data = method.variable(x)
    y_data = method.variable(y)
    gradient = gradient
    if gradient == "dydx":
        gradient_title = f"d({y})/d({x})"
        gradient_data = np.gradient(y_data, x_data)
    elif gradient == "dxdy":
        gradient_title = f"d({x})/d({y})"
        gradient_data = np.gradient(x_data, y_data)
    else:
        raise ValueError("Gradient must be either 'dydx' or 'dxdy'.")

    gradient_result = method.make_result(
        {x: x_data, y: y_data, gradient_title: gradient_data}
    )
    gradient_result.column_definitions = {
        x: rawdata.column_definitions[x],
        y: rawdata.column_definitions[y],
        gradient_title: "The calculated gradient.",
    }
    return gradient_result


def differentiate_LEAN(
    rawdata: Result,
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

    This method assumes :math:`x` datapoints to be evenly spaced, it can return either
    :math:`\frac{dy}{dx}` or :math:`\frac{dx}{dy}` depending on the argument provided
    to the `gradient` parameter.

    Args:
        rawdata (Result):
            The input data to the method.
        x (str):
            The name of the x variable.
        y (str):
            The name of the y variable.
        k (int, optional):
            The integer multiple to apply to the sampling interval for the bin size
            (:math:`\delta R` in paper). Default is 1.
        gradient (str, optional):
            The gradient to calculate, either 'dydx' or 'dxdy'. Default is 'dydx'.
        smoothing_filter (List[float], optional):
            The coefficients of the smoothing matrix.

            Examples provided by :footcite:t:`Feng2020` include:
                - [0.25, 0.5, 0.25] for a 3-point smoothing filter.
                - [0.0668, 0.2417, 0.3830, 0.2417, 0.0668] (default) for a 5-point
                  smoothing filter.
                - [0.1059, 0.121, 0.1745, 0.1972, 0.1745, 0.121, 0.1059] for a 7-point
                  smoothing filter.
        section (str, optional):
            The section of the data with constant sample rate in x to be considered.
            Default is 'longest', which just returns the longest unifomly sampled
            section. Alternative is 'all', which returns all sections.

    Returns:
        Result:
            A result object containing the columns, `x`, `y` and the calculated
            gradient.
    """
    method = BaseMethod(rawdata)
    # identify variables
    x_data = method.variable(x)
    y_data = method.variable(y)
    k = k

    # split input data into uniformly sampled sections
    x_sections = _get_x_sections(x_data)
    if section == "longest":
        x_sections = [max(x_sections, key=lambda x: x.stop - x.start)]
    x_all = np.array([])
    y_all = np.array([])
    calc_gradient_all = np.array([])

    # over each uniformly sampled section, calculate the gradient
    for i in range(len(x_sections)):
        x_data = x_data[x_sections[i]]
        y_data = y_data[x_sections[i]]
        x_pts, y_pts, calculated_gradient = _calc_gradient_with_LEAN(
            x_data, y_data, k, gradient
        )
        x_all = np.append(x_all, x_pts)
        y_all = np.append(y_all, y_pts)
        calc_gradient_all = np.append(calc_gradient_all, calculated_gradient)

    # smooth the calculated gradient
    smoothed_gradient = _smooth_gradient(calc_gradient_all, smoothing_filter)

    # output the results
    gradient_title = f"d({y})/d({x})" if gradient == "dydx" else f"d({x})/d({y})"
    gradient_result = method.make_result(
        {x: x_all, y: y_all, gradient_title: smoothed_gradient}
    )
    gradient_result.column_definitions = {
        x: rawdata.column_definitions[x],
        y: rawdata.column_definitions[y],
        gradient_title: "The calculated gradient.",
    }
    return gradient_result


def _get_x_sections(x: NDArray[np.float64]) -> List[slice]:
    """Split the x data into uniformly sampled sections.

    Args:
        x (NDArray[np.float64]): The x values.

    Returns:
        List[slice]: A list of slices representing the uniformly sampled sections.
    """
    dx = np.diff(x)
    ddx = np.diff(dx)
    # find where ddx is above a threshold
    dx_changes = np.argwhere(abs(ddx) > 0.05 * abs(dx[1:])).reshape(-1) + 2
    x_sections = []
    for i in range(len(dx_changes) + 1):
        if i == 0:
            x_sections.append(slice(0, dx_changes[i]))
        elif i == len(dx_changes):
            x_sections.append(slice(dx_changes[i - 1] - 1, len(x)))
        else:
            x_sections.append(slice(dx_changes[i - 1] - 1, dx_changes[i]))
    # only consider sections where there are more than 5 data points
    x_sections = [s for s in x_sections if (s.stop - s.start) >= 5]
    return x_sections


def _get_dx(x: NDArray[np.float64]) -> float:
    """Get the x sampling interval, assuming uniformly spaced x values.

    Args:
        x (NDArray[np.float64]): The x values.

    Returns:
        float: The x sampling interval.

    Raises:
        ValueError: If the x values are not uniformly spaced.
    """
    dx_all = np.diff(x)
    dx_mean = np.median(dx_all)
    dx_iqr = np.percentile(dx_all, 75) - np.percentile(dx_all, 25)
    if abs(dx_iqr) > 0.1 * abs(dx_mean):
        raise ValueError("x values are not uniformly spaced.")
    else:
        return dx_mean


def _get_dy_and_counts(
    y: NDArray[np.float64], dy: float
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Get the y sampling interval, bin midpoints and counts.

    Args:
        y (NDArray[np.float64]): The y values.
        dy (float): The bin size.

    Returns:
        Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
            The y sampling interval, bin midpoints and counts.
    """
    y_range = y.max() - y.min()
    y_bins = np.linspace(y.min(), y.max(), int(np.ceil(y_range / dy)))
    dy = y_bins[1] - y_bins[0]
    N, _ = np.histogram(y, bins=y_bins)
    y_midpoints = y_bins[:-1] + np.diff(y_bins) / 2
    return dy, y_midpoints, N


def _y_sampling_interval(y: NDArray[np.float64]) -> float:
    r"""Get the y sampling interval, :math:`\delta R` in :footcite:t:`Feng2020`.

    Args:
        y (NDArray[np.float64]): The y values.

    Returns:
        float: The y sampling interval.
    """
    y_unique = np.unique(y)
    y_sorted = np.sort(y_unique)
    y_diff = np.diff(y_sorted)
    return np.min(y_diff)


def _calc_gradient_with_LEAN(
    x: NDArray[np.float64], y: NDArray[np.float64], k: int, gradient: str
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Calculate the gradient of the data, assuming x is uniformly spaced.

    Args:
        x (NDArray[np.float64]): The x values.
        y (NDArray[np.float64]): The y values.
        k (int):
            The integer multiple to apply to the sampling interval for the bin size
            (:math:`\delta R` in paper).
        gradient (str): The gradient to calculate, either 'dydx' or 'dxdy'.

    Returns:
        Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
            The x values, the y midpoints and the calculated gradient.
    """
    dx = _get_dx(x)
    dy = k * _y_sampling_interval(y)
    dy, y_midpoints, N = _get_dy_and_counts(y, dy)
    dxdy = N * dx / dy

    if gradient == "dydx":
        grad = np.divide(1, dxdy, where=dxdy != 0)
    else:
        grad = dxdy
    f = interp1d(y, x, assume_sorted=False)
    x_pts = f(y_midpoints)
    return x_pts, y_midpoints, grad


def _smooth_gradient(
    gradient: NDArray[np.float64], alpha: List[float]
) -> NDArray[np.float64]:
    """Smooth the calculated gradient.

    Args:
        gradient (NDArray[np.float64]): The gradient vector.
        alpha (list[float]): The smoothing coefficients.

    Returns:
        NDArray[np.float64]: The smoothed gradient vector.
    """
    A = np.zeros((len(gradient), len(gradient)))
    w = np.floor(len(alpha) / 2)
    for n in range(len(alpha)):
        k = n - w
        vector = np.ones(int(len(gradient) - abs(k)))
        diag = np.diag(vector, int(k))
        A += alpha[n] * diag
    return A @ gradient


method_dict = {"LEAN": differentiate_LEAN}


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
    result = method_dict[method](input_data, x, y, *args, **kwargs)
    return cast(Result, result)
