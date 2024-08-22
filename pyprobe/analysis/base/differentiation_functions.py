"""Module containing functions for differentiating experimental data."""

from typing import List, Tuple

import numpy as np
import scipy.interpolate as interp
from numpy.typing import NDArray


def get_x_sections(x: NDArray[np.float64]) -> List[slice]:
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
    if len(dx_changes) == 0:
        x_sections.append(slice(0, len(x)))
    else:
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


def get_dx(x: NDArray[np.float64]) -> float:
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


def get_dy_and_counts(
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
    # ensure sign of dy matches direction of input data
    if y[0] < y[-1]:
        dy = y_bins[1] - y_bins[0]
    else:
        dy = -y_bins[1] + y_bins[0]
    N, _ = np.histogram(y, bins=y_bins)
    y_midpoints = y_bins[:-1] + np.diff(y_bins) / 2
    return dy, y_midpoints, N


def y_sampling_interval(y: NDArray[np.float64]) -> float:
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


def calc_gradient_with_LEAN(
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
    dx = get_dx(x)
    dy = k * y_sampling_interval(y)
    dy, y_midpoints, N = get_dy_and_counts(y, dy)
    dxdy = N * dx / dy

    if gradient == "dydx":
        grad = np.divide(1, dxdy, where=dxdy != 0)
    else:
        grad = dxdy
    f = interp.interp1d(y, x, assume_sorted=False)
    x_pts = f(y_midpoints)
    return x_pts, y_midpoints, grad


def smooth_gradient(
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
