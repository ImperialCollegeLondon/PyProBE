"""Module for differentiating experimental data with the LEAN method."""
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from pyprobe.methods.basemethod import BaseMethod
from pyprobe.result import Result


class DifferentiateLEAN(BaseMethod):
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

            Examples are provided by :footcite:t:`Feng2020` include:
                - [0.25, 0.5, 0.25] for a 3-point smoothing filter.
                - [0.0668, 0.2417, 0.3830, 0.2417, 0.0668] (default) for a 5-point
                  smoothing filter.
                - [0.1059, 0.121, 0.1745, 0.1972, 0.1745, 0.121, 0.1059] for a 7-point
                  smoothing filter.

    Attributes:
        x_data (NDArray):
            The x values.
        y_data (NDArray):
            The y values.
        k (int):
            The integer multiple to apply to the sampling interval for the bin size
            (:math:`\delta R` in paper).
        output_data (Result):
            A result object containing the columns, `x`, `y` and the calculated
            gradient.
    """

    def __init__(
        self,
        rawdata: Result,
        x: str,
        y: str,
        k: int = 1,
        gradient: str = "dydx",
        smoothing_filter: List[float] = [0.0668, 0.2417, 0.3830, 0.2417, 0.0668],
    ):
        """Initialize the LEAN method."""
        super().__init__(rawdata)
        self.x_data = self.variable(x)
        self.y_data = self.variable(y)
        self.k = k
        x_pts, y_pts, calculated_gradient = self.gradient(
            self.x_data, self.y_data, self.k, gradient
        )
        smoothed_gradient = self.smooth_gradient(calculated_gradient, smoothing_filter)
        gradient_title = f"d({y})/d({x})" if gradient == "dydx" else f"d({x})/d({y})"
        self.output_data = self.assign_outputs(
            {x: x_pts, y: y_pts, gradient_title: smoothed_gradient}
        )

    @staticmethod
    def get_dx(x: NDArray[np.float64]) -> float:
        """Get the x sampling interval, assuming uniformly spaced x values.

        Args:
            x (NDArray): The x values.

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

    @staticmethod
    def get_dy_and_counts(
        y: NDArray[np.float64], dy: float
    ) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
        """Get the y sampling interval, bin midpoints and counts.

        Args:
            y (NDArray): The y values.
            dy (float): The bin size.

        Returns:
            Tuple[float, NDArray, NDArray]:
                The y sampling interval, bin midpoints and counts.
        """
        y_range = y.max() - y.min()
        y_bins = np.linspace(y.min(), y.max(), int(np.ceil(y_range / dy)))
        dy = y_bins[1] - y_bins[0]
        N, _ = np.histogram(y, bins=y_bins)
        y_midpoints = y_bins[:-1] + np.diff(y_bins) / 2
        return dy, y_midpoints, N

    @staticmethod
    def y_sampling_interval(y: NDArray[np.float64]) -> float:
        r"""Get the y sampling interval, :math:`\delta R` in :footcite:t:`Feng2020`.

        Args:
            y (NDArray): The y values.

        Returns:
            float: The y sampling interval.
        """
        y_unique = np.unique(y)
        y_sorted = np.sort(y_unique)
        y_diff = np.diff(y_sorted)
        return np.min(y_diff)

    @classmethod
    def gradient(
        cls, x: NDArray[np.float64], y: NDArray[np.float64], k: int, gradient: str
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        r"""Calculate the gradient of the data, assuming x is uniformly spaced.

        Args:
            x (NDArray): The x values.
            y (NDArray): The y values.
            k (int):
                The integer multiple to apply to the sampling interval for the bin size
                (:math:`\delta R` in paper).
            gradient (str): The gradient to calculate, either 'dydx' or 'dxdy'.

        Returns:
            Tuple[NDArray, NDArray, NDArray]:
                The x values, the y midpoints and the calculated gradient.
        """
        dx = cls.get_dx(x)
        dy = k * cls.y_sampling_interval(y)
        dy, y_midpoints, N = cls.get_dy_and_counts(y, dy)
        if gradient == "dydx":
            grad = 1 / N * dy / dx
        elif gradient == "dxdy":
            grad = N * dx / dy
        f = interp1d(y, x, assume_sorted=False)
        x_pts = f(y_midpoints)
        return x_pts, y_midpoints, grad

    @staticmethod
    def smooth_gradient(
        gradient: NDArray[np.float64], alpha: List[float]
    ) -> NDArray[np.float64]:
        """Smooth the calculated gradient.

        Args:
            gradient (NDArray[np.float64]): The gradient vector.
            alpha (list[float]): The smoothing coefficients.

        Returns:
            NDArray[float]: The smoothed gradient vector.
        """
        A = np.zeros((len(gradient), len(gradient)))
        w = np.floor(len(alpha) / 2)
        for n in range(len(alpha)):
            k = n - w
            vector = np.ones(int(len(gradient) - abs(k)))
            diag = np.diag(vector, int(k))
            A += alpha[n] * diag
        return A @ gradient
