"""A module for the Step class."""

from typing import Dict, List

import numpy as np
import polars as pl
from numpy.typing import NDArray

from pybatdata.result import Result


class Step(Result):
    """A step in a battery test procedure."""

    def __init__(self, lazyframe: pl.LazyFrame, info: Dict[str, str | int | float]):
        """Create a step.

        Args:
            lazyframe (polars.LazyFrame): The lazyframe of data being filtered.
            info (Dict[str, str | int | float]): A dict containing test info.
        """
        super().__init__(lazyframe, info)

    @property
    def capacity(self) -> float:
        """Calculate the capacity passed during the step.

        Returns:
            float: The capacity passed during the step.
        """
        return abs(self.data["Capacity [Ah]"].max() - self.data["Capacity [Ah]"].min())

    def IC(self, deltaV: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Calculate the normalised incremental capacity of the step.

        Method from: 10.1016/j.etran.2020.100051

        Args:
            deltaV (float): The voltage sampling interval.

        Returns:
            tuple[NDArray[float], NDArray[float]]:
                The midpints of the voltage sampling intervals and the
                normalised incremental capacity.
        """
        V = self.data["Voltage [V]"]

        n = len(V)
        V_range = V.max() - V.min()
        v = np.linspace(V.min(), V.max(), int(V_range / deltaV))
        deltaV = v[1] - v[0]

        N, _ = np.histogram(V, bins=v)
        IC = N / n * 1 / deltaV
        v_midpoints = v[:-1] + np.diff(v) / 2

        IC = self.smooth_IC(IC, [0.0668, 0.2417, 0.3830, 0.2417, 0.0668])
        return v_midpoints, IC

    @staticmethod
    def smooth_IC(IC: NDArray[np.float64], alpha: List[float]) -> NDArray[np.float64]:
        """Smooth the incremental capacity.

        Args:
            IC (NDArray[np.float64]): The incremental capacity vector.
            alpha (list[float]): The smoothing coefficients.

        Returns:
            NDArray[float]: The smoothed incremental capacity.
        """
        A = np.zeros((len(IC), len(IC)))
        w = np.floor(len(alpha) / 2)
        for n in range(len(alpha)):
            k = n - w
            vector = np.ones(int(len(IC) - abs(k)))
            diag = np.diag(vector, int(k))
            A += alpha[n] * diag
        return A @ IC
