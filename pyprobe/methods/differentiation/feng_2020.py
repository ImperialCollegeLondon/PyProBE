"""Module for the Feng et al. (2020) method for ICA."""
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from pyprobe.method import Method
from pyprobe.result import Result


class Feng2020(Method):
    """A method for calculating the incremental capacity analysis."""

    def __init__(self, rawdata: Result, parameters: Dict[str, float]):
        """Initialize the Feng2020 method.

        Args:
            rawdata (Result): The input data to the method.
            parameters (Dict[str, float]): The parameters for the method.
        """
        super().__init__(rawdata, parameters)
        self.voltage = self.variable("Voltage [V]")
        self.deltaV = self.parameter("deltaV")
        v_points, IC = self.calculate_dQdV(self.voltage, self.deltaV)
        self.dQdV = self.assign_outputs({"Voltage [V]": v_points, "IC [Ah/V]": IC})

    @classmethod
    def calculate_dQdV(
        cls, voltage: NDArray[np.float64], deltaV: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Calculate the normalised incremental capacity of the step.

        Method from: 10.1016/j.etran.2020.100051

        Args:
            step (Step): A step object.
            parameter_dict (Dict[str, float]): A dictionary containing
                the parameters for the method

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]:
                - NDArray: The voltage values.
                - NDArray: The incremental capacity values.
        """
        n = len(voltage)
        V_range = voltage.max() - voltage.min()
        v = np.linspace(voltage.min(), voltage.max(), int(V_range / deltaV))
        deltaV = v[1] - v[0]

        N, _ = np.histogram(voltage, bins=v)
        IC = N / n * 1 / deltaV
        v_midpoints = v[:-1] + np.diff(v) / 2

        IC = cls.smooth_IC(IC, [0.0668, 0.2417, 0.3830, 0.2417, 0.0668])
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
