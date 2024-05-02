"""Module for the Feng et al. (2020) method for ICA."""
from typing import TYPE_CHECKING, Dict, List

import numpy as np
import polars as pl
from numpy.typing import NDArray

from pybatdata.result import Result

if TYPE_CHECKING:
    from pybatdata.step import Step


def IC(step: "Step", parameter_dict: Dict[str, float]) -> Result:
    """Calculate the normalised incremental capacity of the step.

    Method from: 10.1016/j.etran.2020.100051

    Args:
        step (Step): A step object.
        parameter_dict (Dict[str, float]): A dictionary containing
            the parameters for the method

    Returns:
        Result: a result object containing the normalised incremental capacity
    """
    V = step.data["Voltage [V]"]
    deltaV = parameter_dict["deltaV"]
    n = len(V)
    V_range = V.max() - V.min()
    v = np.linspace(V.min(), V.max(), int(V_range / deltaV))
    deltaV = v[1] - v[0]

    N, _ = np.histogram(V, bins=v)
    IC = N / n * 1 / deltaV
    v_midpoints = v[:-1] + np.diff(v) / 2

    IC = smooth_IC(IC, [0.0668, 0.2417, 0.3830, 0.2417, 0.0668])
    result = pl.DataFrame({"Voltage [V]": v_midpoints, "IC [Ah/V]": IC})
    return Result(result, step.info)


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
