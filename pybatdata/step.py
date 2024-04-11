"""A module for the Step class."""

import numpy as np
from pybatdata.base import Base
import polars as pl

class Step(Base):
    """A step in a battery test procedure."""
    def __init__(self, 
                 lazyframe: pl.LazyFrame,
                 info: dict):
        """Create a step.

            Args:
                lazyframe (polars.LazyFrame): The lazyframe of data being filtered.
        """
        super().__init__(lazyframe, info)

    @property
    def capacity(self) -> float:
        """Calculate the capacity passed during the step.
        
        Returns:
            float: The capacity passed during the step.
        """
        return abs(self.data['Capacity [Ah]'].max() - self.data['Capacity [Ah]'].min())
    
    def IC(self, deltaV: float) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the normalised incremental capacity ($\frac{1}{Q}\frac{dQ}{dV}$) of the step.
        Method from: 10.1016/j.etran.2020.100051
        
        Args:
            deltaV (float): The voltage sampling interval.

        Returns:
            tuple[np.ndarray, np.ndarray]: The midpints of the voltage sampling intervales and the normalised incremental capacity.
        """
        V = self.data['Voltage [V]']
        
        n = len[V]
        V_range = V.max() - V.min()
        v = np.linspace(V.min(), V.max(), int(V_range/deltaV))
        deltaV = v[1]-v[0]
        
        N, _ = np.histogram(V, bins=v)
        IC = N/n * 1/deltaV
        v_midpoints = v[:-1] + np.diff(v)/2
        
        IC = self.smooth_IC(IC, [0.0668, 0.2417, 0.3830, 0.2417, 0.0668])
        return v_midpoints, IC

    @staticmethod
    def smooth_IC(IC, alpha: list[float]) -> np.ndarray:
        """Smooth the incremental capacity.
        
        Args:
            IC (np.ndarray): The incremental capacity vector.
            alpha (list[float]): The smoothing coefficients.
        
        Returns:
            np.ndarray: The smoothed incremental capacity.
            
        """
        A = np.zeros((len(IC), len(IC)))
        w = np.floor(len(alpha)/2)
        for n in range(len(alpha)):
            k = n - w
            vector = np.ones(int(len(IC) - abs(k)))
            diag = np.diag(vector, int(k))
            A += alpha[n] * diag
        return A @ IC
    
    