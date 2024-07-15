"""Module for calculating stoichiometry limits using dQdV data."""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from pyprobe.methods.basemethod import BaseMethod
from pyprobe.methods.ocv_fitting.Simple_OCV_fit import Simple_OCV_fit
from pyprobe.rawdata import RawData


class dQdV_OCV_fit(BaseMethod):
    """A method for fitting OCV curves."""

    def __init__(
        self,
        rawdata: RawData,
        x_ne: NDArray[np.float64],
        x_pe: NDArray[np.float64],
        ocp_ne: NDArray[np.float64],
        ocp_pe: NDArray[np.float64],
        x_guess: NDArray[np.float64],
    ):
        """Initialize the Simple_OCV_fit method.

        Args:
            rawdata (Result): The input data to the method.
            x_ne (NDArray[np.float64]): The anode stoichiometry data.
            x_pe (NDArray[np.float64]): The cathode stoichiometry data.
            ocp_ne (NDArray[np.float64]): The anode OCP data.
            ocp_pe (NDArray[np.float64]): The cathode OCP data.
            x_guess (NDArray[np.float64]): The initial guess for the fit.
        """
        super().__init__(rawdata)
        self.voltage = self.variable("Voltage [V]")
        self.capacity = self.variable("Capacity [Ah]")
        self.cell_capacity = np.abs(np.ptp(self.capacity))
        self.SOC = (self.capacity - self.capacity.min()) / self.cell_capacity
        self.x_ne = x_ne
        self.x_pe = x_pe
        self.ocp_ne = ocp_ne
        self.ocp_pe = ocp_pe
        self.x_guess = x_guess

        self.dVdQ = np.gradient(self.voltage, self.SOC)

        self.cost_function(self.x_guess)
        fitting_result = minimize(self.cost_function, x_guess)
        self.x_pe_lo, self.x_pe_hi, self.x_ne_lo, self.x_ne_hi = fitting_result.x
        print(fitting_result.x)

    def cost_function(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        """Cost function for the fitting of the OCV curves."""
        x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi = params
        modelled_OCV = Simple_OCV_fit.calc_full_cell_OCV(
            self.SOC,
            x_pe_lo,
            x_pe_hi,
            x_ne_lo,
            x_ne_hi,
            self.x_pe,
            self.ocp_pe,
            self.x_ne,
            self.ocp_ne,
        )
        dVdQ = np.gradient(modelled_OCV, self.SOC)
        # print(dVdQ)
        return np.sum((dVdQ - self.dVdQ) ** 2)
