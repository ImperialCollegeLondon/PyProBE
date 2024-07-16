"""Module for calculating stoichiometry limits using dQdV data."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution

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
        method: Optional[str] = None,
    ):
        """Initialize the Simple_OCV_fit method.

        Args:
            rawdata (Result): The input data to the method.
            x_ne (NDArray[np.float64]): The anode stoichiometry data.
            x_pe (NDArray[np.float64]): The cathode stoichiometry data.
            ocp_ne (NDArray[np.float64]): The anode OCP data.
            ocp_pe (NDArray[np.float64]): The cathode OCP data.
            x_guess (NDArray[np.float64]): The initial guess for the fit.
            method (Optional[str]): The optimization method to use in scipy.minimize.
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
        fitting_result = differential_evolution(
            self.cost_function,
            bounds=[(0.75, 0.95), (0.2, 0.3), (0, 0.05), (0.85, 0.95)],
        )

        self.x_pe_lo, self.x_pe_hi, self.x_ne_lo, self.x_ne_hi = fitting_result.x
        (
            self.pe_capacity,
            self.ne_capacity,
            self.li_inventory,
        ) = Simple_OCV_fit.calc_electrode_capacities(
            self.x_pe_lo, self.x_pe_hi, self.x_ne_lo, self.x_ne_hi, self.cell_capacity
        )

        self.stoichiometry_limits = self.make_result(
            {
                "x_pe low SOC": np.array([self.x_pe_lo]),
                "x_pe high SOC": np.array([self.x_pe_hi]),
                "x_ne low SOC": np.array([self.x_ne_lo]),
                "x_ne high SOC": np.array([self.x_ne_hi]),
                "Cell Capacity [Ah]": np.array([self.cell_capacity]),
                "Cathode Capacity [Ah]": np.array([self.pe_capacity]),
                "Anode Capacity [Ah]": np.array([self.ne_capacity]),
                "Li Inventory [Ah]": np.array([self.li_inventory]),
            }
        )
        self.stoichiometry_limits.column_definitions = {
            "x_pe low SOC": "Positive electrode stoichiometry at lowest SOC point.",
            "x_pe high SOC": "Positive electrode stoichiometry at highest SOC point.",
            "x_ne low SOC": "Negative electrode stoichiometry at lowest SOC point.",
            "x_ne high SOC": "Negative electrode stoichiometry at highest SOC point.",
            "Cell Capacity [Ah]": "Total cell capacity.",
            "Cathode Capacity [Ah]": "Cathode capacity.",
            "Anode Capacity [Ah]": "Anode capacity.",
            "Li Inventory [Ah]": "Lithium inventory.",
        }
        OCV = Simple_OCV_fit.calc_full_cell_OCV(
            self.SOC,
            self.x_pe_lo,
            self.x_pe_hi,
            self.x_ne_lo,
            self.x_ne_hi,
            self.x_pe,
            self.ocp_pe,
            self.x_ne,
            self.ocp_ne,
        )
        self.fitted_OCV = self.make_result({"SOC": self.SOC, "Voltage [V]": OCV})
        self.fitted_OCV.column_definitions = {
            "SOC": "Cell state of charge.",
            "Voltage [V]": "Fitted OCV values.",
        }
        self.output_data = (self.stoichiometry_limits, self.fitted_OCV)

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
