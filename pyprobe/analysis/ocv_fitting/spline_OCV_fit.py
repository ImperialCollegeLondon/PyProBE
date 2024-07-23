"""Module for simple OCV fitting."""

from typing import Optional, Tuple

import numpy as np
import scipy.interpolate as interp
import scipy.optimize as opt
from numpy.typing import NDArray

from pyprobe.analysis.utils import BaseMethod
from pyprobe.rawdata import RawData


class Spline_OCV_fit(BaseMethod):
    """A method for fitting OCV curves."""

    def __init__(
        self,
        rawdata: RawData,
        x_ne: NDArray[np.float64],
        x_pe: NDArray[np.float64],
        ocp_ne: NDArray[np.float64],
        ocp_pe: NDArray[np.float64],
        x_guess: NDArray[np.float64],
        smoothing_lambda: Optional[float] = None,
        fitting_target: str = "OCV",
        optimizer: Optional[str] = None,
    ):
        """Initialize the Simple_OCV_fit method.

        Args:
            rawdata (Result): The input data to the method.
            x_ne (NDArray[np.float64]): The anode stoichiometry data.
            x_pe (NDArray[np.float64]): The cathode stoichiometry data.
            ocp_ne (NDArray[np.float64]): The anode OCP data.
            ocp_pe (NDArray[np.float64]): The cathode OCP data.
            x_guess (NDArray[np.float64]): The initial guess for the fit.
            smoothing_lambda (float, optional): The smoothing parameter. Default None.
            fitting_target (str, optional): The target for the fitting. Default "OCV".
            optimizer (str, optional): The optimizer to use. Default is None.
        """
        super().__init__(rawdata)
        self.voltage = self.variable("Voltage [V]")
        self.capacity = self.variable("Capacity [Ah]")
        self.cell_capacity = np.abs(np.ptp(self.capacity))
        self.SOC = (self.capacity - self.capacity.min()) / self.cell_capacity
        if self.SOC[0] > self.SOC[-1]:
            self.SOC = np.flip(self.SOC)
            self.voltage = np.flip(self.voltage)
        self.voltage_spline = interp.make_smoothing_spline(
            self.SOC, self.voltage, lam=smoothing_lambda
        )
        self.voltage = self.voltage_spline(self.SOC)
        self.dVdSOC_spline = self.voltage_spline.derivative()
        self.dVdSOC = self.dVdSOC_spline(self.SOC)
        self.dSOCdV = 1 / self.dVdSOC
        self.x_ne = x_ne
        self.x_pe = x_pe
        self.ocp_ne = ocp_ne
        self.ocp_pe = ocp_pe
        self.x_guess = x_guess
        self.fitting_target = fitting_target

        if self.fitting_target == "OCV" and optimizer is None:
            optimizer = "minimize"
        elif self.fitting_target in ["dQdV", "dVdQ"] and optimizer is None:
            optimizer = "differential_evolution"

        if optimizer == "minimize":
            fitting_result = opt.minimize(self.cost_function, x_guess)
        elif optimizer == "differential_evolution":
            fitting_result = opt.differential_evolution(
                self.cost_function,
                bounds=[(0.75, 0.95), (0.2, 0.3), (0, 0.05), (0.85, 0.95)],
            )

        self.x_pe_lo, self.x_pe_hi, self.x_ne_lo, self.x_ne_hi = fitting_result.x
        (
            self.pe_capacity,
            self.ne_capacity,
            self.li_inventory,
        ) = self.calc_electrode_capacities(
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
        fitted_voltage = self.calc_full_cell_OCV(
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
        self.fitted_OCV = self.make_result(
            {
                "Capacity [Ah]": self.capacity,
                "SOC": self.SOC,
                "Input Voltage [V]": self.voltage,
                "Fitted Voltage [V]": fitted_voltage,
                "Input dSOCdV [1/V]": self.dSOCdV,
                "Fitted dSOCdV [1/V]": np.gradient(self.SOC, fitted_voltage),
                "Input dVdSOC [V]": self.dVdSOC,
                "Fitted dVdSOC [V]": np.gradient(fitted_voltage, self.SOC),
            }
        )
        self.fitted_OCV.column_definitions = {
            "SOC": "Cell state of charge.",
            "Voltage [V]": "Fitted OCV values.",
        }
        self.output_data = (self.stoichiometry_limits, self.fitted_OCV)

    def cost_function(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        """Cost function for the curve fitting."""
        x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi = params
        modelled_OCV = self.calc_full_cell_OCV(
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
        if self.fitting_target == "dQdV":
            model = np.gradient(self.SOC, modelled_OCV)
            truth = self.dSOCdV
        elif self.fitting_target == "dVdQ":
            model = np.gradient(modelled_OCV, self.SOC)
            truth = self.dVdSOC
        else:
            model = modelled_OCV
            truth = self.voltage
        return np.sum((model - truth) ** 2)

    @staticmethod
    def calc_electrode_capacities(
        x_pe_lo: float,
        x_pe_hi: float,
        x_ne_lo: float,
        x_ne_hi: float,
        cell_capacity: float,
    ) -> Tuple[float, float, float]:
        """Calculate the electrode capacities.

        Args:
            x_pe_lo (float): The cathode stoichiometry at lowest cell SOC.
            x_pe_hi (float): The cathode stoichiometry at highest cell SOC.
            x_ne_lo (float): The anode stoichiometry at lowest cell SOC.
            x_ne_hi (float): The anode stoichiometry at highest cell SOC.

        Returns:
            Tuple[float, float, float]:
                - NDArray: The cathode capacity.
                - NDArray: The anode capacity.
                - NDArray: The lithium inventory.
        """
        pe_capacity = cell_capacity / (x_pe_lo - x_pe_hi)
        ne_capacity = cell_capacity / (x_ne_hi - x_ne_lo)
        li_inventory = (pe_capacity * x_pe_lo) + (ne_capacity * x_ne_lo)
        return pe_capacity, ne_capacity, li_inventory

    @staticmethod
    def calc_full_cell_OCV(
        SOC: NDArray[np.float64],
        x_pe_lo: NDArray[np.float64],
        x_pe_hi: NDArray[np.float64],
        x_ne_lo: NDArray[np.float64],
        x_ne_hi: NDArray[np.float64],
        x_pe: NDArray[np.float64],
        ocp_pe: NDArray[np.float64],
        x_ne: NDArray[np.float64],
        ocp_ne: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate the full cell OCV.

        Args:
            SOC (NDArray[np.float64]): The full cell SOC.
            x_pe_lo (float): The cathode stoichiometry at lowest cell SOC.
            x_pe_hi (float): The cathode stoichiometry at highest cell SOC.
            x_ne_lo (float): The cathode stoichiometry at lowest cell SOC.
            x_ne_hi (float): The anode stoichiometry at highest cell SOC.
            x_pe (NDArray[np.float64]): The cathode stoichiometry data.
            ocp_pe (NDArray[np.float64]): The cathode OCP data.
            x_ne (NDArray[np.float64]): The anode stoichiometry data.
            ocp_ne (NDArray[np.float64]): The anode OCP data.

        Returns:
            NDArray: The full cell OCV.
        """
        n_points = 1000
        # make vectors between stoichiometry limits during charge
        z_ne = np.linspace(x_ne_lo, x_ne_hi, n_points)
        z_pe = np.linspace(
            x_pe_lo, x_pe_hi, n_points
        )  # flip the cathode limits to match charge direction

        # make an SOC vector with the same number of points
        SOC_sampling = np.linspace(0, 1, n_points)

        # interpolate the real electrode OCP data with the created stoichiometry vectors
        OCP_ne = np.interp(z_ne, x_ne, ocp_ne)
        OCP_pe = np.interp(z_pe, x_pe, ocp_pe)
        # OCP_pe = np.flip(OCP_pe) # flip the cathode OCP to match charge direction

        # interpolate the final OCV curve with the original SOC vector
        OCV = np.interp(SOC, SOC_sampling, OCP_pe - OCP_ne)
        return OCV
