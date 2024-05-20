"""Module for simple OCV fitting."""

from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from pyprobe.method import Method
from pyprobe.rawdata import RawData


class Simple_OCV_fit(Method):
    """A method for fitting OCV curves."""

    def __init__(self, rawdata: RawData, parameters: Dict[str, Any]):
        """Initialize the Simple_OCV_fit method.

        Args:
            rawdata (Result): The input data to the method.
            parameters (Dict[str, float]): The parameters for the method.
        """
        super().__init__(rawdata, parameters)
        self.voltage = self.variable("Voltage [V]")
        self.capacity = self.variable("Capacity [Ah]")
        self.x_ne = self.parameter("Anode Stoichiometry")
        self.x_pe = self.parameter("Cathode Stoichiometry")
        self.ocp_ne = self.parameter("Anode OCP [V]")
        self.ocp_pe = self.parameter("Cathode OCP [V]")
        self.x_guess = self.parameter("Initial Guess")
        self.define_outputs(
            [
                "Cathode Lower Stoichiometry Limit",
                "Cathode Upper Stoichiometry Limit",
                "Anode Lower Stoichiometry Limit",
                "Anode Upper Stoichiometry Limit",
                "Cell Capacity",
                "Cathode Capacity",
                "Anode Capacity",
                "Stoichiometry Offset",
            ]
        )
        self.assign_outputs(
            self.fit_ocv(
                self.capacity,
                self.voltage,
                self.x_pe,
                self.ocp_pe,
                self.x_ne,
                self.ocp_ne,
                self.x_guess,
            )
        )

    @classmethod
    def fit_ocv(
        cls,
        capacity: NDArray[np.float64],
        voltage: NDArray[np.float64],
        x_pe: NDArray[np.float64],
        ocp_pe: NDArray[np.float64],
        x_ne: NDArray[np.float64],
        ocp_ne: NDArray[np.float64],
        x_guess: List[float],
    ) -> Tuple[float, float, float, float, float, float, float, float,]:
        """Fit the OCV curve.

        Args:
            capacity (NDArray[np.float64]): The capacity data.
            voltage (NDArray[np.float64]): The voltage data.
            x_pe (NDArray[np.float64]): The cathode stoichiometry data.
            ocp_pe (NDArray[np.float64]): The cathode OCP data.
            x_ne (NDArray[np.float64]): The anode stoichiometry data.
            ocp_ne (NDArray[np.float64]): The anode OCP data.
            x_guess (List[float]): The initial guess for the fit.

        Returns:
            Tuple[float, float, float, float, float, float, float, float]:
                - float: The cathode lower stoichiometry limit.
                - float: The cathode upper stoichiometry limit.
                - float: The anode lower stoichiometry limit.
                - float: The anode upper stoichiometry limit.
                - float: The cell capacity.
                - float: The cathode capacity.
                - float: The anode capacity.
                - float: The stoichiometry offset.
        """
        cell_capacity = np.ptp(capacity)
        SOC = capacity / cell_capacity

        def objective_func(
            SOC: NDArray[np.float64],
            x_pe_lo: float,
            x_pe_hi: float,
            x_ne_lo: float,
            x_ne_hi: float,
        ) -> NDArray[np.float64]:
            return cls.calc_full_cell_OCV(
                SOC, x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi, x_pe, ocp_pe, x_ne, ocp_ne
            )

        x_out = curve_fit(
            objective_func,
            SOC,
            voltage,
            p0=x_guess,
            bounds=([0, 0, 0, 0], [1, 1, 1, 1]),
        )
        x_pe_lo_out = x_out[0][0]
        x_pe_hi_out = x_out[0][1]
        x_ne_lo_out = x_out[0][2]
        x_ne_hi_out = x_out[0][3]

        pe_capacity, ne_capacity, stoich_offset = cls.calc_electrode_capacities(
            x_pe_lo_out, x_pe_hi_out, x_ne_lo_out, x_ne_lo_out, cell_capacity
        )

        return (
            x_pe_lo_out,
            x_pe_hi_out,
            x_ne_lo_out,
            x_ne_hi_out,
            cell_capacity,
            pe_capacity,
            ne_capacity,
            stoich_offset,
        )

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
            pe_stoich_limits (NDArray[np.float64]): The cathode stoichiometry limits.
            ne_stoich_limits (NDArray[np.float64]): The anode stoichiometry limits.
            cell_capacity (NDArray[np.float64]): The cell capacity.

        Returns:
            Tuple[float, float, float]:
                - NDArray: The cathode capacity.
                - NDArray: The anode capacity.
                - NDArray: The stoichiometry offset.
        """
        pe_capacity = cell_capacity / (x_pe_hi - x_pe_lo)
        ne_capacity = cell_capacity / (x_ne_hi - x_ne_lo)
        stoich_offset = ((1 - x_pe_hi) * pe_capacity) - (x_pe_lo * ne_capacity)

        return pe_capacity, ne_capacity, stoich_offset

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
            x_pe_lo (float): The cathode upper stoichiomteric limit.
            x_pe_hi (float): The cathode lower stoichiomteric limit.
            x_ne_lo (float): The anode upper stoichiomteric limit.
            x_ne_hi (float): The anode lower stoichiomteric limit.
            x_pe (NDArray[np.float64]): The cathode stoichiometry data.
            ocp_pe (NDArray[np.float64]): The cathode OCP data.
            x_ne (NDArray[np.float64]): The anode stoichiometry data.
            ocp_ne (NDArray[np.float64]): The anode OCP data.

        Returns:
            NDArray: The full cell OCV.
        """
        n_points = 10000
        # make vectors between stoichiometry limits during charge
        z_ne = np.linspace(x_ne_lo, x_ne_hi, n_points)
        z_pe = np.linspace(
            1 - x_pe_lo, 1 - x_pe_hi, n_points
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
