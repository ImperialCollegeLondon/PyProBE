"""Module for simple OCV fitting."""

from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from pybatdata.method import Method
from pybatdata.rawdata import RawData


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
        self.ne_data = self.variable("Anode Data")
        self.pe_data = self.variable("Cathode Data")
        self.z_guess = self.variable("Initial Guess")
        self.define_outputs(
            [
                "Cathode Stoichiometry Limits",
                "Anode Stoichiometry Limits",
                "Cell Capacity" "Cathode Capacity",
                "Anode Capacity",
                "Stoichiometry Offset",
            ]
        )
        self.assign_outputs(
            self.fit_ocv(
                self.capacity, self.voltage, self.ne_data, self.pe_data, self.z_guess
            )
        )

    @classmethod
    def fit_ocv(
        cls,
        capacity: NDArray[np.float64],
        voltage: NDArray[np.float64],
        ne_data: NDArray[np.float64],
        pe_data: NDArray[np.float64],
        z_guess: List[float],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], float, float, float, float,]:
        """Fit the OCV curve.

        Args:
            capacity (NDArray[np.float64]): The capacity data.
            voltage (NDArray[np.float64]): The voltage data.
            ne_data (NDArray[np.float64]): The anode half cell data.
            pe_data (NDArray[np.float64]): The cathode half cell data.
            z_guess (List[float]): The initial guess for the fit.
        """
        cell_capacity = np.ptp(capacity)
        SOC = capacity / cell_capacity

        def objective_func(
            SOC: NDArray[np.float64],
            z_pe_lo: float,
            z_pe_hi: float,
            z_ne_lo: float,
            z_ne_hi: float,
        ) -> NDArray[np.float64]:
            return cls.calc_full_cell_OCV(
                SOC, z_pe_lo, z_pe_hi, z_ne_lo, z_ne_hi, ne_data, pe_data
            )

        z_out = curve_fit(
            objective_func,
            SOC,
            voltage,
            p0=z_guess,
            bounds=([0, 0, 0, 0], [1, 1, 1, 1]),
        )
        pe_stoich_limits = np.array([z_out[0][0], z_out[0][1]])
        ne_stoich_limits = np.array([z_out[0][2], z_out[0][3]])

        pe_capacity, ne_capacity, stoich_offset = cls.calc_electrode_capacities(
            pe_stoich_limits, ne_stoich_limits, cell_capacity
        )

        return (
            pe_stoich_limits,
            ne_stoich_limits,
            cell_capacity,
            pe_capacity,
            ne_capacity,
            stoich_offset,
        )

    @staticmethod
    def calc_electrode_capacities(
        pe_stoich_limits: NDArray[np.float64],
        ne_stoich_limits: NDArray[np.float64],
        cell_capacity: float,
    ) -> Tuple[float, float, float]:
        """Calculate the electrode capacities.

        Args:
            pe_stoich_limits (NDArray[np.float64]): The cathode stoichiometry limits.
            ne_stoich_limits (NDArray[np.float64]): The anode stoichiometry limits.
            cell_capacity (NDArray[np.float64]): The cell capacity.
        """
        pe_capacity = cell_capacity / (pe_stoich_limits[1] - pe_stoich_limits[0])
        ne_capacity = cell_capacity / (ne_stoich_limits[1] - ne_stoich_limits[0])
        stoich_offset = (pe_stoich_limits[0] * pe_capacity) - (
            ne_stoich_limits[0] * ne_capacity
        )

        return pe_capacity, ne_capacity, stoich_offset

    @staticmethod
    def calc_full_cell_OCV(
        SOC: NDArray[np.float64],
        z_pe_lo: NDArray[np.float64],
        z_pe_hi: NDArray[np.float64],
        z_ne_lo: NDArray[np.float64],
        z_ne_hi: NDArray[np.float64],
        ne_data: NDArray[np.float64],
        pe_data: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate the full cell OCV.

        Args:
            SOC (NDArray[np.float64]): The full cell SOC.
            z_pe_lo (float): The cathode upper stoichiomteric limit.
            z_pe_hi (float): The cathode lower stoichiomteric limit.
            z_ne_lo (float): The anode upper stoichiomteric limit.
            z_ne_hi (float): The anode lower stoichiomteric limit.
            ne_data (NDArray[np.float64]): The anode half cell data.
            pe_data (NDArray[np.float64]): The cathode half cell data.
        """
        n_points = len(SOC)
        z_ne = np.linspace(z_ne_lo, z_ne_hi, n_points)
        z_pe = np.linspace(z_pe_lo, z_pe_hi, n_points)

        OCP_ne = np.interp(z_ne, ne_data[:, 0], ne_data[:, 1])
        OCP_pe = np.interp(z_pe, pe_data[:, 0], pe_data[:, 1])

        return OCP_pe - OCP_ne
