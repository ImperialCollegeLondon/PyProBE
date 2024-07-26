"""Module for degradation mode analysis methods."""
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import pyprobe.analysis.base.degradation_mode_analysis_functions as dma_functions
from pyprobe.analysis.utils import BaseAnalysis, analysismethod
from pyprobe.filters import Cycle, Experiment, RawData
from pyprobe.result import Result


class DMA(Result, BaseAnalysis):
    """A class for degradation mode analysis methods.

    Args:
        rawdata (RawData): The input data to the method.
    """

    def __init__(self, rawdata: Union[Experiment, Cycle, RawData, Result]) -> None:
        """Initialize the DMA object."""
        self.result_list = ["stoichiometry_limits", "fitted_OCV", "dma_result"]
        self.rawdata = rawdata
        self._data = rawdata._data
        self.info = rawdata.info
        self.voltage = rawdata.get_only("Voltage [V]")
        self.capacity = rawdata.get_only("Capacity [Ah]")

        if "SOC" in rawdata.column_list:
            self.SOC = rawdata.get_only("SOC")
            self.cell_capacity = np.abs(np.ptp(self.capacity)) / np.abs(
                np.ptp(self.SOC)
            )
        else:
            self.cell_capacity = np.abs(np.ptp(self.capacity))
            self.SOC = (self.capacity - self.capacity.min()) / self.cell_capacity
        self.dSOCdV = np.gradient(self.SOC, self.voltage)
        self.dVdSOC = np.gradient(self.voltage, self.SOC)

    @analysismethod
    def fit_ocv(
        self,
        x_ne: NDArray[np.float64],
        x_pe: NDArray[np.float64],
        ocp_ne: NDArray[np.float64],
        ocp_pe: NDArray[np.float64],
        x_guess: NDArray[np.float64],
        fitting_target: str = "OCV",
        optimizer: Optional[str] = None,
    ) -> Tuple[Result, Result]:
        """Fit half cell open circuit potential curves to full cell OCV data.

        Args:
            x_ne (NDArray[np.float64]): The anode stoichiometry data.
            x_pe (NDArray[np.float64]): The cathode stoichiometry data.
            ocp_ne (NDArray[np.float64]): The anode OCP data.
            ocp_pe (NDArray[np.float64]): The cathode OCP data.
            x_guess (NDArray[np.float64]): The initial guess for the fit.

        Returns:
            Tuple[Result, Result]:
                - Result: The stoichiometry limits and electrode capacities.
                - Result: The fitted OCV data.
        """
        if optimizer is None:
            if fitting_target == "OCV":
                optimizer = "minimize"
            else:
                optimizer = "differential_evolution"

        fitting_target_data = {
            "OCV": self.voltage,
            "dQdV": self.dSOCdV,
            "dVdQ": self.dVdSOC,
        }[fitting_target]

        x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi = dma_functions.ocv_curve_fit(
            self.SOC,
            fitting_target_data,
            x_pe,
            ocp_pe,
            x_ne,
            ocp_ne,
            fitting_target,
            optimizer,
            x_guess,
        )

        (
            pe_capacity,
            ne_capacity,
            li_inventory,
        ) = dma_functions.calc_electrode_capacities(
            x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi, self.cell_capacity
        )

        self.stoichiometry_limits = self.rawdata.clean_copy(
            {
                "x_pe low SOC": np.array([x_pe_lo]),
                "x_pe high SOC": np.array([x_pe_hi]),
                "x_ne low SOC": np.array([x_ne_lo]),
                "x_ne high SOC": np.array([x_ne_hi]),
                "Cell Capacity [Ah]": np.array([self.cell_capacity]),
                "Cathode Capacity [Ah]": np.array([pe_capacity]),
                "Anode Capacity [Ah]": np.array([ne_capacity]),
                "Li Inventory [Ah]": np.array([li_inventory]),
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

        self.fitted_voltage = dma_functions.calc_full_cell_OCV(
            self.SOC,
            x_pe_lo,
            x_pe_hi,
            x_ne_lo,
            x_ne_hi,
            x_pe,
            ocp_pe,
            x_ne,
            ocp_ne,
        )
        self.fitted_OCV = self.rawdata.clean_copy(
            {
                "Capacity [Ah]": self.capacity,
                "SOC": self.SOC,
                "Input Voltage [V]": self.voltage,
                "Fitted Voltage [V]": self.fitted_voltage,
                "Input dSOCdV [1/V]": self.dSOCdV,
                "Fitted dSOCdV [1/V]": np.gradient(self.SOC, self.fitted_voltage),
                "Input dVdSOC [V]": self.dVdSOC,
                "Fitted dVdSOC [V]": np.gradient(self.fitted_voltage, self.SOC),
            }
        )
        self.fitted_OCV.column_definitions = {
            "SOC": "Cell state of charge.",
            "Voltage [V]": "Fitted OCV values.",
        }

        return self.stoichiometry_limits, self.fitted_OCV

    @analysismethod
    def quantify_degradation_modes(
        self, reference_stoichiometry_limits: Result
    ) -> Result:
        """Quantify the change in degradation modes between at least two OCV fits.

        Args:
            reference_stoichiometry_limits (Result):
                A result object containing the beginning of life stoichiometry limits.

        Returns:
            Result:
                A result object containing the SOH, LAM_pe, LAM_ne, and LLI for each of
                the provided OCV fits.
        """
        if not hasattr(self, "stoichiometry_limits"):
            raise ValueError("No electrode capacities have been calculated.")

        electrode_capacity_results = [
            reference_stoichiometry_limits,
            self.stoichiometry_limits,
        ]
        cell_capacity = self.assemble_array(
            electrode_capacity_results, "Cell Capacity [Ah]"
        )
        pe_capacity = self.assemble_array(
            electrode_capacity_results, "Cathode Capacity [Ah]"
        )
        ne_capacity = self.assemble_array(
            electrode_capacity_results, "Anode Capacity [Ah]"
        )
        li_inventory = self.assemble_array(
            electrode_capacity_results, "Li Inventory [Ah]"
        )
        SOH, LAM_pe, LAM_ne, LLI = dma_functions.calculate_dma_parameters(
            cell_capacity, pe_capacity, ne_capacity, li_inventory
        )

        self.dma_result = electrode_capacity_results[0].clean_copy(
            {
                "SOH": SOH,
                "LAM_pe": LAM_pe,
                "LAM_ne": LAM_ne,
                "LLI": LLI,
            }
        )
        self.dma_result.column_definitions = {
            "SOH": "Cell capacity normalized to initial capacity.",
            "LAM_pe": "Loss of active material in positive electrode.",
            "LAM_ne": "Loss of active material in positive electrode.",
            "LLI": "Loss of lithium inventory.",
        }
        return self.dma_result

    @analysismethod
    def average_ocvs(
        self,
        discharge_filter: Optional[str] = None,
        charge_filter: Optional[str] = None,
    ) -> Result:
        """Average the charge and discharge OCV curves.

        Args:
            discharge_result (Result): The discharge OCV data.
            charge_result (Result): The charge OCV data.

        Returns:
            Tuple[Result, Result]: The averaged charge and discharge OCV curves.
        """
        if not (
            isinstance(self.rawdata, Experiment) or isinstance(self.rawdata, Cycle)
        ):
            raise ValueError(
                "RawData object must be a Cycle or Experiment object to"
                " average OCVs."
            )
        if discharge_filter is None:
            discharge_result = self.rawdata.discharge()
        else:
            discharge_result = eval(f"self.rawdata.{discharge_filter}")
        if charge_filter is None:
            charge_result = self.rawdata.charge()
        else:
            charge_result = eval(f"self.rawdata.{charge_filter}")
        charge_SOC = charge_result.get_only("SOC")
        charge_OCV = charge_result.get_only("Voltage [V]")
        charge_current = charge_result.get_only("Current [A]")
        discharge_SOC = discharge_result.get_only("SOC")
        discharge_OCV = discharge_result.get_only("Voltage [V]")
        discharge_current = discharge_result.get_only("Current [A]")

        average_OCV = dma_functions.average_OCV_curves(
            charge_SOC,
            charge_OCV,
            charge_current,
            discharge_SOC,
            discharge_OCV,
            discharge_current,
        )

        average_result = charge_result.clean_copy(
            {
                "Voltage [V]": average_OCV,
                "Capacity [Ah]": charge_result.get_only("Capacity [Ah]"),
                "SOC": charge_SOC,
            }
        )
        return DMA(rawdata=average_result)
