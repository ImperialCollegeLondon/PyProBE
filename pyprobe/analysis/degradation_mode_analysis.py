"""Module for degradation mode analysis methods."""

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

import pyprobe.analysis.base.degradation_mode_analysis_functions as dma_functions
from pyprobe.analysis import utils
from pyprobe.analysis.utils import AnalysisValidator
from pyprobe.result import Result
from pyprobe.typing import FilterToCycleType, PyProBEDataType


class DMA(BaseModel):
    """A class for degradation mode analysis methods.

    Args:
        input_data (RawData): The input data to the method.
    """

    input_data: PyProBEDataType

    stoichiometry_limits: Optional[Result] = None
    fitted_OCV: Optional[Result] = None
    dma_result: Optional[Result] = None

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
        if "SOC" in self.input_data.column_list:
            required_columns = ["Voltage [V]", "Capacity [Ah]", "SOC"]
            validator = AnalysisValidator(
                input_data=self.input_data, required_columns=required_columns
            )
            voltage, capacity, SOC = validator.variables
            cell_capacity = np.abs(np.ptp(capacity)) / np.abs(np.ptp(SOC))
        else:
            required_columns = ["Voltage [V]", "Capacity [Ah]"]
            validator = AnalysisValidator(
                input_data=self.input_data, required_columns=required_columns
            )
            voltage, capacity = validator.variables
            cell_capacity = np.abs(np.ptp(capacity))
            SOC = (capacity - capacity.min()) / cell_capacity

        dVdSOC = np.gradient(voltage, SOC)
        dSOCdV = np.gradient(SOC, voltage)

        if optimizer is None:
            if fitting_target == "OCV":
                optimizer = "minimize"
            else:
                optimizer = "differential_evolution"

        fitting_target_data = {
            "OCV": voltage,
            "dQdV": dSOCdV,
            "dVdQ": dVdSOC,
        }[fitting_target]

        x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi = dma_functions.ocv_curve_fit(
            SOC,
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
            x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi, cell_capacity
        )

        self.stoichiometry_limits = self.input_data.clean_copy(
            {
                "x_pe low SOC": np.array([x_pe_lo]),
                "x_pe high SOC": np.array([x_pe_hi]),
                "x_ne low SOC": np.array([x_ne_lo]),
                "x_ne high SOC": np.array([x_ne_hi]),
                "Cell Capacity [Ah]": np.array([cell_capacity]),
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

        fitted_voltage = dma_functions.calc_full_cell_OCV(
            SOC,
            x_pe_lo,
            x_pe_hi,
            x_ne_lo,
            x_ne_hi,
            x_pe,
            ocp_pe,
            x_ne,
            ocp_ne,
        )
        self.fitted_OCV = self.input_data.clean_copy(
            {
                "Capacity [Ah]": capacity,
                "SOC": SOC,
                "Input Voltage [V]": voltage,
                "Fitted Voltage [V]": fitted_voltage,
                "Input dSOCdV [1/V]": dSOCdV,
                "Fitted dSOCdV [1/V]": np.gradient(SOC, fitted_voltage),
                "Input dVdSOC [V]": dVdSOC,
                "Fitted dVdSOC [V]": np.gradient(fitted_voltage, SOC),
            }
        )
        self.fitted_OCV.column_definitions = {
            "SOC": "Cell state of charge.",
            "Voltage [V]": "Fitted OCV values.",
        }

        return self.stoichiometry_limits, self.fitted_OCV

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
        required_columns = [
            "Cell Capacity [Ah]",
            "Cathode Capacity [Ah]",
            "Anode Capacity [Ah]",
            "Li Inventory [Ah]",
        ]
        AnalysisValidator(
            input_data=reference_stoichiometry_limits, required_columns=required_columns
        )
        if self.stoichiometry_limits is None:
            raise ValueError("No stoichiometry limits have been calculated.")
        AnalysisValidator(
            input_data=self.stoichiometry_limits, required_columns=required_columns
        )

        electrode_capacity_results = [
            reference_stoichiometry_limits,
            self.stoichiometry_limits,
        ]
        cell_capacity = utils.assemble_array(
            electrode_capacity_results, "Cell Capacity [Ah]"
        )
        pe_capacity = utils.assemble_array(
            electrode_capacity_results, "Cathode Capacity [Ah]"
        )
        ne_capacity = utils.assemble_array(
            electrode_capacity_results, "Anode Capacity [Ah]"
        )
        li_inventory = utils.assemble_array(
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

    @staticmethod
    def average_ocvs(
        input_data: FilterToCycleType,
        discharge_filter: Optional[str] = None,
        charge_filter: Optional[str] = None,
    ) -> "DMA":
        """Average the charge and discharge OCV curves.

        Args:
            discharge_result (Result): The discharge OCV data.
            charge_result (Result): The charge OCV data.

        Returns:
            DMA: A DMA object containing the averaged OCV curve.
        """
        required_columns = ["Voltage [V]", "Capacity [Ah]", "SOC"]
        AnalysisValidator(
            input_data=input_data,
            required_columns=required_columns,
            required_type=FilterToCycleType,
        )

        if discharge_filter is None:
            discharge_result = input_data.discharge()
        else:
            discharge_result = eval(f"input_data.{discharge_filter}")
        if charge_filter is None:
            charge_result = input_data.charge()
        else:
            charge_result = eval(f"input_data.{charge_filter}")
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
        return DMA(input_data=average_result)
