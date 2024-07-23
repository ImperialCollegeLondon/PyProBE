"""Module for degradation mode analysis methods."""
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

import pyprobe.analysis.base.degradation_mode_analysis_functions as dma_functions
import pyprobe.analysis.utils as utils
from pyprobe.rawdata import RawData, Result


def fit_ocv(
    rawdata: RawData,
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
        rawdata (Result): The input data to the method.
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
    voltage = rawdata.get("Voltage [V]")
    capacity = rawdata.get("Capacity [Ah]")

    if "SOC" in rawdata.column_list:
        SOC = rawdata.get("SOC")
        cell_capacity = np.abs(np.ptp(capacity)) / SOC.max()
    else:
        cell_capacity = np.abs(np.ptp(capacity))
        SOC = (capacity - capacity.min()) / cell_capacity
    dSOCdV = np.gradient(SOC, voltage)
    dVdSOC = np.gradient(voltage, SOC)

    if optimizer is None:
        if fitting_target == "OCV":
            optimizer = "minimize"
        else:
            optimizer = "differential_evolution"

    fitting_target_data = {"OCV": voltage, "dQdV": dSOCdV, "dVdQ": dVdSOC}[
        fitting_target
    ]

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

    stoichiometry_limits = rawdata.clean_copy(
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
    stoichiometry_limits.column_definitions = {
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
    fitted_OCV = rawdata.clean_copy(
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
    fitted_OCV.column_definitions = {
        "SOC": "Cell state of charge.",
        "Voltage [V]": "Fitted OCV values.",
    }

    return stoichiometry_limits, fitted_OCV


def quantify_degradation_modes(electrode_capacity_results: List[Result]) -> Result:
    """Quantify the change in degradation modes between at least two OCV fits.

    Args:
        electrode_capacity_results (List[Result]):
            A list of result objects containing the fitted electrode capacities of the
            cell. The first output of the fit_ocv method.

    Returns:
        Result:
            A result object containing the SOH, LAM_pe, LAM_ne, and LLI for each of the
            provided OCV fits.
    """
    cell_capacity = utils.assemble_array(
        electrode_capacity_results, "Cell Capacity [Ah]"
    )
    pe_capacity = utils.assemble_array(
        electrode_capacity_results, "Cathode Capacity [Ah]"
    )
    ne_capacity = utils.assemble_array(
        electrode_capacity_results, "Anode Capacity [Ah]"
    )
    li_inventory = utils.assemble_array(electrode_capacity_results, "Li Inventory [Ah]")
    SOH, LAM_pe, LAM_ne, LLI = dma_functions.calculate_dma_parameters(
        cell_capacity, pe_capacity, ne_capacity, li_inventory
    )

    dma_result = electrode_capacity_results[0].clean_copy(
        {
            "SOH": SOH,
            "LAM_pe": LAM_pe,
            "LAM_ne": LAM_ne,
            "LLI": LLI,
        }
    )
    dma_result.column_definitions = {
        "SOH": "Cell capacity normalized to initial capacity.",
        "LAM_pe": "Loss of active material in positive electrode.",
        "LAM_ne": "Loss of active material in positive electrode.",
        "LLI": "Loss of lithium inventory.",
    }
    return dma_result


def average_ocvs(discharge_result: Result, charge_result: Result) -> Result:
    """Average the charge and discharge OCV curves.

    Args:
        discharge_result (Result): The discharge OCV data.
        charge_result (Result): The charge OCV data.

    Returns:
        Tuple[Result, Result]: The averaged charge and discharge OCV curves.
    """
    charge_SOC = charge_result.get("SOC")
    charge_OCV = charge_result.get("Voltage [V]")
    charge_current = charge_result.get("Current [A]")
    discharge_SOC = discharge_result.get("SOC")
    discharge_OCV = discharge_result.get("Voltage [V]")
    discharge_current = discharge_result.get("Current [A]")

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
            "SOC": charge_SOC,
        }
    )
    return average_result
