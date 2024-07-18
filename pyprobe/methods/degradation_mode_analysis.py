"""Module for degradation mode analysis methods."""
from typing import List, Optional, Tuple

import numpy as np
import scipy.optimize as opt
from numpy.typing import NDArray

from pyprobe.methods.basemethod import BaseMethod
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
    method = BaseMethod(rawdata)
    voltage = method.variable("Voltage [V]")
    capacity = method.variable("Capacity [Ah]")

    cell_capacity = np.abs(np.ptp(capacity))
    SOC = (capacity - capacity.min()) / cell_capacity
    dSOCdV = np.gradient(SOC, voltage)
    dVdSOC = np.gradient(voltage, SOC)

    if fitting_target == "OCV" and optimizer is None:
        optimizer = "minimize"
    elif fitting_target in ["dQdV", "dVdQ"] and optimizer is None:
        optimizer = "differential_evolution"

    def cost_function(params: NDArray[np.float64]) -> NDArray[np.float64]:
        """Cost function for the curve fitting."""
        x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi = params
        modelled_OCV = _calc_full_cell_OCV(
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
        if fitting_target == "dQdV":
            model = np.gradient(SOC, modelled_OCV)
            truth = dSOCdV
        elif fitting_target == "dVdQ":
            model = np.gradient(modelled_OCV, SOC)
            truth = dVdSOC
        else:
            model = modelled_OCV
            truth = voltage
        return np.sum((model - truth) ** 2)

    if optimizer == "minimize":
        fitting_result = opt.minimize(cost_function, x_guess)
    elif optimizer == "differential_evolution":
        fitting_result = opt.differential_evolution(
            cost_function, bounds=[(0.75, 0.95), (0.2, 0.3), (0, 0.05), (0.85, 0.95)]
        )

    x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi = fitting_result.x
    (
        pe_capacity,
        ne_capacity,
        li_inventory,
    ) = _calc_electrode_capacities(x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi, cell_capacity)

    stoichiometry_limits = method.make_result(
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
    fitted_voltage = _calc_full_cell_OCV(
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
    fitted_OCV = method.make_result(
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


def _calc_electrode_capacities(
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


def _calc_full_cell_OCV(
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
    n_points = 10000
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


def quantify_degradation_modes(electrode_capacities: List[Result]) -> Result:
    """Quantify the change in degradation modes between at least two OCV fits.

    Args:
        electrode_capacities (List[Result]):
            A list of result objects containing the fitted electrode capacities of the
            cell. The first output of the fit_ocv method.

    Returns:
        Result:
            A result object containing the SOH, LAM_pe, LAM_ne, and LLI for each of the
            provided OCV fits.
    """
    method = BaseMethod(electrode_capacities)
    cell_capacity = method.variable("Cell Capacity [Ah]")
    pe_capacity = method.variable("Cathode Capacity [Ah]")
    ne_capacity = method.variable("Anode Capacity [Ah]")
    li_inventory = method.variable("Li Inventory [Ah]")
    SOH, LAM_pe, LAM_ne, LLI = _calculate_dma_parameters(
        cell_capacity, pe_capacity, ne_capacity, li_inventory
    )

    dma_result = method.make_result(
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


def _calculate_dma_parameters(
    cell_capacity: NDArray[np.float64],
    pe_capacity: NDArray[np.float64],
    ne_capacity: NDArray[np.float64],
    li_inventory: NDArray[np.float64],
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Calculate the DMA parameters.

    Args:
        pe_stoich_limits (NDArray[np.float64]): The cathode stoichiometry limits.
        ne_stoich_limits (NDArray[np.float64]): The anode stoichiometry limits.
        pe_capacity (NDArray[np.float64]): The cathode capacity.
        ne_capacity (NDArray[np.float64]): The anode capacity.
        li_inventory (NDArray[np.float64]): The lithium inventory.

    Returns:
        Tuple[float, float, float, float]: The SOH, LAM_pe, LAM_ne, and LLI.
    """
    SOH = cell_capacity / cell_capacity[0]
    LAM_pe = 1 - pe_capacity / pe_capacity[0]
    LAM_ne = 1 - ne_capacity / ne_capacity[0]
    LLI = 1 - li_inventory / li_inventory[0]
    return SOH, LAM_pe, LAM_ne, LLI
