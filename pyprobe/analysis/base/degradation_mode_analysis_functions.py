"""A module containing functions for degradation mode analysis."""

from typing import Tuple

import numpy as np
import scipy.interpolate as interp
import scipy.optimize as opt
from numpy.typing import NDArray
from scipy.interpolate import interp1d


def ocv_curve_fit(
    SOC: NDArray[np.float64],
    fitting_target_data: NDArray[np.float64],
    x_pe: NDArray[np.float64],
    ocp_pe: NDArray[np.float64],
    x_ne: NDArray[np.float64],
    ocp_ne: NDArray[np.float64],
    fitting_target: str,
    optimizer: str,
    x_guess: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Fit half cell open circuit potential curves to full cell OCV data.

    Args:
        SOC (NDArray[np.float64]): The full cell SOC.
        fitting_target_data (NDArray[np.float64]): The full cell OCV data.
        x_pe (NDArray[np.float64]): The cathode stoichiometry data.
        ocp_pe (NDArray[np.float64]): The cathode OCP data.
        x_ne (NDArray[np.float64]): The anode stoichiometry data.
        ocp_ne (NDArray[np.float64]): The anode OCP data.
        fitting_target (str): The target for the curve fitting.
        optimizer (str): The optimization algorithm to use.
        x_guess (NDArray[np.float64]): The initial guess for the fitting parameters.

    Returns:
        NDArray: The optimized fitting parameters.
    """

    def cost_function(params: NDArray[np.float64]) -> NDArray[np.float64]:
        """Cost function for the curve fitting."""
        x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi = params
        modelled_OCV = calc_full_cell_OCV(
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
        elif fitting_target == "dVdQ":
            model = np.gradient(modelled_OCV, SOC)
        else:
            model = modelled_OCV
        return np.sum((model - fitting_target_data) ** 2)

    if optimizer == "minimize":
        fitting_result = opt.minimize(cost_function, x_guess, bounds=[(0, 1)] * 4)
    elif optimizer == "differential_evolution":
        fitting_result = opt.differential_evolution(
            cost_function, bounds=[(0.75, 0.95), (0.2, 0.3), (0, 0.05), (0.85, 0.95)]
        )

    return fitting_result.x


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
        cell_capacity (float): The cell capacity.

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


def calc_full_cell_OCV_composite(
    SOC: NDArray[np.float64],
    z_pe_lo: float,
    z_pe_hi: float,
    z_ne_lo: float,
    z_ne_hi: float,
    x_pe: NDArray[np.float64],
    ocp_pe: NDArray[np.float64],
    x_c1: NDArray[np.float64],
    ocp_c1: NDArray[np.float64],
    x_c2: NDArray[np.float64],
    ocp_c2: NDArray[np.float64],
    comp1_frac: float,
) -> NDArray[np.float64]:
    """Calculate the full cell OCV.

    Args:
        SOC (NDArray[np.float64]): The full cell SOC.
        z_pe_lo (float): The cathode stoichiometry at the lowest cell SOC.
        z_pe_hi (float): The cathode stoichiometry at the highest cell SOC.
        z_ne_lo (float): The anode stoichiometry at the lowest cell SOC.
        z_ne_hi (float): The anode stoichiometry at the highest cell SOC.
        x_pe (NDArray[np.float64]): The cathode stoichiometry data.
        ocp_pe (NDArray[np.float64]): The cathode OCP data.
        x_c1 (NDArray[np.float64]): The anode component 1 stoichiometry data.
        ocp_c1 (NDArray[np.float64]): The anode component 1 OCP data.
        x_c2 (NDArray[np.float64]): The anode component 2 stoichiometry data.
        ocp_c2 (NDArray[np.float64]): The anode component 2 OCP data.
        comp1_frac (float): The fraction of the first anode component.

    Returns:
        NDArray[np.float64]: The full cell OCV.
    """
    n_points = 10001

    # Determine common voltage range for anode components
    V_upper = min(ocp_c1.max(), ocp_c2.max())
    V_lower = max(ocp_c1.min(), ocp_c2.min())

    # Create a linearly spaced voltage series
    ocp_composite = np.linspace(V_lower, V_upper, n_points)

    # Filter valid indices for both components
    valid_indices_c1 = (ocp_c1 >= V_lower) & (ocp_c1 <= V_upper)
    valid_indices_c2 = (ocp_c2 >= V_lower) & (ocp_c2 <= V_upper)

    # Create interpolating functions for the anode components
    interp_c1 = interp1d(
        ocp_c1[valid_indices_c1],
        x_c1[valid_indices_c1] * comp1_frac,
        bounds_error=False,
        fill_value="extrapolate",
    )
    interp_c2 = interp1d(
        ocp_c2[valid_indices_c2],
        x_c2[valid_indices_c2] * (1 - comp1_frac),
        bounds_error=False,
        fill_value="extrapolate",
    )

    # Interpolate values for anode stoichiometry
    x_c1_interp = interp_c1(ocp_composite)
    x_c2_interp = interp_c2(ocp_composite)
    x_composite = x_c1_interp + x_c2_interp

    # Create linearly spaced values for stoichiometry
    z_pe = np.linspace(z_pe_lo, z_pe_hi, n_points)
    z_ne = np.linspace(z_ne_lo, z_ne_hi, n_points)
    z_ne_clipped = np.clip(
        z_ne, x_composite.min(), x_composite.max()
    )  # Measured NE cap. is within the composite cap. range

    # Interpolate the OCP data
    OCP_pe = interp1d(x_pe, ocp_pe, fill_value="extrapolate")(z_pe)
    OCP_ne = interp1d(x_composite, ocp_composite, fill_value="extrapolate")(
        z_ne_clipped
    )

    # Calculate the full cell OCV
    OCV = OCP_pe - OCP_ne

    # Interpolate the final OCV with the original SOC vector
    cell_ocv = interp1d(
        np.linspace(0, 1, n_points), OCV, bounds_error=False, fill_value="extrapolate"
    )(SOC)

    return SOC, cell_ocv


def calculate_dma_parameters(
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
        cell_capacity: The cell capacity.
        pe_capacity: The cathode capacity.
        ne_capacity: The anode capacity.
        li_inventory: The lithium inventory.

    Returns:
        Tuple[float, float, float, float]: The SOH, LAM_pe, LAM_ne, and LLI.
    """
    SOH = cell_capacity / cell_capacity[0]
    LAM_pe = 1 - pe_capacity / pe_capacity[0]
    LAM_ne = 1 - ne_capacity / ne_capacity[0]
    LLI = 1 - li_inventory / li_inventory[0]
    return SOH, LAM_pe, LAM_ne, LLI


def average_OCV_curves(
    charge_SOC: NDArray[np.float64],
    charge_OCV: NDArray[np.float64],
    charge_current: NDArray[np.float64],
    discharge_SOC: NDArray[np.float64],
    discharge_OCV: NDArray[np.float64],
    discharge_current: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Average the charge and discharge OCV curves.

    Args:
        charge_SOC: The charge SOC.
        charge_OCV: The charge OCV.
        charge_current: The charge current.
        discharge_SOC: The discharge SOC.
        discharge_OCV: The discharge OCV.
        discharge_current: The discharge current.

    Returns:
        Tuple[NDArray[np.float64], NDArray[np.float64]]:
            An average between the charge and discharge OCV curves.
    """
    f_discharge_OCV = interp.interp1d(discharge_SOC, discharge_OCV, kind="linear")
    f_discharge_current = interp.interp1d(
        discharge_SOC, discharge_current, kind="linear"
    )
    discharge_OCV = f_discharge_OCV(charge_SOC)
    discharge_current = f_discharge_current(charge_SOC)

    return ((charge_current * discharge_OCV) - (discharge_current * charge_OCV)) / (
        charge_current - discharge_current
    )
