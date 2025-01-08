"""Module for degradation mode analysis methods."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast

import numpy as np
import polars as pl
import ray
import sympy as sp
from numpy.typing import NDArray
from pydantic import ConfigDict, validate_call
from scipy import optimize
from scipy.interpolate import PPoly

import pyprobe.analysis.base.degradation_mode_analysis_functions as dma_functions
from pyprobe.analysis import smoothing, utils
from pyprobe.analysis.utils import AnalysisValidator
from pyprobe.pyprobe_types import FilterToCycleType, PyProBEDataType
from pyprobe.result import Result

logger = logging.getLogger(__name__)


def _get_gradient(
    any_function: Union[
        Callable[[NDArray[np.float64]], NDArray[np.float64]],
        PPoly,
        sp.Expr,
    ],
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Retrieve the gradient of the OCP function.

    Args:
        any_function: The OCP function.

    Returns:
        The gradient of the OCP.
    """
    if isinstance(any_function, PPoly):
        return any_function.derivative()
    elif isinstance(any_function, sp.Expr):
        free_symbols = any_function.free_symbols
        sto = free_symbols.pop()
        gradient = sp.diff(any_function, sto)
        return sp.lambdify(sto, gradient, "numpy")
    elif callable(any_function):

        def function_derivative(
            sto: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            """Numerically calculate the derivative."""
            return np.gradient(any_function(sto), sto)

        return function_derivative
    else:
        error_msg = (
            "OCP is not in a differentiable format. OCP must be a"
            " PPoly object, a sympy expression or a callable function with a "
            "single NDArray input and single NDArray output."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)


class _AbstractOCP(ABC):
    """An abstract class for open circuit potential data."""

    @property
    @abstractmethod
    def eval(self) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        """A callable function for ocp as a function of electrode stoichiometry."""
        pass

    @property
    @abstractmethod
    def grad(self) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        """The gradient of the OCP function."""
        pass


class OCP(_AbstractOCP):
    """A class for single-component electrode open circuit potential data.

    Args:
        ocp_function: The OCP function for the electrode.
    """

    def __init__(
        self,
        ocp_function: Union[
            Callable[[NDArray[np.float64]], NDArray[np.float64]],
            PPoly,
            sp.Expr,
        ],
    ) -> None:
        """Initialize the OCP object."""
        self.ocp_function = ocp_function

    @staticmethod
    def from_data(
        stoichiometry: NDArray[np.float64],
        ocp: NDArray[np.float64],
        interpolation_method: Literal["linear", "cubic", "Pchip", "Akima"] = "linear",
    ) -> "OCP":
        """Create an OCP object from stoichiometry and OCP data.

        Appends to the ocp list for the given electrode. Composite electrodes require
        multiple calls to this method to provide the OCP data for each component.

        Args:
            stoichiometry: The stoichiometry data.
            ocp: The OCP data.
            interpolation_method: The interpolation method to use. Defaults to "linear".

        Returns:
            The OCP object.
        """
        interpolator = {
            "linear": smoothing.linear_interpolator,
            "cubic": smoothing.cubic_interpolator,
            "Pchip": smoothing.pchip_interpolator,
            "Akima": smoothing.akima_interpolator,
        }[interpolation_method]
        return OCP(interpolator(stoichiometry, ocp))

    @staticmethod
    def from_expression(sympy_expression: sp.Expr) -> "OCP":
        """Create an OCP object from a sympy expression.

        Args:
            sympy_expression: A sympy expression for the OCP.

        Returns:
            The OCP object.
        """
        return OCP(sympy_expression)

    @property
    def eval(self) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        """A callable function for ocp as a function of electrode stoichiometry."""
        if not hasattr(self, "_eval"):
            if isinstance(self.ocp_function, sp.Expr):
                self._eval = sp.lambdify(
                    self.ocp_function.free_symbols.pop(), self.ocp_function, "numpy"
                )
            else:
                self._eval = self.ocp_function
        return self._eval

    @property
    def grad(self) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        """The gradient of the OCP function."""
        if not hasattr(self, "_grad"):
            self._grad = _get_gradient(self.ocp_function)
        return self._grad


class CompositeOCP(_AbstractOCP):
    """A class for composite electrode open circuit potential data.

    Args:
        ocp_list: A list of OCP functions for the composite electrode.
        ocp_vector: The OCP vector for the composite electrode.
    """

    def __init__(
        self,
        ocp_list: List[
            Union[
                Callable[[NDArray[np.float64]], NDArray[np.float64]],
                PPoly,
                sp.Expr,
            ]
        ],
        ocp_vector: NDArray[np.float64],
    ) -> None:
        """Initialize the CompositeOCP object."""
        self.ocp_list = ocp_list
        self.ocp_vector = ocp_vector
        self.comp_fraction = 0.5

    @staticmethod
    def from_data(
        stoichiometry_comp1: NDArray[np.float64],
        ocp_comp1: NDArray[np.float64],
        stoichiometry_comp2: NDArray[np.float64],
        ocp_comp2: NDArray[np.float64],
        interpolation_method: Literal["linear", "cubic", "Pchip", "Akima"] = "linear",
    ) -> "CompositeOCP":
        """Create a CompositeOCP object from stoichiometry and OCP data.

        Args:
            stoichiometry_comp1: The stoichiometry data for the first component.
            ocp_comp1: The OCP data for the first component.
            stoichiometry_comp2: The stoichiometry data for the second component.
            ocp_comp2: The OCP data for the second component.
            interpolation_method: The interpolation method to use. Defaults to "linear".
        """
        # Determine common voltage range for anode components
        composite_voltage_limits = (
            max(ocp_comp1.min(), ocp_comp2.min()),
            min(ocp_comp1.max(), ocp_comp2.max()),
        )
        # Filter valid indices for both components
        valid_indices_c1 = (ocp_comp1 >= composite_voltage_limits[0]) & (
            ocp_comp1 <= composite_voltage_limits[1]
        )
        valid_indices_c2 = (ocp_comp2 >= composite_voltage_limits[0]) & (
            ocp_comp2 <= composite_voltage_limits[1]
        )

        # Create a linearly spaced voltage series
        ocp_vector_composite = np.linspace(
            composite_voltage_limits[0], composite_voltage_limits[1], 10001
        )

        interpolator = {
            "linear": smoothing.linear_interpolator,
            "cubic": smoothing.cubic_interpolator,
            "Pchip": smoothing.pchip_interpolator,
            "Akima": smoothing.akima_interpolator,
        }[interpolation_method]

        ocp_list = [
            interpolator(
                ocp_comp1[valid_indices_c1], stoichiometry_comp1[valid_indices_c1]
            ),
            interpolator(
                ocp_comp2[valid_indices_c2], stoichiometry_comp2[valid_indices_c2]
            ),
        ]
        return CompositeOCP(ocp_list=ocp_list, ocp_vector=ocp_vector_composite)

    @property
    def eval(self) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        """A callable function for ocp as a function of electrode stoichiometry."""
        # Calculate the electrode stoichiometry vector
        x_composite = self.comp_fraction * self.ocp_list[0](self.ocp_vector) + (
            1 - self.comp_fraction
        ) * self.ocp_list[1](self.ocp_vector)
        return smoothing.linear_interpolator(x_composite, self.ocp_vector)

    @property
    def grad(self) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        """The gradient of the OCP function."""
        return _get_gradient(self.eval)


def _f_OCV(
    ocp_pe: Union[OCP, CompositeOCP],
    ocp_ne: Union[OCP, CompositeOCP],
    SOC: NDArray[np.float64],
    x_pe_lo: NDArray[np.float64],
    x_pe_hi: NDArray[np.float64],
    x_ne_lo: NDArray[np.float64],
    x_ne_hi: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate the full cell OCV as a function of SOC.

    Args:
        SOC: The full cell SOC.
        ocp_pe: The positive electrode OCP as a function of stoichiometry.
        ocp_ne: The negative electrode OCP as a function of stoichiometry.
        x_pe_lo: The cathode stoichiometry at lowest cell SOC.
        x_pe_hi: The cathode stoichiometry at highest cell SOC.
        x_ne_lo: The cathode stoichiometry at lowest cell SOC.
        x_ne_hi: The anode stoichiometry at highest cell SOC.

    Returns:
        A function to calculate the full cell OCV as a function of SOC.
    """
    # Calculate the stoichiometry at the given SOC for each electrode
    z_pe = x_pe_lo + (x_pe_hi - x_pe_lo) * SOC
    z_ne = x_ne_lo + (x_ne_hi - x_ne_lo) * SOC

    # Return the full cell OCV
    return ocp_pe.eval(z_pe) - ocp_ne.eval(z_ne)


def _f_grad_OCV(
    ocp_pe: Union[OCP, CompositeOCP],
    ocp_ne: Union[OCP, CompositeOCP],
    SOC: NDArray[np.float64],
    x_pe_lo: NDArray[np.float64],
    x_pe_hi: NDArray[np.float64],
    x_ne_lo: NDArray[np.float64],
    x_ne_hi: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate the full cell OCV gradient as a function of SOC.

    Derivative is calculated using the chain rule:
    d(OCV)/d(SOC) = d(OCV)/d(z_pe) * d(z_pe)/d(SOC)
                    - d(OCV)/d(z_ne) * d(z_ne)/d(SOC)

    Args:
        SOC: The full cell SOC.
        ocp_pe: The OCP function for the positive electrode.
        ocp_ne: The OCP function for the negative electrode.
        x_pe_lo: The cathode stoichiometry at lowest cell SOC.
        x_pe_hi: The cathode stoichiometry at highest cell SOC.
        x_ne_lo: The cathode stoichiometry at lowest cell SOC.
        x_ne_hi: The anode stoichiometry at highest cell SOC.

    Returns:
        A function to calculate the full cell OCV gradient as a function of SOC.
    """
    # Calculate the stoichiometry at the given SOC for each electrode
    z_pe = x_pe_lo + (x_pe_hi - x_pe_lo) * SOC
    z_ne = x_ne_lo + (x_ne_hi - x_ne_lo) * SOC

    # Calculate the gradient of electrode stoichiometry with respect to SOC
    d_z_pe = x_pe_hi - x_pe_lo
    d_z_ne = x_ne_hi - x_ne_lo

    # Calculate the full cell OCV gradient using the chain rule
    return ocp_pe.grad(z_pe) * d_z_pe - ocp_ne.grad(z_ne) * d_z_ne


def _build_objective_function(
    ocp_pe: Union[OCP, CompositeOCP],
    ocp_ne: Union[OCP, CompositeOCP],
    SOC: NDArray[np.float64],
    fitting_target_data: NDArray[np.float64],
    fitting_target: Literal["OCV", "dQdV", "dVdQ"],
    composite_pe: bool,
    composite_ne: bool,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Get the objective function for the OCV curve fitting.

    Args:
        ocp_pe: An object representing the positive electrode OCP.
        ocp_ne: An object representing the negative electrode OCP.
        SOC: The full cell SOC.
        fitting_target_data: The data to fit.
        fitting_target: The target for the curve fitting.
        composite_pe: Whether the positive electrode is composite.
        composite_ne: Whether the negative electrode is composite.
    """
    # Define the unwrap_params function based on the values of composite_pe and
    # composite_ne
    if not composite_pe and not composite_ne:

        def unwrap_params(
            params: Tuple[np.float64, ...],
        ) -> Tuple[
            float,
            float,
            float,
            float,
            Union[OCP, CompositeOCP],
            Union[OCP, CompositeOCP],
        ]:
            x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi = params
            return (
                x_pe_lo,
                x_pe_hi,
                x_ne_lo,
                x_ne_hi,
                ocp_pe,
                ocp_ne,
            )

    elif composite_pe and not composite_ne:

        def unwrap_params(
            params: Tuple[np.float64, ...],
        ) -> Tuple[
            float,
            float,
            float,
            float,
            Union[OCP, CompositeOCP],
            Union[OCP, CompositeOCP],
        ]:
            x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi, pe_frac = params
            updated_ocp_pe = cast(CompositeOCP, ocp_pe)
            updated_ocp_pe.comp_fraction = pe_frac
            return x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi, updated_ocp_pe, ocp_ne

    elif not composite_pe and composite_ne:

        def unwrap_params(
            params: Tuple[np.float64, ...],
        ) -> Tuple[
            float,
            float,
            float,
            float,
            Union[OCP, CompositeOCP],
            Union[OCP, CompositeOCP],
        ]:
            x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi, ne_frac = params
            updated_ocp_ne = cast(CompositeOCP, ocp_ne)
            updated_ocp_ne.comp_fraction = ne_frac
            return x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi, ocp_pe, updated_ocp_ne

    else:  # composite_pe and composite_ne are both True

        def unwrap_params(
            params: Tuple[np.float64, ...],
        ) -> Tuple[
            float,
            float,
            float,
            float,
            Union[OCP, CompositeOCP],
            Union[OCP, CompositeOCP],
        ]:
            x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi, pe_frac, ne_frac = params
            updated_ocp_pe = cast(CompositeOCP, ocp_pe)
            updated_ocp_ne = cast(CompositeOCP, ocp_ne)
            updated_ocp_pe.comp_fraction = pe_frac
            updated_ocp_ne.comp_fraction = ne_frac
            return x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi, updated_ocp_pe, updated_ocp_ne

    # Define the model function based on the fitting target
    if fitting_target == "OCV":

        def model_function(
            ocp_pe: Union[OCP, CompositeOCP],
            ocp_ne: Union[OCP, CompositeOCP],
            SOC: NDArray[np.float64],
            x_pe_lo: float,
            x_pe_hi: float,
            x_ne_lo: float,
            x_ne_hi: float,
        ) -> NDArray[np.float64]:
            return _f_OCV(
                ocp_pe,
                ocp_ne,
                SOC,
                x_pe_lo,
                x_pe_hi,
                x_ne_lo,
                x_ne_hi,
            )

    elif fitting_target == "dVdQ":

        def model_function(
            ocp_pe: Union[OCP, CompositeOCP],
            ocp_ne: Union[OCP, CompositeOCP],
            SOC: NDArray[np.float64],
            x_pe_lo: float,
            x_pe_hi: float,
            x_ne_lo: float,
            x_ne_hi: float,
        ) -> NDArray[np.float64]:
            return _f_grad_OCV(ocp_pe, ocp_ne, SOC, x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi)

    elif fitting_target == "dQdV":

        def model_function(
            ocp_pe: Union[OCP, CompositeOCP],
            ocp_ne: Union[OCP, CompositeOCP],
            SOC: NDArray[np.float64],
            x_pe_lo: float,
            x_pe_hi: float,
            x_ne_lo: float,
            x_ne_hi: float,
        ) -> NDArray[np.float64]:
            with np.errstate(divide="ignore", invalid="ignore"):
                model = 1 / _f_grad_OCV(
                    ocp_pe, ocp_ne, SOC, x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi
                )
                model[~np.isfinite(model)] = np.inf
            return model

    else:
        raise ValueError(f"Invalid fitting target: {fitting_target}")

    # Define the objective function using the built functions for collecting the
    # parameters and the model
    def _objective_function(
        params: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Objective function for the OCV curve fitting.

        Args:
            params: The fitting parameters.

        Returns:
            The residuals between the data and the fit.
        """
        (
            x_pe_lo,
            x_pe_hi,
            x_ne_lo,
            x_ne_hi,
            updated_ocp_pe,
            updated_ocp_ne,
        ) = unwrap_params(params)
        model = model_function(
            updated_ocp_pe, updated_ocp_ne, SOC, x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi
        )
        return np.sum((model - fitting_target_data) ** 2)

    return _objective_function


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def run_ocv_curve_fit(
    input_data: PyProBEDataType,
    ocp_pe: Union[OCP, CompositeOCP],
    ocp_ne: Union[OCP, CompositeOCP],
    fitting_target: Literal["OCV", "dQdV", "dVdQ"] = "OCV",
    optimizer: Literal["minimize", "differential_evolution"] = "minimize",
    optimizer_options: Dict[str, Any] = {
        "x0": np.array([0.9, 0.1, 0.1, 0.9]),
        "bounds": [(0, 1), (0, 1), (0, 1), (0, 1)],
    },
) -> Tuple[Result, Result]:
    """Fit half cell open circuit potential curves to full cell OCV data.

    Args:
        input_data: The input data for the analysis.
        ocp_pe: The positive electrode OCP in the form of a OCP or CompositeOCP object.
        ocp_ne: The negative electrode OCP in the form of a OCP or CompositeOCP object.
        fitting_target: The target for the curve fitting. Defaults to "OCV".
        optimizer: The optimization algorithm to use. Defaults to "minimize".
        optimizer_options:
            The options for the optimization algorithm. Defaults to
            {"x0": np.array([0.9, 0.1, 0.1, 0.9]),
                "bounds": [(0, 1), (0, 1), (0, 1), (0, 1)]}.
            Where x0 is the initial guess for the fit and bounds are the
            limits for the fit. The fitting parameters are ordered [x_pe_lo,
            x_pe_hi, x_ne_lo, x_ne_hi], where lo and hi indicate the stoichiometry
            limits at low and high full-cell SOC respectively.

    Returns:
        - The stoichiometry limits and electrode capacities.
        - The fitted OCV data.
    """
    if "SOC" in input_data.column_list:
        required_columns = ["Voltage [V]", "Capacity [Ah]", "SOC"]
        validator = AnalysisValidator(
            input_data=input_data, required_columns=required_columns
        )
        voltage, capacity, SOC = validator.variables
        cell_capacity = np.abs(np.ptp(capacity)) / np.abs(np.ptp(SOC))
    else:
        required_columns = ["Voltage [V]", "Capacity [Ah]"]
        validator = AnalysisValidator(
            input_data=input_data, required_columns=required_columns
        )
        voltage, capacity = validator.variables
        cell_capacity = np.abs(np.ptp(capacity))
        SOC = (capacity - capacity.min()) / cell_capacity

    dVdSOC = np.gradient(voltage, SOC)
    dSOCdV = np.gradient(SOC, voltage)

    fitting_target_data = {
        "OCV": voltage,
        "dQdV": dSOCdV,
        "dVdQ": dVdSOC,
    }[fitting_target]

    if isinstance(ocp_pe, CompositeOCP):
        composite_pe = True
    else:
        composite_pe = False
    if isinstance(ocp_ne, CompositeOCP):
        composite_ne = True
    else:
        composite_ne = False

    objective_function = _build_objective_function(
        ocp_pe,
        ocp_ne,
        SOC,
        fitting_target_data,
        fitting_target,
        composite_pe,
        composite_ne,
    )

    selected_optimizer = {
        "minimize": optimize.minimize,
        "differential_evolution": optimize.differential_evolution,
    }[optimizer]

    results = selected_optimizer(objective_function, **optimizer_options).x

    x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi = results[:4]
    if composite_pe:
        pe_frac = results[4]
        if composite_ne:
            ne_frac = results[5]
    elif composite_ne and not composite_pe:
        ne_frac = results[4]
    else:
        pe_frac = None
        ne_frac = None

    (
        pe_capacity,
        ne_capacity,
        li_inventory,
    ) = dma_functions.calc_electrode_capacities(
        x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi, cell_capacity
    )

    data_dict = {
        "x_pe low SOC": np.array([x_pe_lo]),
        "x_pe high SOC": np.array([x_pe_hi]),
        "x_ne low SOC": np.array([x_ne_lo]),
        "x_ne high SOC": np.array([x_ne_hi]),
        "Cell Capacity [Ah]": np.array([cell_capacity]),
        "Cathode Capacity [Ah]": np.array([pe_capacity]),
        "Anode Capacity [Ah]": np.array([ne_capacity]),
        "Li Inventory [Ah]": np.array([li_inventory]),
    }
    if composite_pe:
        data_dict["pe composite fraction"] = np.array([pe_frac])
    if composite_ne:
        data_dict["ne composite fraction"] = np.array([ne_frac])

    input_stoichiometry_limits = input_data.clean_copy(pl.DataFrame(data_dict))
    input_stoichiometry_limits.column_definitions = {
        "x_pe low SOC": "Positive electrode stoichiometry at lowest SOC point.",
        "x_pe high SOC": "Positive electrode stoichiometry at highest SOC point.",
        "x_ne low SOC": "Negative electrode stoichiometry at lowest SOC point.",
        "x_ne high SOC": "Negative electrode stoichiometry at highest SOC point.",
        "Cell Capacity": "Total cell capacity.",
        "Cathode Capacity": "Cathode capacity.",
        "Anode Capacity": "Anode capacity.",
        "Li Inventory": "Lithium inventory.",
    }

    if composite_pe:
        input_stoichiometry_limits.column_definitions["pe composite fraction"] = (
            "Fraction of composite cathode capacity attributed to first component."
        )
    if composite_ne:
        input_stoichiometry_limits.column_definitions["ne composite fraction"] = (
            "Fraction of composite anode capacity attributed to first component."
        )

    fitted_voltage = _f_OCV(
        ocp_pe,
        ocp_ne,
        SOC,
        x_pe_lo,
        x_pe_hi,
        x_ne_lo,
        x_ne_hi,
    )
    fitted_dVdSOC = _f_grad_OCV(
        ocp_pe,
        ocp_ne,
        SOC,
        x_pe_lo,
        x_pe_hi,
        x_ne_lo,
        x_ne_hi,
    )
    fitted_OCV = input_data.clean_copy(
        pl.DataFrame(
            {
                "Capacity [Ah]": capacity,
                "SOC": SOC,
                "Input Voltage [V]": voltage,
                "Fitted Voltage [V]": fitted_voltage,
                "Input dSOCdV [1/V]": dSOCdV,
                "Fitted dSOCdV [1/V]": 1 / fitted_dVdSOC,
                "Input dVdSOC [V]": dVdSOC,
                "Fitted dVdSOC [V]": fitted_dVdSOC,
            }
        )
    )
    fitted_OCV.column_definitions = {
        "SOC": "Cell state of charge.",
        "Voltage": "Fitted OCV values.",
    }

    return input_stoichiometry_limits, fitted_OCV


@validate_call
def quantify_degradation_modes(
    stoichiometry_limits_list: List[Result],
) -> Result:
    """Quantify the change in degradation modes between at least two OCV fits.

    Args:
        stoichiometry_limits_list: A list of Result objects containing the
        stoichiometry limits for the OCV fits.

    Returns:
        A result object containing the SOH, LAM_pe, LAM_ne, and LLI for each of
        the provided OCV fits.
    """
    required_columns = [
        "x_pe low SOC",
        "x_pe high SOC",
        "x_ne low SOC",
        "x_ne high SOC",
        "Cell Capacity [Ah]",
        "Cathode Capacity [Ah]",
        "Anode Capacity [Ah]",
        "Li Inventory [Ah]",
    ]
    for stoichiometry_limits in stoichiometry_limits_list:
        AnalysisValidator(
            input_data=stoichiometry_limits, required_columns=required_columns
        )
    x_pe_lo = utils.assemble_array(stoichiometry_limits_list, "x_pe low SOC")
    x_pe_hi = utils.assemble_array(stoichiometry_limits_list, "x_pe high SOC")
    x_ne_lo = utils.assemble_array(stoichiometry_limits_list, "x_ne low SOC")
    x_ne_hi = utils.assemble_array(stoichiometry_limits_list, "x_ne high SOC")

    cell_capacity = utils.assemble_array(
        stoichiometry_limits_list, "Cell Capacity [Ah]"
    )
    pe_capacity = utils.assemble_array(
        stoichiometry_limits_list, "Cathode Capacity [Ah]"
    )
    ne_capacity = utils.assemble_array(stoichiometry_limits_list, "Anode Capacity [Ah]")
    li_inventory = utils.assemble_array(stoichiometry_limits_list, "Li Inventory [Ah]")
    SOH, LAM_pe, LAM_ne, LLI = dma_functions.calculate_dma_parameters(
        cell_capacity, pe_capacity, ne_capacity, li_inventory
    )

    dma_result = stoichiometry_limits_list[0].clean_copy(
        pl.DataFrame(
            {
                "Index": np.arange(len(stoichiometry_limits_list)),
                "x_pe low SOC": x_pe_lo[:, 0],
                "x_pe high SOC": x_pe_hi[:, 0],
                "x_ne low SOC": x_ne_lo[:, 0],
                "x_ne high SOC": x_ne_hi[:, 0],
                "Cell Capacity [Ah]": cell_capacity[:, 0],
                "Cathode Capacity [Ah]": pe_capacity[:, 0],
                "Anode Capacity [Ah]": ne_capacity[:, 0],
                "Li Inventory [Ah]": li_inventory[:, 0],
                "SOH": SOH[:, 0],
                "LAM_pe": LAM_pe[:, 0],
                "LAM_ne": LAM_ne[:, 0],
                "LLI": LLI[:, 0],
            }
        )
    )
    dma_result.live_dataframe = dma_result.live_dataframe.with_columns(
        pl.col("Index").cast(pl.Int64)
    )
    dma_result.column_definitions = {
        "Index": "The index of the data point from the provided list of input data.",
        "SOH": "Cell capacity normalized to initial capacity.",
        "LAM_pe": "Loss of active material in positive electrode.",
        "LAM_ne": "Loss of active material in positive electrode.",
        "LLI": "Loss of lithium inventory.",
    }
    return dma_result


@ray.remote
def _run_ocv_curve_fit_with_index(
    index: int,
    input_data: PyProBEDataType,
    ocp_pe: Union[OCP, CompositeOCP],
    ocp_ne: Union[OCP, CompositeOCP],
    fitting_target: Literal["OCV", "dQdV", "dVdQ"],
    optimizer: Literal["minimize", "differential_evolution"],
    optimizer_options: Dict[str, Any],
) -> Tuple[int, Tuple[Result, Result]]:
    """Wrapper function for running the OCV curve fitting with an index."""
    result = run_ocv_curve_fit(
        input_data=input_data,
        ocp_pe=ocp_pe,
        ocp_ne=ocp_ne,
        fitting_target=fitting_target,
        optimizer=optimizer,
        optimizer_options=optimizer_options,
    )
    return index, result


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def run_batch_dma_parallel(
    input_data_list: List[PyProBEDataType],
    ocp_pe: Union[OCP, CompositeOCP],
    ocp_ne: Union[OCP, CompositeOCP],
    fitting_target: Literal["OCV", "dQdV", "dVdQ"] = "OCV",
    optimizer: Literal["minimize", "differential_evolution"] = "minimize",
    optimizer_options: Dict[str, Any] = {
        "x0": np.array([0.9, 0.1, 0.1, 0.9]),
        "bounds": [(0, 1), (0, 1), (0, 1), (0, 1)],
    },
) -> Tuple[Result, List[Result]]:
    """Fit half cell open circuit potential curves to full cell OCV data.

    DMA analysis is run in parallel across all provided input_data.

    Args:
        input_data_list: The list of input data for the analysis.
        ocp_pe: The positive electrode OCP in the form of a OCP or CompositeOCP object.
        ocp_ne: The negative electrode OCP in the form of a OCP or CompositeOCP object.
        fitting_target: The target for the curve fitting. Defaults to "OCV".
        optimizer: The optimization algorithm to use. Defaults to "minimize".
        optimizer_options:
            The options for the optimization algorithm. Defaults to
            {"x0": np.array([0.9, 0.1, 0.1, 0.9]),
                "bounds": [(0, 1), (0, 1), (0, 1), (0, 1)]}.
            Where x0 is the initial guess for the fit and bounds are the
            limits for the fit. The fitting parameters are ordered [x_pe_lo,
            x_pe_hi, x_ne_lo, x_ne_hi], where lo and hi indicate the stoichiometry
            limits at low and high full-cell SOC respectively.

    Returns:
        - Result: The stoichiometry limits, electrode capacities and
        degradation modes.
        - List[Result]: The fitted OCV data for each list item in input_data.
    """
    # Run the OCV curve fitting in parallel
    # Initialize Ray (only needs to happen once)
    try:
        if not ray.is_initialized():
            ray.init()
        logger.info(f"Ray using {ray.cluster_resources()['CPU']} CPUs")
        # Submit tasks to Ray
        futures = [
            _run_ocv_curve_fit_with_index.remote(
                index,
                input_data,
                ocp_pe,
                ocp_ne,
                fitting_target=fitting_target,
                optimizer=optimizer,
                optimizer_options=optimizer_options,
            )
            for index, input_data in enumerate(input_data_list)
        ]
        logger.info(f"Submitted {len(futures)} parallel tasks")
        # Get results and sort by index
        fit_results = ray.get(futures)
    finally:
        if ray.is_initialized():
            ray.shutdown()
    fit_results = [result for _, result in sorted(fit_results)]
    # Extract the results
    stoichiometry_limit_list = [result[0] for result in fit_results]
    fitted_OCVs = [result[1] for result in fit_results]

    for index, sto_limit in enumerate(stoichiometry_limit_list):
        sto_limit.live_dataframe = sto_limit.live_dataframe.with_columns(
            pl.lit(index).cast(pl.Int64).alias("Index")
        )

    dma_results = quantify_degradation_modes(stoichiometry_limit_list)
    return dma_results, fitted_OCVs


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def run_batch_dma_sequential(
    input_data_list: List[PyProBEDataType],
    ocp_pe: Union[OCP, CompositeOCP],
    ocp_ne: Union[OCP, CompositeOCP],
    fitting_target: Literal["OCV", "dQdV", "dVdQ"] = "OCV",
    optimizer: List[Literal["minimize", "differential_evolution"]] = ["minimize"],
    optimizer_options: List[Dict[str, Any]] = [
        {
            "x0": np.array([0.9, 0.1, 0.1, 0.9]),
            "bounds": [(0, 1), (0, 1), (0, 1), (0, 1)],
        }
    ],
    link_results: bool = False,
) -> Tuple[Result, List[Result]]:
    """Fit half cell open circuit potential curves to full cell OCV data.

    DMA analysis is run sequentially across all provided input_data.

    Args:
        input_data_list: The list of input data for the analysis.
        ocp_pe: The positive electrode OCP in the form of a OCP or CompositeOCP object.
        ocp_ne: The negative electrode OCP in the form of a OCP or CompositeOCP object.
        fitting_target: The target for the curve fitting. Defaults to "OCV".
        optimizer:
            A list of optimization algorithms to use. The length of the list determines
            how the optimizers will be applied:
                - Length = 1, the same optimizer will be used for all provided input
                data.
                - Length = 2, the first optimizer will be used for the first item in the
                input_data_list and the second optimizer will be used for the remaining
                items.
                - Length = n, the ith optimizer will be used for the ith item in the
                input_data_list. The length of the optimizer list must match the length
                of the input_data_list.
            Defaults to ["minimize"].
        optimizer_options:
            A list of dictionaries containing the options for the optimization
            algorithm. The length of the list determines how the options will be
            applied in the same manner as the optimizer argument. Defaults to
            [{"x0": np.array([0.9, 0.1, 0.1, 0.9]),
                "bounds": [(0, 1), (0, 1), (0, 1), (0, 1)]}].
        link_results: Whether to link the fitted stoichiometry limits from the previous
            input data list item to the next input data list item. Defaults to False.

    Returns:
        - Result: The stoichiometry limits, electrode capacities and
        degradation modes.
        - List[Result]: The fitted OCV data for each list item in input_data.
    """
    # Initialize the results list
    stoichiometry_limit_list: List[Result] = []
    fitted_OCVs: List[Result] = []
    # Run the OCV curve fitting sequentially
    for index, input_data in enumerate(input_data_list):
        if len(optimizer) == 1:
            current_optimizer = optimizer[0]
        else:
            current_optimizer = optimizer[index]
        if len(optimizer_options) == 1:
            current_optimizer_options = optimizer_options[0]
        else:
            current_optimizer_options = optimizer_options[index]
        if index > 0 and link_results:
            previous_stoichiometry_limits = stoichiometry_limit_list[-1]
            (
                previous_x_pe_lo,
                previous_x_pe_hi,
                previous_x_ne_lo,
                previous_x_ne_hi,
            ) = previous_stoichiometry_limits.get(
                "x_pe low SOC", "x_pe high SOC", "x_ne low SOC", "x_ne high SOC"
            )
            if current_optimizer == "minimize":
                current_optimizer_options["x0"] = np.array(
                    [
                        previous_x_pe_lo[0],
                        previous_x_pe_hi[0],
                        previous_x_ne_lo[0],
                        previous_x_ne_hi[0],
                    ]
                )
        stoichiometry_limits, fitted_OCV = run_ocv_curve_fit(
            input_data=input_data,
            ocp_pe=ocp_pe,
            ocp_ne=ocp_ne,
            fitting_target=fitting_target,
            optimizer=current_optimizer,
            optimizer_options=current_optimizer_options,
        )
        stoichiometry_limit_list.append(stoichiometry_limits)
        fitted_OCVs.append(fitted_OCV)
    dma_results = quantify_degradation_modes(stoichiometry_limit_list)
    return dma_results, fitted_OCVs


@validate_call
def average_ocvs(
    input_data: FilterToCycleType,
    discharge_filter: Optional[str] = None,
    charge_filter: Optional[str] = None,
) -> Result:
    """Average the charge and discharge OCV curves.

    Args:
        input_data: The input data for the analysis. Must be a PyProBE object that can
            be filtered to particular steps i.e. a Cycle object or higher.
        discharge_filter: The filter to apply to retrieve the discharge data from
            the input data. If left to default, the first discharge in the input
            data will be used.
        charge_filter: The filter to apply to retrieve the charge data from the
            input data. If left to default, the first charge in the input data will
            be used.

    Returns:
        A Result object containing the averaged OCV curve.
    """
    required_columns = ["Voltage [V]", "Capacity [Ah]", "SOC", "Current [A]"]

    AnalysisValidator(
        input_data=input_data,
        required_columns=required_columns,
    )
    if discharge_filter is None:
        discharge_result = input_data.discharge()
    else:
        discharge_result = eval(f"input_data.{discharge_filter}")
    if charge_filter is None:
        charge_result = input_data.charge()
    else:
        charge_result = eval(f"input_data.{charge_filter}")
    charge_SOC, charge_OCV, charge_current = charge_result.get(
        "SOC", "Voltage [V]", "Current [A]"
    )
    discharge_SOC, discharge_OCV, discharge_current = discharge_result.get(
        "SOC", "Voltage [V]", "Current [A]"
    )

    average_OCV = dma_functions.average_OCV_curves(
        charge_SOC,
        charge_OCV,
        charge_current,
        discharge_SOC,
        discharge_OCV,
        discharge_current,
    )

    return charge_result.clean_copy(
        pl.DataFrame(
            {
                "Voltage [V]": average_OCV,
                "Capacity [Ah]": charge_result.get("Capacity [Ah]"),
                "SOC": charge_SOC,
            }
        )
    )
