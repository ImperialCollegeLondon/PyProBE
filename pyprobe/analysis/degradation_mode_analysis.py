"""Module for degradation mode analysis methods."""

import concurrent.futures
import copy
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast

import numpy as np
import polars as pl
import sympy as sp
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy import optimize
from scipy.interpolate import PPoly

import pyprobe.analysis.base.degradation_mode_analysis_functions as dma_functions
from pyprobe.analysis import smoothing, utils
from pyprobe.analysis.utils import AnalysisValidator
from pyprobe.result import Result
from pyprobe.typing import FilterToCycleType, PyProBEDataType


def _get_gradient(
    any_function: Union[
        Callable[[NDArray[np.float64]], NDArray[np.float64]],
        PPoly,
        sp.Expr,
    ]
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Retrieve the gradient of the OCP function.

    Args:
        any_function: The OCP function.

    Returns:
        Callable[[NDArray[np.float64]], NDArray[np.float64]]:
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
        raise ValueError(
            "OCP is not in a differentiable format. OCP must be a"
            " PPoly object, a sympy expression or a callable function with a "
            "single NDArray input and single NDArray output."
        )


class OCP:
    """A class for single-component electrode open circuit potential data.

    Args:
        ocp_function (Union[Callable[[NDArray[np.float64]],
            NDArray[np.float64]], PPoly, sp.Expr]):
            The OCP function for the electrode.
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
            stoichiometry (NDArray[np.float64]): The stoichiometry data.
            ocp (NDArray[np.float64]): The OCP data.
            interpolation_method
                (Literal["linear", "cubic", "Pchip", "Akima"], optional):
                The interpolation method to use. Defaults to "linear".

        Returns:
            OCP: The OCP object.
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
            OCP: The OCP object.
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


class CompositeOCP:
    """A class for composite electrode open circuit potential data.

    Args:
        ocp_list (List[Union[Callable[[NDArray[np.float64]],
        NDArray[np.float64]], PPoly, sp.Expr]]):
            A list of OCP functions for the composite electrode.

        ocp_vector (NDArray[np.float64]): The OCP vector for the composite electrode.
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
            stoichiometry_comp1 (NDArray[np.float64]):
                The stoichiometry data for the first component.
            ocp_comp1 (NDArray[np.float64]): The OCP data for the first component.
            stoichiometry_comp2 (NDArray[np.float64]):
                The stoichiometry data for the second component.
            ocp_comp2 (NDArray[np.float64]): The OCP data for the second component.
            interpolation_method
                (Literal["linear", "cubic", "Pchip", "Akima"], optional):
                The interpolation method to use. Defaults to "linear".
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


# 1. Define the class as a Pydantic BaseModel.
class DMA(BaseModel):
    """A class for degradation mode analysis methods."""

    # 2. Define the input_data attribute, giving it a type
    input_data: PyProBEDataType
    """The input data for the degradation mode analysis."""

    ocp_pe: OCP | CompositeOCP
    """An object containing the positive electrode OCP functions."""

    ocp_ne: OCP | CompositeOCP
    """An object containing the negative electrode OCP functions."""

    # 3. Define the attributes that will be populated by the methods.
    stoichiometry_limits: Optional[Result] = None
    """The stoichiometry limits and electrode capacities."""
    fitted_OCV: Optional[Result] = None
    """The fitted OCV data."""
    dma_result: Optional[Result] = None
    """The degradation mode analysis results."""

    class Config:
        """Pydantic configuration settings."""

        arbitrary_types_allowed = True

    def downsample_ocv(
        self,
        sampling_interval: float,
        occurrence: Literal["first", "last", "middle"] = "first",
        time_column: str = "Time [s]",
    ) -> Result:
        """Downsample the OCV data to a specified sampling interval.

        This method updates the input_data attribute with the downsampled data, and
        returns a Result object. The logic is based on
        :func:`pyprobe.analysis.smoothing.downsample_data`, which also provides the
        logic for the :func:`pyprobe.analysis.smoothing.Smoothing.downsample` method.

        Args:
            sampling_interval (float): The sampling interval for the downsampled data.
            occurrence (Literal["first", "last", "middle"], optional):
                The occurrence to use when downsampling. Defaults to "first".
            time_column (str, optional):
                The column containing the time data. Defaults to "Time [s]".

        Returns:
            Result: A result object containing the downsampled OCV data.
        """
        required_columns = ["Voltage [V]", time_column]
        AnalysisValidator(input_data=self.input_data, required_columns=required_columns)
        self.input_data.base_dataframe = smoothing.downsample_data(
            df=self.input_data.base_dataframe,
            target="Voltage [V]",
            sampling_interval=sampling_interval,
            occurrence=occurrence,
            time_column=time_column,
        )
        return self.input_data

    def _f_OCV(
        self,
        SOC: NDArray[np.float64],
        x_pe_lo: NDArray[np.float64],
        x_pe_hi: NDArray[np.float64],
        x_ne_lo: NDArray[np.float64],
        x_ne_hi: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate the full cell OCV as a function of SOC.

        Args:
            SOC (NDArray[np.float64]): The full cell SOC.
            ocp_pe (Callable[[NDArray[np.float64]], NDArray[np.float64]]):
                The positive electrode OCP as a function of stoichiometry.
            ocp_ne (Callable[[NDArray[np.float64]], NDArray[np.float64]]):
                The negative electrode OCP as a function of stoichiometry.
            x_pe_lo (float): The cathode stoichiometry at lowest cell SOC.
            x_pe_hi (float): The cathode stoichiometry at highest cell SOC.
            x_ne_lo (float): The cathode stoichiometry at lowest cell SOC.
            x_ne_hi (float): The anode stoichiometry at highest cell SOC.

        Returns:
            Callable[[NDArray[np.float64]], NDArray[np.float64]]:
                A function to calculate the full cell OCV as a function of SOC.
        """
        # Calculate the stoichiometry at the given SOC for each electrode
        z_pe = x_pe_lo + (x_pe_hi - x_pe_lo) * SOC
        z_ne = x_ne_lo + (x_ne_hi - x_ne_lo) * SOC

        # Return the full cell OCV
        return self.ocp_pe.eval(z_pe) - self.ocp_ne.eval(z_ne)

    def _f_grad_OCV(
        self,
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
            SOC (NDArray[np.float64]): The full cell SOC.
            ocp_pe (Callable[[NDArray[np.float64]], NDArray[np.float64]]):
                The OCP function for the positive electrode.
            ocp_ne (Callable[[NDArray[np.float64]], NDArray[np.float64]]):
                The OCP function for the negative electrode.
            x_pe_lo (float): The cathode stoichiometry at lowest cell SOC.
            x_pe_hi (float): The cathode stoichiometry at highest cell SOC.
            x_ne_lo (float): The cathode stoichiometry at lowest cell SOC.
            x_ne_hi (float): The anode stoichiometry at highest cell SOC.

        Returns:
            Callable[[NDArray[np.float64]], NDArray[np.float64]]:
                A function to calculate the full cell OCV gradient as a function of SOC.
        """
        # Calculate the stoichiometry at the given SOC for each electrode
        z_pe = x_pe_lo + (x_pe_hi - x_pe_lo) * SOC
        z_ne = x_ne_lo + (x_ne_hi - x_ne_lo) * SOC

        # Calculate the gradient of electrode stoichiometry with respect to SOC
        d_z_pe = x_pe_hi - x_pe_lo
        d_z_ne = x_ne_hi - x_ne_lo

        # Calculate the full cell OCV gradient using the chain rule
        return self.ocp_pe.grad(z_pe) * d_z_pe - self.ocp_ne.grad(z_ne) * d_z_ne

    def _build_objective_function(
        self,
        SOC: NDArray[np.float64],
        fitting_target_data: NDArray[np.float64],
        fitting_target: Literal["OCV", "dQdV", "dVdQ"],
        composite_pe: bool,
        composite_ne: bool,
    ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        """Get the objective function for the OCV curve fitting.

        Args:
            SOC (NDArray[np.float64]): The full cell SOC.
            fitting_target_data (NDArray[np.float64]): The data to fit.
            fitting_target (Literal["OCV", "dQdV", "dVdQ"]):
                The target for the curve fitting.
            composite_pe (bool): Whether the positive electrode is composite.
            composite_ne (bool): Whether the negative electrode is composite.
        """
        # Define the unwrap_params function based on the values of composite_pe and
        # composite_ne
        if not composite_pe and not composite_ne:

            def unwrap_params(
                params: Tuple[np.float64, ...]
            ) -> Tuple[np.float64, np.float64, np.float64, np.float64]:
                x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi = params
                return (
                    x_pe_lo,
                    x_pe_hi,
                    x_ne_lo,
                    x_ne_hi,
                )

        elif composite_pe and not composite_ne:

            def unwrap_params(
                params: Tuple[np.float64, ...]
            ) -> Tuple[np.float64, np.float64, np.float64, np.float64]:
                x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi, pe_frac = params
                self.ocp_pe = cast(CompositeOCP, self.ocp_pe)
                self.ocp_pe.comp_fraction = pe_frac
                return x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi

        elif not composite_pe and composite_ne:

            def unwrap_params(
                params: Tuple[np.float64, ...]
            ) -> Tuple[np.float64, np.float64, np.float64, np.float64]:
                x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi, ne_frac = params
                self.ocp_ne = cast(CompositeOCP, self.ocp_ne)
                self.ocp_ne.comp_fraction = ne_frac
                return x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi

        else:  # composite_pe and composite_ne are both True

            def unwrap_params(
                params: Tuple[np.float64, ...]
            ) -> Tuple[np.float64, np.float64, np.float64, np.float64]:
                x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi, pe_frac, ne_frac = params
                self.ocp_pe = cast(CompositeOCP, self.ocp_pe)
                self.ocp_ne = cast(CompositeOCP, self.ocp_ne)
                self.ocp_pe.comp_fraction = pe_frac
                self.ocp_ne.comp_fraction = ne_frac
                return x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi

        # Define the model function based on the fitting target
        if fitting_target == "OCV":

            def model_function(
                SOC: NDArray[np.float64],
                x_pe_lo: float,
                x_pe_hi: float,
                x_ne_lo: float,
                x_ne_hi: float,
            ) -> NDArray[np.float64]:
                return self._f_OCV(
                    SOC,
                    x_pe_lo,
                    x_pe_hi,
                    x_ne_lo,
                    x_ne_hi,
                )

        elif fitting_target == "dVdQ":

            def model_function(
                SOC: NDArray[np.float64],
                x_pe_lo: float,
                x_pe_hi: float,
                x_ne_lo: float,
                x_ne_hi: float,
            ) -> NDArray[np.float64]:
                return self._f_grad_OCV(SOC, x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi)

        elif fitting_target == "dQdV":

            def model_function(
                SOC: NDArray[np.float64],
                x_pe_lo: float,
                x_pe_hi: float,
                x_ne_lo: float,
                x_ne_hi: float,
            ) -> NDArray[np.float64]:
                with np.errstate(divide="ignore", invalid="ignore"):
                    model = 1 / self._f_grad_OCV(
                        SOC, x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi
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
                params (NDArray[np.float64]): The fitting parameters.

            Returns:
                NDArray[np.float64]: The residuals between the data and the fit.
            """
            x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi = unwrap_params(params)
            model = model_function(SOC, x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi)
            return np.sum((model - fitting_target_data) ** 2)

        return _objective_function

    def run_ocv_curve_fit(
        self,
        fitting_target: Literal["OCV", "dQdV", "dVdQ"] = "OCV",
        optimizer: Literal["minimize", "differential_evolution"] = "minimize",
        optimizer_options: Dict[str, Any] = {
            "x0": np.array([0.9, 0.1, 0.1, 0.9]),
            "bounds": [(0, 1), (0, 1), (0, 1), (0, 1)],
        },
    ) -> Tuple[Result, Result]:
        """Fit half cell open circuit potential curves to full cell OCV data.

        Args:
            fitting_target (Literal["OCV", "dQdV", "dVdQ"], optional):
                The target for the curve fitting. Defaults to "OCV".
            optimizer (Literal["minimize", "differential_evolution"], optional):
                The optimization algorithm to use. Defaults to "minimize".
            optimizer_options (Dict[str, Any], optional):
                The options for the optimization algorithm. Defaults to
                {"x0": np.array([0.9, 0.1, 0.1, 0.9]),
                 "bounds": [(0, 1), (0, 1), (0, 1), (0, 1)]}.
                Where x0 is the initial guess for the fit and bounds are the
                limits for the fit. The fitting parameters are ordered [x_pe_lo,
                x_pe_hi, x_ne_lo, x_ne_hi], where lo and hi indicate the stoichiometry
                limits at low and high full-cell SOC respectively.

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

        fitting_target_data = {
            "OCV": voltage,
            "dQdV": dSOCdV,
            "dVdQ": dVdSOC,
        }[fitting_target]

        if isinstance(self.ocp_pe, CompositeOCP):
            composite_pe = True
        else:
            composite_pe = False
        if isinstance(self.ocp_ne, CompositeOCP):
            composite_ne = True
        else:
            composite_ne = False

        objective_function = self._build_objective_function(
            SOC, fitting_target_data, fitting_target, composite_pe, composite_ne
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

        self.stoichiometry_limits = self.input_data.clean_copy(pl.DataFrame(data_dict))
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

        if composite_pe:
            self.stoichiometry_limits.column_definitions[
                "pe composite fraction"
            ] = "Fraction of composite cathode capacity attributed to first component."
        if composite_ne:
            self.stoichiometry_limits.column_definitions[
                "ne composite fraction"
            ] = "Fraction of composite anode capacity attributed to first component."

        fitted_voltage = self._f_OCV(
            SOC,
            x_pe_lo,
            x_pe_hi,
            x_ne_lo,
            x_ne_hi,
        )
        fitted_dVdSOC = self._f_grad_OCV(
            SOC,
            x_pe_lo,
            x_pe_hi,
            x_ne_lo,
            x_ne_hi,
        )
        self.fitted_OCV = self.input_data.clean_copy(
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
        self.fitted_OCV.column_definitions = {
            "SOC": "Cell state of charge.",
            "Voltage [V]": "Fitted OCV values.",
        }

        return self.stoichiometry_limits, self.fitted_OCV

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

        Based on the code from :footcite:t:`Kirkaldy_batteryDAT_2023`. The fitting
        algorithm is based on the scipy.optimize.minimize function.

        Args:
            x_ne (NDArray[np.float64]): The anode stoichiometry data.
            x_pe (NDArray[np.float64]): The cathode stoichiometry data.
            ocp_ne (NDArray[np.float64]): The anode OCP data.
            ocp_pe (NDArray[np.float64]): The cathode OCP data.
            x_guess (NDArray[np.float64]): The initial guess for the fit.
            fitting_target (str, optional):
                The target for the curve fitting. Defaults to "OCV".
            optimizer (str, optional):
                The optimization algorithm to use. Defaults to None. If None, the
                optimizer will be set to scipy.optimize.minimize for
                fitting_target="OCV" and scipy.optimize.differential_evolution for
                fitting_target="dQdV" or "dVdQ".

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
            pl.DataFrame(
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
            pl.DataFrame(
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
            pl.DataFrame(
                {
                    "SOH": SOH[:, 0],
                    "LAM_pe": LAM_pe[:, 0],
                    "LAM_ne": LAM_ne[:, 0],
                    "LLI": LLI[:, 0],
                }
            )
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
        ocp_pe: OCP,
        ocp_ne: OCP,
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
            pl.DataFrame(
                {
                    "Voltage [V]": average_OCV,
                    "Capacity [Ah]": charge_result.get_only("Capacity [Ah]"),
                    "SOC": charge_SOC,
                }
            )
        )
        return DMA(input_data=average_result, ocp_pe=ocp_pe, ocp_ne=ocp_ne)


def _run_ocv_curve_fit_with_index(
    index: int,
    dma_object: DMA,
    fitting_target: Literal["OCV", "dQdV", "dVdQ"],
    optimizer: Literal["minimize", "differential_evolution"],
    optimizer_options: Dict[str, Any],
) -> Tuple[int, Tuple[Result, Result]]:
    """Wrapper function for running the OCV curve fitting with an index."""
    result = dma_object.run_ocv_curve_fit(
        fitting_target=fitting_target,
        optimizer=optimizer,
        optimizer_options=optimizer_options,
    )
    return index, result


def _quantify_degradation_modes_with_index(
    index: int, dma_object: DMA, reference_stoichiometry_limits: Result
) -> Tuple[int, Result]:
    result = dma_object.quantify_degradation_modes(
        reference_stoichiometry_limits=reference_stoichiometry_limits,
    )
    return index, result


class BatchDMA(BaseModel):
    """A class for running DMA analysis across multiple input data sets."""

    input_data: List[PyProBEDataType]
    """The a list of input data for the degradation mode analysis."""

    ocp_pe: OCP

    ocp_ne: OCP

    fitted_OCV: Optional[List[Result]] = None
    """A list of the fitted OCV data."""

    batch_DMA_results: Optional[Result] = None
    """The result of the DMA analysis across all provided input_data.

    Contains the stoichiometry limits, SOH, LAM_pe, LAM_ne, and LLI for each of the
    provided OCV fits.
    """

    class Config:
        """Pydantic configuration settings."""

        arbitrary_types_allowed = True

    @property
    def dma_objects(self) -> List[DMA]:
        """Return the DMA objects."""
        if not hasattr(self, "_dma_objects"):
            self._dma_objects = [
                DMA(input_data=data, ocp_ne=self.ocp_ne, ocp_pe=self.ocp_pe)
                for data in self.input_data
            ]
        return self._dma_objects

    def downsample_ocv(
        self,
        sampling_interval: float,
        occurrence: Literal["first", "last", "middle"] = "first",
        time_column: str = "Time [s]",
    ) -> List[Result]:
        """Downsample the OCV data to a specified sampling interval.

        This method updates the input_data attribute with the downsampled data, and
        returns a Result object. The logic is based on
        :func:`pyprobe.analysis.smoothing.downsample_data`, which also provides the
        logic for the :func:`pyprobe.analysis.smoothing.Smoothing.downsample` method.

        Args:
            sampling_interval (float): The sampling interval for the downsampled data.
            occurrence (Literal["first", "last", "middle"], optional):
                The occurrence to use when downsampling. Defaults to "first".
            time_column (str, optional):
                The column containing the time data. Defaults to "Time [s]".

        Returns:
            List[Result]: A list of result objects containing the downsampled OCV data.
        """
        return [
            dma_object.downsample_ocv(
                sampling_interval=sampling_interval,
                occurrence=occurrence,
                time_column=time_column,
            )
            for dma_object in self.dma_objects
        ]

    def run_batch_dma_parallel(
        self,
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
            fitting_target (Literal["OCV", "dQdV", "dVdQ"], optional):
                The target for the curve fitting. Defaults to "OCV".
            optimizer (Literal["minimize", "differential_evolution"], optional):
                The optimization algorithm to use. Defaults to "minimize".
            optimizer_options (Dict[str, Any], optional):
                The options for the optimization algorithm. Defaults to
                {"x0": np.array([0.9, 0.1, 0.1, 0.9]),
                 "bounds": [(0, 1), (0, 1), (0, 1), (0, 1)]}.
                Where x0 is the initial guess for the fit and bounds are the
                limits for the fit. The fitting parameters are ordered [x_pe_lo,
                x_pe_hi, x_ne_lo, x_ne_hi], where lo and hi indicate the stoichiometry
                limits at low and high full-cell SOC respectively.

        Returns:
            Tuple[Result, List[Result]]:
                - Result: The stoichiometry limits, electrode capacities and
                degradation modes.
                - List[Result]: The fitted OCV data for each list item in input_data.
        """
        # Run the OCV curve fitting in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            fit_futures = [
                executor.submit(
                    _run_ocv_curve_fit_with_index,
                    index,
                    dma_object,
                    fitting_target=fitting_target,
                    optimizer=optimizer,
                    optimizer_options=optimizer_options,
                )
                for index, dma_object in enumerate(self.dma_objects)
            ]
        # collate the results
        fit_results = [
            future.result() for future in concurrent.futures.as_completed(fit_futures)
        ]

        # Extract the results
        fit_index = [result[0] for result in fit_results]
        stoichiometry_limits = [result[1][0] for result in fit_results]
        fitted_OCVs = [result[1][1] for result in fit_results]
        sorted_fitted_ocvs = copy.deepcopy(fitted_OCVs)
        fit_0_position = -1
        # Update the DMA objects with the results
        for position, index in enumerate(fit_index):
            if index == 0:
                fit_0_position = position
            sorted_fitted_ocvs[index] = fitted_OCVs[position]
            stoichiometry_limits[position].base_dataframe = stoichiometry_limits[
                position
            ].base_dataframe.with_columns(pl.lit(index).alias("Index"))
            self.dma_objects[index].stoichiometry_limits = stoichiometry_limits[
                position
            ]
            self.dma_objects[index].fitted_OCV = fitted_OCVs[position]

        # Run the DMA quantification in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            if self.dma_objects[0].stoichiometry_limits is None:
                raise ValueError(
                    "No stoichiometry limits have been calculated for the reference "
                    "data point."
                )
            dma_futures = [
                executor.submit(
                    _quantify_degradation_modes_with_index,
                    index,
                    dma_object,
                    reference_stoichiometry_limits=self.dma_objects[
                        0
                    ].stoichiometry_limits,
                )
                for index, dma_object in enumerate(self.dma_objects)
            ]
        # collate the results
        all_results = [
            future.result() for future in concurrent.futures.as_completed(dma_futures)
        ]

        # Extract the results
        dma_index = [result[0] for result in all_results]
        dma_results = [result[1] for result in all_results]

        # Update the DMA objects with the results
        for position, index in enumerate(dma_index):
            if index == 0:
                dma_0_position = position
            dma_results[position].base_dataframe = dma_results[
                position
            ].base_dataframe.with_columns(pl.lit(index).alias("Index"))
            self.dma_objects[index].dma_result = dma_results[position]
            # reduce the dataframe down to only the last row (the row relevant to the
            # data point)
            dma_results[position].base_dataframe = dma_results[
                position
            ].base_dataframe.tail(1)

        # compile the results into single result objects
        all_stoichiometry_limits = copy.deepcopy(stoichiometry_limits[fit_0_position])
        all_dma_results = copy.deepcopy(dma_results[dma_0_position])
        all_stoichiometry_limits.extend(
            [item for i, item in enumerate(stoichiometry_limits) if i != fit_0_position]
        )
        all_dma_results.extend(
            [item for i, item in enumerate(dma_results) if i != dma_0_position]
        )
        all_final_results = all_stoichiometry_limits
        all_final_results.join(all_dma_results, on="Index", how="left")
        all_final_results.base_dataframe = all_final_results.base_dataframe.sort(
            "Index"
        )
        all_final_results.define_column(
            "Index", "The index of the data point from the provided list of input data."
        )
        self.batch_DMA_results = all_final_results
        return all_final_results, sorted_fitted_ocvs
