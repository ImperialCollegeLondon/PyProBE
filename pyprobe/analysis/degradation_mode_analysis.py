"""Module for degradation mode analysis methods."""

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import polars as pl
import sympy as sp
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy import optimize
from scipy.interpolate import PPoly

import pyprobe.analysis.base.degradation_mode_analysis_functions as dma_functions
from pyprobe.analysis import utils
from pyprobe.analysis.utils import AnalysisValidator
from pyprobe.result import Result
from pyprobe.typing import FilterToCycleType, PyProBEDataType


# 1. Define the class as a Pydantic BaseModel.
class DMA(BaseModel):
    """A class for degradation mode analysis methods."""

    # 2. Define the input_data attribute, giving it a type
    input_data: PyProBEDataType
    """The input data for the degradation mode analysis."""

    # 3. Define the attributes that will be populated by the methods.
    stoichiometry_limits: Optional[Result] = None
    """The stoichiometry limits and electrode capacities."""
    fitted_OCV: Optional[Result] = None
    """The fitted OCV data."""
    dma_result: Optional[Result] = None
    """The degradation mode analysis results."""

    def _ocp_derivative(
        self, ocp_list: List[None | Callable[[NDArray], NDArray]]
    ) -> List[Callable[[NDArray], NDArray]]:
        """Calculate the derivative of each OCP.

        Args:
            ocp (Callable[[NDArray], NDArray]):
                The OCP function. Must be a differentiable function. Currently supported
                formats are scipy.interpolate.PPoly objects (from utils.interpolators)
                or sympy expressions.

        Returns:
            Callable[[NDArray], NDArray]: The derivative of the OCP.
        """

        def _check_free_symbols(free_symbols: set[sp.Symbol]) -> sp.Symbol:
            if len(free_symbols) == 1:
                return free_symbols.pop()
            else:
                raise ValueError(
                    "OCP must be a function of a single variable, " "the stoichiometry."
                )

        derivatives = []
        for ocp in ocp_list:
            if isinstance(ocp, PPoly):
                derivatives.append(ocp.derivative())
            elif isinstance(ocp, sp.Expr):
                free_symbols = ocp.free_symbols
                sto = _check_free_symbols(free_symbols)
                gradient = sp.diff(ocp, sto)
                derivatives.append(sp.lambdify(sto, gradient, "numpy"))
            else:
                raise ValueError(
                    "OCP is not in a differentiable format. OCP must be a"
                    " PPoly object or a sympy expression."
                )
        return derivatives

    @property
    def d_ocp_pe(self) -> List[Callable[[NDArray], NDArray]]:
        """Return the derivative of the positive electrode OCP.

        Returns:
            List[Callable[[NDArray], NDArray]]: The derivative of the positive
            electrode OCP.
        """
        return self._ocp_derivative(self._ocp_pe)

    @property
    def d_ocp_ne(self) -> List[Callable[[NDArray], NDArray]]:
        """Return the derivative of the negative electrode OCP.

        Returns:
            List[Callable[[NDArray], NDArray]]: The derivative of the negative
            electrode OCP.
        """
        return self._ocp_derivative(self._ocp_ne)

    @property
    def ocp_pe(
        self,
    ) -> List[Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]]]:
        """Return a list of positive electrode OCPs as a function of stoichiometry.

        Returns:
            List[Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]]]:
                A list of positive electrode OCPs as a function of stoichiometry.
        """
        if not hasattr(self, "_f_ocp_pe"):
            self._f_ocp_pe = []
            for ocp in self._ocp_pe:
                if isinstance(ocp, sp.Expr):
                    self._f_ocp_pe.append(
                        sp.lambdify(ocp.free_symbols.pop(), ocp, "numpy")
                    )
                else:
                    self._f_ocp_pe.append(ocp)
        return self._f_ocp_pe

    @ocp_pe.setter
    def ocp_pe(
        self,
        value: List[Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]]],
    ) -> None:
        """Set the OCP data for the positive electrode."""
        self._f_ocp_pe = value
        self._ocp_pe = value

    @property
    def ocp_ne(
        self,
    ) -> List[Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]]]:
        """Return a list of negative electrode OCPs as a function of stoichiometry.

        Returns:
            List[Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]]]:
                A list of negative electrode OCPs as a function of stoichiometry.
        """
        if not hasattr(self, "_f_ocp_ne"):
            self._f_ocp_ne = []
            for ocp in self._ocp_ne:
                if isinstance(ocp, sp.Expr):
                    self._f_ocp_ne.append(
                        sp.lambdify(ocp.free_symbols.pop(), ocp, "numpy")
                    )
                else:
                    self._f_ocp_ne.append(ocp)
        return self._f_ocp_ne

    @ocp_ne.setter
    def ocp_ne(
        self,
        value: List[Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]]],
    ) -> None:
        """Set the OCP data for the negative electrode."""
        self._f_ocp_ne = value
        self._ocp_ne = value

    def set_ocp_from_data(
        self,
        stoichiometry: NDArray[np.float64],
        ocp: NDArray[np.float64],
        electrode: Literal["pe", "ne"],
        interpolation_method: Literal["linear", "cubic", "Pchip", "Akima"] = "linear",
        component_index: int = 0,
        total_electrode_components: int = 1,
    ) -> None:
        """Provide the OCP data for a given electrode.

        Args:
            stoichiometry (NDArray[np.float64]): The stoichiometry data.
            ocp (NDArray[np.float64]): The OCP data.
            electrode (Literal["pe", "ne"]): The electrode to set the OCP data for.
            interpolation_method
                (Literal["linear", "cubic", "Pchip", "Akima"], optional):
                The interpolation method to use. Defaults to "linear".
            component_index (int, optional):
                The index of the electrode component to set the OCP data for.
                Defaults to 0.
            total_electrode_components (int, optional):
                The total number of electrode components. Defaults to 1.
        """
        interpolator = utils.interpolators[interpolation_method](x=stoichiometry, y=ocp)
        if electrode == "pe":
            if not hasattr(self, "_ocp_pe"):
                self._ocp_pe = [None] * total_electrode_components
            self._ocp_pe[component_index] = interpolator
            if hasattr(self, "_f_ocp_pe"):
                del self._f_ocp_pe
        elif electrode == "ne":
            if not hasattr(self, "_ocp_ne"):
                self._ocp_ne = [None] * total_electrode_components
            self._ocp_ne[component_index] = interpolator
            if hasattr(self, "_f_ocp_ne"):
                del self._f_ocp_ne

    def set_ocp_from_expression(
        self,
        ocp: sp.Expr,
        electrode: Literal["pe", "ne"],
        component_index: int = 0,
        total_electrode_components: int = 1,
    ) -> None:
        """Provide the OCP data for a given electrode.

        Args:
            ocp (sp.Expr): _description_
            electrode (Literal[&quot;pe&quot;, &quot;ne&quot;]): _description_
            component_index (int, optional): _description_. Defaults to 0.
            total_electrode_components (int, optional): _description_. Defaults to 1.
        """
        if electrode == "pe":
            if not hasattr(self, "_ocp_pe"):
                self._ocp_pe = [None] * total_electrode_components
            self._ocp_pe[component_index] = ocp
        elif electrode == "ne":
            if not hasattr(self, "_ocp_ne"):
                self._ocp_ne = [None] * total_electrode_components
            self._ocp_ne[component_index] = ocp

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
            x_pe_lo (float): The cathode stoichiometry at lowest cell SOC.
            x_pe_hi (float): The cathode stoichiometry at highest cell SOC.
            x_ne_lo (float): The cathode stoichiometry at lowest cell SOC.
            x_ne_hi (float): The anode stoichiometry at highest cell SOC.

        Returns:
            Callable[[NDArray[np.float64]], NDArray[np.float64]]:
                A function to calculate the full cell OCV as a function of SOC.
        """
        if self.ocp_pe[0] is None:
            raise ValueError("Positive electrode OCP data not provided.")
        if self.ocp_ne[0] is None:
            raise ValueError("Negative electrode OCP data not provided.")
        # Calculate the stoichiometry at the given SOC for each electrode
        z_pe = x_pe_lo + (x_pe_hi - x_pe_lo) * SOC
        z_ne = x_ne_lo + (x_ne_hi - x_ne_lo) * SOC

        # Return the full cell OCV
        return self.ocp_pe[0](z_pe) - self.ocp_ne[0](z_ne)

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
            x_pe_lo (float): The cathode stoichiometry at lowest cell SOC.
            x_pe_hi (float): The cathode stoichiometry at highest cell SOC.
            x_ne_lo (float): The cathode stoichiometry at lowest cell SOC.
            x_ne_hi (float): The anode stoichiometry at highest cell SOC.

        Returns:
            Callable[[NDArray[np.float64]], NDArray[np.float64]]:
                A function to calculate the full cell OCV gradient as a function of SOC.
        """
        if self._ocp_pe[0] is None:
            raise ValueError("Positive electrode OCP data not provided.")
        if self._ocp_ne[0] is None:
            raise ValueError("Negative electrode OCP data not provided.")
        # Calculate the stoichiometry at the given SOC for each electrode
        z_pe = x_pe_lo + (x_pe_hi - x_pe_lo) * SOC
        z_ne = x_ne_lo + (x_ne_hi - x_ne_lo) * SOC

        # Calculate the gradient of electrode stoichiometry with respect to SOC
        d_z_pe = x_pe_hi - x_pe_lo
        d_z_ne = x_ne_hi - x_ne_lo

        # Calculate the full cell OCV gradient using the chain rule
        return self.d_ocp_pe[0](z_pe) * d_z_pe - self.d_ocp_ne[0](z_ne) * d_z_ne

    def _curve_fit_ocv(
        self,
        SOC: NDArray[np.float64],
        fitting_target_data: NDArray[np.float64],
        fitting_target: Literal["OCV", "dQdV", "dVdQ"],
        optimizer: Literal["minimize", "differential_evolution"],
        optimizer_options: Dict[str, Any],
    ) -> NDArray[np.float64]:
        """Fit half cell open circuit potential curves to full cell OCV data.

        Args:
            SOC (NDArray[np.float64]): The full cell SOC.
            fitting_target_data (NDArray[np.float64]): The data to fit.
            fitting_target (Literal["OCV", "dQdV", "dVdQ"]):
                The target for the curve fitting.
            optimizer (Literal["minimize", "differential_evolution"]):
                The optimization algorithm to use from the scipy.optimize package.
            optimizer_options (Dict[str, Any]):
                The options for the optimization algorithm. See the documentation for
                scipy.optimize.minimize and scipy.optimize.differential_evolution.

        Returns:
            NDArray[np.float64]:
                The fitted stoichiometry limits - [x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi].
        """

        def _ocv_curve_fit_objective(
            params: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            """Objective function for the OCV curve fitting.

            Args:
                params (NDArray[np.float64]): The fitting parameters.

            Returns:
                NDArray[np.float64]: The residuals between the data and the fit.
            """
            x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi = params
            match fitting_target:
                case "OCV":
                    model = self._f_OCV(SOC, x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi)
                case "dVdQ":
                    model = self._f_grad_OCV(SOC, x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi)
                case "dQdV":
                    with np.errstate(divide="ignore", invalid="ignore"):
                        model = 1 / self._f_grad_OCV(
                            SOC, x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi
                        )
                        model[
                            ~np.isfinite(model)
                        ] = np.inf  # Set infinities and NaNs to zero
                case _:
                    raise ValueError(f"Invalid fitting target: {fitting_target}")
            return np.sum((model - fitting_target_data) ** 2)

        selected_optimizer = {
            "minimize": optimize.minimize,
            "differential_evolution": optimize.differential_evolution,
        }[optimizer]

        return selected_optimizer(_ocv_curve_fit_objective, **optimizer_options).x

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
        return DMA(input_data=average_result)
