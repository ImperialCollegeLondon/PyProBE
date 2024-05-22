"""Module for the base Method class."""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from numpy.typing import NDArray

from pyprobe.result import Result


class Method:
    """A base class for a method.

    Attributes:
        input_data (Result): The input data to the method, a result object.
        parameters (Dict[str, float]): The parameters for the method.
        variable_list (List[str]): The list of variables used in the method.
        parameter_list (List[str]): The list of parameters used in the method.
        output_list (List[str]): The list of outputs from the method.
        output_dict (Dict[str, NDArray]): The dictionary of outputs from the method.
    """

    def __init__(
        self,
        input_data: Result | List[Result],
        parameters: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize the Method object.

        Args:
            input_data (Result): The result object.
            parameters (Dict[str, float]): The parameters for the method.
        """
        self.input_data = input_data
        self.parameters = parameters
        self.variable_list: List[str] = []
        self.parameter_list: List[str] = []
        self.output_dict: Dict[str, NDArray[Any]] = {}

    def variable(self, name: str) -> NDArray[Any]:
        """Return a variable from the input data.

        Args:
            name (str): The name of the variable.

        Returns:
            NDArray: The variable as a numpy array.
        """
        self.variable_list.append(name)

        if not isinstance(self.input_data, list):
            return self.input_data.data[name].to_numpy()
        else:
            return np.vstack([input.data[name].to_numpy() for input in self.input_data])

    def parameter(self, name: str) -> Any:
        """Return a parameter.

        Args:
            name (str): The name of the parameter.

        Returns:
            float: The parameter.
        """
        if self.parameters is None:
            raise ValueError("No parameters provided to method.")
        if name not in self.parameters:
            raise KeyError(f"Parameter {name} not found in parameter dict provided.")
        self.parameter_list.append(name)
        return self.parameters[name]

    def assign_outputs(
        self, output_list: List[str], function_call: Tuple[NDArray[np.float64], ...]
    ) -> Result:
        """Assign the outputs of the method.

        Args:
            output_list (List[str]): The list of output names.
            function_call (Tuple): The tuple of outputs from the method.

        Returns:
            Result: A result object containing the method outputs.
        """
        if all([np.ndim(item) == np.ndim(function_call[0]) for item in function_call]):
            for i, name in enumerate(output_list):
                self.output_dict[name] = np.asarray(function_call[i]).reshape(-1)
        else:
            for i, name in enumerate(output_list):
                self.output_dict[name] = function_call[i]

        if isinstance(self.input_data, list):
            info = self.input_data[0].info
        else:
            info = self.input_data.info

        return Result(pl.DataFrame(self.output_dict), info)
