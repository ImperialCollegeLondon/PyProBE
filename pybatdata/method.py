"""Module for the base Method class."""
from typing import Dict, List, Tuple

import polars as pl
from numpy.typing import NDArray

from pybatdata.result import Result


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

    def __init__(self, input_data: Result, parameters: Dict[str, float]) -> None:
        """Initialize the Method object.

        Args:
            input_data (Result): The result object.
            parameters (Dict[str, float]): The parameters for the method.
        """
        self.input_data = input_data
        self.parameters = parameters
        self.variable_list: List[str] = []
        self.parameter_list: List[str] = []
        self.output_list: List[str] = []

    def variable(self, name: str) -> NDArray:
        """Return a variable from the input data.

        Args:
            name (str): The name of the variable.

        Returns:
            NDArray: The variable as a numpy array.
        """
        self.variable_list.append(name)
        return self.input_data.data[name].to_numpy()

    def parameter(self, name: str) -> float:
        """Return a parameter.

        Args:
            name (str): The name of the parameter.

        Returns:
            float: The parameter.
        """
        self.parameter_list.append(name)
        return self.parameters[name]

    def set_outputs(self, output_list: List[str]) -> None:
        """Set the output list.

        Args:
            output_list (List[str]): The list of outputs.
        """
        self.output_list = output_list

    def assign_outputs(self, function_call: Tuple[NDArray, NDArray]) -> None:
        """Assign the outputs of the method.

        Args:
            function_call (Tuple): The tuple of outputs from the method.
        """
        self.output_dict = {}
        for i, name in enumerate(self.output_list):
            self.output_dict[name] = function_call[i]

    @property
    def result(self) -> Result:
        """Return the result object of the method."""
        return Result(pl.DataFrame(self.output_dict), self.input_data.info)
