"""Module for the base Method class."""
from typing import Any, Dict, List

import numpy as np
import polars as pl
from numpy.typing import NDArray

from pyprobe.result import Result


class Method:
    """A base class for a method.

    Attributes:
        input_data (Result): The input data to the method, a result object.
        variable_list (List[str]): The list of variables used in the method.
        output_list (List[str]): The list of outputs from the method.
        output_dict (Dict[str, NDArray]): The dictionary of outputs from the method.
    """

    def __init__(
        self,
        input_data: Result | List[Result],
    ) -> None:
        """Initialize the Method object.

        Args:
            input_data (Result): The result object.
        """
        self.input_data = input_data
        self.variable_list: List[str] = []

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

    def assign_outputs(self, output_dict: Dict[str, NDArray[Any]]) -> Result:
        """Assign the outputs of the method.

        Args:
            Dict[str, NDArray]: A dictionary of the outputs.

        Returns:
            Result: A result object containing the method outputs.
        """
        first_dim = np.ndim(list(output_dict.values())[0])
        if all(np.ndim(array) == first_dim for array in output_dict.values()):
            for name in output_dict.keys():
                output_dict[name] = np.asarray(output_dict[name]).reshape(-1)

        if isinstance(self.input_data, list):
            info = self.input_data[0].info
        else:
            info = self.input_data.info

        return Result(pl.DataFrame(output_dict), info)
