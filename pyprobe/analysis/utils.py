"""Module for utilities for analysis classes."""
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, model_validator
from scipy import interpolate

from pyprobe.result import Result
from pyprobe.typing import PyProBEDataType


def assemble_array(input_data: List[Result], name: str) -> NDArray[Any]:
    """Assemble an array from a list of results.

    Args:
        input_data (List[Result]): A list of results.
        name (str): The name of the variable.

    Returns:
        NDArray: The assembled array.
    """
    return np.vstack([input.get_only(name) for input in input_data])


class AnalysisValidator(BaseModel):
    """A base class for analysis classes."""

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    input_data: PyProBEDataType
    """The input data to an analysis class."""
    required_columns: List[str]
    """The columns required to conduct the analysis."""
    required_type: Optional[Any] = Field(default=None)
    """The required data type of the input data."""

    @model_validator(mode="after")
    def validate_input_data_type(self) -> "AnalysisValidator":
        """Check if the input_data is of the required type.

        Returns:
            AnalysisValidator: The validated instance.

        Raises:
            ValueError: If the input data is not of the required type.
        """
        if self.required_type is not None and not isinstance(
            self.input_data, self.required_type
        ):
            raise ValueError(f"Input data is not of type {self.required_type}")
        return self

    @model_validator(mode="after")
    def validate_required_columns(self) -> "AnalysisValidator":
        """Check if the required columns are present in the input_data.

        Returns:
            AnalysisValidator: The validated instance.

        Raises:
            ValueError: If any of the required columns are missing.
        """
        missing_columns = []
        for col in self.required_columns:
            if col not in self.input_data.column_list:
                try:
                    self.input_data._check_units(col)
                except ValueError:
                    missing_columns.append(col)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        return self

    @property
    def variables(self) -> Tuple[NDArray[np.float64], ...]:
        """Return the required columns in the input data as NDArrays.

        Returns:
            Tuple[NDArray[np.float64], ...]: The required columns as NDArrays.
        """
        return self.input_data.get(*tuple(self.required_columns))


class _LinearInterpolator(interpolate.PPoly):
    """A class to interpolate data linearly."""

    def __init__(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Initialize the interpolator."""
        slopes = np.diff(y) / np.diff(x)
        coefficients = np.vstack([slopes, y[:-1]])
        super().__init__(coefficients, x)


def _validate_interp_input_vectors(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate the input vectors x and y.

    Args:
        x (NDArray[np.float64]): The x data.
        y (NDArray[np.float64]): The y data.

    Raises:
        ValueError: If the input vectors are not valid.
    """
    if not np.all(np.diff(x) > 0) and not np.all(np.diff(x) < 0):
        raise ValueError("x must be strictly increasing or decreasing")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if not np.all(np.diff(x) > 0):
        x = np.flip(x)
        y = np.flip(y)
    return x, y


def _create_interpolator(
    interpolator_class: Callable[..., Any],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Create an interpolator after validating the input vectors.

    Args:
        interpolator_class (Callable[..., Any]): The interpolator class to use.
        x (NDArray[np.float64]): The x data.
        y (NDArray[np.float64]): The y data.

    Returns:
        Any: The interpolator object.
    """
    x, y = _validate_interp_input_vectors(x, y)
    return interpolator_class(x, y)


interpolators = {
    "linear": lambda x, y: _create_interpolator(_LinearInterpolator, x, y),
    "cubic": lambda x, y: _create_interpolator(interpolate.CubicSpline, x, y),
    "Pchip": lambda x, y: _create_interpolator(interpolate.PchipInterpolator, x, y),
    "Akima": lambda x, y: _create_interpolator(interpolate.Akima1DInterpolator, x, y),
}
"""Dictionary of available interpolators.

Available options include:
    - "linear": Linear interpolation from numpy.
    - "cubic": Cubic spline interpolation from scipy.
    - "Pchip": Pchip interpolation from scipy.
    - "Akima": Akima interpolation from scipy.
"""
