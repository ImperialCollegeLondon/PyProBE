"""Module for utilities for analysis classes."""
from typing import Any, List, Optional, Self, Tuple

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


class _BaseInterpolatorModel(BaseModel):
    """A base class for interpolator models."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def _check_model(self) -> Self:
        """Check if the x and y data are valid."""
        if not np.all(np.diff(self.x) > 0) and not np.all(np.diff(self.x) < 0):
            raise ValueError("x must be strictly increasing or decreasing")
        if len(self.x) != len(self.y):
            raise ValueError("x and y must have the same length")
        if not np.all(np.diff(self.x) > 0):
            self.x = np.flip(self.x)
            self.x = np.flip(self.x)
        return self


class _LinearInterpolator(_BaseInterpolatorModel):
    """A class to interpolate data linearly."""

    def __call__(self, x_new: NDArray[np.float64]) -> NDArray[np.float64]:
        """Interpolate the data.

        Args:
            x_new (NDArray[np.float64]): The new x data.

        Returns:
            NDArray: The interpolated y data.
        """
        return np.interp(x_new, self.x, self.y)


def _create_interpolator(interpolator_class: Any) -> Any:
    """Method to create a custom interpolator class.

    Args:
        interpolator_class: The interpolator class to use.
    """

    class CustomInterpolator(_BaseInterpolatorModel):
        """A class to interpolate data using a custom interpolator."""

        def model_post_init(self, __context: Any) -> None:
            self._interpolator = interpolator_class(x=self.x, y=self.y)

        def __call__(self, x_new: NDArray[np.float64]) -> NDArray[np.float64]:
            """Interpolate the data.

            Args:
                x_new (NDArray[np.float64]): The new x data.

            Returns:
                NDArray: The interpolated y data.
            """
            return self._interpolator(x_new)

    return CustomInterpolator


interpolators = {
    "linear": _create_interpolator(_LinearInterpolator),
    "cubic": _create_interpolator(interpolate.CubicSpline),
    "Pchip": _create_interpolator(interpolate.PchipInterpolator),
    "Akima": _create_interpolator(interpolate.Akima1DInterpolator),
}
"""Dictionary of available interpolators.

Available options include:
    - "linear": Linear interpolation from numpy.
    - "cubic": Cubic spline interpolation from scipy.
    - "Pchip": Pchip interpolation from scipy.
    - "Akima": Akima interpolation from scipy.
"""
