"""Module containing methods for smoothing noisy experimental data."""

import copy
import logging
from typing import Any, Callable, Literal, Optional, Tuple

import numpy as np
import polars as pl
from deprecated import deprecated
from numpy.typing import NDArray
from pydantic import BaseModel, validate_call
from scipy import interpolate
from scipy.interpolate import make_smoothing_spline
from scipy.signal import savgol_filter

from pyprobe.analysis.utils import AnalysisValidator
from pyprobe.pyprobe_types import PyProBEDataType
from pyprobe.result import Result

logger = logging.getLogger(__name__)


@validate_call
def spline_smoothing(
    input_data: PyProBEDataType,
    target_column: str,
    smoothing_lambda: Optional[float] = None,
    x: str = "Time [s]",
) -> Result:
    """A method for smoothing noisy data using a spline.

    Args:
        input_data:
            The input data to smooth.
        target_column:
            The name of the target variable to smooth.
        smoothing_lambda:
            The smoothing parameter. Default is None.
        x:
            The name of the x variable for the spline curve fit.
            Default is "Time [s]".

    Returns:
        Result:
            A result object containing the data from input data with the target
            column smoothed using a spline, and the gradient of the smoothed data
            with respect to the x variable.
    """
    # validate and identify variables
    validator = AnalysisValidator(
        input_data=input_data, required_columns=[x, target_column]
    )
    x_data, y_data = validator.variables

    data_flipped = False
    if x_data[0] > x_data[-1]:  # flip the data if it is not in ascending order
        x_data = np.flip(x_data)
        y_data = np.flip(y_data)
        data_flipped = True

    y_spline = make_smoothing_spline(x_data, y_data, lam=smoothing_lambda)

    smoothed_y = y_spline(x_data)
    if data_flipped:
        x_data = np.flip(x_data)
        smoothed_y = np.flip(smoothed_y)

    derivative = y_spline.derivative()
    smoothed_dydx = derivative(x_data)

    result = copy.deepcopy(input_data)
    smoothed_data_column = pl.Series(target_column, smoothed_y)
    result.base_dataframe = result.base_dataframe.with_columns(
        smoothed_data_column.alias(target_column)
    )

    gradient_column_name = f"d({target_column})/d({x})"
    dydx_column = pl.Series(gradient_column_name, smoothed_dydx)
    result.base_dataframe = result.base_dataframe.with_columns(
        dydx_column.alias(gradient_column_name)
    )
    result.define_column(
        f"d({target_column})/d({x})",
        "The gradient of the smoothed data.",
    )
    return result


def _downsample_monotonic_data(
    df: pl.DataFrame | pl.LazyFrame,
    target: str,
    sampling_interval: float,
    occurrence: Literal["first", "last", "middle"] = "first",
) -> pl.DataFrame | pl.LazyFrame:
    """Resample a DataFrame to a specified interval.

    This method bins the data into intervals of the specified size and then selects
    the desired occurrence within each bin. The target column must be monotonic.

    Args:
        df (pl.DataFrame | pl.LazyFrame):
            The DataFrame to downsample.
        target (str):
            The target column to downsample.
        sampling_interval (float):
            The desired minimum interval between points.
        occurrence (Literal['first', 'last', 'middle'], optional):
            The occurrence to take when downsampling. Default is 'first'.

    Returns:
        pl.DataFrame | pl.LazyFrame:
            The downsampled DataFrame.
    """
    df = df.with_columns(
        [
            ((pl.col(target) / sampling_interval).floor() * sampling_interval).alias(
                "bin"
            )
        ]
    )
    # Group by 'group' and select the desired occurrence
    if occurrence == "first":
        return df.group_by("bin", maintain_order=True).first().drop("bin")
    elif occurrence == "last":
        return df.group_by("bin", maintain_order=True).last().drop("bin")
    elif occurrence == "middle":
        return (
            df.group_by("bin", maintain_order=True).quantile(0.5, "nearest").drop("bin")
        )


def _downsample_non_monotonic_data(
    df: pl.DataFrame | pl.LazyFrame,
    target: str,
    sampling_interval: float,
) -> pl.DataFrame | pl.LazyFrame:
    """Resample a DataFrame to a specified interval.

    This method loops through the data and selects each point that meets the interval
    condition over the previously sampled point.

    Args:
        df (pl.DataFrame | pl.LazyFrame):
            The DataFrame to downsample.
        target (str):
            The target column to downsample.
        sampling_interval (float):
            The desired minimum interval between points.

    Returns:
        pl.DataFrame | pl.LazyFrame:
            The downsampled DataFrame.
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    x = df.select(target).to_numpy()
    indices = [0]
    last_val = x[0]

    for i in range(1, len(x)):
        if abs(x[i] - last_val) >= sampling_interval:
            indices.append(i)
            last_val = x[i]
    df = df.with_row_index()
    return df.filter(pl.col("index").is_in(indices)).drop("index")


@validate_call
def downsample(
    input_data: PyProBEDataType,
    target_column: str,
    sampling_interval: float,
    monotonic: bool = True,
    occurrence: Literal["first", "last", "middle"] = "first",
) -> Result:
    """Downsample a DataFrame to a specified interval.

    This function uses two different methods for downsampling depending on whether the
    target column is monotonic or not.
    - If the target column is monotonic, the data is
    binned into intervals of the specified size and the desired occurrence within each
    bin is selected.
    - If the target column is not monotonic, the data is looped through
    and compared to the previously sampled point to determine if the interval condition
    is met.

    The monotonic algorithm is faster and more efficient, while the non-monotonic
    guarantees that the interval condition is met.

    Args:
        input_data:
            The input data to downsample.
        target_column:
            The target column to downsample.
        sampling_interval:
            The desired minimum interval between points.
        monotonic:
            If True, the target_column is assumed to be monotonic. Default is True.
            If False, the target_column is assumed to be non-monotonic. Each point in
            the target column is compared to the previously sampled point to determine
            if the interval condition is met.
        occurrence:
            The occurrence to take when downsampling. Default is 'first'.
            This argument is only used when the target column is monotonic, otherwise
            the first occurrence when the interval condition is met is taken.

    Returns:
        Result:
            A result object containing the downsampled DataFrame.
    """
    AnalysisValidator(input_data=input_data, required_columns=[target_column])
    result = copy.deepcopy(input_data)
    if monotonic:
        result.base_dataframe = _downsample_monotonic_data(
            result.base_dataframe,
            target_column,
            sampling_interval,
            occurrence,
        )
    else:
        result.base_dataframe = _downsample_non_monotonic_data(
            result.base_dataframe,
            target_column,
            sampling_interval,
        )
    return result


@validate_call
def savgol_smoothing(
    input_data: PyProBEDataType,
    target_column: str,
    window_length: int,
    polyorder: int,
    derivative: int = 0,
) -> Result:
    """Smooth noisy data using a Savitzky-Golay filter.

    Args:
        input_data:
            The input data to smooth.
        target_column:
            The name of the target variable to smooth.
        window_length:
            The length of the filter window. Must be a positive odd integer.
        polynomial_order:
            The order of the polynomial used to fit the samples.
        derivative:
            The order of the derivative to compute. Default is 0.

    Returns:
        Result:
            A result object containing all of the columns of input_data smoothed
            using the Savitzky-Golay filter.
    """
    # validate and identify variables
    validator = AnalysisValidator(
        input_data=input_data, required_columns=[target_column]
    )
    x = validator.variables
    smoothed_y = savgol_filter(
        x=x, window_length=window_length, polyorder=polyorder, deriv=derivative
    )

    smoothed_data_column = pl.Series(target_column, smoothed_y)
    result = copy.deepcopy(input_data)
    result.base_dataframe = result.base_dataframe.with_columns(
        smoothed_data_column.alias(target_column)
    )
    return result


class _LinearInterpolator(interpolate.PPoly):
    """A class to interpolate data linearly."""

    def __init__(
        self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
    ) -> None:
        """Initialize the interpolator."""
        slopes = np.diff(y) / np.diff(x)
        coefficients = np.vstack([slopes, y[:-1]])
        super().__init__(coefficients, x, **kwargs)


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
        error_msg = "x must be strictly increasing or decreasing"
        logger.error(error_msg)
        raise ValueError(error_msg)
    if len(x) != len(y):
        error_msg = "x and y must have the same length"
        logger.error(error_msg)
        raise ValueError(error_msg)
    if not np.all(np.diff(x) > 0):
        x = np.flip(x)
        y = np.flip(y)
    return x, y


def _create_interpolator(
    interpolator_class: Callable[..., Any],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    **kwargs: Any,
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
    return interpolator_class(x, y, **kwargs)


def linear_interpolator(
    x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Create a linear interpolator.

    Args:
        x (NDArray[np.float64]): The x data.
        y (NDArray[np.float64]): The y data.

    Returns:
        Callable[[NDArray[np.float64]], NDArray[np.float64]]: The linear interpolator.
    """
    return _create_interpolator(_LinearInterpolator, x, y, **kwargs)


def cubic_interpolator(
    x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Create a Scipy cubic spline interpolator.

    Args:
        x (NDArray[np.float64]): The x data.
        y (NDArray[np.float64]): The y data.

    Returns:
        Callable[[NDArray[np.float64]], NDArray[np.float64]]:
            The cubic spline interpolator.
    """
    return _create_interpolator(interpolate.CubicSpline, x, y, **kwargs)


def pchip_interpolator(
    x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Create a Scipy Pchip interpolator.

    Args:
        x (NDArray[np.float64]): The x data.
        y (NDArray[np.float64]): The y data.

    Returns:
        Callable[[NDArray[np.float64]], NDArray[np.float64]]: The Pchip interpolator.
    """
    return _create_interpolator(interpolate.PchipInterpolator, x, y, **kwargs)


def akima_interpolator(
    x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Create a Scipy Akima interpolator.

    Args:
        x (NDArray[np.float64]): The x data.
        y (NDArray[np.float64]): The y data.

    Returns:
        Callable[[NDArray[np.float64]], NDArray[np.float64]]: The Akima interpolator.
    """
    return _create_interpolator(interpolate.Akima1DInterpolator, x, y, **kwargs)


class Smoothing(BaseModel):
    """A class for smoothing noisy experimental data.

    Smoothing methods return result objects containing the same columns as the input
    data, but with the target column smoothed using the specified method.
    """

    input_data: Result
    """The input data for the smoothing."""

    @deprecated(
        reason="Use the module-level smoothing.spline_smoothing method instead.",
        version="1.1.0",
    )
    def spline_smoothing(
        self,
        target_column: str,
        smoothing_lambda: Optional[float] = None,
        x: str = "Time [s]",
    ) -> Result:
        """A method for smoothing noisy data using a spline.

        Args:
            target_column (str):
                The name of the target variable to smooth.
            smoothing_lambda (float, optional):
                The smoothing parameter. Default is None.
            x (str, optional):
                The name of the x variable for the spline curve fit.
                Default is "Time [s]".

        Returns:
            Result:
                A result object containing the data from input data with the target
                column smoothed using a spline, and the gradient of the smoothed data
                with respect to the x variable.
        """
        # validate and identify variables
        validator = AnalysisValidator(
            input_data=self.input_data, required_columns=[x, target_column]
        )
        x_data, y_data = validator.variables

        data_flipped = False
        if x_data[0] > x_data[-1]:  # flip the data if it is not in ascending order
            x_data = np.flip(x_data)
            y_data = np.flip(y_data)
            data_flipped = True

        y_spline = make_smoothing_spline(x_data, y_data, lam=smoothing_lambda)

        smoothed_y = y_spline(x_data)
        if data_flipped:
            x_data = np.flip(x_data)
            smoothed_y = np.flip(smoothed_y)

        derivative = y_spline.derivative()
        smoothed_dydx = derivative(x_data)

        result = copy.deepcopy(self.input_data)
        smoothed_data_column = pl.Series(target_column, smoothed_y)
        result.base_dataframe = result.base_dataframe.with_columns(
            smoothed_data_column.alias(target_column)
        )

        gradient_column_name = f"d({target_column})/d({x})"
        dydx_column = pl.Series(gradient_column_name, smoothed_dydx)
        result.base_dataframe = result.base_dataframe.with_columns(
            dydx_column.alias(gradient_column_name)
        )
        result.define_column(
            f"d({target_column})/d({x})",
            "The gradient of the smoothed data.",
        )
        return result

    @deprecated(
        reason="Use the module-level smoothing.downsample method instead.",
        version="1.1.0",
    )
    def downsample(
        self,
        target_column: str,
        sampling_interval: float,
        occurrence: Literal["first", "last", "middle"] = "first",
    ) -> Result:
        """Downsample a DataFrame to a specified interval.

        Requires the target column to be monotonic.

        Args:
            target_column (str):
                The target column to downsample.
            sampling_interval (float):
                The desired minimum interval between points.
            occurrence (Literal['first', 'last', 'middle'], optional):
                The occurrence to take when downsampling. Default is 'first'.
            time_column (str, optional):
                The time column to use for downsampling. Default is 'Time [s]'.

        Returns:
            Result:
                A result object containing the downsampled DataFrame.
        """
        result = copy.deepcopy(self.input_data)
        result.base_dataframe = _downsample_monotonic_data(
            result.base_dataframe,
            target_column,
            sampling_interval,
            occurrence,
        )
        return result

    @deprecated(
        reason="Use the module-level smoothing.downsample method instead.",
        version="1.1.0",
    )
    def level_smoothing(
        self,
        target_column: str,
        interval: float,
        monotonic: Optional[bool] = False,
    ) -> Result:
        """Smooth noisy data by resampling to a specified interval.

        Args:
            target_column (str):
                The name of the target variable to smooth.
            interval (float):
                The desired minimum interval between points.
            monotonic (bool, optional):
                If True, the target_column is assumed to be monotonic. Default is False.

        Returns:
            Result:
                A result object containing all of the columns of input_data resampled
                to the specified interval for the target column.
        """
        # validate and identify variables
        validator = AnalysisValidator(
            input_data=self.input_data, required_columns=[target_column]
        )
        x = validator.variables[0]

        if monotonic:
            if x[0] > x[-1]:
                x = np.flip(x)

        last_x = x[0]  # Start with the first point
        x_resampled = [last_x]  # Add the first point to the resampled list

        # Use a while loop to find the next point that meets the interval condition
        i = 1
        while i < len(x):
            # Find the next index where the difference meets the interval condition
            if monotonic:  # if monotinic, only consider positive differences
                comparison = x[i:] - last_x
            else:  # if not monotonic, consider all differences
                comparison = abs(x[i:] - last_x)

            next_indices = np.where(comparison >= interval)[0]
            if len(next_indices) == 0:
                break
            next_index = next_indices[0] + i
            x_resampled.append(x[next_index])
            last_x = x[next_index]
            i = next_index + 1

        # Filter the dataframe to only include the resampled points
        dataframe = self.input_data.base_dataframe.filter(
            [
                pl.col(target_column).is_in(x_resampled),
                pl.col(target_column).is_first_distinct(),
            ]
        )

        # Create a new result object with the resampled data
        result = self.input_data
        result.base_dataframe = dataframe
        return result

    @deprecated(
        reason="Use the module-level smoothing.savgol_smoothing method instead.",
        version="1.1.0",
    )
    def savgol_smoothing(
        self,
        target_column: str,
        window_length: int,
        polyorder: int,
        derivative: int = 0,
    ) -> Result:
        """Smooth noisy data using a Savitzky-Golay filter.

        Args:
            target_column (str):
                The name of the target variable to smooth.
            window_length (int):
                The length of the filter window. Must be a positive odd integer.
            polynomial_order (int):
                The order of the polynomial used to fit the samples.
            derivative (int, optional):
                The order of the derivative to compute. Default is 0.

        Returns:
            Result:
                A result object containing all of the columns of input_data smoothed
                using the Savitzky-Golay filter.
        """
        # validate and identify variables
        validator = AnalysisValidator(
            input_data=self.input_data, required_columns=[target_column]
        )
        x = validator.variables
        smoothed_y = savgol_filter(
            x=x, window_length=window_length, polyorder=polyorder, deriv=derivative
        )

        smoothed_data_column = pl.Series(target_column, smoothed_y)
        result = copy.deepcopy(self.input_data)
        result.base_dataframe = result.base_dataframe.with_columns(
            smoothed_data_column.alias(target_column)
        )
        return result
