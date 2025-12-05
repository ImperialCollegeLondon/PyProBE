"""Analysis functions for manipulating time series data."""

import datetime
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from loguru import logger
from scipy import ndimage, signal

from pyprobe.analysis.utils import AnalysisValidator

if TYPE_CHECKING:
    from pyprobe.result import Result


def _clean_data(
    date_arr: np.ndarray, y_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert date to float, remove NaNs/NaTs, and sort by date."""
    # Check if date_arr is datetime64
    if np.issubdtype(date_arr.dtype, np.datetime64):
        mask_date = ~np.isnat(date_arr)
        t = date_arr.astype("datetime64[us]").astype(float)
    else:
        # Assume it's already float (epoch)
        mask_date = ~np.isnan(date_arr)
        t = date_arr.astype(float)

    # Filter NaNs in y
    mask_y = ~np.isnan(y_arr)

    mask = mask_date & mask_y

    t = t[mask]
    y = y_arr[mask]

    sort_idx = np.argsort(t)
    return t[sort_idx], y[sort_idx]


def _parabolic_peak(corr: np.ndarray, peak_idx: int, lags: np.ndarray) -> float:
    """Refine peak location using parabolic interpolation.

    Fits a parabola to the peak and its two neighbors to find the true maximum.
    This provides sub-sample precision for the correlation peak.

    Args:
        corr: Correlation values.
        peak_idx: Index of the discrete peak.
        lags: Lag values corresponding to correlation.

    Returns:
        Refined lag value at the interpolated peak.
    """
    if 0 < peak_idx < len(corr) - 1:
        y_m1 = corr[peak_idx - 1]
        y_0 = corr[peak_idx]
        y_p1 = corr[peak_idx + 1]

        denom = 2 * (y_m1 - 2 * y_0 + y_p1)
        if abs(denom) > 1e-10:
            delta = (y_m1 - y_p1) / denom
            return lags[peak_idx] + delta * (lags[1] - lags[0])
    return lags[peak_idx]


def align_data(
    result1: "Result",
    result2: "Result",
    column1: str,
    column2: str,
) -> tuple["Result", "Result"]:
    """Align the data of two Result objects from the cross-correlation of two columns.

    The date column of result2 is shifted to best align column2 with column1 from
    result1.

    Args:
        result1 (Result): The first Result object (reference).
        result2 (Result): The second Result object (to be shifted).
        column1 (str): The column name in the first Result object to align on.
        column2 (str): The column name in the second Result object to align on.

    Returns:
        Tuple[Result, Result]: The two Result objects, with the second one shifted.
    """
    logger.info(f"Aligning data on {column1} and {column2}...")

    # Get data from result1
    validator1 = AnalysisValidator(
        input_data=result1,
        required_columns=["Date", column1],
    )
    date1, y1 = validator1.variables
    t1, y1 = _clean_data(date1, y1)

    # Get data from result2
    validator2 = AnalysisValidator(
        input_data=result2,
        required_columns=["Date", column2],
    )
    date2, y2 = validator2.variables
    t2, y2 = _clean_data(date2, y2)

    # Determine sampling rate (median dt)
    if len(t1) < 2 or len(t2) < 2:
        error_msg = (
            "Insufficient data points for alignment after cleaning. Need at least 2 "
            "valid points in each dataset."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    dt1 = np.median(np.diff(t1))
    dt2 = np.median(np.diff(t2))
    dt = min(dt1, dt2)

    # Create uniform grid
    t_start = min(t1[0], t2[0])
    t_end = max(t1[-1], t2[-1])
    t_grid = np.arange(t_start, t_end, dt)

    # Interpolate onto uniform grid
    y1_interp = np.interp(t_grid, t1, y1, left=0, right=0)
    y2_interp = np.interp(t_grid, t2, y2, left=0, right=0)

    # Remove mean
    y1_interp = y1_interp - np.mean(y1_interp)
    y2_interp = y2_interp - np.mean(y2_interp)

    # Apply Gaussian smoothing to enable sub-sample peak interpolation
    # This converts sharp edges into smooth curves that can be accurately
    # interpolated. A sigma of 2 samples provides good smoothing without
    # excessive blurring.
    sigma = 2  # samples
    y1_smooth = ndimage.gaussian_filter1d(y1_interp, sigma)
    y2_smooth = ndimage.gaussian_filter1d(y2_interp, sigma)

    # Correlate smoothed signals
    correlation = signal.correlate(y1_smooth, y2_smooth, mode="full")
    lags = signal.correlation_lags(len(y1_smooth), len(y2_smooth), mode="full")

    # Find discrete peak
    peak_idx = np.argmax(correlation)

    # Refine peak using parabolic interpolation for sub-sample precision
    lag = _parabolic_peak(correlation, peak_idx, lags.astype(float))

    time_shift = lag * dt
    time_shift_duration = datetime.timedelta(microseconds=time_shift)

    logger.info(f"Applying time shift of {time_shift_duration} to new data.")

    # Shift result2
    result2.lf = result2.lf.with_columns(pl.col("Date") + time_shift_duration)

    return result1, result2
