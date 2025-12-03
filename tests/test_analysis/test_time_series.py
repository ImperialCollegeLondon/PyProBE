"""Analysis tests for time series functions."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from pyprobe.analysis.time_series import align_data
from pyprobe.result import Result


def test_align_data():
    """Test align_data with LazyFrames."""
    dt = 0.1
    t = np.arange(0, 20, dt)

    # Create square wave signals by sampling a continuous signal
    # This simulates real data where edge timing is preserved in sample values
    t_continuous = np.linspace(0, 20, 100000)
    y1_continuous = np.zeros_like(t_continuous)
    y1_continuous[t_continuous >= 5.0] = 1.0
    y1_continuous[t_continuous >= 10.0] = 0.0
    y1_continuous[t_continuous >= 12.0] = -1.0
    y1_continuous[t_continuous >= 17.0] = 0.0

    shift = 2.35
    y2_continuous = np.zeros_like(t_continuous)
    y2_continuous[t_continuous >= (5.0 + shift)] = 1.0
    y2_continuous[t_continuous >= (10.0 + shift)] = 0.0
    y2_continuous[t_continuous >= (12.0 + shift)] = -1.0
    y2_continuous[t_continuous >= (17.0 + shift)] = 0.0

    # Sample the continuous signals
    y1 = np.interp(t, t_continuous, y1_continuous)
    y2 = np.interp(t, t_continuous, y2_continuous)

    start_time = datetime(2023, 1, 1, 10, 0, 0)

    df1 = pl.DataFrame(
        {
            "Date": [start_time + timedelta(seconds=float(val)) for val in t],
            "Signal": y1,
        }
    ).lazy()

    df2 = pl.DataFrame(
        {
            "Date": [start_time + timedelta(seconds=float(val)) for val in t],
            "Signal": y2,
        }
    ).lazy()

    result1 = Result(base_dataframe=df1, info={})
    result2 = Result(base_dataframe=df2, info={})

    r1, r2 = align_data(result1, result2, "Signal", "Signal")

    # Trigger collection
    r2_df = r2.live_dataframe.collect()

    original_date = start_time
    new_date = r2_df["Date"][0]

    diff = (new_date - original_date).total_seconds()
    # The shift applied to result2 should be negative of the delay to align it back
    # y2 is delayed by 2.35s, so we need to shift it by -2.35s to match y1
    # Tolerance of 0.01s accounts for sub-sample precision of the alignment algorithm
    assert np.isclose(diff, -shift, atol=0.01)
