"""Analysis tests for time series functions."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from pyprobe.analysis.time_series import align_data
from pyprobe.result import Result


def test_align_data():
    """Test align_data with LazyFrames."""
    t = np.linspace(0, 20, 200)
    y1 = np.sin(t)
    y2 = np.sin(t - 2.0)

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
    assert np.isclose(diff, -2.0, atol=0.2)
