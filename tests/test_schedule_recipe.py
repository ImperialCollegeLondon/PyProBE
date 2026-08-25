"""Tests for schedule accumulator recipes.

Tests the _global_net_from_resetting_ch_dch and
_global_cumulative_from_resetting_ch_dch recipes that derive net and
cumulative columns from schedule-level charging and discharging accumulators
that reset at segment boundaries rather than step boundaries.
"""

from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from pyprobe.columns import BDF, ColumnDict


class TestScheduleRecipes:
    """Tests for schedule-level accumulator recipes."""

    def test_net_with_charge_segment_and_discharge_segment(self) -> None:
        """Net column reconstructs correctly from schedule accumulators with charge and discharge.

        Frame: t=[0,1,2,3], I=[1.0,2.0,1.5,1.0], ch=[0,0.01,0,0], dch=[0,0,0,0.01].
        ch decreases at row 2 (0.01 → 0), so segment key is [0,0,1,1] and row 2
        is the only boundary. Seam at row 2 is the trapezoid over current and
        its previous value: 0.5 * (1.5 + 2.0) * 1.0 / 3600 = 0.0004861111111111111.
        Row 1: diff(ch)=0.01, diff(dch)=0, seam=0 → net = 0.01.
        Row 2: diff(ch)=-0.01 (clipped to 0), diff(dch)=0, seam=0.0004861111 →
        net = 0.01 + 0.0004861111 = 0.0104861111.
        Row 3: diff(ch)=0, diff(dch)=0.01, seam=0 → net = 0.0104861111 - 0.01 = 0.0004861111.
        """
        df = pl.DataFrame(
            {
                "Test Time / s": [0.0, 1.0, 2.0, 3.0],
                "Current / A": [1.0, 2.0, 1.5, 1.0],
                "Schedule Charging Capacity / Ah": [0.0, 0.01, 0.00, 0.00],
                "Schedule Discharging Capacity / Ah": [0.0, 0.00, 0.00, 0.01],
            }
        )

        cs = ColumnDict(df.columns)
        expr = cs.resolve(BDF.NET_CAPACITY_AH)
        result = df.select(expr)

        expected = pl.DataFrame(
            {
                "Net Capacity / Ah": [
                    0.0,
                    0.01,
                    0.010486111111111111,
                    0.0004861111111111108,
                ]
            }
        )

        assert_frame_equal(result, expected, atol=1e-10, rtol=1e-12)

    def test_cumulative_with_charge_segment_and_discharge_segment(self) -> None:
        """Cumulative column reconstructs correctly from schedule accumulators.

        Frame: t=[0,1,2,3], I=[1.0,2.0,1.5,1.0], ch=[0,0.01,0,0], dch=[0,0,0,0.01].
        ch decreases at row 2, so segment key is [0,0,1,1] and row 2 is a
        boundary. Seam at row 2 is 0.5 * (1.5 + 2.0) * 1.0 / 3600 = 0.0004861111.
        Row 1: ch.cumsum()=0.01, dch.cumsum()=0, seam.abs().cumsum()=0 → cumulative = 0.01.
        Row 2: ch.cumsum()=0.01, dch.cumsum()=0, seam.abs().cumsum()=0.0004861111 →
        cumulative = 0.01 + 0.0004861111 = 0.0104861111.
        Row 3: ch.cumsum()=0.01, dch.cumsum()=0.01, seam.abs().cumsum()=0.0004861111 →
        cumulative = 0.01 + 0.01 + 0.0004861111 = 0.02048611111.
        """
        df = pl.DataFrame(
            {
                "Test Time / s": [0.0, 1.0, 2.0, 3.0],
                "Current / A": [1.0, 2.0, 1.5, 1.0],
                "Schedule Charging Capacity / Ah": [0.0, 0.01, 0.00, 0.00],
                "Schedule Discharging Capacity / Ah": [0.0, 0.00, 0.00, 0.01],
            }
        )

        cs = ColumnDict(df.columns)
        expr = cs.resolve(BDF.CUMULATIVE_CAPACITY_AH)
        result = df.select(expr)

        expected = pl.DataFrame(
            {
                "Cumulative Capacity / Ah": [
                    0.0,
                    0.01,
                    0.010486111111111111,
                    0.02048611111111111,
                ]
            }
        )

        assert_frame_equal(result, expected, atol=1e-10, rtol=1e-12)

    def test_seam_charge_difference_with_reset(self) -> None:
        """Seam term makes a measurable difference at the reset row.

        Frame: t=[0,1], I=[2.0,2.0], ch=[0.01,0], dch=[0,0].
        ch resets at row 1 (0.01 → 0), so row 1 is a boundary. Both clipped
        diffs are zero, so the seam is the whole answer. Seam at row 1 is the
        trapezoid over current and its previous value:
        0.5 * (2.0 + 2.0) * 1.0 / 3600 = 0.0005555555555555556.
        Row 1 net = 0 + 0 + 0.0005555555 = 0.0005555555.
        Without the seam term, row 1 would incorrectly be 0.0.
        """
        df = pl.DataFrame(
            {
                "Test Time / s": [0.0, 1.0],
                "Current / A": [2.0, 2.0],
                "Schedule Charging Capacity / Ah": [0.0100, 0.0000],
                "Schedule Discharging Capacity / Ah": [0.0, 0.0],
            }
        )

        cs = ColumnDict(df.columns)
        expr = cs.resolve(BDF.NET_CAPACITY_AH)
        result = df.select(expr)

        expected = pl.DataFrame({"Net Capacity / Ah": [0.0, 0.0005555555555555556]})

        assert_frame_equal(result, expected, atol=1e-10, rtol=1e-12)
