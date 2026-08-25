"""Tests for the frame route of :meth:`pyprobe.filters.Procedure.load`.

A polars or pandas frame loads directly, with a ``column_map`` that names
the source column of each BDF output column. Every key of that map must be
a BDF column name in the ``"Quantity / unit"`` form.
"""

import polars as pl
import pytest

from pyprobe.filters import Procedure


class TestFrameRoute:
    """A frame loads directly, and a column map renames its columns."""

    @pytest.mark.xfail(
        strict=True,
        reason="Procedure.load takes no column_map argument",
    )
    def test_frame_maps_to_bdf_columns(self) -> None:
        """A mapped source column reaches the output under its BDF name."""
        frame = pl.DataFrame(
            {
                "time_s": [0.0, 1.0, 2.0],
                "curr_a": [1.0, -1.0, 0.0],
                "volt_v": [3.7, 3.6, 3.55],
            },
        )

        procedure = Procedure.load(  # type: ignore[call-arg]
            frame,
            column_map={
                "Test Time / s": "time_s",
                "Current / A": "curr_a",
                "Voltage / V": "volt_v",
            },
        )

        data = procedure.data
        assert data["Current / A"].to_list() == [1.0, -1.0, 0.0]
        assert data["Voltage / V"].to_list() == [3.7, 3.6, 3.55]
        assert "curr_a" not in data.columns

    @pytest.mark.xfail(
        strict=True,
        reason="Procedure.load takes no column_map argument",
    )
    def test_mapped_output_name_that_is_not_a_bdf_name_raises(self) -> None:
        """A column map key outside the 'Quantity / unit' form fails by name."""
        frame = pl.DataFrame({"curr_a": [1.0, -1.0]})

        with pytest.raises(ValueError, match="Current"):
            Procedure.load(  # type: ignore[call-arg]
                frame,
                column_map={"Current": "curr_a"},
            )
