"""Tests for the extend that merges the files of one test.

A cycling data object orders the data by start time, continues the test time,
offsets the step identifier, and rebuilds the step count. A plain table object
concatenates and applies none of those rules.
"""

import shutil
from pathlib import Path

import polars as pl
import pytest

from pyprobe.filters import Procedure
from pyprobe.io import process_cycler
from pyprobe.result import Table

ARBIN_SAMPLE = Path("tests/sample_data/arbin/sample_data_arbin.csv")
"""A raw Arbin cycler file of thirteen rows."""


def _procedure(**columns: list[float] | list[int]) -> Procedure:
    """Build a procedure over a frame of the given columns.

    Args:
        columns: The column values, keyed by the BDF column name with each
            space and slash replaced by an underscore.

    Returns:
        Procedure: A procedure over the frame.
    """
    names = {
        "test_time": "Test Time / s",
        "unix_time": "Unix Time / s",
        "current": "Current / A",
        "voltage": "Voltage / V",
        "step_id": "Step ID",
        "step_count": "Step Count / 1",
    }
    return Procedure.load(
        pl.DataFrame({names[key]: value for key, value in columns.items()}),
    )


class TestExtendOrder:
    """The extend orders the objects before it concatenates them."""

    def test_extend_orders_by_start_time(self) -> None:
        """The object with the earliest Unix time leads the extended data."""
        late = _procedure(
            unix_time=[100.0, 101.0, 102.0],
            current=[1.0, 1.0, 1.0],
            voltage=[4.0, 4.1, 4.2],
        )
        early = _procedure(
            unix_time=[0.0, 1.0, 2.0],
            current=[1.0, 1.0, 1.0],
            voltage=[3.0, 3.1, 3.2],
        )

        late.extend(early)

        assert late.data["Voltage / V"].to_list() == [3.0, 3.1, 3.2, 4.0, 4.1, 4.2]

    def test_extend_without_unix_time_keeps_the_given_order(self) -> None:
        """Without a Unix time column the objects stay in the order given."""
        first = _procedure(
            test_time=[10.0, 11.0, 12.0],
            current=[1.0, 1.0, 1.0],
            voltage=[4.0, 4.1, 4.2],
        )
        second = _procedure(
            test_time=[0.0, 1.0, 2.0],
            current=[1.0, 1.0, 1.0],
            voltage=[3.0, 3.1, 3.2],
        )

        first.extend(second)

        assert first.data["Voltage / V"].to_list() == [4.0, 4.1, 4.2, 3.0, 3.1, 3.2]


class TestExtendTime:
    """The extend states how the test time crosses a boundary."""

    @pytest.mark.xfail(
        strict=True,
        reason="CyclingData.extend does not implement the time rule",
    )
    def test_continuous_test_time_starts_at_the_last_value(self) -> None:
        """The second test time starts at the last test time of the first."""
        first = _procedure(
            test_time=[0.0, 1.0, 2.0],
            current=[1.0, 1.0, 1.0],
            voltage=[3.7, 3.8, 3.9],
        )
        second = _procedure(
            test_time=[0.0, 1.0, 2.0],
            current=[1.0, 1.0, 1.0],
            voltage=[3.6, 3.5, 3.4],
        )

        first.extend(second, time="continue")  # type: ignore[call-arg]

        assert first.data["Test Time / s"].to_list() == [0.0, 1.0, 2.0, 2.0, 3.0, 4.0]

    @pytest.mark.xfail(
        strict=True,
        reason="CyclingData.extend does not implement the time rule",
    )
    def test_elapsed_test_time_keeps_the_real_gap(self) -> None:
        """An elapsed test time follows the Unix time, so the gap survives."""
        first = _procedure(
            unix_time=[0.0, 1.0, 2.0],
            current=[1.0, 1.0, 1.0],
            voltage=[3.7, 3.8, 3.9],
        )
        second = _procedure(
            unix_time=[10.0, 11.0, 12.0],
            current=[1.0, 1.0, 1.0],
            voltage=[3.6, 3.5, 3.4],
        )

        first.extend(second, time="elapsed")  # type: ignore[call-arg]

        assert first.data["Test Time / s"].to_list() == [
            0.0,
            1.0,
            2.0,
            10.0,
            11.0,
            12.0,
        ]

    @pytest.mark.xfail(
        strict=True,
        reason="CyclingData.extend does not implement the time rule",
    )
    def test_elapsed_test_time_without_unix_time_raises(self) -> None:
        """An elapsed test time needs a Unix time column on every object."""
        first = _procedure(
            test_time=[0.0, 1.0, 2.0],
            current=[1.0, 1.0, 1.0],
            voltage=[3.7, 3.8, 3.9],
        )
        second = _procedure(
            test_time=[0.0, 1.0, 2.0],
            current=[1.0, 1.0, 1.0],
            voltage=[3.6, 3.5, 3.4],
        )

        with pytest.raises(ValueError, match="Unix Time / s"):
            first.extend(second, time="elapsed")  # type: ignore[call-arg]


class TestExtendSteps:
    """The extend states how the step columns cross a boundary."""

    @pytest.mark.xfail(
        strict=True,
        reason="CyclingData.extend does not implement the step identifier rule",
    )
    def test_offset_step_identifier_continues(self) -> None:
        """An offset step identifier continues past the maximum of the first."""
        first = _procedure(
            test_time=[0.0, 1.0, 2.0],
            current=[1.0, 1.0, 1.0],
            voltage=[3.7, 3.8, 3.9],
            step_id=[1, 2, 3],
        )
        second = _procedure(
            test_time=[0.0, 1.0, 2.0],
            current=[1.0, 1.0, 1.0],
            voltage=[3.6, 3.5, 3.4],
            step_id=[1, 2, 3],
        )

        first.extend(second, step_id="offset")  # type: ignore[call-arg]

        assert first.data["Step ID"].to_list() == [1, 2, 3, 4, 5, 6]

    @pytest.mark.xfail(
        strict=True,
        reason="CyclingData.extend does not implement the step identifier rule",
    )
    def test_verbatim_step_identifier_stacks(self) -> None:
        """A verbatim step identifier stacks the recorded values."""
        first = _procedure(
            test_time=[0.0, 1.0, 2.0],
            current=[1.0, 1.0, 1.0],
            voltage=[3.7, 3.8, 3.9],
            step_id=[1, 2, 3],
        )
        second = _procedure(
            test_time=[0.0, 1.0, 2.0],
            current=[1.0, 1.0, 1.0],
            voltage=[3.6, 3.5, 3.4],
            step_id=[1, 2, 3],
        )

        first.extend(second, step_id="keep")  # type: ignore[call-arg]

        assert first.data["Step ID"].to_list() == [1, 2, 3, 1, 2, 3]

    @pytest.mark.xfail(
        strict=True,
        reason="CyclingData.extend does not rebuild the step count",
    )
    def test_step_count_is_rebuilt_across_the_boundary(self) -> None:
        """A recorded step count is replaced by one that never resets."""
        first = _procedure(
            test_time=[0.0, 1.0, 2.0, 3.0],
            current=[1.0, 1.0, 1.0, 1.0],
            voltage=[3.7, 3.8, 3.9, 4.0],
            step_id=[1, 1, 2, 2],
            step_count=[0, 0, 1, 1],
        )
        second = _procedure(
            test_time=[0.0, 1.0, 2.0, 3.0],
            current=[1.0, 1.0, 1.0, 1.0],
            voltage=[3.6, 3.5, 3.4, 3.3],
            step_id=[1, 1, 2, 2],
            step_count=[0, 0, 1, 1],
        )

        first.extend(second)

        counts = first.data["Step Count / 1"].to_list()
        assert counts == sorted(counts)
        assert len(set(counts)) == 4
        assert counts[4] > counts[3]


class TestExtendResult:
    """A plain table object applies none of the rules."""

    def test_result_extends_without_the_rules(self) -> None:
        """A table with no BDF time and no step identifier stacks verbatim."""
        first = Table(pl.DataFrame({"Time": [0.0, 1.0], "Value": [1.0, 2.0]}))
        second = Table(pl.DataFrame({"Time": [0.0, 1.0], "Value": [3.0, 4.0]}))

        first.extend(second)

        assert first.data["Time"].to_list() == [0.0, 1.0, 0.0, 1.0]
        assert first.data["Value"].to_list() == [1.0, 2.0, 3.0, 4.0]
        assert "Step Count / 1" not in first.data.columns


class TestGlobSource:
    """A conversion of a glob loads every file it matches."""

    @pytest.mark.xfail(
        strict=True,
        reason="process_cycler does not compose the load and the extend",
    )
    def test_glob_loads_every_file_and_extends(self, tmp_path: Path) -> None:
        """Two matched files become one artifact with a continuous test time."""
        for name in ("session_2.csv", "session_10.csv"):
            shutil.copy(ARBIN_SAMPLE, tmp_path / name)
        output = tmp_path / "converted.parquet"

        process_cycler(str(tmp_path / "session_*.csv"), output_path=output)

        data = pl.read_parquet(output)
        assert data.height == 26
        times = data["Test Time / s"].to_list()
        assert times == sorted(times)

    def test_glob_that_matches_no_file_raises(self, tmp_path: Path) -> None:
        """A pattern that matches nothing fails and names the pattern."""
        pattern = str(tmp_path / "session_*.csv")

        with pytest.raises(FileNotFoundError, match="session_"):
            process_cycler(pattern, output_path=tmp_path / "converted.parquet")
