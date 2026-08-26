"""Tests for the read path of :meth:`pyprobe.filters.Procedure.load`.

The read routes a raw cycler file through the BDF reader, reduces the result
to the core column set, and keeps the columns that the user names.
"""

import shutil
from pathlib import Path
from unittest.mock import patch

import bdf
import bdf.io
import polars as pl
import pytest

from pyprobe.filters import Procedure

ARBIN_SAMPLE = Path("tests/sample_data/arbin/sample_data_arbin.csv")
"""A raw Arbin cycler file of thirteen rows."""


def _write_bdf_csv(path: Path, header: str, rows: list[str]) -> Path:
    """Write a CSV file whose header holds BDF column names.

    Args:
        path: The file to write.
        header: The header line, without a line break.
        rows: The data lines, each without a line break.

    Returns:
        Path: The file that was written.
    """
    path.write_text("\n".join([header, *rows]) + "\n")
    return path


class TestRawFileRoute:
    """A raw cycler file reads through the BDF reader."""

    def test_raw_file_loads_and_writes_nothing(self, tmp_path: Path) -> None:
        """A raw cycler file becomes a procedure, and no file is written."""
        source = tmp_path / ARBIN_SAMPLE.name
        shutil.copy(ARBIN_SAMPLE, source)
        before = sorted(p.name for p in tmp_path.iterdir())

        procedure = Procedure.load(source)

        assert isinstance(procedure, Procedure)
        data = procedure.data
        assert data.height == 13
        assert data["Voltage / V"].to_list()[0] == pytest.approx(3.534595)
        assert data["Voltage / V"].to_list()[-1] == pytest.approx(3.599601)
        assert data["Current / A"].to_list()[-1] == pytest.approx(2.650138)
        assert sorted(p.name for p in tmp_path.iterdir()) == before

    def test_reader_controls_pass_through(self, tmp_path: Path) -> None:
        """The time zone, the date order and the reconciliation reach the reader."""
        source = _write_bdf_csv(
            tmp_path / "raw.csv",
            "Test Time / s,Current / A,Voltage / V",
            ["0.0,1.0,3.7", "1.0,-1.0,3.6"],
        )
        frame = pl.LazyFrame(
            {
                "Test Time / s": [0.0, 1.0],
                "Current / A": [1.0, -1.0],
                "Voltage / V": [3.7, 3.6],
            },
        )

        with patch(
            "bdf.io.scan",
            return_value=(frame, bdf.Metadata()),
        ) as scan:
            Procedure.load(  # type: ignore[call-arg]
                source,
                tz="Europe/London",
                day_month_order="day_first",
                reconcile_time=True,
            )

        keywords = scan.call_args.kwargs
        assert keywords["tz"] == "Europe/London"
        assert keywords["day_month_order"] == "day_first"
        assert keywords["reconcile_time"] is True


class TestCoreColumnReduction:
    """The read keeps the core column set alone."""

    def test_core_columns_are_kept_and_the_rest_are_dropped(
        self,
        tmp_path: Path,
    ) -> None:
        """A core column survives the read, and a non-core column does not."""
        source = tmp_path / ARBIN_SAMPLE.name
        shutil.copy(ARBIN_SAMPLE, source)

        procedure = Procedure.load(source)

        names = procedure.data.columns
        assert "Current / A" in names
        assert "Voltage / V" in names
        assert "Step ID" in names
        assert "Temperature T1 / degC" in names

    def test_absent_required_column_raises(self, tmp_path: Path) -> None:
        """A source without a current column fails and names that column."""
        source = _write_bdf_csv(
            tmp_path / "no_current.csv",
            "Test Time / s,Voltage / V",
            ["0.0,3.7", "1.0,3.6"],
        )

        with pytest.raises(bdf.BDFValidationError, match="Current / A"):
            Procedure.load(source)

    def test_no_time_column_resolves_raises(self) -> None:
        """A frame without either time column fails and names both columns."""
        frame = pl.DataFrame(
            {
                "Current / A": [1.0, -1.0],
                "Voltage / V": [3.7, 3.6],
            },
        )

        with pytest.raises(ValueError) as failure:
            Procedure.load(frame)

        assert "Unix Time / s" in str(failure.value)
        assert "Test Time / s" in str(failure.value)

    def test_absent_optional_column_warns_once(
        self,
        tmp_path: Path,
        caplog,
    ) -> None:
        """An optional column that does not resolve is dropped and reported."""
        source = _write_bdf_csv(
            tmp_path / "no_step_id.csv",
            "Test Time / s,Current / A,Voltage / V",
            ["0.0,1.0,3.7", "1.0,-1.0,3.6"],
        )

        procedure = Procedure.load(source)

        assert "Step ID" not in procedure.data.columns
        warnings = [
            record
            for record in caplog.records
            if record.levelname == "WARNING" and "Step ID" in record.getMessage()
        ]
        assert len(warnings) == 1


class TestExtraColumns:
    """A user names a source column that the ontology does not define."""

    def test_named_extra_column_is_kept_under_the_given_name(
        self,
        tmp_path: Path,
    ) -> None:
        """A named source column reaches the output under its given name."""
        source = _write_bdf_csv(
            tmp_path / "extra.csv",
            "Test Time / s,Current / A,Voltage / V,Pressure(kPa)",
            ["0.0,1.0,3.7,101.0", "1.0,-1.0,3.6,102.0"],
        )

        procedure = Procedure.load(  # type: ignore[call-arg]
            source,
            extra_columns={"Pressure(kPa)": "Ambient Pressure / kPa"},
        )

        data = procedure.data
        assert "Ambient Pressure / kPa" in data.columns
        assert "Pressure(kPa)" not in data.columns
        assert data["Ambient Pressure / kPa"].to_list() == [101.0, 102.0]

    def test_absent_named_source_column_raises(self, tmp_path: Path) -> None:
        """A named source column that the data does not hold fails by name."""
        source = _write_bdf_csv(
            tmp_path / "extra.csv",
            "Test Time / s,Current / A,Voltage / V",
            ["0.0,1.0,3.7", "1.0,-1.0,3.6"],
        )

        with pytest.raises(KeyError, match="Pressure"):
            Procedure.load(  # type: ignore[call-arg]
                source,
                extra_columns={"Pressure(kPa)": "Ambient Pressure / kPa"},
            )
