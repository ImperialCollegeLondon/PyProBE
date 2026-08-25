"""Tests for the write path of PyProBE data objects.

A write produces a Parquet data file and its BDF metadata sidecar together.
:func:`pyprobe.io.process_cycler` composes a read and that write, and it reuses
an existing output file unless the user asks for an overwrite.
"""

import shutil
from pathlib import Path
from unittest.mock import patch

import bdf
import bdf.io
import polars as pl
import polars.testing as pl_testing
import pytest

from pyprobe.filters import Procedure
from pyprobe.io import process_cycler
from pyprobe.result import Table

ARBIN_SAMPLE = Path("tests/sample_data/arbin/sample_data_arbin.csv")
"""A raw Arbin cycler file of thirteen rows."""


class TestTableSave:
    """A table object writes itself to a BDF artifact."""

    @pytest.mark.xfail(strict=True, reason="Table.save is not implemented")
    def test_procedure_writes_the_data_file_and_the_sidecar(
        self,
        tmp_path: Path,
        procedure: Procedure,
    ) -> None:
        """A saved procedure lands as a data file that a BDF reader opens."""
        path = tmp_path / "procedure.parquet"

        written = procedure.save(path)

        assert written == path
        assert path.exists()
        assert (tmp_path / "procedure.metadata.json").exists()
        frame, _ = bdf.io.scan(path, plugin="bdf_parquet")
        pl_testing.assert_frame_equal(frame.collect(), procedure.data)

    @pytest.mark.xfail(strict=True, reason="Table.save is not implemented")
    def test_filtered_object_writes_its_own_data(
        self,
        tmp_path: Path,
        procedure: Procedure,
    ) -> None:
        """A step writes the rows of that step alone."""
        step = procedure.step(0)
        path = tmp_path / "step.parquet"

        step.save(path)

        pl_testing.assert_frame_equal(pl.read_parquet(path), step.data)
        assert (tmp_path / "step.metadata.json").exists()

    @pytest.mark.xfail(strict=True, reason="Table.save is not implemented")
    def test_written_artifact_reloads(
        self,
        tmp_path: Path,
        procedure: Procedure,
    ) -> None:
        """A saved procedure reloads with the same data and the same record."""
        path = tmp_path / "procedure.parquet"
        procedure.metadata = bdf.Metadata(raw={"Name": "A"})
        procedure.save(path)

        loaded = Procedure.load(path)

        pl_testing.assert_frame_equal(loaded.data, procedure.data)
        assert loaded.metadata.raw == {"Name": "A"}  # type: ignore[attr-defined]

    @pytest.mark.xfail(strict=True, reason="Table.save is not implemented")
    def test_save_path_with_a_wrong_suffix_raises(self, tmp_path: Path) -> None:
        """A save to a path that is not Parquet fails and names the suffix."""
        table = Table(
            pl.DataFrame(
                {
                    "Test Time / s": [0.0, 1.0],
                    "Current / A": [1.0, -1.0],
                    "Voltage / V": [3.7, 3.6],
                },
            ),
        )

        with pytest.raises(ValueError, match=r"\.csv"):
            table.save(tmp_path / "table.csv")

    @pytest.mark.xfail(strict=True, reason="Table.save is not implemented")
    def test_save_without_a_required_column_raises(self, tmp_path: Path) -> None:
        """A frame that holds no current column fails the BDF validation."""
        table = Table(
            pl.DataFrame(
                {
                    "Test Time / s": [0.0, 1.0],
                    "Voltage / V": [3.7, 3.6],
                },
            ),
        )

        with pytest.raises(bdf.BDFValidationError, match="Current / A"):
            table.save(tmp_path / "table.parquet")

    @pytest.mark.xfail(strict=True, reason="Table.save is not implemented")
    def test_save_over_an_existing_file_needs_an_overwrite(
        self,
        tmp_path: Path,
        procedure: Procedure,
    ) -> None:
        """A second save fails unless the user asks for an overwrite."""
        path = tmp_path / "procedure.parquet"
        procedure.save(path)

        with pytest.raises(FileExistsError, match="procedure.parquet"):
            procedure.save(path)

        assert procedure.save(path, overwrite=True) == path


class TestProcessCycler:
    """One call reads a raw cycler file and writes a stored artifact."""

    @pytest.mark.xfail(
        strict=True,
        reason="process_cycler does not compose the BDF write",
    )
    def test_absent_output_file_is_written(self, tmp_path: Path) -> None:
        """A conversion with no output file reads the raw file and writes one."""
        source = tmp_path / ARBIN_SAMPLE.name
        shutil.copy(ARBIN_SAMPLE, source)
        output = tmp_path / "converted.parquet"

        written = process_cycler(source, output_path=output)

        assert written == output
        assert pl.read_parquet(output).height == 13
        assert (tmp_path / "converted.metadata.json").exists()

    def test_existing_output_file_is_reused(self, tmp_path: Path) -> None:
        """A conversion with an output file present reads no raw data."""
        source = tmp_path / ARBIN_SAMPLE.name
        shutil.copy(ARBIN_SAMPLE, source)
        output = tmp_path / "converted.parquet"
        pl.DataFrame({"Voltage / V": [1.0]}).write_parquet(output)

        with patch("bdf.io.scan") as scan:
            written = process_cycler(source, output_path=output)

        scan.assert_not_called()
        assert written == output
        assert pl.read_parquet(output).columns == ["Voltage / V"]

    @pytest.mark.xfail(
        strict=True,
        reason="process_cycler does not compose the BDF write",
    )
    def test_overwrite_replaces_the_output_file(self, tmp_path: Path) -> None:
        """A requested overwrite reads the raw file again and replaces the output."""
        source = tmp_path / ARBIN_SAMPLE.name
        shutil.copy(ARBIN_SAMPLE, source)
        output = tmp_path / "converted.parquet"
        pl.DataFrame({"Voltage / V": [1.0]}).write_parquet(output)

        written = process_cycler(source, output_path=output, overwrite_data=True)

        assert written == output
        assert pl.read_parquet(output).height == 13
        assert (tmp_path / "converted.metadata.json").exists()

    def test_output_path_with_a_wrong_suffix_raises(self, tmp_path: Path) -> None:
        """An output path that is not Parquet fails and names the suffix."""
        source = tmp_path / ARBIN_SAMPLE.name
        shutil.copy(ARBIN_SAMPLE, source)

        with pytest.raises(ValueError, match=r"\.csv"):
            process_cycler(source, output_path=tmp_path / "converted.csv")

    @pytest.mark.xfail(strict=True, reason="Table.save is not implemented")
    def test_pyprobe_artifact_is_not_a_raw_source(
        self,
        tmp_path: Path,
        procedure: Procedure,
    ) -> None:
        """A conversion of a written artifact directs the user to the load path."""
        artifact = tmp_path / "procedure.parquet"
        procedure.save(artifact)

        with pytest.raises(ValueError, match="load"):
            process_cycler(artifact, output_path=tmp_path / "again.parquet")
