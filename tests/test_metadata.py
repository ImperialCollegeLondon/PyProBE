"""Tests for the typed metadata record and the sidecar store.

Every PyProBE data object holds one :class:`bdf.Metadata` record. The BDF
sidecar beside the data file is the only store on disk.
"""

import datetime
import json
from pathlib import Path

import bdf
import polars as pl
import pyarrow.parquet as pq
import pytest

import pyprobe._version
from pyprobe.filters import Procedure
from pyprobe.io import is_pyprobe_file, read_sidecar
from pyprobe.result import Table


class TestTypedRecord:
    """Every object holds one typed record."""

    def test_object_without_metadata_holds_an_empty_record(
        self,
        cycling_frame: pl.DataFrame,
    ) -> None:
        """An object built without metadata holds an empty typed record."""
        table = Table(cycling_frame)

        assert table.metadata == bdf.Metadata()

    def test_dictionary_metadata_raises(self, cycling_frame: pl.DataFrame) -> None:
        """A dictionary as the metadata fails and names the expected type."""
        with pytest.raises(TypeError, match="Metadata"):
            Table(cycling_frame, metadata={"Name": "A"})

    def test_derived_object_holds_its_own_copy(self, procedure: Procedure) -> None:
        """A change to the record of a filtered object leaves the source alone."""
        procedure.metadata = bdf.Metadata(raw={"Name": "A"})
        step = procedure.step(0)

        step.metadata.raw = {"Name": "B"}  # type: ignore[attr-defined]

        assert procedure.metadata.raw == {"Name": "A"}  # type: ignore[attr-defined]


class TestExtras:
    """The free-form keys live under the extras field."""

    def test_info_returns_the_extras_mapping(self, procedure: Procedure) -> None:
        """The information mapping is the extras mapping of the record."""
        procedure.metadata = bdf.Metadata(extras={"Name": "A", "Channel": 3})

        assert procedure.info == {"Name": "A", "Channel": 3}

    def test_object_without_extras_returns_an_empty_mapping(
        self,
        procedure: Procedure,
    ) -> None:
        """A record that holds no extras gives an empty mapping."""
        procedure.metadata = bdf.Metadata()

        assert procedure.info == {}

    @pytest.mark.xfail(strict=True, reason="Table.save is not implemented")
    def test_write_records_the_pyprobe_provenance(
        self,
        tmp_path: Path,
        procedure: Procedure,
    ) -> None:
        """A write records the PyProBE version and the write time."""
        path = tmp_path / "procedure.parquet"

        procedure.save(path)

        provenance = read_sidecar(path).extras["pyprobe"]
        assert provenance["version"] == pyprobe._version.__version__
        written_at = datetime.datetime.fromisoformat(provenance["written_at"])
        assert written_at.utcoffset() == datetime.timedelta(0)


class TestSidecarStore:
    """The sidecar beside the data file is the only store."""

    @pytest.mark.xfail(strict=True, reason="Table.save is not implemented")
    def test_write_produces_one_store(
        self,
        tmp_path: Path,
        procedure: Procedure,
    ) -> None:
        """A write leaves the record in the sidecar and not in the data file."""
        path = tmp_path / "procedure.parquet"

        procedure.save(path)

        assert (tmp_path / "procedure.metadata.json").exists()
        footer = pq.read_schema(path).metadata or {}
        assert b"bdf_metadata" not in footer

    @pytest.mark.xfail(strict=True, reason="Table.save is not implemented")
    def test_load_reads_the_sidecar(
        self,
        tmp_path: Path,
        procedure: Procedure,
    ) -> None:
        """A loaded object holds the record that the sidecar carries."""
        path = tmp_path / "procedure.parquet"
        procedure.metadata = bdf.Metadata(raw={"Name": "A"})
        procedure.save(path)

        loaded = Procedure.load(path)

        assert loaded.metadata.raw == {"Name": "A"}  # type: ignore[attr-defined]

    def test_absent_sidecar_gives_an_empty_record(
        self,
        tmp_path: Path,
        cycling_frame: pl.DataFrame,
    ) -> None:
        """A data file without a sidecar loads with an empty record."""
        path = tmp_path / "plain.parquet"
        cycling_frame.write_parquet(path)

        loaded = Procedure.load(path)

        assert loaded.metadata == bdf.Metadata()

    def test_sidecar_read_of_an_absent_data_file_raises(
        self,
        tmp_path: Path,
    ) -> None:
        """A sidecar read of a data file that does not exist fails."""
        with pytest.raises(FileNotFoundError):
            read_sidecar(tmp_path / "missing.parquet")

    def test_invalid_sidecar_raises_the_bdf_error(
        self,
        tmp_path: Path,
        cycling_frame: pl.DataFrame,
    ) -> None:
        """A sidecar that does not parse fails with the BDF metadata error."""
        path = tmp_path / "plain.parquet"
        cycling_frame.write_parquet(path)
        (tmp_path / "plain.metadata.json").write_text("{not json")

        with pytest.raises(bdf.BDFMetadataError):
            Procedure.load(path)

    @pytest.mark.xfail(strict=True, reason="Table.save is not implemented")
    def test_pyprobe_file_is_identified_from_the_sidecar(
        self,
        tmp_path: Path,
        procedure: Procedure,
        cycling_frame: pl.DataFrame,
    ) -> None:
        """A file is a PyProBE file where its sidecar holds the PyProBE key."""
        written = tmp_path / "procedure.parquet"
        procedure.save(written)
        plain = tmp_path / "plain.parquet"
        cycling_frame.write_parquet(plain)

        assert is_pyprobe_file(written) is True
        assert is_pyprobe_file(plain) is False


class TestSaveReplacesTheRecord:
    """A save writes the record that the object holds."""

    @pytest.mark.xfail(strict=True, reason="Table.save is not implemented")
    def test_changed_field_reaches_the_sidecar(
        self,
        tmp_path: Path,
        procedure: Procedure,
    ) -> None:
        """A changed field replaces the field that the sidecar held."""
        path = tmp_path / "procedure.parquet"
        procedure.metadata = bdf.Metadata(raw={"Name": "A"})
        procedure.save(path)
        loaded = Procedure.load(path)

        loaded.metadata.raw = {"Name": "B"}  # type: ignore[attr-defined]
        loaded.save(path, overwrite=True)

        assert read_sidecar(path).raw == {"Name": "B"}

    @pytest.mark.xfail(strict=True, reason="Table.save is not implemented")
    def test_dropped_field_leaves_the_sidecar(
        self,
        tmp_path: Path,
        procedure: Procedure,
    ) -> None:
        """A dropped field is absent from the sidecar after the save."""
        path = tmp_path / "procedure.parquet"
        procedure.metadata = bdf.Metadata(raw={"Name": "A"})
        procedure.save(path)
        loaded = Procedure.load(path)

        loaded.metadata.raw = None  # type: ignore[attr-defined]
        loaded.save(path, overwrite=True)

        assert read_sidecar(path).raw is None
        assert "raw" not in json.loads(
            (tmp_path / "procedure.metadata.json").read_text(),
        )


class TestExtendRecord:
    """An extend keeps the record of the object that it extends."""

    def test_agreeing_records_log_no_warning(
        self,
        cycling_frame: pl.DataFrame,
        caplog,
    ) -> None:
        """Two records that agree leave the record in place and log nothing."""
        first = Procedure.load(cycling_frame)
        second = Procedure.load(cycling_frame)
        first.metadata = bdf.Metadata(raw={"Name": "A"})
        second.metadata = bdf.Metadata(raw={"Name": "A"})
        caplog.clear()

        first.extend(second)

        assert first.metadata.raw == {"Name": "A"}  # type: ignore[attr-defined]
        assert [
            record for record in caplog.records if record.levelname == "WARNING"
        ] == []

    @pytest.mark.xfail(
        strict=True,
        reason="Table.extend does not report a differing record",
    )
    def test_differing_records_warn_and_keep_the_first(
        self,
        cycling_frame: pl.DataFrame,
        caplog,
    ) -> None:
        """A differing record is reported by field, and the first one stays."""
        first = Procedure.load(cycling_frame)
        second = Procedure.load(cycling_frame)
        first.metadata = bdf.Metadata(raw={"Name": "A"})
        second.metadata = bdf.Metadata(raw={"Name": "B"})
        caplog.clear()

        first.extend(second)

        assert first.metadata.raw == {"Name": "A"}  # type: ignore[attr-defined]
        warnings = [
            record for record in caplog.records if record.levelname == "WARNING"
        ]
        assert len(warnings) == 1
        assert "raw" in warnings[0].getMessage()
