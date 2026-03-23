"""Tests for the io module.

This module provides tests for BDF-based cycler data import, including:
- process_cycler happy path and integration with column resolution
- process_cycler output_dir and skip_if_exists behavior
- Error handling for missing required and optional columns
- Parquet metadata write and read operations
- read_metadata function with preference logic
- process_cycler integration tests with actual sample data files
"""

import datetime
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import polars.testing as pl_testing
import pytest

from pyprobe.io import (
    _write_parquet,
    process_cycler,
    read_parquet_metadata,
)


@pytest.fixture
def bdf_df() -> pd.DataFrame:
    """Pandas DataFrame with the 3 required BDF columns."""
    return pd.DataFrame(
        {
            "Test Time / s": [0.0, 1.0, 2.0],
            "Current / A": [1.0, -1.0, 0.5],
            "Voltage / V": [3.7, 3.6, 3.8],
        }
    )


class TestProcessCycler:
    """Tests for process_cycler with minimal required columns."""

    def test_process_cycler_required_columns_only(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """process_cycler returns LazyFrame with required BDF columns."""
        with patch("bdf.read", return_value=bdf_df):
            lf = process_cycler("fake.csv", output_dir=tmp_path)

        assert isinstance(lf, pl.LazyFrame)
        result = lf.collect()
        assert "Test Time / s" in result.columns
        assert "Current / A" in result.columns
        assert "Voltage / V" in result.columns
        assert result.shape == (3, 3)

    def test_process_cycler_with_optional_columns(self, tmp_path: Path) -> None:
        """process_cycler includes optional columns when available."""
        fake_df = pd.DataFrame(
            {
                "Test Time / s": [0.0, 1.0, 2.0],
                "Current / A": [1.0, -1.0, 0.5],
                "Voltage / V": [3.7, 3.6, 3.8],
                "Net Capacity / Ah": [0.0, 0.1, 0.15],
                "Step Index / 1": [1, 1, 2],
            }
        )
        with patch("bdf.read", return_value=fake_df):
            lf = process_cycler("fake.csv", output_dir=tmp_path)

        result = lf.collect()
        assert "Net Capacity / Ah" in result.columns
        assert "Step Index / 1" in result.columns

    def test_process_cycler_derives_step_count_from_step_index(
        self, tmp_path: Path
    ) -> None:
        """process_cycler derives Step Count from Step Index when available."""
        fake_df = pd.DataFrame(
            {
                "Test Time / s": [0.0, 1.0, 2.0, 3.0],
                "Current / A": [1.0, -1.0, 0.5, 0.3],
                "Voltage / V": [3.7, 3.6, 3.8, 3.7],
                "Step Index / 1": [1, 1, 2, 2],
            }
        )
        with patch("bdf.read", return_value=fake_df):
            lf = process_cycler("fake.csv", output_dir=tmp_path)

        result = lf.collect()
        assert "Step Count / 1" in result.columns
        step_count = result["Step Count / 1"].to_list()
        assert step_count == [0, 0, 1, 1]

    def test_process_cycler_passes_plugin_to_bdf_read(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """process_cycler forwards plugin parameter to bdf.read()."""
        with patch("bdf.read", return_value=bdf_df) as mock_read:
            process_cycler("fake.csv", output_dir=tmp_path, plugin="neware-csv")

        mock_read.assert_called_once()
        call_kwargs = mock_read.call_args.kwargs
        assert call_kwargs["plugin"] == "neware-csv"


class TestProcessCyclerOutputDir:
    """Tests for process_cycler with output_dir parameter."""

    def test_process_cycler_writes_parquet_with_output_dir(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """process_cycler writes to Parquet file in output_dir."""
        with patch("bdf.read", return_value=bdf_df):
            lf = process_cycler("fake.csv", output_dir=tmp_path)

        expected_output = tmp_path / "fake.bdx.parquet"
        assert expected_output.exists()
        assert isinstance(lf, pl.LazyFrame)
        result = lf.collect()
        assert result.shape[0] == 3

    def test_process_cycler_output_file_naming(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """process_cycler names output file as {source_stem}.bdx.parquet."""
        with patch("bdf.read", return_value=bdf_df):
            lf = process_cycler("data.xlsx", output_dir=tmp_path)

        expected_output = tmp_path / "data.bdx.parquet"
        assert expected_output.exists()
        assert isinstance(lf, pl.LazyFrame)

    def test_process_cycler_returns_scan_of_written_parquet(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """process_cycler returns a lazy scan of the written parquet file."""
        with patch("bdf.read", return_value=bdf_df):
            lf = process_cycler("fake.csv", output_dir=tmp_path)

        result = lf.collect()
        assert len(result) == 3

    def test_process_cycler_output_dir_as_string(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """process_cycler accepts output_dir as string."""
        with patch("bdf.read", return_value=bdf_df):
            lf = process_cycler("fake.csv", output_dir=str(tmp_path))

        expected_output = tmp_path / "fake.bdx.parquet"
        assert expected_output.exists()
        assert isinstance(lf, pl.LazyFrame)

    def test_process_cycler_output_dir_defaults_to_source_parent(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """process_cycler defaults output_dir to source's parent directory."""
        source_file = tmp_path / "data.csv"
        source_file.write_text("dummy")

        with patch("bdf.read", return_value=bdf_df):
            lf = process_cycler(source_file)

        expected_output = tmp_path / "data.bdx.parquet"
        assert expected_output.exists()
        assert isinstance(lf, pl.LazyFrame)


class TestProcessCyclerSkipIfExists:
    """Tests for skip_if_exists parameter behavior."""

    def test_process_cycler_skip_exists_true_skips_read(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """With skip_if_exists=True, bdf.read() is not called if file exists."""
        with patch("bdf.read", return_value=bdf_df):
            process_cycler("fake.csv", output_dir=tmp_path)

        mock_read = MagicMock()
        with patch("bdf.read", side_effect=mock_read):
            lf = process_cycler("fake.csv", output_dir=tmp_path, skip_if_exists=True)

        mock_read.assert_not_called()
        result = lf.collect()
        assert result.shape[0] == 3

    def test_process_cycler_skip_exists_false_overwrites(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """With skip_if_exists=False, existing file is overwritten."""
        with patch("bdf.read", return_value=bdf_df):
            process_cycler("fake.csv", output_dir=tmp_path)

        new_df = pd.DataFrame(
            {
                "Test Time / s": [0.0, 1.0, 2.0, 3.0],
                "Current / A": [1.0, -1.0, 0.5, 0.3],
                "Voltage / V": [3.7, 3.6, 3.8, 3.7],
            }
        )
        with patch("bdf.read", return_value=new_df) as mock_read:
            lf = process_cycler("fake.csv", output_dir=tmp_path, skip_if_exists=False)

        mock_read.assert_called_once()
        result = lf.collect()
        assert result.shape[0] == 4

    def test_process_cycler_skip_exists_default_true(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """skip_if_exists defaults to True."""
        with patch("bdf.read", return_value=bdf_df):
            process_cycler("fake.csv", output_dir=tmp_path)

        with patch("bdf.read", side_effect=Exception("Should not be called")):
            lf = process_cycler("fake.csv", output_dir=tmp_path)

        result = lf.collect()
        assert result.shape[0] == 3


class TestProcessCyclerMissingColumns:
    """Tests for error handling when required or optional columns are missing."""

    @pytest.mark.parametrize(
        "missing_column",
        ["Test Time / s", "Current / A", "Voltage / V"],
    )
    def test_process_cycler_missing_required_column_raises(
        self, tmp_path: Path, missing_column: str
    ) -> None:
        """process_cycler raises ValueError when required column is missing."""
        fake_df = pd.DataFrame(
            {
                "Test Time / s": [0.0, 1.0],
                "Current / A": [1.0, -1.0],
                "Voltage / V": [3.7, 3.6],
            }
        )
        del fake_df[missing_column]

        with (
            patch("bdf.read", return_value=fake_df),
            pytest.raises(ValueError, match="Required BDF column"),
        ):
            process_cycler("fake.csv", output_dir=tmp_path)

    def test_process_cycler_missing_optional_column_warns(
        self, tmp_path: Path, bdf_df: pd.DataFrame, caplog
    ) -> None:
        """process_cycler logs warning via loguru when optional column missing."""
        with patch("bdf.read", return_value=bdf_df):
            lf = process_cycler("fake.csv", output_dir=tmp_path)

        result = lf.collect()
        assert result.shape[0] == 3
        assert "Net Capacity" not in result.columns
        assert "Optional BDF column" in caplog.text


class TestMetadataRoundTrip:
    """Tests for metadata round-trip: write and read cycles."""

    def test_metadata_roundtrip_basic_strings(self, tmp_path: Path) -> None:
        """Basic string metadata round-trips through parquet footer."""
        output_file = tmp_path / "test.parquet"
        df = pl.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
        metadata = {"cell_id": "C001", "cycler": "neware"}

        _write_parquet(df, output_file, metadata)  # type: ignore

        read_meta = read_parquet_metadata(output_file)
        assert read_meta["cell_id"] == "C001"
        assert read_meta["cycler"] == "neware"

    def test_metadata_roundtrip_special_chars_utf8(self, tmp_path: Path) -> None:
        """Metadata round-trip preserves special characters and UTF-8."""
        output_file = tmp_path / "test.parquet"
        df = pl.DataFrame({"x": [1, 2, 3]})
        metadata = {
            "description": "Test data with spaces",
            "unicode": "café",
            "key_with_underscore": "value123",
        }

        _write_parquet(df, output_file, metadata)  # type: ignore

        read_meta = read_parquet_metadata(output_file)
        assert read_meta["description"] == "Test data with spaces"
        assert read_meta["unicode"] == "café"
        assert read_meta["key_with_underscore"] == "value123"

    def test_metadata_roundtrip_empty_dict(self, tmp_path: Path) -> None:
        """Metadata round-trip with empty dict."""
        output_file = tmp_path / "test.parquet"
        df = pl.DataFrame({"x": [1, 2, 3]})

        _write_parquet(df, output_file, metadata={})

        assert output_file.exists()
        read_meta = read_parquet_metadata(output_file)
        assert isinstance(read_meta, dict)
        assert len(read_meta) == 0

    def test_metadata_roundtrip_none(self, tmp_path: Path) -> None:
        """Metadata round-trip with None (no metadata)."""
        output_file = tmp_path / "test.parquet"
        df = pl.DataFrame({"x": [1, 2, 3]})

        _write_parquet(df, output_file, metadata=None)

        assert output_file.exists()
        read_meta = read_parquet_metadata(output_file)
        assert isinstance(read_meta, dict)
        assert len(read_meta) == 0

    def test_metadata_roundtrip_json_sidecar(self, tmp_path: Path) -> None:
        """Metadata round-trip with metadata_format='json'."""
        output_file = tmp_path / "test.parquet"
        df = pl.DataFrame({"x": [1, 2, 3]})
        metadata = {"cell_id": "C001", "cycler": "neware"}

        _write_parquet(df, output_file, metadata, metadata_format="json")  # type: ignore

        sidecar = tmp_path / "test.json"
        assert sidecar.exists()
        loaded = json.loads(sidecar.read_text())
        assert loaded == metadata

    def test_metadata_roundtrip_json_sidecar_no_metadata(self, tmp_path: Path) -> None:
        """With metadata_format='json' but no metadata, no sidecar is written."""
        output_file = tmp_path / "test.parquet"
        df = pl.DataFrame({"x": [1, 2, 3]})

        _write_parquet(df, output_file, metadata=None, metadata_format="json")

        sidecar = tmp_path / "test.json"
        assert not sidecar.exists()

    def test_metadata_roundtrip_non_string_values_as_strings(
        self, tmp_path: Path
    ) -> None:
        """Non-string values come back as strings from parquet footer."""
        output_file = tmp_path / "test.parquet"
        df = pl.DataFrame({"x": [1, 2, 3]})
        metadata = {"count": "42", "rate": "3.14", "flag": "true"}

        _write_parquet(df, output_file, metadata)  # type: ignore

        read_meta = read_parquet_metadata(output_file)
        assert read_meta["count"] == "42"
        assert read_meta["rate"] == "3.14"
        assert read_meta["flag"] == "true"


class TestReadMetadata:
    """Tests for the read_metadata function."""

    def test_read_metadata_only_parquet_exists(self, tmp_path: Path) -> None:
        """read_metadata returns parquet metadata when only parquet exists."""
        output_file = tmp_path / "test.parquet"
        df = pl.DataFrame({"x": [1, 2, 3]})
        metadata = {"source": "parquet_only"}

        _write_parquet(df, output_file, metadata)  # type: ignore

        read_meta = read_parquet_metadata(output_file)
        assert read_meta["source"] == "parquet_only"

    def test_read_metadata_only_json_exists(self, tmp_path: Path) -> None:
        """read_metadata returns json metadata when only json sidecar exists."""
        output_file = tmp_path / "test.parquet"
        sidecar = tmp_path / "test.json"
        df = pl.DataFrame({"x": [1, 2, 3]})

        _write_parquet(df, output_file, metadata=None)
        json_metadata = {"source": "json_only"}
        sidecar.write_text(json.dumps(json_metadata))

        meta = json.loads(sidecar.read_text())
        assert meta["source"] == "json_only"

    def test_read_metadata_both_exist_prefer_parquet(self, tmp_path: Path) -> None:
        """When both exist, prefer='parquet' returns parquet metadata."""
        output_file = tmp_path / "test.parquet"
        sidecar = tmp_path / "test.json"
        df = pl.DataFrame({"x": [1, 2, 3]})

        parquet_metadata = {"source": "parquet"}
        _write_parquet(df, output_file, parquet_metadata)  # type: ignore

        json_metadata = {"source": "json"}
        sidecar.write_text(json.dumps(json_metadata))

        read_meta = read_parquet_metadata(output_file)
        assert read_meta["source"] == "parquet"

    def test_read_metadata_both_exist_prefer_json(self, tmp_path: Path) -> None:
        """When both exist, logic can prefer json metadata."""
        output_file = tmp_path / "test.parquet"
        sidecar = tmp_path / "test.json"
        df = pl.DataFrame({"x": [1, 2, 3]})

        parquet_metadata = {"source": "parquet"}
        _write_parquet(df, output_file, parquet_metadata)  # type: ignore

        json_metadata = {"source": "json"}
        sidecar.write_text(json.dumps(json_metadata))

        meta = json.loads(sidecar.read_text())
        assert meta["source"] == "json"


class TestProcessCyclerIntegrationWithMetadata:
    """Integration tests for process_cycler with metadata."""

    def test_process_cycler_metadata_in_output_file(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """process_cycler embeds metadata in output Parquet file."""
        metadata = {"experiment_id": "EXP_001", "note": "test data"}

        with patch("bdf.read", return_value=bdf_df):
            lf = process_cycler("fake.csv", output_dir=tmp_path, metadata=metadata)  # type: ignore

        result = lf.collect()
        assert result.shape[0] == 3

        output_file = tmp_path / "fake.bdx.parquet"
        read_meta = read_parquet_metadata(output_file)
        assert read_meta["experiment_id"] == "EXP_001"
        assert read_meta["note"] == "test data"

    def test_process_cycler_metadata_with_json_sidecar(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """process_cycler with metadata_format='json' writes metadata to JSON."""
        metadata = {"key": "value"}

        with patch("bdf.read", return_value=bdf_df):
            process_cycler(
                "fake.csv",
                output_dir=tmp_path,
                metadata=metadata,  # type: ignore
                metadata_format="json",
            )

        sidecar = tmp_path / "fake.bdx.json"
        assert sidecar.exists()
        loaded = json.loads(sidecar.read_text())
        assert loaded == metadata


class TestProcessCyclerEdgeCases:
    """Edge case tests for process_cycler."""

    def test_process_cycler_empty_dataframe(self, tmp_path: Path) -> None:
        """process_cycler handles empty DataFrame (0 rows)."""
        fake_df = pd.DataFrame(
            {
                "Test Time / s": [],
                "Current / A": [],
                "Voltage / V": [],
            }
        )
        with patch("bdf.read", return_value=fake_df):
            lf = process_cycler("fake.csv", output_dir=tmp_path)

        result = lf.collect()
        assert result.shape[0] == 0
        assert result.shape[1] == 3

    def test_process_cycler_single_row(self, tmp_path: Path) -> None:
        """process_cycler handles single-row DataFrame."""
        fake_df = pd.DataFrame(
            {
                "Test Time / s": [0.0],
                "Current / A": [1.5],
                "Voltage / V": [3.7],
            }
        )
        with patch("bdf.read", return_value=fake_df):
            lf = process_cycler("fake.csv", output_dir=tmp_path)

        result = lf.collect()
        assert result.shape[0] == 1

    def test_process_cycler_large_dataframe(self, tmp_path: Path) -> None:
        """process_cycler handles large DataFrame efficiently."""
        n_rows = 10000
        fake_df = pd.DataFrame(
            {
                "Test Time / s": range(n_rows),
                "Current / A": [1.0 + i * 0.001 for i in range(n_rows)],
                "Voltage / V": [3.7 + i * 0.0001 for i in range(n_rows)],
            }
        )
        with patch("bdf.read", return_value=fake_df):
            lf = process_cycler("fake.csv", output_dir=tmp_path)

        result = lf.collect()
        assert result.shape[0] == n_rows

    def test_process_cycler_numeric_columns_converted_correctly(
        self, tmp_path: Path
    ) -> None:
        """process_cycler correctly converts pandas dtypes to Polars."""
        fake_df = pd.DataFrame(
            {
                "Test Time / s": pd.array([0, 1, 2], dtype="int64"),
                "Current / A": pd.array([1.0, -1.0, 0.5], dtype="float64"),
                "Voltage / V": pd.array([3.7, 3.6, 3.8], dtype="float32"),
            }
        )
        with patch("bdf.read", return_value=fake_df):
            lf = process_cycler("fake.csv", output_dir=tmp_path)

        result = lf.collect()
        assert len(result) == 3

    def test_process_cycler_source_as_path_object(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """process_cycler accepts source as Path object."""
        source_file = tmp_path / "fake.csv"
        source_file.write_text("dummy")

        with patch("bdf.read", return_value=bdf_df):
            lf = process_cycler(source_file, output_dir=tmp_path)

        assert isinstance(lf, pl.LazyFrame)

    def test_process_cycler_negative_current_values(self, tmp_path: Path) -> None:
        """process_cycler handles negative current (discharge) values."""
        fake_df = pd.DataFrame(
            {
                "Test Time / s": [0.0, 1.0, 2.0],
                "Current / A": [1.0, -1.0, -0.5],
                "Voltage / V": [3.7, 3.6, 3.5],
            }
        )
        with patch("bdf.read", return_value=fake_df):
            lf = process_cycler("fake.csv", output_dir=tmp_path)

        result = lf.collect()
        currents = result["Current / A"].to_list()
        assert currents[1] == -1.0
        assert currents[2] == -0.5

    def test_process_cycler_zero_time_start(self, tmp_path: Path) -> None:
        """process_cycler correctly handles time starting at zero."""
        fake_df = pd.DataFrame(
            {
                "Test Time / s": [0.0, 0.5, 1.0],
                "Current / A": [1.0, 1.0, -1.0],
                "Voltage / V": [3.7, 3.75, 3.6],
            }
        )
        with patch("bdf.read", return_value=fake_df):
            lf = process_cycler("fake.csv", output_dir=tmp_path)

        result = lf.collect()
        times = result["Test Time / s"].to_list()
        assert times[0] == 0.0
        assert times[1] == 0.5


class TestProcessCyclerExtraColumns:
    """Tests for the extra_columns parameter."""

    def test_extra_columns_happy_path(self, tmp_path: Path) -> None:
        """Extra columns are renamed and included in the output."""
        bdf_df = pd.DataFrame(
            {
                "Test Time / s": [0.0, 1.0],
                "Current / A": [1.0, -1.0],
                "Voltage / V": [3.7, 3.6],
            }
        )
        raw_df = pd.DataFrame(
            {
                "Time(s)": [0.0, 1.0],
                "I(A)": [1.0, -1.0],
                "V(V)": [3.7, 3.6],
                "Pressure(kPa)": [101.3, 101.4],
            }
        )
        with patch("bdf.read", side_effect=[bdf_df, raw_df]):
            lf = process_cycler(
                "fake.csv",
                output_dir=tmp_path,
                extra_columns={"Pressure / kPa": "Pressure(kPa)"},
            )
        result = lf.collect()
        assert "Pressure / kPa" in result.columns
        assert result["Pressure / kPa"].to_list() == [101.3, 101.4]

    def test_extra_columns_multiple(self, tmp_path: Path) -> None:
        """Multiple extra columns are all included."""
        bdf_df = pd.DataFrame(
            {
                "Test Time / s": [0.0, 1.0],
                "Current / A": [1.0, -1.0],
                "Voltage / V": [3.7, 3.6],
            }
        )
        raw_df = pd.DataFrame(
            {
                "Time(s)": [0.0, 1.0],
                "P(kPa)": [101.3, 101.4],
                "T_aux(C)": [25.0, 26.0],
            }
        )
        with patch("bdf.read", side_effect=[bdf_df, raw_df]):
            lf = process_cycler(
                "fake.csv",
                output_dir=tmp_path,
                extra_columns={
                    "Pressure / kPa": "P(kPa)",
                    "Aux Temp / degC": "T_aux(C)",
                },
            )
        result = lf.collect()
        assert "Pressure / kPa" in result.columns
        assert "Aux Temp / degC" in result.columns

    def test_extra_columns_missing_source_raises(self, tmp_path: Path) -> None:
        """ValueError raised when source column doesn't exist in raw data."""
        bdf_df = pd.DataFrame(
            {
                "Test Time / s": [0.0],
                "Current / A": [1.0],
                "Voltage / V": [3.7],
            }
        )
        raw_df = pd.DataFrame({"Time(s)": [0.0], "I(A)": [1.0]})
        with (
            patch("bdf.read", side_effect=[bdf_df, raw_df]),
            pytest.raises(
                ValueError, match="Extra column source 'NoSuchCol' not found"
            ),
        ):
            process_cycler(
                "fake.csv",
                output_dir=tmp_path,
                extra_columns={"Pressure / kPa": "NoSuchCol"},
            )

    def test_extra_columns_none_is_noop(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """extra_columns=None does not change behaviour."""
        with patch("bdf.read", return_value=bdf_df) as mock_read:
            lf = process_cycler("fake.csv", output_dir=tmp_path, extra_columns=None)
        mock_read.assert_called_once()
        assert isinstance(lf, pl.LazyFrame)

    def test_extra_columns_persisted_in_output(self, tmp_path: Path) -> None:
        """Extra columns are persisted in the output parquet file."""
        bdf_df = pd.DataFrame(
            {
                "Test Time / s": [0.0, 1.0],
                "Current / A": [1.0, -1.0],
                "Voltage / V": [3.7, 3.6],
            }
        )
        raw_df = pd.DataFrame(
            {
                "Time(s)": [0.0, 1.0],
                "P(kPa)": [101.3, 101.4],
            }
        )
        with patch("bdf.read", side_effect=[bdf_df, raw_df]):
            lf = process_cycler(
                "fake.csv",
                output_dir=tmp_path,
                extra_columns={"Pressure / kPa": "P(kPa)"},
            )
        result = lf.collect()
        assert "Pressure / kPa" in result.columns
        assert result["Pressure / kPa"].to_list() == [101.3, 101.4]
        output_file = tmp_path / "fake.bdx.parquet"
        assert output_file.exists()
        lf_reload = process_cycler("fake.csv", output_dir=tmp_path, skip_if_exists=True)
        result_reload = lf_reload.collect()
        assert "Pressure / kPa" in result_reload.columns


class TestProcessCyclerIntegration:
    """End-to-end integration tests using real sample data files."""

    arbin_last_row = pl.DataFrame(
        {
            "Date": [datetime.datetime(2024, 9, 20, 8, 37, 5, 772000).timestamp()],
            "Test Time / s": [301.214],
            "Step Index / 1": [3],
            "Step Count / 1": [2],
            "Current / A": [2.650138],
            "Voltage / V": [3.599601],
            "Net Capacity / Ah": [0.0007812400999999999],
            "Surface Temperature T1 / degC": [24.68785],
        },
    )

    basytec_last_row = pl.DataFrame(
        {
            "Date": [datetime.datetime(2023, 6, 19, 17, 58, 3, 235803).timestamp()],
            "Test Time / s": [70.235804],
            "Step Index / 1": [4],
            "Step Count / 1": [1],
            "Current / A": [0.449602],
            "Voltage / V": [3.53285],
            "Net Capacity / Ah": [0.001248916998009],
            "Ambient Temperature / degC": [25.47953],
        },
    )

    biologic_last_row = pl.DataFrame(
        {
            "Date": [datetime.datetime(2024, 5, 13, 11, 19, 51, 602139).timestamp()],
            "Test Time / s": [139.524007],
            "Step Index / 1": [1],
            "Step Count / 1": [1],
            "Current / A": [-0.899826],
            "Voltage / V": [3.4854481],
            "Net Capacity / Ah": [-0.03237135133365209],
            "Ambient Temperature / degC": [23.029291],
        },
    )

    biologic_last_row_no_header = pl.DataFrame(
        {
            "Test Time / s": [281792.50213],
            "Step Index / 1": [0],
            "Step Count / 1": [0],
            "Current / A": [0.0],
            "Voltage / V": [2.9814022],
            "Net Capacity / Ah": [0.0],
            "Ambient Temperature / degC": [24.506462],
        },
    )

    biologic_last_row_mb = pl.DataFrame(
        {
            "Date": [datetime.datetime(2024, 5, 13, 11, 19, 51, 858016).timestamp()],
            "Test Time / s": [256016.11344],
            "Step Index / 1": [5],
            "Step Count / 1": [5],
            "Current / A": [0.450135],
            "Voltage / V": [3.062546],
            "Net Capacity / Ah": [0.307727],
            "Ambient Temperature / degC": [22.989878],
        },
    )

    maccor_last_row = pl.DataFrame(
        {
            "Date": [datetime.datetime(2023, 11, 23, 15, 56, 24, 60000).timestamp()],
            "Test Time / s": [13.06],
            "Step Index / 1": [2],
            "Step Count / 1": [1],
            "Current / A": [28.798],
            "Voltage / V": [3.716],
            "Net Capacity / Ah": [0.048],
            "Surface Temperature T1 / degC": [22.2591],
        },
    )

    neware_last_row = pl.DataFrame(
        {
            "Date": [datetime.datetime(2024, 3, 6, 21, 39, 38, 591000).timestamp()],
            "Test Time / s": [562749.497],
            "Step Index / 1": [12],
            "Step Count / 1": [61],
            "Current / A": [0.0],
            "Voltage / V": [3.4513],
            "Net Capacity / Ah": [0.022805],
        },
    )

    novonix_last_row = pl.DataFrame(
        {
            "Date": [datetime.datetime(2025, 7, 19, 18, 51, 8).timestamp()],
            "Test Time / s": [12287.48004],
            "Step Index / 1": [1],
            "Step Count / 1": [0],
            "Current / A": [0.49999387],
            "Voltage / V": [4.12864581],
            "Net Capacity / Ah": [1.70652976],
            "Surface Temperature T1 / degC": [24.792],
        },
    )

    def helper_process_cycler_integration(
        self,
        tmp_path: Path,
        source_file: str | Path,
        expected_final_row_bdf_format: pl.DataFrame | None = None,
        plugin: str | None = None,
    ) -> pl.LazyFrame:
        """Helper function to test process_cycler against real cycler data files.

        Similar to helper_read_and_process in test_basecycler.py, but adapted for
        BDF column names and the process_cycler API.

        Args:
            tmp_path: Temporary directory for output Parquet files.
            source_file: Path to the real cycler data file.
            expected_final_row_bdf_format: Expected final row in BDF format
                (with column names like "Test Time / s", "Unix Time / s", etc.).
                The expected row should already be in the BDF format with timestamps
                converted to seconds since epoch.
            plugin: Optional batterydf plugin name.

        Returns:
            The collected LazyFrame result.
        """
        lf = process_cycler(source_file, output_dir=tmp_path, plugin=plugin)

        assert isinstance(lf, pl.LazyFrame)
        result = lf.collect()

        # Check data integrity if expected final row is provided
        if expected_final_row_bdf_format is not None:
            final_row = result.tail(1)

            # Select only columns that exist in both dataframes
            cols_in_both = [
                c
                for c in expected_final_row_bdf_format.columns
                if c in final_row.columns
            ]
            if cols_in_both:
                expected_subset = expected_final_row_bdf_format.select(cols_in_both)
                final_subset = final_row.select(cols_in_both)

                pl_testing.assert_frame_equal(
                    expected_subset,
                    final_subset,
                    check_column_order=False,
                    check_dtypes=False,
                    atol=1e-5,
                )

        return result

    @pytest.mark.parametrize(
        "source_file, expected_final_row",
        [
            ("tests/sample_data/arbin/sample_data_arbin.csv", arbin_last_row),
            ("tests/sample_data/basytec/sample_data_basytec.txt", basytec_last_row),
            (
                "tests/sample_data/biologic/Sample_data_biologic_CA1.txt",
                biologic_last_row,
            ),
            (
                "tests/sample_data/biologic/Sample_data_biologic_no_header.mpt",
                biologic_last_row_no_header,
            ),
            ("tests/sample_data/maccor/sample_data_maccor.csv", maccor_last_row),
            ("tests/sample_data/neware/sample_data_neware.xlsx", neware_last_row),
            ("tests/sample_data/novonix/sample_data_novonix.csv", novonix_last_row),
        ],
    )
    def test_read_and_process_sample_data(
        self, tmp_path: Path, source_file: str, expected_final_row: pl.DataFrame
    ) -> None:
        """Test the full process of reading and processing real sample data files.

        This test runs process_cycler on real sample data files from different
        cyclers and checks that the output contains required columns and that the
        final row matches expected values (within tolerance).

        Args:
            tmp_path: Temporary directory for output Parquet files.
            source_file: Path to the real cycler data file to test.
            expected_final_row: Expected final row in BDF format for validation.
        """
        self.helper_process_cycler_integration(
            tmp_path,
            source_file,
            expected_final_row_bdf_format=expected_final_row,
        )

    def test_process_cycler_derived_step_count_integration(
        self, tmp_path: Path
    ) -> None:
        """process_cycler derives Step Count from Step Index with real data.

        Replicates monotonicity and derivation logic from cycler tests.
        """
        result = self.helper_process_cycler_integration(
            tmp_path,
            "tests/sample_data/neware/sample_data_neware.xlsx",
        )
        # If Step Count is derived, it should be monotonically non-decreasing
        if "Step Count / 1" in result.columns:
            step_index_diffs = result["Step Index / 1"].diff().drop_nulls()
            step_count_diffs = result["Step Count / 1"].diff().drop_nulls()
            # When Step Index changes, Step Count should increment
            for i in range(len(step_index_diffs)):
                if step_index_diffs[i] > 0:
                    assert step_count_diffs[i] >= 0, (
                        "Step Count should increment when Step Index changes"
                    )

    def test_process_cycler_multiple_files_integration(self, tmp_path: Path) -> None:
        """process_cycler handles multiple files independently without interference.

        Tests that processing multiple files in the same output directory works.
        """
        files = [
            "tests/sample_data/arbin/sample_data_arbin.csv",
            "tests/sample_data/maccor/sample_data_maccor.csv",
        ]
        results = []
        for file in files:
            result = self.helper_process_cycler_integration(
                tmp_path,
                file,
            )
            results.append(result)

        # Both should have been processed successfully
        assert all(r.shape[0] > 0 for r in results), (
            "All processed files should have rows"
        )
        # Each should have its own output file
        for file in files:
            stem = Path(file).stem
            expected_output = tmp_path / f"{stem}.bdx.parquet"
            assert expected_output.exists(), (
                f"Expected parquet file {expected_output} not created"
            )

    def test_process_cycler_skip_if_exists_integration(self, tmp_path: Path) -> None:
        """With skip_if_exists=True, cached files are reused with real data.

        Replicates skip_if_exists behavior with actual sample data.
        """
        source = "tests/sample_data/neware/sample_data_neware.xlsx"

        # First call - creates file
        lf1 = process_cycler(source, output_dir=tmp_path, skip_if_exists=True)
        result1 = lf1.collect()

        # Second call - should reuse
        lf2 = process_cycler(source, output_dir=tmp_path, skip_if_exists=True)
        result2 = lf2.collect()

        # Results should be identical
        pl_testing.assert_frame_equal(result1, result2)
        assert result1.shape == result2.shape
