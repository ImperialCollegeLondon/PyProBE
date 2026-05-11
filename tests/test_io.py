"""Tests for the io module.

This module provides tests for BDF-based cycler data import, including:
- process_cycler happy path and integration with column resolution
- process_cycler output_path and skip_if_exists behavior
- Error handling for missing required and optional columns
- Parquet metadata write and read operations
- read_metadata function with preference logic
- process_cycler integration tests with actual sample data files
- process_generic with different DataFrame sources (polars, lazy, pandas)
"""

import datetime
import json
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import polars.testing as pl_testing
import pyarrow.parquet as pq
import pytest

from pyprobe.columns import BDF
from pyprobe.io import (
    attach_metadata,
    process_cycler,
    process_generic,
    read_metadata,
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
            result = process_cycler("fake.csv", output_path=tmp_path)

        assert isinstance(result, Path)
        result = pl.scan_parquet(result).collect()
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
            result = process_cycler("fake.csv", output_path=tmp_path)

        result = pl.scan_parquet(result).collect()
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
            result = process_cycler("fake.csv", output_path=tmp_path)

        result = pl.scan_parquet(result).collect()
        assert "Step Count / 1" in result.columns
        step_count = result["Step Count / 1"].to_list()
        assert step_count == [0, 0, 1, 1]

    def test_process_cycler_passes_plugin_to_bdf_read(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """process_cycler forwards plugin parameter to bdf.read()."""
        with patch("bdf.read", return_value=bdf_df) as mock_read:
            process_cycler("fake.csv", output_path=tmp_path, plugin="neware-csv")

        mock_read.assert_called_once()
        call_kwargs = mock_read.call_args.kwargs
        assert call_kwargs["plugin"] == "neware-csv"


class TestProcessCyclerOutputPath:
    """Tests for process_cycler with output_path parameter."""

    def test_process_cycler_writes_parquet_with_output_path(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """process_cycler writes to Parquet file at specified output_path."""
        with patch("bdf.read", return_value=bdf_df):
            result = process_cycler("fake.csv", output_path=tmp_path)

        expected_output = tmp_path / "fake.bdx.parquet"
        assert expected_output.exists()
        assert isinstance(result, Path)
        result = pl.scan_parquet(result).collect()
        assert result.shape[0] == 3

    def test_process_cycler_output_file_naming(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """process_cycler names output file as {source_stem}.bdx.parquet."""
        with patch("bdf.read", return_value=bdf_df):
            result = process_cycler("data.xlsx", output_path=tmp_path)

        expected_output = tmp_path / "data.bdx.parquet"
        assert expected_output.exists()
        assert isinstance(result, Path)

    def test_process_cycler_returns_path_to_written_parquet(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """process_cycler returns Path to the written parquet file."""
        with patch("bdf.read", return_value=bdf_df):
            result = process_cycler("fake.csv", output_path=tmp_path)

        result = pl.scan_parquet(result).collect()
        assert len(result) == 3

    def test_process_cycler_output_path_as_string(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """process_cycler accepts output_path as string."""
        with patch("bdf.read", return_value=bdf_df):
            result = process_cycler("fake.csv", output_path=str(tmp_path))

        expected_output = tmp_path / "fake.bdx.parquet"
        assert expected_output.exists()
        assert isinstance(result, Path)

    def test_process_cycler_output_path_defaults_to_source_parent(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """process_cycler defaults output_path to source's parent directory."""
        source_file = tmp_path / "data.csv"
        source_file.write_text("dummy")

        with patch("bdf.read", return_value=bdf_df):
            result = process_cycler(source_file)

        expected_output = tmp_path / "data.bdx.parquet"
        assert expected_output.exists()
        assert isinstance(result, Path)

    def test_process_cycler_accepts_source_as_path_object(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """process_cycler accepts source as Path object."""
        source_file = tmp_path / "fake.csv"
        source_file.write_text("dummy")

        with patch("bdf.read", return_value=bdf_df):
            result = process_cycler(source_file, output_path=tmp_path)

        assert isinstance(result, Path)


class TestProcessCyclerOverwriteData:
    """Tests for overwrite_data parameter behavior."""

    def test_process_cycler_overwrite_false_skips_read(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """With overwrite_data=False, bdf.read() is not called if file exists."""
        with patch("bdf.read", return_value=bdf_df):
            process_cycler("fake.csv", output_path=tmp_path)

        mock_read = MagicMock()
        with patch("bdf.read", side_effect=mock_read):
            result = process_cycler(
                "fake.csv", output_path=tmp_path, overwrite_data=False
            )

        mock_read.assert_not_called()
        result = pl.scan_parquet(result).collect()
        assert result.shape[0] == 3

    def test_process_cycler_overwrite_true_overwrites(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """With overwrite_data=True, existing file is overwritten."""
        with patch("bdf.read", return_value=bdf_df):
            process_cycler("fake.csv", output_path=tmp_path)

        new_df = pd.DataFrame(
            {
                "Test Time / s": [0.0, 1.0, 2.0, 3.0],
                "Current / A": [1.0, -1.0, 0.5, 0.3],
                "Voltage / V": [3.7, 3.6, 3.8, 3.7],
            }
        )
        with patch("bdf.read", return_value=new_df) as mock_read:
            result = process_cycler(
                "fake.csv", output_path=tmp_path, overwrite_data=True
            )

        mock_read.assert_called_once()
        result = pl.scan_parquet(result).collect()
        assert result.shape[0] == 4

    def test_process_cycler_overwrite_data_defaults_false(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """overwrite_data defaults to False (skip if exists)."""
        with patch("bdf.read", return_value=bdf_df):
            process_cycler("fake.csv", output_path=tmp_path)

        with patch("bdf.read", side_effect=Exception("Should not be called")):
            result = process_cycler("fake.csv", output_path=tmp_path)

        result = pl.scan_parquet(result).collect()
        assert result.shape[0] == 3


class TestProcessCyclerMissingColumns:
    """Tests for error handling when required or optional columns are missing."""

    @pytest.mark.parametrize(
        "missing_column",
        ["Current / A", "Voltage / V"],
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
            process_cycler("fake.csv", output_path=tmp_path)

    def test_process_cycler_missing_time_column_raises(self, tmp_path: Path) -> None:
        """Raise ValueError when both Unix Time and Test Time are missing."""
        fake_df = pd.DataFrame(
            {
                "Current / A": [1.0, -1.0],
                "Voltage / V": [3.7, 3.6],
            }
        )

        with (
            patch("bdf.read", return_value=fake_df),
            pytest.raises(ValueError, match="Required time column"),
        ):
            process_cycler("fake.csv", output_path=tmp_path)

    def test_process_cycler_missing_optional_column_warns(
        self, tmp_path: Path, bdf_df: pd.DataFrame, caplog
    ) -> None:
        """process_cycler logs warning via loguru when optional column missing."""
        with patch("bdf.read", return_value=bdf_df):
            result = process_cycler("fake.csv", output_path=tmp_path)

        result = pl.scan_parquet(result).collect()
        assert result.shape[0] == 3
        assert "Net Capacity" not in result.columns
        assert "Optional BDF column" in caplog.text


class TestProcessCyclerEdgeCases:
    """Edge case and boundary tests for process_cycler."""

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
            result = process_cycler("fake.csv", output_path=tmp_path)

        result = pl.scan_parquet(result).collect()
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
            result = process_cycler("fake.csv", output_path=tmp_path)

        result = pl.scan_parquet(result).collect()
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
            result = process_cycler("fake.csv", output_path=tmp_path)

        result = pl.scan_parquet(result).collect()
        assert result.shape[0] == n_rows

    def test_process_cycler_test_time_derived_from_unix_time(
        self, tmp_path: Path
    ) -> None:
        """Test that Test Time is derived from Unix Time."""
        fake_df = pd.DataFrame(
            {
                "Unix Time / s": [0, 1, 2],
                "Test Time / s": [0, 2, 4],  # different time, should be ignored
                "Current / A": [1.0, -1.0, 0.5],
                "Voltage / V": [3.7, 3.6, 3.8],
            }
        )
        with patch("bdf.read", return_value=fake_df):
            result = process_cycler("fake.csv", output_path=tmp_path)

        result = pl.scan_parquet(result).collect()
        assert "Test Time / s" in result.columns
        test_time = result["Test Time / s"].to_list()
        assert test_time == pytest.approx([0.0, 1.0, 2.0])


class TestProcessCyclerIntegration:
    """End-to-end integration tests using real sample data files."""

    arbin_last_row = pl.DataFrame(
        {
            "Unix Time / s": [
                datetime.datetime(2024, 9, 20, 8, 37, 5, 772000).timestamp()
            ],
            "Test Time / s": [301.214 - 30.0005],  # first datapoint at 30 s
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
            "Unix Time / s": [
                datetime.datetime(2023, 6, 19, 17, 58, 3, 235803).timestamp()
            ],
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
            "Unix Time / s": [
                datetime.datetime(2024, 5, 13, 11, 19, 51, 602139).timestamp()
            ],
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
            "Unix Time / s": [
                datetime.datetime(2024, 5, 13, 11, 19, 51, 858016).timestamp()
            ],
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
            "Unix Time / s": [
                datetime.datetime(2023, 11, 23, 15, 56, 24, 60000).timestamp()
            ],
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
            "Unix Time / s": [
                datetime.datetime(2024, 3, 6, 21, 39, 38, 591000).timestamp()
            ],
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
            "Unix Time / s": [datetime.datetime(2025, 7, 19, 18, 51, 8).timestamp()],
            "Test Time / s": [12288.0],
            "Step Count / 1": [1],
            "Step Index / 1": [0],
            "Current / A": [0.49999387],
            "Voltage / V": [4.12864581],
            "Net Capacity / Ah": [1.70652976],
            "Surface Temperature T1 / degC": [25.262],
            "Ambient Temperature / degC": [24.792],
        },
    )

    @pytest.mark.parametrize(
        "source_file, plugin, expected_final_row",
        [
            (
                "tests/sample_data/arbin/sample_data_arbin.csv",
                "arbin-csv",
                arbin_last_row,
            ),
            (
                "tests/sample_data/basytec/sample_data_basytec.txt",
                "basytec-txt",
                basytec_last_row,
            ),
            (
                "tests/sample_data/biologic/Sample_data_biologic_CA1.txt",
                "biologic-mpt",
                biologic_last_row,
            ),
            (
                "tests/sample_data/biologic/Sample_data_biologic_no_header.mpt",
                "biologic-mpt",
                biologic_last_row_no_header,
            ),
            (
                "tests/sample_data/maccor/sample_data_maccor.csv",
                "maccor-csv",
                maccor_last_row,
            ),
            (
                "tests/sample_data/neware/sample_data_neware.xlsx",
                "neware-xlsx",
                neware_last_row,
            ),
            (
                "tests/sample_data/novonix/sample_data_novonix.csv",
                "novonix-csv",
                novonix_last_row,
            ),
        ],
    )
    def test_read_and_process_sample_data(
        self,
        tmp_path: Path,
        source_file: str,
        plugin: str,
        expected_final_row: pl.DataFrame,
    ) -> None:
        """Test the full process of reading and processing real sample data files.

        This test runs process_cycler on real sample data files from different
        cyclers and checks that the output contains required columns and that the
        final row matches expected values (within tolerance).

        Args:
            tmp_path: Temporary directory for output Parquet files.
            source_file: Path to the real cycler data file to test.
            plugin: The cycler plugin name to use for parsing.
            expected_final_row: Expected final row in BDF format for validation.
        """
        result = process_cycler(source_file, output_path=tmp_path, plugin=plugin)

        assert isinstance(result, Path)
        result = pl.scan_parquet(result).collect()

        # Check data integrity if expected final row is provided
        if expected_final_row is not None:
            final_row = result.tail(1)

            pl_testing.assert_frame_equal(
                expected_final_row,
                final_row,
                check_column_order=False,
                check_dtypes=False,
                abs_tol=1e-5,
            )

    def test_process_cycler_timezone_shifts_unix_time(self, tmp_path: Path) -> None:
        """Test timezone parameter shifts Unix Time values when treating source.

        The basytec sample file contains tz-naive timestamps recorded in local time.
        Specifying timezone="Europe/Berlin" (CEST = UTC+2 in June 2023) causes those
        timestamps to be interpreted as Berlin local time and converted to UTC,
        producing Unix timestamps that are 7200 seconds earlier than when no
        timezone is given (i.e. when the naive timestamps are assumed to be UTC).
        """
        source = "tests/sample_data/basytec/sample_data_basytec.txt"
        utc_dir = tmp_path / "utc"
        berlin_dir = tmp_path / "berlin"
        utc_dir.mkdir()
        berlin_dir.mkdir()

        result_utc_path = process_cycler(
            source, output_path=utc_dir, plugin="basytec-txt"
        )
        result_berlin_path = process_cycler(
            source,
            output_path=berlin_dir,
            plugin="basytec-txt",
            timezone="Europe/Berlin",
        )

        unix_utc = (
            pl.scan_parquet(result_utc_path)
            .select("Unix Time / s")
            .collect()["Unix Time / s"]
        )
        unix_berlin = (
            pl.scan_parquet(result_berlin_path)
            .select("Unix Time / s")
            .collect()["Unix Time / s"]
        )

        # June 2023: CEST is UTC+2, so Berlin-local times are 7200 s ahead of UTC.
        # When the naive timestamps are reinterpreted as Berlin time, the resulting
        # UTC Unix timestamps are 7200 s earlier.
        offset = (unix_berlin - unix_utc).to_list()
        assert all(abs(v - (-7200.0)) < 1e-3 for v in offset)

    def test_process_cycler_overwrite_data_false_integration(
        self, tmp_path: Path
    ) -> None:
        """With overwrite_data=False, cached files are reused with real data."""
        source = "tests/sample_data/neware/sample_data_neware.xlsx"

        # First call - creates file
        result1_path = process_cycler(
            source, output_path=tmp_path, overwrite_data=False
        )
        result1 = pl.scan_parquet(result1_path).collect()

        # Second call - should reuse
        result2_path = process_cycler(
            source, output_path=tmp_path, overwrite_data=False
        )
        result2 = pl.scan_parquet(result2_path).collect()

        # Results should be identical
        pl_testing.assert_frame_equal(result1, result2)
        assert result1.shape == result2.shape


class TestCorruptedParquetMetadataRecovery:
    """Tests for handling corrupted Parquet metadata gracefully."""

    def test_metadata_manager_read_parquet_json_decode_error(
        self, tmp_path: Path
    ) -> None:
        """MetadataManager.read_parquet() raises ValueError for corrupted JSON."""
        from pyprobe.io import MetadataManager

        # Create a valid Parquet file with corrupted metadata
        output_file = tmp_path / "test.parquet"
        df = pl.DataFrame({"x": [1, 2, 3]})
        table = df.to_arrow()

        # Inject corrupted (non-JSON) metadata
        corrupted_metadata: dict[bytes, bytes] = {
            b"bdx_metadata": b"this is not valid json }{[",
        }
        table = table.replace_schema_metadata(corrupted_metadata)
        pq.write_table(table, output_file)

        # Try to read the corrupted metadata
        manager = MetadataManager(output_file)

        # Should raise ValueError due to corrupted metadata
        with pytest.raises(ValueError, match="invalid JSON"):
            manager.read_parquet()

    def test_metadata_manager_read_parquet_unicode_decode_error(
        self, tmp_path: Path
    ) -> None:
        """MetadataManager.read_parquet() raises ValueError for invalid UTF-8."""
        from pyprobe.io import MetadataManager

        output_file = tmp_path / "test.parquet"
        df = pl.DataFrame({"x": [1, 2, 3]})
        table = df.to_arrow()

        # Inject invalid UTF-8 sequence as metadata
        corrupted_metadata: dict[bytes, bytes] = {
            b"bdx_metadata": b"\x80\x81\x82\x83",
        }
        table = table.replace_schema_metadata(corrupted_metadata)
        pq.write_table(table, output_file)

        # Try to read the corrupted metadata
        manager = MetadataManager(output_file)

        # Should raise ValueError due to invalid encoding
        with pytest.raises(ValueError, match="invalid UTF-8"):
            manager.read_parquet()

    def test_metadata_manager_read_both_with_corrupted_parquet(
        self, tmp_path: Path
    ) -> None:
        """With corrupted parquet metadata and no sidecar, read_both raises."""
        from pyprobe.io import MetadataManager

        output_file = tmp_path / "test.parquet"
        df = pl.DataFrame({"x": [1, 2, 3]})
        table = df.to_arrow()

        # Parquet metadata is corrupted
        corrupted_metadata: dict[bytes, bytes] = {
            b"bdx_metadata": b"invalid json",
        }
        table = table.replace_schema_metadata(corrupted_metadata)
        pq.write_table(table, output_file)

        # No JSON sidecar exists
        manager = MetadataManager(output_file)

        # Should raise ValueError since preferred source is corrupted
        with pytest.raises(ValueError, match="corrupted"):
            manager.read_both(prefer="parquet")

    def test_metadata_manager_read_both_with_corrupted_parquet_but_valid_sidecar(
        self, tmp_path: Path
    ) -> None:
        """With corrupted parquet metadata but valid JSON sidecar, returns sidecar."""
        from pyprobe.io import MetadataManager

        output_file = tmp_path / "test.parquet"
        df = pl.DataFrame({"x": [1, 2, 3]})
        table = df.to_arrow()

        # Parquet metadata is corrupted
        corrupted_metadata: dict[bytes, bytes] = {
            b"bdx_metadata": b"invalid json",
        }
        table = table.replace_schema_metadata(corrupted_metadata)
        pq.write_table(table, output_file)

        # But JSON sidecar (test.json) has valid metadata
        json_metadata = {"cell_id": "C001", "source": "json"}
        (tmp_path / "test.json").write_text(json.dumps(json_metadata))

        # read_both should return the JSON metadata
        manager = MetadataManager(output_file)
        result = manager.read_both(prefer="json")

        assert result == json_metadata


class TestAttachMetadata:
    """Tests for attach_metadata function."""

    def test_attach_metadata_parquet_footer(self, tmp_path: Path) -> None:
        """attach_metadata stores metadata in parquet footer."""
        df = pl.DataFrame(
            {
                "Test Time / s": [0.0, 1.0, 2.0],
                "Current / A": [1.0, -1.0, 0.5],
                "Voltage / V": [3.7, 3.6, 3.8],
            }
        )
        output_file = tmp_path / "test.bdx.parquet"
        df.write_parquet(str(output_file))

        metadata = {"cell_id": "C001", "cycler": "neware"}
        attach_metadata(output_file, metadata, metadata_format="parquet")

        read_meta = read_metadata(output_file)
        assert read_meta["cell_id"] == "C001"
        assert read_meta["cycler"] == "neware"

    def test_attach_metadata_json_sidecar(self, tmp_path: Path) -> None:
        """attach_metadata creates JSON sidecar when format='json'."""
        df = pl.DataFrame(
            {
                "Test Time / s": [0.0, 1.0, 2.0],
                "Current / A": [1.0, -1.0, 0.5],
                "Voltage / V": [3.7, 3.6, 3.8],
            }
        )
        output_file = tmp_path / "test.bdx.parquet"
        df.write_parquet(str(output_file))

        metadata = {"cell_id": "C001", "cycler": "neware"}
        attach_metadata(output_file, metadata, metadata_format="json")

        sidecar = tmp_path / "test.bdx.json"
        assert sidecar.exists()
        loaded = json.loads(sidecar.read_text())
        assert loaded == metadata

    def test_attach_metadata_merges_with_existing(self, tmp_path: Path) -> None:
        """attach_metadata merges with existing metadata."""
        df = pl.DataFrame({"x": [1, 2, 3]})
        output_file = tmp_path / "test.bdx.parquet"
        df.write_parquet(str(output_file))

        attach_metadata(output_file, {"cell_id": "A"}, metadata_format="parquet")
        attach_metadata(output_file, {"batch": "1"}, metadata_format="parquet")

        read_meta = read_metadata(output_file)
        assert read_meta["cell_id"] == "A"
        assert read_meta["batch"] == "1"

    def test_attach_metadata_no_write_when_unchanged(self, tmp_path: Path) -> None:
        """attach_metadata skips file write when metadata is already up to date."""
        df = pl.DataFrame({"x": [1, 2, 3]})
        output_file = tmp_path / "test.bdx.parquet"
        df.write_parquet(str(output_file))

        metadata = {"cell_id": "C001"}
        attach_metadata(output_file, metadata, metadata_format="parquet")
        mtime_after_first = output_file.stat().st_mtime_ns

        attach_metadata(output_file, metadata, metadata_format="parquet")
        mtime_after_second = output_file.stat().st_mtime_ns

        assert mtime_after_first == mtime_after_second

    def test_attach_metadata_file_not_found(self, tmp_path: Path) -> None:
        """attach_metadata raises FileNotFoundError if file doesn't exist."""
        missing_file = tmp_path / "missing.parquet"
        with pytest.raises(FileNotFoundError):
            attach_metadata(missing_file, {"key": "value"})


class TestProcessCyclerGlob:
    """Tests for glob pattern handling in process_cycler."""

    def test_glob_concat_two_files(self, tmp_path: Path) -> None:
        """process_cycler concatenates multiple files matched by glob."""
        df1 = pd.DataFrame(
            {
                "Test Time / s": [0.0, 1.0],
                "Current / A": [1.0, -1.0],
                "Voltage / V": [3.7, 3.6],
            }
        )
        df2 = pd.DataFrame(
            {
                "Test Time / s": [2.0, 3.0],
                "Current / A": [0.5, 0.3],
                "Voltage / V": [3.8, 3.7],
            }
        )

        file1 = tmp_path / "data_1.csv"
        file2 = tmp_path / "data_2.csv"
        file1.write_text("dummy")
        file2.write_text("dummy")

        pattern = str(tmp_path / "data_*.csv")
        with patch(
            "bdf.read",
            side_effect=[df1, df2],
        ):
            result = process_cycler(
                pattern,
                output_path=tmp_path / "out.bdx.parquet",
            )

        result_df = pl.scan_parquet(result).collect()
        assert result_df.shape[0] == 4

    def test_glob_no_matching_files_raises(self, tmp_path: Path) -> None:
        """process_cycler raises FileNotFoundError when glob matches no files."""
        pattern = str(tmp_path / "nonexistent_*.csv")
        with pytest.raises(FileNotFoundError, match="No files found matching"):
            process_cycler(pattern, output_path=tmp_path)

    def test_glob_output_named_from_first_file(self, tmp_path: Path) -> None:
        """process_cycler output file is named from first sorted glob match."""
        df = pd.DataFrame(
            {
                "Test Time / s": [0.0],
                "Current / A": [1.0],
                "Voltage / V": [3.7],
            }
        )

        file1 = tmp_path / "zzz_1.csv"
        file1.write_text("dummy")

        pattern = str(tmp_path / "zzz_*.csv")
        with patch("bdf.read", return_value=df):
            result = process_cycler(pattern, output_path=tmp_path)

        assert isinstance(result, Path)


class TestProcessCyclerColumnMap:
    """Tests for column_map parameter in process_cycler."""

    def test_column_map_overrides_auto_resolved_with_custom_source(
        self, tmp_path: Path
    ) -> None:
        """column_map overrides auto-resolved BDF columns with different values."""
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
                "I(A)": [2.0, -2.0],  # Different values from BDF auto-resolution
                "V(V)": [3.7, 3.6],
                "Another Current(A)": [101.3, 101.4],
            }
        )

        with patch(
            "bdf.read",
            side_effect=[bdf_df, raw_df],
        ):
            result = process_cycler(
                "fake.csv",
                output_path=tmp_path / "out.bdx.parquet",
                column_map={"Current / A": "Another Current(A)"},
            )

        result_df = pl.scan_parquet(result).collect()
        # Verify that column_map override was used (values match raw_df, not bdf_df)
        assert "Current / A" in result_df.columns
        currents = result_df["Current / A"].to_list()
        assert currents[0] == 101.3  # From raw_df, proving override worked
        assert currents[1] == 101.4

    def test_column_map_appends_new_column(self, tmp_path: Path) -> None:
        """column_map can add new columns not in auto-resolved set."""
        bdf_df = pd.DataFrame(
            {
                "Test Time / s": [0.0, 1.0],
                "Current / A": [1.0, -1.0],
                "Voltage / V": [3.7, 3.6],
            }
        )
        raw_df = pd.DataFrame(
            {
                "Test Time / s": [0.0, 1.0],
                "Current / A": [1.0, -1.0],
                "Voltage / V": [3.7, 3.6],
                "Pressure(kPa)": [101.3, 101.4],
            }
        )

        with patch(
            "bdf.read",
            side_effect=[bdf_df, raw_df],
        ):
            result = process_cycler(
                "fake.csv",
                output_path=tmp_path / "out.bdx.parquet",
                column_map={"Pressure / kPa": "Pressure(kPa)"},
            )

        result_df = pl.scan_parquet(result).collect()
        assert "Pressure / kPa" in result_df.columns

    def test_column_map_missing_source_column_raises(self, tmp_path: Path) -> None:
        """column_map raises ValueError when source column not found."""
        bdf_df = pd.DataFrame(
            {
                "Test Time / s": [0.0],
                "Current / A": [1.0],
                "Voltage / V": [3.7],
            }
        )
        raw_df = pd.DataFrame(
            {
                "Test Time / s": [0.0],
                "Current / A": [1.0],
                "Voltage / V": [3.7],
            }
        )

        with (
            patch(
                "bdf.read",
                side_effect=[bdf_df, raw_df],
            ),
            pytest.raises(ValueError, match="column_map source 'NoSuchCol' not found"),
        ):
            process_cycler(
                "fake.csv",
                output_path=tmp_path / "out.bdx.parquet",
                column_map={"Pressure / kPa": "NoSuchCol"},
            )


class TestProcessCyclerCompression:
    """Tests for compression_priority parameter."""

    def test_default_compression_is_lz4(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """Default compression_priority='performance' uses lz4."""
        with patch("bdf.read", return_value=bdf_df):
            result = process_cycler("fake.csv", output_path=tmp_path)

        pf = pq.ParquetFile(result)
        assert pf.metadata.row_group(0).column(0).compression == "LZ4"

    def test_file_size_compression_is_zstd(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """compression_priority='file size' uses zstd."""
        with patch("bdf.read", return_value=bdf_df):
            result = process_cycler(
                "fake.csv",
                output_path=tmp_path / "out.bdx.parquet",
                compression_priority="file size",
            )

        pf = pq.ParquetFile(result)
        assert pf.metadata.row_group(0).column(0).compression == "ZSTD"


class TestProcessGeneric:
    """Tests for process_generic function with different DataFrame sources."""

    @pytest.mark.parametrize(
        "input_data",
        [
            pytest.param(
                pl.DataFrame(
                    {
                        "Time [s]": [0.0, 1.0, 2.0],
                        "Current [A]": [1.0, -1.0, 0.5],
                        "Voltage [V]": [3.7, 3.6, 3.8],
                    }
                ),
                id="polars_dataframe",
            ),
            pytest.param(
                pl.LazyFrame(
                    {
                        "Time [s]": [0.0, 1.0, 2.0],
                        "Current [A]": [1.0, -1.0, 0.5],
                        "Voltage [V]": [3.7, 3.6, 3.8],
                    }
                ),
                id="polars_lazyframe",
            ),
            pytest.param(
                pd.DataFrame(
                    {
                        "Time [s]": [0.0, 1.0, 2.0],
                        "Current [A]": [1.0, -1.0, 0.5],
                        "Voltage [V]": [3.7, 3.6, 3.8],
                    }
                ),
                id="pandas_dataframe",
            ),
        ],
    )
    def test_process_generic_accepts_different_sources(
        self, tmp_path: Path, input_data
    ) -> None:
        """process_generic accepts polars DataFrame, LazyFrame, and pandas DataFrame."""
        column_map: dict[str | BDF, str] = {
            "Test Time / s": "Time [s]",
            "Current / A": "Current [A]",
            "Voltage / V": "Voltage [V]",
        }
        output_path = tmp_path / "output.bdx.parquet"

        result = process_generic(input_data, column_map, output_path)

        assert isinstance(result, Path)
        assert result.exists()
        result_df = pl.scan_parquet(result).collect()
        assert "Test Time / s" in result_df.columns
        assert "Current / A" in result_df.columns
        assert "Voltage / V" in result_df.columns

    def test_process_generic_missing_required_column_raises(
        self, tmp_path: Path
    ) -> None:
        """process_generic raises when required column cannot be resolved."""
        df = pl.DataFrame(
            {
                "Time [s]": [0.0, 1.0],
                "Current [A]": [1.0, -1.0],
            }
        )

        column_map: dict[str | BDF, str] = {
            "Test Time / s": "Time [s]",
            "Current / A": "Current [A]",
        }
        output_path = tmp_path / "output.bdx.parquet"

        with pytest.raises(
            ValueError, match="Required BDF column 'Voltage' could not be resolved"
        ):
            process_generic(df, column_map, output_path)

    def test_process_generic_uses_column_map_keys(self, tmp_path: Path) -> None:
        """process_generic uses column_map keys to determine output column names."""
        df = pl.DataFrame(
            {
                "t": [0.0, 1.0],
                "i": [1.0, -1.0],
                "v": [3.7, 3.6],
            }
        )

        column_map: dict[str | BDF, str] = {
            "Test Time / s": "t",
            "Current / A": "i",
            "Voltage / V": "v",
        }
        output_path = tmp_path / "output.bdx.parquet"

        result = process_generic(df, column_map, output_path)
        result_df = pl.scan_parquet(result).collect()
        # Column names should match the keys, not the source names
        assert "Test Time / s" in result_df.columns
        assert "Current / A" in result_df.columns
        assert "Voltage / V" in result_df.columns
        assert "t" not in result_df.columns
        assert "i" not in result_df.columns
        assert "v" not in result_df.columns

    def test_process_generic_returns_path_to_output(self, tmp_path: Path) -> None:
        """process_generic returns Path to the written file."""
        df = pl.DataFrame(
            {
                "Test Time / s": [0.0, 1.0],
                "Current / A": [1.0, -1.0],
                "Voltage / V": [3.7, 3.6],
            }
        )

        column_map: dict[str | BDF, str] = {
            "Test Time / s": "Test Time / s",
            "Current / A": "Current / A",
            "Voltage / V": "Voltage / V",
        }
        output_path = tmp_path / "output.bdx.parquet"

        result = process_generic(df, column_map, output_path)

        assert isinstance(result, Path)
        assert result == output_path
        assert result.exists()

    def test_process_generic_compression_priority(self, tmp_path: Path) -> None:
        """process_generic respects compression_priority parameter."""
        df = pl.DataFrame(
            {
                "Test Time / s": [0.0, 1.0],
                "Current / A": [1.0, -1.0],
                "Voltage / V": [3.7, 3.6],
            }
        )

        column_map: dict[str | BDF, str] = {
            "Test Time / s": "Test Time / s",
            "Current / A": "Current / A",
            "Voltage / V": "Voltage / V",
        }
        output_path = tmp_path / "output.bdx.parquet"

        result = process_generic(
            df,
            column_map,
            output_path,
            compression_priority="file size",
        )

        pf = pq.ParquetFile(result)
        assert pf.metadata.row_group(0).column(0).compression == "ZSTD"


class TestHelperFunctions:
    """Direct unit tests for internal io module helper functions."""

    def test_resolve_glob_single_file(self, tmp_path: Path) -> None:
        """_resolve_glob returns single file as a list."""
        from pyprobe.io import _resolve_glob

        test_file = tmp_path / "test.csv"
        test_file.write_text("data")

        result = _resolve_glob(test_file)

        assert result == [test_file]

    def test_resolve_glob_pattern_multiple_files(self, tmp_path: Path) -> None:
        """_resolve_glob expands glob patterns in sorted order."""
        from pyprobe.io import _resolve_glob

        file1 = tmp_path / "file_01.csv"
        file2 = tmp_path / "file_02.csv"
        file3 = tmp_path / "file_10.csv"
        file1.write_text("data1")
        file2.write_text("data2")
        file3.write_text("data3")

        pattern = str(tmp_path / "file_*.csv")
        result = _resolve_glob(pattern)

        # Should be sorted numerically by glob
        assert len(result) == 3
        assert result == sorted([file1, file2, file3])

    def test_resolve_glob_pattern_no_matches_raises(self, tmp_path: Path) -> None:
        """_resolve_glob raises FileNotFoundError when glob matches no files."""
        from pyprobe.io import _resolve_glob

        pattern = str(tmp_path / "nonexistent_*.csv")

        with pytest.raises(FileNotFoundError, match="No files found matching"):
            _resolve_glob(pattern)

    def test_resolve_glob_with_path_object(self, tmp_path: Path) -> None:
        """_resolve_glob works with Path objects as input."""
        from pyprobe.io import _resolve_glob

        test_file = tmp_path / "test.csv"
        test_file.write_text("data")

        result = _resolve_glob(test_file)

        assert result == [test_file]

    def test_handle_existing_cached_file_exists(self, tmp_path: Path) -> None:
        """_handle_existing_cached_file returns path if file exists."""
        from pyprobe.io import _handle_existing_cached_file

        cached_file = tmp_path / "cached.parquet"
        cached_file.write_text("mock parquet data")

        result = _handle_existing_cached_file(cached_file)

        assert result == cached_file

    def test_handle_existing_cached_file_not_exists(self, tmp_path: Path) -> None:
        """_handle_existing_cached_file returns None if file doesn't exist."""
        from pyprobe.io import _handle_existing_cached_file

        missing_file = tmp_path / "missing.parquet"

        result = _handle_existing_cached_file(missing_file)

        assert result is None

    def test_build_column_map_exprs_with_bdf_enum_keys(self) -> None:
        """_build_column_map_exprs builds expressions for BDF enum keys."""
        from pyprobe.io import _build_column_map_exprs

        columns = ["time", "current", "voltage"]
        column_map = cast(
            dict[str | BDF, str],
            {
                BDF.TEST_TIME_SECOND: "time",
                BDF.CURRENT_AMPERE: "current",
                BDF.VOLTAGE_VOLT: "voltage",
            },
        )

        exprs = _build_column_map_exprs(columns, column_map)

        assert len(exprs) == 3
        # Verify expressions can be used in select
        df = pl.DataFrame(
            {"time": [0.0, 1.0], "current": [1.0, 2.0], "voltage": [3.7, 3.8]}
        )
        result = df.select(exprs)
        assert "Test Time / s" in result.columns
        assert "Current / A" in result.columns
        assert "Voltage / V" in result.columns

    def test_build_column_map_exprs_with_string_keys(self) -> None:
        """_build_column_map_exprs builds expressions for string BDF keys."""
        from pyprobe.io import _build_column_map_exprs

        columns = ["t", "i", "v"]
        column_map = cast(
            dict[str | BDF, str],
            {
                "Test Time / s": "t",
                "Current / A": "i",
                "Voltage / V": "v",
            },
        )

        exprs = _build_column_map_exprs(columns, column_map)

        assert len(exprs) == 3
        df = pl.DataFrame({"t": [0.0, 1.0], "i": [1.0, 2.0], "v": [3.7, 3.8]})
        result = df.select(exprs)
        assert result.columns == ["Test Time / s", "Current / A", "Voltage / V"]
        assert result.shape == (2, 3)

    def test_build_column_map_exprs_missing_source_column_raises(self) -> None:
        """_build_column_map_exprs raises ValueError for missing source column."""
        from pyprobe.io import _build_column_map_exprs

        columns = ["time", "current"]
        column_map = cast(
            dict[str | BDF, str],
            {"Test Time / s": "time", "Voltage / V": "missing_voltage"},
        )

        with pytest.raises(ValueError, match="not found in data"):
            _build_column_map_exprs(columns, column_map)

    def test_build_column_map_exprs_invalid_bdf_format_raises(self) -> None:
        """_build_column_map_exprs raises ValueError for invalid BDF string format."""
        from pyprobe.io import _build_column_map_exprs

        columns = ["time"]
        column_map = cast(
            dict[str | BDF, str], {"Invalid Format": "time"}
        )  # Missing "/ unit"

        with pytest.raises(ValueError):
            _build_column_map_exprs(columns, column_map)

    def test_concat_dataframes_same_schema(self) -> None:
        """_concat_dataframes concatenates DataFrames with same schema."""
        from pyprobe.io import _concat_dataframes

        df1 = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        df2 = pl.DataFrame({"a": [5, 6], "b": [7.0, 8.0]})

        result = _concat_dataframes([df1, df2])

        assert result.shape == (4, 2)
        assert result.columns == ["a", "b"]
        assert result["a"].to_list() == [1, 2, 5, 6]

    def test_concat_dataframes_different_schemas(self) -> None:
        """_concat_dataframes concatenates DataFrames with different schemas."""
        from pyprobe.io import _concat_dataframes

        df1 = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        df2 = pl.DataFrame({"b": [5.0, 6.0], "c": [7, 8]})

        result = _concat_dataframes([df1, df2])

        # Diagonal mode fills missing columns with null
        assert "a" in result.columns
        assert "b" in result.columns
        assert "c" in result.columns
        assert result.shape == (4, 3)
        # Check that nulls are filled correctly
        assert result["a"][2] is None or result["a"][2] != result["a"][2]  # null check
        assert result["c"][0] is None or result["c"][0] != result["c"][0]  # null check

    def test_concat_dataframes_empty_list(self) -> None:
        """_concat_dataframes handles empty list (should error)."""
        from pyprobe.io import _concat_dataframes

        with pytest.raises(Exception):  # polars concat will error on empty list
            _concat_dataframes([])

    def test_extract_column_map_columns_subset(self) -> None:
        """_extract_column_map_columns extracts and renames a subset of columns."""
        from pyprobe.io import _extract_column_map_columns

        df = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "current": [1.0, -1.0, 0.5],
                "voltage": [3.7, 3.6, 3.8],
                "temp": [25.0, 25.1, 25.2],
            }
        )
        column_map = cast(
            dict[str | BDF, str],
            {
                "Test Time / s": "time",
                "Current / A": "current",
                "Voltage / V": "voltage",
            },
        )

        result = _extract_column_map_columns(df, column_map)

        assert result.columns == ["Test Time / s", "Current / A", "Voltage / V"]
        assert result.shape == (3, 3)
        assert "temp" not in result.columns

    def test_extract_column_map_columns_with_bdf_enum(self) -> None:
        """_extract_column_map_columns works with BDF enum keys."""
        from pyprobe.io import _extract_column_map_columns

        df = pl.DataFrame(
            {
                "t": [0.0, 1.0],
                "i": [1.0, 2.0],
                "v": [3.7, 3.8],
            }
        )
        column_map = cast(
            dict[str | BDF, str],
            {
                BDF.TEST_TIME_SECOND: "t",
                BDF.CURRENT_AMPERE: "i",
                BDF.VOLTAGE_VOLT: "v",
            },
        )

        result = _extract_column_map_columns(df, column_map)

        assert "Test Time / s" in result.columns
        assert "Current / A" in result.columns
        assert "Voltage / V" in result.columns
        assert result.shape == (2, 3)

    def test_extract_column_map_columns_missing_source_raises(self) -> None:
        """_extract_column_map_columns raises ValueError for missing source column."""
        from pyprobe.io import _extract_column_map_columns

        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        column_map = cast(dict[str | BDF, str], {"Output / unit": "missing_col"})

        with pytest.raises(ValueError, match="not found in data"):
            _extract_column_map_columns(df, column_map)


class TestIsProbeFile:
    """Tests for is_pyprobe_file()."""

    def test_is_pyprobe_file_true_after_process_cycler(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """is_pyprobe_file returns True for a file written by process_cycler."""
        from pyprobe.io import is_pyprobe_file

        with patch("bdf.read", return_value=bdf_df):
            path = process_cycler("fake.csv", output_path=tmp_path)

        assert is_pyprobe_file(path) is True

    def test_is_pyprobe_file_false_for_plain_parquet(self, tmp_path: Path) -> None:
        """is_pyprobe_file returns False for a file without pyprobe key."""
        from pyprobe.io import is_pyprobe_file

        p = tmp_path / "plain.parquet"
        pl.DataFrame({"x": [1, 2]}).write_parquet(p)
        assert is_pyprobe_file(p) is False

    def test_is_pyprobe_file_raises_for_missing_file(self, tmp_path: Path) -> None:
        """is_pyprobe_file raises FileNotFoundError for non-existent path."""
        from pyprobe.io import is_pyprobe_file

        with pytest.raises(FileNotFoundError):
            is_pyprobe_file(tmp_path / "nonexistent.parquet")

    def test_pyprobe_footer_contains_version_and_written_at_after_process_cycler(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """Parquet footer after process_cycler has pyprobe.version and written_at."""
        from pyprobe.io import MetadataManager

        with patch("bdf.read", return_value=bdf_df):
            path = process_cycler("fake.csv", output_path=tmp_path)

        meta = MetadataManager(path).read_parquet()
        assert "pyprobe" in meta
        assert "version" in meta["pyprobe"]
        assert "written_at" in meta["pyprobe"]

    def test_pyprobe_footer_present_after_process_generic(self, tmp_path: Path) -> None:
        """Parquet footer after process_generic has pyprobe sub-dict."""
        from pyprobe.io import MetadataManager

        df = pl.DataFrame(
            {
                "Test Time / s": [0.0, 1.0],
                "Current / A": [1.0, -1.0],
                "Voltage / V": [3.7, 3.6],
            }
        )
        column_map: dict[str | BDF, str] = {
            "Test Time / s": "Test Time / s",
            "Current / A": "Current / A",
            "Voltage / V": "Voltage / V",
        }
        path = process_generic(df, column_map, tmp_path / "out.parquet")
        meta = MetadataManager(path).read_parquet()
        assert isinstance(meta.get("pyprobe"), dict)

    def test_process_cycler_raises_on_pyprobe_file_input(
        self, tmp_path: Path, bdf_df: pd.DataFrame
    ) -> None:
        """process_cycler raises ValueError when source is a PyProBE-written file."""
        with patch("bdf.read", return_value=bdf_df):
            path = process_cycler("fake.csv", output_path=tmp_path)

        with pytest.raises(ValueError, match="Procedure.load"):
            process_cycler(path, output_path=tmp_path / "other")
