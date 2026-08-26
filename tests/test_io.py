"""Tests for the io module.

This module provides tests for BDF-based cycler data import, including:
- process_cycler as a composition of the load, the extend and the save
- process_cycler output_path resolution, including glob sources
- process_cycler integration tests with actual sample data files, including
  the overwrite_data behavior
- process_generic with different DataFrame sources (polars, lazy, pandas)
- is_pyprobe_file, backed by the BDF metadata sidecar
"""

import datetime
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pandas as pd
import polars as pl
import polars.testing as pl_testing
import pyarrow.parquet as pq
import pytest

from pyprobe.columns import BDF
from pyprobe.filters import Procedure
from pyprobe.io import (
    is_pyprobe_file,
    process_cycler,
    process_generic,
)


def _fake_procedure(n: int = 3, offset: float = 0.0) -> Procedure:
    """Build a minimal procedure over a frame with the given row count.

    Args:
        n: The number of rows to build.
        offset: The starting value of the test time column.

    Returns:
        Procedure: A procedure holding a Test Time, a Current, and a Voltage
            column.
    """
    return Procedure.load(
        pl.DataFrame(
            {
                "Test Time / s": [offset + i for i in range(n)],
                "Current / A": [1.0] * n,
                "Voltage / V": [3.7] * n,
            }
        ),
    )


class TestProcessCyclerComposition:
    """process_cycler composes the load, the extend and the save."""

    def test_forwards_load_kwargs_to_procedure_load(self, tmp_path: Path) -> None:
        """Every keyword outside the process_cycler signature reaches the load."""
        with patch(
            "pyprobe.filters.Procedure.load", return_value=_fake_procedure()
        ) as mock_load:
            process_cycler(
                "fake.csv",
                output_path=tmp_path,
                plugin="neware_csv",
                extra_columns={"Pressure(kPa)": "Ambient Pressure / kPa"},
            )

        mock_load.assert_called_once_with(
            Path("fake.csv"),
            plugin="neware_csv",
            extra_columns={"Pressure(kPa)": "Ambient Pressure / kPa"},
        )

    def test_multiple_files_extend_before_the_save(self, tmp_path: Path) -> None:
        """Every matched file loads, and every one but the first extends onto it."""
        file1 = tmp_path / "run_1.csv"
        file2 = tmp_path / "run_2.csv"
        file1.write_text("dummy")
        file2.write_text("dummy")

        procedures = [_fake_procedure(offset=0.0), _fake_procedure(offset=10.0)]
        output = tmp_path / "out.bdf.parquet"
        with patch("pyprobe.filters.Procedure.load", side_effect=procedures):
            process_cycler(str(tmp_path / "run_*.csv"), output_path=output)

        result = pl.read_parquet(output)
        assert result.height == 6

    def test_compression_priority_forwards_to_save(self, tmp_path: Path) -> None:
        """compression_priority reaches Table.save unchanged."""
        output = tmp_path / "out.bdf.parquet"
        with patch("pyprobe.filters.Procedure.load", return_value=_fake_procedure()):
            process_cycler(
                "fake.csv", output_path=output, compression_priority="file size"
            )

        pf = pq.ParquetFile(output)
        assert pf.metadata.row_group(0).column(0).compression == "ZSTD"


class TestProcessCyclerOutputPath:
    """process_cycler resolves the destination of the write."""

    def test_output_path_defaults_to_source_parent(self, tmp_path: Path) -> None:
        """With no output_path, the write lands beside the source."""
        source_file = tmp_path / "data.csv"
        source_file.write_text("dummy")

        with patch("pyprobe.filters.Procedure.load", return_value=_fake_procedure()):
            result = process_cycler(source_file)

        assert result == tmp_path / "data.bdf.parquet"
        assert result.exists()

    def test_output_path_as_a_directory_names_the_file_from_the_source(
        self, tmp_path: Path
    ) -> None:
        """A directory output_path takes the file name from the source stem."""
        with patch("pyprobe.filters.Procedure.load", return_value=_fake_procedure()):
            result = process_cycler("data.xlsx", output_path=tmp_path)

        assert result == tmp_path / "data.bdf.parquet"
        assert result.exists()

    def test_output_path_as_a_string(self, tmp_path: Path) -> None:
        """process_cycler accepts output_path as a string."""
        with patch("pyprobe.filters.Procedure.load", return_value=_fake_procedure()):
            result = process_cycler("fake.csv", output_path=str(tmp_path))

        assert result == tmp_path / "fake.bdf.parquet"
        assert result.exists()


class TestProcessCyclerGlob:
    """Tests for glob pattern handling in process_cycler."""

    def test_glob_no_matching_files_raises(self, tmp_path: Path) -> None:
        """process_cycler raises FileNotFoundError when glob matches no files."""
        pattern = str(tmp_path / "nonexistent_*.csv")
        with pytest.raises(FileNotFoundError, match="No files found matching"):
            process_cycler(pattern, output_path=tmp_path)

    def test_glob_output_named_from_first_file(self, tmp_path: Path) -> None:
        """process_cycler output file is named from first sorted glob match."""
        file1 = tmp_path / "zzz_1.csv"
        file1.write_text("dummy")

        pattern = str(tmp_path / "zzz_*.csv")
        with patch("pyprobe.filters.Procedure.load", return_value=_fake_procedure()):
            result = process_cycler(pattern, output_path=tmp_path)

        assert result == tmp_path / "zzz_1.bdf.parquet"


class TestProcessCyclerIntegration:
    """End-to-end integration tests using real sample data files."""

    arbin_last_row = pl.DataFrame(
        {
            "Unix Time / s": [
                datetime.datetime(2024, 9, 20, 8, 37, 5, 772000).timestamp()
            ],
            "Test Time / s": [301.214],
            "Step ID": [3],
            "Step Count / 1": [2],
            "Current / A": [2.650138],
            "Voltage / V": [3.599601],
            "Net Capacity / Ah": [0.00038040109999999997],
            "Temperature T1 / degC": [24.68785],
        },
    )

    basytec_last_row = pl.DataFrame(
        {
            "Test Time / s": [70.2358036666668],
            "Step ID": [4],
            "Step Count / 1": [1],
            "Current / A": [0.449601734416934],
            "Voltage / V": [3.53285012323902],
            "Net Capacity / Ah": [0.001248916998009],
        },
    )

    biologic_last_row = pl.DataFrame(
        {
            "Test Time / s": [139.5240066270344],
            "Step ID": [1],
            "Step Count / 1": [1],
            "Current / A": [-0.8998263500000001],
            "Voltage / V": [3.4854481],
            "Net Capacity / Ah": [-0.03237135133365207],
        },
    )

    biologic_last_row_no_header = pl.DataFrame(
        {
            "Test Time / s": [282092.50213],
            "Current / A": [0.0],
            "Voltage / V": [2.9814022],
            "Net Capacity / Ah": [0.0],
            "Step Count / 1": [0],
            "Step ID": [0],
        },
    )

    maccor_last_row = pl.DataFrame(
        {
            "Test Time / s": [15.06],
            "Current / A": [28.798],
            "Voltage / V": [3.716],
            "Unix Time / s": [datetime.datetime(2023, 11, 23, 15, 56, 24).timestamp()],
            "Net Capacity / Ah": [0.04024425555555555],
            "Step Count / 1": [2],
            "Temperature T1 / degC": [22.2591],
        },
    )

    neware_last_row = pl.DataFrame(
        {
            "Unix Time / s": [
                datetime.datetime(2024, 3, 6, 21, 39, 38, 591000).timestamp()
            ],
            "Test Time / s": [562784.5],
            "Step ID": [12],
            "Step Count / 1": [61],
            "Current / A": [0.0],
            "Voltage / V": [3.4513],
            "Net Capacity / Ah": [-0.01857910226387168],
        },
    )

    novonix_last_row = pl.DataFrame(
        {
            "Unix Time / s": [datetime.datetime(2025, 7, 19, 18, 51, 8).timestamp()],
            "Test Time / s": [12287.48004],
            "Step Count / 1": [1],
            "Step ID": [0],
            "Current / A": [0.49999387],
            "Voltage / V": [4.12864581],
            "Net Capacity / Ah": [1.70652976],
            "Temperature T1 / degC": [24.792],
            "Temperature T2 / degC": [25.262],
        },
    )

    @pytest.mark.parametrize(
        "source_file, plugin, expected_final_row",
        [
            (
                "tests/sample_data/arbin/sample_data_arbin.csv",
                "arbin_csv",
                arbin_last_row,
            ),
            (
                "tests/sample_data/basytec/sample_data_basytec.txt",
                "basytec_txt",
                basytec_last_row,
            ),
            (
                "tests/sample_data/biologic/Sample_data_biologic_CA1.txt",
                "biologic_mpt",
                biologic_last_row,
            ),
            (
                "tests/sample_data/biologic/Sample_data_biologic_no_header.mpt",
                "biologic_mpt",
                biologic_last_row_no_header,
            ),
            (
                "tests/sample_data/maccor/sample_data_maccor.csv",
                "maccor_csv",
                maccor_last_row,
            ),
            (
                "tests/sample_data/neware/sample_data_neware.xlsx",
                "neware_xlsx",
                neware_last_row,
            ),
            (
                "tests/sample_data/novonix/sample_data_novonix.csv",
                "novonix_csv",
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
        output_path = tmp_path / "output.bdf.parquet"

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
        output_path = tmp_path / "output.bdf.parquet"

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
        output_path = tmp_path / "output.bdf.parquet"

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
        output_path = tmp_path / "output.bdf.parquet"

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
        output_path = tmp_path / "output.bdf.parquet"

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


class TestIsProbeFile:
    """Tests for is_pyprobe_file(), backed by the BDF metadata sidecar."""

    def test_is_pyprobe_file_true_after_process_cycler(self, tmp_path: Path) -> None:
        """is_pyprobe_file returns True for a file written by process_cycler."""
        with patch("pyprobe.filters.Procedure.load", return_value=_fake_procedure()):
            path = process_cycler("fake.csv", output_path=tmp_path)

        assert is_pyprobe_file(path) is True

    def test_is_pyprobe_file_false_for_plain_parquet(self, tmp_path: Path) -> None:
        """is_pyprobe_file returns False for a file without a pyprobe sidecar key."""
        p = tmp_path / "plain.parquet"
        pl.DataFrame({"x": [1, 2]}).write_parquet(p)
        assert is_pyprobe_file(p) is False

    def test_is_pyprobe_file_raises_for_missing_file(self, tmp_path: Path) -> None:
        """is_pyprobe_file raises FileNotFoundError for non-existent path."""
        with pytest.raises(FileNotFoundError):
            is_pyprobe_file(tmp_path / "nonexistent.parquet")

    def test_process_cycler_raises_on_pyprobe_file_input(self, tmp_path: Path) -> None:
        """process_cycler raises ValueError when source is a PyProBE-written file."""
        with patch("pyprobe.filters.Procedure.load", return_value=_fake_procedure()):
            path = process_cycler("fake.csv", output_path=tmp_path)

        with pytest.raises(ValueError, match="Procedure.load"):
            process_cycler(path, output_path=tmp_path / "other")
