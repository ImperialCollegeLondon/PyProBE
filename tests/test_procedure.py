"""Module containing tests of the procedure class."""

from typing import Any

import numpy as np
import polars as pl
import pytest

from pyprobe.filters import Procedure
from tests.metadata_helpers import read_extras


def test_experiment(procedure_fixture, steps_fixture, benchmark):
    """Test creating an experiment."""

    def make_experiment():
        return procedure_fixture.experiment("Break-in Cycles")

    experiment = benchmark(make_experiment)
    assert experiment.data["Step ID"].unique().to_list() == steps_fixture[1]
    assert experiment.cycle_info == [(4, 7, 5)]

    experiment = procedure_fixture.experiment("Discharge Pulses")
    assert experiment.data["Step ID"].unique().to_list() == steps_fixture[2]
    assert experiment.cycle_info == [(9, 12, 10)]

    """Test filtering by multiple experiment names."""
    with pytest.warns(UserWarning):
        experiment = procedure_fixture.experiment("Break-in Cycles", "Discharge Pulses")

    assert experiment.cycle_info == []


def test_remove_experiment(procedure_fixture):
    """Test removing an experiment."""
    procedure_fixture.remove_experiment("Break-in Cycles")
    assert "Break-in Cycles" not in procedure_fixture.experiment_names
    assert procedure_fixture.data["Step ID"].unique().to_list() == [
        2,
        3,
        9,
        10,
        11,
        12,
    ]
    assert procedure_fixture.step_descriptions["Step"] == [1, 2, 3, 9, 10, 11, 12]


def test_init(procedure_fixture, step_descriptions_fixture):
    """Test initialising a procedure."""
    assert procedure_fixture.step_descriptions == step_descriptions_fixture


def test_experiment_no_description():
    """Test creating a procedure with no step descriptions."""
    procedure = Procedure.load(
        "tests/sample_data/neware/sample_data_neware.bdf.parquet",
        readme_path="tests/sample_data/neware/README_total_steps.yaml",
    )
    assert np.all(np.isnan(procedure.step_descriptions["Description"]))


def test_experiment_names(procedure_fixture, titles_fixture):
    """Test the experiment_names method."""
    assert procedure_fixture.experiment_names == titles_fixture


class TestProcedureLoad:
    """Tests for Procedure.load() classmethod."""

    def test_load_auto_guesses_readme_when_present(self, tmp_path) -> None:
        """Procedure.load auto-guesses README.yaml in parquet parent directory."""
        from pyprobe.filters import Procedure

        df = pl.DataFrame(
            {
                "Test Time / s": [0.0, 1.0, 2.0],
                "Current / A": [1.0, -1.0, 0.5],
                "Voltage / V": [3.7, 3.6, 3.8],
                "Step ID": [1, 1, 2],
            }
        )

        parquet_path = tmp_path / "data.bdf.parquet"
        df.write_parquet(parquet_path)

        readme_path = tmp_path / "README.yaml"
        readme_path.write_text("Initial Charge:\n  Steps: [1]\n")

        procedure = Procedure.load(parquet_path, readme_path=None)

        assert procedure.readme_dict is not None
        assert "Initial Charge" in procedure.readme_dict

    def test_load_no_readme_proceeds_without_definitions(self, tmp_path) -> None:
        """Procedure.load proceeds without README when file doesn't exist."""
        from pyprobe.filters import Procedure

        df = pl.DataFrame(
            {
                "Test Time / s": [0.0, 1.0, 2.0],
                "Current / A": [1.0, -1.0, 0.5],
                "Voltage / V": [3.7, 3.6, 3.8],
            }
        )

        parquet_path = tmp_path / "data.bdf.parquet"
        df.write_parquet(parquet_path)

        procedure = Procedure.load(parquet_path, readme_path=None)

        assert procedure.readme_dict == {}

    def test_load_explicit_readme_used(self, tmp_path) -> None:
        """Procedure.load uses explicit readme_path when provided."""
        from pyprobe.filters import Procedure

        df = pl.DataFrame(
            {
                "Test Time / s": [0.0, 1.0, 2.0],
                "Current / A": [1.0, -1.0, 0.5],
                "Voltage / V": [3.7, 3.6, 3.8],
                "Step ID": [1, 1, 2],
            }
        )

        parquet_path = tmp_path / "data.bdf.parquet"
        df.write_parquet(parquet_path)

        readme_path = tmp_path / "custom_readme.yaml"
        readme_path.write_text("My Experiment:\n  Steps: [1]\n")

        procedure = Procedure.load(parquet_path, readme_path=readme_path)

        assert "My Experiment" in procedure.readme_dict

    def test_load_missing_parquet_raises(self, tmp_path) -> None:
        """Procedure.load raises FileNotFoundError if parquet file doesn't exist."""
        from pyprobe.filters import Procedure

        missing_path = tmp_path / "missing.bdf.parquet"

        with pytest.raises(FileNotFoundError):
            Procedure.load(missing_path)

    def test_load_materialises_test_time_when_only_unix_time(self, tmp_path) -> None:
        """Test Time / s is in schema when parquet has only Unix Time / s."""
        df = pl.DataFrame(
            {
                "Unix Time / s": [1_700_000_000.0, 1_700_000_001.0, 1_700_000_002.0],
                "Current / A": [1.0, -1.0, 0.5],
                "Voltage / V": [3.7, 3.6, 3.8],
            }
        )
        parquet_path = tmp_path / "data.bdf.parquet"
        df.write_parquet(parquet_path)

        procedure = Procedure.load(parquet_path)

        assert "Test Time / s" in procedure.lf.collect_schema().names()

    def test_load_parquet_sets_path(self, tmp_path) -> None:
        """Procedure.load with a .parquet file sets _path to the resolved path."""
        df = pl.DataFrame(
            {
                "Test Time / s": [0.0, 1.0],
                "Current / A": [1.0, -1.0],
                "Voltage / V": [3.7, 3.6],
            }
        )
        parquet_path = tmp_path / "data.parquet"
        df.write_parquet(parquet_path)

        procedure = Procedure.load(parquet_path)

        assert procedure._path == parquet_path  # noqa: SLF001

    def test_load_csv_path_returns_none_path(self, tmp_path) -> None:
        """Procedure.load with a .csv file returns Procedure with _path=None."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "Test Time / s,Current / A,Voltage / V\n0.0,1.0,3.7\n1.0,-1.0,3.6\n"
        )

        procedure = Procedure.load(csv_path)

        assert procedure._path is None  # noqa: SLF001

    def test_load_lazyframe_returns_none_path(self) -> None:
        """Procedure.load with a LazyFrame returns Procedure with _path=None."""
        lf = pl.LazyFrame(
            {
                "Test Time / s": [0.0, 1.0],
                "Current / A": [1.0, -1.0],
                "Voltage / V": [3.7, 3.6],
            }
        )

        procedure = Procedure.load(lf)

        assert procedure._path is None  # noqa: SLF001

    def test_load_dataframe_returns_none_path(self) -> None:
        """Procedure.load with a DataFrame returns Procedure with _path=None."""
        df = pl.DataFrame(
            {
                "Test Time / s": [0.0, 1.0],
                "Current / A": [1.0, -1.0],
                "Voltage / V": [3.7, 3.6],
            }
        )

        procedure = Procedure.load(df)

        assert procedure._path is None  # noqa: SLF001

    def test_load_raises_on_missing_required_bdf_columns(self, tmp_path) -> None:
        """Procedure.load raises ValueError when required BDF columns are absent."""
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("Test Time / s,Some Column\n0.0,1.0\n1.0,2.0\n")

        with pytest.raises(ValueError, match="Required BDF column"):
            Procedure.load(csv_path)

    def test_path_propagates_through_filter(self, tmp_path) -> None:
        """_path is preserved after a filter operation."""
        df = pl.DataFrame(
            {
                "Test Time / s": [0.0, 1.0, 2.0],
                "Current / A": [1.0, -1.0, 0.5],
                "Voltage / V": [3.7, 3.6, 3.8],
                "Step ID": [1, 1, 2],
                "Step Count / 1": [0, 0, 1],
            }
        )
        parquet_path = tmp_path / "data.parquet"
        df.write_parquet(parquet_path)

        procedure = Procedure.load(parquet_path)
        step = procedure.step(1)

        assert step._path == parquet_path  # noqa: SLF001


class TestSyncMetadata:
    """Tests for Procedure.sync_metadata()."""

    def _make_procedure_with_metadata(
        self, tmp_path, initial_meta: dict[str, Any]
    ) -> "Procedure":
        """Helper: create parquet with metadata, load as Procedure."""
        import json

        import pyarrow.parquet as pq

        from pyprobe.io import _PARQUET_METADATA_KEY

        df = pl.DataFrame(
            {
                "Test Time / s": [0.0, 1.0],
                "Current / A": [1.0, -1.0],
                "Voltage / V": [3.7, 3.6],
            }
        )
        parquet_path = tmp_path / "data.parquet"
        table = df.to_arrow()
        existing_meta: dict[bytes, bytes] = table.schema.metadata or {}
        table = table.replace_schema_metadata(
            {**existing_meta, _PARQUET_METADATA_KEY: json.dumps(initial_meta).encode()}
        )
        pq.write_table(table, parquet_path)
        return Procedure.load(parquet_path)

    def test_sync_metadata_writes_new_key(self, tmp_path) -> None:
        """sync_metadata writes a new key to the backing file."""
        from pyprobe.io import MetadataManager

        proc = self._make_procedure_with_metadata(tmp_path, {"existing": "value"})
        read_extras(proc)["new_key"] = "new_value"

        proc.sync_metadata()

        assert proc._path is not None  # noqa: SLF001
        written = MetadataManager(proc._path).read_parquet()  # noqa: SLF001
        assert written["new_key"] == "new_value"
        assert written["existing"] == "value"

    def test_sync_metadata_protect_existing_raises_on_removed_key(
        self, tmp_path
    ) -> None:
        """sync_metadata(protect_existing=True) raises when a key is removed."""
        proc = self._make_procedure_with_metadata(tmp_path, {"key_a": "val_a"})
        del read_extras(proc)["key_a"]

        with pytest.raises(ValueError, match="absent from metadata"):
            proc.sync_metadata(protect_existing=True)

    def test_sync_metadata_protect_existing_raises_on_changed_value(
        self, tmp_path
    ) -> None:
        """sync_metadata(protect_existing=True) raises when a value changes."""
        proc = self._make_procedure_with_metadata(tmp_path, {"key_a": "original"})
        read_extras(proc)["key_a"] = "changed"

        with pytest.raises(ValueError, match="changed value"):
            proc.sync_metadata(protect_existing=True)

    def test_sync_metadata_protect_false_allows_value_change(self, tmp_path) -> None:
        """sync_metadata(protect_existing=False) allows changing an existing value."""
        from pyprobe.io import MetadataManager

        proc = self._make_procedure_with_metadata(tmp_path, {"key_a": "original"})
        read_extras(proc)["key_a"] = "changed"

        proc.sync_metadata(protect_existing=False)

        assert proc._path is not None  # noqa: SLF001
        written = MetadataManager(proc._path).read_parquet()  # noqa: SLF001
        assert written["key_a"] == "changed"

    def test_sync_metadata_raises_when_no_path(self) -> None:
        """sync_metadata raises RuntimeError when _path is None."""
        proc = Procedure.load(
            pl.DataFrame(
                {
                    "Test Time / s": [0.0],
                    "Current / A": [1.0],
                    "Voltage / V": [3.7],
                }
            )
        )

        with pytest.raises(RuntimeError, match="_path is None"):
            proc.sync_metadata()
