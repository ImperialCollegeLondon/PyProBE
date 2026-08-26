"""Module containing tests of the procedure class."""

import polars as pl
import pytest

from pyprobe.filters import Procedure


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
    )
    procedure.attach_legacy_readme(
        "tests/sample_data/neware/README_total_steps.yaml",
    )
    assert procedure.step_descriptions["Description"] == [None] * 11


def test_experiment_names(procedure_fixture, titles_fixture):
    """Test the experiment_names method."""
    assert procedure_fixture.experiment_names == titles_fixture


class TestProcedureLoad:
    """Tests for Procedure.load() classmethod."""

    def test_load_leaves_the_protocol_empty(self, tmp_path) -> None:
        """Procedure.load populates no experiment definitions from a README."""
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
        (tmp_path / "README.yaml").write_text("Initial Charge:\n  Steps: [1]\n")

        procedure = Procedure.load(parquet_path)

        assert procedure.experiment_names == []

    def test_load_missing_parquet_raises(self, tmp_path) -> None:
        """Procedure.load raises FileNotFoundError if parquet file doesn't exist."""
        from pyprobe.filters import Procedure

        missing_path = tmp_path / "missing.bdf.parquet"

        with pytest.raises(FileNotFoundError):
            Procedure.load(missing_path)

    def test_load_materialises_test_time_when_only_unix_time(self) -> None:
        """Test Time / s is in schema when a frame has only Unix Time / s."""
        df = pl.DataFrame(
            {
                "Unix Time / s": [1_700_000_000.0, 1_700_000_001.0, 1_700_000_002.0],
                "Current / A": [1.0, -1.0, 0.5],
                "Voltage / V": [3.7, 3.6, 3.8],
            }
        )

        procedure = Procedure.load(df)

        assert "Test Time / s" in procedure.lf.collect_schema().names()

    def test_load_parquet_with_only_unix_time_raises(self, tmp_path) -> None:
        """A parquet file without a literal Test Time column fails validation."""
        import bdf

        df = pl.DataFrame(
            {
                "Unix Time / s": [1_700_000_000.0, 1_700_000_001.0, 1_700_000_002.0],
                "Current / A": [1.0, -1.0, 0.5],
                "Voltage / V": [3.7, 3.6, 3.8],
            }
        )
        parquet_path = tmp_path / "data.bdf.parquet"
        df.write_parquet(parquet_path)

        with pytest.raises(bdf.BDFValidationError):
            Procedure.load(parquet_path)

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
        """Procedure.load raises where a raw file is missing a required column."""
        import bdf

        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("Test Time / s,Some Column\n0.0,1.0\n1.0,2.0\n")

        with pytest.raises(bdf.BDFValidationError):
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
