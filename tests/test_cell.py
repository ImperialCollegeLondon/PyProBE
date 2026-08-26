"""Tests for the Cell class."""

import json
import logging
import os

import polars as pl
import pytest
from numpy.testing import assert_array_equal
from polars.testing import assert_frame_equal

import pyprobe
from pyprobe import cell
from pyprobe._version import __version__
from pyprobe.filters import Procedure
from pyprobe.protocol import Step, leaves, step_id_of
from tests.protocol_helpers import attach_protocol


@pytest.fixture
def cell_instance():
    """Return a Cell instance."""
    return cell.Cell()


def test_init(cell_instance):
    """Test Cell initialises with empty procedure dict and no info attribute."""
    assert cell_instance.procedure == {}
    assert not hasattr(cell_instance, "info")


def test_make_cell_list():
    """make_cell_list emits DeprecationWarning and returns a list of Cell objects."""
    filepath = "tests/sample_data/neware/Experiment_Record.xlsx"
    record_name = "sample_data_neware"
    with pytest.warns(DeprecationWarning, match="make_cell_list"):
        cell_list = pyprobe.make_cell_list(filepath, record_name)
    assert len(cell_list) == 3
    for c in cell_list:
        assert isinstance(c, cell.Cell)
        assert c.procedure == {}


@pytest.fixture
def caplog_fixture(caplog):
    """A fixture to capture log messages."""
    caplog.set_level(logging.INFO)
    return caplog


def test_import_pybamm_solution(benchmark, tmp_path):
    """Test the import_pybamm_solution method."""
    pybamm = pytest.importorskip("pybamm")
    parameter_values = pybamm.ParameterValues("Chen2020")
    spm = pybamm.lithium_ion.SPM()
    experiment = pybamm.Experiment(
        [
            (
                "Discharge at C/10 for 10 hours or until 3.3 V",
                "Rest for 1 hour",
                "Charge at 1 A until 4.1 V",
                "Hold at 4.1 V until 50 mA",
                "Rest for 1 hour",
            ),
        ]
        * 3
        + [
            "Discharge at 2A until 3.3 V",
            "Charge at 1 A until 4.1 V",
            "Discharge at 1A until 3.3 V",
        ],
    )
    sim = pybamm.Simulation(
        spm,
        experiment=experiment,
        parameter_values=parameter_values,
    )
    sol = sim.solve()
    cell_instance = cell.Cell()
    with pytest.warns(DeprecationWarning, match="import_pybamm_solution"):
        cell_instance.import_pybamm_solution(
            procedure_name="PyBaMM",
            pybamm_solutions=sol,
            experiment_names="Test",
        )
    assert_array_equal(
        cell_instance.procedure["PyBaMM"].experiment("Test").get("Voltage / V"),
        sol["Terminal voltage [V]"].entries,
    )
    assert_array_equal(
        cell_instance.procedure["PyBaMM"].experiment("Test").get("Current / A"),
        sol["Current [A]"].entries * -1,
    )
    assert_array_equal(
        cell_instance.procedure["PyBaMM"].experiment("Test").get("Test Time / s"),
        sol["Time [s]"].entries,
    )
    assert_array_equal(
        cell_instance.procedure["PyBaMM"].experiment("Test").get("Net Capacity / Ah"),
        sol["Discharge capacity [A.h]"].entries * -1,
    )

    # test filtering by cycle and step
    assert_array_equal(
        cell_instance.procedure["PyBaMM"]
        .experiment("Test")
        .cycle(1)
        .get("Voltage / V"),
        sol.cycles[1]["Terminal voltage [V]"].entries,
    )
    assert_array_equal(
        cell_instance.procedure["PyBaMM"]
        .experiment("Test")
        .cycle(1)
        .step(3)
        .get("Current / A"),
        sol.cycles[1].steps[3]["Current [A]"].entries * -1,
    )

    assert cell_instance.procedure["PyBaMM"].experiment("Test").step_descriptions[
        "Step"
    ] == [0, 1, 2, 3, 4]

    # test with multiple experiments from different simulations
    experiment2 = pybamm.Experiment(
        [
            (
                "Rest for 1 hour",
                "Charge at 1 A until 4.1 V",
                "Hold at 4.1 V until 50 mA",
                "Rest for 1 hour",
                "Discharge at 1C until 3.3 V",
            ),
        ]
        * 5,
    )
    sim2 = pybamm.Simulation(
        spm,
        experiment=experiment2,
        parameter_values=parameter_values,
    )

    sol2 = sim2.solve(starting_solution=sol)

    def add_two_experiments():
        with pytest.warns(DeprecationWarning):
            return cell_instance.import_pybamm_solution(
                procedure_name="PyBaMM two experiments",
                pybamm_solutions=[sol, sol2],
                experiment_names=["Test1", "Test2"],
            )

    benchmark(add_two_experiments)
    assert set(
        cell_instance.procedure["PyBaMM two experiments"].experiment_names,
    ) == {"Test1", "Test2"}
    assert_array_equal(
        cell_instance.procedure["PyBaMM two experiments"].get("Voltage / V"),
        sol2["Terminal voltage [V]"].entries,
    )
    assert_array_equal(
        cell_instance.procedure["PyBaMM two experiments"]
        .experiment("Test1")
        .get("Voltage / V"),
        sol["Terminal voltage [V]"].entries,
    )
    sol_length = len(sol["Terminal voltage [V]"].entries)
    assert_array_equal(
        cell_instance.procedure["PyBaMM two experiments"]
        .experiment("Test2")
        .get("Voltage / V"),
        sol2["Terminal voltage [V]"].entries[sol_length:],
    )

    # test reading and writing to parquet
    parquet_path = tmp_path / "pybamm.parquet"
    with pytest.warns(DeprecationWarning):
        cell_instance.import_pybamm_solution(
            procedure_name="PyBaMM",
            pybamm_solutions=sol,
            experiment_names="Test",
            output_data_path=str(parquet_path),
        )
    written_data = pl.read_parquet(parquet_path)
    assert_frame_equal(
        cell_instance.procedure["PyBaMM"].data,
        written_data,
        check_column_order=False,
    )


def test_archive(cell_instance, tmp_path, sample_data_neware_parquet):
    """Test archiving and loading a cell."""
    title = "Test"

    cell_instance.add_procedure(title, sample_data_neware_parquet)
    archive_path = tmp_path / "archive"
    with pytest.warns(DeprecationWarning, match="archive"):
        cell_instance.archive(str(archive_path))
    assert os.path.exists(archive_path)

    with pytest.warns(DeprecationWarning, match="load_archive"):
        cell_from_file = pyprobe.load_archive(str(archive_path))
    assert cell_instance.procedure.keys() == cell_from_file.procedure.keys()
    assert (
        cell_instance.procedure[title].column_definitions
        == cell_from_file.procedure[title].column_definitions
    )
    assert (
        cell_instance.procedure[title].step_descriptions
        == cell_from_file.procedure[title].step_descriptions
    )
    assert (
        cell_instance.procedure[title].cycle_info
        == cell_from_file.procedure[title].cycle_info
    )
    assert (
        cell_instance.procedure[title].experiment_names
        == cell_from_file.procedure[title].experiment_names
    )
    assert_frame_equal(
        cell_instance.procedure[title].lf,
        cell_from_file.procedure[title].lf,
    )

    # test loading an incorrect pyprobe version
    with open(archive_path / "metadata.json") as f:
        metadata = json.load(f)
    metadata["PyProBE Version"] = "0.0.0"
    with open(archive_path / "metadata.json", "w") as f:
        json.dump(metadata, f)
    with pytest.warns(
        (DeprecationWarning, UserWarning),
    ):
        cell_from_file = pyprobe.load_archive(str(archive_path))

    # test with zip file
    archive_zip_path = tmp_path / "archive.zip"
    with pytest.warns(DeprecationWarning, match="archive"):
        cell_instance.archive(str(archive_zip_path))
    assert os.path.exists(archive_zip_path)
    assert not os.path.exists(tmp_path / "archive")
    with pytest.warns(DeprecationWarning, match="load_archive"):
        cell_from_file = pyprobe.load_archive(str(archive_zip_path))
    assert cell_instance.procedure.keys() == cell_from_file.procedure.keys()
    assert (
        cell_instance.procedure[title].column_definitions
        == cell_from_file.procedure[title].column_definitions
    )
    assert (
        cell_instance.procedure[title].step_descriptions
        == cell_from_file.procedure[title].step_descriptions
    )
    assert (
        cell_instance.procedure[title].cycle_info
        == cell_from_file.procedure[title].cycle_info
    )
    assert (
        cell_instance.procedure[title].experiment_names
        == cell_from_file.procedure[title].experiment_names
    )
    assert_frame_equal(
        cell_instance.procedure[title].lf,
        cell_from_file.procedure[title].lf,
    )


def test_archive_round_trips_the_protocol(cell_instance, tmp_path):
    """An archived procedure keeps the experiments and the repeats of its protocol."""
    frame = pl.DataFrame(
        {
            "Test Time / s": [0.0, 1.0, 2.0, 3.0],
            "Current / A": [1.0, 1.0, -1.0, -1.0],
            "Voltage / V": [3.7, 3.8, 3.6, 3.5],
            "Step ID": [1, 2, 3, 4],
        },
    )
    procedure = Procedure.load(frame)
    attach_protocol(
        procedure,
        [
            Step(
                mode="group",
                description="Formation",
                steps=[
                    Step(description="Charge", tags=["step_id:1"]),
                    Step(description="Rest", tags=["step_id:2"]),
                ],
            ),
            Step(
                mode="group",
                description="Break-in",
                count=5,
                steps=[
                    Step(description="Discharge", tags=["step_id:3"]),
                    Step(description="Rest", tags=["step_id:4"]),
                ],
            ),
        ],
    )
    cell_instance.procedure["Test"] = procedure
    archive_path = tmp_path / "archive"
    with pytest.warns(DeprecationWarning, match="archive"):
        cell_instance.archive(str(archive_path))

    with pytest.warns(DeprecationWarning, match="load_archive"):
        loaded = pyprobe.load_archive(str(archive_path)).procedure["Test"]

    assert loaded.experiment_names == ["Formation", "Break-in"]
    assert loaded.cycle_info == [(3, 4, 5)]
    assert loaded.step_descriptions == {
        "Step": [1, 2, 3, 4],
        "Description": ["Charge", "Rest", "Discharge", "Rest"],
    }
    assert loaded.experiment("Formation").data["Step ID"].to_list() == [1, 2]


ARCHIVE_FRAME = pl.DataFrame(
    {
        "Test Time / s": [0.0, 1.0, 2.0, 3.0],
        "Current / A": [1.0, 1.0, -1.0, -1.0],
        "Voltage / V": [3.7, 3.8, 3.6, 3.5],
        "Step ID": [1, 2, 3, 4],
    },
)
"""The data that a hand-written archive holds."""


def write_released_archive(directory, procedure_record):
    """Write an archive in the shape that a released version wrote.

    Args:
        directory: The directory to write the archive into.
        procedure_record: The record of the one procedure of the archive,
            beside its data file name.

    Returns:
        str: The path of the archive.
    """
    directory.mkdir(parents=True, exist_ok=True)
    ARCHIVE_FRAME.write_parquet(directory / "Test.parquet")
    metadata = {
        "info": {"Name": "Cell A"},
        "procedure": {"Test": {"lf": "Test.parquet", **procedure_record}},
        "PyProBE Version": __version__,
    }
    with open(directory / "metadata.json", "w") as file:
        json.dump(metadata, file)
    return str(directory)


def test_load_archive_reads_an_archive_that_holds_no_protocol(tmp_path):
    """An archive without a protocol loads, and its procedure holds no experiment."""
    archive_path = write_released_archive(tmp_path / "archive", {"readme_dict": {}})

    with pytest.warns(DeprecationWarning, match="load_archive"):
        loaded = pyprobe.load_archive(archive_path).procedure["Test"]

    assert loaded.experiment_names == []
    assert loaded.info == {"Name": "Cell A"}
    assert loaded.data["Step ID"].to_list() == [1, 2, 3, 4]


def test_load_archive_builds_a_tree_from_the_released_definitions(tmp_path):
    """An archive of a released version gives the experiments it defines."""
    archive_path = write_released_archive(
        tmp_path / "archive",
        {
            "readme_dict": {
                "Formation": {
                    "Steps": [1, 2],
                    "Step Descriptions": ["Charge", "Rest"],
                    "Cycles": [],
                },
                "Break-in": {
                    "Steps": [3, 4],
                    "Step Descriptions": ["Discharge", "Rest"],
                    "Cycles": [[3, 4, 5]],
                },
            },
        },
    )

    with pytest.warns(DeprecationWarning, match="load_archive"):
        loaded = pyprobe.load_archive(archive_path).procedure["Test"]

    assert loaded.experiment_names == ["Formation", "Break-in"]
    assert loaded.cycle_info == [(3, 4, 5)]
    assert loaded.step_descriptions == {
        "Step": [1, 2, 3, 4],
        "Description": ["Charge", "Rest", "Discharge", "Rest"],
    }
    assert loaded.experiment("Break-in").data["Step ID"].to_list() == [3, 4]


def test_archive_keeps_a_written_column_definition(cell_instance, tmp_path):
    """A definition that a user wrote survives the archive and the load."""
    procedure = Procedure.load(ARCHIVE_FRAME)
    procedure.define_column("Voltage / V", "The voltage across the cell terminals")
    cell_instance.procedure["Test"] = procedure
    archive_path = tmp_path / "archive"
    with pytest.warns(DeprecationWarning, match="archive"):
        cell_instance.archive(str(archive_path))

    with pytest.warns(DeprecationWarning, match="load_archive"):
        loaded = pyprobe.load_archive(str(archive_path)).procedure["Test"]

    assert (
        loaded.column_definitions["Voltage / V"]
        == "The voltage across the cell terminals"
    )


def test_protocol_from_step_ranges_names_each_experiment():
    """The PyBaMM protocol holds one group per experiment, in step order."""
    step_ranges = pl.DataFrame(
        {
            "Experiment": ["Test2", "Test1"],
            "Step Range": [[3, 4], [0, 1, 2]],
        },
    )

    method = cell._protocol_from_step_ranges(step_ranges)

    assert [group.description for group in method] == ["Test1", "Test2"]
    assert [[step_id_of(leaf) for leaf in leaves(group)] for group in method] == [
        [0, 1, 2],
        [3, 4],
    ]


class TestCellAddProcedure:
    """Tests for Cell.add_procedure() method."""

    def test_add_procedure_with_parquet_path(self, sample_data_neware_parquet):
        """add_procedure loads and stores a Procedure from a parquet path."""
        c = cell.Cell()
        c.add_procedure("Sample", sample_data_neware_parquet)
        assert "Sample" in c.procedure
        assert isinstance(c.procedure["Sample"], Procedure)

    def test_add_procedure_with_procedure_object(self, sample_data_neware_parquet):
        """add_procedure stores an existing Procedure object directly."""
        proc = Procedure.load(sample_data_neware_parquet)
        c = cell.Cell()
        c.add_procedure("Sample", proc)
        assert c.procedure["Sample"] is proc

    def test_add_procedure_with_lazyframe(self, lazyframe_fixture):
        """add_procedure wraps a LazyFrame in a Procedure."""
        c = cell.Cell()
        c.add_procedure("Sample", lazyframe_fixture)
        assert isinstance(c.procedure["Sample"], Procedure)

    def test_add_procedure_with_dataframe(self, lazyframe_fixture):
        """add_procedure wraps a DataFrame in a Procedure."""
        c = cell.Cell()
        c.add_procedure("Sample", lazyframe_fixture.collect())
        assert isinstance(c.procedure["Sample"], Procedure)

    def test_add_procedure_bdf_incompatible_dataframe_raises_error(self):
        """add_procedure raises ValueError when DataFrame lacks required BDF columns."""
        import polars as pl

        df = pl.DataFrame({"time": [0.0, 1.0], "arbitrary_col": [1.0, 2.0]})
        c = cell.Cell()
        with pytest.raises(ValueError, match="Required"):
            c.add_procedure("Sample", df)


class TestCellDeprecations:
    """Tests that deprecated Cell methods emit DeprecationWarning."""

    def test_archive_emits_deprecation(self, tmp_path, sample_data_neware_parquet):
        """Cell.archive() emits DeprecationWarning."""
        c = cell.Cell()
        c.add_procedure("Sample", sample_data_neware_parquet)
        with pytest.warns(DeprecationWarning, match="archive"):
            c.archive(str(tmp_path / "arc"))

    def test_load_archive_emits_deprecation(self, tmp_path, sample_data_neware_parquet):
        """load_archive() emits DeprecationWarning."""
        c = cell.Cell()
        c.add_procedure("Sample", sample_data_neware_parquet)
        with pytest.warns(DeprecationWarning):
            c.archive(str(tmp_path / "arc"))
        with pytest.warns(DeprecationWarning, match="load_archive"):
            pyprobe.load_archive(str(tmp_path / "arc"))

    def test_make_cell_list_emits_deprecation(self):
        """make_cell_list() emits DeprecationWarning."""
        filepath = "tests/sample_data/neware/Experiment_Record.xlsx"
        with pytest.warns(DeprecationWarning, match="make_cell_list"):
            pyprobe.make_cell_list(filepath, "sample_data_neware")

    def test_import_pybamm_solution_emits_deprecation(self):
        """Cell.import_pybamm_solution() emits DeprecationWarning."""
        import contextlib

        pytest.importorskip("pybamm")
        c = cell.Cell()
        with (
            pytest.warns(DeprecationWarning, match="import_pybamm_solution"),
            contextlib.suppress(Exception),
        ):
            c.import_pybamm_solution("x", [], [])
