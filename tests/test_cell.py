"""Tests for the Cell class."""

import copy
import datetime
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


@pytest.fixture
def cell_instance(info_fixture):
    """Return a Cell instance."""
    return cell.Cell(info=info_fixture)


def test_init(cell_instance, info_fixture):
    """Test the __init__ method."""
    expected_info = copy.copy(info_fixture)
    assert cell_instance.info == expected_info
    assert cell_instance.procedure == {}


def test_make_cell_list():
    """Test the make_cell_list method."""
    filepath = "tests/sample_data/neware/Experiment_Record.xlsx"
    record_name = "sample_data_neware"
    cell_list = pyprobe.make_cell_list(filepath, record_name)
    assert cell_list[0].info == {
        "Name": "Cell1",
        "Chemistry": "NMC622",
        "Nominal Capacity [Ah]": 5.0,
        "Start date": datetime.datetime(2024, 3, 20, 9, 3, 23),
    }
    assert cell_list[1].info == {
        "Name": "Cell2",
        "Chemistry": "NMC811",
        "Nominal Capacity [Ah]": 3.0,
        "Start date": datetime.datetime(2024, 3, 20, 9, 2, 23),
    }
    assert cell_list[2].info == {
        "Name": "Cell3",
        "Chemistry": "LFP",
        "Nominal Capacity [Ah]": 2.5,
        "Start date": datetime.datetime(2024, 3, 20, 9, 3, 23),
    }


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
    cell_instance = cell.Cell(info={})
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

    assert cell_instance.procedure["PyBaMM"].readme_dict["Test"]["Steps"] == [
        0,
        1,
        2,
        3,
        4,
    ]

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
    cell_instance.archive(str(archive_path))
    assert os.path.exists(archive_path)

    cell_from_file = pyprobe.load_archive(str(archive_path))
    assert cell_instance.procedure.keys() == cell_from_file.procedure.keys()
    assert cell_instance.info == cell_from_file.info
    assert (
        cell_instance.procedure[title].readme_dict
        == cell_from_file.procedure[title].readme_dict
    )
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
        UserWarning,
        match=(
            f"The PyProBE version used to archive the cell was "
            f"{metadata['PyProBE Version']}, the current version is "
            f"{__version__}. There may be compatibility"
            f" issues."
        ),
    ):
        cell_from_file = pyprobe.load_archive(str(archive_path))

    # test with zip file
    archive_zip_path = tmp_path / "archive.zip"
    cell_instance.archive(str(archive_zip_path))
    assert os.path.exists(archive_zip_path)
    assert not os.path.exists(tmp_path / "archive")
    cell_from_file = pyprobe.load_archive(str(archive_zip_path))
    assert cell_instance.procedure.keys() == cell_from_file.procedure.keys()
    assert cell_instance.info == cell_from_file.info
    assert (
        cell_instance.procedure[title].readme_dict
        == cell_from_file.procedure[title].readme_dict
    )
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
    assert_frame_equal(
        cell_instance.procedure[title].lf,
        cell_from_file.procedure[title].lf,
    )


class TestCellAddProcedure:
    """Tests for Cell.add_procedure() method."""

    def test_add_procedure_basic(self, cell_instance, mocker):
        """add_procedure processes cycler and loads procedure."""
        from pathlib import Path
        from unittest.mock import MagicMock

        source = "fake_cycler_file.xlsx"
        procedure_name = "TestProcedure"

        mock_path = Path("/tmp/output.bdx.parquet")
        mock_procedure = MagicMock()

        mock_process = mocker.patch(
            "pyprobe.io.process_cycler",
            return_value=mock_path,
        )
        mock_attach = mocker.patch("pyprobe.io.attach_metadata")
        mock_load = mocker.patch(
            "pyprobe.filters.Procedure.load",
            return_value=mock_procedure,
        )

        cell_instance.add_procedure(
            procedure_name,
            source,
            output_path="/tmp/out.bdx.parquet",
        )

        mock_process.assert_called_once()
        mock_attach.assert_called_once()
        mock_load.assert_called_once_with(mock_path, readme_path=None)
        assert cell_instance.procedure[procedure_name] == mock_procedure

    def test_add_procedure_merges_metadata(self, cell_instance, mocker):
        """add_procedure merges cell.info with provided metadata."""
        from pathlib import Path
        from unittest.mock import MagicMock

        source = "fake_cycler_file.xlsx"
        procedure_name = "TestProcedure"
        additional_metadata = {"batch": "B001"}

        mock_path = Path("/tmp/output.bdx.parquet")
        mock_procedure = MagicMock()

        mocker.patch(
            "pyprobe.io.process_cycler",
            return_value=mock_path,
        )
        mock_attach = mocker.patch("pyprobe.io.attach_metadata")
        mocker.patch(
            "pyprobe.filters.Procedure.load",
            return_value=mock_procedure,
        )

        cell_instance.add_procedure(
            procedure_name,
            source,
            metadata=additional_metadata,
        )

        expected_metadata = {**cell_instance.info, **additional_metadata}
        mock_attach.assert_called_once()
        call_args = mock_attach.call_args
        assert call_args[0][1] == expected_metadata

    def test_add_procedure_custom_readme(self, cell_instance, mocker):
        """add_procedure uses explicit readme_path when provided."""
        from pathlib import Path
        from unittest.mock import MagicMock

        source = "fake_cycler_file.xlsx"
        procedure_name = "TestProcedure"
        readme_path = Path("/custom/README.yaml")

        mock_path = Path("/tmp/output.bdx.parquet")
        mock_procedure = MagicMock()

        mocker.patch(
            "pyprobe.io.process_cycler",
            return_value=mock_path,
        )
        mocker.patch("pyprobe.io.attach_metadata")
        mock_load = mocker.patch(
            "pyprobe.filters.Procedure.load",
            return_value=mock_procedure,
        )

        cell_instance.add_procedure(
            procedure_name,
            source,
            readme_path=readme_path,
        )

        mock_load.assert_called_once()
        call_kwargs = mock_load.call_args.kwargs
        assert call_kwargs["readme_path"] == readme_path
