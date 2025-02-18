"""Tests for the Cell class."""

import copy
import datetime
import json
import logging
import os
import shutil

import polars as pl
import polars.testing as pl_testing
import pytest
from numpy.testing import assert_array_equal
from polars.testing import assert_frame_equal
from pydantic import ValidationError

import pyprobe
from pyprobe._version import __version__
from pyprobe.cell import Cell
from pyprobe.cyclers import basecycler


@pytest.fixture
def cell_instance(info_fixture):
    """Return a Cell instance."""
    return Cell(info=info_fixture)


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


def test_get_filename(info_fixture):
    """Test the _get_filename method."""
    filename_inputs = ["Name"]

    def filename(name):
        return f"Cell_named_{name}.xlsx"

    file = Cell._get_filename(info_fixture, filename, filename_inputs)
    assert file == "Cell_named_Test_Cell.xlsx"


def test_verify_filename():
    """Test the _verify_parquet method."""
    file = "path/to/sample_data_neware"
    assert Cell._verify_parquet(file) == "path/to/sample_data_neware.parquet"

    file = "path/to/sample_data_neware.parquet"
    assert Cell._verify_parquet(file) == "path/to/sample_data_neware.parquet"

    file = "path/to/sample_data_neware.csv"
    assert Cell._verify_parquet(file) == "path/to/sample_data_neware.parquet"

    file = "path/to/sample_data_neware*.parquet"
    with pytest.raises(ValueError):
        Cell._verify_parquet(file)


@pytest.fixture
def caplog_fixture(caplog):
    """A fixture to capture log messages."""
    caplog.set_level(logging.INFO)
    return caplog


def test_write_parquet(mocker):
    """Test the _write_parquet method."""
    data = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    importer = mocker.Mock()
    importer.get_pyprobe_dataframe.return_value = data
    for compression in ["uncompressed", "performance", "file size"]:
        Cell._write_parquet(
            importer=importer,
            output_data_path=compression + ".parquet",
            compression=compression,
        )
        written_data = pl.read_parquet(compression + ".parquet")
        pl_testing.assert_frame_equal(written_data, data)
        os.remove(compression + ".parquet")


def test_process_cycler_file(cell_instance, lazyframe_fixture, caplog_fixture, mocker):
    """Test the process_cycler_file method."""
    folder_path = "tests/sample_data/biologic/"
    file_name = "sample_data_biologic.mpt"
    output_name = "sample_data_biologic.parquet"
    mocker.patch("pyprobe.cell.Cell._get_data_paths")
    mocker.patch("pyprobe.cell.Cell._convert_to_parquet")

    cyclers = ["neware", "maccor", "biologic", "basytec"]

    for cycler in cyclers:
        mocker.patch(f"pyprobe.cyclers.{cycler}.{cycler.capitalize()}")
        cell_instance.process_cycler_file(
            cycler,
            folder_path,
            file_name,
            output_name,
            compression_priority="file size",
            overwrite_existing=True,
        )
        eval(f"pyprobe.cyclers.{cycler}.{cycler.capitalize()}.assert_called_once()")


def test_convert_to_parquet(mocker):
    """Test the _convert_to_parquet method."""
    mocker.patch("pyprobe.cell.Cell._write_parquet")
    importer = mocker.Mock()
    Cell._convert_to_parquet(
        importer=importer,
        output_data_path="tests/sample_data/neware/sample_data_neware.parquet",
        overwrite_existing=False,
    )
    pyprobe.cell.Cell._write_parquet.assert_not_called()

    Cell._convert_to_parquet(
        importer=importer,
        output_data_path="tests/sample_data/neware/sample_data_neware.parquet",
        overwrite_existing=True,
    )
    pyprobe.cell.Cell._write_parquet.assert_called_once()


def test_process_generic_file(cell_instance):
    """Test the process_generic_file method."""
    folder_path = "tests/sample_data/"
    df = pl.DataFrame(
        {
            "T [s]": [1.0, 2.0, 3.0],
            "V [V]": [4.0, 5.0, 6.0],
            "I [A]": [7.0, 8.0, 9.0],
            "Q [Ah]": [10.0, 11.0, 12.0],
            "Count": [1, 2, 3],
        }
    )

    column_importers = [
        basecycler.ConvertUnits("Time [s]", "T [*]"),
        basecycler.ConvertUnits("Voltage [V]", "V [*]"),
        basecycler.ConvertUnits("Current [A]", "I [*]"),
        basecycler.ConvertUnits("Capacity [Ah]", "Q [*]"),
        basecycler.CastAndRename("Step", "Count", pl.Int64),
    ]

    df.write_csv(f"{folder_path}/test_generic_file.csv")

    cell_instance.process_generic_file(
        folder_path=folder_path,
        input_filename="test_generic_file.csv",
        output_filename="test_generic_file.parquet",
        column_importers=column_importers,
    )
    expected_df = pl.DataFrame(
        {
            "Time [s]": [1.0, 2.0, 3.0],
            "Step": [1, 2, 3],
            "Event": [0, 1, 2],
            "Current [A]": [7.0, 8.0, 9.0],
            "Voltage [V]": [4.0, 5.0, 6.0],
            "Capacity [Ah]": [10.0, 11.0, 12.0],
        }
    )
    saved_df = pl.read_parquet(f"{folder_path}/test_generic_file.parquet")
    assert_frame_equal(expected_df, saved_df, check_column_order=False)

    with pytest.raises(ValidationError):
        cell_instance.process_generic_file(
            folder_path=folder_path,
            input_filename="test_generic_file.csv",
            output_filename="test_generic_file.parquet",
        )

    os.remove(f"{folder_path}/test_generic_file.csv")
    os.remove(f"{folder_path}/test_generic_file.parquet")


def test_add_procedure(cell_instance, procedure_fixture, benchmark):
    """Test the add_procedure method."""
    input_path = "tests/sample_data/neware/"
    file_name = "sample_data_neware.parquet"
    title = "Test"

    def add_procedure():
        return cell_instance.add_procedure(title, input_path, file_name)

    benchmark(add_procedure)
    assert_frame_equal(
        cell_instance.procedure[title].data,
        procedure_fixture.data,
        check_column_order=False,
    )

    cell_instance.add_procedure(
        "Test_custom", input_path, file_name, readme_name="README_total_steps.yaml"
    )
    assert_frame_equal(
        cell_instance.procedure["Test_custom"].data,
        procedure_fixture.data,
        check_column_order=False,
    )


def test_quick_add_procedure(cell_instance, procedure_fixture):
    """Test the quick_add_procedure method."""
    input_path = "tests/sample_data/neware/"
    file_name = "sample_data_neware.parquet"
    title = "Test"

    cell_instance.quick_add_procedure(title, input_path, file_name)
    assert_frame_equal(
        cell_instance.procedure[title].data,
        procedure_fixture.data,
        check_column_order=False,
    )


def test_import_pybamm_solution(benchmark):
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
            )
        ]
        * 3
        + [
            "Discharge at 1C until 3.3 V",
        ]
    )
    sim = pybamm.Simulation(
        spm, experiment=experiment, parameter_values=parameter_values
    )
    sol = sim.solve()
    cell_instance = Cell(info={})
    cell_instance.import_pybamm_solution(
        procedure_name="PyBaMM",
        pybamm_solutions=sol,
        experiment_names="Test",
    )
    assert_array_equal(
        cell_instance.procedure["PyBaMM"].experiment("Test").get("Voltage [V]"),
        sol["Terminal voltage [V]"].entries,
    )
    assert_array_equal(
        cell_instance.procedure["PyBaMM"].experiment("Test").get("Current [A]"),
        sol["Current [A]"].entries * -1,
    )
    assert_array_equal(
        cell_instance.procedure["PyBaMM"].experiment("Test").get("Time [s]"),
        sol["Time [s]"].entries,
    )
    assert_array_equal(
        cell_instance.procedure["PyBaMM"].experiment("Test").get("Capacity [Ah]"),
        sol["Discharge capacity [A.h]"].entries * -1,
    )

    # test filtering by cycle and step
    assert_array_equal(
        cell_instance.procedure["PyBaMM"]
        .experiment("Test")
        .cycle(1)
        .get("Voltage [V]"),
        sol.cycles[1]["Terminal voltage [V]"].entries,
    )
    assert_array_equal(
        cell_instance.procedure["PyBaMM"]
        .experiment("Test")
        .cycle(1)
        .step(3)
        .get("Current [A]"),
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
                "Discharge at 1C for 10 hours or until 3.3 V",
                "Rest for 1 hour",
                "Charge at 1 A until 4.1 V",
                "Hold at 4.1 V until 50 mA",
                "Rest for 1 hour",
            )
        ]
        * 5
    )
    sim2 = pybamm.Simulation(
        spm, experiment=experiment2, parameter_values=parameter_values
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
        cell_instance.procedure["PyBaMM two experiments"].experiment_names
    ) == set(["Test1", "Test2"])
    assert_array_equal(
        cell_instance.procedure["PyBaMM two experiments"].get("Voltage [V]"),
        sol2["Terminal voltage [V]"].entries,
    )
    assert_array_equal(
        cell_instance.procedure["PyBaMM two experiments"]
        .experiment("Test1")
        .get("Voltage [V]"),
        sol["Terminal voltage [V]"].entries,
    )
    sol_length = len(sol["Terminal voltage [V]"].entries)
    assert_array_equal(
        cell_instance.procedure["PyBaMM two experiments"]
        .experiment("Test2")
        .get("Voltage [V]"),
        sol2["Terminal voltage [V]"].entries[sol_length:],
    )

    # test reading and writing to parquet
    cell_instance.import_pybamm_solution(
        procedure_name="PyBaMM",
        pybamm_solutions=sol,
        experiment_names="Test",
        output_data_path="tests/sample_data/pybamm.parquet",
    )
    written_data = pl.read_parquet("tests/sample_data/pybamm.parquet")
    assert_frame_equal(
        cell_instance.procedure["PyBaMM"].data.drop(
            ["Procedure Time [s]", "Procedure Capacity [Ah]"]
        ),
        written_data,
        check_column_order=False,
    )
    os.remove("tests/sample_data/pybamm.parquet")


def test_archive(cell_instance):
    """Test archiving and loading a cell."""
    input_path = "tests/sample_data/neware/"
    file_name = "sample_data_neware.parquet"
    title = "Test"

    cell_instance.add_procedure(title, input_path, file_name)
    cell_instance.archive(input_path + "archive")
    assert os.path.exists(input_path + "archive")

    cell_from_file = pyprobe.load_archive(input_path + "archive")
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
        cell_instance.procedure[title].live_dataframe,
        cell_from_file.procedure[title].live_dataframe,
    )

    # test loading an incorrect pyprobe version
    with open(os.path.join(input_path, "archive", "metadata.json"), "r") as f:
        metadata = json.load(f)
    metadata["PyProBE Version"] = "0.0.0"
    with open(os.path.join(input_path, "archive", "metadata.json"), "w") as f:
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
        cell_from_file = pyprobe.load_archive(input_path + "archive")

    shutil.rmtree(input_path + "archive")

    # test with zip file
    cell_instance.archive(input_path + "archive.zip")
    assert os.path.exists(input_path + "archive.zip")
    assert not os.path.exists(input_path + "archive")
    cell_from_file = pyprobe.load_archive(input_path + "archive.zip")
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
        cell_instance.procedure[title].live_dataframe,
        cell_from_file.procedure[title].live_dataframe,
    )

    shutil.rmtree(input_path + "archive")


def test_get_data_paths(cell_instance):
    """Test _get_data_paths with string filename."""
    folder_path = "test/folder"
    filename = "test.csv"
    result = cell_instance._get_data_paths(folder_path, filename)
    assert result == os.path.join("test/folder", "test.csv")

    """Test _get_data_paths with function filename."""

    def filename_func(name):
        return f"cell_{name}.csv"

    folder_path = "test/folder"
    filename_inputs = ["Name"]
    result = cell_instance._get_data_paths(folder_path, filename_func, filename_inputs)
    assert result == os.path.join(
        "test/folder", f"cell_{cell_instance.info['Name']}.csv"
    )

    """Test _get_data_paths with function filename but missing inputs."""
    folder_path = "test/folder"
    with pytest.raises(
        ValueError, match="filename_inputs must be provided when filename is a function"
    ):
        cell_instance._get_data_paths(folder_path, filename_func)

    """Test _get_data_paths with absolute folder path."""
    folder_path = "/absolute/path"
    filename = "test.csv"
    result = cell_instance._get_data_paths(folder_path, filename)
    assert result == os.path.join("/absolute/path", "test.csv")

    """Test _get_data_paths with relative folder path."""
    cell_instance = Cell(
        info={
            "Name": "Test_Cell",
            "Chemistry": "NMC622",
        }
    )

    folder_path = "../relative/path"
    filename = "test.csv"
    result = cell_instance._get_data_paths(folder_path, filename)
    assert result == os.path.join("../relative/path", "test.csv")

    """Test _get_data_paths with complex filename function using multiple inputs."""

    def filename_func(name, chemistry):
        return f"cell_{name}_{chemistry}.csv"

    folder_path = "test/folder"
    filename_inputs = ["Name", "Chemistry"]
    result = cell_instance._get_data_paths(folder_path, filename_func, filename_inputs)
    expected = os.path.join(
        "test/folder",
        f"cell_{cell_instance.info['Name']}_{cell_instance.info['Chemistry']}.csv",
    )
    assert result == expected
