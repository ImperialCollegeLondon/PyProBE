"""Tests for the Cell class."""

import copy
import datetime
import json
import logging
import os
import shutil

import polars as pl
import pytest
from numpy.testing import assert_array_equal
from polars.testing import assert_frame_equal

import pyprobe
from pyprobe import cell
from pyprobe._version import __version__
from pyprobe.cyclers import basecycler
from pyprobe.readme_processor import process_readme


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


def test_get_filename(info_fixture):
    """Test the _get_filename method."""
    filename_inputs = ["Name"]

    def filename(name):
        return f"Cell_named_{name}.xlsx"

    file = cell.Cell._get_filename(info_fixture, filename, filename_inputs)
    assert file == "Cell_named_Test_Cell.xlsx"


@pytest.fixture
def caplog_fixture(caplog):
    """A fixture to capture log messages."""
    caplog.set_level(logging.INFO)
    return caplog


def test_process_cycler_file(cell_instance, mocker):
    """Test the process_cycler_file method."""
    output_name = "test.parquet"

    cyclers = ["neware", "maccor", "biologic", "basytec", "arbin"]
    file_paths = [
        "tests/sample_data/neware/sample_data_neware.xlsx",
        "tests/sample_data/maccor/sample_data_maccor.csv",
        "tests/sample_data/biologic/Sample_data_biologic_CA1.txt",
        "tests/sample_data/basytec/sample_data_basytec.txt",
        "tests/sample_data/arbin/sample_data_arbin.csv",
    ]

    for cycler, file in zip(cyclers, file_paths):
        process_mock = mocker.patch(
            f"pyprobe.cyclers.{cycler}.{cycler.capitalize()}.process"
        )
        folder_path = os.path.dirname(file)
        file_name = os.path.basename(file)
        cell_instance.process_cycler_file(
            cycler,
            folder_path,
            file_name,
            output_name,
            compression_priority="file size",
            overwrite_existing=True,
        )
        process_mock.assert_called_once()


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
        },
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
        },
    )
    saved_df = pl.read_parquet(f"{folder_path}/test_generic_file.parquet")
    assert_frame_equal(expected_df, saved_df, check_column_order=False)

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
        "Test_custom",
        input_path,
        file_name,
        readme_name="README_total_steps.yaml",
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
            ),
        ]
        * 3
        + [
            "Discharge at 1C until 3.3 V",
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
            ["Procedure Time [s]", "Procedure Capacity [Ah]"],
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
    with open(os.path.join(input_path, "archive", "metadata.json")) as f:
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
        "test/folder",
        f"cell_{cell_instance.info['Name']}.csv",
    )

    """Test _get_data_paths with function filename but missing inputs."""
    folder_path = "test/folder"
    with pytest.raises(
        ValueError,
        match="filename_inputs must be provided when filename is a function",
    ):
        cell_instance._get_data_paths(folder_path, filename_func)

    """Test _get_data_paths with absolute folder path."""
    folder_path = "/absolute/path"
    filename = "test.csv"
    result = cell_instance._get_data_paths(folder_path, filename)
    assert result == os.path.join("/absolute/path", "test.csv")

    """Test _get_data_paths with relative folder path."""
    cell_instance = cell.Cell(
        info={
            "Name": "Test_Cell",
            "Chemistry": "NMC622",
        },
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


def test_check_parquet_exists():
    """Test the _check_parquet_exists method."""
    cell.Cell._check_parquet("tests/sample_data/neware/sample_data_neware.parquet")

    with pytest.raises(
        FileNotFoundError,
        match="File tests/sample_data/sample_data_3.parquet does not exist.",
    ):
        cell.Cell._check_parquet("tests/sample_data/sample_data_3.parquet")

    with pytest.raises(
        ValueError,
        match="Files must be in parquet format. sample_data_neware.csv is not.",
    ):
        cell.Cell._check_parquet("tests/sample_data/neware/sample_data_neware.csv")


def test_import_data(cell_instance, mocker, caplog):
    """Test the import_data method."""
    procedure_name = "test_procedure"
    data_path = "tests/sample_data/neware/sample_data_neware.parquet"
    readme_path = "tests/sample_data/neware/README.yaml"

    sample_df = pl.LazyFrame(
        {
            "Time [s]": [1, 2, 3],
            "Voltage [V]": [4, 5, 6],
            "Current [A]": [7, 8, 9],
            "Capacity [Ah]": [10, 11, 12],
            "Step": [1, 2, 3],
            "Event": [0, 1, 2],
        },
    )
    mocker.patch("polars.scan_parquet", return_value=sample_df)

    cell_instance.import_data(procedure_name, data_path, readme_path)

    assert procedure_name in cell_instance.procedure
    assert (
        cell_instance.procedure[procedure_name].readme_dict
        == process_readme(readme_path).experiment_dict
    )
    assert_frame_equal(
        cell_instance.procedure[procedure_name].base_dataframe,
        sample_df,
    )

    # test with no readme
    procedure_name = "test_procedure_no_readme"
    cell_instance.import_data(procedure_name, data_path)
    assert procedure_name in cell_instance.procedure
    assert (
        cell_instance.procedure[procedure_name].readme_dict
        == process_readme(readme_path).experiment_dict
    )

    # test with no readme in the folder
    with caplog.at_level(logging.WARNING):
        procedure_name = "test_procedure_no_readme"
        mocker.patch("os.path.exists", return_value=False)
        cell_instance.import_data(procedure_name, data_path)
        assert cell_instance.procedure[procedure_name].readme_dict == {}
        assert caplog.messages[0] == (
            "No README file found for test_procedure_no_readme. Proceeding without"
            " README."
        )

    # Test with invalid readme path
    with pytest.raises(
        ValueError, match="README file tests/sample_data/README.yaml does not exist."
    ):
        cell_instance.import_data(
            procedure_name, data_path, "tests/sample_data/README.yaml"
        )


def test_import_from_cycler(cell_instance, mocker):
    """Test the import_from_cycler method."""
    procedure_name = "test_procedure"
    cycler = "neware"
    input_data_path = "tests/sample_data/neware/sample_data_neware.xlsx"
    output_data_path = "tests/sample_data/neware/sample_data_neware.parquet"
    readme_path = "tests/sample_data/neware/README.yaml"

    sample_df = pl.LazyFrame(
        {
            "Time [s]": [1, 2, 3],
            "Voltage [V]": [4, 5, 6],
            "Current [A]": [7, 8, 9],
            "Capacity [Ah]": [10, 11, 12],
            "Step": [1, 2, 3],
            "Event": [0, 1, 2],
        },
    )

    process_cycler_data = mocker.patch("pyprobe.cell.process_cycler_data")
    mocker.patch("polars.scan_parquet", return_value=sample_df)

    cell_instance.import_from_cycler(
        procedure_name,
        cycler,
        input_data_path,
        output_data_path,
        readme_path,
    )

    process_cycler_data.assert_called_once_with(
        cycler,
        input_data_path,
        output_data_path,
        column_importers=[],
        compression_priority="performance",
        overwrite_existing=False,
    )
    assert procedure_name in cell_instance.procedure
    assert (
        cell_instance.procedure[procedure_name].readme_dict
        == process_readme(readme_path).experiment_dict
    )
    assert_frame_equal(
        cell_instance.procedure[procedure_name].base_dataframe,
        sample_df,
    )

    # Test with no readme_path provided
    cell_instance.import_from_cycler(
        procedure_name,
        cycler,
        input_data_path,
        output_data_path,
    )
    assert (
        cell_instance.procedure[procedure_name].readme_dict
        == process_readme(readme_path).experiment_dict
    )

    # Test with no output_data_path provided
    cell_instance.import_from_cycler(
        procedure_name,
        cycler,
        input_data_path,
    )
    process_cycler_data.assert_called_with(
        cycler,
        input_data_path,
        None,
        column_importers=[],
        compression_priority="performance",
        overwrite_existing=False,
    )

    # Test with different compression priority
    cell_instance.import_from_cycler(
        procedure_name,
        cycler,
        input_data_path,
        output_data_path,
        readme_path,
        compression_priority="file size",
    )
    process_cycler_data.assert_called_with(
        cycler,
        input_data_path,
        output_data_path,
        column_importers=[],
        compression_priority="file size",
        overwrite_existing=False,
    )

    # Test with overwrite_existing set to True
    cell_instance.import_from_cycler(
        procedure_name,
        cycler,
        input_data_path,
        output_data_path,
        readme_path,
        overwrite_existing=True,
    )
    process_cycler_data.assert_called_with(
        cycler,
        input_data_path,
        output_data_path,
        column_importers=[],
        compression_priority="performance",
        overwrite_existing=True,
    )

    # Test with column_importers provided
    column_importers = [basecycler.ConvertUnits("Time [s]", "T [*]")]
    cell_instance.import_from_cycler(
        procedure_name,
        cycler,
        input_data_path,
        output_data_path,
        readme_path,
        column_importers=column_importers,
    )
    process_cycler_data.assert_called_with(
        cycler,
        input_data_path,
        output_data_path,
        column_importers=column_importers,
        compression_priority="performance",
        overwrite_existing=False,
    )


def test_process_cycler_data(mocker):
    """Test the process_cycler_file method."""
    output_name = "test.parquet"

    cyclers = ["neware", "maccor", "biologic", "basytec", "arbin"]
    file_paths = [
        "tests/sample_data/neware/sample_data_neware.xlsx",
        "tests/sample_data/maccor/sample_data_maccor.csv",
        "tests/sample_data/biologic/Sample_data_biologic_01_MB_CA1.txt",
        "tests/sample_data/basytec/sample_data_basytec.txt",
        "tests/sample_data/arbin/sample_data_arbin.csv",
    ]

    for cycler, file in zip(cyclers, file_paths):
        process_patch = mocker.patch(
            f"pyprobe.cyclers.{cycler}.{cycler.capitalize()}.process"
        )
        cell.process_cycler_data(
            cycler,
            file,
            output_name,
            overwrite_existing=True,
        )
        process_patch.assert_called_once()


def test_process_cycler_data_generic():
    """Test the process_generic_file method."""
    data_path = "tests/sample_data/test_generic_file.csv"
    df = pl.DataFrame(
        {
            "T [s]": [1.0, 2.0, 3.0],
            "V [V]": [4.0, 5.0, 6.0],
            "I [A]": [7.0, 8.0, 9.0],
            "Q [Ah]": [10.0, 11.0, 12.0],
            "Count": [1, 2, 3],
        },
    )

    column_importers = [
        basecycler.ConvertUnits("Time [s]", "T [*]"),
        basecycler.ConvertUnits("Voltage [V]", "V [*]"),
        basecycler.ConvertUnits("Current [A]", "I [*]"),
        basecycler.ConvertUnits("Capacity [Ah]", "Q [*]"),
        basecycler.CastAndRename("Step", "Count", pl.Int64),
    ]

    df.write_csv(data_path)

    cell.process_cycler_data(
        cycler_type="generic",
        input_data_path=data_path,
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
        },
    )
    saved_df = pl.read_parquet(data_path.replace(".csv", ".parquet"))
    assert_frame_equal(expected_df, saved_df, check_column_order=False)

    with pytest.raises(ValueError):
        cell.process_cycler_data(
            cycler_type="generic",
            input_data_path=data_path,
        )

    os.remove(data_path)
    os.remove(data_path.replace(".csv", ".parquet"))
