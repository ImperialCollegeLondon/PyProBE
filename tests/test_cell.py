"""Tests for the Cell class."""
import copy
import os

import polars as pl
import pybamm
import pytest
from numpy.testing import assert_array_equal
from polars.testing import assert_frame_equal

import pyprobe
from pyprobe.cell import Cell


@pytest.fixture
def cell_instance(info_fixture):
    """Return a Cell instance."""
    return Cell(info=info_fixture)


def test_init(cell_instance, info_fixture):
    """Test the __init__ method."""
    expected_info = copy.copy(info_fixture)
    expected_info["color"] = "#ff00ff"
    assert cell_instance.info == expected_info
    assert cell_instance.procedure == {}

    info = {"not name": "test"}
    Cell(info=info)
    with pytest.warns(UserWarning):
        cell = Cell(info=info)
        assert cell.info["Name"] == "Default Name"


def test_make_cell_list(info_fixture):
    """Test the make_cell_list method."""
    filepath = "tests/sample_data/neware/Experiment_Record.xlsx"
    record_name = "sample_data_neware"
    cell_list = pyprobe.make_cell_list(filepath, record_name)
    expected_info = copy.copy(info_fixture)
    expected_info["color"] = "#ff00ff"
    assert cell_list[0].info == expected_info


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


def test_process_cycler_file(cell_instance, lazyframe_fixture):
    """Test the process_cycler_file method."""
    folder_path = "tests/sample_data/neware/"
    file_name = "sample_data_neware.xlsx"
    output_name = "sample_data_neware.parquet"
    cell_instance.process_cycler_file("neware", folder_path, file_name, output_name)
    expected_dataframe = lazyframe_fixture.collect()
    expected_dataframe = expected_dataframe.with_columns(
        pl.col("Date").dt.cast_time_unit("us")
    )
    saved_dataframe = pl.read_parquet(f"{folder_path}/{output_name}")
    saved_dataframe = saved_dataframe.select(pl.all().exclude("Temperature [C]"))
    assert_frame_equal(expected_dataframe, saved_dataframe)


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

    df.write_csv(f"{folder_path}/test_generic_file.csv")
    column_dict = {
        "Date": "Date",
        "T [*]": "Time [*]",
        "V [*]": "Voltage [*]",
        "I [*]": "Current [*]",
        "Q [*]": "Capacity [*]",
        "Count": "Step",
        "Temp [*]": "Temperature [C]",
    }
    cell_instance.process_generic_file(
        folder_path=folder_path,
        input_filename="test_generic_file.csv",
        output_filename="test_generic_file.parquet",
        column_dict=column_dict,
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
    assert_frame_equal(cell_instance.procedure[title].data, procedure_fixture.data)

    cell_instance.add_procedure(
        "Test_custom", input_path, file_name, readme_name="README_total_steps.yaml"
    )
    assert_frame_equal(
        cell_instance.procedure["Test_custom"].data, procedure_fixture.data
    )


def test_import_pybamm_solution(benchmark):
    """Test the import_pybamm_solution method."""
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
    )
    os.remove("tests/sample_data/pybamm.parquet")
