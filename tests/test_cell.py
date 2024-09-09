"""Tests for the Cell class."""
import copy
import os

import polars as pl
import pybamm
import pytest
from polars.testing import assert_frame_equal

import pyprobe
from pyprobe.cell import Cell, ReadmeModel


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
            "Cycle": [0, 0, 0],
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


def test_set_color_scheme(cell_instance):
    """Test the set_color_scheme method."""
    assert cell_instance.set_color_scheme(5) == [
        "#ff00ff",
        "#0080ff",
        "#00db21",
        "#f03504",
        "#a09988",
    ]


def test_process_readme(cell_instance, titles_fixture, benchmark):
    """Test processing a readme file in yaml format."""
    expected_steps = [
        [1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]

    def _process_readme():
        return cell_instance._process_readme(
            "tests/sample_data/neware/README_implicit.yaml"
        )

    readme = benchmark(_process_readme)
    assert readme.titles == titles_fixture
    assert readme.step_numbers == expected_steps
    assert readme.pybamm_experiment is None

    # Test with total steps
    readme = cell_instance._process_readme(
        "tests/sample_data/neware/README_total_steps.yaml"
    )
    assert readme.titles == titles_fixture
    assert readme.step_numbers == [
        [1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]
    assert readme.pybamm_experiment is None

    # Test with defined step numbers
    readme = cell_instance._process_readme("tests/sample_data/neware/README.yaml")
    assert readme.titles == titles_fixture
    assert readme.step_numbers == [
        [1, 2, 3],
        [4, 5, 6, 7],
        [9, 10, 11, 12],
    ]
    assert all(
        isinstance(item, pybamm.Experiment) for item in readme.pybamm_experiment_list
    )
    assert isinstance(readme.pybamm_experiment, pybamm.Experiment)


def test_expand_cycles():
    """Test the _expand_cycles method."""
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    cycles = [(0, 3, 2), (1, 2, 3), (7, 8, 2)]
    expected_result = [
        0,
        1,
        2,
        1,
        2,
        1,
        2,
        3,
        0,
        1,
        2,
        1,
        2,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        7,
        8,
    ]
    assert ReadmeModel._expand_cycles(indices, cycles) == expected_result


def test_readme_model():
    """Test the ReadmeModel class."""
    exp_dict = {
        "Experiment 1": {
            "Steps": {
                1: "Rest for 1 hour",
                2: "Rest for 2 hours",
                3: "Rest for 3 hours",
                4: "Rest for 4 hours",
                5: "Rest for 5 hours, Rest for 6 hours",
            },
            "Cycle 1": {
                "Start": 1,
                "End": 4,
                "Count": 2,
            },
            "Cycle 2": {
                "Start": 2,
                "End": 3,
                "Count": 3,
            },
        },
        "Experiment 2": {
            "Steps": ["Step 1", "Step 2", "Step 3", "Step 4"],
        },
        "Experiment 3": {
            "Total Steps": 8,
        },
    }
    model = ReadmeModel(readme_dict=exp_dict)

    assert model.readme_type == ["explicit", "implicit", "total"]
    assert model.step_numbers == [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9],
        [10, 11, 12, 13, 14, 15, 16, 17],
    ]
    assert model.cycle_details == [[(0, 3, 2), (1, 2, 3)], [], []]
    assert model.step_indices == [
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3],
        [0, 1, 2, 3, 4, 5, 6, 7],
    ]
    assert model.step_descriptions == [
        [
            "Rest for 1 hour",
            "Rest for 2 hours",
            "Rest for 3 hours",
            "Rest for 4 hours",
            "Rest for 5 hours, Rest for 6 hours",
        ],
        ["Step 1", "Step 2", "Step 3", "Step 4"],
        [],
    ]
    assert model.pybamm_experiment_descriptions == [
        (
            "Rest for 1 hour",
            "Rest for 2 hours",
            "Rest for 3 hours",
            "Rest for 2 hours",
            "Rest for 3 hours",
            "Rest for 2 hours",
            "Rest for 3 hours",
            "Rest for 4 hours",
            "Rest for 1 hour",
            "Rest for 2 hours",
            "Rest for 3 hours",
            "Rest for 2 hours",
            "Rest for 3 hours",
            "Rest for 2 hours",
            "Rest for 3 hours",
            "Rest for 4 hours",
            "Rest for 5 hours",
            "Rest for 6 hours",
        ),
        ("Step 1", "Step 2", "Step 3", "Step 4"),
        (),
    ]
    assert isinstance(model.pybamm_experiment_list[0], pybamm.Experiment)
    assert model.pybamm_experiment_list[1] is None
    assert model.pybamm_experiment_list[2] is None
    assert model.pybamm_experiment is None
