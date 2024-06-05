"""Tests for the Cell class."""
import copy

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pyprobe.cell import Cell


@pytest.fixture
def cell_instance(info_fixture):
    """Return a Cell instance."""
    return Cell(info_fixture)


def test_init(cell_instance, info_fixture):
    """Test the __init__ method."""
    assert cell_instance.info == info_fixture
    assert cell_instance.procedure == {}
    assert cell_instance.processed_data == {}


def test_make_cell_list(info_fixture):
    """Test the make_cell_list method."""
    filepath = "tests/sample_data/neware/Experiment_Record.xlsx"
    record_name = "sample_data_neware"
    cell_list = Cell.make_cell_list(filepath, record_name)
    expected_info = copy.copy(info_fixture)
    expected_info["color"] = "#ff00ff"
    assert cell_list[0].info == expected_info


def test_get_filename(info_fixture):
    """Test the get_filename method."""
    filename_inputs = ["Name"]

    def filename(name):
        return f"Cell_named_{name}.xlsx"

    file = Cell.get_filename(info_fixture, filename, filename_inputs)
    assert file == "Cell_named_Test_Cell.xlsx"


def test_verify_filename():
    """Test the verify_parquet method."""
    file = "path/to/sample_data_neware"
    assert Cell.verify_parquet(file) == "path/to/sample_data_neware.parquet"

    file = "path/to/sample_data_neware.parquet"
    assert Cell.verify_parquet(file) == "path/to/sample_data_neware.parquet"

    file = "path/to/sample_data_neware.csv"
    assert Cell.verify_parquet(file) == "path/to/sample_data_neware.parquet"


def test_process_cycler_file(cell_instance, lazyframe_fixture):
    """Test the process_cycler_file method."""
    folder_path = "tests/sample_data/neware/"
    file_name = "sample_data_neware.xlsx"
    output_name = "sample_data_neware.parquet"
    cell_instance.process_cycler_file("neware", folder_path, file_name, output_name)
    expected_dataframe = lazyframe_fixture.collect()
    saved_dataframe = pl.read_parquet(f"{folder_path}/{output_name}")
    assert_frame_equal(expected_dataframe, saved_dataframe)


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
        "Test_custom", input_path, file_name, custom_readme_name="README_total_steps"
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
