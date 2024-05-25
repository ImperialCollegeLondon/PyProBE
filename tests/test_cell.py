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
    root_directory = "tests/sample_data_neware"
    record_name = "sample_data_neware"
    cell_list = Cell.make_cell_list(root_directory, record_name)
    expected_info = copy.copy(info_fixture)
    expected_info["color"] = "#ff00ff"
    assert cell_list[0].info == expected_info


def test_read_record():
    """Test the read_record method."""
    root_directory = "tests/sample_data_neware"
    record_name = "sample_data_neware"
    record = Cell.read_record(root_directory, record_name)
    pl.testing.assert_frame_equal(record, pl.DataFrame({"Name": "Test_Cell"}))


def test_get_filename(info_fixture):
    """Test the get_filename method."""
    filename_inputs = ["Name"]

    def filename(name):
        return f"Cell_named_{name}.xlsx"

    file = Cell.get_filename(info_fixture, filename, filename_inputs)
    assert file == "Cell_named_Test_Cell.xlsx"


def test_process_cycler_file(cell_instance, lazyframe_fixture):
    """Test the process_cycler_file method."""
    folder_path = "tests/sample_data_neware"
    file_name = "sample_data_neware.xlsx"
    output_name = "sample_data_neware.parquet"
    cell_instance.process_cycler_file("neware", folder_path, file_name)
    expected_dataframe = lazyframe_fixture.collect()
    saved_dataframe = pl.read_parquet(f"{folder_path}/{output_name}")
    assert_frame_equal(expected_dataframe, saved_dataframe)


def test_add_procedure(cell_instance, procedure_fixture, benchmark):
    """Test the add_procedure method."""
    input_path = "tests/sample_data_neware"
    file_name = "sample_data_neware.parquet"
    title = "Test"

    def add_procedure():
        return cell_instance.add_procedure(title, input_path, file_name)

    benchmark(add_procedure)
    assert_frame_equal(cell_instance.procedure[title].data, procedure_fixture.data)


def test_set_color_scheme(cell_instance):
    """Test the set_color_scheme method."""
    assert cell_instance.set_color_scheme(5) == [
        "#ff00ff",
        "#0080ff",
        "#00db21",
        "#f03504",
        "#a09988",
    ]
