"""Tests for the Cell class."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pybatdata.cell import Cell
from pybatdata.cyclers.neware import Neware


@pytest.fixture
def cell_instance(info_fixture):
    """Return a Cell instance."""
    return Cell(info_fixture)


def test_init(cell_instance, info_fixture):
    """Test the __init__ method."""
    assert cell_instance.info == info_fixture
    assert cell_instance.procedure == {}
    assert cell_instance.processed_data == {}


def test_read_record(info_fixture):
    """Test the read_record method."""
    root_directory = "tests/sample_data_neware"
    record_name = "sample_data_neware"
    record = Cell.read_record(root_directory, record_name)
    pl.testing.assert_frame_equal(record, pl.DataFrame([info_fixture]))


@pytest.fixture
def filename_function():
    """Return a function that returns a filename."""

    def filename(name):
        return f"Cell_named_{name}.xlsx"

    return filename


def test_get_filename(info_fixture, filename_function):
    """Test the get_filename method."""
    filename_inputs = ["Name"]
    filename = Cell.get_filename(info_fixture, filename_function, filename_inputs)
    assert filename == "Cell_named_Test_Cell.xlsx"


def test_add_data(cell_instance, procedure_fixture):
    """Test the add_data method."""
    input_path = "tests/sample_data_neware/sample_data_neware.xlsx"
    title = "Test"
    cycler = Neware
    skip_writing = False
    cell_instance.add_data(input_path, title, cycler, skip_writing)
    assert_frame_equal(cell_instance.procedure[title].data, procedure_fixture.data)


def test_verify_parquet():
    """Test the verify_parquet method."""
    input_path = "tests/sample_data_neware/sample_data_neware.xlsx"
    assert Cell.verify_parquet(input_path, Neware) is True
    assert Cell.verify_parquet("false/path", Neware) is False


def test_set_color_scheme(cell_instance):
    """Test the set_color_scheme method."""
    assert cell_instance.set_color_scheme(5) == [
        "#ff00ff",
        "#0080ff",
        "#00db21",
        "#f03504",
        "#a09988",
    ]
