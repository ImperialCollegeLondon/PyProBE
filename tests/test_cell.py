import pytest
from pybatdata.cell import Cell
from pybatdata.cyclers.neware import Neware
import pandas as pd
from polars.testing import assert_frame_equal

@pytest.fixture
def cell_instance(metadata_fixture):
    return Cell(metadata_fixture)

def test_init(cell_instance, metadata_fixture):
    assert cell_instance.metadata == metadata_fixture
    assert cell_instance.raw_data == {}
    assert cell_instance.processed_data == {}

def test_read_record(metadata_fixture):
    root_directory = 'tests/sample_data_neware'
    record_name = 'sample_data_neware'
    record = Cell.read_record(root_directory, record_name)
    pd.testing.assert_frame_equal(record, pd.DataFrame([metadata_fixture]))

@pytest.fixture
def filename_function():
    def filename(name):
        return f'Cell_named_{name}.xlsx'
    return filename

def test_get_filename(metadata_fixture, filename_function):
    filename_inputs=['Name']
    filename = Cell.get_filename(metadata_fixture, filename_function, filename_inputs)
    assert filename == 'Cell_named_Test_Cell.xlsx'

def test_add_data(cell_instance, procedure_fixture):
    input_path = 'tests/sample_data_neware/sample_data_neware.xlsx'
    title = 'Test'
    cycler = Neware
    skip_writing = True
    cell_instance.add_data(input_path, title, cycler, skip_writing)
    assert_frame_equal(cell_instance.raw_data[title].RawData, procedure_fixture.RawData)

def test_verify_parquet():
    input_path = 'tests/sample_data_neware/sample_data_neware.xlsx'
    assert Cell.verify_parquet(input_path, Neware) == True