import pytest
import polars as pl
from pybatdata.cell import Cell
from pybatdata.procedure import Procedure
from pybatdata.cyclers.neware import Neware
neware = Neware()
Cell.write_parquet('tests/sample_data_neware/sample_data_neware.xlsx', 'tests/sample_data_neware/sample_data_neware.parquet', neware)

@pytest.fixture(scope='module')
def info_fixture():
    return {"Name": 'Test_Cell'}

@pytest.fixture(scope='module')
def lazyframe_fixture():
    return pl.scan_parquet('tests/sample_data_neware/sample_data_neware.parquet')

@pytest.fixture(scope='module')
def titles_fixture():
    return {'Initial Charge': 'SOC Reset', 
            'Break-in Cycles': 'Cycling', 
            'Discharge Pulses': 'Pulsing', }
    
@pytest.fixture(scope='module')
def steps_fixture():
    return [[[1, 2, 3]], 
            [[4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7]], 
            [[9, 10, 11, 12], [9, 10, 11, 12], [9, 10, 11, 12], [9, 10, 11, 12], 
             [9, 10, 11, 12], [9, 10, 11, 12], [9, 10, 11, 12], [9, 10, 11, 12], 
             [9, 10, 11, 12], [9, 10, 11, 12]], ]

@pytest.fixture(scope='module')
def cycles_fixture():
    return [[1], [1, 2, 3, 4, 5], [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]

@pytest.fixture(scope='module')
def step_names_fixture():
    return [None, 
            'Rest', 
            'CCCV Chg', 
            'Rest', 
            'CC DChg', 
            'Rest', 
            'CCCV Chg', 
            'Rest', 
            None, 
            'Rest', 
            'CC DChg', 
            'Rest', 
            'Rest']

@pytest.fixture(scope='module')
def procedure_fixture(info_fixture):
    return Procedure('tests/sample_data_neware/sample_data_neware.parquet', info_fixture)

@pytest.fixture(scope='module')
def BreakinCycles_fixture(procedure_fixture):
    return procedure_fixture.experiment('Break-in Cycles')

@pytest.fixture(scope='module')
def Pulsing_fixture(procedure_fixture):
    return procedure_fixture.experiment('Discharge Pulses')