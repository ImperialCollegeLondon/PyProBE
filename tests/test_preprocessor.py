import pytest

from pybatdata.preprocessor import Preprocessor
from pybatdata.cyclers import Neware



def test_preprocessor_initialization(preprocessor_fixture):
    assert preprocessor_fixture.folderpath == "tests"
    assert preprocessor_fixture.procedure_name == "sample_data_neware"
    assert preprocessor_fixture.cycler == Neware
    
def test_process_readme(preprocessor_fixture):
    titles = {'Initial Charge': 'SOC Reset', 
            'Break-in Cycles': 'Cycling', 
            'Discharge Pulses': 'Pulsing', }
    
    steps = [[[1, 2, 3]], 
            [[4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7]], 
            [[9, 10, 11, 12], [9, 10, 11, 12], [9, 10, 11, 12], [9, 10, 11, 12], 
             [9, 10, 11, 12], [9, 10, 11, 12], [9, 10, 11, 12], [9, 10, 11, 12], 
             [9, 10, 11, 12], [9, 10, 11, 12]], ]
    
    cycles = [[1], [1, 2, 3, 4, 5], [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
    
    step_names = [None, 
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
    
    assert preprocessor_fixture.titles == titles
    assert preprocessor_fixture.steps == steps
    assert preprocessor_fixture.cycles == cycles
    assert preprocessor_fixture.step_names == step_names