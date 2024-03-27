import pytest
from pybatdata.procedure import Procedure

@pytest.fixture
def procedure_instance(lazyframe, preprocessor_fixture):
    return Procedure(lazyframe, preprocessor_fixture.titles, preprocessor_fixture.cycles, preprocessor_fixture.steps, preprocessor_fixture.step_names)

def test_experiment(procedure_instance):
    experiment = procedure_instance.experiment('Break-in Cycles')
    assert experiment.cycles_idx == [1, 2, 3, 4, 5]
    assert experiment.steps_idx == [[4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7]]
    assert experiment.step_names ==  [None, 
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
