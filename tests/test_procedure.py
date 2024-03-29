import pytest
from pybatdata.procedure import Procedure


def test_experiment(procedure_fixture):
    experiment = procedure_fixture.experiment('Break-in Cycles')
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
