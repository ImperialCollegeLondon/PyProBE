def test_experiment(procedure_fixture, cycles_fixture, steps_fixture, step_names_fixture):
    experiment = procedure_fixture.experiment('Break-in Cycles')
    assert experiment.RawData['Cycle'].unique().to_list() == cycles_fixture[1]
    assert experiment.RawData['Step'].unique().to_list() ==  steps_fixture[1][0]

def test_process_readme(procedure_fixture,
                        titles_fixture,
                        steps_fixture,
                        cycles_fixture,
                        step_names_fixture):
    
    assert procedure_fixture.titles == titles_fixture
    assert procedure_fixture.steps_idx == steps_fixture
    assert procedure_fixture.cycles_idx == cycles_fixture
    assert procedure_fixture.step_names == step_names_fixture

def test_flatten(procedure_fixture):
    lst = [[1, 2, 3], [4, 5], 6]
    flat_list = procedure_fixture.flatten(lst)
    assert flat_list == [1, 2, 3, 4, 5, 6]