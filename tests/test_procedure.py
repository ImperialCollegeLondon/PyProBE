def test_experiment(procedure_fixture, cycles_fixture, steps_fixture, step_names_fixture):
    experiment = procedure_fixture.experiment('Break-in Cycles')
    assert experiment.cycles_idx == cycles_fixture[1]
    assert experiment.steps_idx ==  steps_fixture[1]
    assert experiment.step_names ==  step_names_fixture

def test_process_readme(procedure_fixture,
                        titles_fixture,
                        steps_fixture,
                        cycles_fixture,
                        step_names_fixture):
    
    assert procedure_fixture.titles == titles_fixture
    assert procedure_fixture.steps_idx == steps_fixture
    assert procedure_fixture.cycles_idx == cycles_fixture
    assert procedure_fixture.step_names == step_names_fixture