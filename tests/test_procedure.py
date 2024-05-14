"""Module containing tests of the procedure class."""


def test_experiment(
    procedure_fixture, cycles_fixture, steps_fixture, step_names_fixture
):
    """Test creating an experiment."""
    experiment = procedure_fixture.experiment("Break-in Cycles")
    assert experiment.data["Cycle"].unique().to_list() == cycles_fixture[1]
    assert experiment.data["Step"].unique().to_list() == steps_fixture[1]

    experiment = procedure_fixture.experiment("Discharge Pulses")
    assert experiment.data["Cycle"].unique().to_list() == cycles_fixture[2]
    assert experiment.data["Step"].unique().to_list() == steps_fixture[2]

    """Test filtering by multiple experiment names."""
    experiment = procedure_fixture.experiment("Break-in Cycles", "Discharge Pulses")
    assert set(experiment.data["Cycle"].unique().to_list()) == set(
        cycles_fixture[1] + cycles_fixture[2]
    )
    assert set(experiment.data["Step"].unique().to_list()) == set(
        steps_fixture[1] + steps_fixture[2]
    )


def test_process_readme(
    procedure_fixture, titles_fixture, steps_fixture, cycles_fixture, step_names_fixture
):
    """Test processing a readme file in yaml format."""
    titles, steps = procedure_fixture.process_readme(
        "tests/sample_data_neware/README.yaml"
    )
    assert titles == titles_fixture
    assert steps == steps_fixture

    # Test without step numbers
    titles, steps = procedure_fixture.process_readme(
        "tests/sample_data_neware/README_no_step_num.yaml"
    )
    assert titles == titles_fixture
    assert steps == [
        [1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]


def test_experiment_names(procedure_fixture, titles_fixture):
    """Test the experiment_names method."""
    assert procedure_fixture.experiment_names == list(titles_fixture.keys())


def test_flatten(procedure_fixture):
    """Test flattening lists."""
    lst = [[1, 2, 3], [4, 5], 6]
    flat_list = procedure_fixture.flatten(lst)
    assert flat_list == [1, 2, 3, 4, 5, 6]
