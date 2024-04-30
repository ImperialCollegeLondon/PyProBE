"""Module containing tests of the procedure class."""


def test_experiment(
    procedure_fixture, cycles_fixture, steps_fixture, step_names_fixture
):
    """Test creating an experiment."""
    experiment = procedure_fixture.experiment("Break-in Cycles")
    assert experiment.data["Cycle"].unique().to_list() == cycles_fixture[1]
    assert experiment.data["Step"].unique().to_list() == steps_fixture[1][0]

    experiment = procedure_fixture.experiment("Discharge Pulses")
    assert experiment.data["Cycle"].unique().to_list() == cycles_fixture[2]
    assert experiment.data["Step"].unique().to_list() == steps_fixture[2][0]


def test_process_readme(
    procedure_fixture, titles_fixture, steps_fixture, cycles_fixture, step_names_fixture
):
    """Test processing a readme file in yaml format."""
    titles, cycles, steps = procedure_fixture.process_readme(
        "tests/sample_data_neware/README.yaml"
    )
    assert titles == titles_fixture
    assert steps == steps_fixture
    assert cycles == cycles_fixture

    # Test without step numbers
    titles, cycles, steps = procedure_fixture.process_readme(
        "tests/sample_data_neware/README_no_step_num.yaml"
    )
    assert titles == titles_fixture
    assert steps == [
        [[1, 2, 3]],
        [[4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7]],
        [
            [8, 9, 10, 11],
            [8, 9, 10, 11],
            [8, 9, 10, 11],
            [8, 9, 10, 11],
            [8, 9, 10, 11],
            [8, 9, 10, 11],
            [8, 9, 10, 11],
            [8, 9, 10, 11],
            [8, 9, 10, 11],
            [8, 9, 10, 11],
        ],
    ]


def test_flatten(procedure_fixture):
    """Test flattening lists."""
    lst = [[1, 2, 3], [4, 5], 6]
    flat_list = procedure_fixture.flatten(lst)
    assert flat_list == [1, 2, 3, 4, 5, 6]
