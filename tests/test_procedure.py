"""Module containing tests of the procedure class."""


def test_experiment(procedure_fixture, cycles_fixture, steps_fixture, benchmark):
    """Test creating an experiment."""

    def make_experiment():
        return procedure_fixture.experiment("Break-in Cycles")

    experiment = benchmark(make_experiment)
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

    assert experiment.data["Experiment Time [s]"][0] == 0
    assert experiment.data["Experiment Capacity [Ah]"][0] == 0


def test_experiment_names(procedure_fixture, titles_fixture):
    """Test the experiment_names method."""
    assert procedure_fixture.experiment_names == titles_fixture


def test_flatten(procedure_fixture):
    """Test flattening lists."""
    lst = [[1, 2, 3], [4, 5], 6]
    flat_list = procedure_fixture._flatten(lst)
    assert flat_list == [1, 2, 3, 4, 5, 6]


def test_zero_columns(procedure_fixture):
    """Test methods to set the first value of columns to zero."""
    assert procedure_fixture.data["Procedure Time [s]"][0] == 0
    assert procedure_fixture.data["Procedure Capacity [Ah]"][0] == 0
