"""Tests for the readme_processor module."""
import pybamm

from pyprobe.readme_processor import ReadmeModel, process_readme


def test_process_readme(titles_fixture, benchmark):
    """Test processing a readme file in yaml format."""
    expected_steps = [
        [1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]

    def _process_readme():
        return process_readme("tests/sample_data/neware/README_implicit.yaml")

    readme = benchmark(_process_readme)
    assert readme.titles == titles_fixture
    assert readme.step_numbers == expected_steps
    assert readme.pybamm_experiment is None

    # Test with total steps
    readme = process_readme("tests/sample_data/neware/README_total_steps.yaml")
    assert readme.titles == titles_fixture
    assert readme.step_numbers == [
        [1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]
    assert readme.pybamm_experiment is None

    # Test with defined step numbers
    readme = process_readme("tests/sample_data/neware/README.yaml")
    assert readme.titles == titles_fixture
    assert readme.step_numbers == [
        [1, 2, 3],
        [4, 5, 6, 7],
        [9, 10, 11, 12],
    ]
    assert all(
        isinstance(item, pybamm.Experiment) for item in readme.pybamm_experiment_list
    )
    assert isinstance(readme.pybamm_experiment, pybamm.Experiment)


def test_expand_cycles():
    """Test the _expand_cycles method."""
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    cycles = [(0, 3, 2), (1, 2, 3), (7, 8, 2)]
    expected_result = [
        0,
        1,
        2,
        1,
        2,
        1,
        2,
        3,
        0,
        1,
        2,
        1,
        2,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        7,
        8,
    ]
    assert ReadmeModel._expand_cycles(indices, cycles) == expected_result


def test_readme_model():
    """Test the ReadmeModel class."""
    exp_dict = {
        "Experiment 1": {
            "Steps": {
                1: "Rest for 1 hour",
                2: "Rest for 2 hours",
                3: "Rest for 3 hours",
                4: "Rest for 4 hours",
                5: "Rest for 5 hours, Rest for 6 hours",
            },
            "Cycle 1": {
                "Start": 1,
                "End": 4,
                "Count": 2,
            },
            "Cycle 2": {
                "Start": 2,
                "End": 3,
                "Count": 3,
            },
        },
        "Experiment 2": {
            "Steps": ["Step 1", "Step 2", "Step 3", "Step 4"],
        },
        "Experiment 3": {
            "Total Steps": 8,
        },
    }
    model = ReadmeModel(readme_dict=exp_dict)

    assert model.readme_type == ["explicit", "implicit", "total"]
    assert model.step_numbers == [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9],
        [10, 11, 12, 13, 14, 15, 16, 17],
    ]
    assert model.cycle_details == [[(0, 3, 2), (1, 2, 3)], [], []]
    assert model.step_indices == [
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3],
        [0, 1, 2, 3, 4, 5, 6, 7],
    ]
    assert model.step_descriptions == [
        [
            "Rest for 1 hour",
            "Rest for 2 hours",
            "Rest for 3 hours",
            "Rest for 4 hours",
            "Rest for 5 hours, Rest for 6 hours",
        ],
        ["Step 1", "Step 2", "Step 3", "Step 4"],
        [],
    ]
    assert model.pybamm_experiment_descriptions == [
        (
            "Rest for 1 hour",
            "Rest for 2 hours",
            "Rest for 3 hours",
            "Rest for 2 hours",
            "Rest for 3 hours",
            "Rest for 2 hours",
            "Rest for 3 hours",
            "Rest for 4 hours",
            "Rest for 1 hour",
            "Rest for 2 hours",
            "Rest for 3 hours",
            "Rest for 2 hours",
            "Rest for 3 hours",
            "Rest for 2 hours",
            "Rest for 3 hours",
            "Rest for 4 hours",
            "Rest for 5 hours",
            "Rest for 6 hours",
        ),
        ("Step 1", "Step 2", "Step 3", "Step 4"),
        (),
    ]
    assert isinstance(model.pybamm_experiment_list[0], pybamm.Experiment)
    assert model.pybamm_experiment_list[1] is None
    assert model.pybamm_experiment_list[2] is None
    assert model.pybamm_experiment is None
