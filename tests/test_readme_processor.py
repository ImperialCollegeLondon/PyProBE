"""Tests for the readme_processor module."""

import pytest

from pyprobe.readme_processor import ReadmeModel, process_readme


@pytest.fixture
def readme_dict_fixture():
    """Return a readme dictionary for testing."""
    return {
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


def test_readme(readme_dict_fixture):
    """Test the process_readme function."""
    readme = ReadmeModel(readme_dict=readme_dict_fixture)
    assert list(readme.experiment_dict.keys()) == [
        "Experiment 1",
        "Experiment 2",
        "Experiment 3",
    ]
    assert readme.experiment_dict["Experiment 1"]["Steps"] == [1, 2, 3, 4, 5]
    assert readme.experiment_dict["Experiment 1"]["Step Descriptions"] == [
        "Rest for 1 hour",
        "Rest for 2 hours",
        "Rest for 3 hours",
        "Rest for 4 hours",
        "Rest for 5 hours, Rest for 6 hours",
    ]
    assert readme.experiment_dict["Experiment 1"]["Cycles"] == [
        (1, 4, 2),
        (2, 3, 3),
    ]

    assert readme.experiment_dict["Experiment 2"]["Steps"] == [6, 7, 8, 9]
    assert readme.experiment_dict["Experiment 2"]["Step Descriptions"] == [
        "Step 1",
        "Step 2",
        "Step 3",
        "Step 4",
    ]
    assert readme.experiment_dict["Experiment 2"]["Cycles"] == []


def test_process_readme_file_explicit(titles_fixture, benchmark):
    """Test processing a readme file in yaml format."""

    def _process_readme():
        return process_readme("tests/sample_data/neware/README.yaml")

    readme = benchmark(_process_readme)

    assert list(readme.experiment_dict.keys()) == titles_fixture
    assert readme.experiment_dict["Break-in Cycles"]["Steps"] == [4, 5, 6, 7]
    assert readme.experiment_dict["Break-in Cycles"]["Step Descriptions"] == [
        "Discharge at 4 mA until 3 V",
        "Rest for 2 hours",
        "Charge at 4 mA until 4.2 V, Hold at 4.2 V until 0.04 A",
        "Rest for 2 hours",
    ]

    assert readme.experiment_dict["Discharge Pulses"]["Steps"] == [9, 10, 11, 12]
    assert readme.experiment_dict["Discharge Pulses"]["Step Descriptions"] == [
        "Rest for 10 seconds",
        "Discharge at 20 mA for 0.2 hours or until 3 V",
        "Rest for 30 minutes",
        "Rest for 1.5 hours",
    ]
    assert readme.experiment_dict["Discharge Pulses"]["Cycles"] == [(9, 12, 10)]


def test_process_readme_file_implicit(titles_fixture, benchmark):
    """Test processing a readme file in yaml format."""

    def _process_readme():
        return process_readme("tests/sample_data/neware/README_implicit.yaml")

    readme = benchmark(_process_readme)

    assert list(readme.experiment_dict.keys()) == titles_fixture
    assert readme.experiment_dict["Break-in Cycles"]["Steps"] == [4, 5, 6, 7]
    assert readme.experiment_dict["Break-in Cycles"]["Step Descriptions"] == [
        "Discharge at 4 mA until 3 V",
        "Rest for 2 hours",
        "Charge at 4 mA until 4.2 V, Hold at 4.2 V until 0.04 A",
        "Rest for 2 hours",
    ]
    assert readme.experiment_dict["Break-in Cycles"]["Cycles"] == []

    assert readme.experiment_dict["Discharge Pulses"]["Steps"] == [8, 9, 10, 11]
    assert readme.experiment_dict["Discharge Pulses"]["Step Descriptions"] == [
        "Rest for 10 seconds",
        "Discharge at 20 mA for 0.2 hours or until 3 V",
        "Rest for 30 minutes",
        "Rest for 1.5 hours",
    ]
    assert readme.experiment_dict["Discharge Pulses"]["Cycles"] == []
