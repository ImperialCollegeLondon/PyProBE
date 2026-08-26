"""Tests for the readme_processor module."""

import pytest
import yaml

from pyprobe.protocol import Step, leaves, step_id_of
from pyprobe.readme_processor import readme_to_method


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


def _method_from_file(readme_path):
    """Return the protocol tree of a README.yaml file."""
    with open(readme_path) as file:
        return readme_to_method(yaml.safe_load(file))


def _step_ids(node: Step):
    """Return the step identifier of each leaf under a node, in tree order."""
    return [step_id_of(leaf) for leaf in leaves(node)]


def _descriptions(node: Step):
    """Return the description of each leaf under a node, in tree order."""
    return [leaf.description for leaf in leaves(node)]


def test_readme_to_method(readme_dict_fixture):
    """Test the conversion of a readme dictionary to a protocol tree."""
    method = readme_to_method(readme_dict_fixture)

    assert [group.description for group in method] == [
        "Experiment 1",
        "Experiment 2",
        "Experiment 3",
    ]

    assert _step_ids(method[0]) == [1, 2, 3, 4, 5]
    assert _descriptions(method[0]) == [
        "Rest for 1 hour",
        "Rest for 2 hours",
        "Rest for 3 hours",
        "Rest for 4 hours",
        "Rest for 5 hours, Rest for 6 hours",
    ]
    outer = method[0].steps[0]
    assert outer.count == 2
    assert _step_ids(outer) == [1, 2, 3, 4]
    inner = outer.steps[1]
    assert inner.count == 3
    assert _step_ids(inner) == [2, 3]

    assert _step_ids(method[1]) == [6, 7, 8, 9]
    assert _descriptions(method[1]) == ["Step 1", "Step 2", "Step 3", "Step 4"]
    assert method[1].count is None

    assert _step_ids(method[2]) == [10, 11, 12, 13, 14, 15, 16, 17]
    assert _descriptions(method[2]) == [None] * 8


def test_readme_to_method_file_explicit(titles_fixture):
    """Test the conversion of a readme file with explicit step numbers."""
    method = _method_from_file("tests/sample_data/neware/README.yaml")

    assert [group.description for group in method] == titles_fixture

    break_in = method[1]
    assert _step_ids(break_in) == [4, 5, 6, 7]
    assert _descriptions(break_in) == [
        "Discharge at 4 mA until 3 V",
        "Rest for 2 hours",
        "Charge at 4 mA until 4.2 V, Hold at 4.2 V until 0.04 A",
        "Rest for 2 hours",
    ]
    assert break_in.count == 5

    pulses = method[2]
    assert _step_ids(pulses) == [9, 10, 11, 12]
    assert _descriptions(pulses) == [
        "Rest for 10 seconds",
        "Discharge at 20 mA for 0.2 hours or until 3 V",
        "Rest for 30 minutes",
        "Rest for 1.5 hours",
    ]
    assert pulses.count == 10


def test_readme_to_method_file_implicit(titles_fixture):
    """Test the conversion of a readme file with implicit step numbers."""
    method = _method_from_file("tests/sample_data/neware/README_implicit.yaml")

    assert [group.description for group in method] == titles_fixture

    break_in = method[1]
    assert _step_ids(break_in) == [4, 5, 6, 7]
    assert _descriptions(break_in) == [
        "Discharge at 4 mA until 3 V",
        "Rest for 2 hours",
        "Charge at 4 mA until 4.2 V, Hold at 4.2 V until 0.04 A",
        "Rest for 2 hours",
    ]
    assert break_in.count is None

    pulses = method[2]
    assert _step_ids(pulses) == [8, 9, 10, 11]
    assert pulses.count is None


def test_readme_to_method_file_total_steps(titles_fixture):
    """Test the conversion of a readme file that states a total step count."""
    method = _method_from_file("tests/sample_data/neware/README_total_steps.yaml")

    assert [group.description for group in method] == titles_fixture

    break_in = method[1]
    assert _step_ids(break_in) == [4, 5, 6, 7]
    assert _descriptions(break_in) == [None] * 4

    pulses = method[2]
    assert _step_ids(pulses) == [8, 9, 10, 11]
    assert _descriptions(pulses) == [None] * 4
