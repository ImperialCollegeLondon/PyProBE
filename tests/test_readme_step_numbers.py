"""Tests for the step numbering of the legacy README converter.

An experiment that states its steps as a list of descriptions, or as a total
count, numbers them from the highest step number that any experiment before
it holds.
"""

from pyprobe.protocol import leaves, step_id_of
from pyprobe.readme_processor import readme_to_method


def _step_ids(group):
    """Return the step identifier of each leaf under a group, in tree order."""
    return [step_id_of(leaf) for leaf in leaves(group)]


def test_implicit_steps_follow_the_highest_number_so_far():
    """A step list numbers from the highest number of any earlier experiment."""
    readme = {
        "Conditioning": {"Steps": {4: "Rest for 4 hours", 5: "Charge at 1C"}},
        "Reference Test": {"Steps": {1: "Rest for 1 hour", 2: "Discharge at 1C"}},
        "Ageing": {"Steps": ["Charge at 1C", "Rest for 1 hour", "Discharge at 1C"]},
    }

    method = readme_to_method(readme)

    assert _step_ids(method[0]) == [4, 5]
    assert _step_ids(method[1]) == [1, 2]
    assert _step_ids(method[2]) == [6, 7, 8]


def test_a_total_step_count_follows_the_highest_number_so_far():
    """A total step count numbers from the highest number of any earlier one."""
    readme = {
        "Conditioning": {"Steps": {4: "Rest for 4 hours", 5: "Charge at 1C"}},
        "Reference Test": {"Steps": {1: "Rest for 1 hour", 2: "Discharge at 1C"}},
        "Ageing": {"Total Steps": 3},
    }

    method = readme_to_method(readme)

    assert _step_ids(method[2]) == [6, 7, 8]
