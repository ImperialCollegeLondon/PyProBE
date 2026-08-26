"""Tests for the legacy experiment definitions that the test helper builds.

Each expected mapping is written as a literal, so a fixture that routes
through :func:`tests.readme_helpers.attach_readme` never derives its
expectation from the converter it exercises.
"""

import yaml

from pyprobe.filters import Procedure
from tests.readme_helpers import attach_readme

PARQUET = "tests/sample_data/neware/sample_data_neware.bdf.parquet"
"""A sample procedure to attach a README to."""

INITIAL_CHARGE_DESCRIPTIONS = [
    "Rest for 4 hours",
    "Charge at 4mA until 4.2 V, Hold at 4.2 V until 0.04 A",
    "Rest for 2 hours",
]
BREAK_IN_DESCRIPTIONS = [
    "Discharge at 4 mA until 3 V",
    "Rest for 2 hours",
    "Charge at 4 mA until 4.2 V, Hold at 4.2 V until 0.04 A",
    "Rest for 2 hours",
]
PULSE_DESCRIPTIONS = [
    "Rest for 10 seconds",
    "Discharge at 20 mA for 0.2 hours or until 3 V",
    "Rest for 30 minutes",
    "Rest for 1.5 hours",
]


def test_explicit_readme_definitions():
    """A README with explicit step numbers and cycles builds those definitions."""
    procedure = attach_readme(
        Procedure.load(PARQUET),
        "tests/sample_data/neware/README.yaml",
    )

    assert procedure.readme_dict == {
        "Initial Charge": {
            "Steps": [1, 2, 3],
            "Step Descriptions": INITIAL_CHARGE_DESCRIPTIONS,
            "Cycles": [],
        },
        "Break-in Cycles": {
            "Steps": [4, 5, 6, 7],
            "Step Descriptions": BREAK_IN_DESCRIPTIONS,
            "Cycles": [(4, 7, 5)],
        },
        "Discharge Pulses": {
            "Steps": [9, 10, 11, 12],
            "Step Descriptions": PULSE_DESCRIPTIONS,
            "Cycles": [(9, 12, 10)],
        },
    }


def test_implicit_readme_definitions():
    """A README with implicit step numbers numbers the steps in order."""
    procedure = attach_readme(
        Procedure.load(PARQUET),
        "tests/sample_data/neware/README_implicit.yaml",
    )

    assert procedure.readme_dict == {
        "Initial Charge": {
            "Steps": [1, 2, 3],
            "Step Descriptions": INITIAL_CHARGE_DESCRIPTIONS,
            "Cycles": [],
        },
        "Break-in Cycles": {
            "Steps": [4, 5, 6, 7],
            "Step Descriptions": BREAK_IN_DESCRIPTIONS,
            "Cycles": [],
        },
        "Discharge Pulses": {
            "Steps": [8, 9, 10, 11],
            "Step Descriptions": PULSE_DESCRIPTIONS,
            "Cycles": [],
        },
    }


def test_total_steps_readme_definitions():
    """A README that states a total step count names no step description."""
    procedure = attach_readme(
        Procedure.load(PARQUET),
        "tests/sample_data/neware/README_total_steps.yaml",
    )

    assert procedure.readme_dict == {
        "Initial Charge": {
            "Steps": [1, 2, 3],
            "Step Descriptions": [],
            "Cycles": [],
        },
        "Break-in Cycles": {
            "Steps": [4, 5, 6, 7],
            "Step Descriptions": [],
            "Cycles": [],
        },
        "Discharge Pulses": {
            "Steps": [8, 9, 10, 11],
            "Step Descriptions": [],
            "Cycles": [],
        },
    }


def test_an_empty_description_stays_in_the_definitions(tmp_path):
    """A step that names an empty description keeps its place in the list."""
    readme_path = tmp_path / "README.yaml"
    readme_path.write_text(yaml.safe_dump({"Formation": {"Steps": {1: "", 2: ""}}}))

    procedure = attach_readme(Procedure.load(PARQUET), readme_path)

    assert procedure.readme_dict["Formation"]["Step Descriptions"] == ["", ""]
    assert procedure.step_descriptions == {
        "Step": [1, 2],
        "Description": ["", ""],
    }
