"""Tests for the protocol tree under a filter.

A structural filter — an experiment call, a cycle call or a step call —
knows its target protocol node before it touches the data, so the result
holds that node. A condition filter, such as a charge call, builds a mask
from the data instead, and the result keeps the protocol node of its
source unchanged.
"""

from unittest.mock import patch

import polars as pl
import pytest

from pyprobe.filters import Procedure
from pyprobe.protocol import Step
from tests.protocol_helpers import attach_protocol


class TestStructuralFilters:
    """An experiment, a cycle and a step filter each reduce the tree."""

    def test_experiment_filter_reduces_to_the_named_group(
        self,
        procedure: Procedure,
    ) -> None:
        """Filtering to an experiment holds that experiment's group as its protocol."""
        group = Step(
            mode="group",
            description="Formation",
            steps=[Step(description="Charge", tags=["step_id:1"])],
        )
        attach_protocol(procedure, [group])

        experiment = procedure.experiment("Formation")

        assert experiment._protocol_node is group

    def test_step_filter_reduces_to_the_leaf(self, procedure: Procedure) -> None:
        """Filtering to a step holds that step's leaf as its protocol."""
        leaf = Step(description="Charge", tags=["step_id:1"])
        attach_protocol(procedure, [leaf])

        step = procedure.step(0)

        assert step._protocol_node is leaf


class TestConditionFilters:
    """A condition filter keeps the protocol of its source unchanged."""

    def test_charge_filter_keeps_the_source_protocol_without_a_collect(
        self,
        procedure: Procedure,
    ) -> None:
        """A charge filter reports the protocol of its source, and collects nothing."""
        marker = Step(mode="group", description="Formation")
        procedure._protocol_node = marker

        with patch(
            "polars.LazyFrame.collect",
            side_effect=AssertionError("a condition filter must not collect"),
        ):
            result = procedure.charge()

        assert result._protocol_node is marker


class TestCycleDescent:
    """A cycle filter descends to the first group below that repeats."""

    def test_outer_cycle_filter_reduces_to_the_repeating_group(
        self,
        procedure: Procedure,
    ) -> None:
        """An outer cycle filter reduces the protocol to the group that repeats."""
        outer = Step(
            mode="group",
            description="Formation",
            count=10,
            steps=[
                Step(description=f"Step {i}", tags=[f"step_id:{i}"])
                for i in range(10, 16)
            ],
        )
        attach_protocol(procedure, [outer])

        experiment = procedure.experiment("Formation")
        cycle = experiment.cycle(0)

        assert cycle._protocol_node is outer

    def test_inner_cycle_filter_reduces_to_the_nested_group(
        self,
        procedure: Procedure,
    ) -> None:
        """A further cycle filter descends into the nested repeating group."""
        inner = Step(
            mode="group",
            count=50,
            steps=[
                Step(description="Charge", tags=["step_id:11"]),
                Step(description="Rest", tags=["step_id:12"]),
            ],
        )
        outer = Step(
            mode="group",
            description="Formation",
            count=10,
            steps=[inner],
        )
        attach_protocol(procedure, [outer])

        experiment = procedure.experiment("Formation")
        outer_cycle = experiment.cycle(0)
        inner_cycle = outer_cycle.cycle(0)

        assert inner_cycle._protocol_node is inner

    def test_no_repeating_group_infers_the_boundary_and_warns(
        self,
        procedure: Procedure,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """With no repeating group below, the boundary is inferred and warned once."""
        attach_protocol(
            procedure,
            [
                Step(
                    mode="group",
                    description="Formation",
                    steps=[
                        Step(description="Charge", tags=["step_id:1"]),
                        Step(description="Rest", tags=["step_id:2"]),
                        Step(description="Discharge", tags=["step_id:3"]),
                    ],
                ),
            ],
        )

        cycle = procedure.experiment("Formation").cycle(0)

        assert cycle.data.height > 0
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1

    def test_two_sibling_groups_with_a_count_raises(
        self,
        procedure: Procedure,
    ) -> None:
        """Two sibling groups that each repeat fail, and the message names both."""
        attach_protocol(
            procedure,
            [
                Step(
                    mode="group",
                    description="Formation A",
                    count=3,
                    steps=[Step(description="Charge", tags=["step_id:1"])],
                ),
                Step(
                    mode="group",
                    description="Formation B",
                    count=4,
                    steps=[Step(description="Discharge", tags=["step_id:2"])],
                ),
            ],
        )

        with pytest.raises(ValueError, match="Formation A") as failure:
            procedure.cycle()

        assert "Formation B" in str(failure.value)


class TestExperimentNavigation:
    """A user reads the experiments at a tree level, and filters to one of them."""

    def test_procedure_reports_its_top_experiments(
        self,
        procedure: Procedure,
    ) -> None:
        """The procedure's experiment names are the descriptions at the top level."""
        attach_protocol(
            procedure,
            [
                Step(
                    mode="group",
                    description="Initial Charge",
                    steps=[Step(description="Rest", tags=["step_id:1"])],
                ),
                Step(
                    mode="group",
                    description="Break-in",
                    steps=[Step(description="Charge", tags=["step_id:2"])],
                ),
            ],
        )

        assert procedure.experiment_names == ["Initial Charge", "Break-in"]

    def test_experiment_reports_its_child_experiments(
        self,
        procedure: Procedure,
    ) -> None:
        """An experiment's names are the descriptions of the groups below it."""
        attach_protocol(
            procedure,
            [
                Step(
                    mode="group",
                    description="Formation",
                    steps=[
                        Step(
                            mode="group",
                            description="Charge Phase",
                            steps=[Step(description="Charge", tags=["step_id:1"])],
                        ),
                        Step(
                            mode="group",
                            description="Discharge Phase",
                            steps=[Step(description="Discharge", tags=["step_id:2"])],
                        ),
                    ],
                ),
            ],
        )

        experiment = procedure.experiment("Formation")

        assert experiment.experiment_names == [  # type: ignore[attr-defined]
            "Charge Phase",
            "Discharge Phase",
        ]

    def test_filtering_to_a_child_experiment_returns_its_data_alone(self) -> None:
        """Filtering to a child experiment selects the rows of that child alone."""
        frame = pl.DataFrame(
            {
                "Test Time / s": [0.0, 1.0, 2.0, 3.0],
                "Current / A": [1.0, 1.0, -1.0, -1.0],
                "Voltage / V": [3.7, 3.8, 3.6, 3.5],
                "Step ID": [1, 2, 3, 4],
            },
        )
        procedure = Procedure.load(frame)
        attach_protocol(
            procedure,
            [
                Step(
                    mode="group",
                    description="Formation",
                    steps=[
                        Step(
                            mode="group",
                            description="Charge Phase",
                            steps=[
                                Step(description="Charge", tags=["step_id:1"]),
                                Step(description="Rest", tags=["step_id:2"]),
                            ],
                        ),
                        Step(
                            mode="group",
                            description="Discharge Phase",
                            steps=[
                                Step(description="Discharge", tags=["step_id:3"]),
                                Step(description="Rest", tags=["step_id:4"]),
                            ],
                        ),
                    ],
                ),
            ],
        )

        child = procedure.experiment("Formation").experiment(  # type: ignore[attr-defined]
            "Charge Phase",
        )

        assert set(child.data["Step ID"].to_list()) == {1, 2}

    def test_name_not_present_at_the_current_level_raises(
        self,
        procedure_fixture: Procedure,
    ) -> None:
        """A name the current tree level does not hold fails, naming the request."""
        experiment = procedure_fixture.experiment("Break-in Cycles")

        with pytest.raises(ValueError, match="Ghost Phase"):
            experiment.experiment("Ghost Phase")  # type: ignore[attr-defined]
