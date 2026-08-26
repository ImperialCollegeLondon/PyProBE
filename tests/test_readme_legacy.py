"""Tests for the legacy README conversion.

A ``README.yaml`` file names an experiment, its steps and its cycles in a
format that predates the protocol tree.
:func:`~pyprobe.readme_processor.readme_to_method` converts that
dictionary to a tree, and
:meth:`~pyprobe.filters.Procedure.attach_legacy_readme` attaches the
result to a procedure's metadata on an explicit call alone.
"""

from pathlib import Path

import pytest

from pyprobe.filters import Procedure
from pyprobe.protocol import leaves, step_id_of
from pyprobe.readme_processor import readme_to_method

NEWARE_README = Path("tests/sample_data/neware/README.yaml")
"""A legacy README.yaml file naming three experiments with cycles."""


class TestExplicitAttach:
    """A legacy README converts on an explicit call alone."""

    @pytest.mark.xfail(
        strict=True,
        reason="Procedure.attach_legacy_readme is not implemented",
    )
    def test_attach_writes_the_tree_and_the_step_identifiers(
        self,
        procedure: Procedure,
    ) -> None:
        """Attaching a README writes a tree whose leaves carry the step numbers."""
        procedure.attach_legacy_readme(NEWARE_README)

        method = procedure.metadata.battinfo_test_protocol.method  # type: ignore[attr-defined]
        experiment_names = [group.description for group in method]
        assert "Break-in Cycles" in experiment_names

        break_in = next(
            group for group in method if group.description == "Break-in Cycles"
        )
        step_ids = [step_id_of(leaf) for leaf in leaves(break_in)]
        assert step_ids == [4, 5, 6, 7]

    def test_load_ignores_a_readme_beside_the_file(
        self,
        sample_data_neware_parquet: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A load beside a README.yaml file leaves the procedure without a protocol."""
        procedure = Procedure.load(sample_data_neware_parquet)

        assert procedure.experiment_names == []
        assert not [r for r in caplog.records if "README" in r.getMessage()]


class TestCycleConversion:
    """A README cycle becomes a repeat count on a contiguous group."""

    @pytest.mark.xfail(
        strict=True,
        reason="readme_to_method is not implemented",
    )
    def test_contiguous_cycle_becomes_one_repeating_group(self) -> None:
        """A cycle over the whole experiment becomes one group with that count."""
        readme = {
            "Formation": {
                "Steps": {
                    1: "Charge at 1C",
                    2: "Rest",
                    3: "Discharge at 1C",
                },
                "Cycle": {"Start": 1, "End": 3, "Count": 5},
            },
        }

        method = readme_to_method(readme)

        assert len(method) == 1
        group = method[0]
        assert group.description == "Formation"
        assert group.count == 5
        assert [step_id_of(leaf) for leaf in leaves(group)] == [1, 2, 3]

    @pytest.mark.xfail(
        strict=True,
        reason="readme_to_method is not implemented",
    )
    def test_cycle_that_cuts_across_the_step_list_raises(self) -> None:
        """A cycle whose bounds do not name a contiguous run of steps fails."""
        readme = {
            "Formation": {
                "Steps": {
                    1: "Charge at 1C",
                    2: "Rest",
                    3: "Discharge at 1C",
                },
                "Cycle 1": {"Start": 1, "End": 5, "Count": 5},
            },
        }

        with pytest.raises(ValueError, match="Formation") as failure:
            readme_to_method(readme)

        assert "Cycle 1" in str(failure.value)

    @pytest.mark.xfail(
        strict=True,
        reason="readme_to_method is not implemented",
    )
    def test_nested_cycles_nest_the_inner_group_in_the_outer(self) -> None:
        """An inner cycle declared after an outer one becomes a nested group."""
        readme = {
            "Formation": {
                "Steps": {
                    1: "Rest",
                    2: "Charge at 1C",
                    3: "Rest",
                    4: "Discharge at 1C",
                    5: "Rest",
                    6: "Rest",
                },
                "Cycle 1": {"Start": 1, "End": 6, "Count": 2},
                "Cycle 2": {"Start": 2, "End": 5, "Count": 3},
            },
        }

        method = readme_to_method(readme)

        outer = method[0]
        assert outer.count == 2
        inner_groups = [step for step in (outer.steps or []) if step.mode == "group"]
        assert len(inner_groups) == 1
        inner = inner_groups[0]
        assert inner.count == 3
        assert [step_id_of(leaf) for leaf in leaves(inner)] == [2, 3, 4, 5]
