"""Tests for the protocol tree and its derived views.

PyProBE stores the experiment definitions of a procedure in
``metadata.battinfo_test_protocol.method``, an ordered list of protocol
:class:`~pyprobe.protocol.Step` nodes. A group node names an experiment
through its description and repeats as a cycle through its ``count``. A
leaf node carries its cycler step identifier in a ``"step_id:"`` tag, and
the step descriptions, the cycle information and the column definitions
all derive from that tree.
"""

import bdf
import bdf.spec
import polars as pl
import pytest

from pyprobe.filters import Procedure
from pyprobe.protocol import Step, step_id_of
from pyprobe.result import Table
from tests.protocol_helpers import attach_protocol


class TestGroupConvention:
    """A group node names an experiment, repeats as a cycle, or both."""

    @pytest.mark.xfail(
        strict=True,
        reason="Procedure.experiment_names reads the legacy readme_dict "
        "rather than the protocol tree",
    )
    def test_group_with_a_description_names_an_experiment(
        self,
        procedure: Procedure,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A group's description becomes an experiment name.

        An object with no protocol tree reports no experiment names, and it
        logs no warning about it.
        """
        assert procedure.experiment_names == []
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

        attach_protocol(
            procedure,
            [
                Step(
                    mode="group",
                    description="Formation",
                    steps=[Step(description="Rest", tags=["step_id:1"])],
                ),
            ],
        )

        assert procedure.experiment_names == ["Formation"]

    @pytest.mark.xfail(
        strict=True,
        reason="Procedure.cycle_info reads a stored list rather than the protocol tree",
    )
    def test_group_with_a_count_is_a_cycle(self, procedure: Procedure) -> None:
        """A group that repeats a count is treated as a cycle."""
        attach_protocol(
            procedure,
            [
                Step(
                    mode="group",
                    count=5,
                    steps=[
                        Step(description="Discharge", tags=["step_id:4"]),
                        Step(description="Rest", tags=["step_id:5"]),
                        Step(description="Charge", tags=["step_id:6"]),
                        Step(description="Rest", tags=["step_id:7"]),
                    ],
                ),
            ],
        )

        assert procedure.cycle_info == [(4, 7, 5)]

    @pytest.mark.xfail(
        strict=True,
        reason="the protocol tree does not yet drive experiment_names and cycle_info",
    )
    def test_group_with_a_description_and_a_count_is_both(
        self,
        procedure: Procedure,
    ) -> None:
        """A group that both names and repeats is an experiment and a cycle."""
        attach_protocol(
            procedure,
            [
                Step(
                    mode="group",
                    description="Formation",
                    count=3,
                    steps=[
                        Step(description="Charge", tags=["step_id:1"]),
                        Step(description="Rest", tags=["step_id:2"]),
                        Step(description="Discharge", tags=["step_id:3"]),
                    ],
                ),
            ],
        )

        assert procedure.experiment_names == ["Formation"]
        assert procedure.cycle_info == [(1, 3, 3)]


class TestExperimentDataSelection:
    """Filtering to an experiment selects the data rows of its leaves alone."""

    @pytest.mark.xfail(
        strict=True,
        reason="Procedure.experiment resolves names against the legacy "
        "readme_dict, not the protocol tree",
    )
    def test_experiment_selects_only_the_rows_of_its_leaves(self) -> None:
        """A skipped step number is excluded even where the source data holds it."""
        frame = pl.DataFrame(
            {
                "Test Time / s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "Current / A": [1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.5],
                "Voltage / V": [3.7, 3.8, 3.6, 3.5, 3.55, 3.56, 3.6],
                "Step ID": [4, 5, 6, 7, 8, 9, 10],
            },
        )
        procedure = Procedure.load(frame)
        attach_protocol(
            procedure,
            [
                Step(
                    mode="group",
                    description="Break-in",
                    steps=[
                        Step(description="Discharge", tags=["step_id:4"]),
                        Step(description="Rest", tags=["step_id:5"]),
                        Step(description="Charge", tags=["step_id:6"]),
                        Step(description="Rest", tags=["step_id:7"]),
                        Step(description="Rest", tags=["step_id:9"]),
                        Step(description="Discharge", tags=["step_id:10"]),
                    ],
                ),
            ],
        )

        data = procedure.experiment("Break-in").data

        assert set(data["Step ID"].to_list()) == {4, 5, 6, 7, 9, 10}


class TestStepIdentifierValidation:
    """A leaf that carries no identifier, or an invalid one, fails and names itself."""

    @pytest.mark.xfail(
        strict=True,
        reason="Procedure.experiment passes the bdf.Metadata record into "
        "Table's dict-only metadata check, before it validates step "
        "identifiers",
    )
    def test_leaf_without_a_step_identifier_raises(
        self,
        procedure_fixture: Procedure,
    ) -> None:
        """An experiment whose leaves carry no step identifier fails, naming it."""
        attach_protocol(
            procedure_fixture,
            [
                Step(
                    mode="group",
                    description="Break-in Cycles",
                    steps=[
                        Step(description="Discharge"),
                        Step(description="Rest"),
                    ],
                ),
            ],
        )

        with pytest.raises(ValueError, match="Break-in Cycles"):
            procedure_fixture.experiment("Break-in Cycles")

    def test_invalid_step_identifier_raises(self) -> None:
        """A tag that does not hold an integer identifier fails, naming the leaf."""
        leaf = Step(description="Hold at 4.2 V", tags=["step_id:four"])

        with pytest.raises(ValueError, match="Hold at 4.2 V"):
            step_id_of(leaf)


class TestDerivedViews:
    """The step descriptions, the cycle information and the column definitions.

    Each view derives from the protocol tree.
    """

    @pytest.mark.xfail(
        strict=True,
        reason="Procedure.step_descriptions reads a stored dict rather than "
        "the protocol tree",
    )
    def test_step_descriptions_pair_each_leaf_tag_with_its_description(
        self,
        procedure: Procedure,
    ) -> None:
        """The step descriptions pair each leaf's step identifier with its text."""
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

        assert procedure.step_descriptions == {
            "Step": [1, 2, 3],
            "Description": ["Charge", "Rest", "Discharge"],
        }

    @pytest.mark.xfail(
        strict=True,
        reason="Procedure.experiment resolves names against the legacy "
        "readme_dict, not the protocol tree",
    )
    def test_cycle_info_reads_the_repeat_count_and_the_bounds(
        self,
        procedure: Procedure,
    ) -> None:
        """The cycle information of an experiment reports its bounds and count."""
        attach_protocol(
            procedure,
            [
                Step(
                    mode="group",
                    description="Break-in",
                    count=5,
                    steps=[
                        Step(description="Discharge", tags=["step_id:4"]),
                        Step(description="Rest", tags=["step_id:5"]),
                        Step(description="Charge", tags=["step_id:6"]),
                        Step(description="Rest", tags=["step_id:7"]),
                    ],
                ),
            ],
        )

        experiment = procedure.experiment("Break-in")

        assert experiment.cycle_info == [(4, 7, 5)]

    @pytest.mark.xfail(
        strict=True,
        reason="Table.column_definitions returns the constructor argument rather "
        "than the merged ontology view",
    )
    def test_column_definition_comes_from_the_bdf_ontology(
        self,
        cycling_frame: pl.DataFrame,
    ) -> None:
        """A BDF column's definition comes from the ontology's definition field."""
        table = Table(cycling_frame)

        match = bdf.spec.COLUMN_ONTOLOGY.quantity_from_label("Current / A")
        assert match is not None
        expected = match[0].definition

        assert table.column_definitions["Current / A"] == expected

    @pytest.mark.xfail(
        strict=True,
        reason="Table.column_definitions returns the constructor argument rather "
        "than the merged ontology view",
    )
    def test_column_definition_of_a_non_ontology_column_comes_from_extras(
        self,
        cycling_frame: pl.DataFrame,
    ) -> None:
        """A column the ontology does not define reads its definition from extras."""
        table = Table(cycling_frame)
        table.metadata = bdf.Metadata(
            extras={
                "column_definitions": {
                    "Ambient Pressure / kPa": "the site's ambient pressure sensor",
                },
            },
        )

        assert (
            table.column_definitions["Ambient Pressure / kPa"]
            == "the site's ambient pressure sensor"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="Table.define_column writes to the released flat attribute "
        "rather than to extras",
    )
    def test_define_column_writes_into_the_metadata_extras(
        self,
        cycling_frame: pl.DataFrame,
    ) -> None:
        """A defined column lands in the record's extras, not a flat attribute."""
        table = Table(cycling_frame)

        table.define_column(
            "Ambient Pressure / kPa",
            "the site's ambient pressure sensor",
        )

        assert (
            table.metadata.extras[  # type: ignore[attr-defined]
                "column_definitions"
            ]["Ambient Pressure / kPa"]
            == "the site's ambient pressure sensor"
        )
