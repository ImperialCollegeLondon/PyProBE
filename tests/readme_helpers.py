"""Helpers for attaching a legacy README to a loaded procedure in tests."""

from pathlib import Path
from typing import cast

from pyprobe.filters import Procedure
from pyprobe.protocol import Step, leaves, step_id_of


def attach_readme(procedure: Procedure, readme_path: str | Path) -> Procedure:
    """Attach a legacy README's experiment definitions to *procedure*.

    Args:
        procedure: The procedure to update.
        readme_path: Path to a README.yaml file.

    Returns:
        Procedure: *procedure*, with its experiment definitions populated.
    """
    procedure.attach_legacy_readme(readme_path)
    protocol = procedure.metadata.battinfo_test_protocol
    method = (protocol.method or []) if protocol is not None else []
    procedure.readme_dict = {
        str(group.description): _experiment_definition(group) for group in method
    }
    procedure._populate_step_descriptions()  # noqa: SLF001
    return procedure


def _experiment_definition(
    group: Step,
) -> dict[str, list[str | int | tuple[int, int, int]]]:
    """Return the legacy definition of the experiment that *group* holds.

    Args:
        group: The group node of one experiment.

    Returns:
        dict[str, list[str | int | tuple[int, int, int]]]: The step numbers,
            the step descriptions and the cycles of the experiment. The
            descriptions are empty where no step of the experiment names one.
    """
    step_leaves = leaves(group)
    descriptions = [leaf.description for leaf in step_leaves]
    return {
        "Steps": [step_id_of(leaf) for leaf in step_leaves],
        "Step Descriptions": descriptions if _named(descriptions) else [],
        "Cycles": _cycles(group),
    }


def _named(descriptions: list[str | None]) -> bool:
    """Report whether any step of an experiment names a description.

    Args:
        descriptions: The description of each step, in order.

    Returns:
        bool: True where at least one step names a description.
    """
    return any(description is not None for description in descriptions)


def _cycles(node: Step) -> list[str | int | tuple[int, int, int]]:
    """Return the bounds and the count of each repeat under *node*.

    Args:
        node: The node to walk. A count on the node itself counts as a repeat.

    Returns:
        list[str | int | tuple[int, int, int]]: The first step number, the
            last step number and the repeat count of each repeating group, in
            tree order.
    """
    found: list[str | int | tuple[int, int, int]] = []
    if node.count is not None:
        step_ids = cast(list[int], [step_id_of(leaf) for leaf in leaves(node)])
        found.append((step_ids[0], step_ids[-1], node.count))
    for child in node.steps or []:
        if child.mode == "group":
            found.extend(_cycles(child))
    return found
