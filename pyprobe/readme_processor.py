"""Module for the conversion of a legacy README file to a protocol tree."""

from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from pyprobe.protocol import Step, leaves, step_id_of, step_id_tag


def readme_to_method(readme: dict[str, Any]) -> list[Step]:
    """Convert a legacy README dictionary to a test protocol tree.

    Each experiment of the README becomes a group node that carries the
    experiment name as its description. Each step becomes a leaf node under
    that group, and it carries its step number as a ``"step_id:"`` tag. Each
    cycle becomes a ``count`` on the group that repeats.

    Args:
        readme: The dictionary that a README.yaml file holds.

    Returns:
        list[Step]: The protocol tree, with one group node per experiment.

    Raises:
        ValueError: If a cycle does not bound a contiguous group of steps.
            The message names the experiment and the cycle key.
    """
    method: list[Step] = []
    max_step = 0
    for name, experiment in readme.items():
        steps = _experiment_steps(experiment, max_step)
        max_step = max([max_step, *(number for number, _ in steps)])
        group = Step(
            mode="group",
            description=name,
            steps=[
                Step(description=description, tags=[step_id_tag(number)])
                for number, description in steps
            ],
        )
        for key in [key for key in experiment if "cycle" in key.lower()]:
            cycle = experiment[key]
            if not _apply_cycle(group, cycle["Start"], cycle["End"], cycle["Count"]):
                error_msg = (
                    f"'{key}' of experiment '{name}' bounds steps "
                    f"{cycle['Start']} to {cycle['End']}, which are not a "
                    "contiguous group of that experiment's steps."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        method.append(group)
    return method


def read_readme(readme_path: str | Path) -> list[Step]:
    """Read a legacy README.yaml file and convert it to a protocol tree.

    Args:
        readme_path: The path to the README.yaml file.

    Returns:
        list[Step]: The protocol tree, with one group node per experiment.

    Raises:
        FileNotFoundError: If the README file does not exist.
        ValueError: If a cycle does not bound a contiguous group of steps.
            The message names the experiment and the cycle key.
    """
    path = Path(readme_path)
    if not path.is_file():
        error_msg = f"README file not found: {path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    with path.open() as file:
        return readme_to_method(yaml.safe_load(file))


def _experiment_steps(
    experiment: dict[str, Any],
    max_step: int,
) -> list[tuple[int, str | None]]:
    """Return the step numbers and descriptions of one README experiment.

    An experiment states its steps as a mapping of number to description, as
    a list of descriptions, or as a total count. The last two forms number
    their steps from the step that follows *max_step*.

    Args:
        experiment: The definition of one experiment.
        max_step: The highest step number of the experiments before this one.

    Returns:
        list[tuple[int, str | None]]: The step number and the description of
            each step, in order. A description is None where the experiment
            states a total count alone.

    Raises:
        ValueError: If the experiment states its steps in no known form.
    """
    if "Steps" in experiment:
        steps = experiment["Steps"]
        if isinstance(steps, dict):
            return list(steps.items())
        if isinstance(steps, list):
            return [
                (max_step + offset, description)
                for offset, description in enumerate(steps, start=1)
            ]
        error_msg = "Invalid format for steps in README file"
        logger.error(error_msg)
        raise ValueError(error_msg)
    if "Total Steps" in experiment:
        total = experiment["Total Steps"]
        return [(number, None) for number in range(max_step + 1, max_step + total + 1)]
    error_msg = "Each experiment must have a 'Steps' or 'Total Steps' key."
    logger.error(error_msg)
    raise ValueError(error_msg)


def _apply_cycle(node: Step, start: int, end: int, count: int) -> bool:
    """Repeat the run of nodes that a cycle bounds, and report the success.

    The run repeats through the ``count`` of the node that holds it. Where the
    run covers every child of *node*, the count lands on *node* itself.
    Otherwise the run becomes a new group node in its place.

    Args:
        node: The node to search, and the node the count can land on.
        start: The step number that the cycle starts at.
        end: The step number that the cycle ends at.
        count: The number of repeats that the cycle declares.

    Returns:
        bool: True where the cycle bounds a contiguous run under this node.
    """
    children = node.steps or []
    leaf_ids = [[step_id_of(leaf) for leaf in leaves(child)] for child in children]
    first = [index for index, ids in enumerate(leaf_ids) if ids and ids[0] == start]
    last = [index for index, ids in enumerate(leaf_ids) if ids and ids[-1] == end]
    if first and last and first[0] <= last[0]:
        opening, closing = first[0], last[0]
        if opening == 0 and closing == len(children) - 1 and node.count is None:
            node.count = count
        else:
            node.steps = [
                *children[:opening],
                Step(
                    mode="group",
                    count=count,
                    steps=children[opening : closing + 1],
                ),
                *children[closing + 1 :],
            ]
        return True
    return any(
        child.mode == "group" and _apply_cycle(child, start, end, count)
        for child in children
    )
