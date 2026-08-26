"""Module for processing PyPrBE README files."""

from pathlib import Path
from typing import Any, cast

import yaml
from loguru import logger

from pyprobe import utils
from pyprobe.protocol import Step, leaves, step_id_of, step_id_tag


class ReadmeModel:
    """A class for processing the README.yaml file."""

    def __init__(self, readme_dict: dict[str, Any]) -> None:
        """Initialize the ReadmeModel class."""
        self.readme_dict = readme_dict
        experiment_names = self.readme_dict.keys()

        self.experiment_dict: dict[
            str,
            dict[str, list[str | int | tuple[int, int, int]]],
        ] = {name: {} for name in experiment_names}
        self.step_details = None
        for experiment_name in experiment_names:
            if "Steps" in self.readme_dict[experiment_name]:
                if isinstance(self.readme_dict[experiment_name]["Steps"], dict):
                    self._process_explicit_experiment(experiment_name)
                elif isinstance(self.readme_dict[experiment_name]["Steps"], list):
                    self._process_implicit_experiment(experiment_name)
                else:
                    error_msg = "Invalid format for steps in README file"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            elif "Total Steps" in self.readme_dict[experiment_name]:
                self._process_total_steps_experiment(experiment_name)
            else:
                error_msg = "Each experiment must have a 'Steps' or 'Total Steps' key."
                logger.error(error_msg)
                raise ValueError(error_msg)

    def _process_explicit_experiment(self, experiment_name: str) -> None:
        """Process an experiment with explicit step numbers.

        Args:
            experiment_name (str): The name of the experiment.
        """
        step_numbers = list(self.readme_dict[experiment_name]["Steps"].keys())
        step_descriptions = list(self.readme_dict[experiment_name]["Steps"].values())
        cycle_keys = [
            key for key in self.readme_dict[experiment_name] if "cycle" in key.lower()
        ]
        exp_cycles: list[str | int | tuple[int, int, int]] = []
        for cycle in cycle_keys:
            start = self.readme_dict[experiment_name][cycle]["Start"]
            end = self.readme_dict[experiment_name][cycle]["End"]
            count = self.readme_dict[experiment_name][cycle]["Count"]
            exp_cycles.append((start, end, count))
        self.experiment_dict[experiment_name]["Steps"] = step_numbers
        self.experiment_dict[experiment_name]["Step Descriptions"] = step_descriptions
        self.experiment_dict[experiment_name]["Cycles"] = exp_cycles

    def _process_implicit_experiment(self, experiment_name: str) -> None:
        """Process an experiment with implicit step numbers.

        Args:
            experiment_name (str): The name of the experiment.
        """
        max_step = self._get_max_step()
        step_descriptions = self.readme_dict[experiment_name]["Steps"]
        step_numbers = list(range(max_step + 1, max_step + len(step_descriptions) + 1))

        self.experiment_dict[experiment_name]["Steps"] = cast(
            list[str | int | tuple[int, int, int]],
            step_numbers,
        )  # cast to satisfy mypy
        self.experiment_dict[experiment_name]["Step Descriptions"] = step_descriptions
        self.experiment_dict[experiment_name]["Cycles"] = []

    def _process_total_steps_experiment(self, experiment_name: str) -> None:
        """Process an experiment with total steps.

        Args:
            experiment_name (str): The name of the experiment.
        """
        total_steps = self.readme_dict[experiment_name]["Total Steps"]
        max_step = self._get_max_step()
        step_numbers = list(range(max_step + 1, max_step + total_steps + 1))
        self.experiment_dict[experiment_name]["Steps"] = cast(
            list[str | int | tuple[int, int, int]],
            step_numbers,
        )  # cast to satisfy mypy
        self.experiment_dict[experiment_name]["Step Descriptions"] = []
        self.experiment_dict[experiment_name]["Cycles"] = []

    def _get_max_step(self) -> int:
        """Get the maximum step number from the experiment dictionary.

        Returns:
            int: The maximum step number from previously processed experiments.
        """
        all_steps = [
            experiment["Steps"]
            for experiment in self.experiment_dict.values()
            if "Steps" in experiment
        ]
        return max(utils.flatten_list(all_steps)) if all_steps else 0


def process_readme(
    readme_path: str,
) -> "ReadmeModel":
    """Function to process the README.yaml file.

    Args:
        readme_path (str): The path to the README.yaml file.

    Returns:
        Tuple[List[str], List[List[int]], Optional[pybamm.Experiment]]
            - List[str]: The list of titles from the README.yaml file.
            - List[List[int]]: The list of steps from the README.yaml file.
            - Optional[pybamm.Experiment]: The PyBaMM experiment object.
    """
    with open(readme_path) as file:
        readme_dict = yaml.safe_load(file)
    return ReadmeModel(readme_dict=readme_dict)


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
