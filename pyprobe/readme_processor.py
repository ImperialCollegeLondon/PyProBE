"""Module for processing PyPrBE README files."""

import logging
from typing import Any, Dict, List, Tuple, Union, cast

import yaml

from pyprobe import utils

logger = logging.getLogger(__name__)


class ReadmeModel:
    """A class for processing the README.yaml file."""

    def __init__(self, readme_dict: Dict[str, Any]) -> None:
        """Initialize the ReadmeModel class."""
        self.readme_dict = readme_dict
        experiment_names = self.readme_dict.keys()

        self.experiment_dict: Dict[
            str, Dict[str, List[str | int | Tuple[int, int, int]]]
        ] = {name: {} for name in experiment_names}
        self.step_details = None
        for experiment_name in experiment_names:
            if "Steps" in self.readme_dict[experiment_name].keys():
                if isinstance(self.readme_dict[experiment_name]["Steps"], dict):
                    self._process_explicit_experiment(experiment_name)
                elif isinstance(self.readme_dict[experiment_name]["Steps"], list):
                    self._process_implicit_experiment(experiment_name)
                else:
                    error_msg = "Invalid format for steps in README file"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            elif "Total Steps" in self.readme_dict[experiment_name].keys():
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
        exp_cycles: List[str | int | Tuple[int, int, int]] = []
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
            List[Union[str, int, Tuple[int, int, int]]], step_numbers
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
            List[Union[str, int, Tuple[int, int, int]]], step_numbers
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
    with open(readme_path, "r") as file:
        readme_dict = yaml.safe_load(file)
    return ReadmeModel(readme_dict=readme_dict)
