"""Module for processing PyPrBE README files."""
from typing import Any, Dict, List, Tuple

import yaml


class ReadmeModel:
    """A class for processing the README.yaml file."""

    def __init__(self, readme_dict: Dict[str, Any]) -> None:
        """Initialize the ReadmeModel class."""
        experiment_names = readme_dict.keys()

        self.experiment_dict: Dict[
            str, Dict[str, List[str | int | Tuple[int, int, int]]]
        ] = {name: {} for name in experiment_names}
        self.step_details = None
        for experiment_name in experiment_names:
            step_numbers = list(readme_dict[experiment_name]["Steps"].keys())
            step_descriptions = list(readme_dict[experiment_name]["Steps"].values())
            cycle_keys = [
                key for key in readme_dict[experiment_name] if "cycle" in key.lower()
            ]
            exp_cycles: List[str | int | Tuple[int, int, int]] = []
            for cycle in cycle_keys:
                start = readme_dict[experiment_name][cycle]["Start"]
                end = readme_dict[experiment_name][cycle]["End"]
                count = readme_dict[experiment_name][cycle]["Count"]
                exp_cycles.append((start, end, count))
            self.experiment_dict[experiment_name]["Steps"] = step_numbers
            self.experiment_dict[experiment_name][
                "Step Descriptions"
            ] = step_descriptions
            self.experiment_dict[experiment_name]["Cycles"] = exp_cycles


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
