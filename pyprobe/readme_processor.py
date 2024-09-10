"""Module for processing PyPrBE README files."""
import warnings
from typing import Any, Dict, List, Optional, Tuple

import pybamm
import yaml
from pydantic import BaseModel, Field, model_validator


class ReadmeModel(BaseModel):
    """A Pydantic BaseModel class for processing the README.yaml file."""

    readme_dict: Dict[str, Any]
    """A dictionary containing the contents of the README.yaml file."""
    readme_type: List[str] = Field(default_factory=list)
    """A list of strings indicating the README format for each experiment."""
    number_of_experiments: int = Field(default_factory=int)
    """The number of experiments in the README file."""
    titles: List[str] = Field(default_factory=list)
    """A list of strings containing the titles of the experiments."""
    step_numbers: List[List[int]] = Field(default_factory=list)
    """A list of lists containing the step numbers for each experiment."""
    step_indices: List[List[int]] = Field(default_factory=list)
    """A list of lists containing the step indices for each experiment."""
    step_descriptions: List[List[str]] = Field(default_factory=list)
    """A list of lists containing the step descriptions for each experiment."""
    cycle_details: List[List[Tuple[int, int, int]]] = Field(default_factory=list)
    """A list of lists containing the cycle details for each experiment."""
    pybamm_experiment_descriptions: List[Tuple[str, ...]] = Field(default_factory=list)
    """A list of tuples containing the PyBaMM experiment descriptions for each
    experiment."""
    pybamm_experiment_list: List[pybamm.Experiment] = Field(default_factory=list)
    """A list of PyBaMM experiment objects for each experiment."""
    pybamm_experiment: Optional[pybamm.Experiment] = Field(default=None)
    """A PyBaMM experiment object for all experiments."""

    class Config:
        """Pydantic configuration settings."""

        arbitrary_types_allowed = True

    @model_validator(mode="before")
    @classmethod
    def check_readme_dict(cls, data: Any) -> "ReadmeModel":
        """Validate the structure of the README.yaml file.

        Args:
            data (Any): The data to be validated.

        Returns:
            ReadmeModel: The validated data.

        Raises:
            TypeError: If the 'Steps' field is not a dictionary or a list.
            TypeError:
                If the 'Steps' field is a dictionary and the keys are not integers
                or the values are not strings.
            TypeError: If the 'Steps' field is a list and the values are not strings.
            TypeError: If the 'cycle' fields are not dictionaries.
            TypeError:
                If the 'cycle' fields are dictionaries and the keys are not strings
                or the values are not integers.
        """
        readme_dict = data["readme_dict"]
        data["readme_type"] = []
        cls.number_of_experiments = len(readme_dict)
        for experiment in readme_dict:
            if "Steps" in readme_dict[experiment]:
                steps = readme_dict[experiment]["Steps"]
                if isinstance(steps, dict):
                    if not all(
                        isinstance(k, int) and isinstance(v, str)
                        for k, v in steps.items()
                    ):
                        raise TypeError(
                            "The 'Steps' field must be a dictionary with keys of type"
                            " int and values of type str"
                        )
                    cycle_keys = [
                        key
                        for key in readme_dict[experiment].keys()
                        if "cycle" in key.lower()
                    ]
                    for cycle in cycle_keys:
                        cycle_dict = readme_dict[experiment][cycle]
                        if not all(
                            isinstance(k, str) and isinstance(v, int)
                            for k, v in cycle_dict.items()
                        ):
                            raise TypeError(
                                f"{cycle} must be a dictionary with keys of type str"
                                " and values of type int"
                            )
                    data["readme_type"].append("explicit")
                elif isinstance(steps, list):
                    if not all(isinstance(step, str) for step in steps):
                        raise TypeError("The 'Steps' field must be a list of strings")
                    data["readme_type"].append("implicit")
            elif "Total Steps" in readme_dict[experiment]:
                data["readme_type"].append("total")
        return data

    def model_post_init(self, __context: Any) -> None:
        """Get all the attributes of the class."""
        self.titles = list(self.readme_dict.keys())
        self.step_numbers = self._get_step_numbers()
        self.step_indices = self._get_step_indices()
        self.step_descriptions = self._get_step_descriptions()
        self.cycle_details = self._get_cycle_details()
        self.pybamm_experiment_descriptions = self._get_pybamm_experiment_descriptions()
        self.pybamm_experiment_list = self._get_pybamm_experiment_list()
        self.pybamm_experiment = self._get_pybamm_experiment()

    def _get_step_numbers(self) -> List[List[int]]:
        """Get the step numbers from the README.yaml file.

        Returns:
            List[List[int]]:
                A list of lists containing the step numbers for each
                experiment.
        """
        max_step = 0
        all_steps = []
        for experiment, readme_format in zip(self.readme_dict, self.readme_type):
            if readme_format == "explicit":
                exp_steps = list(self.readme_dict[experiment]["Steps"].keys())
            elif readme_format == "total":
                exp_steps = list(range(self.readme_dict[experiment]["Total Steps"]))
                exp_steps = [x + max_step + 1 for x in exp_steps]
            else:
                exp_steps = list(range(len(self.readme_dict[experiment]["Steps"])))
                exp_steps = [x + max_step + 1 for x in exp_steps]
            max_step = exp_steps[-1]
            all_steps.append(exp_steps)
        return all_steps

    def _get_step_indices(self) -> List[List[int]]:
        """Get the step indices from the README.yaml file.

        Returns:
            List[List[int]]:
                A list of lists containing the step indices for each
                experiment.
        """
        step_indices = []
        for exp_step_numbers in self.step_numbers:
            step_indices.append(list(range(len(exp_step_numbers))))
        return step_indices

    def _get_step_descriptions(self) -> List[List[str]]:
        """Get the step descriptions from the README.yaml file.

        Returns:
            List[List[str]]:
                A list of lists containing the step descriptions for each
                experiment.
        """
        all_descriptions = []
        for experiment, readme_format in zip(self.readme_dict, self.readme_type):
            if readme_format == "explicit":
                exp_step_descriptions = list(
                    self.readme_dict[experiment]["Steps"].values()
                )
            elif readme_format == "implicit":
                exp_step_descriptions = self.readme_dict[experiment]["Steps"]
            else:
                exp_step_descriptions = []
            all_descriptions.append(exp_step_descriptions)
        return all_descriptions

    def _get_cycle_details(self) -> List[List[Tuple[int, int, int]]]:
        """Get the cycle details from the README.yaml file.

        Returns:
            List[List[Tuple[int, int, int]]]:
                A list of lists containing the cycle details for each
                experiment.
        """
        cycles = []
        for experiment, readme_format, step_numbers in zip(
            self.readme_dict, self.readme_type, self.step_numbers
        ):
            exp_cycles = []
            if readme_format == "explicit":
                cycle_keys = [
                    key
                    for key in self.readme_dict[experiment].keys()
                    if "cycle" in key.lower()
                ]
                for cycle in cycle_keys:
                    cycle_dict = self.readme_dict[experiment][cycle]
                    start = cycle_dict["Start"]
                    end = cycle_dict["End"]
                    count = cycle_dict["Count"]
                    exp_cycles.append(
                        (step_numbers.index(start), step_numbers.index(end), count)
                    )
            cycles.append(exp_cycles)
        return cycles

    def _get_pybamm_experiment_descriptions(self) -> List[Tuple[str, ...]]:
        """Get the PyBaMM experiment objects from the README.yaml file.

        Returns:
            List[Tuple[str, ...]]:
                A list of tuples containing the PyBaMM experiment descriptions for each
                experiment.
        """
        all_descriptions = []
        for step_descriptions, step_indices, step_numbers, cycle_details in zip(
            self.step_descriptions,
            self.step_indices,
            self.step_numbers,
            self.cycle_details,
        ):
            final_descriptions = []
            if len(step_descriptions) > 0:
                expanded_indices = self._expand_cycles(step_indices, cycle_details)
                expanded_descriptions = [step_descriptions[i] for i in expanded_indices]
                # split any descriptions seperated by commas
                for desciption in expanded_descriptions:
                    line = desciption.split(",")
                    for item in line:
                        final_descriptions.append(item.strip())
            all_descriptions.append(tuple(final_descriptions))
        return all_descriptions

    def _get_pybamm_experiment_list(self) -> List[pybamm.Experiment]:
        """Get the PyBaMM experiment objects from the README.yaml file.

        Returns:
            List[pybamm.Experiment]:
                A list of PyBaMM experiment objects for each experiment.
        """
        pybamm_experiments = []
        for experiment, descriptions in zip(
            self.readme_dict, self.pybamm_experiment_descriptions
        ):
            if len(descriptions) > 0:
                try:
                    pybamm_experiments.append(pybamm.Experiment(descriptions))
                except Exception as e:
                    warnings.warn(
                        f"PyBaMM experiment could not be created for experiment:"
                        f" {experiment}. {e}"
                    )
                    pybamm_experiments.append(None)
            else:
                pybamm_experiments.append(None)
                warnings.warn(
                    f"PyBaMM experiment could not be created for experiment:"
                    f" {experiment} as there are no step descriptions."
                )
        return pybamm_experiments

    def _get_pybamm_experiment(self) -> Optional[pybamm.Experiment]:
        """Get the PyBaMM experiment object from the README.yaml file.

        Returns:
            Optional[pybamm.Experiment]:
                A PyBaMM experiment object for all experiments.
        """
        if any(exp is None for exp in self.pybamm_experiment_list):
            warnings.warn(
                "Some experiments do not have valid step descriptions."
                " Unable to create PyBaMM experiment."
            )
            return None
        else:
            all_descriptions = [exp for exp in self.pybamm_experiment_descriptions]
            return pybamm.Experiment(all_descriptions)

    @staticmethod
    def _expand_cycles(
        indices: List[int], cycles: List[Tuple[int, int, int]]
    ) -> List[int]:
        """Expand a list of cycle details into a list of step indices.

        Args:
            indices (List[int]): A list of step indices.
            cycles (List[Tuple[int, int, int]]): A list of cycle details.

        Returns:
            List[int]:
            A list of step indices expanded with every step in the order of occurrence.
        """
        if len(cycles) == 0:
            return indices
        repeated_list = indices
        for cycle in cycles:
            # cycle = (start, end, repeats)
            start = cycle[0]
            end = cycle[1] + 1
            repeats = cycle[2]

            # get sublist
            sublist = indices[start:end]

            # repeat sublist
            repeated_sublist = sublist * repeats

            # insert repeated sublist into repeated_list
            result: List[int] = []
            i = 0
            while i < len(repeated_list):
                if repeated_list[i : i + len(sublist)] == sublist:
                    result.extend(repeated_sublist)
                    i += len(sublist)
                else:
                    result.append(repeated_list[i])
                    i += 1
            repeated_list = result
        return result


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
