"""A module for the Procedure class."""
import os
from typing import Any, Dict, List, Optional

import polars as pl
import yaml

from pyprobe.experiments.baseexperiment import BaseExperiment
from pyprobe.experiments.cycling import Cycling
from pyprobe.experiments.pOCV import pOCV
from pyprobe.experiments.pulsing import Pulsing
from pyprobe.filter import Filter


class Procedure(Filter):
    """A class for a procedure in a battery experiment."""

    def __init__(
        self,
        data_path: str,
        info: Dict[str, str | int | float],
        custom_readme_name: Optional[str] = None,
    ) -> None:
        """Create a procedure class.

        Args:
            data_path (str): The path to the data parquet file.
            info (Dict[str, str | int | float]): A dict containing test info.
            custom_readme_name (str, optional): The name of the custom README file.
                Defaults to None.
        """
        _data = pl.scan_parquet(data_path)
        data_folder = os.path.dirname(data_path)
        if custom_readme_name:
            readme_path = os.path.join(data_folder, f"{custom_readme_name}.yaml")
        else:
            readme_path = os.path.join(data_folder, "README.yaml")
        (
            self.titles,
            self.steps_idx,
        ) = self.process_readme(readme_path)
        super().__init__(_data, info)

    def experiment(self, *experiment_names: str) -> BaseExperiment:
        """Return an experiment object from the procedure.

        Args:
            experiment_names (str): Variable-length argument list of
                experiment names.

        Returns:
            BaseExperiment: An experiment object from the procedure.
        """
        experiment_types = {
            "Constant Current": BaseExperiment,
            "Pulsing": Pulsing,
            "Cycling": Cycling,
            "pOCV": pOCV,
            "SOC Reset": BaseExperiment,
            "General": BaseExperiment,
        }
        steps_idx = []
        for experiment_name in experiment_names:
            if experiment_name not in self.titles:
                raise ValueError(f"{experiment_name} not in procedure.")
            experiment_number = list(self.titles.keys()).index(experiment_name)
            steps_idx.append(self.steps_idx[experiment_number])
        flattened_steps = self.flatten(steps_idx)
        conditions = [
            pl.col("Step").is_in(flattened_steps),
        ]
        lf_filtered = self._data.filter(conditions)
        if len(experiment_names) == 1:
            experiment_obj = experiment_types[self.titles[experiment_names[0]]]
        else:
            experiment_obj = BaseExperiment
        return experiment_obj(lf_filtered, self.info)

    @property
    def experiment_names(self) -> List[str]:
        """Return the names of the experiments in the procedure.

        Returns:
            List[str]: The names of the experiments in the procedure.
        """
        return list(self.titles.keys())

    def verify_yaml(self, readme_name: str) -> str:
        """Verify that the readme has YAML extension.

        Args:
            readme_name (str): The name of the README file.
        """
        # Get the file extension of output_filename
        _, ext = os.path.splitext(readme_name)

        # If the file extension is not .parquet, replace it with .parquet
        if ext != ".yaml":
            readme_name = os.path.splitext(readme_name)[0] + ".yaml"
        return readme_name

    @staticmethod
    def process_readme(
        readme_path: str,
    ) -> tuple[Dict[str, str], List[List[int]]]:
        """Function to process the README.yaml file.

        Args:
            readme_path (str): The path to the README.yaml file.

        Returns:
            Tuple[Dict[str, str], List[int], List[int]]:
                - dict: The titles of the experiments inside a procedure.
                    Format {title: experiment type}.
                - list: The cycle numbers inside the procedure.
                - list: The step numbers inside the procedure.
        """
        with open(readme_path, "r") as file:
            readme_dict = yaml.safe_load(file)

        titles = {
            experiment: readme_dict[experiment]["Type"] for experiment in readme_dict
        }

        max_step = 0
        steps: List[List[int]] = []
        for experiment in readme_dict:
            if "Step Numbers" in readme_dict[experiment]:
                step_list = readme_dict[experiment]["Step Numbers"]
            elif "Total Steps" in readme_dict[experiment]:
                step_list = list(range(readme_dict[experiment]["Total Steps"]))
                step_list = [x + max_step + 1 for x in step_list]
            else:
                step_list = list(range(len(readme_dict[experiment]["Steps"])))
                step_list = [x + max_step + 1 for x in step_list]
            max_step = step_list[-1]
            steps.append(step_list)

        return titles, steps

    @classmethod
    def flatten(cls, lst: int | List[Any]) -> List[int]:
        """Flatten a list of lists into a single list.

        Args:
            lst (list): The list of lists to flatten.

        Returns:
            list: The flattened list.
        """
        if not isinstance(lst, list):
            return [lst]
        else:
            return [item for sublist in lst for item in cls.flatten(sublist)]
