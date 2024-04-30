"""A module for the Procedure class."""
import os
import re
from typing import Any, Dict, List

import polars as pl
import yaml

from pybatdata.experiment import Experiment
from pybatdata.experiments.cycling import Cycling
from pybatdata.experiments.pulsing import Pulsing
from pybatdata.result import Result


class Procedure(Result):
    """A class for a procedure in a battery experiment."""

    def __init__(
        self,
        data_path: str,
        info: Dict[str, str | int | float],
    ) -> None:
        """Create a procedure class.

        Args:
            data_path (str): The path to the data parquet file.
            info (Dict[str, str | int | float]): A dict containing test info.
        """
        lazyframe = pl.scan_parquet(data_path)
        data_folder = os.path.dirname(data_path)
        readme_path = os.path.join(data_folder, "README.yaml")
        (
            self.titles,
            self.steps_idx,
        ) = self.process_readme(readme_path)
        super().__init__(lazyframe, info)

    def experiment(self, experiment_name: str) -> Experiment:
        """Return an experiment object from the procedure.

        Args:
            experiment_name (str): The name of the experiment.

        Returns:
            Experiment: An experiment object from the procedure.
        """
        experiment_number = list(self.titles.keys()).index(experiment_name)
        steps_idx = self.steps_idx[experiment_number]
        conditions = [
            pl.col("Step").is_in(self.flatten(steps_idx)),
        ]
        lf_filtered = self.lazyframe.filter(conditions)
        experiment_types = {
            "Constant Current": Experiment,
            "Pulsing": Pulsing,
            "Cycling": Cycling,
            "SOC Reset": Experiment,
        }
        return experiment_types[self.titles[experiment_name]](lf_filtered, self.info)

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

    @classmethod
    def get_exp_conditions(cls, column: str, indices: List[int]) -> pl.Expr:
        """Convert a list of indices for a column into a polars expr for filtering.

        Args:
            column (str): The column to filter.
            indices (List[int]): The indices to filter.

        Returns:
            pl.Expr: The polars expression for filtering the column.
        """
        return pl.col(column).is_in(cls.flatten(indices)).alias(column)

    @staticmethod
    def process_txt(
        readme_path: str,
    ) -> tuple[Dict[str, str], List[List[int]], List[List[list[int]]]]:
        """Function to process the README.txt file and extract the relevant information.

        Args:
            readme_path (str): The path to the README.txt file.

        Returns:
            dict: The titles of the experiments inside a procddure.
                Fomat {title: experiment type}.
            list: The cycle numbers inside the procedure.
            list: The step numbers inside the procedure.
        """
        with open(readme_path, "r") as file:
            lines = file.readlines()

        titles = {}
        title_index = 0
        for line in lines:
            if line.startswith("##"):
                splitted_line = line[3:].split(":")
                titles[splitted_line[0].strip()] = splitted_line[1].strip()

        steps: List[List[List[int]]] = [[[]] for _ in range(len(titles))]
        cycles: List[List[int]] = [[] for _ in range(len(titles))]
        line_index = 0
        title_index = -1
        cycle_index = 0
        while line_index < len(lines):
            if lines[line_index].startswith("##"):
                title_index += 1
                cycle_index = 0
            if lines[line_index].startswith("#-"):
                match = re.search(r"Step (\d+)", lines[line_index])
                if match is not None:
                    steps[title_index][cycle_index].append(
                        int(match.group(1))
                    )  # Append step number to the corresponding title's list
                    latest_step = int(match.group(1))
            if lines[line_index].startswith("#x"):
                line_index += 1
                match = re.search(r"Starting step: (\d+)", lines[line_index])
                if match is not None:
                    starting_step = int(match.group(1))
                line_index += 1
                match = re.search(r"Cycle count: (\d+)", lines[line_index])
                if match is not None:
                    cycle_count = int(match.group(1))
                for i in range(cycle_count - 1):
                    steps[title_index].append(
                        list(range(starting_step, latest_step + 1))
                    )
                    cycle_index += 1
            line_index += 1

        cycles = [list(range(len(sublist))) for sublist in steps]
        for i in range(len(cycles) - 1):
            cycles[i + 1] = [item + cycles[i][-1] for item in cycles[i + 1]]
        for i in range(len(cycles)):
            cycles[i] = [item + 1 for item in cycles[i]]
        return titles, cycles, steps

    @staticmethod
    def process_readme(
        readme_path: str,
    ) -> tuple[Dict[str, str], List[List[List[int]]]]:
        """Function to process the README.yaml file.

        Args:
            readme_path (str): The path to the README.yaml file.

        Returns:
            dict: The titles of the experiments inside a procddure.
                Fomat {title: experiment type}.
            list: The cycle numbers inside the procedure.
            list: The step numbers inside the procedure.
        """
        with open(readme_path, "r") as file:
            readme_dict = yaml.safe_load(file)

        titles = {
            experiment: readme_dict[experiment]["Type"] for experiment in readme_dict
        }

        max_step = 0
        steps: List[List[List[int]]] = []
        for experiment in readme_dict:
            if "Step Numbers" in readme_dict[experiment]:
                step_list = readme_dict[experiment]["Step Numbers"]
            else:
                step_list = list(range(len(readme_dict[experiment]["Steps"])))
                step_list = [x + max_step + 1 for x in step_list]
            max_step = step_list[-1]
            steps_and_cycles = [
                step_list for _ in range(readme_dict[experiment]["Repeat"])
            ]
            steps.append(steps_and_cycles)

        return titles, steps
