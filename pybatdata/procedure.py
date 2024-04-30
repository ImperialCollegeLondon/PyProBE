"""A module for the Procedure class."""
import os
from typing import Dict, List

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
            pl.col("Step").is_in(steps_idx),
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
    def get_exp_conditions(cls, column: str, indices: List[int]) -> pl.Expr:
        """Convert a list of indices for a column into a polars expr for filtering.

        Args:
            column (str): The column to filter.
            indices (List[int]): The indices to filter.

        Returns:
            pl.Expr: The polars expression for filtering the column.
        """
        return pl.col(column).is_in(indices).alias(column)

    @staticmethod
    def process_readme(
        readme_path: str,
    ) -> tuple[Dict[str, str], List[List[int]]]:
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
        steps: List[List[int]] = []
        for experiment in readme_dict:
            if "Step Numbers" in readme_dict[experiment]:
                step_list = readme_dict[experiment]["Step Numbers"]
            else:
                step_list = list(range(len(readme_dict[experiment]["Steps"])))
                step_list = [x + max_step + 1 for x in step_list]
            max_step = step_list[-1]
            steps.append(step_list)

        return titles, steps
