"""A module for the Procedure class."""
from pybatdata.experiment import Experiment
from pybatdata.experiments.pulsing import Pulsing
from pybatdata.experiments.cycling import Cycling
import polars as pl
from pybatdata.base import Base
import re
import os

class Procedure(Base):
    """A class for a procedure in a battery experiment."""
    def __init__(self, 
                 data_path: str):
        """ Create a procedure class.
        
        Args:
            data_path (str): The path to the data parquet file.
        """
        lazyframe = pl.scan_parquet(data_path)
        data_folder = os.path.dirname(data_path)
        readme_path = os.path.join(data_folder, 'README.txt')
        self.titles, self.cycles_idx, self.steps_idx, self.step_names = self.process_readme(readme_path)
        super().__init__(lazyframe)
        
    def experiment(self, experiment_name: str)->Experiment:
        """Return an experiment object from the procedure.
        
        Args:
            experiment_name (str): The name of the experiment.
            
        Returns:
            Experiment: An experiment object from the procedure.
        """
        experiment_number = list(self.titles.keys()).index(experiment_name)
        cycles_idx = self.cycles_idx[experiment_number]
        steps_idx = self.steps_idx[experiment_number]
        conditions = [pl.col('Cycle').is_in(self.flatten(cycles_idx)),
                      pl.col('Step').is_in(self.flatten(steps_idx))]
        lf_filtered = self.lazyframe.filter(conditions)
        experiment_types = {'Constant Current': Experiment, 
                            'Pulsing': Pulsing, 
                            'Cycling': Cycling, 
                            'SOC Reset': Experiment}
        return experiment_types[self.titles[experiment_name]](lf_filtered)
    
    @classmethod
    def flatten(cls, lst: list) -> list:
        """Flatten a list of lists into a single list.
        
        Args:
            lst (list): The list of lists to flatten.
            
        Returns:
            list: The flattened list."""
        if not isinstance(lst, list):
            return [lst]
        if lst == []:
            return lst
        if isinstance(lst[0], list):
            return cls.flatten(lst[0]) + cls.flatten(lst[1:])
        return lst[:1] + cls.flatten(lst[1:])

    @classmethod
    def get_exp_conditions(cls, column: str, indices: list) -> pl.Expr:
        """Convert a list of indices for a column into a polars expression for filtering.
        
        Args:
            column (str): The column to filter.
            indices (list): The indices to filter.
            
        Returns:
            pl.Expr: The polars expression for filtering the column."""
        return pl.col(column).is_in(cls.flatten(indices)).alias(column)

    @staticmethod
    def process_readme(readme_path):
        """Function to process the README.txt file and extract the relevant information.
        
        Args:
            readme_path (str): The path to the README.txt file.
            
        Returns:
            dict: The titles of the experiments inside a procddure. Fomat {title: experiment type}.
            list: The step numbers inside the procedure.
            list: The cycle numbers inside the procedure.
            list: The names of the steps inside the procedure.
        """
        with open(readme_path, 'r') as file:
            lines = file.readlines()

        titles = {}
        title_index = 0
        for line in lines:
            if line.startswith('##'):    
                splitted_line = line[3:].split(":")
                titles[splitted_line[0].strip()] = splitted_line[1].strip()

        steps = [[[]] for _ in range(len(titles))]
        cycles = [[] for _ in range(len(titles))]
        line_index = 0
        title_index = -1
        cycle_index = 0
        while line_index < len(lines):
            if lines[line_index].startswith('##'):    
                title_index += 1
                cycle_index = 0
            if lines[line_index].startswith('#-'):
                match = re.search(r'Step (\d+)', lines[line_index])
                if match:
                    steps[title_index][cycle_index].append(int(match.group(1)))  # Append step number to the corresponding title's list
                latest_step = int(match.group(1))
            if lines[line_index].startswith('#x'):
                line_index += 1
                match = re.search(r'Starting step: (\d+)', lines[line_index])
                if match:
                    starting_step = int(match.group(1))
                line_index += 1
                match = re.search(r'Cycle count: (\d+)', lines[line_index])
                if match:
                    cycle_count = int(match.group(1))
                for i in range(cycle_count-1):
                    steps[title_index].append(list(range(starting_step, latest_step+1)))
                    cycle_index += 1
            line_index += 1

        cycles = [list(range(len(sublist))) for sublist in steps]
        for i in range(len(cycles)-1):
            cycles[i+1] = [item+cycles[i][-1] for item in cycles[i+1]]
        for i in range(len(cycles)): 
            cycles[i] = [item+1 for item in cycles[i]]
        
        step_names = [None for _ in range(steps[-1][-1][-1]+1)]
        line_index = 0
        while line_index < len(lines):
            if lines[line_index].startswith('#-'):    
                match = re.search(r'Step (\d+)', lines[line_index])
                if match: 
                    step_names[int(match.group(1))] = lines[line_index].split(': ')[1].strip()
            line_index += 1
        return titles, cycles, steps,  step_names