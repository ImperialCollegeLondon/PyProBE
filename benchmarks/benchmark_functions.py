"""Functions to be used in the benchmarks."""

import os

from pyprobe.cell import Cell
from pyprobe.experiment import Experiment
from pyprobe.procedure import Procedure


def create_cell() -> Cell:
    """Create a Cell object."""
    return Cell(info={"Name": "Test_Cell"})


def add_procedure_from_parquet() -> Cell:
    """Add data to a Cell object from a parquet file."""
    cell = create_cell()
    cell.add_procedure(
        procedure_name="sample procedure",
        folder_path=os.path.join(
            os.path.dirname(__file__), "../tests/sample_data_neware/"
        ),
        filename="sample_data_neware.parquet",
    )
    return cell


add_procedure_from_parquet()


def return_procedure() -> Procedure:
    """Return a Procedure object."""
    cell = add_procedure_from_parquet()
    return cell.procedure["sample procedure"]


def return_cycling_experiment() -> Experiment:
    """Return a cycling Experiment object."""
    cell = add_procedure_from_parquet()
    return cell.procedure["sample procedure"].experiment("Break-in Cycles")
