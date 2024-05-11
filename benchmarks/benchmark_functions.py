"""Functions to be used in the benchmarks."""

import os

from pybatdata.cell import Cell
from pybatdata.cyclers.neware import Neware
from pybatdata.experiment import Experiment
from pybatdata.procedure import Procedure


def create_cell() -> Cell:
    """Create a Cell object."""
    return Cell(info={"Name": "Test_Cell"})


def add_data_from_parquet() -> Cell:
    """Add data to a Cell object from a parquet file."""
    cell = create_cell()
    cell.add_data(
        os.path.join(
            os.path.dirname(__file__),
            "../tests/sample_data_neware/sample_data_neware.parquet",
        ),
        "sample procedure",
        Neware(),
        skip_writing=True,
    )
    return cell


def return_procedure() -> Procedure:
    """Return a Procedure object."""
    cell = add_data_from_parquet()
    return cell.procedure["sample procedure"]


def return_cycling_experiment() -> Experiment:
    """Return a cycling Experiment object."""
    cell = add_data_from_parquet()
    return cell.procedure["sample procedure"].experiment("Break-in Cycles")
