"""A module for the Pulsing class."""

from pybatdata.experiment import Experiment
from pybatdata.step import Step
import polars as pl
import numpy as np

class Pulsing(Experiment):
    """A pulsing experiment in a battery procedure."""

    def __init__(self, lazyframe, cycles_idx, steps_idx, step_names):
        """Create a pulsing experiment.

            Args:
                lazyframe (polars.LazyFrame): The lazyframe of data being filtered.
                cycles_idx (list): The indices of the cycles of the pulsing experiment.
                steps_idx (list): The indices of the steps in the pulsing experiment.
                step_names (list): The names of all of the steps in the procedure.
        """
        super().__init__(lazyframe, cycles_idx, steps_idx, step_names)
        self.rests = [None]*len(cycles_idx)
        self.pulses = [None]*len(cycles_idx)

    @property
    def pulse_starts(self)->pl.DataFrame:
        """Find the start of the pulses in the pulsing experiment.
        
        Returns:
            pl.DataFrame: A dataframe with rows for the start of each pulse.
        """
        df = self.RawData.with_columns(pl.col('Current (A)').shift().alias('Prev Current'))
        df = df.with_columns(pl.col('Voltage (V)').shift().alias('Prev Voltage'))
        return  df.filter((df['Current (A)'].shift() == 0) & (df['Current (A)'] != 0))

    @property
    def V0(self) -> np.ndarray:
        """Find the voltage values immediately before each pulse.
        
        Returns:
            numpy.ndarray: The voltage values immediately before each pulse.
        """
        return self.pulse_starts['Prev Voltage'].to_numpy()
    
    @property
    def V1(self) -> np.ndarray:
        """Find the voltage values immediately after each pulse.
        
        Returns:
            numpy.ndarray: The voltage values immediately after each pulse.
        """
        return self.pulse_starts['Voltage (V)'].to_numpy()
    
    @property
    def I1(self) -> np.ndarray:
        """Find the current values immediately after each pulse.
        
        Returns:
            numpy.ndarray: The current values immediately after each pulse.
        """
        return self.pulse_starts['Current (A)'].to_numpy()

    @property
    def R0(self) -> np.ndarray:
        """Find the ohmic resistance for each pulse.
        
        Returns:
            numpy.ndarray: The ohmic resistance for each pulse."""
        return (self.V1-self.V0)/self.I1
    
    def Rt(self, t: float) -> np.ndarray:
        """Find the cell resistance at a given time after each pulse.
        
        Returns:
            numpy.ndarray: The cell resistance at a given time after each pulse.
        """
        t_point = self.pulse_starts['Time']+t
        Vt = np.zeros(len(t_point))
        for i in range(len(Vt)):
            condition = self.RawData['Time'] >= t_point[i]
            first_row = self.RawData.filter(condition).sort('Time').head(1)
            Vt[i] = first_row['Voltage (V)'].to_numpy()[0]
        return (Vt-self.V0)/self.I1    
        
    def pulse(self, pulse_number: int) -> Step:
        """Return a step object for a pulse in the pulsing experiment.
        
        Args:
            pulse_number (int): The pulse number to return.
            
        Returns:
            Step: A step object for a pulse in the pulsing experiment.
        """
        if self.pulses[pulse_number] is None:
            self.pulses[pulse_number] = self.cycle(pulse_number).chargeordischarge(0)
        return self.pulses[pulse_number]
    
    def rest(self, rest_number: int) -> Step:
        """Return a step object for a rest in the pulsing experiment.
        
        Args:
            rest_number (int): The rest number to return.
            
        Returns:
            Step: A step object for a rest in the pulsing experiment.
        """
        if self.rests[rest_number] is None:
            self.rests[rest_number] = self.cycle(rest_number).rest(0)
        return self.rests[rest_number]