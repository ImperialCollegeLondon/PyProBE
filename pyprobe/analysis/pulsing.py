"""A module for the Pulsing class."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import polars as pl
from numpy.typing import NDArray

from pyprobe.analysis.utils import BaseAnalysis, analysismethod
from pyprobe.filters import Experiment
from pyprobe.result import Result


@dataclass(kw_only=True)
class Pulsing(BaseAnalysis):
    """A pulsing experiment in a battery procedure.

    Args:
        experiment (RawData): The raw data for the pulsing experiment.
    """

    experiment: Experiment
    rests: List[Optional[Result]] = field(default_factory=list)
    pulses: List[Optional[Result]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Create a pulsing experiment."""
        self.rests: List[Optional[Result]] = [None] * self.experiment.data.select(
            "Cycle"
        ).n_unique("Cycle")
        self.pulses: List[Optional[Result]] = [None] * self.experiment.data.select(
            "Cycle"
        ).n_unique("Cycle")

    @property
    def pulse_starts(self) -> pl.DataFrame:
        """Find the start of the pulses in the pulsing experiment.

        Returns:
            pl.DataFrame: A dataframe with rows for the start of each pulse.
        """
        df = self.experiment.data.with_columns(
            pl.col("Current [A]").shift().alias("Prev Current")
        )
        df = df.with_columns(pl.col("Voltage [V]").shift().alias("Prev Voltage"))
        return df.filter((df["Current [A]"].shift() == 0) & (df["Current [A]"] != 0))

    @property
    def V0(self) -> NDArray[np.float64]:
        """Find the voltage values immediately before each pulse.

        Returns:
            numpy.ndarray: The voltage values immediately before each pulse.
        """
        return self.pulse_starts["Prev Voltage"].to_numpy()

    @property
    def V1(self) -> NDArray[np.float64]:
        """Find the voltage values immediately after each pulse.

        Returns:
            numpy.ndarray: The voltage values immediately after each pulse.
        """
        return self.pulse_starts["Voltage [V]"].to_numpy()

    @property
    def I1(self) -> NDArray[np.float64]:
        """Find the current values immediately after each pulse.

        Returns:
            numpy.ndarray: The current values immediately after each pulse.
        """
        return self.pulse_starts["Current [A]"].to_numpy()

    @property
    def R0(self) -> NDArray[np.float64]:
        """Find the ohmic resistance for each pulse.

        Returns:
            numpy.ndarray: The ohmic resistance for each pulse.
        """
        return (self.V1 - self.V0) / self.I1

    @property
    @analysismethod
    def pulse_summary(self) -> Result:
        """Find the resistance values for each pulse.

        Returns:
            Result: A result object containing resistance values for each pulse.
        """
        return Result(
            base_dataframe=pl.DataFrame(
                {
                    "Pulse number": list(range(len(self.R0))),
                    "R0 [Ohm]": self.R0,
                    "V0 [V]": self.V0,
                }
            ),
            info=self.experiment.info,
            column_definitions={
                "Pulse number": "The pulse number.",
                "R0 [Ohm]": "The ohmic resistance for each pulse.",
                "V0 [V]": "The voltage values immediately before each pulse.",
            },
        )

    def Rt(self, t: float) -> NDArray[np.float64]:
        """Find the cell resistance at a given time after each pulse.

        Returns:
            numpy.ndarray: The cell resistance at a given time after each pulse.
        """
        t_point = self.pulse_starts["Time [s]"] + t
        Vt = np.zeros(len(t_point))
        for i in range(len(Vt)):
            condition = self.experiment.data["Time [s]"] >= t_point[i]
            first_row = self.experiment.data.filter(condition).sort("Time [s]").head(1)
            Vt[i] = first_row["Voltage [V]"].to_numpy()[0]
        return (Vt - self.V0) / self.I1

    def pulse(self, pulse_number: int) -> Optional[Result]:
        """Return a step object for a pulse in the pulsing experiment.

        Args:
            pulse_number (int): The pulse number to return.

        Returns:
            Result: A step object for a pulse in the pulsing experiment.
        """
        if self.pulses[pulse_number] is None:
            self.pulses[pulse_number] = self.experiment.cycle(
                pulse_number
            ).chargeordischarge(0)
        return self.pulses[pulse_number]

    def pulse_rest(self, rest_number: int) -> Optional[Result]:
        """Return a step object for a rest in the pulsing experiment.

        Args:
            rest_number (int): The rest number to return.

        Returns:
            Result: A step object for a rest in the pulsing experiment.
        """
        if self.rests[rest_number] is None:
            self.rests[rest_number] = self.experiment.cycle(rest_number).rest(0)
        return self.rests[rest_number]
