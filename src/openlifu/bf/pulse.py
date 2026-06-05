from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pandas as pd

from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.dict_conversion import DictMixin
from openlifu.util.field_display import summarize_fields


@dataclass
class Pulse(DictMixin):
    """
    Class for representing a sinusoidal pulse
    """

    frequency: Annotated[float, OpenLIFUFieldData(
        name="Frequency",
        description="Frequency of the pulse",
        units="Hz", display_units="kHz", precision=1,
    )] = 1.0  # Hz
    """Frequency of the pulse in Hz"""

    amplitude: Annotated[float, OpenLIFUFieldData(
        name="Amplitude",
        description="Amplitude of the pulse (between 0 and 1).",
        precision=2,
    )] = 1.0  # AU
    """Amplitude of the pulse in arbitrary units (AU) between 0 and 1"""

    duration: Annotated[float, OpenLIFUFieldData(
        name="Duration",
        description="Duration of the pulse",
        units="s", display_units="ms", precision=1,
    )] = 1.0  # s
    """Duration of the pulse in s"""

    def __post_init__(self):
        if self.frequency <= 0:
            raise ValueError("Frequency must be greater than 0")
        if self.amplitude < 0 or self.amplitude > 1:
            raise ValueError("Amplitude must be between 0 and 1")
        if self.duration <= 0:
            raise ValueError("Duration must be greater than 0")

    def calc_pulse(self, t: np.array):
        """
        Calculate the pulse at the given times

        :param t: Array of times to calculate the pulse at (s)
        :returns: Array of pulse values at the given times
        """
        return self.amplitude * np.sin(2*np.pi*self.frequency*t)

    def calc_time(self, dt: float):
        """
        Calculate the time array for the pulse for a particular timestep

        :param dt: Time step (s)
        :returns: Array of times for the pulse (s)
        """
        return np.arange(0, self.duration, dt)

    def to_table(self) -> pd.DataFrame:
        """
        Get a table of the pulse parameters

        :returns: Pandas DataFrame of the pulse parameters
        """
        records = [{"Name": "Frequency", "Value": self.frequency, "Unit": "Hz"},
                   {"Name": "Amplitude", "Value": self.amplitude, "Unit": "AU"},
                   {"Name": "Duration", "Value": self.duration, "Unit": "s"}]
        return pd.DataFrame.from_records(records)

    def get_summary(self) -> str:
        """Return a one-liner summary like ``Frequency: 400 kHz, Amplitude: 1, Duration: 5 ms``."""
        return summarize_fields(self, ("frequency", "amplitude", "duration"))
