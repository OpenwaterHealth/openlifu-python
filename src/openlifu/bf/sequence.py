from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import pandas as pd

from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.dict_conversion import DictMixin


@dataclass
class Sequence(DictMixin):
    """
    Class for representing a sequence of pulses
    """

    pulse_interval: Annotated[float, OpenLIFUFieldData("Pulse interval (s)", "Interval between pulses in the sequence (s)")] = 1.0  # s
    """Interval between pulses in the sequence (s)"""

    pulse_count: Annotated[int, OpenLIFUFieldData("Pulse count", "Number of pulses in the sequence")] = 1
    """Number of pulses in the sequence"""

    pulse_train_interval: Annotated[float, OpenLIFUFieldData("Pulse train interval (s)", "Interval between pulse trains in the sequence (s)")] = 1.0  # s
    """Interval between pulse trains in the sequence (s)"""

    pulse_train_count: Annotated[int, OpenLIFUFieldData("Pulse train count", "Number of pulse trains in the sequence")] = 1
    """Number of pulse trains in the sequence"""

    focus_order: Annotated[list[int] | None, OpenLIFUFieldData("Focus order", "Optional focus index order for each pulse")] = None
    """Optional focus index order for each pulse"""

    def __post_init__(self):
        if self.pulse_interval <= 0:
            raise ValueError("Pulse interval must be positive")
        if self.pulse_count <= 0:
            raise ValueError("Pulse count must be positive")
        if self.pulse_train_interval < 0:
            raise ValueError("Pulse train interval must be non-negative")
        elif (self.pulse_train_interval > 0) and (self.pulse_train_interval < (self.pulse_interval * self.pulse_count)):
            raise ValueError("Pulse train interval must be greater than or equal to the total pulse interval")
        if self.pulse_train_count <= 0:
            raise ValueError("Pulse train count must be positive")
        if self.focus_order is not None:
            if len(self.focus_order) == 0:
                raise ValueError("Focus order must not be empty")
            if len(self.focus_order) != self.pulse_count:
                raise ValueError("Focus order length must match pulse count")
            if any(not isinstance(focus_index, int) for focus_index in self.focus_order):
                raise TypeError("Focus order entries must be integers")
            if any(focus_index < 1 for focus_index in self.focus_order):
                raise ValueError("Focus order entries must be positive")

    def to_table(self) -> pd.DataFrame:
        """
        Get a table of the sequence parameters

        :returns: Pandas DataFrame of the sequence parameters
        """
        records = [
            {"Name": "Pulse Interval", "Value": self.pulse_interval, "Unit": "s"},
            {"Name": "Pulse Count", "Value": self.pulse_count, "Unit": ""},
            {"Name": "Pulse Train Interval", "Value": self.pulse_train_interval, "Unit": "s"},
            {"Name": "Pulse Train Count", "Value": self.pulse_train_count, "Unit": ""},
            {"Name": "Focus Order", "Value": self.focus_order, "Unit": ""}
        ]
        return pd.DataFrame.from_records(records)

    def get_pulse_train_duration(self) -> float:
        """
        Get the duration of a single pulse train in seconds

        :returns: Duration of a single pulse train in seconds
        """
        return self.pulse_interval * self.pulse_count

    def get_sequence_duration(self) -> float:
        """
        Get the total duration of the sequence in seconds

        :returns: Total duration of the sequence in seconds
        """
        if self.pulse_train_interval == 0:
            interval = self.get_pulse_train_duration()
        else:
            interval = self.pulse_train_interval
        return interval * self.pulse_train_count
