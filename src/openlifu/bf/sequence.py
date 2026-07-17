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

    pulse_interval: Annotated[float, OpenLIFUFieldData(
        name="Pulse interval",
        description="Interval between pulses in the sequence",
        units="s", display_units="ms", precision=1,
    )] = 1.0  # s
    """Interval between pulses in the sequence (s)"""

    pulse_count: Annotated[int, OpenLIFUFieldData(
        name="Pulse count",
        description="Number of pulses in the sequence",
    )] = 1
    """Number of pulses in the sequence"""

    pulse_train_interval: Annotated[float, OpenLIFUFieldData(
        name="Pulse train interval",
        description="Interval between pulse trains in the sequence",
        units="s", display_units="s", precision=2,
    )] = 1.0  # s
    """Interval between pulse trains in the sequence (s)"""

    pulse_train_count: Annotated[int, OpenLIFUFieldData(
        name="Pulse train count",
        description="Number of pulse trains in the sequence",
    )] = 1
    """Number of pulse trains in the sequence"""

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

    def to_table(self) -> pd.DataFrame:
        """
        Get a table of the sequence parameters

        :returns: Pandas DataFrame of the sequence parameters
        """
        records = [
            {"Name": "Pulse Interval", "Value": self.pulse_interval, "Unit": "s"},
            {"Name": "Pulse Count", "Value": self.pulse_count, "Unit": ""},
            {"Name": "Pulse Train Interval", "Value": self.pulse_train_interval, "Unit": "s"},
            {"Name": "Pulse Train Count", "Value": self.pulse_train_count, "Unit": ""}
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

    def get_summary(self) -> str:
        """Return a one-liner summary of the sequence parameters.

        Format: ``"{pulse_count} pulses every {pulse_interval}ms,
        repeated {pulse_train_count}x every {pulse_train_interval}s"``.
        Numeric values use ``%g`` formatting (no trailing zeros).
        """
        pulse_interval_ms = self.pulse_interval * 1000.0
        pulse_train_interval_s = self.pulse_train_interval
        return (
            f"{int(self.pulse_count)} pulses every {pulse_interval_ms:g}ms, "
            f"repeated {int(self.pulse_train_count)}x every {pulse_train_interval_s:g}s"
        )
