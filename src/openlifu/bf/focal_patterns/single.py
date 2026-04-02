from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from openlifu.bf.focal_patterns import FocalPattern
from openlifu.geo.point import Point


@dataclass
class SinglePoint(FocalPattern):
    """
    Class for representing a single focus

    :ivar target_pressure: Target pressure of the focal pattern in Pa
    """
    def get_targets(self, target: Point):
        """
        Get the targets of the focal pattern

        :param target: Target point of the focal pattern
        :returns: List of target points
        """
        return [target.copy()]

    def num_foci(self):
        """
        Get the number of foci in the focal pattern

        :returns: Number of foci (1)
        """
        return 1

    def to_table(self) -> pd.DataFrame:
        """
        Get a table of the focal pattern parameters

        :returns: Pandas DataFrame of the focal pattern parameters
        """
        records = [{"Name": "Type", "Value": "Single Point", "Unit": ""},
                   {"Name": "Target Pressure", "Value": self.target_pressure, "Unit": self.units}]
        return pd.DataFrame.from_records(records)
