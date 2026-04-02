from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated

import pandas as pd

from openlifu.bf import focal_patterns
from openlifu.geo.point import Point
from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.units import getunittype


@dataclass
class FocalPattern(ABC):
    """
    Abstract base class for representing a focal pattern
    """

    target_pressure: Annotated[float, OpenLIFUFieldData("Target pressure", "Target pressure of the focal pattern in given units")] = 1.0
    """Target pressure of the focal pattern in given units"""

    units: Annotated[str, OpenLIFUFieldData("Pressure units", "Pressure units")] = "Pa"
    """Pressure units"""

    def __post_init__(self):
        if self.target_pressure <= 0:
            raise ValueError("Target pressure must be greater than 0")
        if not isinstance(self.units, str):
            raise TypeError("Units must be a string")
        if getunittype(self.units) != 'pressure':
            raise ValueError(f"Units must be a pressure unit, got {self.units}")

    @abstractmethod
    def get_targets(self, target: Point):
        """
        Get the targets of the focal pattern

        :param target: Target point of the focal pattern
        :returns: List of target points
        """
        pass

    @abstractmethod
    def num_foci(self):
        """
        Get the number of foci in the focal pattern

        :returns: Number of foci
        """
        pass

    def to_dict(self):
        """
        Convert the focal pattern to a dictionary

        :returns: Dictionary of the focal pattern parameters
        """
        d = self.__dict__.copy()
        d['class'] = self.__class__.__name__
        return d

    @staticmethod
    def from_dict(d):
        """
        Create a focal pattern from a dictionary

        :param d: Dictionary of the focal pattern parameters
        :returns: FocalPattern object
        """
        d = d.copy()
        short_classname = d.pop("class")
        module_dict = focal_patterns.__dict__
        class_constructor = module_dict[short_classname]
        return class_constructor(**d)

    @abstractmethod
    def to_table(self) -> pd.DataFrame:
        """
        Get a table of the focal pattern parameters

        :returns: Pandas DataFrame of the focal pattern parameters
        """
        pass
