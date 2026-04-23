from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pandas as pd
import xarray as xa

from openlifu.bf.apod_methods import ApodizationMethod
from openlifu.geo.point import Point
from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.units import getunittype
from openlifu.xdc import Transducer


@dataclass
class PiecewiseLinear(ApodizationMethod):
    zero_angle: Annotated[float, OpenLIFUFieldData("Zero Apodization Angle", "Angle at and beyond which the piecewise linear apodization is 0%")] = 90.0
    """Angle at and beyond which the piecewise linear apodization is 0%"""

    rolloff_angle: Annotated[float, OpenLIFUFieldData("Rolloff start angle", "Angle below which the piecewise linear apodization is 100%")] = 45.0
    """Angle below which the piecewise linear apodization is 100%"""

    units: Annotated[str, OpenLIFUFieldData("Angle units", "Angle units")] = "deg"
    """Angle units"""

    def __post_init__(self):
        if not isinstance(self.zero_angle, int | float):
            raise TypeError(f"Zero angle must be a number, got {type(self.zero_angle).__name__}.")
        if self.zero_angle < 0:
            raise ValueError(f"Zero angle must be non-negative, got {self.zero_angle}.")
        if not isinstance(self.rolloff_angle, int | float):
            raise TypeError(f"Rolloff angle must be a number, got {type(self.rolloff_angle).__name__}.")
        if self.rolloff_angle < 0:
            raise ValueError(f"Rolloff angle must be non-negative, got {self.rolloff_angle}.")
        if self.rolloff_angle >= self.zero_angle:
            raise ValueError(f"Rolloff angle must be less than zero angle, got {self.rolloff_angle} >= {self.zero_angle}.")
        if getunittype(self.units) != "angle":
            raise ValueError(f"Units must be an angle type, got {self.units}.")

    def calc_apodization(self, arr: Transducer, target: Point, params: xa.Dataset, transform:np.ndarray | None=None):
        target_pos = target.get_position(units="m")
        matrix = transform if transform is not None else np.eye(4)
        angles = np.array([el.angle_to_point(target_pos, units="m", matrix=matrix, return_as=self.units) for el in arr.elements])
        apod = np.zeros(arr.numelements())
        f = ((self.zero_angle - angles) / (self.zero_angle - self.rolloff_angle))
        apod = np.maximum(0, np.minimum(1, f))
        return apod

    def to_table(self) -> pd.DataFrame:
        """
        Get a table of the apodization method parameters

        :returns: Pandas DataFrame of the apodization method parameters
        """
        records = [
            {"Name": "Type", "Value": "Piecewise-Linear", "Unit": ""},
            {"Name": "Zero Angle", "Value": self.zero_angle, "Unit": self.units},
            {"Name": "Rolloff Angle", "Value": self.rolloff_angle, "Unit": self.units},
        ]
        return pd.DataFrame.from_records(records)
