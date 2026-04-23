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
class MaxAngle(ApodizationMethod):
    max_angle: Annotated[float, OpenLIFUFieldData("Maximum acceptance angle", "Maximum acceptance angle for each element from the vector normal to the element surface")] = 30.0
    """Maximum acceptance angle for each element from the vector normal to the element surface"""

    units: Annotated[str, OpenLIFUFieldData("Angle units", "Angle units")] = "deg"
    """Angle units"""

    def __post_init__(self):
        if not isinstance(self.max_angle, int | float):
            raise TypeError(f"Max angle must be a number, got {type(self.max_angle).__name__}.")
        if self.max_angle < 0:
            raise ValueError(f"Max angle must be non-negative, got {self.max_angle}.")
        if getunittype(self.units) != "angle":
            raise ValueError(f"Units must be an angle type, got {self.units}.")

    def calc_apodization(self, arr: Transducer, target: Point, params: xa.Dataset, transform:np.ndarray | None=None):
        target_pos = target.get_position(units="m")
        matrix = transform if transform is not None else np.eye(4)
        angles = np.array([el.angle_to_point(target_pos, units="m", matrix=matrix, return_as=self.units) for el in arr.elements])
        apod = np.zeros(arr.numelements())
        apod[angles <= self.max_angle] = 1
        return apod

    def to_table(self) -> pd.DataFrame:
        """
        Get a table of the apodization method parameters

        :returns: Pandas DataFrame of the apodization method parameters
        """
        records = [{"Name": "Type", "Value": "Max Angle", "Unit": ""},
                   {"Name": "Max Angle", "Value": self.max_angle, "Unit": self.units}]
        return pd.DataFrame.from_records(records)
