from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pandas as pd
import xarray as xa

from openlifu.bf.delay_methods import DelayMethod
from openlifu.geo.point import Point
from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.xdc import Transducer


@dataclass
class Direct(DelayMethod):
    c0: Annotated[float, OpenLIFUFieldData("Speed of Sound (m/s)", "Speed of sound in the medium (m/s)")] = 1480.0
    """Speed of sound in the medium (m/s)"""

    def __post_init__(self):
        if not isinstance(self.c0, int | float):
            raise TypeError("Speed of sound must be a number")
        if self.c0 <= 0:
            raise ValueError("Speed of sound must be greater than 0")
        self.c0 = float(self.c0)

    def calc_delays(self, arr: Transducer, target: Point, params: xa.Dataset | None=None, transform:np.ndarray | None=None):
        if params is None:
            c = self.c0
        else:
            c = params['sound_speed'].attrs['ref_value']
        target_pos = target.get_position(units="m")
        matrix = transform if transform is not None else np.eye(4)
        dists = np.array([el.distance_to_point(target_pos, units="m", matrix=matrix) for el in arr.elements])
        tof = dists / c
        delays = max(tof) - tof
        return delays

    def to_table(self) -> pd.DataFrame:
        """
        Get a table of the delay method parameters

        :returns: Pandas DataFrame of the delay method parameters
        """
        records = [{"Name": "Type", "Value": "Direct", "Unit": ""},
                   {"Name": "Default Sound Speed", "Value": self.c0, "Unit": "m/s"}]
        return pd.DataFrame.from_records(records)
