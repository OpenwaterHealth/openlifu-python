from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pandas as pd
import xarray as xa

from openlifu.bf.apod_methods import ApodizationMethod
from openlifu.geo.point import Point
from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.field_display import summarize_fields
from openlifu.xdc import Transducer


@dataclass
class Uniform(ApodizationMethod):
    value: Annotated[float, OpenLIFUFieldData(
        name="Value",
        description="Uniform apodization value between 0 and 1.",
        precision=2,
    )] = 1.0
    """Uniform apodization value between 0 and 1."""

    def calc_apodization(self, arr: Transducer, target: Point, params: xa.Dataset, transform:np.ndarray | None=None):
        return np.full(arr.numelements(), self.value)

    def to_table(self):
        """
        Get a table of the apodization method parameters

        :returns: Pandas DataFrame of the apodization method parameters
        """
        records = [{"Name": "Type", "Value": "Uniform", "Unit": ""},
                   {"Name": "Value", "Value": self.value, "Unit": ""}]
        return pd.DataFrame.from_records(records)

    def get_summary(self) -> str:
        """Return a one-liner summary of the apodization parameters."""
        return summarize_fields(self, ("value",))
