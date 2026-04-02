from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xa

from openlifu.bf import apod_methods
from openlifu.geo.point import Point
from openlifu.xdc import Transducer


@dataclass
class ApodizationMethod(ABC):
    @abstractmethod
    def calc_apodization(self, arr: Transducer, target: Point, params: xa.Dataset, transform:np.ndarray | None=None) -> Any:
        pass

    def to_dict(self):
        d = self.__dict__.copy()
        d['class'] = self.__class__.__name__
        return d

    @staticmethod
    def from_dict(d):
        d = d.copy()
        short_classname = d.pop("class")
        module_dict = apod_methods.__dict__
        class_constructor = module_dict[short_classname]
        return class_constructor(**d)

    @abstractmethod
    def to_table(self) -> pd.DataFrame:
        """
        Get a table of the apodization method parameters

        :returns: Pandas DataFrame of the apodization method parameters
        """
        pass
