from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, List

import numpy as np
import pandas as pd

from openlifu.bf.focal_patterns import FocalPattern
from openlifu.geo.point import Point
from openlifu.util.annotations import OpenLIFUFieldData


@dataclass
class Wheel(FocalPattern):
    """
    Class for representing a wheel pattern
    """

    center: Annotated[bool, OpenLIFUFieldData("Include center point?", "Whether to include the center for the wheel pattern")] = True
    """Whether to include the center for the wheel pattern"""

    num_spokes: Annotated[int | List[int], OpenLIFUFieldData(
        name="Number of spokes",
        description="Number of spokes in the wheel pattern",
    )] = 4
    """Number of spokes in the wheel pattern"""

    spoke_radius: Annotated[float | List[float], OpenLIFUFieldData(
        name="Spoke radius",
        description="Radius of the spokes in the wheel pattern",
        units_field="distance_units", display_units="mm", precision=1,
    )] = 1.0  # mm
    """Radius of the spokes in the wheel pattern"""

    spoke_phase: Annotated[float | List[float], OpenLIFUFieldData(
        name="Spoke phase",
        description="Phase of the spokes in the wheel pattern",
        units_field="angle_units", display_units="deg", precision=1,
    )] = 0.0  # degrees
    """Phase of the spokes in the wheel pattern"""

    distance_units: Annotated[str, OpenLIFUFieldData(
        name="Distance units",
        description="Units of the wheel pattern parameters",
        unit_options=("mm", "cm", "m"),
    )] = "mm"
    """Units of the wheel pattern parameters"""

    order: Annotated[list[int] | None, OpenLIFUFieldData("Focus order", "Order of Foci (1-indexed) in the sequence")] = None
    """Order of Foci (1-indexed) in the sequence. This is a list of integers that specifies the order in which the foci are used in the pulse sequence. If None, the foci are used in the order they are listed in the `foci` attribute."""

    def __post_init__(self):
        if not isinstance(self.center, bool):
            raise TypeError(f"Center must be a boolean, got {type(self.center).__name__}.")
        if isinstance(self.num_spokes, int):
            if self.num_spokes < 1:
                raise ValueError(f"Number of spokes must be a positive integer, got {self.num_spokes}.")
        elif isinstance(self.num_spokes, list):
            if not all(isinstance(n, int) and n > 0 for n in self.num_spokes):
                raise ValueError(f"All elements of num_spokes must be positive integers, got {self.num_spokes}.")
        else:
            raise TypeError(f"num_spokes must be an int or list of ints, got {type(self.num_spokes).__name__}.")

        if isinstance(self.spoke_radius, int | float):
            if self.spoke_radius <= 0:
                raise ValueError(f"Spoke radius must be a positive number, got {self.spoke_radius}.")
        elif isinstance(self.spoke_radius, list):
            if not all(isinstance(r, int | float) and r > 0 for r in self.spoke_radius):
                raise ValueError(f"All elements of spoke_radius must be positive numbers, got {self.spoke_radius}.")
            if isinstance(self.spoke_phase, list) and len(self.spoke_phase) != len(self.spoke_radius):
                raise ValueError(f"Length of spoke_phase list must match length of spoke_radius list, got {len(self.spoke_phase)} and {len(self.spoke_radius)}.")
            if isinstance(self.num_spokes, list) and len(self.spoke_radius) != len(self.num_spokes):
                raise ValueError(f"Length of spoke_radius list must match length of num_spokes list, got {len(self.spoke_radius)} and {len(self.num_spokes)}.")
        else:
            raise TypeError(f"spoke_radius must be a number or list of numbers, got {type(self.spoke_radius).__name__}.")
        super().__post_init__()

    def get_targets(self, target: Point):
        """
        Get the targets of the focal pattern

        :param target: Target point of the focal pattern
        :returns: List of target points
        """
        m = target.get_matrix(center_on_point=True)
        if isinstance(self.num_spokes, int):
            if isinstance(self.spoke_radius, int | float):
                spoke_radius_list = [self.spoke_radius]
                spoke_phase_list = [self.spoke_phase]
            else:
                spoke_radius_list = self.spoke_radius
                spoke_phase_list = self.spoke_phase if isinstance(self.spoke_phase, list) else [self.spoke_phase]*len(spoke_radius_list)
            num_spokes_list = [self.num_spokes]*len(spoke_radius_list)
        else:
            num_spokes_list = self.num_spokes
            spoke_radius_list = self.spoke_radius
            spoke_phase_list = self.spoke_phase if isinstance(self.spoke_phase, list) else [self.spoke_phase]*len(spoke_radius_list)

        n_points = sum(num_spokes_list) + int(self.center)

        if self.center:
            targets = [target.copy()]
            targets[0].id = f"{target.id}_01"
            targets[0].id = f"{target.id} (1/{n_points}, Center)"
        else:
            targets = []

        for (num_spokes, spoke_radius, spoke_phase) in zip(num_spokes_list, spoke_radius_list, spoke_phase_list):
            for j in range(num_spokes):
                point_index = len(targets) + 1
                theta = 2*np.pi*j/num_spokes + np.deg2rad(spoke_phase)
                local_position = spoke_radius * np.array([np.cos(theta), np.sin(theta), 0.0])
                position = np.dot(m, np.append(local_position, 1.0))[:3]
                spoke = Point(id=f"{target.id}_{point_index:02d}",
                              name=f"{target.name} ({point_index}/{n_points}, {spoke_radius:.1f} mm, {np.rad2deg(theta):.0f}°)",
                              position=position,
                              units=self.distance_units,
                              radius=target.radius)
                targets.append(spoke)
        return targets

    def num_foci(self) -> int:
        """
        Get the number of foci in the focal pattern

        :returns: Number of foci
        """
        return int(self.center) + self.num_spokes

    def get_order(self):
        """
        Get the order of foci in the focal pattern

        :returns: List of indices of foci in the order they are used in the pulse sequence
        """
        if self.order is not None:
            return self.order
        else:
            return list(range(1, self.num_foci() + 1))

    def to_table(self) -> pd.DataFrame:
        """
        Get a table of the focal pattern parameters

        :returns: Pandas DataFrame of the focal pattern parameters
        """
        records = [
            {"Name": "Type", "Value": "Wheel", "Unit": ""},
            {"Name": "Target Pressure", "Value": self.target_pressure, "Unit": self.units},
            {"Name": "Center", "Value": self.center, "Unit": ""},
            {"Name": "Number of Spokes", "Value": self.num_spokes, "Unit": ""},
            {"Name": "Spoke Radius", "Value": self.spoke_radius, "Unit": self.distance_units},
        ]
        return pd.DataFrame.from_records(records)
