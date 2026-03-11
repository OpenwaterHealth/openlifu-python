from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Annotated, Literal, Tuple

import numpy as np
import pandas as pd
import xarray as xa

from openlifu.geo import Point
from openlifu.seg import SegmentationMethod
from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.dict_conversion import DictMixin
from openlifu.util.units import getunitconversion, getunittype
from openlifu.xdc import Transducer

COORD_DIMS = ("x", "y", "z")
COORD_NAMES = ("Lateral", "Elevation", "Axial")

@dataclass
class SimSetup(DictMixin):

    spacing: Annotated[float, OpenLIFUFieldData("Spacing", "Simulation grid spacing")] = 1.0
    """Simulation grid spacing"""

    units: Annotated[str, OpenLIFUFieldData("Spatial units", "Units used for spatial measurements")] = "mm"
    """Units used for spatial measurements"""

    x_extent: Annotated[Tuple[float, float], OpenLIFUFieldData("X-extent", "Simulation grid extent along the first dimension")] = (-30., 30.)
    """Simulation grid extent along the first dimension"""

    y_extent: Annotated[Tuple[float, float], OpenLIFUFieldData("Y-extent", "Simulation grid extend along the second dimension")] = (-30., 30.)
    """Simulation grid extend along the second dimension"""

    z_extent: Annotated[Tuple[float, float], OpenLIFUFieldData("Z-extent", "Simulation grid extend along the third dimension")] = (-4., 60.)
    """Simulation grid extend along the third dimension"""

    dt: Annotated[float, OpenLIFUFieldData("Time step", "Simulation time step")] = 0.
    """Simulation time step"""

    t_end: Annotated[float, OpenLIFUFieldData("End time", """Simulation end time""")] = 0.
    """Simulation end time"""

    c0: Annotated[float, OpenLIFUFieldData("Speed of Sound (m/s)", "Reference speed of sound for converting distance to time")] = 1500.0
    """Reference speed of sound for converting distance to time"""

    cfl: Annotated[float, OpenLIFUFieldData("CFL number", "Courant-Friedrichs-Lewy number")] = 0.3
    """Courant-Friedrichs-Lewy number"""

    options: Annotated[dict[str, str], OpenLIFUFieldData("Simulation options", "Additional simulation options")] = field(default_factory=dict)
    """Additional simulation options"""

    def __post_init__(self):
        if len(self.x_extent) != 2:
            raise ValueError("x_extent must have length 2.")
        if self.x_extent[0] >= self.x_extent[1]:
            raise ValueError("x_extent must be in the form (min, max) with min < max.")
        if len(self.y_extent) != 2:
            raise ValueError("y_extent must have length 2.")
        if self.y_extent[0] >= self.y_extent[1]:
            raise ValueError("y_extent must be in the form (min, max) with min < max.")
        if len(self.z_extent) != 2:
            raise ValueError("z_extent must have length 2.")
        if self.z_extent[0] >= self.z_extent[1]:
            raise ValueError("z_extent must be in the form (min, max) with min < max.")
        if not isinstance(self.spacing, int | float):
            raise TypeError("spacing must be a number.")
        if self.spacing <= 0:
            raise ValueError("spacing must be a positive number.")
        if not isinstance(self.units, str):
            raise TypeError("units must be a string.")
        if getunittype(self.units) != 'distance':
            raise ValueError(f"units must be a length unit, got {self.units}.")
        if not isinstance(self.c0, int | float):
            raise TypeError("c0 must be a number.")
        if self.c0 <= 0:
            raise ValueError("c0 must be a positive number.")
        if not isinstance(self.cfl, int | float):
            raise TypeError("cfl must be a number.")
        if self.cfl <= 0:
            raise ValueError("cfl must be a positive number.")
        if not isinstance(self.dt, int | float):
            raise TypeError("dt must be a number.")
        if self.dt < 0:
            raise ValueError("dt must be a non-negative number.")
        if not isinstance(self.t_end, int | float):
            raise TypeError("t_end must be a number.")
        if self.t_end < 0:
            raise ValueError("t_end must be a non-negative number.")
        nx = np.diff(self.x_extent)/self.spacing
        x_extent = tuple(np.arange(2)*np.round(nx)*self.spacing + self.x_extent[0])
        if ((0.5-np.abs((nx % 1) - 0.5))/ np.round(nx)) > 1e-3:
            logging.warning(f"x_extent {self.x_extent} does not evenly divide by spacing ({self.spacing}). Rounding to {x_extent}.")
        self.x_extent = x_extent
        ny = np.diff(self.y_extent)/self.spacing
        y_extent = tuple(np.arange(2)*np.round(ny)*self.spacing + self.y_extent[0])
        if ((0.5-np.abs((ny % 1) - 0.5))/ np.round(ny)) > 1e-3:
            logging.warning(f"y_extent {self.y_extent} does not evenly divide by spacing ({self.spacing}). Rounding to {y_extent}.")
        self.y_extent = y_extent
        nz = np.diff(self.z_extent)/self.spacing
        z_extent = tuple(np.arange(2)*np.round(nz)*self.spacing + self.z_extent[0])
        if ((0.5-np.abs((nz % 1) - 0.5))/ np.round(nz)) > 1e-3:
            logging.warning(f"z_extent {self.z_extent} does not evenly divide by spacing ({self.spacing}). Rounding to {z_extent}.")
        self.z_extent = z_extent

    def get_coords(self, dims=None, units: str | None = None):
        dims = COORD_DIMS if dims is None else dims
        units = self.units if units is None else units
        sizes = self.get_size(dims)
        extents = self.get_extent(dims, units)
        coords = xa.Coordinates({dim: np.linspace(extents[i][0], extents[i][1], sizes[i]) for i, dim in enumerate(dims)})
        for dim in dims:
            coords[dim].attrs['units'] = units
            coords[dim].attrs['long_name'] = COORD_NAMES[COORD_DIMS.index(dim)]
        return coords

    def get_corners(self, units: str | None = None):
        units = self.units if units is None else units
        scl = getunitconversion(self.units, units)
        xyz = np.array(np.meshgrid(self.x_extent, self.y_extent, self.z_extent, indexing='ij'))
        corners = xyz.reshape(3,-1)
        return corners*scl

    def get_extent(self, dims: str | None=None, units: str | None = None):
        dims = COORD_DIMS if dims is None else dims
        units = self.units if units is None else units
        scl = getunitconversion(self.units, units)
        extents = [self.x_extent, self.y_extent, self.z_extent]
        return np.array([extents[COORD_DIMS.index(dim)] for dim in dims])*scl

    def get_max_cycle_offset(self, arr:Transducer, frequency: float | None = None, delays: np.ndarray | None=None, zmin: float =10e-3):
        frequency = arr.frequency if frequency is None else frequency
        delays = np.zeros(arr.numelements()) if delays is None else delays
        coords = self.get_coords(units="m")
        dims = coords.dims
        cvals = [coords[dims[0]], coords[dims[1]], coords[dims[2]].sel(ax=slice(zmin, None))]
        ndg = np.meshgrid(*cvals)
        dists = [np.sqrt((ndg[0]-pos[0])**2 + (ndg[1]-pos[1])**2 + (ndg[2]-pos[2])**2) for pos in arr.get_positions(units="m")]
        tof = [dist/self.c0 + delays[i] for i, dist in enumerate(dists)]
        dtof = np.array(tof).max(axis=0) - np.array(tof).min(axis=0)
        max_cycle_offset = dtof.max()*frequency
        return max_cycle_offset

    def get_max_distance(self, arr: Transducer, units: str | None = None):
        units = self.units if units is None else units
        corners = self.get_corners(units=units)
        distances = np.array([[el.distance_to_point(corner, units=units) for corner in corners.T] for el in arr.elements])
        max_distance = np.max(distances)
        return max_distance

    def get_size(self, dims: str | None=None):
        dims = COORD_DIMS if dims is None else dims
        n = [int((np.round(np.diff(ext)/self.spacing)).item())+1 for ext in [self.x_extent, self.y_extent, self.z_extent]]
        return np.array([n[COORD_DIMS.index(dim)] for dim in dims]).squeeze()

    def get_spacing(self, units: str | None = None):
        units = self.units if units is None else units
        return getunitconversion(self.units, units)*self.spacing

    def setup_sim_scene(
            self,
            seg_method: SegmentationMethod,
            volume: xa.DataArray | None = None
        ) -> Tuple[xa.DataArray, Transducer, Point]:
        """ Prepare a simulation scene composed of a simulation grid

        Setup a simulation scene with a simulation grid including physical properties.
        A segmentation is performed to detect the medium, so we can assign
        physical properties to each voxel, later used by the ultrasound simulation.
        This assume that the input volume is resampled to the geo-referenced simulation grid (lon, lat, ele).

        Args:
            seg_method: seg.SegmentationMethod
            volume: xa.DataArray
                Optional volume to be used for simulation grid definition (Default: None).
                The volume is assumed to be resampled on sim grid coordinates.

        Returns
            params: The xa.DataArray simulation grid with physical properties for each voxel
        """
        if volume is None:
            sim_coords = self.get_coords()
            params = seg_method.ref_params(sim_coords)
        else:
            params = seg_method.seg_params(volume)

        return params

    def to_table(self) -> pd.DataFrame:
        """
        Get a table of the simulation setup parameters

        :returns: Pandas DataFrame of the simulation setup parameters
        """
        records = [
            {"Name": "Spacing", "Value": self.spacing, "Unit": self.units},
            {"Name": "X Extent", "Value": f"{self.x_extent[0]} to {self.x_extent[1]}", "Unit": self.units},
            {"Name": "Y Extent", "Value": f"{self.y_extent[0]} to {self.y_extent[1]}", "Unit": self.units},
            {"Name": "Z Extent", "Value": f"{self.z_extent[0]} to {self.z_extent[1]}", "Unit": self.units},
            {"Name": "Time Step", "Value": self.dt, "Unit": "s"},
            {"Name": "End Time", "Value": self.t_end, "Unit": "s"},
            {"Name": "Speed of Sound", "Value": self.c0, "Unit": "m/s"},
            {"Name": "CFL", "Value": self.cfl, "Unit": ""},
        ]
        return pd.DataFrame.from_records(records)

    @staticmethod
    def from_dict(d: dict, on_keyword_mismatch: Literal['warn', 'raise', 'ignore'] = 'warn') -> SimSetup:
        """Create a SimSetup instance from a dictionary."""
        if not isinstance(d, dict):
            raise TypeError("Input must be a dictionary.")

        expected_keywords = [
            'spacing', 'units', 'x_extent', 'y_extent', 'z_extent',
            'dt', 't_end', 'c0', 'cfl', 'options'
        ]

        input_args = {
            k: v for k, v in d.items() if k in expected_keywords
        }
        unexpected_keywords = [k for k in d if k not in expected_keywords]

        if unexpected_keywords:
            if on_keyword_mismatch == 'raise':
                raise TypeError(f"Unexpected keyword arguments for SimSetup: {unexpected_keywords}")
            elif on_keyword_mismatch == 'warn':
                logging.warning(f"Ignoring unexpected keyword arguments for SimSetup: {unexpected_keywords}")

        return SimSetup(**input_args)
