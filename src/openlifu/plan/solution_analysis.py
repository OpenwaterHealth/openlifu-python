from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, Tuple, Type

import numpy as np
import pandas as pd
import xarray as xa

from openlifu.plan.param_constraint import PARAM_STATUS_SYMBOLS, ParameterConstraint
from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.dict_conversion import DictMixin
from openlifu.util.units import getunitconversion, getunittype

DEFAULT_ORIGIN = np.zeros(3)

PARAM_FORMATS = {
    "mainlobe_pnp_MPa": ["max", "0.3f", "MPa", "Mainlobe Peak Negative Pressure"],
    "mainlobe_isppa_Wcm2": ["max", "0.1f", "W/cm^2", "Mainlobe I_SPPA"],
    "mainlobe_ispta_mWcm2": ["mean", "0.1f", "mW/cm^2", "Mainlobe I_SPTA"],
    "target_position_lat_mm": ["mean", "0.1f", "mm", "Target Position (Lateral)"],
    "target_position_ele_mm": ["mean", "0.1f", "mm", "Target Position (Elevation)"],
    "target_position_ax_mm": ["mean", "0.1f", "mm", "Target Position (Axial)"],
    "focal_centroid_lat_mm": ["mean", "0.1f", "mm", "Focal Centroid (Lateral)"],
    "focal_centroid_ele_mm": ["mean", "0.1f", "mm", "Focal Centroid (Elevation)"],
    "focal_centroid_ax_mm": ["mean", "0.1f", "mm", "Focal Centroid (Axial)"],
    "beamwidth_lat_3dB_mm": ["mean", "0.2f", "mm", "3dB Beamwidth (Lateral)"],
    "beamwidth_ele_3dB_mm": ["mean", "0.2f", "mm", "3dB Beamwidth (Elevational)"],
    "beamwidth_ax_3dB_mm": ["mean", "0.2f", "mm", "3dB Beamwidth (Axial)"],
    "beamwidth_lat_6dB_mm": ["mean", "0.2f", "mm", "6dB Beamwidth (Lateral)"],
    "beamwidth_ele_6dB_mm": ["mean", "0.2f", "mm", "6dB Beamwidth (Elevational)"],
    "beamwidth_ax_6dB_mm": ["mean", "0.2f", "mm", "6dB Beamwidth (Axial)"],
    "sidelobe_pnp_MPa": ["max", "0.3f", "MPa", "Sidelobe Peak Negative Pressure"],
    "sidelobe_isppa_Wcm2": ["max", "0.1f", "W/cm^2", "Sidelobe I_SPPA"],
    "sidelobe_to_mainlobe_pressure_ratio": ["mean", "0.2f", "", "Sidelobe/Mainlobe Pressure Ratio"],
    "sidelobe_to_mainlobe_intensity_ratio": ["mean", "0.2f", "", "Sidelobe/Mainlobe Intensity Ratio"],
    "global_pnp_MPa": ["max", "0.3f", "MPa", "Global Peak Negative Pressure"],
    "global_isppa_Wcm2": ["max", "0.1f", "W/cm^2", "Global I_SPPA"],
    "global_ispta_mWcm2": [None, "0.1f", "mW/cm^2", "Global I_SPTA"],
    "MI": [None, "0.2f", "", "MI"],
    "TIC": [None, "0.2f", "", "TIC"],
    "voltage_V": [None, "0.1f", "V", "Voltage"],
    "p0_MPa": ["max", "0.3f", "MPa", "Emitted Pressure"],
    "focal_gain": ["max", "0.2f", "", "Focal Gain"],
    "power_W": [None, "0.2f", "W", "Emitted Power"],
    "duty_cycle_pulse_train_pct": [None, "0.1f", "%", "Pulse Train Duty Cycle"],
    "duty_cycle_sequence_pct": [None, "0.1f", "%", "Sequence Duty Cycle"],
    "sequence_duration_s": [None, "0.0f", "s", "Sequence Duration"],}

@dataclass
class SolutionAnalysis(DictMixin):
    mainlobe_pnp_MPa: Annotated[list[float], OpenLIFUFieldData("Mainlobe PNP", "Peak negative pressure in the mainlobe, in MPa")] = field(default_factory=list)
    """Peak negative pressure in the mainlobe, in MPa"""

    mainlobe_isppa_Wcm2: Annotated[list[float], OpenLIFUFieldData("Mainlobe ISPPA", "Spatial peak pulse average intensity in the mainlobe, in W/cm²")] = field(default_factory=list)
    """Spatial peak pulse average intensity in the mainlobe, in W/cm²"""

    mainlobe_ispta_mWcm2: Annotated[list[float], OpenLIFUFieldData("Mainlobe ISPTA", "Spatial peak time average intensity in the mainlobe, in mW/cm²")] = field(default_factory=list)
    """Spatial peak time average intensity in the mainlobe, in mW/cm²"""

    target_position_lat_mm: Annotated[list[float], OpenLIFUFieldData("Target position lateral (mm)", "Lateral position of the target, in mm")] = field(default_factory=list)
    """Lateral position of the target, in mm"""

    target_position_ele_mm: Annotated[list[float], OpenLIFUFieldData("Target position elevation (mm)", "Elevation position of the target, in mm")] = field(default_factory=list)
    """Elevation position of the target, in mm"""

    target_position_ax_mm: Annotated[list[float], OpenLIFUFieldData("Target position axial (mm)", "Axial position of the target, in mm")] = field(default_factory=list)
    """Axial position of the target, in mm"""

    focal_centroid_lat_mm: Annotated[list[float], OpenLIFUFieldData("Focal centroid lateral (mm)", "Lateral centroid of the focus, in mm")] = field(default_factory=list)
    """Lateral centroid of the focus, in mm"""

    focal_centroid_ele_mm: Annotated[list[float], OpenLIFUFieldData("Focal centroid elevation (mm)", "Elevation centroid of the focus, in mm")] = field(default_factory=list)
    """Elevation centroid of the focus, in mm"""

    focal_centroid_ax_mm: Annotated[list[float], OpenLIFUFieldData("Focal centroid axial (mm)", "Axial centroid of the focus, in mm")] = field(default_factory=list)
    """Axial centroid of the focus, in mm"""

    beamwidth_lat_3dB_mm: Annotated[list[float], OpenLIFUFieldData("3dB lateral beamwidth", "Lateral beamwidth at -3 dB, in mm")] = field(default_factory=list)
    """Lateral beamwidth at -3 dB, in mm"""

    beamwidth_ele_3dB_mm: Annotated[list[float], OpenLIFUFieldData("3dB elevation beamwidth", "Elevation beamwidth at -3 dB, in mm")] = field(default_factory=list)
    """Elevation beamwidth at -3 dB, in mm"""

    beamwidth_ax_3dB_mm: Annotated[list[float], OpenLIFUFieldData("3dB axial beamwidth", "Axial beamwidth at -3 dB, in mm")] = field(default_factory=list)
    """Axial beamwidth at -3 dB, in mm"""

    beamwidth_lat_6dB_mm: Annotated[list[float], OpenLIFUFieldData("6dB lateral beamwidth", "Lateral beamwidth at -6 dB, in mm")] = field(default_factory=list)
    """Lateral beamwidth at -6 dB, in mm"""

    beamwidth_ele_6dB_mm: Annotated[list[float], OpenLIFUFieldData("6dB elevation beamwidth", "Elevation beamwidth at -6 dB, in mm")] = field(default_factory=list)
    """Elevation beamwidth at -6 dB, in mm"""

    beamwidth_ax_6dB_mm: Annotated[list[float], OpenLIFUFieldData("6dB axial beamwidth", "Axial beamwidth at -6 dB, in mm")] = field(default_factory=list)
    """Axial beamwidth at -6 dB, in mm"""

    sidelobe_pnp_MPa: Annotated[list[float], OpenLIFUFieldData("Sidelobe PNP", "Peak negative pressure in the sidelobes, in MPa")] = field(default_factory=list)
    """Peak negative pressure in the sidelobes, in MPa"""

    sidelobe_isppa_Wcm2: Annotated[list[float], OpenLIFUFieldData("Sidelobe ISPPA", "Spatial peak pulse average intensity in the sidelobes, in W/cm²")] = field(default_factory=list)
    """Spatial peak pulse average intensity in the sidelobes, in W/cm²"""

    sidelobe_to_mainlobe_pressure_ratio: Annotated[list[float], OpenLIFUFieldData("Sidelobe/Mainlobe Pressure Ratio", "Ratio of sidelobe peak negative pressure to mainlobe peak negative pressure")] = field(default_factory=list)
    """Ratio of sidelobe peak negative pressure to mainlobe peak negative pressure"""

    sidelobe_to_mainlobe_intensity_ratio: Annotated[list[float], OpenLIFUFieldData("Sidelobe/Mainlobe Intensity Ratio", "Ratio of sidelobe spatial peak pulse average intensity to mainlobe spatial peak pulse average intensity")] = field(default_factory=list)
    """Ratio of sidelobe spatial peak pulse average intensity to mainlobe spatial peak pulse average intensity"""

    global_pnp_MPa: Annotated[list[float], OpenLIFUFieldData("Global PNP", "Maximum peak negative pressure in the entire field, in MPa")] = field(default_factory=list)
    """Maximum peak negative pressure in the entire field, in MPa"""

    global_isppa_Wcm2: Annotated[list[float], OpenLIFUFieldData("Global ISPPA", "Maximum spatial peak pulse average intensity in the entire field, in W/cm²")] = field(default_factory=list)
    """Maximum spatial peak pulse average intensity in the entire field, in W/cm²"""

    global_ispta_mWcm2: Annotated[float | None, OpenLIFUFieldData("Global ISPTA (mW/cm²)", "Global Intensity at Spatial-Peak, Time-Average (I_SPTA) (mW/cm²)")] = None
    """Global Intensity at Spatial-Peak, Time-Average (I_SPTA) (mW/cm²)"""

    MI: Annotated[float | None, OpenLIFUFieldData("Mechanical index (MI)", "Mechanical index (MI)")] = None
    """Mechanical index (MI)"""

    TIC: Annotated[float | None, OpenLIFUFieldData("Thermal index (TIC)", "Thermal index in cranium (TIC)")] = None
    """Thermal index in cranium (TIC)"""

    voltage_V: Annotated[float | None, OpenLIFUFieldData("Voltage (V)", "Voltage applied to the transducer (V)")] = None
    """Voltage applied to the transducer (V)"""

    p0_MPa: Annotated[list[float], OpenLIFUFieldData("Emitted pressure (MPa)", "Initial pressure values in the field, in MPa")] = field(default_factory=list)
    """Initial pressure values in the field (MPa)"""

    focal_gain: Annotated[list[float], OpenLIFUFieldData("Focal gain", "Ratio of mainlobe peak pressure to emitted pressure")] = field(default_factory=list)
    """Ratio of mainlobe peak pressure to emitted pressure"""

    power_W: Annotated[float | None, OpenLIFUFieldData("Emitted Power (W)", "Emitted power from the transducer face (W)")] = None
    """Emitted power from the transducer face (W)"""

    duty_cycle_pulse_train_pct: Annotated[float | None, OpenLIFUFieldData("Duty cycle pulse train", "Duty cycle within a pulse train (0-1)")] = None
    """Duty cycle within a pulse train (0-1)"""

    duty_cycle_sequence_pct: Annotated[float | None, OpenLIFUFieldData("Duty cycle sequence", "Duty cycle of the overall sequence (0-1)")] = None
    """Duty cycle of the overall sequence (0-1)"""

    sequence_duration_s: Annotated[float | None, OpenLIFUFieldData("Sequence duration (s)", "Total duration of the sequence (s)")] = None
    """Total duration of the sequence (s)"""


    param_constraints: Annotated[Dict[str, ParameterConstraint], OpenLIFUFieldData("Parameter constraints", None)] = field(default_factory=dict)
    """TODO: Add description"""

    def to_table(self, constraints:Dict[str,ParameterConstraint]|None=None, focus_index=None) -> pd.DataFrame:
        records = []
        if constraints is None:
            constraints = self.param_constraints
        for p in constraints:
            if p not in PARAM_FORMATS:
                raise ValueError(f"Unknown parameter constraint for '{p}'. Must be one of: {list(PARAM_FORMATS.keys())}")
        for param, fmt in PARAM_FORMATS.items():
            value_by_focus = None
            agg_value = None
            if fmt[0] is None:
                value_by_focus = None
                agg_value = self.__dict__[param]
            elif fmt[0] == "max":
                value_by_focus = self.__dict__[param]
                agg_value = max(value_by_focus)
            elif fmt[0] == "mean":
                value_by_focus = self.__dict__[param]
                agg_value = np.mean(value_by_focus)
            else:
                raise ValueError(f"Unknown aggregation method '{fmt[0]}' for parameter '{param}'.")
            if agg_value is not None:
                record = {"id": param,
                          "Param": fmt[3],
                          "Value" : "",
                          "Units" : fmt[2],
                          "Status": "",
                          "_value" : agg_value,
                          "_value_by_focus": value_by_focus,
                          "_warning": False,
                          "_error": False}
                if np.isnan(agg_value):
                    record["Value"] = "NaN"
                else:
                    if focus_index is None:
                        record["Value"] = f"{agg_value:{fmt[1]}}"
                    elif value_by_focus is None:
                        record["Value"] = "N/A"
                    else:
                        value = value_by_focus[focus_index]
                        record["Value"] = f"{value:{fmt[1]}}"
                    if param in constraints:
                        record['_warning'] = constraints[param].is_warning(agg_value)
                        record['_error'] = constraints[param].is_error(agg_value)
                        record["Status"] = PARAM_STATUS_SYMBOLS[constraints[param].get_status(agg_value)]
                records.append(record)
        return pd.DataFrame.from_records(records)

    @classmethod
    def from_dict(cls: Type[SolutionAnalysis], parameter_dict: Dict[str, Any]) -> SolutionAnalysis:
        """
        Create an object from a dictionary

        Args:
            parameter_dict: dictionary of parameters to define the object
        Returns: new object
        """
        parameter_dict["param_constraints"] = {
            k: ParameterConstraint.from_dict(v)
            for k, v in parameter_dict.get("param_constraints", {}).items()
        }

        return cls(**parameter_dict)

    @staticmethod
    def from_json(json_string : str) -> SolutionAnalysis:
        """Load a SolutionAnalysis from a json string"""
        return SolutionAnalysis.from_dict(json.loads(json_string))

    def to_json(self, compact:bool) -> str:
        """Serialize a SolutionAnalysis to a json string

        Args:
            compact: if enabled then the string is compact (not pretty). Disable for pretty.

        Returns: A json string representing the complete SolutionAnalysis object.
        """
        if compact:
            return json.dumps(self.to_dict(), separators=(',', ':'))
        else:
            return json.dumps(self.to_dict(), indent=4)

@dataclass
class SolutionAnalysisOptions(DictMixin):
    standoff_sound_speed: Annotated[float, OpenLIFUFieldData("Standoff sound speed (m/s)", "Speed of sound in standoff, for calculating initial impedance")] = 1500.0
    """Speed of sound in standoff, for calculating initial impedance"""

    standoff_density: Annotated[float, OpenLIFUFieldData("Standoff density (kg/m³)", "Density of standoff medium (kg/m³)")] = 1000.0
    """Density of standoff medium (kg/m³)"""

    ref_sound_speed: Annotated[float, OpenLIFUFieldData("Reference sound speed (m/s)", "Reference speed of sound in the medium (m/s)")] = 1500.0
    """Reference speed of sound in the medium (m/s)"""

    ref_density: Annotated[float, OpenLIFUFieldData("Reference density (kg/m³)", "Reference density (kg/m³)")] = 1000.0
    """Reference density (kg/m³)"""

    mainlobe_aspect_ratio: Annotated[Tuple[float, float, float], OpenLIFUFieldData("Mainlobe aspect ratio (lat,ele,ax)", "Aspect ratio of the mainlobe mask")] = (1., 1., 5.)
    """Aspect ratio of the mainlobe ellipsoid mask, in the form (lat,ele,ax). (1,1,5) means an ellipsoid 5x as long as it is wide."""

    mainlobe_radius: Annotated[float, OpenLIFUFieldData("Mainlobe mask radius", "Size of the mainlobe mask, in the units provided for Distance units (`distance_units`)")] = 2.5e-3
    """Size of the mainlobe mask, in the units provided for Distance units (`distance_units`). The mainlobe mask is an ellipsoid with this radius, scaled by the `mainlobe_aspect_ratio`."""

    beamwidth_radius: Annotated[float, OpenLIFUFieldData("Beamwidth search radius", "Size of the beamwidth search, in the units provided for Distance units (`distance_units`)")] = 5e-3
    """Size of the beamwidth search, in the units provided for Distance units (`distance_units`). The beamwidth is found along the lateral and elevation lines perpendicular to the focus axis."""

    sidelobe_radius: Annotated[float, OpenLIFUFieldData("Sidelobe radius", "Size of the sidelobe mask, in the units provided for Distance units (`distance_units`)")] = 3e-3
    """Size of the sidelobe mask, in the units provided for Distance units (`distance_units`). Pressure outside of this ellipsoid (scaled by `mainlobe_aspect_ratio`) is considered outside of the focal region."""

    sidelobe_zmin: Annotated[float, OpenLIFUFieldData("Sidelobe minimum z", "Minimum z coordinate of the sidelobe mask, in the units provided for Distance units (`distance_units`)")] = 1e-3
    """Minimum z coordinate of the sidelobe mask, in the units provided for Distance units (`distance_units`). This value is used to ignore emitted pressure artifacts."""

    distance_units: Annotated[str, OpenLIFUFieldData("Distance units", "The units used for distance measurements")] = "m"
    """The units used for distance measurements"""

    param_constraints: Annotated[Dict[str, ParameterConstraint], OpenLIFUFieldData("Parameter constraints", None)] = field(default_factory=dict)
    """TODO: Add description"""

    def __post_init__(self):
        if self.standoff_sound_speed <= 0:
            raise ValueError("Standoff sound speed must be greater than 0")
        if self.standoff_density <= 0:
            raise ValueError("Standoff density must be greater than 0")
        if self.ref_sound_speed <= 0:
            raise ValueError("Reference sound speed must be greater than 0")
        if self.ref_density <= 0:
            raise ValueError("Reference density must be greater than 0")
        if not isinstance(self.mainlobe_aspect_ratio, tuple | list) or len(self.mainlobe_aspect_ratio) != 3:
            raise TypeError("Mainlobe aspect ratio must be a tuple or list of three floats (lat, ele, ax)")
        self.mainlobe_aspect_ratio = tuple(self.mainlobe_aspect_ratio)  # Ensure it's a tuple
        if not all(isinstance(x, int | float) for x in self.mainlobe_aspect_ratio):
            raise TypeError("Mainlobe aspect ratio must contain only numbers")
        if not isinstance(self.mainlobe_radius, int | float) or self.mainlobe_radius <= 0:
            raise ValueError("Mainlobe radius must be a positive number")
        if not isinstance(self.beamwidth_radius, int | float) or self.beamwidth_radius <= 0:
            raise ValueError("Beamwidth radius must be a positive number")
        if not isinstance(self.sidelobe_radius, int | float) or self.sidelobe_radius <= 0:
            raise ValueError("Sidelobe radius must be a positive number")
        if not isinstance(self.sidelobe_zmin, int | float) or self.sidelobe_zmin < 0:
            raise ValueError("Sidelobe minimum z must be a non-negative number")
        if not isinstance(self.distance_units, str):
            raise TypeError("Distance units must be a string")
        if getunittype(self.distance_units) != 'distance':
            raise ValueError(f"Distance units must be a length unit, got {self.distance_units}")

    @classmethod
    def from_dict(cls: Type[SolutionAnalysisOptions], parameter_dict: Dict[str, Any]) -> SolutionAnalysisOptions:
        """
        Create an object from a dictionary

        Args:
            parameter_dict: dictionary of parameters to define the object
        Returns: new object
        """
        parameter_dict["param_constraints"] = {
            k: ParameterConstraint.from_dict(v)
            for k, v in parameter_dict.get("param_constraints", {}).items()
        }

        return cls(**parameter_dict)

def find_centroid(da: xa.DataArray, cutoff:float, units:None) -> np.ndarray:
    """Find the centroid of a thresholded region of a DataArray"""
    if units is not None and getunittype(units) != 'distance':
        raise ValueError(f"Units must be a length unit, got {units}")
    da = da.where(da > cutoff, 0)
    dims = list(da.dims)
    coords = np.meshgrid(*[da.coords[coord] for coord in dims], indexing='ij')
    centroid = np.array([np.sum(da*coords[dims.index(dim)])/da.sum() for dim in da.dims])
    if units is not None:
        da_units = [da.coords[dim].attrs.get('units', None) for dim in dims]
        centroid = np.array([getunitconversion(coord_units, units) * c for coord_units, c in zip(da_units, centroid)])
    return centroid

def get_focus_matrix(focus, origin=[0,0,0]) -> np.ndarray:
    """Get the coordinate transform from the focus to the transducer

    The "focal coordinate system" here refers to a coordinate system whose z-axis is along
    the ray from the transducer's "effective origin" (see `Transducer.get_effective_origin`)
    to the focus center.

    Args:
        focus: A 3D point describing the focal coordinate system location in transducer coordinates
        origin: A 3D point describing the transducer "effective origin" in transducer coordinates

    Returns: A 4x4 affine transform matrix that describes the coordinate transformation from the focus
        coordinate system to the transducer coordinates.
    """
    focus = np.array(focus)
    origin = np.array(origin)
    zvec = (focus-origin)/np.linalg.norm(focus-origin)
    az = -np.arctan2(zvec[0],zvec[2])
    xvec = np.array([np.cos(az), 0, np.sin(az)])
    yvec = np.cross(zvec, xvec)
    uv = np.array([xvec, yvec, zvec, focus])
    M = np.concatenate([uv.T,np.zeros([1,4])],axis=0)
    M[3,3] = 1
    return M

def get_gridded_transformed_coords(da: xa.DataArray, matrix: np.ndarray, as_dataset=True):
    """Transform the coords of a DataArray using a transform matrix.

    Args:
        da: DataArray whose coordinates will be used
        matrix: a 4x4 coordinate transformation matrix, transforming from the desired coordinate system
            to the coordinate system of `da`
        as_dataset: Whether to return the transformed coords as a numpy array or an xarray Dataset

    Returns the transformed coordinate grid as either a numpy array or an xarray Dataset.
    """
    shape = tuple(da.sizes[d] for d in da.dims)
    coords = np.meshgrid(*[da.coords[coord] for coord in da.dims], indexing='ij')
    coords = np.concatenate([coord.reshape(-1,1) for coord in coords], axis=1)
    coords = np.concatenate([coords, np.ones((coords.shape[0],1))], axis=1)
    coords = np.dot(np.linalg.inv(matrix), coords.T).T
    coords = coords[:,0:3].reshape(*shape, 3)
    if as_dataset:
        coords = xa.Dataset({f'd_{dim}': (da.dims, coords[...,i]) for i, dim in enumerate(da.dims)}, coords=da.coords)
    return coords

def get_offset_grid(da: xa.DataArray, focus, origin=DEFAULT_ORIGIN, as_dataset=True):
    """Transform the coords of a DataArray that is in transducer coordinates to focus coordinates

    See `get_focus_matrix` for the meaning of "focus coordinates"

    Args:
        da: DataArray whose coordinates will be used (presumably the transducer coordinates)
        focus: A 3D point describing the focus location in the coordinates of `da`
        origin: A 3D point describing the "effective origin" in the coordinates of `da`
            (see `Transducer.get_effective_origin` for the meaning of this).
        as_dataset: Whether to return the transformed coords as a numpy array or an xarray Dataset

    Returns the transformed coordinate grid as either a numpy array or an xarray Dataset.
    """
    M = get_focus_matrix(focus, origin=origin)
    #M = focus.get_matrix()
    coords = get_gridded_transformed_coords(da, M, as_dataset=as_dataset)
    return coords

def calc_dist_from_focus(da: xa.DataArray, focus, origin=DEFAULT_ORIGIN, aspect_ratio=[1,1,1], as_dataarray=True):
    """Compute a distance map from a focus point in transducer space, using a possibly distorted metric that respects
    the symmetry of the focus shape (e.g. it could be cigar-shaped).

    Args:
        da: DataArray that will supply the coordnate grid (presumably transducer coordinates)
        focus: A 3D point describing the focus location in the coordinates of `da`
        origin: A 3D point describing the "effective origin" in the coordinates of `da`
            (see `Transducer.get_effective_origin` for the meaning of this).
        aspect_ratio: x,y,z scalings on the focal coordinate system to distort the space before computing distance
            (see `get_focus_matrix` for the meaning of "focus coordinates").
        as_dataarray: Whether to return the distance map as a numpy array or an xarray DataArray

    Returns the distance map as either a numpy array or an xarray DataArray.
    """
    coords = get_offset_grid(da, focus, origin=origin, as_dataset=False)
    dist = np.sqrt(np.sum((coords/aspect_ratio)**2, axis=-1))
    if as_dataarray:
        dist = xa.DataArray(dist, coords=da.coords, dims=da.dims)
    return dist

def get_mask(
    da: xa.DataArray,
    focus,
    distance:float,
    origin=DEFAULT_ORIGIN,
    aspect_ratio=[1,1,1],
    operator='<',
) -> xa.DataArray:
    """Compute a boolean mask of the focus region in transducer space.

    The focus region is an ellipsoid centered at the focus point.

    Args:
        da: DataArray that will supply the coordnate grid (presumably transducer coordinates)
        focus: A 3D point describing the focus location in the coordinates of `da`
        distance: How far from the `focus` to include points in the mask. See `calc_dist_from_focus`
            for the distorted metric under which a ball of points becomes an ellispoid in euclidean space.
        origin: A 3D point describing the "effective origin" in the coordinates of `da`
            (see `Transducer.get_effective_origin` for the meaning of this).
        aspect_ratio: x,y,z scalings on the focal coordinate system to distort the space before computing distances
            (see `get_focus_matrix` for the meaning of "focus coordinates").
        operator: a string representation of an inequality operator that represents the desired masking operation.
            The default '<' for example includes points strictly *inside* the focal ellipsoid.

    Returns: the focal mask as an xarray DataArray
    """
    dist = calc_dist_from_focus(da, focus, origin=origin, aspect_ratio=aspect_ratio)
    if operator == '<':
        mask = dist < distance
    elif operator == '<=':
        mask = dist <= distance
    elif operator == '>':
        mask = dist > distance
    elif operator == '>=':
        mask = dist >= distance
    else:
        raise ValueError("Operator must be '<', '>', '<=', or '>='.")
    return mask

def interp_transformed_axis(
    da: xa.DataArray,
    focus,
    dim,
    origin=DEFAULT_ORIGIN,
    min_offset:float | None=None,
    max_offset:float | None=None,
) -> xa.DataArray:
    """Interpolate data along a focal coordinate system axis.

    See `get_focus_matrix` for the meaning of "focal coordinate system."

    Args:
        da: DataArray whose values will be sampled (presumably defined on transducer coordinates).
        focus: A 3D point describing the focus location in the coordinates of `da`
        dim: The name of the dimension of `da` whose corresponding focal coordinate system axis should be sampled along.
            For example, the "axial" dimension of a transducer corresponds to the focal axis (z-axis) in the focal coordinate
            system, that is, the ray from the transducer's "effective origin" (see `Transducer.get_effective_origin`)
            to the focus center.
        origin: A 3D point describing the "effective origin" in the coordinates of `da`
            (see `Transducer.get_effective_origin` for the meaning of this).
        min_offset: How far along the negative focal `dim` direction to sample. By default samples as far as the coordinate
            grid allows.
        max_offset: How far along the positive focal `dim` direction to sample. By default samples as far as the coordinate
            grid allows.

    Returns: a 1D DataArray of interpolated values from `da`.
    """
    matrix = get_focus_matrix(focus, origin=origin)
    coords = get_gridded_transformed_coords(da, matrix, as_dataset=True)
    if min_offset is None:
        min_offset = float(coords[f'd_{dim}'].min())
    if max_offset is None:
        max_offset = float(coords[f'd_{dim}'].max())
    n = da.sizes[dim]*2
    interp_dim = np.linspace(min_offset, max_offset, n)
    interp_xyz = np.zeros((n,3))
    interp_xyz[:,da.dims.index(dim)] = interp_dim
    interp_xyz = np.dot(matrix, np.concatenate([interp_xyz.T, np.ones((1,n))], axis=0)).T[:,0:3]
    interpolants = {d: xa.DataArray(interp_xyz[:,i], coords={f'offset_d{dim}':interp_dim}) for i, d in enumerate(da.dims)}
    interp_da = da.interp(**interpolants)
    for d in da.dims:
        interp_da.assign_coords(dim=(f'offset_d{dim}', interpolants[d].to_numpy()))
    return interp_da

def get_beam_bounds(
    da: xa.DataArray,
    focus,
    dim,
    cutoff:float,
    origin=DEFAULT_ORIGIN,
    min_offset:float | None=None,
    max_offset:float | None=None,
) -> Tuple[float, float]:
    """Determine how far along a focal coordinate system axis a DataArray's value stays above a certain cutoff.

    See `get_focus_matrix` for the meaning of "focal coordinate system."

    Args:
        da: DataArray whose values will be considered (presumably defined on transducer coordinates).
        focus: A 3D point describing the focus location in the coordinates of `da`
        dim: The name of the dimension of `da` whose corresponding focal coordinate system axis should be sampled along.
            For example, the "axial" dimension of a transducer corresponds to the focal axis (z-axis) in the focal coordinate
            system, that is, the ray from the transducer's "effective origin" (see `Transducer.get_effective_origin`)
            to the focus center.
        cutoff: The threshold against which `da` values are compared.
        origin: A 3D point describing the "effective origin" in the coordinates of `da`
            (see `Transducer.get_effective_origin` for the meaning of this).
        min_offset: How far along the negative focal `dim` direction to sample. By default samples as far as the coordinate
            grid allows.
        max_offset: How far along the positive focal `dim` direction to sample. By default samples as far as the coordinate
            grid allows.

    Returns:
        negoff: Distance from the `focus` along the negative focal `dim` direction until `da` first goes below `cutoff`.
        posoff: Distance from the `focus` along the positive focal `dim` direction until `da` first goes below `cutoff`.
    """
    interp_da = interp_transformed_axis(da, focus=focus, dim=dim, origin=origin, min_offset=min_offset, max_offset=max_offset)
    offset = interp_da.coords[f'offset_d{dim}']
    da_negoff = interp_da.where(offset <= 0, drop=True)
    da_posoff = interp_da.where(offset >= 0, drop=True)
    da_negoff = da_negoff.where(da_negoff < float(cutoff), drop=True)
    if da_negoff.size > 0:
        negoff = float(da_negoff.coords[f'offset_d{dim}'][-1])
    else:
        negoff = np.nan
    da_posoff = da_posoff.where(da_posoff < float(cutoff), drop=True)
    if da_posoff.size > 0:
        posoff = float(da_posoff.coords[f'offset_d{dim}'][0])
    else:
        posoff = np.nan
    return negoff, posoff

def get_beamwidth(
    da: xa.DataArray,
    focus,
    dim,
    cutoff:float | None=None,
    origin=DEFAULT_ORIGIN,
    min_offset:float | None=None,
    max_offset:float | None=None
) -> float:
    """Determine the FWHM (or differently thresholded width) of a DataArray along a focal coordinate system axis.

    See `get_focus_matrix` for the meaning of "focal coordinate system."

    Args:
        da: DataArray whose values will be considered (presumably defined on transducer coordinates).
        focus: A 3D point describing the focus location in the coordinates of `da`
        dim: The name of the dimension of `da` whose corresponding focal coordinate system axis should be sampled along.
            For example, the "axial" dimension of a transducer corresponds to the focal axis (z-axis) in the focal coordinate
            system, that is, the ray from the transducer's "effective origin" (see `Transducer.get_effective_origin`)
            to the focus center.
        cutoff: The threshold against which `da` values are compared. If not provided then the half-max is used, making
            this an "FWHM" function.
        origin: A 3D point describing the "effective origin" in the coordinates of `da`
            (see `Transducer.get_effective_origin` for the meaning of this).
        min_offset: How far along the negative focal `dim` direction to sample. By default samples as far as the coordinate
            grid allows.
        max_offset: How far along the positive focal `dim` direction to sample. By default samples as far as the coordinate
            grid allows.

    Returns: The "beam width" along the focal `dim` direction, by measuring from the from closest-to-focus point along the
        *negative* focal `dim` axis for which `da` goes below `cutoff` up to the closest-to-focus point along the
        *positive* focal `dim` axis for which `da` goes below `cutoff`.
    """
    if cutoff is None:
        cutoff = float(da.max())/2
    negoff, posoff = get_beam_bounds(da, focus=focus, dim=dim, cutoff=float(cutoff), origin=origin, min_offset=min_offset, max_offset=max_offset)
    bw = posoff - negoff
    return bw
