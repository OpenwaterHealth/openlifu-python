from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Literal

import numpy as np
import vtk

from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.units import getunitconversion
from openlifu.xdc.element import Element

DIMS = ['x', 'y', 'z']
LDIMS = Literal['x','y','z']

@dataclass
class Transducer:
    id: Annotated[str, OpenLIFUFieldData("Transducer ID", "Unique identifier for transducer")] = "transducer"
    """Unique identifier for transducer"""

    name: Annotated[str, OpenLIFUFieldData("Transducer name", "Human readable name for transducer")] = ""
    """Human readable name for transducer"""

    elements: Annotated[List[Element], OpenLIFUFieldData("Elements", "Collection of transducer Elements")] = field(default_factory=list)
    """Collection of transducer Elements"""

    frequency: Annotated[float, OpenLIFUFieldData("Frequency (Hz)", "Nominal array frequency (Hz)")] = 400.6e3
    """Nominal array frequency (Hz)"""

    units: Annotated[str, OpenLIFUFieldData("Units", "Native units of transducer local coordinate space")] = "m"
    """Native units of transducer local coordinate space"""

    attrs: Annotated[Dict[str, Any], OpenLIFUFieldData("Attributes", "Additional transducer attributes")] = field(default_factory=dict)
    """Additional transducer attributes"""

    registration_surface_filename: Annotated[str | None, OpenLIFUFieldData("Registration surface filename", "Relative path to an open surface of the transducer to be used for registration")] = None
    """Relative path to an open surface of the transducer to be used for registration"""

    transducer_body_filename: Annotated[str | None, OpenLIFUFieldData("Transducer body filename", "Relative path to the closed surface mesh for visualizing the transducer body")] = None
    """Relative path to the closed surface mesh for visualizing the transducer body"""

    standoff_transform: Annotated[np.ndarray, OpenLIFUFieldData("Standoff transform", "Affine transform representing the way in which the standoff for this transducer displaces the transducer.\n\nA \"standoff transform\" applies a displacement in transducer space that moves a transducer to where it would\nbe situated with the standoff in place. The idea is that if you start with a transform that places a transducer\ndirectly against skin, then pre-composing that transform by a \"standoff transform\" serves to nudge the transducer\nsuch that there is space for the standoff to be between it and the skin.\n\nSee also `openlifu.geo.create_standoff_transform`.\n\nThe units of this transform are assumed to be the native units of the transducer, the `Transducer.units` field.")] = field(default_factory=lambda: np.eye(4, dtype=float))
    """Affine transform representing the way in which the standoff for this transducer displaces the transducer.

    A "standoff transform" applies a displacement in transducer space that moves a transducer to where it would
    be situated with the standoff in place. The idea is that if you start with a transform that places a transducer
    directly against skin, then pre-composing that transform by a "standoff transform" serves to nudge the transducer
    such that there is space for the standoff to be between it and the skin.

    See also `openlifu.geo.create_standoff_transform`.

    The units of this transform are assumed to be the native units of the transducer, the `Transducer.units` field.
    """

    sensitivity: Annotated[float | None, OpenLIFUFieldData("Sensitivity", "Sensitivity of the element (Pa/V)")] = None
    """Sensitivity of the element (Pa/V)"""

    impulse_response: Annotated[np.ndarray | None, OpenLIFUFieldData("Impulse response", "Impulse response of the element")] = None
    """Impulse response of the element, can be a single value or an array of values. If an array, `impulse_dt` must be set to the time step of the impulse response. Is convolved with the input signal."""

    impulse_dt: Annotated[float | None, OpenLIFUFieldData("Impulse response timestep", """Impulse response timestep""")] = None
    """Impulse response timestep. If `impulse_response` is an array, this is the time step of the impulse response."""

    module_invert: Annotated[List[bool], OpenLIFUFieldData("Invert polarity", "Whether to invert the polarity of the transducer output, per module")] = field(default_factory=lambda: [False])
    """Whether to invert the polarity of the transducer output"""

    def __post_init__(self):
        logging.info("Initializing transducer array")
        if self.name == "":
            self.name = self.id
        for element in self.elements:
            element.rescale(self.units)
        if self.impulse_response is not None:
            self.impulse_response = np.array(self.impulse_response, dtype=np.float64)
            if self.impulse_response.ndim != 1 or len(self.impulse_response)<2:
                raise ValueError("Impulse response must be a 1-dimensional array.")
            if self.impulse_dt is None:
                raise ValueError("Impulse response timestep must be set if impulse response is set.")


    def interp_impulse_response(self, dt=None):
        if dt is None:
            dt = self.impulse_dt
        n0 = len(self.impulse_response)
        t0 = self.impulse_dt * np.arange(n0)
        t1 = np.arange(0, t0[-1] + dt, dt)
        impulse_response = np.interp(t1, t0, self.impulse_response)
        impulse_t = np.arange(len(impulse_response)) * dt
        impulse_t = impulse_t - np.mean(impulse_t)
        return impulse_response, impulse_t

    def calc_output(self, input_signal, dt, delays: np.ndarray = None, apod: np.ndarray = None):
        if delays is None:
            delays = np.zeros(self.numelements())
        if apod is None:
            apod = np.ones(self.numelements())
        if self.impulse_response is None:
            filtered_input_signal = input_signal * 1
        else:
            impulse = self.interp_impulse_response(dt)
            filtered_input_signal = np.convolve(input_signal, impulse, mode='full')
        if self.sensitivity is not None:
            filtered_input_signal = filtered_input_signal * self.sensitivity
        outputs = [np.concatenate([np.zeros(int(delay/dt)), a*element.calc_output(filtered_input_signal, dt)],axis=0) for element, delay, a, in zip(self.elements, delays, apod)]
        max_len = max([len(o) for o in outputs])
        output_signal = np.zeros([self.numelements(), max_len])
        for i, o in enumerate(outputs):
            output_signal[i, :len(o)] = o
        return output_signal

    def copy(self):
        return copy.deepcopy(self)

    def draw(self,
             transform:np.ndarray | None=None,
             units:str | None=None,
             facecolor=[0,1,1,0.5]):
        units = self.units if units is None else units
        actor = self.get_actor(units=units, transform=transform, facecolor=facecolor)
        renderWindow = vtk.vtkRenderWindow()
        renderer = vtk.vtkRenderer()
        renderWindow.AddRenderer(renderer)
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        renderer.AddActor(actor)
        renderWindow.Render()
        renderWindowInteractor.Start()

    def get_actor(self, transform:np.ndarray | None=None, units:str | None=None, facecolor=[0,1,1,0.5]):
        units = self.units if units is None else units
        polydata = self.get_polydata(units=units, transform=transform, facecolor=facecolor)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetInterpolationToFlat()
        return actor

    def get_polydata(self, transform:np.ndarray | None=None, units:str | None=None, facecolor=None):
        """Get a vtk polydata of the transducer. Optionally provide a transform, and units in which to interpret
        that transform. If a transform is provided with no units specified, it is assumed that the units
        are the same as those of the transducer itself. Optionally provide an RGBA color to set."""
        units = self.units if units is None else units
        N = self.numelements()
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(4*N)
        cell_array = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(4)
        facecolor = np.array(facecolor)
        is_color_el_none = np.vectorize(lambda color_el: color_el is None)
        if np.all(is_color_el_none(facecolor)):
            facecolors = np.tile((np.array(None)), (N, 1))
        elif facecolor.ndim == 1:
            facecolors = np.tile((np.array([*facecolor])*255).astype(np.uint8), (N, 1))
        else:
            facecolors = np.array([np.array([*fc])*255 for fc in facecolor]).astype(np.uint8)
        point_index = 0
        matrix = transform if transform is not None else np.eye(4)
        for el, color in zip(self.elements, facecolors):
            corners = el.get_corners(matrix=matrix, units=units)
            rect = vtk.vtkQuad()
            point_ids = rect.GetPointIds()
            for i in range(4):
                points.SetPoint(point_index, corners[:,i])
                point_ids.SetId(i, point_index)
                if color[0] is not None:
                    colors.InsertNextTuple4(*color)
                point_index += 1
            cell_array.InsertNextCell(rect)
        polydata = vtk.vtkPolyData()
        polydata.SetPolys(cell_array)
        polydata.SetPoints(points)
        if not np.all(is_color_el_none(facecolor)):
            polydata.GetPointData().SetScalars(colors)
        return polydata

    def get_area(self, units=None):
        units = self.units if units is None else units
        widths, lengths = zip(*[element.get_size(units=units) for element in self.elements])
        return sum(w * l for w, l in zip(widths, lengths))

    def get_corners(self, transform:np.ndarray | None=None, units:str | None=None):
        units = self.units if units is None else units
        matrix = transform if transform is not None else np.eye(4)
        return [element.get_corners(units=units, matrix=matrix) for element in self.elements]

    def get_effective_origin(self, apodizations:np.ndarray, units:str | None=None):
        """Get the centroid of the effective active region of the transducer based on apodizations.

        Args:
            apodizations: vector of apodizations for the transducer elements
            units: units in which to describe the centroid. If not provided then transducer native units are used.

        Returns: a 3-element array describing the centroid in the transducer coordinate system
        """
        units = self.units if units is None else units
        return (apodizations.reshape(-1,1) * self.get_positions(units=units)).sum(axis=0)/apodizations.sum()

    def get_positions(self, transform:np.ndarray | None=None, units:str | None=None):
        units = self.units if units is None else units
        matrix = transform if transform is not None else np.eye(4)
        positions = [element.get_position(units=units, matrix=matrix) for element in self.elements]
        return np.array(positions)

    def convert_transform(self, matrix:np.ndarray, units:str) -> np.ndarray:
        """Given a transform matrix in some units, convert it to this transducer's native units.

        Args:
            matrix: 4x4 affine transform matrix
            units: units of the coordinate space on which the provided transform matrix operates

        Returns: 4x4 affine transform matrix, now operating on a the transducer's native coordinate space
            (i.e. in the transducer's native units)
        """
        matrix = matrix.copy()
        matrix[0:3, 3] *= getunitconversion(units, self.units)
        return matrix

    def get_standoff_transform_in_units(self, units:str) -> np.ndarray:
        """Get the transducer's standoff transform in the desired units."""
        matrix = self.standoff_transform.copy()
        matrix[0:3, 3] *= getunitconversion(self.units, units)
        return matrix

    @staticmethod
    def merge(list_of_transducers:List[Transducer], offset_pins:bool=False, offset_indices:bool=False, merge_mismatched_sensitivity=True, merged_attrs:dict={}) -> Transducer:
        array_copies = [arr.copy() for arr in list_of_transducers]
        sensitivities = np.array([arr.sensitivity for arr in array_copies if arr.sensitivity is not None])
        if len(sensitivities) > 0 and len(sensitivities) < len(array_copies):
            raise ValueError("If one transducer has a sensitivity, all must have a sensitivity.")
        if len(set(sensitivities)) > 1:
            if not merge_mismatched_sensitivity:
                raise ValueError("Transducers have different sensitivities. Use merge_mismatched_sensitivity=True to merge the relative sensitivities into the merged elements")
            else:
                max_sensitivity = sensitivities.max()
                relative_sensitivities = sensitivities/max_sensitivity
                for array, relative_sensitivity in zip(array_copies, relative_sensitivities):
                    for el in array.elements:
                        if el.sensitivity is not None:
                            el.sensitivity = el.sensitivity * relative_sensitivity
                        else:
                            el.sensitivity = relative_sensitivity
                    array.sensitivity = max_sensitivity
        merged_array = array_copies[0]
        for xform_array in array_copies[1:]:
            if offset_pins:
                for el in xform_array.elements:
                    el.pin = el.pin + merged_array.numelements()
            if offset_indices:
                for el in xform_array.elements:
                    el.index = el.index + merged_array.numelements()
            merged_array.elements += xform_array.elements
            merged_array.module_invert += xform_array.module_invert
        for k, v in merged_attrs.items():
            merged_array.__setattr__(k, v)
        return merged_array

    def numelements(self):
        return len(self.elements)

    def rescale(self, units):
        if self.units != units:
            for element in self.elements:
                element.rescale(units)
            self.units = units

    def sort_by_index(self):
        """Sort the elements of the transducer by their element number."""
        element_order = np.argsort([element.index for element in self.elements])
        self.elements = [self.elements[i] for i in element_order]

    def sort_by_pin(self):
        """Sort the elements of the transducer by their pin number."""
        element_order = np.argsort([element.pin for element in self.elements])
        self.elements = [self.elements[i] for i in element_order]

    def to_dict(self):
        d = self.__dict__.copy()
        d["elements"] = [element.to_dict() for element in d["elements"]]
        if self.impulse_response is None:
            del d["impulse_response"]
        else:
            d["impulse_response"] = d["impulse_response"].tolist()
        if self.impulse_dt is None:
            del d["impulse_dt"]
        d["standoff_transform"] =  d["standoff_transform"].tolist()
        return d

    def to_file(self, filename):
        from openlifu.util.json import to_json
        to_json(self.to_dict(), filename)

    def transform(self, matrix, units=None):
        if units is not None:
            self.rescale(units)
        for el in self.elements:
            el.set_matrix(np.dot(np.linalg.inv(matrix), el.get_matrix()))

    def translate(self, dim: LDIMS, amount: float, units=None):
        if units is not None:
            self.rescale(units)
        matrix = np.eye(4)
        dim_index = list(DIMS).index(dim)
        matrix[dim_index, 3] = amount
        self.transform(matrix, units=units)

    def rotate(self, dim: LDIMS, angle: float, units: Literal["deg", "rad"]="deg"):
        if units == "deg":
            angle_rad = np.deg2rad(angle)
        else:
            angle_rad = angle
        matrix = np.eye(4)
        if dim == 'x':
            matrix[1,1] = np.cos(angle_rad)
            matrix[1,2] = -np.sin(angle_rad)
            matrix[2,1] = np.sin(angle_rad)
            matrix[2,2] = np.cos(angle_rad)
        elif dim == 'y':
            matrix[0,0] = np.cos(angle_rad)
            matrix[0,2] = np.sin(angle_rad)
            matrix[2,0] = -np.sin(angle_rad)
            matrix[2,2] = np.cos(angle_rad)
        elif dim == 'z':
            matrix[0,0] = np.cos(angle_rad)
            matrix[0,1] = -np.sin(angle_rad)
            matrix[1,0] = np.sin(angle_rad)
            matrix[1,1] = np.cos(angle_rad)
        self.transform(matrix, units=units)

    @staticmethod
    def from_file(filename):
        with open(filename) as file:
            data = json.load(file)
        return Transducer.from_dict(data)

    @staticmethod
    def from_dict(d, **kwargs):
        d = d.copy()
        d["elements"] = [Element.from_dict(element) for element in d["elements"]]
        if "impulse_response" in d and d["impulse_response"] is not None:
            if len(d["impulse_response"]) == 1 and "sensitivity" not in d:
                d["sensitivity"] = d["impulse_response"][0]
                del d["impulse_response"]
            else:
                d["impulse_response"] = np.array(d["impulse_response"])
        if "standoff_transform" in d and d["standoff_transform"] is not None:
            d["standoff_transform"] = np.array(d["standoff_transform"])
        return Transducer(**d, **kwargs)

    @staticmethod
    def from_json(json_string : str) -> Transducer:
        """Load a Transducer from a json string"""
        return Transducer.from_dict(json.loads(json_string))

    def to_json(self, compact:bool=False) -> str:
        """Serialize a Transducer to a json string

        Args:
            compact: if enabled then the string is compact (not pretty). Disable for pretty.

        Returns: A json string representing the complete Transducer object.
        """
        if compact:
            return json.dumps(self.to_dict(), separators=(',', ':'))
        else:
            return json.dumps(self.to_dict(), indent=4)

    @staticmethod
    def gen_matrix_array(nx=2, ny=2, pitch=1, kerf=0, units="mm", **kwargs):
        """Generate a 2D flat matrix array

        Args:
            nx: number of elements in the x direction
            ny: number of elements in the y direction
            pitch: distance between element centers
            kerf: distance between element edges
            units: units of the array dimensions
            impulse_response: impulse response of the elements
            impulse_dt: time step of the impulse response
            id: unique identifier
            name: name of the array
            attrs: additional attributes

        Returns: a Transducer object representing the array
        """
        N = nx * ny
        xpos = (np.arange(nx) - (nx - 1) / 2) * pitch # x positions, centered about x=0
        ypos = -(np.arange(ny) - (ny - 1) / 2) * pitch # y positions, centered about y=0
        elements = []
        for i in range(N):
            x = xpos[i // ny] # inner loop through x positions
            y = ypos[i % ny] # outer loop through y positions
            elements.append(Element(
                index=i+1,
                pin=i+1,
                position = np.array([x, y, 0]),
                orientation = np.array([0, 0, 0]),
                size = np.array([pitch - kerf, pitch - kerf]),
                units=units
            ))
        arr = Transducer(elements=elements, units=units, **kwargs)
        return arr

@dataclass
class TransformedTransducer(Transducer):
    transform: np.ndarray = field(default_factory= lambda: np.eye(4))

    def bake(self):
        tdict = self.to_dict()
        tdict.pop("transform")
        t = Transducer.from_dict(tdict)
        t.transform(self.transform, units=self.units)
        return t

    def translate_global(self, dim: LDIMS, amount, units=None):
        if units is None:
            units = self.units
        matrix = np.eye(4)
        dim_index = DIMS.index(dim)
        matrix[dim_index, 3] = amount
        self.transform = self.transform @ np.linalg.inv(matrix)

    def translate_local(self, dim: LDIMS, amount, units=None):
        if units is None:
            units = self.units
        matrix = np.eye(4)
        dim_index = DIMS.index(dim)
        matrix[dim_index, 3] = amount
        self.transform = np.linalg.inv(matrix) @ self.transform

    def rotate_global(self, dim: LDIMS, angle: float, units: Literal["deg", "rad"]="deg"):
        if units == "deg":
            angle_rad = np.deg2rad(angle)
        else:
            angle_rad = angle
        matrix = np.eye(4)
        if dim == 'x':
            matrix[1,1] = np.cos(angle_rad)
            matrix[1,2] = -np.sin(angle_rad)
            matrix[2,1] = np.sin(angle_rad)
            matrix[2,2] = np.cos(angle_rad)
        elif dim == 'y':
            matrix[0,0] = np.cos(angle_rad)
            matrix[0,2] = np.sin(angle_rad)
            matrix[2,0] = -np.sin(angle_rad)
            matrix[2,2] = np.cos(angle_rad)
        elif dim == 'z':
            matrix[0,0] = np.cos(angle_rad)
            matrix[0,1] = -np.sin(angle_rad)
            matrix[1,0] = np.sin(angle_rad)
            matrix[1,1] = np.cos(angle_rad)
        self.transform = self.transform @ matrix

    def rotate_local(self, dim: LDIMS, angle: float, units: Literal["deg", "rad"]="deg"):
        if units == "deg":
            angle_rad = np.deg2rad(angle)
        else:
            angle_rad = angle
        matrix = np.eye(4)
        if dim == 'x':
            matrix[1,1] = np.cos(angle_rad)
            matrix[1,2] = -np.sin(angle_rad)
            matrix[2,1] = np.sin(angle_rad)
            matrix[2,2] = np.cos(angle_rad)
        elif dim == 'y':
            matrix[0,0] = np.cos(angle_rad)
            matrix[0,2] = np.sin(angle_rad)
            matrix[2,0] = -np.sin(angle_rad)
            matrix[2,2] = np.cos(angle_rad)
        elif dim == 'z':
            matrix[0,0] = np.cos(angle_rad)
            matrix[0,1] = -np.sin(angle_rad)
            matrix[1,0] = np.sin(angle_rad)
            matrix[1,1] = np.cos(angle_rad)
        self.transform = matrix @ self.transform

    def to_dict(self):
        tdict = super().to_dict()
        tdict["transform"] = self.transform.tolist()
        return tdict

    @staticmethod
    def from_dict(data, **kwargs):
        d = data.copy()
        transform = np.array(d.pop("transform"))
        t = Transducer.from_dict(d, **kwargs)
        return TransformedTransducer.from_transducer(t, transform)

    @staticmethod
    def from_transducer(t: Transducer, transform: np.ndarray) -> TransformedTransducer:
        tdict = t.__dict__
        return TransformedTransducer(**tdict, transform=np.array(transform))
