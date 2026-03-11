from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Annotated

import numpy as np

from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.units import getunitconversion


def matrix2xyz(matrix):
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]
    az = np.arctan2(matrix[0, 2], matrix[2, 2])
    el = -np.arctan2(matrix[1, 2], np.sqrt(matrix[2, 2]**2 + matrix[0, 2]**2))
    Raz = np.array([[np.cos(az), 0, np.sin(az)],
                    [0, 1, 0],
                    [-np.sin(az), 0, np.cos(az)]])
    Rel = np.array([[1, 0, 0],
                    [0, np.cos(el), -np.sin(el)],
                    [0, np.sin(el), np.cos(el)]])
    Razel = np.dot(Raz, Rel)
    xv = matrix[:3, 0]
    xyp = np.dot(xv, Razel[:3,1])
    xxp = np.dot(xv, Razel[:3,0])
    roll = np.arctan2(xyp, xxp)
    return x, y, z, az, el, roll

@dataclass
class Element:
    index: Annotated[int, OpenLIFUFieldData("Element index", "Element index")] = 0
    """Element index to identify the element in the array."""

    position: Annotated[np.ndarray, OpenLIFUFieldData("Position", "Position of the element in 3D space")] = field(default_factory=lambda: np.array([0., 0., 0.]))
    """ Position of the element in 3D space as a numpy array [x, y, z]."""

    orientation: Annotated[np.ndarray, OpenLIFUFieldData("Orientation", "Orientation of the element in 3D space")] = field(repr=False, default_factory=lambda: np.array([0., 0., 0.]))
    """ Orientation of the element in 3D space as a numpy array around the [y, x', z''] axes [az, el, roll] in radians."""

    size: Annotated[np.ndarray, OpenLIFUFieldData("Size", "Size of the element in 2D")] = field(default_factory=lambda: np.array([1., 1.]))
    """ Size of the element in 2D as a numpy array [width, length]."""

    sensitivity: Annotated[float | None, OpenLIFUFieldData("Sensitivity", "Sensitivity of the element (Pa/V)")] = None
    """Sensitivity of the element (Pa/V)"""

    impulse_response: Annotated[np.ndarray | None, OpenLIFUFieldData("Impulse response", "Impulse response of the element")] = None
    """Impulse response of the element, can be a single value or an array of values. If an array, `impulse_dt` must be set to the time step of the impulse response. Is convolved with the input signal."""

    impulse_dt: Annotated[float | None, OpenLIFUFieldData("Impulse response timestep", """Impulse response timestep""")] = None
    """Impulse response timestep. If `impulse_response` is an array, this is the time step of the impulse response."""

    pin: Annotated[int, OpenLIFUFieldData("Pin", "Channel pin to which the element is connected")] = -1
    """Channel pin to which the element is connected. 1-(64*number of modules)."""

    units: Annotated[str, OpenLIFUFieldData("Units", "Spatial units")] = "mm"
    """Spatial units of the element specification."""

    def __post_init__(self):
        self.position = np.array(self.position, dtype=np.float64)
        if self.position.shape != (3,):
            raise ValueError("Position must be a 3-element array.")
        self.orientation = np.array(self.orientation, dtype=np.float64)
        if self.orientation.shape != (3,):
            raise ValueError("Orientation must be a 3-element array.")
        self.size = np.array(self.size, dtype=np.float64)
        if self.size.shape != (2,):
            raise ValueError("Size must be a 2-element array.")
        if self.impulse_response is not None:
            if isinstance(self.impulse_response, int | float):
                self.impulse_response = np.array([self.impulse_response])
            self.impulse_response = np.array(self.impulse_response, dtype=np.float64)
            if self.impulse_response.ndim != 1:
                raise ValueError("Impulse response must be a 1-dimensional array.")
            if len(self.impulse_response)>1 and self.impulse_dt is None:
                raise ValueError("Impulse response timestep must be set if impulse response is an array.")

    @property
    def x(self):
        return self.position[0]

    @x.setter
    def x(self, value):
        self.position[0] = value

    @property
    def y(self):
        return self.position[1]

    @y.setter
    def y(self, value):
        self.position[1] = value

    @property
    def z(self):
        return self.position[2]

    @z.setter
    def z(self, value):
        self.position[2] = value

    @property
    def az(self):
        return self.orientation[0]

    @az.setter
    def az(self, value):
        self.orientation[0] = value

    @property
    def el(self):
        return self.orientation[1]

    @el.setter
    def el(self, value):
        self.orientation[1] = value

    @property
    def roll(self):
        return self.orientation[2]

    @roll.setter
    def roll(self, value):
        self.orientation[2] = value

    @property
    def width(self):
        return self.size[0]

    @width.setter
    def width(self, value):
        self.size[0] = value

    @property
    def length(self):
        return self.size[1]

    @length.setter
    def length(self, value):
        self.size[1] = value

    def calc_output(self, input_signal, dt):
        if self.impulse_response is None:
            filtered_signal = input_signal * 1
        elif len(self.impulse_response) == 1:
            filtered_signal = input_signal * self.impulse_response[0]
        else:
            impulse = self.interp_impulse_response(dt)
            filtered_signal = np.convolve(input_signal, impulse, mode='full')
        if self.sensitivity is not None:
            filtered_signal = filtered_signal * self.sensitivity
        return filtered_signal

    def copy(self):
        return copy.deepcopy(self)

    def rescale(self, units):
        if self.units != units:
            scl = getunitconversion(self.units, units)
            self.position *= scl
            self.size *= scl
            self.units = units

    def get_position(self, units=None, matrix=np.eye(4)):
        units = self.units if units is None else units
        scl = getunitconversion(self.units, units)
        pos = self.position * scl
        pos = np.append(pos, 1)
        pos = np.dot(matrix, pos)
        return pos[:3]

    def get_size(self, units=None):
        units = self.units if units is None else units
        scl = getunitconversion(self.units, units)
        ele_width = self.size[0] * scl
        ele_length = self.size[1] * scl
        return ele_width, ele_length

    def get_area(self, units=None):
        units = self.units if units is None else units
        ele_width, ele_length = self.get_size(units)
        return ele_width * ele_length

    def get_corners(self, units=None, matrix=np.eye(4)):
        units = self.units if units is None else units
        scl = getunitconversion(self.units, units)
        rect = np.array([np.array([-1, -1.,  1,  1]) * 0.5 * self.width,
                            np.array([-1,  1,  1, -1]) * 0.5 * self.length,
                            np.zeros(4) ,
                            np.ones(4)])
        xyz = np.dot(self.get_matrix(), rect)
        xyz1 = np.dot(matrix, xyz)
        corner = []
        for j in range(3):
            corner.append(xyz1[j, :] * scl)
        return np.array(corner)

    def get_matrix(self, units=None):
        units = self.units if units is None else units
        Raz = np.array([[np.cos(self.az), 0, np.sin(self.az)],
                        [0, 1, 0],
                        [-np.sin(self.az), 0, np.cos(self.az)]])
        Rel = np.array([[1, 0, 0],
                        [0, np.cos(self.el), -np.sin(self.el)],
                        [0, np.sin(self.el), np.cos(self.el)]])
        Rroll = np.array([[np.cos(self.roll), -np.sin(self.roll), 0],
                            [np.sin(self.roll), np.cos(self.roll), 0],
                            [0, 0, 1]])
        pos = self.get_position(units=units)
        m = np.concatenate((np.dot(Raz, np.dot(Rel,Rroll)), pos.reshape([3,1])), axis=1)
        m = np.concatenate((m, [[0, 0, 0, 1]]), axis=0)
        return m

    def get_angle(self, units="rad"):
        # Return angles about the x, y', and z'' axes (el, az, roll)
        if units == "rad":
            el = self.el
            az = self.az
            roll = self.roll
        elif units == "deg":
            el = np.degrees(self.el)
            az = np.degrees(self.az)
            roll = np.degrees(self.roll)
        return el, az, roll

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

    def distance_to_point(self, point, units=None, matrix=np.eye(4)):
        units = self.units if units is None else units
        pos = np.concatenate([self.get_position(units=units), [1]])
        m = self.get_matrix(units=units)
        gpos = np.dot(matrix, pos)
        vec = point - gpos[:3]
        dist = np.linalg.norm(vec, 2)
        return dist

    def angle_to_point(self, point, units=None, return_as="rad", matrix=np.eye(4)):
        units = self.units if units is None else units
        m = self.get_matrix(units=units)
        gm = np.dot(matrix, m)
        v1 = point - gm[:3, 3]
        v2 = gm[:3, 2]
        v1 = v1 / np.linalg.norm(v1, 2)
        v2 = v2 / np.linalg.norm(v2, 2)
        vcross = np.cross(v1, v2)
        theta = np.arcsin(np.linalg.norm(vcross, 2))
        if return_as == "deg":
            theta = np.degrees(theta)
        return theta

    def set_matrix(self, matrix, units=None):
        if units is not None:
            self.rescale(units)
        x, y, z, az, el, roll = matrix2xyz(matrix)
        self.position = np.array([x, y, z])
        self.orientation = np.array([az, el, roll])

    def to_dict(self):
        d = {"index": self.index,
                "position": self.position.tolist(),
                "orientation": self.orientation.tolist(),
                "size": self.size.tolist(),
                "pin": self.pin,
                "units": self.units}
        if self.impulse_response is not None:
            d["impulse_response"] = self.impulse_response.tolist()
        if self.impulse_dt is not None:
            d["impulse_dt"] = self.impulse_dt
        return d

    @staticmethod
    def from_dict(d):
        if 'x' in d:
            d = copy.deepcopy(d)
            d["position"] = np.array([d.pop('x'), d.pop('y'), d.pop('z')])
            d["orientation"] = np.array([d.pop('az'), d.pop('el'), d.pop('roll')])
            d["size"] = np.array([d.pop('w'), d.pop('l')])
        if "impulse_response" in d and d["impulse_response"] is not None:
            d["impulse_response"] = np.array(d["impulse_response"])
        if "impulse_dt" in d and d["impulse_dt"] is not None:
            d["impulse_dt"] = float(d["impulse_dt"])
        return Element(**d)
