from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Annotated

import numpy as np

from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.units import getunitconversion

SENS_FREQ_KEY = "freq_Hz"
SENS_VALUE_KEY = "values_Pa_per_V"


def normalize_sensitivity(sensitivity: float | dict) -> float | dict[str, list[float]]:
    """Normalize sensitivity to a canonical representation.

    Canonical frequency-dependent representation is:
        {"freq_Hz": [...], "values_Pa_per_V": [...]}

    Backward-compatible legacy representation with frequency keys is accepted and
    converted to the canonical representation.
    """
    if isinstance(sensitivity, dict):
        if SENS_FREQ_KEY in sensitivity or SENS_VALUE_KEY in sensitivity:
            if SENS_FREQ_KEY not in sensitivity or SENS_VALUE_KEY not in sensitivity:
                raise ValueError("Sensitivity dictionary must include both 'freq_Hz' and 'values_Pa_per_V'.")
            freqs = np.asarray(sensitivity[SENS_FREQ_KEY], dtype=np.float64).reshape(-1)
            values = np.asarray(sensitivity[SENS_VALUE_KEY], dtype=np.float64).reshape(-1)
        else:
            # Legacy format: {frequency_hz: sensitivity}
            if len(sensitivity) == 0:
                raise ValueError("Sensitivity dictionary must not be empty.")
            mapping = {float(k): float(v) for k, v in sensitivity.items()}
            freqs = np.array(list(mapping.keys()), dtype=np.float64)
            values = np.array(list(mapping.values()), dtype=np.float64)

        if len(freqs) == 0:
            raise ValueError("Sensitivity frequency list must not be empty.")
        if len(freqs) != len(values):
            raise ValueError("Sensitivity frequency and value lists must have the same length.")

        order = np.argsort(freqs)
        freqs = freqs[order]
        values = values[order]
        if np.any(np.diff(freqs) <= 0):
            raise ValueError("Sensitivity frequencies must be strictly increasing.")

        return {
            SENS_FREQ_KEY: [float(f) for f in freqs],
            SENS_VALUE_KEY: [float(v) for v in values],
        }

    return float(sensitivity)


def sensitivity_at_frequency(sensitivity: float | dict, frequency: float) -> float:
    sensitivity = normalize_sensitivity(sensitivity)
    if isinstance(sensitivity, dict):
        if frequency in sensitivity[SENS_FREQ_KEY]:
            idx = sensitivity[SENS_FREQ_KEY].index(frequency)
            return float(sensitivity[SENS_VALUE_KEY][idx])
        else:
            freqs = np.array(sensitivity[SENS_FREQ_KEY], dtype=np.float64)
            values = np.array(sensitivity[SENS_VALUE_KEY], dtype=np.float64)
            return float(np.interp(frequency, freqs, values, left=values[0], right=values[-1]))
    return float(sensitivity)


def generate_drive_signal(cycles: float, frequency: float, dt: float, amplitude: float = 1.0) -> np.ndarray:
    """Generate a drive signal with duration constrained by cycles/frequency."""
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if frequency <= 0:
        raise ValueError("frequency must be positive.")
    if cycles <= 0:
        raise ValueError("cycles must be positive.")
    n_samples = max(1, int(np.round(cycles / (frequency * dt))))
    t = np.arange(n_samples, dtype=np.float64) * dt
    return amplitude * np.sin(2 * np.pi * frequency * t)

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

    sensitivity: Annotated[float | dict[float, float], OpenLIFUFieldData("Sensitivity", "Sensitivity of the element (Pa/V), scalar or {frequency_hz: sensitivity}")] = 1.0
    """Sensitivity of the element (Pa/V)"""

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
        if self.sensitivity is None:
            self.sensitivity = 1.0
        elif isinstance(self.sensitivity, dict):
            if len(self.sensitivity) == 0:
                raise ValueError("Sensitivity dictionary must not be empty.")
            mapping = {float(k): float(v) for k, v in self.sensitivity.items()}
            freqs = np.array(sorted(mapping.keys()), dtype=np.float64)
            if np.any(np.diff(freqs) <= 0):
                raise ValueError("Sensitivity dictionary frequencies must be strictly increasing.")
            self.sensitivity = {float(f): mapping[float(f)] for f in freqs}
        else:
            self.sensitivity = float(self.sensitivity)

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

    def get_sensitivity(self, frequency: float) -> float:
        return sensitivity_at_frequency(self.sensitivity, frequency)

    def calc_output(self, input_signal, cycles: float, frequency: float, dt: float):
        drive_signal = generate_drive_signal(input_signal, cycles=cycles, frequency=frequency, dt=dt)
        return drive_signal * self.get_sensitivity(frequency)

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
        d["sensitivity"] = self.sensitivity
        return d

    @staticmethod
    def from_dict(d):
        if 'x' in d:
            d = copy.deepcopy(d)
            d["position"] = np.array([d.pop('x'), d.pop('y'), d.pop('z')])
            d["orientation"] = np.array([d.pop('az'), d.pop('el'), d.pop('roll')])
            d["size"] = np.array([d.pop('w'), d.pop('l')])
        # Backward compatibility: legacy impulse keys are ignored.
        d.pop("impulse_response", None)
        d.pop("impulse_dt", None)
        if "sensitivity" not in d or d["sensitivity"] is None:
            d["sensitivity"] = 1.0
        return Element(**d)
