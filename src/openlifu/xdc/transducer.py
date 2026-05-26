from __future__ import annotations

import copy
import html
import json
import logging
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Literal

import numpy as np

from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.units import getunitconversion
from openlifu.xdc.element import (
    Element,
    generate_drive_signal,
    sensitivity_at_frequency,
)

DIMS = ['x', 'y', 'z']
LDIMS = Literal['x','y','z']


def _combine_sensitivities(
    base_sensitivity: float | List[tuple[float, float]],
    scale_sensitivity: float | List[tuple[float, float]],
) -> float | List[tuple[float, float]]:

    if isinstance(base_sensitivity, list) and isinstance(scale_sensitivity, list):
        base_freqs = np.asarray([f for f, _ in base_sensitivity], dtype=np.float64)
        scale_freqs = np.asarray([f for f, _ in scale_sensitivity], dtype=np.float64)
        if not np.array_equal(base_freqs, scale_freqs):
            raise ValueError("Cannot combine sensitivity dictionaries with different frequency keys.")
        base_values = np.asarray([v for _, v in base_sensitivity], dtype=np.float64)
        scale_values = np.asarray([v for _, v in scale_sensitivity], dtype=np.float64)
        return [(float(f), float(v)) for f, v in zip(base_freqs, base_values * scale_values)]
    elif isinstance(base_sensitivity, list):
        factor = float(scale_sensitivity)
        values = np.asarray([v for _, v in base_sensitivity], dtype=np.float64)
        return [(float(f), float(v)) for (f, _), v in zip(base_sensitivity, values * factor)]
    elif isinstance(scale_sensitivity, list):
        factor = float(base_sensitivity)
        values = np.asarray([v for _, v in scale_sensitivity], dtype=np.float64)
        return [(float(f), float(v)) for (f, _), v in zip(scale_sensitivity, factor * values)]
    else:
        return float(base_sensitivity) * float(scale_sensitivity)


def _format_scalar(value: float, precision: int = 3) -> str:
    return np.format_float_positional(float(value), precision=precision, trim="-")


def _format_sensitivity_summary(sensitivity: float | List[tuple[float, float]]) -> str:
    if isinstance(sensitivity, list):
        if not sensitivity:
            return "[]"
        low_f, low_v = sensitivity[0]
        high_f, high_v = sensitivity[-1]
        if len(sensitivity) == 1:
            return (
                f"[{_format_scalar(low_f, precision=0)} Hz: "
                f"{_format_scalar(low_v)} Pa/V]"
            )
        return (
            f"[{len(sensitivity)} pts, "
            f"{_format_scalar(low_f, precision=0)}-"
            f"{_format_scalar(high_f, precision=0)} Hz, "
            f"{_format_scalar(low_v)}-{_format_scalar(high_v)} Pa/V]"
        )
    return f"{_format_scalar(sensitivity)} Pa/V"

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

    sensitivity: Annotated[float | List[tuple[float, float]], OpenLIFUFieldData("Sensitivity", "Sensitivity of the transducer (Pa/V), scalar or list of (frequency, value) tuples")] = 1.0
    """Sensitivity of the transducer (Pa/V), scalar or frequency-dependent list of tuples."""

    crosstalk_frac: Annotated[float, OpenLIFUFieldData("Crosstalk fraction", "Fraction of the signal that leaks into other elements due to crosstalk")] = 0.0
    """Fraction of the signal that leaks into other elements due to crosstalk"""

    crosstalk_dist: Annotated[float, OpenLIFUFieldData("Crosstalk distance", "Distance within which elements experience crosstalk")] = 0.0
    """Distance within which elements experience crosstalk"""

    module_invert: Annotated[List[bool], OpenLIFUFieldData("Invert polarity", "Whether to invert the polarity of the transducer output, per module")] = field(default_factory=lambda: [False])
    """Whether to invert the polarity of the transducer output"""

    def __post_init__(self):
        logging.info("Initializing transducer array")
        if self.name == "":
            self.name = self.id
        for element in self.elements:
            element.rescale(self.units)
        if self.sensitivity is None:
            self.sensitivity = 1.0
        elif isinstance(self.sensitivity, list):
            self.sensitivity = sorted(((float(f), float(v)) for f, v in self.sensitivity), key=lambda t: t[0])

    def __repr__(self) -> str:
        return (
            "Transducer("
            f"id='{self.id}', name='{self.name}', "
            f"elements={self.numelements()}, "
            f"frequency={_format_scalar(self.frequency, precision=0)} Hz, "
            f"units='{self.units}', "
            f"sensitivity={_format_sensitivity_summary(self.sensitivity)}"
            ")"
        )

    def __str__(self) -> str:
        lines = [
            f"Transducer '{self.name}' ({self.id})",
            f"  Elements: {self.numelements()}",
            f"  Frequency: {_format_scalar(self.frequency, precision=0)} Hz",
            f"  Units: {self.units}",
            f"  Sensitivity: {_format_sensitivity_summary(self.sensitivity)}",
            f"  Crosstalk: frac={_format_scalar(self.crosstalk_frac)}, "
            f"dist={_format_scalar(self.crosstalk_dist)} m",
            f"  Meshes: registration={self.registration_surface_filename}, "
            f"body={self.transducer_body_filename}",
        ]
        if self.attrs:
            attr_keys = sorted(str(k) for k in self.attrs)
            preview = ", ".join(attr_keys[:5])
            suffix = " ..." if len(attr_keys) > 5 else ""
            lines.append(f"  Attr Keys ({len(attr_keys)}): {preview}{suffix}")
        return "\n".join(lines)

    def _repr_pretty_(self, p, cycle: bool) -> None:
        if cycle:
            p.text("Transducer(...)")
            return
        p.text(str(self))

    def _repr_html_(self) -> str:
        def line(label: str, value_html: str) -> str:
            return (
                "<div style='margin:1px 0;'>"
                f"<span style='font-weight:600;'>{html.escape(label)}:</span> "
                f"{value_html}"
                "</div>"
            )

        summary_lines = [
            line("ID", html.escape(self.id)),
            line("Name", html.escape(self.name)),
            line("Frequency", html.escape(f"{_format_scalar(self.frequency, precision=0)} Hz")),
            line("Units", html.escape(self.units)),
            line("Sensitivity", html.escape(_format_sensitivity_summary(self.sensitivity))),
            line(
                "Crosstalk",
                html.escape(
                    f"frac={_format_scalar(self.crosstalk_frac)}, "
                    f"dist={_format_scalar(self.crosstalk_dist)} m"
                ),
            ),
            line("Registration Mesh", html.escape(str(self.registration_surface_filename))),
            line("Body Mesh", html.escape(str(self.transducer_body_filename))),
            line("Attr Keys", html.escape(", ".join(sorted(str(k) for k in self.attrs)) or "-")),
        ]

        # Per-element sensitivity is the product of the element's stored
        # sensitivity and the module's sensitivity. For frequency-dependent
        # module sensitivity we evaluate it at the module's center frequency
        # so the displayed per-element value matches what would be applied
        # at the nominal drive frequency.
        module_sens_at_f = sensitivity_at_frequency(self.sensitivity, self.frequency)

        element_rows = "".join(
            "<details style='margin:1px 0;'>"
            "<summary style='cursor:pointer;'>"
            f"<span style='display:inline-block;min-width:48px;'>#{element.index}</span>"
            f"<span style='display:inline-block;min-width:56px;'>pin {element.pin}</span>"
            f"<span style='display:inline-block;min-width:170px;'>pos [{_format_scalar(element.position[0])}, {_format_scalar(element.position[1])}, {_format_scalar(element.position[2])}]</span>"
            f"<span style='display:inline-block;min-width:120px;'>size [{_format_scalar(element.size[0])}, {_format_scalar(element.size[1])}]</span>"
            f"<span>{html.escape(_format_sensitivity_summary(_combine_sensitivities(element.sensitivity, module_sens_at_f)))}</span>"
            "</summary>"
            "<div style='margin:6px 0 0 14px;padding-left:10px;border-left:2px solid rgba(127,127,127,0.35);'>"
            f"{element._repr_html_()}"
            "</div>"
            "</details>"
            for element in self.elements
        )

        elements_section = (
            "<details style='margin:1px 0;'>"
            f"<summary style='cursor:pointer;display:inline;'>"
            f"<span style='font-weight:600;'>Elements:</span> {self.numelements()}"
            "</summary>"
            "<div style='margin:6px 0 0 14px;padding-left:10px;border-left:2px solid rgba(127,127,127,0.35);max-height:340px;overflow:auto;'>"
            f"{element_rows}"
            "</div>"
            "</details>"
        )

        return (
            "<div style='font-family:ui-monospace,monospace;line-height:1.35;'>"
            "<div style='font-weight:600;margin-bottom:4px;'>Transducer</div>"
            f"{''.join(summary_lines)}"
            f"{elements_section}"
            "</div>"
        )

    def calc_output(self, cycles: float, frequency: float, dt: float, delays: np.ndarray = None, apod: np.ndarray = None, amplitude: float = 1.0) -> np.ndarray:
        if delays is None:
            delays = np.zeros(self.numelements())
        if apod is None:
            apod = np.ones(self.numelements())
        drive_signal = generate_drive_signal(cycles=cycles, frequency=frequency, dt=dt, amplitude=amplitude)
        base_output = drive_signal * sensitivity_at_frequency(self.sensitivity, frequency)
        outputs = [
            np.concatenate(
                [np.zeros(int(delay / dt)), a * sensitivity_at_frequency(element.sensitivity, frequency) * base_output],
                axis=0,
            )
            for element, delay, a, in zip(self.elements, delays, apod)
        ]
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
        import vtk
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
        import vtk
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
        import vtk
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
        dict_key_sets = set()
        for array in array_copies:
            if isinstance(array.sensitivity, list):
                dict_key_sets.add(tuple(f for f, _ in array.sensitivity))
            for el in array.elements:
                if isinstance(el.sensitivity, list):
                    dict_key_sets.add(tuple(f for f, _ in el.sensitivity))
        if len(dict_key_sets) > 1:
            raise ValueError("Cannot merge sensitivities with different frequency keys.")

        sensitivity_signatures = []
        for array in array_copies:
            if isinstance(array.sensitivity, list):
                sensitivity_signatures.append((
                    tuple(f for f, _ in array.sensitivity),
                    tuple(v for _, v in array.sensitivity),
                ))
            else:
                sensitivity_signatures.append(float(array.sensitivity))

        if not merge_mismatched_sensitivity and len(set(sensitivity_signatures)) > 1:
            raise ValueError("Transducers have different sensitivities. Use merge_mismatched_sensitivity=True to merge them into the merged elements")

        for array in array_copies:
            transducer_sensitivity = array.sensitivity
            for el in array.elements:
                el.sensitivity = _combine_sensitivities(el.sensitivity, transducer_sensitivity)
            array.sensitivity = 1.0

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
        # Backward compatibility: legacy impulse fields are ignored.
        d.pop("impulse_response", None)
        d.pop("impulse_dt", None)
        if "sensitivity" not in d or d["sensitivity"] is None:
            d["sensitivity"] = 1.0
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

    @classmethod
    def from_module_user_config(cls, user_config: dict) -> Transducer:
        """Build a single-module ``Transducer`` from a module ``user_config`` dict.

        The dict is expected to follow the structure produced by the SDK's
        :py:meth:`openlifu_sdk.io.LIFUTXDevice.TxDevice.read_config`: a top-level
        record with a nested ``"module"`` sub-dict whose fields are the kwargs
        for :py:meth:`gen_matrix_array` (``nx``, ``ny``, ``pitch``, ``kerf``,
        plus any ``Transducer`` fields such as ``frequency``, ``sensitivity``,
        ``crosstalk_frac``, ``crosstalk_dist``, ``id``, ``name``). The
        top-level ``"hwid"`` is preserved in the resulting transducer's
        ``attrs`` dict under the key ``"hwid"`` so callers can identify the
        physical module later.

        Mesh filenames, standoff transforms, and any other per-module data
        that cannot be carried in the user_config remain at their dataclass
        defaults; supply a template via :py:meth:`TransducerArray.from_module_user_configs`
        to inject those.
        """
        module_cfg = (user_config.get("module") or {})
        if not module_cfg:
            raise ValueError("user_config has no 'module' sub-dict")
        t = cls.gen_matrix_array(**module_cfg)
        hwid = user_config.get("hwid")
        if hwid is not None:
            t.attrs = dict(t.attrs)
            t.attrs["hwid"] = hwid
        return t

    @staticmethod
    def gen_matrix_array(nx=2, ny=2, pitch=1, kerf=0, units="mm", **kwargs):
        """Generate a 2D flat matrix array

        Args:
            nx: number of elements in the x direction
            ny: number of elements in the y direction
            pitch: distance between element centers
            kerf: distance between element edges
            units: units of the array dimensions
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
