from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from openlifu.util.dict_conversion import DictMixin
from openlifu.util.units import getunitconversion
from openlifu.xdc import Transducer, TransformedTransducer


def get_angle_from_gap(width, gap, roc):
    a = roc
    b = width/2
    c = gap/2
    mag = np.sqrt(a**2 + b**2)
    A = a/mag
    B = b/mag
    dth = np.arcsin(c/mag) + np.arcsin(B)
    return dth if A >= 0 else -dth

def get_roc_from_angle(width, gap, dth):
    return (0.5*gap + (0.5 * width * np.cos(dth))) / np.sin(dth)

@dataclass
class TransducerArray(DictMixin):
    id: str = "transducer_array"
    name: str = "Transducer Array"
    modules: list[TransformedTransducer] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)

    def to_transducer(self, offset_pins=True, offset_indices=True):
        t = Transducer.merge([t.bake() for t in self.modules], offset_pins=offset_pins, offset_indices=offset_indices, merged_attrs=self.attrs)
        t.name = self.name
        t.id = self.id
        return t

    @staticmethod
    def from_dict(data: dict):
        d = data.copy()
        if "type" in d:
            d.pop("type")
        d["modules"] = [TransformedTransducer.from_dict(t) for t in data["modules"]]
        if "attrs" in d:
            if "standoff_transform" in d["attrs"] and d["attrs"]["standoff_transform"] is not None:
                d["attrs"]["standoff_transform"] = np.array(d["attrs"]["standoff_transform"])
            d["attrs"].pop("impulse_response", None)
            d["attrs"].pop("impulse_dt", None)
        return TransducerArray(**d)


    def to_dict(self):
        d = {"type": "TransducerArray"}
        d.update(self.__dict__)
        d["modules"] = [t.to_dict() for t in self.modules]
        for k, v in self.attrs.items():
            if isinstance(v, np.ndarray):
                d["attrs"][k] = v.tolist()
        return d

    def to_json(self, compact:bool=False) -> str:
        """Serialize a TransducerArray to a json string

        Args:
            compact: if enabled then the string is compact (not pretty). Disable for pretty.

        Returns: A json string representing the complete TransducerArray object.
        """
        if compact:
            return json.dumps(self.to_dict(), separators=(',', ':'))
        else:
            return json.dumps(self.to_dict(), indent=4)

    def to_file(self, file_path: str, compact: bool = False) -> None:
        """Serialize a TransducerArray to a json file

        Args:
            file_path: The path to the file where the json string will be written.
            compact: if enabled then the string is compact (not pretty). Disable for pretty.
        """
        json_string = self.to_json(compact=compact)
        with open(file_path, 'w') as f:
            f.write(json_string)

    @staticmethod
    def get_concave_cylinder(trans, rows=1, cols=1, width=40, gap=0, dth=None, roc=np.inf, units="mm", id="transducer_array", name="Transducer Array", attrs: dict={}):
        scl = getunitconversion(units, trans.units)
        modules = []
        if roc == np.inf:
            for i in range(rows):
                y = (width+gap)*(i-(rows-1)/2)*scl
                for j in range(cols):
                    dx = (width+gap)*(j-(cols-1)/2)*scl
                    M = np.array([[1,0,0,dx], [0,1,0,y], [0,0,1,0], [0,0,0,1]])
                    trans_new = TransformedTransducer.from_transducer(trans, transform=np.linalg.inv(M))
                    modules.append(trans_new)
        else:
            if dth is None:
                dth = get_angle_from_gap(width, gap, roc)
            elif roc is None and dth is not None and gap is not None:
                roc = get_roc_from_angle(width, gap, dth)
            for i in range(rows):
                y = (width+gap)*(i-(rows-1)/2)*scl
                for j in range(cols):
                    th = dth*2*(j-(cols-1)/2)
                    x = roc*np.sin(th)*scl
                    z = roc*(1-np.cos(th))*scl
                    M = np.array([[np.cos(th),0,-np.sin(th),x],
                                [0,1,0,y],
                                [np.sin(th),0,np.cos(th),z],
                                [0,0,0,1]])
                    trans_new = TransformedTransducer.from_transducer(trans, transform=np.linalg.inv(M))
                    modules.append(trans_new)
        return TransducerArray(modules=modules, id=id, name=name, attrs=attrs)

    @staticmethod
    def from_file(filename: str) -> TransducerArray:
        with open(filename) as f:
            data = json.load(f)
        return TransducerArray.from_dict(data)

    @property
    def registration_surface_filename(self):
        if "registration_surface_filename" in self.attrs:
            return self.attrs["registration_surface_filename"]
        return None

    @registration_surface_filename.setter
    def registration_surface_filename(self, value):
        self.attrs["registration_surface_filename"] = value

    @property
    def transducer_body_filename(self):
        if "transducer_body_filename" in self.attrs:
            return self.attrs["transducer_body_filename"]
        return None

    @transducer_body_filename.setter
    def transducer_body_filename(self, value):
        self.attrs["transducer_body_filename"] = value
