from __future__ import annotations

import copy
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
import xarray as xa

from openlifu.seg.material import MATERIALS, PARAM_INFO, Material
from openlifu.util.annotations import OpenLIFUFieldData


@dataclass
class SegmentationMethod(ABC):
    materials: Annotated[dict[str, Material], OpenLIFUFieldData("Segmentation materials", "Dictionary mapping of label names to material definitions used during segmentation")] = field(default_factory=lambda: MATERIALS.copy())
    """Dictionary mapping of label names to material definitions used during segmentation"""

    ref_material: Annotated[str, OpenLIFUFieldData("Reference material", "Reference material ID to use")] = "water"
    """Reference material ID to use"""

    def __post_init__(self):
        if self.materials is None:
            self.materials = MATERIALS.copy()
        if not isinstance(self.materials, dict):
            raise TypeError(f"Materials must be a dictionary, got {type(self.materials).__name__}.")
        if not all(isinstance(m, Material) for m in self.materials.values()):
            raise TypeError("All materials must be instances of Material class.")
        if self.ref_material not in self.materials:
            raise ValueError(f"Reference material {self.ref_material} not found.")

    @abstractmethod
    def _segment(self, volume: xa.DataArray) -> xa.DataArray:
        pass

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d['materials'] = { k: v.to_dict() for k, v in self.materials.items() }
        d['class'] = self.__class__.__name__
        return d

    @staticmethod
    def from_dict(d: dict, on_keyword_mismatch: Literal['warn', 'raise', 'ignore'] = 'warn') -> SegmentationMethod:
        from openlifu.seg import seg_methods
        if not isinstance(d, dict):  # previous implementations might pass str
            raise TypeError(f"Expected dict for from_dict, got {type(d).__name__}")

        d = copy.deepcopy(d)
        short_classname = d.pop("class")

        # Recursively construct Material instances
        materials_dict = d.get("materials")
        if materials_dict is not None:
            d["materials"] = {
                k: v if isinstance(v, Material) else Material.from_dict(v)
                for k, v in materials_dict.items()
            }

        class_constructor = getattr(seg_methods, short_classname)

        # Filter out unexpected keywords
        sig = inspect.signature(class_constructor)
        expected_keywords = [p.name for p in sig.parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD]
        unexpected_keywords = [k for k in d if k not in expected_keywords]

        if unexpected_keywords:
            if on_keyword_mismatch == 'raise':
                raise TypeError(f"Unexpected keyword arguments for {short_classname}: {unexpected_keywords}")
            elif on_keyword_mismatch == 'warn':
                logging.warning(f"Ignoring unexpected keyword arguments for {short_classname}: {unexpected_keywords}")
            for k in unexpected_keywords:
                d.pop(k)

        return class_constructor(**d)

    def _material_indices(self, materials: dict | None = None):
        materials = self.materials if materials is None else materials
        return {material_id: i for i, material_id in enumerate(materials.keys())}

    def _map_params(self, seg: xa.DataArray, materials: dict | None = None):
        materials = self.materials if materials is None else materials
        material_dict = self._material_indices(materials=materials)
        params = xa.Dataset()
        ref_mat = materials[self.ref_material]
        for param_id in PARAM_INFO:
            info = Material.param_info(param_id)
            param = xa.DataArray(np.zeros(seg.shape), coords=seg.coords, attrs={"units": info["units"], "long_name": info["name"], "ref_value": ref_mat.get_param(param_id)})
            for material_id, material in materials.items():
                midx = material_dict[material_id]
                param.data[seg.data == midx] = getattr(material, param_id)
            params[param_id] = param
        params.attrs['ref_material'] = ref_mat
        return params

    def seg_params(self, volume: xa.DataArray, materials: dict | None = None):
        materials = self.materials if materials is None else materials
        seg = self._segment(volume)
        params = self._map_params(seg, materials=materials)
        return params

    def ref_params(self, coords: xa.Coordinates):
        seg = self._ref_segment(coords)
        params = self._map_params(seg)
        return params

    def _ref_segment(self, coords: xa.Coordinates):
        material_dict = self._material_indices()
        m_idx = material_dict[self.ref_material]
        sz = list(coords.sizes.values())
        seg = xa.DataArray(np.full(sz, m_idx, dtype=int), coords=coords)
        return seg

    @abstractmethod
    def to_table(self) -> pd.DataFrame:
        """
        Get a table of the segmentation method parameters

        :returns: Pandas DataFrame of the segmentation method parameters
        """
        pass

    def get_summary(self) -> str:
        """Return a one-liner summary of the segmentation method.

        Default implementation returns the human-friendly form of the class name
        (e.g. ``"Uniform Tissue"``); subclasses may override to provide more
        detail.
        """
        # Insert spaces before capital letters: "UniformTissue" -> "Uniform Tissue"
        name = type(self).__name__
        result = []
        for i, ch in enumerate(name):
            if i > 0 and ch.isupper() and not name[i - 1].isupper():
                result.append(" ")
            result.append(ch)
        return "".join(result)
