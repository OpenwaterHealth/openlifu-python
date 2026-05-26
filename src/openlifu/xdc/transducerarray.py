from __future__ import annotations

import copy
import html
import json
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from openlifu.util.dict_conversion import DictMixin
from openlifu.util.units import getunitconversion
from openlifu.xdc import Transducer, TransformedTransducer

try:
    from openlifu_sdk.io import LIFUInterface as _SDKLIFUInterface
except ImportError:
    _SDKLIFUInterface = None


def _format_scalar(value: float, precision: int = 3) -> str:
    return np.format_float_positional(float(value), precision=precision, trim="-")

# Mapping from (num_connected_modules, freq_khz) to the canonical
# template id consumed by :py:meth:`TransducerArray.get_connected`.
_DEFAULT_TEMPLATE_IDS: dict[tuple[int, int], str] = {
    (1, 155): "openlifu_1x155",
    (1, 400): "openlifu_1x400",
    (2, 155): "openlifu_2x155",
    (2, 400): "openlifu_2x400",
}

# Locally-embedded per-module transforms and array-level standoff for
# the canonical default templates. Used as a meshless fallback when no
# database is provided to :py:meth:`TransducerArray.get_connected`.
# The 2x155 entries currently mirror the openlifu_2x180_evt1 template
# as a stand-in until a dedicated 155 kHz template ships.
_DEFAULT_TEMPLATE_DATA: dict[str, dict] = {
    "openlifu_1x155": {
        "name": "OpenLIFU 1x 155kHz",
        "module_transforms": [np.eye(4, dtype=float)],
        "standoff_transform": np.eye(4, dtype=float),
    },
    "openlifu_1x400": {
        "name": "OpenLIFU 1x 400kHz",
        "module_transforms": [np.eye(4, dtype=float)],
        "standoff_transform": np.eye(4, dtype=float),
    },
    "openlifu_2x155": {
        "name": "OpenLIFU 2x 155kHz",
        "module_transforms": [
            np.array([
                [0.9697859993972769, 0.0, -0.2439571998794554, 25.84571998794554],
                [0.0, 1.0, 0.0, 0.0],
                [0.24395719987945538, 0.0, 0.9697859993972772, 3.20098197421292],
                [0.0, 0.0, 0.0, 1.0],
            ], dtype=float),
            np.array([
                [0.9697859993972769, 0.0, 0.2439571998794554, -25.84571998794554],
                [0.0, 1.0, 0.0, 0.0],
                [-0.24395719987945538, 0.0, 0.9697859993972772, 3.20098197421292],
                [0.0, 0.0, 0.0, 1.0],
            ], dtype=float),
        ],
        "standoff_transform": np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.997684, -0.0680153, 0.0],
            [0.0, 0.0680153, 0.997684, -8.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float),
    },
    "openlifu_2x400": {
        "name": "OpenLIFU 2x 400kHz",
        "module_transforms": [
            np.array([
                [0.9659258262890683, 0.0, -0.25881904510252074, 25.84571998794554],
                [0.0, 1.0, 0.0, 0.0],
                [0.25881904510252074, 0.0, 0.9659258262890683, 3.20098197421292],
                [0.0, 0.0, 0.0, 1.0],
            ], dtype=float),
            np.array([
                [0.9659258262890683, 0.0, 0.25881904510252074, -25.84571998794554],
                [0.0, 1.0, 0.0, 0.0],
                [-0.25881904510252074, 0.0, 0.9659258262890683, 3.20098197421292],
                [0.0, 0.0, 0.0, 1.0],
            ], dtype=float),
        ],
        "standoff_transform": np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -8.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float),
    },
}


def _build_meshless_default_template(template_id: str) -> TransducerArray:
    """Build a meshless template :class:`TransducerArray` from embedded transforms."""
    spec = _DEFAULT_TEMPLATE_DATA[template_id]
    modules: list[TransformedTransducer] = []
    for tform in spec["module_transforms"]:
        t = Transducer(id=template_id, elements=[])
        modules.append(TransformedTransducer.from_transducer(t, transform=np.array(tform, dtype=float)))
    attrs = {"standoff_transform": np.array(spec["standoff_transform"], dtype=float)}
    return TransducerArray(id=template_id, name=spec["name"], modules=modules, attrs=attrs)


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

def get_gap_from_angle(width, dth, roc):
    a = roc
    b = width/2
    mag = np.sqrt(a**2 + b**2)
    A = a/mag
    B = b/mag
    gap = 2*mag*np.sin(dth - np.arcsin(B))
    return gap if A >= 0 else -gap

@dataclass
class TransducerArray(DictMixin):
    id: str = "transducer_array"
    name: str = "Transducer Array"
    modules: list[TransformedTransducer] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        total_elements = sum(m.numelements() for m in self.modules)
        return (
            "TransducerArray("
            f"id='{self.id}', name='{self.name}', "
            f"modules={len(self.modules)}, total_elements={total_elements}"
            ")"
        )

    def __str__(self) -> str:
        total_elements = sum(m.numelements() for m in self.modules)
        lines = [
            f"TransducerArray '{self.name}' ({self.id})",
            f"  Modules: {len(self.modules)}",
            f"  Total Elements: {total_elements}",
        ]
        if self.modules:
            module_preview = [m.id for m in self.modules[:4]]
            suffix = " ..." if len(self.modules) > 4 else ""
            lines.append(f"  Module IDs: {', '.join(module_preview)}{suffix}")
            lines.append(
                "  Module HWIDs: "
                + ", ".join(str((m.attrs or {}).get("hwid")) for m in self.modules[:4])
                + (" ..." if len(self.modules) > 4 else "")
            )
        if self.attrs:
            attr_keys = sorted(str(k) for k in self.attrs)
            preview = ", ".join(attr_keys[:6])
            lines.append(f"  Attr Keys ({len(attr_keys)}): {preview}{' ...' if len(attr_keys) > 6 else ''}")
        return "\n".join(lines)

    def _repr_pretty_(self, p, cycle: bool) -> None:
        if cycle:
            p.text("TransducerArray(...)")
            return
        p.text(str(self))

    def _repr_html_(self) -> str:
        total_elements = sum(m.numelements() for m in self.modules)
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
            line("Total Elements", html.escape(str(total_elements))),
        ]

        module_rows = "".join(
            "<details style='margin:3px 0;'>"
            f"<summary style='cursor:pointer;'>"
            f"Module {i}: {html.escape(m.id)} | {m.numelements()} elements | "
            f"HWID={html.escape(str((m.attrs or {}).get('hwid')))} | "
            f"t=[{_format_scalar(m.transform[0, 3])}, {_format_scalar(m.transform[1, 3])}, {_format_scalar(m.transform[2, 3])}]"
            "</summary>"
            "<div style='margin:6px 0 0 14px;padding-left:10px;border-left:2px solid rgba(127,127,127,0.35);'>"
            f"{m._repr_html_()}"
            "</div>"
            "</details>"
            for i, m in enumerate(self.modules)
        )

        attr_keys_line = line(
            "Attr Keys", html.escape(", ".join(sorted(str(k) for k in self.attrs)) or "-")
        )

        return (
            "<div style='font-family:ui-monospace,monospace;line-height:1.35;'>"
            "<div style='font-weight:600;margin-bottom:4px;'>TransducerArray</div>"
            f"{''.join(summary_lines)}"
            "<details style='margin:1px 0;'>"
            f"<summary style='cursor:pointer;display:inline;'>"
            f"<span style='font-weight:600;'>Modules:</span> {len(self.modules)}"
            "</summary>"
            "<div style='margin:6px 0 0 14px;padding-left:10px;border-left:2px solid rgba(127,127,127,0.35);max-height:340px;overflow:auto;'>"
            f"{module_rows}"
            "</div>"
            "</details>"
            f"{attr_keys_line}"
            "</div>"
        )

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
    def get_concave_cylinder(trans, rows=1, cols=1, width=40, gap=None, dth=None, roc=None, units="mm", id="transducer_array", name="Transducer Array", attrs: dict={}):

        modules = []
        if isinstance(trans, Transducer):
            trans_arr = np.array([[trans]*cols for _ in range(rows)])
        else:
            trans_arr = np.array(trans).reshape(rows, cols)
        scl = getunitconversion(units, trans_arr[0,0].units)
        if gap is None:
            if dth is not None and roc is not None:
                gap = get_gap_from_angle(width, dth, roc)
            else:
                gap = 0
        elif dth is not None and roc is not None:
            raise ValueError("Invalid combination of parameters: cannot specify all of gap, dth, and roc.")

        if dth is None:
            if roc is not None:
                dth = get_angle_from_gap(width, gap, roc)
            else:
                dth = 0
                roc = np.inf

        if roc is None:
            if np.isclose(dth, 0.0):
                roc = np.inf
            else:
                roc = get_roc_from_angle(width, gap, dth)

        if dth == 0:
            for i in range(rows):
                y = (width+gap)*(i-(rows-1)/2)*scl
                for j in range(cols):
                    dx = (width+gap)*(j-(cols-1)/2)*scl
                    M = np.array([[1,0,0,dx], [0,1,0,y], [0,0,1,0], [0,0,0,1]])
                    trans_new = TransformedTransducer.from_transducer(trans_arr[i,j], transform=np.linalg.inv(M))
                    modules.append(trans_new)
        else:
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
                    trans_new = TransformedTransducer.from_transducer(trans_arr[i,j], transform=np.linalg.inv(M))
                    modules.append(trans_new)
        return TransducerArray(modules=modules, id=id, name=name, attrs=attrs)

    @staticmethod
    def from_file(filename: str) -> TransducerArray:
        with open(filename) as f:
            data = json.load(f)
        return TransducerArray.from_dict(data)

    @classmethod
    def from_module_user_configs(
        cls,
        user_configs: Sequence[dict],
        template: TransducerArray | None = None,
        module_transforms: Sequence[np.ndarray] | None = None,
        arr_id: str | None = None,
        arr_name: str | None = None,
    ) -> TransducerArray:
        """Construct a :class:`TransducerArray` from one or more module ``user_config`` dicts.

        Each ``user_config`` describes a single physical module as reported by
        the SDK (``hwid``, ``module`` sub-dict suitable for
        :py:meth:`Transducer.gen_matrix_array`, optional ``device`` sub-dict on
        the lead module). User configs cannot carry mesh data, so a
        ``template`` :class:`TransducerArray` is normally supplied to inject
        per-module mesh filenames / standoff transforms / placement transforms
        and array-level metadata (id, name, attrs).

        Sources of array-level metadata, lowest priority first:

        1. ``template``: provides ``id``, ``name``, ``attrs``, and per-module
           ``transform``, ``standoff_transform``, ``registration_surface_filename``,
           ``transducer_body_filename``. Modules are matched to ``user_configs``
           positionally.
        2. ``user_configs[0]["device"]`` (if present): overrides ``id``,
           ``name``, merges into ``attrs``, and supplies per-module transforms
           keyed by ``hwid`` (falling back to positional matching when
           ``hwid`` is not present in the device entry).
        3. ``module_transforms`` (if given): per-module 4x4 transforms that
           override everything else. Length must match ``user_configs``.
        4. ``arr_id`` / ``arr_name`` (if given): explicit array id/name that
           override the values picked up from the device config or template.

        The per-module ``Transducer`` is always rebuilt from the user_config's
        ``module`` field (this is the on-device truth for nx/ny/pitch/kerf/
        frequency/sensitivity/etc.); only metadata that cannot live in the
        user_config is taken from the template.

        Args:
            user_configs: ordered list of user_config dicts. Order corresponds
                to module index as reported by the device.
            template: optional template array; see above for what it supplies.
            module_transforms: optional list of explicit 4x4 transforms,
                one per user_config, that override template/device transforms.
            arr_id: optional explicit array id. Highest-priority source for the
                resulting ``TransducerArray.id`` (overrides device/template/default).
            arr_name: optional explicit array name. Highest-priority source for
                the resulting ``TransducerArray.name``.

        Returns:
            A :class:`TransducerArray` whose ``modules`` are
            :class:`TransformedTransducer` instances built from the
            user_configs.
        """
        if not user_configs:
            raise ValueError("user_configs must contain at least one user_config dict")
        if module_transforms is not None and len(module_transforms) != len(user_configs):
            raise ValueError(
                f"module_transforms length ({len(module_transforms)}) does not match "
                f"user_configs length ({len(user_configs)})"
            )

        # ---- Resolve array-level metadata ----
        # Priority (highest first): explicit arg > device config > template > default.
        resolved_id: str = "transducer_array"
        resolved_name: str = "Transducer Array"
        arr_attrs: dict = {}
        if template is not None:
            resolved_id = template.id
            resolved_name = template.name
            arr_attrs = copy.deepcopy(template.attrs)

        device_cfg = user_configs[0].get("device") or None
        device_modules_in_order: list = []
        device_modules_by_hwid: dict = {}
        if device_cfg:
            resolved_id = device_cfg.get("id", resolved_id)
            resolved_name = device_cfg.get("name", resolved_name)
            for k, v in (device_cfg.get("attrs") or {}).items():
                arr_attrs[k] = v
            device_modules_in_order = list(device_cfg.get("modules") or [])
            device_modules_by_hwid = {
                m["hwid"]: m
                for m in device_modules_in_order
                if isinstance(m, dict) and m.get("hwid")
            }

        if arr_id is not None:
            resolved_id = arr_id
        if arr_name is not None:
            resolved_name = arr_name

        # Normalize array-level standoff_transform to ndarray if present
        st = arr_attrs.get("standoff_transform")
        if st is not None and not isinstance(st, np.ndarray):
            arr_attrs["standoff_transform"] = np.array(st, dtype=float)

        # ---- Build each module ----
        template_modules: list = list(template.modules) if template is not None else []
        modules: list[TransformedTransducer] = []
        for i, cfg in enumerate(user_configs):
            t = Transducer.from_module_user_config(cfg)
            template_mod = template_modules[i] if i < len(template_modules) else None

            # Inherit per-module data from template (mesh/standoff/etc.)
            if template_mod is not None:
                t.registration_surface_filename = template_mod.registration_surface_filename
                t.transducer_body_filename = template_mod.transducer_body_filename
                if template_mod.standoff_transform is not None:
                    t.standoff_transform = np.array(template_mod.standoff_transform, dtype=float)
                if template_mod.module_invert:
                    t.module_invert = list(template_mod.module_invert)

            # Resolve transform: template < device < explicit override
            transform = np.eye(4)
            if template_mod is not None:
                transform = np.array(template_mod.transform, dtype=float)

            hwid = cfg.get("hwid")
            device_mod: dict | None = None
            if hwid and hwid in device_modules_by_hwid:
                device_mod = device_modules_by_hwid[hwid]
            elif device_modules_in_order and i < len(device_modules_in_order):
                candidate = device_modules_in_order[i]
                if isinstance(candidate, dict):
                    device_mod = candidate
            if device_mod is not None and device_mod.get("transform") is not None:
                transform = np.array(device_mod["transform"], dtype=float)

            if module_transforms is not None:
                transform = np.array(module_transforms[i], dtype=float)

            modules.append(TransformedTransducer.from_transducer(t, transform=transform))

        return cls(id=resolved_id, name=resolved_name, modules=modules, attrs=arr_attrs)

    @classmethod
    def get_connected(
        cls,
        interface=None,
        db=None,
        arr_id: str | None = None,
        arr_name: str | None = None,
        module_transforms: Sequence[np.ndarray] | None = None,
        use_default_template: bool = True,
    ) -> TransducerArray:
        """Read ``user_config`` from every connected TX module and build a :class:`TransducerArray`.

        Picks a default template based on the number of connected modules
        and the per-module ``freq`` value (which must agree across modules
        when more than one is connected). The mapping is:

        ====================== =====================
        ``(n_modules, freq)``  template id
        ====================== =====================
        ``(1, 155)``           ``openlifu_1x155``
        ``(1, 400)``           ``openlifu_1x400``
        ``(2, 155)``           ``openlifu_2x155``
        ``(2, 400)``           ``openlifu_2x400``
        ====================== =====================

        When ``db`` is provided, the template (with its meshes) is loaded
        from the database via ``db.load_transducer(template_id, convert_array=False)``.
        If no database is provided (or the lookup fails) and
        ``use_default_template`` is ``True``, a meshless fallback template
        is constructed from the transforms embedded in this module — the
        resulting array has correct module/standoff transforms but no
        mesh filenames.

        Args:
            interface: an :py:class:`openlifu_sdk.io.LIFUInterface`-like
                object exposing ``txdevice.get_tx_module_count()`` and
                ``txdevice.read_config(module=i)``. A fresh
                :py:class:`LIFUInterface` is constructed when omitted
                (requires ``openlifu_sdk`` to be installed).
            db: optional :py:class:`openlifu.db.Database` used to load the
                template by id (so the resulting array references the
                database's mesh files).
            arr_id: optional explicit override for the resulting array id.
            arr_name: optional explicit override for the resulting array name.
            module_transforms: optional explicit per-module 4x4 transforms
                (e.g. from a per-module calibration step) that override
                both the template and any device-config transforms.
            use_default_template: when ``True`` (default), fall back to a
                meshless embedded template if no database template can be
                found. Set to ``False`` to skip the template entirely and
                rely on identity / device-config / explicit transforms.

        Returns:
            A :class:`TransducerArray` representing the connected device.
        """
        if interface is None:
            if _SDKLIFUInterface is None:
                raise ImportError(
                    "openlifu_sdk is required to auto-create a LIFUInterface; "
                    "install it or pass an explicit `interface=` argument."
                )
            interface = _SDKLIFUInterface()

        txdevice = interface.txdevice
        count = int(txdevice.get_tx_module_count())
        if count <= 0:
            raise RuntimeError("No TX modules are connected.")

        user_configs: list[dict] = []
        for i in range(count):
            cfg = txdevice.read_config(module=i)
            if cfg is None:
                raise RuntimeError(f"Failed to read user_config from module {i}.")
            user_configs.append(json.loads(cfg.get_json_str()))

        # All connected modules must report the same frequency for the
        # template lookup to be unambiguous.
        freqs = {c.get("freq") for c in user_configs}
        if len(freqs) > 1:
            raise ValueError(
                f"Connected modules have mismatched frequencies: "
                f"{sorted(f for f in freqs if f is not None)}"
            )
        freq = next(iter(freqs)) if freqs else None

        # Resolve a template: prefer db lookup, fall back to embedded transforms.
        template: TransducerArray | None = None
        template_id: str | None = None
        if freq is not None:
            template_id = _DEFAULT_TEMPLATE_IDS.get((count, int(freq)))
        if template_id is not None:
            if db is not None:
                try:
                    loaded = db.load_transducer(template_id, convert_array=False)
                except Exception:  # -- db can raise many things; treat as "not found"
                    loaded = None
                if isinstance(loaded, TransducerArray):
                    template = loaded
            if template is None and use_default_template and template_id in _DEFAULT_TEMPLATE_DATA:
                template = _build_meshless_default_template(template_id)

        return cls.from_module_user_configs(
            user_configs,
            template=template,
            module_transforms=module_transforms,
            arr_id=arr_id,
            arr_name=arr_name,
        )

    def to_device_config(self) -> dict:
        """Serialize array-level info to a ``device`` dict for the lead module's user_config.

        Captures the array ``id``, ``name``, and ``attrs`` (mesh filenames and
        array-level ``standoff_transform``) along with per-module ``hwid`` +
        ``transform`` entries. Mesh files themselves are not stored — consumers
        must combine this with a template :class:`TransducerArray` (which
        provides the mesh files via :py:attr:`Transducer.registration_surface_filename`
        / :py:attr:`Transducer.transducer_body_filename`) when reconstructing
        the array via :py:meth:`from_module_user_configs`.
        """
        attrs_serialized: dict = {}
        for k, v in self.attrs.items():
            attrs_serialized[k] = v.tolist() if isinstance(v, np.ndarray) else v
        modules_entries: list[dict] = []
        for m in self.modules:
            entry = {
                "hwid": (m.attrs or {}).get("hwid"),
                "transform": np.array(m.transform).tolist(),
            }
            modules_entries.append(entry)
        return {
            "id": self.id,
            "name": self.name,
            "modules": modules_entries,
            "attrs": attrs_serialized,
        }

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
