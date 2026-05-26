from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from helpers import dataclasses_are_equal

from openlifu.xdc import Element, Transducer, TransducerArray
from openlifu.xdc.transducerarray import (
    get_angle_from_gap,
    get_gap_from_angle,
    get_roc_from_angle,
)


@pytest.fixture()
def example_transducer() -> Transducer:
    return Transducer.from_file(Path(__file__).parent/'resources/example_db/transducers/example_transducer/example_transducer.json')

def load_transducer_array(transducer_array_id : str) -> TransducerArray:
    """Load an example TransducerArray given the transducer ID."""
    return TransducerArray.from_file(Path(__file__).parent/f'resources/example_db/transducers/{transducer_array_id}/{transducer_array_id}.json')

@pytest.mark.parametrize("compact_representation", [True, False])
def test_serialize_deserialize_transducer(example_transducer : Transducer, compact_representation: bool):
    reconstructed_transducer = example_transducer.from_json(example_transducer.to_json(compact_representation))
    dataclasses_are_equal(example_transducer, reconstructed_transducer)

def test_get_polydata_color_options(example_transducer : Transducer):
    """Ensure that the color is set correctly on the polydata"""
    polydata_with_default_color = example_transducer.get_polydata()
    point_scalars = polydata_with_default_color.GetPointData().GetScalars()
    assert point_scalars is None

    polydata_with_given_color = example_transducer.get_polydata(facecolor=[0,1,1,0.5])
    point_scalars = polydata_with_given_color.GetPointData().GetScalars()
    assert point_scalars is not None

def test_default_transducer():
    """Ensure it is possible to construct a default transducer"""
    Transducer()

def test_convert_transform():
    transducer = Transducer(units='cm')
    transform = transducer.convert_transform(
        matrix = np.array([
            [1,0,0,2],
            [0,1,0,3],
            [0,0,1,4],
            [0,0,0,1],
        ], dtype=float),
        units = "m",
    )
    expected_transform = np.array([
        [1,0,0,200],
        [0,1,0,300],
        [0,0,1,400],
        [0,0,0,1],
    ], dtype=float)
    assert np.allclose(transform,expected_transform)

def test_get_effective_origin():
    transducer = Transducer.gen_matrix_array(nx=3, ny=2, units='cm')
    effective_origin_with_all_active = transducer.get_effective_origin(apodizations = np.ones(transducer.numelements()))
    assert np.allclose(effective_origin_with_all_active, np.zeros(3))

    rng = np.random.default_rng()
    element_index_to_turn_on = rng.integers(transducer.numelements())
    apodizations_with_just_one_element = np.zeros(transducer.numelements())
    apodizations_with_just_one_element[element_index_to_turn_on] = 0.5 # It is allowed to be a number between 0 and 1
    assert np.allclose(
        transducer.get_effective_origin(apodizations = apodizations_with_just_one_element, units = "um"),
        transducer.get_positions(units="um")[element_index_to_turn_on],
    )

def test_get_standoff_transform_in_units():
    standoff_transform_in_mm = np.array([
            [-0.1,0.9,0,20],
            [0.9,0.1,0,30],
            [0,0,1,40],
            [0,0,0,1],
    ])
    standoff_transform_in_cm = np.array([
            [-0.1,0.9,0,2],
            [0.9,0.1,0,3],
            [0,0,1,4],
            [0,0,0,1],
    ])
    transducer = Transducer(units='mm')
    transducer.standoff_transform = standoff_transform_in_mm
    assert np.allclose(
        transducer.get_standoff_transform_in_units("cm"),
        standoff_transform_in_cm,
    )

def test_read_data_types(example_transducer:Transducer):
    assert isinstance(example_transducer.standoff_transform, np.ndarray)
    if len(example_transducer.elements) > 0:
        assert isinstance(example_transducer.elements[0], Element)

@pytest.mark.parametrize(
    "transducer_array_id",
    [
        "example_transducer_array",
        "example_transducer_array2",
    ]
)
def test_transducer_array_to_transducer_data_types(transducer_array_id):
    transducer_array : TransducerArray = load_transducer_array(transducer_array_id)
    transducer = transducer_array.to_transducer()
    assert isinstance(transducer.standoff_transform, np.ndarray)
    assert not hasattr(transducer, "impulse_response")
    assert not hasattr(transducer, "impulse_dt")
    if len(transducer.elements) > 0:
        assert isinstance(transducer.elements[0], Element)


def test_transducer_calc_output_interpolates_dictionary_sensitivity():
    transducer = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity=[(100e3, 1.0), (300e3, 3.0)],
    )
    transducer.elements[0].sensitivity = 1.0
    cycles = 3
    dt = 1e-7

    output_mid = transducer.calc_output(cycles=cycles, frequency=200e3, dt=dt)
    output_low = transducer.calc_output(cycles=cycles, frequency=100e3, dt=dt)

    n_samples_mid = int(np.round(cycles / (200e3 * dt)))
    n_samples_low = int(np.round(cycles / (100e3 * dt)))
    t_mid = np.arange(n_samples_mid) * dt
    t_low = np.arange(n_samples_low) * dt
    expected_mid = 2.0 * np.sin(2 * np.pi * 200e3 * t_mid)
    expected_low = 1.0 * np.sin(2 * np.pi * 100e3 * t_low)

    np.testing.assert_allclose(output_mid[0], expected_mid)
    np.testing.assert_allclose(output_low[0], expected_low)


def test_element_calc_output_generates_signal_from_scalar_input():
    element = Element(sensitivity=2.0)
    cycles = 4
    frequency = 100e3
    dt = 1e-7
    n_samples = int(np.round(cycles / (frequency * dt)))

    output = element.calc_output(cycles=cycles, frequency=frequency, dt=dt, amplitude=3.0)
    t = np.arange(n_samples) * dt
    expected = 2.0 * 3.0 * np.sin(2 * np.pi * frequency * t)

    np.testing.assert_allclose(output, expected)


def test_element_calc_output_enforces_cycles_duration_for_generated_signal():
    element = Element(sensitivity=1.0)
    cycles = 1
    frequency = 200e3
    dt = 1e-6
    n_samples = int(np.round(cycles / (frequency * dt)))
    output = element.calc_output(cycles=cycles, frequency=frequency, dt=dt)
    t = np.arange(n_samples) * dt
    expected = np.sin(2 * np.pi * frequency * t)

    assert len(output) == n_samples
    np.testing.assert_allclose(output, expected)


def test_merge_pushes_transducer_sensitivity_into_elements():
    transducer_a = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity=[(100e3, 2.0), (300e3, 4.0)],
    )
    transducer_b = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity=[(100e3, 3.0), (300e3, 6.0)],
    )
    transducer_a.elements[0].sensitivity = 5.0
    transducer_b.elements[0].sensitivity = 7.0

    merged = Transducer.merge([transducer_a, transducer_b], merge_mismatched_sensitivity=True)

    assert merged.sensitivity == 1.0
    assert merged.elements[0].sensitivity == [(100e3, 10.0),(300e3, 20.0)]
    assert merged.elements[1].sensitivity == [(100e3, 21.0),(300e3, 42.0)]


def test_merge_rejects_mismatched_sensitivity_keys():
    transducer_a = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity=[(100e3, 2.0), (300e3, 4.0)],
    )
    transducer_b = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity=[(100e3, 2.0), (300e3, 4.0)],
    )
    transducer_a.elements[0].sensitivity = [(100e3, 5.0), (300e3, 7.0)]
    transducer_b.elements[0].sensitivity = [(100e3, 11.0), (400e3, 13.0)]

    with pytest.raises(ValueError, match="different frequency keys"):
        Transducer.merge([transducer_a, transducer_b], merge_mismatched_sensitivity=True)


@pytest.mark.parametrize(
    ("width", "dth", "roc"),
    [
        (8.0, 0.08, 25.0),
        (10.0, 0.12, 30.0),
        (12.0, 0.18, 45.0),
    ],
)
def test_concave_geometry_helpers_are_mutual_inverses(width: float, dth: float, roc: float):
    gap = get_gap_from_angle(width, dth, roc)
    recovered_roc = get_roc_from_angle(width, gap, dth)
    recovered_dth = get_angle_from_gap(width, gap, roc)
    recovered_gap = get_gap_from_angle(width, recovered_dth, roc)

    assert np.isclose(recovered_roc, roc)
    assert np.isclose(recovered_dth, dth)
    assert np.isclose(recovered_gap, gap)


def test_get_concave_cylinder_computes_gap_from_dth_and_roc_layout_spacing():
    base = Transducer.gen_matrix_array(nx=1, ny=1, units="mm")
    width = 8.0
    dth = 0.12
    roc = 25.0
    array = TransducerArray.get_concave_cylinder(
        base,
        rows=2,
        cols=1,
        width=width,
        dth=dth,
        roc=roc,
        units="mm",
    )
    merged = array.to_transducer()
    positions = merged.get_positions(units="mm")

    expected_gap = get_gap_from_angle(width, dth, roc)
    y_spacing = np.abs(positions[1, 1] - positions[0, 1])

    assert np.isclose(y_spacing, width + expected_gap)


def test_get_concave_cylinder_handles_zero_dth_without_roc():
    base = Transducer.gen_matrix_array(nx=1, ny=1, units="mm")
    width = 10.0
    gap = 2.0
    array = TransducerArray.get_concave_cylinder(
        base,
        rows=1,
        cols=2,
        width=width,
        gap=gap,
        dth=0.0,
        units="mm",
    )
    merged = array.to_transducer()
    positions = merged.get_positions(units="mm")

    x_spacing = np.abs(positions[1, 0] - positions[0, 0])
    z_values = positions[:, 2]

    assert np.isclose(x_spacing, width + gap)
    np.testing.assert_allclose(z_values, np.zeros_like(z_values))


def test_get_concave_cylinder_rejects_gap_dth_roc_together():
    base = Transducer.gen_matrix_array(nx=1, ny=1, units="mm")
    with pytest.raises(ValueError, match="cannot specify all of gap, dth, and roc"):
        TransducerArray.get_concave_cylinder(
            base,
            rows=1,
            cols=2,
            width=10.0,
            gap=1.0,
            dth=0.2,
            roc=20.0,
            units="mm",
        )


def test_transducer_calc_output_combines_frequency_dependent_sensitivities():
    transducer = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity=[(100e3, 2.0), (300e3, 4.0)],
    )
    transducer.elements[0].sensitivity = [(100e3, 5.0), (300e3, 9.0)]

    frequency = 200e3
    dt = 1e-7
    cycles = 3
    n_samples = int(np.round(cycles / (frequency * dt)))
    t = np.arange(n_samples) * dt
    expected_drive = np.sin(2 * np.pi * frequency * t)

    output = transducer.calc_output(cycles=cycles, frequency=frequency, dt=dt)

    np.testing.assert_allclose(output[0], 21.0 * expected_drive)


def test_transducer_array_to_transducer_preserves_frequency_dependent_sensitivities():
    transducer_a = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity=[(100e3, 2.0), (300e3, 4.0)],
    )
    transducer_b = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity=[(100e3, 1.0), (300e3, 3.0)],
    )
    transducer_a.elements[0].sensitivity = 5.0
    transducer_b.elements[0].sensitivity = 7.0

    array = TransducerArray.get_concave_cylinder(
        [transducer_a, transducer_b],
        rows=1,
        cols=2,
        width=10.0,
        gap=0.0,
        units="mm",
    )
    merged = array.to_transducer()

    frequency = 200e3
    dt = 1e-7
    cycles = 2
    n_samples = int(np.round(cycles / (frequency * dt)))
    t = np.arange(n_samples) * dt
    expected_drive = np.sin(2 * np.pi * frequency * t)

    output = merged.calc_output(cycles=cycles, frequency=frequency, dt=dt)

    np.testing.assert_allclose(output[0], 15.0 * expected_drive)
    np.testing.assert_allclose(output[1], 14.0 * expected_drive)


def test_element_sensitivity_from_json_is_list_of_tuples():
    """Sensitivity read from a JSON dict (list-of-lists) is converted to List[tuple[float, float]]."""
    d = {
        "index": 1,
        "position": [0.0, 0.0, 0.0],
        "orientation": [0.0, 0.0, 0.0],
        "size": [1.0, 1.0],
        "pin": 1,
        "units": "mm",
        "sensitivity": [[100e3, 1.0], [300e3, 3.0]],  # JSON encodes tuples as lists
    }
    element = Element.from_dict(d)
    assert isinstance(element.sensitivity, list)
    assert all(isinstance(pair, tuple) for pair in element.sensitivity)
    assert all(isinstance(f, float) and isinstance(v, float) for f, v in element.sensitivity)
    assert element.sensitivity == [(100e3, 1.0), (300e3, 3.0)]


def test_transducer_sensitivity_from_json_is_list_of_tuples():
    """Transducer-level sensitivity survives a to_json/from_json round-trip as List[tuple[float, float]]."""
    transducer = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity=[(100e3, 2.0), (300e3, 4.0)],
    )
    reconstructed = Transducer.from_json(transducer.to_json())
    assert isinstance(reconstructed.sensitivity, list)
    assert all(isinstance(pair, tuple) for pair in reconstructed.sensitivity)
    assert all(isinstance(f, float) and isinstance(v, float) for f, v in reconstructed.sensitivity)
    assert reconstructed.sensitivity == [(100e3, 2.0), (300e3, 4.0)]


def test_element_in_transducer_sensitivity_from_json_is_list_of_tuples():
    """Element-level sensitivity inside a Transducer survives a to_json/from_json round-trip as List[tuple[float, float]]."""
    transducer = Transducer.gen_matrix_array(nx=1, ny=1, units="mm")
    transducer.elements[0].sensitivity = [(100e3, 5.0), (300e3, 9.0)]
    reconstructed = Transducer.from_json(transducer.to_json())
    el_sensitivity = reconstructed.elements[0].sensitivity
    assert isinstance(el_sensitivity, list)
    assert all(isinstance(pair, tuple) for pair in el_sensitivity)
    assert all(isinstance(f, float) and isinstance(v, float) for f, v in el_sensitivity)
    assert el_sensitivity == [(100e3, 5.0), (300e3, 9.0)]


# ---------------------------------------------------------------------------
# Module user_config integration
# ---------------------------------------------------------------------------

def _example_module_user_config(hwid: str = "ABCD1234") -> dict:
    return {
        "sn": "EVT2B-400K-TEST",
        "hwid": hwid,
        "freq": 400,
        "module": {
            "id": f"txm_400_{hwid.lower()}",
            "name": f"TXM 400kHz ({hwid})",
            "nx": 8,
            "ny": 8,
            "pitch": 5,
            "frequency": 400000.0,
            "kerf": 0.3,
            "crosstalk_frac": 0.12,
            "crosstalk_dist": 0.00505,
            "sensitivity": [(400e3, 2800.0), (405e3, 1950.0)],
        },
        "device": {},
    }


def test_transducer_from_module_user_config():
    cfg = _example_module_user_config(hwid="HW1")
    t = Transducer.from_module_user_config(cfg)
    assert isinstance(t, Transducer)
    assert t.numelements() == 64
    assert t.id == "txm_400_hw1"
    assert t.frequency == 400000.0
    assert t.attrs["hwid"] == "HW1"
    assert t.sensitivity == [(400e3, 2800.0), (405e3, 1950.0)]


def test_transducer_from_module_user_config_missing_module():
    with pytest.raises(ValueError, match="no 'module'"):
        Transducer.from_module_user_config({"hwid": "X"})


def test_transducer_array_from_module_user_configs_bare():
    cfgs = [_example_module_user_config("HW1"), _example_module_user_config("HW2")]
    arr = TransducerArray.from_module_user_configs(cfgs)
    assert isinstance(arr, TransducerArray)
    assert len(arr.modules) == 2
    assert arr.id == "transducer_array"
    # No template/device/override -> identity transforms.
    for m in arr.modules:
        np.testing.assert_allclose(m.transform, np.eye(4))
    assert {m.attrs.get("hwid") for m in arr.modules} == {"HW1", "HW2"}


def test_transducer_array_from_module_user_configs_with_device_field():
    cfg1 = _example_module_user_config("HW1")
    cfg2 = _example_module_user_config("HW2")
    cfg1["device"] = {
        "id": "test_array",
        "name": "Test Array",
        "modules": [
            {"hwid": "HW2", "transform": np.diag([1, 1, 1, 1]).tolist()},
            {"hwid": "HW1",
             "transform": [[1, 0, 0, 10.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
        ],
        "attrs": {"registration_surface_filename": "x.obj"},
    }
    arr = TransducerArray.from_module_user_configs([cfg1, cfg2])
    assert arr.id == "test_array"
    assert arr.name == "Test Array"
    assert arr.attrs["registration_surface_filename"] == "x.obj"
    # HW1 is module index 0 but its transform is keyed by hwid -> the
    # (10, 0, 0) offset must end up on module 0, not module 1.
    np.testing.assert_allclose(arr.modules[0].transform[0, 3], 10.0)
    np.testing.assert_allclose(arr.modules[1].transform, np.eye(4))


def test_transducer_array_from_module_user_configs_with_template():
    cfgs = [_example_module_user_config("HW1"), _example_module_user_config("HW2")]
    # Build a template with mesh metadata + nontrivial transforms.
    base_template = TransducerArray.get_concave_cylinder(
        Transducer.gen_matrix_array(nx=8, ny=8, pitch=5, kerf=0.3, units="mm"),
        rows=1, cols=2, width=40, gap=0.0, units="mm",
        id="template_array", name="Template Array",
        attrs={"registration_surface_filename": "tpl.obj"},
    )
    for m in base_template.modules:
        m.registration_surface_filename = "module.surf.obj"
        m.transducer_body_filename = "module.body.obj"

    arr = TransducerArray.from_module_user_configs(cfgs, template=base_template)
    assert arr.id == "template_array"
    assert arr.attrs["registration_surface_filename"] == "tpl.obj"
    for m in arr.modules:
        assert m.registration_surface_filename == "module.surf.obj"
        assert m.transducer_body_filename == "module.body.obj"
    # Template's transforms should propagate when no device/override provided.
    np.testing.assert_allclose(arr.modules[0].transform, base_template.modules[0].transform)


def test_transducer_array_from_module_user_configs_module_transforms_override():
    cfgs = [_example_module_user_config("HW1"), _example_module_user_config("HW2")]
    cfgs[0]["device"] = {
        "id": "x",
        "name": "x",
        "modules": [
            {"hwid": "HW1", "transform": np.eye(4).tolist()},
            {"hwid": "HW2", "transform": np.eye(4).tolist()},
        ],
        "attrs": {},
    }
    overrides = [
        np.array([[1, 0, 0, 1.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float),
        np.array([[1, 0, 0, 2.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float),
    ]
    arr = TransducerArray.from_module_user_configs(cfgs, module_transforms=overrides)
    np.testing.assert_allclose(arr.modules[0].transform[0, 3], 1.0)
    np.testing.assert_allclose(arr.modules[1].transform[0, 3], 2.0)


def test_transducer_array_to_device_config_roundtrip():
    cfg1 = _example_module_user_config("HW1")
    cfg2 = _example_module_user_config("HW2")
    cfg1["device"] = {
        "id": "rt_array",
        "name": "Roundtrip Array",
        "modules": [
            {"hwid": "HW1",
             "transform": [[1, 0, 0, 5.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
            {"hwid": "HW2",
             "transform": [[1, 0, 0, -5.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
        ],
        "attrs": {"standoff_transform": np.eye(4).tolist()},
    }
    arr = TransducerArray.from_module_user_configs([cfg1, cfg2])
    device_dict = arr.to_device_config()
    assert device_dict["id"] == "rt_array"
    assert [m["hwid"] for m in device_dict["modules"]] == ["HW1", "HW2"]
    np.testing.assert_allclose(device_dict["modules"][0]["transform"][0][3], 5.0)
    np.testing.assert_allclose(device_dict["modules"][1]["transform"][0][3], -5.0)


def test_transducer_array_from_module_user_configs_empty_raises():
    with pytest.raises(ValueError, match="at least one user_config"):
        TransducerArray.from_module_user_configs([])


def test_transducer_array_from_module_user_configs_length_mismatch_raises():
    cfgs = [_example_module_user_config("HW1")]
    with pytest.raises(ValueError, match="module_transforms length"):
        TransducerArray.from_module_user_configs(cfgs, module_transforms=[np.eye(4), np.eye(4)])


def test_transducer_array_from_module_user_configs_explicit_arr_id_name_override():
    cfg1 = _example_module_user_config("HW1")
    cfg2 = _example_module_user_config("HW2")
    cfg1["device"] = {
        "id": "from_device",
        "name": "From Device",
        "modules": [
            {"hwid": "HW1", "transform": np.eye(4).tolist()},
            {"hwid": "HW2", "transform": np.eye(4).tolist()},
        ],
        "attrs": {},
    }
    arr = TransducerArray.from_module_user_configs(
        [cfg1, cfg2], arr_id="explicit_id", arr_name="Explicit Name",
    )
    assert arr.id == "explicit_id"
    assert arr.name == "Explicit Name"


def test_transducer_array_from_module_user_configs_arr_id_falls_through():
    """No explicit arg / no device id -> template id is used."""
    cfgs = [_example_module_user_config("HW1"), _example_module_user_config("HW2")]
    template = TransducerArray.get_concave_cylinder(
        Transducer.gen_matrix_array(nx=8, ny=8, pitch=5, kerf=0.3, units="mm"),
        rows=1, cols=2, width=40, gap=0.0, units="mm",
        id="tpl_id", name="Tpl Name",
    )
    arr = TransducerArray.from_module_user_configs(cfgs, template=template)
    assert arr.id == "tpl_id"
    assert arr.name == "Tpl Name"


class _FakeUserConfig:
    def __init__(self, payload: dict):
        self._payload = payload

    def get_json_str(self) -> str:
        return json.dumps(self._payload)


class _FakeTxDevice:
    def __init__(self, configs: list[dict]):
        self._configs = configs

    def get_tx_module_count(self) -> int:
        return len(self._configs)

    def read_config(self, module: int = 0):
        return _FakeUserConfig(self._configs[module])


class _FakeInterface:
    def __init__(self, configs: list[dict]):
        self.txdevice = _FakeTxDevice(configs)


def _user_cfg_for_freq(hwid: str, freq_khz: int) -> dict:
    cfg = _example_module_user_config(hwid)
    cfg["freq"] = freq_khz
    cfg["module"]["frequency"] = float(freq_khz) * 1e3
    return cfg


def test_transducer_array_get_connected_no_db_2x400():
    """No db -> meshless embedded template, but transforms applied."""
    interface = _FakeInterface([_user_cfg_for_freq("HW1", 400), _user_cfg_for_freq("HW2", 400)])
    arr = TransducerArray.get_connected(interface=interface)
    assert arr.id == "openlifu_2x400"
    assert len(arr.modules) == 2
    # Transforms from the embedded default should be applied (not identity).
    np.testing.assert_allclose(arr.modules[0].transform[0, 3], 25.84571998794554)
    np.testing.assert_allclose(arr.modules[1].transform[0, 3], -25.84571998794554)
    # Standoff transform present, no mesh filenames in the meshless fallback.
    assert "standoff_transform" in arr.attrs
    assert arr.attrs.get("registration_surface_filename") is None


def test_transducer_array_get_connected_no_db_1x400():
    interface = _FakeInterface([_user_cfg_for_freq("HW1", 400)])
    arr = TransducerArray.get_connected(interface=interface)
    assert arr.id == "openlifu_1x400"
    assert len(arr.modules) == 1
    np.testing.assert_allclose(arr.modules[0].transform, np.eye(4))


def test_transducer_array_get_connected_mismatched_freqs_raises():
    interface = _FakeInterface([_user_cfg_for_freq("HW1", 400), _user_cfg_for_freq("HW2", 155)])
    with pytest.raises(ValueError, match="mismatched frequencies"):
        TransducerArray.get_connected(interface=interface)


def test_transducer_array_get_connected_with_db_uses_template_meshes():
    """A db that returns a TransducerArray template should provide mesh filenames."""
    template = TransducerArray.get_concave_cylinder(
        Transducer.gen_matrix_array(nx=8, ny=8, pitch=5, kerf=0.3, units="mm"),
        rows=1, cols=2, width=40, gap=0.0, units="mm",
        id="openlifu_2x400", name="OpenLIFU 2x 400kHz",
        attrs={
            "registration_surface_filename": "fake.surf.obj",
            "transducer_body_filename": "fake.body.obj",
        },
    )
    for m in template.modules:
        m.registration_surface_filename = "module.surf.obj"
        m.transducer_body_filename = "module.body.obj"

    class _FakeDb:
        def load_transducer(self, transducer_id, convert_array=True):
            assert transducer_id == "openlifu_2x400"
            assert convert_array is False
            return template

    interface = _FakeInterface([_user_cfg_for_freq("HW1", 400), _user_cfg_for_freq("HW2", 400)])
    arr = TransducerArray.get_connected(interface=interface, db=_FakeDb())
    assert arr.attrs["registration_surface_filename"] == "fake.surf.obj"
    assert arr.modules[0].registration_surface_filename == "module.surf.obj"


def test_transducer_array_get_connected_explicit_overrides():
    interface = _FakeInterface([_user_cfg_for_freq("HW1", 400), _user_cfg_for_freq("HW2", 400)])
    overrides = [
        np.array([[1, 0, 0, 1.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float),
        np.array([[1, 0, 0, 2.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float),
    ]
    arr = TransducerArray.get_connected(
        interface=interface,
        arr_id="my_array", arr_name="My Array",
        module_transforms=overrides,
    )
    assert arr.id == "my_array"
    assert arr.name == "My Array"
    np.testing.assert_allclose(arr.modules[0].transform[0, 3], 1.0)
    np.testing.assert_allclose(arr.modules[1].transform[0, 3], 2.0)


def test_transducer_array_get_connected_no_modules_raises():
    interface = _FakeInterface([])
    with pytest.raises(RuntimeError):
        TransducerArray.get_connected(interface=interface)


def test_transducer_array_get_connected_unknown_combo_no_template():
    """Unknown (n, freq) combo -> no template, identity transforms."""
    interface = _FakeInterface([
        _user_cfg_for_freq("HW1", 250),
        _user_cfg_for_freq("HW2", 250),
        _user_cfg_for_freq("HW3", 250),
    ])
    arr = TransducerArray.get_connected(interface=interface)
    # No template id matched (3, 250) -> bare default
    assert arr.id == "transducer_array"
    for m in arr.modules:
        np.testing.assert_allclose(m.transform, np.eye(4))
