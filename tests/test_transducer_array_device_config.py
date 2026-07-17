"""Tests for the device-config validation + template-override behavior of
:py:meth:`openlifu.xdc.TransducerArray.get_connected` / ``from_module_user_configs``.
"""
from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Sequence

import pytest

from openlifu.xdc import DeviceConfigMismatchError, TransducerArray
from openlifu.xdc.transducerarray import (
    _DEFAULT_TEMPLATE_IDS,
    _validate_device_config_against_connected,
    arrays_structurally_equal,
)


def _module_user_config(hwid: str, freq: int = 400, nx: int = 2, ny: int = 2) -> dict:
    """Minimal user_config dict accepted by ``Transducer.from_module_user_config``."""
    return {
        "hwid": hwid,
        "freq": freq,
        "module": {
            "nx": nx,
            "ny": ny,
            "pitch": 1.0,
            "kerf": 0.0,
            "units": "mm",
            "frequency": freq * 1000.0,
        },
    }


class _FakeTxDevice:
    def __init__(self, user_configs: Sequence[dict]):
        self._user_configs = list(user_configs)

    def get_tx_module_count(self) -> int:
        return len(self._user_configs)

    def read_config(self, module: int):
        cfg = self._user_configs[module]
        # Mimic the SDK's read_config object that has a .get_json_str() method.
        import json
        return SimpleNamespace(get_json_str=lambda c=cfg: json.dumps(c))


def _fake_interface(user_configs: Sequence[dict]):
    return SimpleNamespace(txdevice=_FakeTxDevice(user_configs))


# ---------- _validate_device_config_against_connected ----------

def test_validate_matches_when_count_and_hwids_agree():
    device_cfg = {
        "id": "dev",
        "modules": [{"hwid": "AAA"}, {"hwid": "BBB"}],
    }
    user_configs = [_module_user_config("AAA"), _module_user_config("BBB")]
    # Order-insensitive: should also pass with swapped hardware order.
    _validate_device_config_against_connected(device_cfg, user_configs)
    _validate_device_config_against_connected(device_cfg, list(reversed(user_configs)))


def test_validate_raises_on_count_mismatch():
    device_cfg = {"id": "dev", "modules": [{"hwid": "AAA"}, {"hwid": "BBB"}]}
    with pytest.raises(DeviceConfigMismatchError, match="lists 2 module"):
        _validate_device_config_against_connected(device_cfg, [_module_user_config("AAA")])


def test_validate_raises_on_hwid_mismatch():
    device_cfg = {"id": "dev", "modules": [{"hwid": "AAA"}, {"hwid": "BBB"}]}
    user_configs = [_module_user_config("AAA"), _module_user_config("ZZZ")]
    with pytest.raises(DeviceConfigMismatchError, match="HWIDs do not match"):
        _validate_device_config_against_connected(device_cfg, user_configs)


def test_validate_skips_hwid_check_when_device_cfg_has_no_hwids():
    # Count matches; device cfg omits hwids. Should not raise.
    device_cfg = {"id": "dev", "modules": [{}, {}]}
    user_configs = [_module_user_config("AAA"), _module_user_config("BBB")]
    _validate_device_config_against_connected(device_cfg, user_configs)


# ---------- get_connected: template selection + validation ----------

def test_get_connected_uses_device_template_when_present():
    user_configs = [
        {
            **_module_user_config("AAA", freq=400),
            "device": {
                "id": "my-array",
                "name": "My Array",
                "template": "openlifu_1x155",  # deliberately != (1, 400) default
                "modules": [{"hwid": "AAA"}],
                "attrs": {},
            },
        }
    ]
    arr = TransducerArray.get_connected(
        interface=_fake_interface(user_configs),
        db=None,
        use_default_template=True,
    )
    assert arr.id == "my-array"
    assert arr.name == "My Array"
    # The template id used to look up the meshless default should be the
    # device-specified one (1x155), not the (1, 400) default.
    assert "standoff_transform" in arr.attrs


def test_get_connected_falls_back_to_count_freq_mapping_without_device():
    user_configs = [_module_user_config("AAA", freq=400)]
    arr = TransducerArray.get_connected(
        interface=_fake_interface(user_configs),
        db=None,
        use_default_template=True,
    )
    # Default (1, 400) -> openlifu_1x400.
    assert arr.id == _DEFAULT_TEMPLATE_IDS[(1, 400)]


def test_get_connected_validates_hwid_mismatch():
    user_configs = [
        {
            **_module_user_config("AAA"),
            "device": {
                "id": "dev",
                "modules": [{"hwid": "ZZZ"}],
                "template": "openlifu_1x400",
            },
        }
    ]
    with pytest.raises(DeviceConfigMismatchError):
        TransducerArray.get_connected(
            interface=_fake_interface(user_configs),
            db=None,
            use_default_template=True,
        )


def test_get_connected_validates_count_mismatch():
    user_configs = [
        {
            **_module_user_config("AAA"),
            "device": {
                "id": "dev",
                "modules": [{"hwid": "AAA"}, {"hwid": "BBB"}],  # expects 2
                "template": "openlifu_2x400",
            },
        },
    ]
    with pytest.raises(DeviceConfigMismatchError, match="lists 2 module"):
        TransducerArray.get_connected(
            interface=_fake_interface(user_configs),
            db=None,
            use_default_template=True,
        )


# ---------- db-comparison warning ----------

class _FakeDB:
    """Stand-in for openlifu.db.Database for the db-mismatch warning test."""

    def __init__(self, stored: dict[str, TransducerArray]):
        self._stored = stored

    def get_transducer_ids(self):
        return list(self._stored)

    def load_transducer(self, transducer_id: str, convert_array: bool = True):
        return self._stored.get(transducer_id)


def test_get_connected_warns_when_db_copy_differs():
    user_configs = [
        {
            **_module_user_config("AAA", freq=400),
            "device": {
                "id": "my-array",
                "name": "My Array",
                "template": "openlifu_1x400",
                "modules": [{"hwid": "AAA"}],
                "attrs": {},
            },
        }
    ]
    # Build a stored array with a clearly different name to force a mismatch.
    stored = TransducerArray.get_connected(
        interface=_fake_interface(user_configs), db=None
    )
    stored.name = "Something Different"
    fake_db = _FakeDB({"my-array": stored})

    with pytest.warns(UserWarning, match="differs from the version stored"):
        TransducerArray.get_connected(
            interface=_fake_interface(user_configs),
            db=fake_db,
            use_default_template=True,
        )


def test_get_connected_does_not_warn_when_db_copy_matches():
    user_configs = [_module_user_config("AAA", freq=400)]
    # First call without db produces the canonical array; stash a copy.
    stored = TransducerArray.get_connected(
        interface=_fake_interface(user_configs), db=None
    )
    fake_db = _FakeDB({stored.id: stored})
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # promote any UserWarning to an error
        TransducerArray.get_connected(
            interface=_fake_interface(user_configs),
            db=fake_db,
            use_default_template=True,
        )


def test_arrays_structurally_equal_handles_path_basename_normalization():
    a = TransducerArray.get_connected(
        interface=_fake_interface([_module_user_config("AAA", freq=400)]), db=None
    )
    b = TransducerArray.get_connected(
        interface=_fake_interface([_module_user_config("AAA", freq=400)]), db=None
    )
    # Inject differently-prefixed mesh paths that share a basename.
    b.modules[0].registration_surface_filename = "/abs/path/regsurf.stl"
    a.modules[0].registration_surface_filename = "regsurf.stl"
    assert arrays_structurally_equal(a, b)
