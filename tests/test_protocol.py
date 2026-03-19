from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import xarray as xa
from pytest_mock import MockerFixture

from openlifu import Point, Protocol, Transducer
from openlifu.bf.focal_patterns import Wheel
from openlifu.db import Session
from openlifu.plan.protocol import OnPulseMismatchAction
from openlifu.plan.target_constraints import TargetConstraints


@pytest.fixture()
def example_protocol() -> Protocol:
    return Protocol.from_file(Path(__file__).parent/'resources/example_db/protocols/example_protocol/example_protocol.json')

@pytest.fixture()
def example_transducer() -> Transducer:
    return Transducer.from_file(Path(__file__).parent/"resources/example_db/transducers/example_transducer/example_transducer.json")

@pytest.fixture()
def example_session() -> Session:
    return Session.from_file(Path(__file__).parent/"resources/example_db/subjects/example_subject/sessions/example_session/example_session.json")

@pytest.fixture()
def example_wheel_pattern() -> Wheel:
    return Wheel(num_spokes=6)

def test_to_dict_from_dict(example_protocol: Protocol):
    proto_dict = example_protocol.to_dict()
    new_protocol = Protocol.from_dict(proto_dict)
    assert new_protocol == example_protocol

@pytest.mark.parametrize("compact_representation", [True, False])
def test_serialize_deserialize_protocol(example_protocol : Protocol, compact_representation: bool):
    assert example_protocol.from_json(example_protocol.to_json(compact_representation)) == example_protocol

def test_default_protocol():
    """Ensure it is possible to construct a default protocol"""
    Protocol()

def test_to_table(example_protocol: Protocol):
    """Ensure that the protocol can be correctly converted to a table."""
    t = example_protocol.to_table()
    assert t is not None
    assert "Category" in t.columns
    assert "Name" in t.columns
    assert "Value" in t.columns
    assert "Unit" in t.columns
    tm =t.set_index(["Category", "Name"])
    assert tm.loc["","ID"]["Value"] == example_protocol.id
    assert tm.loc["Delay Method", "Default Sound Speed"]["Value"] == 1540.0
    assert tm.loc["Delay Method", "Default Sound Speed"]["Unit"] == "m/s"

@pytest.mark.parametrize(
    "target_constraints",
    [
        [
            TargetConstraints(dim="P", units="mm", min=0.0, max=float("inf")),
        ],
        [
            TargetConstraints(dim="P", units="m", min=-0.001, max=0.0),
        ],
        [
            TargetConstraints(dim="L", units="mm", min=-100.0, max=0.0),
            TargetConstraints(dim="P", units="mm", min=-100.0, max=0.0),
            TargetConstraints(dim="S", units="mm", min=-100.0, max=-10.0),
        ]
    ]
)
def test_check_target(example_protocol: Protocol, example_session: Session, target_constraints: TargetConstraints):
    """Ensure that the target can be correctly verified."""
    example_protocol.target_constraints = target_constraints
    with pytest.raises(ValueError, match="not within bounds"):
        example_protocol.check_target(example_session.targets[0])

@pytest.mark.parametrize("on_pulse_mismatch", [
            OnPulseMismatchAction.ERROR,
            OnPulseMismatchAction.ROUND,
            OnPulseMismatchAction.ROUNDUP,
            OnPulseMismatchAction.ROUNDDOWN
        ]
    )
def test_fix_pulse_mismatch(
        example_protocol: Protocol,
        example_session: Session,
        example_wheel_pattern: Wheel,
        on_pulse_mismatch: OnPulseMismatchAction
    ):
    """Test if sequence is correctly fixed for all pulse mismatch actions."""
    logging.disable(logging.CRITICAL)

    target = example_session.targets[0]
    foci = example_wheel_pattern.get_targets(target)
    num_foci = len(foci)
    if on_pulse_mismatch is OnPulseMismatchAction.ERROR:
        with pytest.raises(ValueError, match="not a multiple of the number of foci"):
            example_protocol.fix_pulse_mismatch(on_pulse_mismatch, foci)
    else:
        example_protocol.fix_pulse_mismatch(on_pulse_mismatch, foci)
        if on_pulse_mismatch is OnPulseMismatchAction.ROUND:
            assert example_protocol.sequence.pulse_count == num_foci
        elif on_pulse_mismatch is OnPulseMismatchAction.ROUNDUP:
            assert example_protocol.sequence.pulse_count == 2*num_foci
        elif on_pulse_mismatch is OnPulseMismatchAction.ROUNDDOWN:
            assert example_protocol.sequence.pulse_count == num_foci

@pytest.mark.parametrize("use_gpu", [True, False, None])
@pytest.mark.parametrize("gpu_is_available", [True, False])
def test_calc_solution_use_gpu(
    mocker:MockerFixture,
    example_protocol:Protocol,
    example_transducer:Transducer,
    use_gpu:bool | None,
    gpu_is_available:bool,
):
    """Test that the correct value of use_gpu is passed to the simulation runner"""
    example_simulation_output = xa.Dataset(
        {
            'p_min': xa.DataArray(data=np.empty((3, 2, 3)), dims=["x", "y", "z"], attrs={'units': "Pa"}),
            'p_max': xa.DataArray(data=np.empty((3, 2, 3)),dims=["x", "y", "z"],attrs={'units': "Pa"}),
            'intensity': xa.DataArray(data=np.empty((3, 2, 3)),dims=["x", "y", "z"],attrs={'units': "W/cm^2"}),
        },
        coords={
            'x': xa.DataArray(dims=["x"], data=np.linspace(0, 1, 3), attrs={'units': "m"}),
            'y': xa.DataArray(dims=["y"], data=np.linspace(0, 1, 2), attrs={'units': "m"}),
            'z': xa.DataArray(dims=["z"], data=np.linspace(0, 1, 3), attrs={'units': "m"}),
        },
    )
    mocker.patch(
        "openlifu.plan.protocol.gpu_available",
        return_value = gpu_is_available,
    )
    run_simulation_mock = mocker.patch(
        "openlifu.plan.solution.run_simulation",
        return_value = example_simulation_output
    )
    example_protocol.calc_solution(
        target = Point(),
        transducer = example_transducer,
        simulate = True,
        scale = False,
        use_gpu=use_gpu,
    )
    args, kwargs = run_simulation_mock.call_args
    if use_gpu is None:
        assert kwargs['gpu'] == gpu_is_available
    else:
        assert kwargs['gpu'] == use_gpu
