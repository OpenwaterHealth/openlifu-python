from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import xarray as xa

from openlifu import Protocol, Transducer
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
    example_protocol.scaling_options = {
        "balance_method": "ispta_repeats",
        "balance_metric": "mainlobe_ispta_mWcm2",
        "ordering": "minimize_repeats",
    }
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


def test_calc_solution_skips_pulse_mismatch_when_focus_order_present(
        example_protocol: Protocol,
        example_transducer: Transducer,
        example_session: Session,
        mocker
    ):
    """Test explicit focus_order allows pulse counts that are not divisible by number of foci."""
    example_protocol.focal_pattern = Wheel(num_spokes=3)
    num_foci = example_protocol.focal_pattern.num_foci()
    example_protocol.sequence.pulse_count = 5
    example_protocol.sequence.focus_order = [1, 2, 3, 1, 2]
    beamform_mock = mocker.patch.object(
        example_protocol,
        "beamform",
        return_value=(np.zeros(len(example_transducer.elements)), np.ones(len(example_transducer.elements))),
    )
    fix_pulse_mismatch_mock = mocker.patch.object(example_protocol, "fix_pulse_mismatch")

    solution, simulation_result_aggregated, solution_analysis = example_protocol.calc_solution(
        target=example_session.targets[0],
        transducer=example_transducer,
        params=xa.Dataset(),
        simulate=False,
        scale=False,
    )

    assert solution.sequence.focus_order == [1, 2, 3, 1, 2]
    assert solution.sequence.pulse_count == 5
    assert beamform_mock.call_count == num_foci
    fix_pulse_mismatch_mock.assert_not_called()
    assert simulation_result_aggregated is None
    assert solution_analysis is None
