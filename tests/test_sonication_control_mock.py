from __future__ import annotations

import numpy as np
import pytest

from openlifu.bf import Pulse, Sequence
from openlifu.geo.point import Point
from openlifu.io.LIFUInterface import LIFUInterface, LIFUInterfaceStatus
from openlifu.plan.solution import Solution


@pytest.fixture()
def example_solution() -> Solution:
    pt = Point(position=(0,0,30), units="mm")
    return Solution(
        id="solution",
        name="Solution",
        protocol_id="example_protocol",
        transducer=None,
        delays = np.zeros((1,64)),
        apodizations = np.ones((1,64)),
        pulse = Pulse(frequency=500e3, duration=2e-5),
        voltage=1.0,
        sequence = Sequence(
            pulse_interval=0.1,
            pulse_count=10,
            pulse_train_interval=1,
            pulse_train_count=1
        ),
        target=pt,
        foci=[pt],
        approved=True
    )

def test_lifuinterface_mock(example_solution:Solution):
    """Test that LIFUInterface can be used in mock mode (i.e. test_mode=True)"""
    lifu_interface = LIFUInterface(TX_test_mode=True, HV_test_mode=True)
    lifu_interface.txdevice.enum_tx7332_devices(num_devices=2)
    lifu_interface.set_solution(example_solution.to_dict())
    lifu_interface.start_sonication()
    status = lifu_interface.get_status()
    assert status == LIFUInterfaceStatus.STATUS_READY
    lifu_interface.stop_sonication()
