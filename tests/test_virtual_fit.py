from __future__ import annotations

from openlifu.seg.virtual_fit import VirtualFitOptions


def test_unit_conversion():
    vfo = VirtualFitOptions(
        units="cm",
        transducer_steering_center_distance=45.,
        pitch_range=(-12,14),
        yaw_range=(-13,15),
        pitch_step = 9,
        yaw_step = 10,
        planefit_dyaw_extent = 2.3,
        steering_limits=((-10,11),(-12,13),(-14,15)),
    )
    vfo_converted = vfo.to_units("mm")
    assert vfo_converted.transducer_steering_center_distance == 10*vfo.transducer_steering_center_distance
    assert vfo_converted.planefit_dyaw_extent == 10*vfo.planefit_dyaw_extent
    assert vfo_converted.yaw_step == vfo.yaw_step
    assert vfo_converted.pitch_range == vfo.pitch_range
    assert isinstance(vfo_converted.steering_limits, tuple)
    assert all(isinstance(sl, tuple) for sl in vfo_converted.steering_limits)
    assert vfo_converted.steering_limits[2][0] == 10*vfo.steering_limits[2][0]
