from __future__ import annotations

import sys

import pytest
import xarray

import openlifu


@pytest.mark.skipif(
    sys.platform == 'darwin',
    reason=(
        "This test is skipped on macOS due to some unresolved known issues with kwave."
        " See https://github.com/OpenwaterHealth/OpenLIFU-python/pull/259#issuecomment-2923230777"
    )
)
def test_run_simulation_runs():
    """Test that run_simulation can run and outputs something of the correct type."""

    transducer = openlifu.Transducer.gen_matrix_array(nx=2, ny=2, pitch=2, kerf=.5, units="mm", sensitivity=1e5)
    dt = 2e-7
    sim_setup = openlifu.SimSetup(
        dt=dt,
        t_end=3*dt, # only 3 time steps. we just want to test that the simulation code can run
        x_extent=(-10,10),
        y_extent=(-10,10),
        z_extent=(-2,10),
    )
    pulse = openlifu.Pulse(frequency=400e3, duration=1/400e3)
    protocol = openlifu.Protocol(
        pulse=pulse,
        sequence=openlifu.Sequence(),
        sim_setup=sim_setup
    )
    coords = sim_setup.get_coords()
    default_seg_method = protocol.seg_method
    params = default_seg_method.ref_params(coords)
    delays, apod = protocol.beamform(arr=transducer, target=openlifu.Point(position=(0,0,50)), params=params)
    delays[:] = 0.0
    apod[:] = 1.0


    dataset = openlifu.sim.run_simulation(
        arr=transducer,
        params=params,
        delays=delays,
        apod= apod,
        freq = pulse.frequency,
        cycles = 1,
        dt=protocol.sim_setup.dt,
        t_end=protocol.sim_setup.t_end,
        amplitude = 1,
        gpu = False,
    )

    assert isinstance(dataset, xarray.Dataset)
    assert 'p_max' in dataset
    assert 'p_min' in dataset
    assert 'intensity' in dataset
