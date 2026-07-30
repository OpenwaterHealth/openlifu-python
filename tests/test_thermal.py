from __future__ import annotations

import math

import numpy as np
import pytest
import xarray as xa

from openlifu.bf import Pulse, Sequence
from openlifu.geo.point import Point
from openlifu.plan import Solution
from openlifu.seg import Material
from openlifu.seg.material import WATER
from openlifu.sim.thermal import (
    compute_heat_source,
    db2neper,
    generate_pulse_events,
    run_thermal_simulation,
)
from openlifu.xdc import Transducer
from openlifu.xdc.element import Element


def _uniform_params(shape=(5, 5, 5), spacing_m=1e-3, material: Material = WATER) -> xa.Dataset:
    """Return a params Dataset with uniform material properties on a regular m-grid."""
    coords = xa.Coordinates({
        'x': xa.DataArray(np.arange(shape[0]) * spacing_m, dims=['x'], attrs={'units': 'm'}),
        'y': xa.DataArray(np.arange(shape[1]) * spacing_m, dims=['y'], attrs={'units': 'm'}),
        'z': xa.DataArray(np.arange(shape[2]) * spacing_m, dims=['z'], attrs={'units': 'm'}),
    })
    ones = xa.DataArray(np.ones(shape), coords=coords)
    return xa.Dataset({
        'density': ones * material.density,
        'sound_speed': ones * material.sound_speed,
        'attenuation': ones * material.attenuation,
        'specific_heat': ones * material.specific_heat,
        'thermal_conductivity': ones * material.thermal_conductivity,
    })


def test_db2neper_matches_kwave_reference():
    """Reference value: 1 dB/(MHz cm) with y=1 → ~1.152e-5 Np/(rad/s m)."""
    val = db2neper(1.0, y=1.0)
    expected = 100.0 * 1.0 * (1e-6 / (2.0 * math.pi)) / (20.0 * np.log10(np.exp(1)))
    assert np.isclose(val, expected)


def test_compute_heat_source_uniform_medium():
    """Q = alpha_np * p^2 / (rho * c) should match a hand computation."""
    tissue = Material(
        name='tissue', sound_speed=1500.0, density=1000.0, attenuation=0.5,
        specific_heat=3600.0, thermal_conductivity=0.5,
    )
    params = _uniform_params(shape=(3, 3, 3), material=tissue)
    p = 1e6 * np.ones((3, 3, 3))  # 1 MPa
    Q = compute_heat_source(params, p, freq=500e3, alpha_power=0.9)

    alpha_np = db2neper(0.5, y=0.9) * (2 * np.pi * 500e3) ** 0.9
    expected = alpha_np * (1e6) ** 2 / (1000.0 * 1500.0)
    assert np.allclose(Q, expected)


def test_generate_pulse_events_default_order():
    """Default order cycles through foci 0..nfoc-1."""
    events = generate_pulse_events(
        pulse_count=4,
        pulse_interval=1.0,
        pulse_train_count=2,
        pulse_train_interval=5.0,
        pulse_duration=0.1,
        num_foci=2,
    )
    # 2 trains x 4 pulses x 2 events per pulse = 16
    assert len(events) == 16
    # Alternating focus 0, 1, 0, 1 within train
    on_events = [e for e in events if e[1] >= 0]
    focus_seq = [e[1] for e in on_events]
    assert focus_seq == [0, 1, 0, 1, 0, 1, 0, 1]
    # Every "on" is followed by an "off" at t+pulse_duration
    for i in range(0, len(events), 2):
        assert events[i + 1][0] == pytest.approx(events[i][0] + 0.1)
        assert events[i + 1][1] == -1


def test_generate_pulse_events_custom_order():
    """`order` is 1-indexed and overrides the default cycling."""
    events = generate_pulse_events(
        pulse_count=3,
        pulse_interval=1.0,
        pulse_train_count=1,
        pulse_train_interval=0.0,
        pulse_duration=0.1,
        num_foci=3,
        order=[3, 1, 2],
    )
    on = [e[1] for e in events if e[1] >= 0]
    assert on == [2, 0, 1]  # converted 1→0-indexed


def test_generate_pulse_events_zero_train_interval_backtoback():
    """pulse_train_interval == 0 means back-to-back trains."""
    events = generate_pulse_events(
        pulse_count=2,
        pulse_interval=1.0,
        pulse_train_count=2,
        pulse_train_interval=0.0,
        pulse_duration=0.5,
        num_foci=1,
    )
    on_times = [e[0] for e in events if e[1] >= 0]
    # First train: t=0, 1. Second train: t=2, 3.
    assert on_times == pytest.approx([0.0, 1.0, 2.0, 3.0])


def test_generate_pulse_events_rejects_bad_order():
    with pytest.raises(ValueError, match="1-indexed"):
        generate_pulse_events(
            pulse_count=1, pulse_interval=1.0, pulse_train_count=1,
            pulse_train_interval=0.0, pulse_duration=0.1, num_foci=2,
            order=[0, 3],
        )


def test_run_thermal_simulation_no_heat_gives_zero():
    """With Q=0 and T_init=0 the temperature stays at 0."""
    params = _uniform_params(shape=(4, 4, 4))
    Q = xa.DataArray(
        np.zeros((1, 4, 4, 4)),
        dims=['focal_point_index', 'x', 'y', 'z'],
        coords={
            'focal_point_index': [0],
            'x': params.coords['x'],
            'y': params.coords['y'],
            'z': params.coords['z'],
        },
    )
    events = [(0.0, 0), (1.0, -1)]
    ds = run_thermal_simulation(
        heat_sources=Q, params=params, events=events,
        dt=0.5, t_end=2.0, oversample=1, T0=37.0,
        mode='superpose', pulse_duration=1.0,
    )
    assert np.allclose(ds['temperature_rise'].data, 0.0)
    assert ds['Tmax'].item() == pytest.approx(37.0)
    assert ds.attrs['T0'] == 37.0


def test_run_thermal_simulation_uniform_heating_rate():
    """With uniform Q and Neumann BCs the temperature rises spatially uniformly
    at rate ``Q/(rho * c_p)`` while the source is on.
    """
    tissue = Material(
        name='tissue', sound_speed=1500.0, density=1000.0, attenuation=0.5,
        specific_heat=3600.0, thermal_conductivity=0.5,
    )
    params = _uniform_params(shape=(4, 4, 4), material=tissue)
    Q_val = 1e5  # W/m^3
    Q_data = Q_val * np.ones((1, 4, 4, 4))
    Q = xa.DataArray(
        Q_data,
        dims=['focal_point_index', 'x', 'y', 'z'],
        coords={
            'focal_point_index': [0],
            'x': params.coords['x'],
            'y': params.coords['y'],
            'z': params.coords['z'],
        },
    )
    # Single continuous pulse of 1s duration on focus 0.
    events = [(0.0, 0), (1.0, -1)]
    ds = run_thermal_simulation(
        heat_sources=Q, params=params, events=events,
        dt=0.1, t_end=1.0, oversample=1, T0=0.0,
        mode='direct', pulse_duration=1.0,
    )
    T = ds['temperature_rise'].data
    # Diffusion across uniform T contributes no divergence, so T rises linearly.
    expected_rate = Q_val / (tissue.density * tissue.specific_heat)
    # Check midpoint value
    t_mid_idx = 5  # t = 0.5s
    expected = expected_rate * (t_mid_idx * 0.1)
    assert np.allclose(T[t_mid_idx], expected, rtol=1e-4)
    # Field is spatially uniform (nothing to diffuse)
    assert T[t_mid_idx].std() < 1e-6


def test_run_thermal_simulation_cools_after_pulse():
    """After the heat source turns off the peak temperature must decrease."""
    tissue = Material(
        name='tissue', sound_speed=1500.0, density=1000.0, attenuation=0.5,
        specific_heat=3600.0, thermal_conductivity=0.5,
    )
    params = _uniform_params(shape=(9, 9, 9), spacing_m=1e-3, material=tissue)
    # Localized Gaussian-ish heat source in the center voxel only.
    Q_data = np.zeros((1, 9, 9, 9))
    Q_data[0, 4, 4, 4] = 5e8  # W/m^3, concentrated
    Q = xa.DataArray(
        Q_data,
        dims=['focal_point_index', 'x', 'y', 'z'],
        coords={
            'focal_point_index': [0],
            'x': params.coords['x'],
            'y': params.coords['y'],
            'z': params.coords['z'],
        },
    )
    events = [(0.0, 0), (0.5, -1)]  # 0.5s pulse
    ds = run_thermal_simulation(
        heat_sources=Q, params=params, events=events,
        dt=0.05, t_end=3.0, oversample=1, T0=0.0,
        mode='direct', pulse_duration=0.5,
    )
    T = ds['temperature_rise'].data
    peak = T.reshape(T.shape[0], -1).max(axis=1)
    # Peak grows while source is on and decays after
    on_idx = round(0.5 / 0.05)
    assert peak[on_idx] > peak[0]
    assert peak[-1] < peak[on_idx]


def test_direct_and_superpose_agree_single_pulse():
    """Modes must agree for a single-pulse sequence (a base sanity check that
    the two solvers implement the same physics).
    """
    tissue = Material(
        name='tissue', sound_speed=1500.0, density=1000.0, attenuation=0.3,
        specific_heat=3600.0, thermal_conductivity=0.5,
    )
    params = _uniform_params(shape=(7, 7, 7), spacing_m=1e-3, material=tissue)
    Q_data = np.zeros((1, 7, 7, 7))
    Q_data[0, 3, 3, 3] = 2e8
    Q = xa.DataArray(
        Q_data,
        dims=['focal_point_index', 'x', 'y', 'z'],
        coords={
            'focal_point_index': [0],
            'x': params.coords['x'],
            'y': params.coords['y'],
            'z': params.coords['z'],
        },
    )
    events = [(0.0, 0), (0.2, -1)]
    kw = {
        "heat_sources": Q, "params": params, "events": events,
        "dt": 0.05, "t_end": 1.0, "oversample": 1, "T0": 0.0, "pulse_duration": 0.2,
    }
    ds_direct = run_thermal_simulation(mode='direct', **kw)
    ds_super = run_thermal_simulation(mode='superpose', **kw)
    T_d = ds_direct['temperature_rise'].data
    T_s = ds_super['temperature_rise'].data
    # These are single-precision FE integrations of the same equations, so
    # equality up to a modest tolerance is expected.
    assert np.allclose(T_d, T_s, atol=5e-5, rtol=1e-3)


def test_direct_and_superpose_agree_multi_pulse_single_focus():
    """For a repeated pulse train on a single focus the superposition result
    must reproduce the event-driven direct simulation.
    """
    tissue = Material(
        name='tissue', sound_speed=1500.0, density=1000.0, attenuation=0.5,
        specific_heat=3600.0, thermal_conductivity=0.5,
    )
    params = _uniform_params(shape=(7, 7, 7), spacing_m=1e-3, material=tissue)
    Q_data = np.zeros((1, 7, 7, 7))
    Q_data[0, 3, 3, 3] = 1e8
    Q = xa.DataArray(
        Q_data,
        dims=['focal_point_index', 'x', 'y', 'z'],
        coords={
            'focal_point_index': [0],
            'x': params.coords['x'],
            'y': params.coords['y'],
            'z': params.coords['z'],
        },
    )
    events = generate_pulse_events(
        pulse_count=3, pulse_interval=0.5,
        pulse_train_count=1, pulse_train_interval=0.0,
        pulse_duration=0.1, num_foci=1,
    )
    kw = {
        "heat_sources": Q, "params": params, "events": events,
        "dt": 0.05, "t_end": 2.0, "oversample": 1, "T0": 0.0, "pulse_duration": 0.1,
    }
    ds_direct = run_thermal_simulation(mode='direct', **kw)
    ds_super = run_thermal_simulation(mode='superpose', **kw)
    assert np.allclose(
        ds_direct['temperature_rise'].data,
        ds_super['temperature_rise'].data,
        atol=1e-4, rtol=5e-3,
    )


def test_direct_and_superpose_agree_multi_focus():
    """Two-focus alternating sequence: superposition matches direct."""
    tissue = Material(
        name='tissue', sound_speed=1500.0, density=1000.0, attenuation=0.5,
        specific_heat=3600.0, thermal_conductivity=0.5,
    )
    params = _uniform_params(shape=(7, 7, 7), spacing_m=1e-3, material=tissue)
    Q_data = np.zeros((2, 7, 7, 7))
    Q_data[0, 2, 3, 3] = 1e8
    Q_data[1, 4, 3, 3] = 1e8
    Q = xa.DataArray(
        Q_data,
        dims=['focal_point_index', 'x', 'y', 'z'],
        coords={
            'focal_point_index': [0, 1],
            'x': params.coords['x'],
            'y': params.coords['y'],
            'z': params.coords['z'],
        },
    )
    events = generate_pulse_events(
        pulse_count=4, pulse_interval=0.4,
        pulse_train_count=1, pulse_train_interval=0.0,
        pulse_duration=0.1, num_foci=2,
    )
    kw = {
        "heat_sources": Q, "params": params, "events": events,
        "dt": 0.05, "t_end": 2.0, "oversample": 1, "T0": 0.0, "pulse_duration": 0.1,
    }
    ds_direct = run_thermal_simulation(mode='direct', **kw)
    ds_super = run_thermal_simulation(mode='superpose', **kw)
    assert np.allclose(
        ds_direct['temperature_rise'].data,
        ds_super['temperature_rise'].data,
        atol=1e-4, rtol=5e-3,
    )


def _make_solution_with_pressure(shape=(5, 5, 5), spacing_m=1e-3, p_amp=1e6) -> Solution:
    """Build a minimal Solution with a synthetic acoustic simulation_result."""
    coords = xa.Coordinates({
        'x': xa.DataArray(np.arange(shape[0]) * spacing_m, dims=['x'], attrs={'units': 'm'}),
        'y': xa.DataArray(np.arange(shape[1]) * spacing_m, dims=['y'], attrs={'units': 'm'}),
        'z': xa.DataArray(np.arange(shape[2]) * spacing_m, dims=['z'], attrs={'units': 'm'}),
        'focal_point_index': [0],
    })
    p_field = np.zeros((1, *shape))
    p_field[0, shape[0] // 2, shape[1] // 2, shape[2] // 2] = p_amp
    sim = xa.Dataset({
        'p_min': xa.DataArray(
            p_field,
            dims=['focal_point_index', 'x', 'y', 'z'],
            coords=coords,
            attrs={'units': 'Pa'},
        ),
    })
    return Solution(
        id='sol_thermal',
        transducer=Transducer(
            id='t1', elements=[Element(index=1, position=[0, 0, 0], units='m')],
            frequency=500e3, units='m',
        ),
        delays=np.array([[0.0]]),
        apodizations=np.array([[1.0]]),
        pulse=Pulse(frequency=500e3, duration=0.1, amplitude=1.0),
        sequence=Sequence(pulse_interval=0.5, pulse_count=2, pulse_train_interval=1.0, pulse_train_count=1),
        foci=[Point(id='f0', position=(0.0, 0.0, spacing_m * (shape[2] // 2)), units='m')],
        simulation_result=sim,
    )


def test_solution_simulate_thermal_uses_simulation_result():
    """`Solution.simulate_thermal` picks up self.simulation_result when
    ``acoustic_result`` is not supplied and returns a Dataset with the
    expected coordinates and metadata.
    """
    solution = _make_solution_with_pressure(shape=(5, 5, 5))
    params = _uniform_params(shape=(5, 5, 5), spacing_m=1e-3)
    ds = solution.simulate_thermal(params=params, dt=0.05, t_end=0.5, T0=37.0)
    assert 'temperature_rise' in ds
    assert ds['temperature_rise'].dims == ('t', 'x', 'y', 'z')
    assert ds['temperature_rise'].sizes['t'] == round(0.5 / 0.05) + 1
    assert ds.attrs['T0'] == 37.0
    # Peak temperature must at least equal T0 (delta-T is >= 0 with a positive source)
    assert ds['Tmax'].item() >= 37.0


def test_solution_simulate_thermal_missing_result_errors():
    """simulate_thermal raises if the Solution has no simulation_result and none is passed."""
    solution = _make_solution_with_pressure()
    solution.simulation_result = xa.Dataset()
    params = _uniform_params()
    with pytest.raises(ValueError, match="No acoustic_result provided"):
        solution.simulate_thermal(params=params)


def test_solution_simulate_thermal_missing_pressure_field_errors():
    solution = _make_solution_with_pressure()
    params = _uniform_params()
    with pytest.raises(ValueError, match="not found in acoustic_result"):
        solution.simulate_thermal(params=params, pressure_field='not_a_field')


def _absorbing_tissue() -> Material:
    """A tissue-like material with realistic (nonzero) attenuation."""
    return Material(
        name='absorbing_tissue', sound_speed=1500.0, density=1000.0, attenuation=0.5,
        specific_heat=3600.0, thermal_conductivity=0.5,
    )


def _localized_heat_source(shape=(9, 9, 9), spacing_m=1e-3, q_center=1e6) -> xa.DataArray:
    """A single-focus heat source concentrated at the grid center."""
    Q_data = np.zeros((1, *shape))
    Q_data[0, shape[0] // 2, shape[1] // 2, shape[2] // 2] = q_center
    return xa.DataArray(
        Q_data,
        dims=['focal_point_index', 'x', 'y', 'z'],
        coords={
            'focal_point_index': [0],
            'x': xa.DataArray(np.arange(shape[0]) * spacing_m, dims=['x'], attrs={'units': 'm'}),
            'y': xa.DataArray(np.arange(shape[1]) * spacing_m, dims=['y'], attrs={'units': 'm'}),
            'z': xa.DataArray(np.arange(shape[2]) * spacing_m, dims=['z'], attrs={'units': 'm'}),
        },
    )


def test_impulse_mode_gives_expected_per_pulse_dt():
    """A single pulse in impulse mode deposits ``Q * pulse_duration / rho_cp``
    at the source voxel, matching the exact formula.
    """
    tissue = _absorbing_tissue()
    params = _uniform_params(shape=(9, 9, 9), spacing_m=1e-3, material=tissue)
    Q_val = 1e7  # W/m^3
    Q = _localized_heat_source(shape=(9, 9, 9), q_center=Q_val)
    pulse_dur = 5e-3  # 5 ms

    # A single pulse at t=0. Under the "sample before events" convention used
    # by all three modes, T[j=0] = 0 (nothing has happened yet). The impulse
    # deposit only appears in T[j >= 1].
    events = [(0.0, 0)]
    ds = run_thermal_simulation(
        heat_sources=Q, params=params, events=events,
        dt=0.05, t_end=1.0, T0=0.0,
        mode='impulse', pulse_duration=pulse_dur,
    )
    T = ds['temperature_rise'].data
    center = T[:, 4, 4, 4]
    expected_deposit = Q_val * pulse_dur / (tissue.density * tissue.specific_heat)
    # Sample j=0 (pre-impulse) is zero.
    assert center[0] == 0.0
    # Sample j=1 has one full dt of diffusion after the impulse, but the
    # diffusion length is tiny compared to dx=1 mm, so nearly all the deposit
    # is still at the center.
    assert center[1] > 0.9 * expected_deposit
    assert center[1] < expected_deposit * 1.01
    # Tmax is the peak temperature-rise + T0. With T0=0 the peak is just under
    # the ideal instantaneous deposit due to finite dt of diffusion.
    assert ds['Tmax'].item() > 0.9 * expected_deposit
    assert ds['Tmax'].item() <= expected_deposit + 1e-6


def test_impulse_mode_accumulates_across_pulses():
    """Repeated pulses in impulse mode accumulate temperature at the focus."""
    tissue = _absorbing_tissue()
    params = _uniform_params(shape=(9, 9, 9), spacing_m=1e-3, material=tissue)
    Q = _localized_heat_source(shape=(9, 9, 9), q_center=1e7)
    pulse_dur = 5e-3

    events = generate_pulse_events(
        pulse_count=10, pulse_interval=0.2,
        pulse_train_count=1, pulse_train_interval=0.0,
        pulse_duration=pulse_dur, num_foci=1,
    )
    ds = run_thermal_simulation(
        heat_sources=Q, params=params, events=events,
        dt=0.05, t_end=2.5, T0=0.0,
        mode='impulse', pulse_duration=pulse_dur,
    )
    center = ds['temperature_rise'].data[:, 4, 4, 4]
    # After all pulses have deposited energy the peak must be larger than
    # after just one pulse.
    single_pulse_deposit = 1e7 * pulse_dur / (tissue.density * tissue.specific_heat)
    assert center.max() > 5 * single_pulse_deposit
    # And the peak must be bounded above by the deposit-count times the
    # per-pulse deposit (no gain from diffusion).
    assert center.max() < 10 * single_pulse_deposit + 1e-6


def test_impulse_and_superpose_agree_for_short_pulses():
    """When ``pulse_duration`` is much shorter than the thermal diffusion
    timescale, the impulse limit and the exact ON-then-OFF response give
    essentially the same trace at every output sample (both modes now use
    the same sample-before-events convention).
    """
    tissue = _absorbing_tissue()
    params = _uniform_params(shape=(7, 7, 7), spacing_m=1e-3, material=tissue)
    Q = _localized_heat_source(shape=(7, 7, 7), q_center=5e6)
    pulse_dur = 1e-3  # 1 ms - short compared to dt=50 ms.

    events = generate_pulse_events(
        pulse_count=5, pulse_interval=0.25,
        pulse_train_count=1, pulse_train_interval=0.0,
        pulse_duration=pulse_dur, num_foci=1,
    )
    kw = {
        "heat_sources": Q, "params": params, "events": events,
        "dt": 0.05, "t_end": 2.0, "T0": 0.0, "pulse_duration": pulse_dur,
    }
    ds_imp = run_thermal_simulation(mode='impulse', **kw)
    ds_sup = run_thermal_simulation(mode='superpose', **kw)
    # The pulse (1 ms) is 50x shorter than dt (50 ms) and much shorter than
    # the diffusion timescale, so both traces should agree to a few percent
    # of the per-pulse deposit at every sample.
    tolerance = 5e-2 * (5e6 * pulse_dur / (tissue.density * tissue.specific_heat))
    max_diff = np.max(np.abs(ds_imp['temperature_rise'].data - ds_sup['temperature_rise'].data))
    assert max_diff < tolerance


def test_zero_attenuation_warns(caplog):
    """The zero-attenuation trap must produce a diagnostic warning."""
    params = _uniform_params(shape=(5, 5, 5), material=WATER)  # attenuation=0
    Q = xa.DataArray(
        np.zeros((1, 5, 5, 5)),
        dims=['focal_point_index', 'x', 'y', 'z'],
        coords={
            'focal_point_index': [0],
            'x': params.coords['x'],
            'y': params.coords['y'],
            'z': params.coords['z'],
        },
    )
    with caplog.at_level('WARNING', logger='openlifu.sim.thermal'):
        run_thermal_simulation(
            heat_sources=Q, params=params, events=[(0.0, 0)],
            dt=0.1, t_end=0.5, mode='impulse', pulse_duration=1e-3,
        )
    assert any(
        'All heat sources are zero' in rec.message
        and "attenuation" in rec.message
        for rec in caplog.records
    )


def test_solution_simulate_thermal_defaults_to_impulse():
    """The default mode on the Solution API is 'impulse'."""
    solution = _make_solution_with_pressure(shape=(5, 5, 5))
    params = _uniform_params(shape=(5, 5, 5), spacing_m=1e-3)
    ds = solution.simulate_thermal(params=params, dt=0.05, t_end=0.5, T0=37.0)
    assert ds.attrs['mode'] == 'impulse'


def test_impulse_captures_multiple_events_per_output_step():
    """Multiple pulse events falling within a single output ``dt`` must all
    deposit their energy. Energy conservation is total: the volume-integrated
    temperature rise at the last sample equals ``n_pulses * total_deposit``.
    """
    tissue = _absorbing_tissue()
    params = _uniform_params(shape=(7, 7, 7), spacing_m=1e-3, material=tissue)
    Q = _localized_heat_source(shape=(7, 7, 7), q_center=5e6)
    pulse_dur = 1e-3

    # 5 pulses within a single output step: pulse_interval = dt/5.
    dt = 0.5
    events = generate_pulse_events(
        pulse_count=5, pulse_interval=dt / 5,   # sub-dt pulses
        pulse_train_count=1, pulse_train_interval=0.0,
        pulse_duration=pulse_dur, num_foci=1,
    )
    ds = run_thermal_simulation(
        heat_sources=Q, params=params, events=events,
        dt=dt, t_end=1.0, T0=0.0,
        mode='impulse', pulse_duration=pulse_dur,
    )
    total_energy = ds['temperature_rise'].data[-1].sum()
    # Each pulse deposits ``deposit`` at one voxel, and Neumann BCs conserve
    # the volume integral of T. Five pulses -> volume-integral is 5 * deposit.
    per_pulse = 5e6 * pulse_dur / (tissue.density * tissue.specific_heat)
    assert total_energy == pytest.approx(5 * per_pulse, rel=1e-4)


def test_impulse_captures_off_grid_events_exactly():
    """Events at arbitrary sub-``dt`` times must all be captured exactly, with
    total volume-integrated energy independent of how events line up with the
    output-``dt`` grid.
    """
    tissue = _absorbing_tissue()
    params = _uniform_params(shape=(7, 7, 7), spacing_m=1e-3, material=tissue)
    Q = _localized_heat_source(shape=(7, 7, 7), q_center=5e6)
    pulse_dur = 1e-3
    dt = 0.1

    # Deliberately off-grid event times: 0.017, 0.163, 0.284, 0.451, 0.752.
    # None of these snap cleanly to any output sample time k*dt = 0.0, 0.1, ...
    off_grid_events = [(0.017, 0), (0.163, 0), (0.284, 0), (0.451, 0), (0.752, 0)]
    ds = run_thermal_simulation(
        heat_sources=Q, params=params, events=off_grid_events,
        dt=dt, t_end=2.0, T0=0.0,
        mode='impulse', pulse_duration=pulse_dur,
    )
    total_energy = ds['temperature_rise'].data[-1].sum()
    per_pulse = 5e6 * pulse_dur / (tissue.density * tissue.specific_heat)
    assert total_energy == pytest.approx(5 * per_pulse, rel=1e-4)


def test_impulse_floating_point_alignment_of_events():
    """Regression test for the floating-point drift bug in event-time
    accounting: pulses at exactly ``k * dt`` (which suffer bit-level rounding
    error under ``t += dt`` accumulation) must land in the correct output
    step. ``T`` at sample ``k`` must be *before* the pulse-start event at
    ``t = k * dt`` has taken effect.
    """
    tissue = _absorbing_tissue()
    params = _uniform_params(shape=(5, 5, 5), spacing_m=1e-3, material=tissue)
    Q = _localized_heat_source(shape=(5, 5, 5), q_center=5e6)
    pulse_dur = 1e-3
    # pulse_interval = 0.25, dt = 0.05 -> pulse start times k = 0, 5, 10, 15,
    # 20 * dt. Under naive accumulation, 15 * dt drifts to
    # 0.7500000000000001, which misplaces the 0.75 s pulse in the wrong
    # output step.
    events = generate_pulse_events(
        pulse_count=5, pulse_interval=0.25,
        pulse_train_count=1, pulse_train_interval=0.0,
        pulse_duration=pulse_dur, num_foci=1,
    )
    ds = run_thermal_simulation(
        heat_sources=Q, params=params, events=events,
        dt=0.05, t_end=2.0, T0=0.0,
        mode='impulse', pulse_duration=pulse_dur,
    )
    T = ds['temperature_rise'].data[:, 2, 2, 2]
    # The pulse start at t=0.75 (sample j=15) must first show up at j=16 --
    # the trace between j=14 and j=15 must be a smooth decay, NOT a big jump.
    per_pulse = 5e6 * pulse_dur / (tissue.density * tissue.specific_heat)
    assert (T[15] - T[14]) < 0 or abs(T[15] - T[14]) < 0.1 * per_pulse
    # The pulse should show up at j=16 (jump of ~ per_pulse from j=15 to j=16).
    assert (T[16] - T[15]) > 0.5 * per_pulse


def test_impulse_and_direct_agree_for_impulsive_pulses():
    """For short pulses, time-marching impulse mode and event-driven direct
    mode should agree closely: both use the sample-before-events convention
    and both integrate the same physics.
    """
    tissue = _absorbing_tissue()
    params = _uniform_params(shape=(7, 7, 7), spacing_m=1e-3, material=tissue)
    Q = _localized_heat_source(shape=(7, 7, 7), q_center=5e6)
    pulse_dur = 1e-3

    events = generate_pulse_events(
        pulse_count=5, pulse_interval=0.25,
        pulse_train_count=1, pulse_train_interval=0.0,
        pulse_duration=pulse_dur, num_foci=1,
    )
    kw = {
        "heat_sources": Q, "params": params, "events": events,
        "dt": 0.05, "t_end": 2.0, "T0": 0.0, "pulse_duration": pulse_dur,
    }
    ds_imp = run_thermal_simulation(mode='impulse', **kw)
    ds_dir = run_thermal_simulation(mode='direct', **kw)
    per_pulse = 5e6 * pulse_dur / (tissue.density * tissue.specific_heat)
    max_diff = float(np.max(np.abs(
        ds_imp['temperature_rise'].data - ds_dir['temperature_rise'].data
    )))
    # Difference should be tiny -- driven only by the fact that direct
    # integrates Q * dt_pulse across the pulse-duration while impulse deposits
    # it all at t = t_start.
    assert max_diff < 0.05 * per_pulse
