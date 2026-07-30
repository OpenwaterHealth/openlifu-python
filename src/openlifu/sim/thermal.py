from __future__ import annotations

import logging
import math
import time
from typing import List, Literal, Sequence, Tuple

import numpy as np
import xarray as xa

from openlifu.util.units import getunitconversion

logger = logging.getLogger(__name__)


def db2neper(alpha_db: np.ndarray | float, y: float = 1.0) -> np.ndarray | float:
    """Convert attenuation from dB / (MHz^y cm) to Nepers / ((rad/s)^y m).

    Matches the convention used by k-Wave's ``db2neper``. Multiplying the returned
    value by ``(2*pi*f)**y`` (with ``f`` in Hz) gives an attenuation coefficient
    in Np/m suitable for computing volumetric heat deposition from pressure.
    """
    return 100.0 * alpha_db * (1e-6 / (2.0 * math.pi)) ** y / (20.0 * np.log10(np.exp(1)))


def compute_heat_source(
    params: xa.Dataset,
    pressure: np.ndarray,
    freq: float,
    alpha_power: float = 0.9,
) -> np.ndarray:
    """Compute the volumetric rate of heat deposition from an acoustic pressure field.

    Uses the standard absorbed-power relation ``Q = alpha * p^2 / (rho * c)``
    where ``alpha`` is the attenuation coefficient in Np/m, ``p`` is the peak
    pressure amplitude in Pa, ``rho`` is density in kg/m^3, and ``c`` is speed
    of sound in m/s. The result has units of W/m^3.

    Args:
        params: Tissue parameter maps as an xarray Dataset. Must include
            ``density`` (kg/m^3), ``sound_speed`` (m/s), and ``attenuation``
            (dB/cm/MHz) variables.
        pressure: Peak pressure amplitude (Pa). Shape must be broadcastable to
            the parameter maps (typically ``(nx, ny, nz)`` or
            ``(num_foci, nx, ny, nz)``).
        freq: Frequency in Hz used when converting attenuation to Np/m.
        alpha_power: Power law exponent for attenuation. Default 0.9.

    Returns:
        Q with the same shape as ``pressure`` and units of W/m^3.
    """
    alpha_db = np.asarray(params['attenuation'].data)
    density = np.asarray(params['density'].data)
    sound_speed = np.asarray(params['sound_speed'].data)
    alpha_np = db2neper(alpha_db, y=alpha_power) * (2.0 * np.pi * freq) ** alpha_power
    return alpha_np * np.asarray(pressure) ** 2 / (density * sound_speed)


def generate_pulse_events(
    pulse_count: int,
    pulse_interval: float,
    pulse_train_count: int,
    pulse_train_interval: float,
    pulse_duration: float,
    num_foci: int,
    order: Sequence[int] | None = None,
) -> List[Tuple[float, int]]:
    """Generate a list of ``(time, focus_index)`` events for a pulse sequence.

    Positive ``focus_index`` (0-indexed) marks the start of a sonication at that
    focus. ``focus_index == -1`` marks the end of a sonication (Q returns to 0).

    Args:
        pulse_count: Number of pulses per pulse train.
        pulse_interval: Time between the start of successive pulses in a train (s).
        pulse_train_count: Number of pulse trains in the sequence.
        pulse_train_interval: Time between the start of successive pulse trains (s).
            If 0, trains are treated as back-to-back with no gap.
        pulse_duration: Duration of a single pulse (s).
        num_foci: Number of foci to cycle through.
        order: Optional list of 1-indexed focus indices specifying the order in
            which foci are sonicated. If None, the foci are cycled through in
            order 1, 2, ..., num_foci, matching the MATLAB LOFUgen1 convention.

    Returns:
        A list of ``(time_s, focus_index)`` tuples, sorted by time.
    """
    if num_foci <= 0:
        return []
    if order is None:
        order_idx = list(range(num_foci))
    else:
        order_idx = [int(i) - 1 for i in order]
        if any(i < 0 or i >= num_foci for i in order_idx):
            raise ValueError(f"order values must be 1-indexed integers in [1, {num_foci}].")
    if len(order_idx) == 0:
        return []

    if pulse_train_interval == 0:
        pt_interval = pulse_count * pulse_interval
    else:
        pt_interval = pulse_train_interval

    events: List[Tuple[float, int]] = []
    for pt in range(pulse_train_count):
        pt_start = pt * pt_interval
        for p in range(pulse_count):
            p_start = pt_start + p * pulse_interval
            p_end = p_start + pulse_duration
            p_index = order_idx[p % len(order_idx)]
            events.append((float(p_start), int(p_index)))
            events.append((float(p_end), -1))
    events.sort(key=lambda e: e[0])
    return events


def _get_dx(coords: xa.Coordinates) -> Tuple[float, float, float]:
    """Return grid spacing along (x, y, z) in meters."""
    dxs = []
    for dim in ('x', 'y', 'z'):
        unit = coords[dim].attrs.get('units', 'm')
        scl = getunitconversion(unit, 'm')
        data = coords[dim].data
        if len(data) < 2:
            raise ValueError(f"Coordinate '{dim}' must have at least 2 samples.")
        dxs.append(float(np.abs(np.diff(data)[0])) * scl)
    return tuple(dxs)  # type: ignore[return-value]


def _diffuse_step(
    T: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
    dz: float,
    k_field: np.ndarray,
    inv_rho_cp: np.ndarray,
    Q: np.ndarray | float,
) -> np.ndarray:
    """One forward-Euler step of the heterogeneous heat equation.

    Solves ``rho*C_p * dT/dt = div(k * grad T) + Q`` with zero-flux (Neumann)
    boundary conditions using cell-centered finite differences. The
    conductivity at cell faces is the arithmetic mean of neighboring values.
    """
    kx = 0.5 * (k_field[1:, :, :] + k_field[:-1, :, :])
    ky = 0.5 * (k_field[:, 1:, :] + k_field[:, :-1, :])
    kz = 0.5 * (k_field[:, :, 1:] + k_field[:, :, :-1])

    fx = kx * (T[1:, :, :] - T[:-1, :, :]) / dx
    fy = ky * (T[:, 1:, :] - T[:, :-1, :]) / dy
    fz = kz * (T[:, :, 1:] - T[:, :, :-1]) / dz

    div = np.zeros_like(T)
    div[:-1, :, :] += fx / dx
    div[1:, :, :] -= fx / dx
    div[:, :-1, :] += fy / dy
    div[:, 1:, :] -= fy / dy
    div[:, :, :-1] += fz / dz
    div[:, :, 1:] -= fz / dz

    return T + dt * inv_rho_cp * (div + Q)


def _cfl_dt(dx: float, dy: float, dz: float, k_field: np.ndarray, rho_cp: np.ndarray) -> float:
    """CFL-stable time step for explicit heat diffusion with heterogeneous coefficients."""
    with np.errstate(divide='ignore', invalid='ignore'):
        D = k_field / rho_cp
        D_max = float(np.nanmax(D))
    if D_max <= 0 or not np.isfinite(D_max):
        return math.inf
    return 0.5 / (D_max * (1.0 / dx ** 2 + 1.0 / dy ** 2 + 1.0 / dz ** 2))


def _advance(
    T: np.ndarray,
    duration: float,
    dstep: float,
    dx: float, dy: float, dz: float,
    k_field: np.ndarray,
    inv_rho_cp: np.ndarray,
    Q: np.ndarray | float,
) -> np.ndarray:
    """Advance ``T`` by ``duration`` using at most ``dstep``-sized forward-Euler steps."""
    if duration <= 0:
        return T
    n_full = math.floor(duration / dstep + 1e-12)
    frac = duration - n_full * dstep
    for _ in range(n_full):
        T = _diffuse_step(T, dstep, dx, dy, dz, k_field, inv_rho_cp, Q)
    if frac > 1e-15:
        T = _diffuse_step(T, frac, dx, dy, dz, k_field, inv_rho_cp, Q)
    return T


def _run_direct(
    heat_sources: np.ndarray,
    events: List[Tuple[float, int]],
    N: int,
    dt: float,
    dstep: float,
    dx: float, dy: float, dz: float,
    k_field: np.ndarray,
    inv_rho_cp: np.ndarray,
    T_init: np.ndarray | None,
) -> np.ndarray:
    """Run the thermal simulation via direct time-stepping through events.

    Mirrors the LOFUgen1 MATLAB ``run_thermal_sim`` control flow: at each
    output step the simulation is advanced by ``dt`` in ``dstep``-sized
    increments, with fractional sub-steps taken to align exactly with event
    times where Q changes.
    """
    shape = heat_sources.shape[1:]
    if T_init is None:
        T = np.zeros(shape, dtype=np.float32)
    else:
        T = T_init.astype(np.float32).copy()

    T_out = np.zeros((N, *shape), dtype=np.float32)
    # Sentinel event to terminate the while loop below without special-casing
    sorted_events = [*sorted(events, key=lambda e: e[0]), (math.inf, -1)]
    ii = 0
    Q: np.ndarray | float = 0.0

    for i in range(N):
        T_out[i] = T
        # Compute step boundaries directly from i to avoid the floating-point
        # drift that would otherwise accumulate in ``t += dt`` and cause
        # events at exact grid points (e.g. t = 0.75 with dt = 0.05) to fall
        # into the wrong output step.
        t = i * dt
        t_target = (i + 1) * dt
        if sorted_events[ii][0] > t_target:
            T = _advance(T, dt, dstep, dx, dy, dz, k_field, inv_rho_cp, Q)
        else:
            t1 = t
            while sorted_events[ii][0] <= t_target:
                e_time = sorted_events[ii][0]
                if e_time > t1:
                    T = _advance(T, e_time - t1, dstep, dx, dy, dz, k_field, inv_rho_cp, Q)
                    t1 = e_time
                e_idx = sorted_events[ii][1]
                Q = 0.0 if e_idx < 0 else heat_sources[e_idx]
                ii += 1
            if t_target > t1:
                T = _advance(T, t_target - t1, dstep, dx, dy, dz, k_field, inv_rho_cp, Q)
    return T_out


def _run_superpose(
    heat_sources: np.ndarray,
    events: List[Tuple[float, int]],
    N: int,
    dt: float,
    dstep: float,
    dx: float, dy: float, dz: float,
    k_field: np.ndarray,
    inv_rho_cp: np.ndarray,
    T_init: np.ndarray | None,
    pulse_duration: float,
) -> np.ndarray:
    """Run the thermal simulation by superposition of per-focus single-pulse responses.

    For each focus ``i`` we integrate the diffusion equation with ``Q_i`` on
    for ``pulse_duration``, then off, sampling ``T_i(x, k*dt)`` for ``k=0..N-1``.
    Because the diffusion equation is linear, the total temperature rise for
    the pulse sequence is the sum of appropriately time-shifted single-pulse
    responses.

    This is typically much faster than the direct mode when the sequence
    contains many pulses (each pulse contributes an O(N * grid) addition,
    whereas a direct time-stepped simulation must resolve every pulse edge).
    Pulse event times are snapped to the nearest ``dt`` sample; a warning is
    logged if the snapping error is significant.
    """
    nfoc = heat_sources.shape[0]
    shape = heat_sources.shape[1:]

    if T_init is None:
        T_out = np.zeros((N, *shape), dtype=np.float32)
    else:
        T_out = np.broadcast_to(T_init.astype(np.float32), (N, *shape)).copy()

    # Bucket "on" events by focus and snap their times to the output grid.
    max_snap_err = 0.0
    per_focus_events: List[List[int]] = [[] for _ in range(nfoc)]
    for event_time, focus_idx in events:
        if focus_idx < 0:
            continue
        j_e_float = event_time / dt
        j_e = round(j_e_float)
        snap_err = abs(j_e - j_e_float) * dt
        max_snap_err = max(max_snap_err, snap_err)
        if j_e >= N:
            continue
        per_focus_events[focus_idx].append(j_e)

    if max_snap_err > 0.05 * dt:
        logger.warning(
            "Event times are not aligned to the output time step (dt=%g); "
            "max snap error is %g s. Consider decreasing dt for better precision.",
            dt, max_snap_err,
        )

    for i in range(nfoc):
        if not per_focus_events[i]:
            continue
        logger.info("Computing single-pulse response for focus %d/%d ...", i + 1, nfoc)
        j_events = np.asarray(sorted(per_focus_events[i]), dtype=np.int64)
        T_i = np.zeros(shape, dtype=np.float32)
        Q_active = np.asarray(heat_sources[i], dtype=np.float32)
        for j in range(N):
            # Distribute T_i (which equals T_i(x, j*dt)) into T_out for every
            # on-event at this focus: contribution to sample j_e + j.
            valid = j_events + j
            valid = valid[valid < N]
            for j_out in valid:
                T_out[j_out] += T_i
            # Advance T_i by one output step from tau=j*dt to (j+1)*dt.
            tau = j * dt
            tau_target = tau + dt
            if tau_target <= pulse_duration + 1e-15:
                T_i = _advance(T_i, dt, dstep, dx, dy, dz, k_field, inv_rho_cp, Q_active)
            elif tau >= pulse_duration - 1e-15:
                T_i = _advance(T_i, dt, dstep, dx, dy, dz, k_field, inv_rho_cp, 0.0)
            else:
                on_dur = pulse_duration - tau
                T_i = _advance(T_i, on_dur, dstep, dx, dy, dz, k_field, inv_rho_cp, Q_active)
                T_i = _advance(T_i, dt - on_dur, dstep, dx, dy, dz, k_field, inv_rho_cp, 0.0)
    return T_out


def _run_impulse(
    heat_sources: np.ndarray,
    events: List[Tuple[float, int]],
    N: int,
    dt: float,
    dstep: float,
    dx: float, dy: float, dz: float,
    k_field: np.ndarray,
    inv_rho_cp: np.ndarray,
    T_init: np.ndarray | None,
    pulse_duration: float,
) -> np.ndarray:
    """Time-marching impulse mode: apply instantaneous deposits at each pulse
    event and diffuse between them.

    Marches a single temperature field forward in time (all foci contribute
    to the same field). At each output sample time ``t_j = j*dt`` the current
    ``T`` is stored *before* any events at that time (matching the
    sample-before-events convention used by the other two modes). Between
    samples, ``T`` is diffused with ``Q = 0``, and a fractional sub-step is
    inserted at every pulse-start event so that the instantaneous deposit

    .. math::

        E_i(x) = Q_i(x) \\cdot \\mathrm{pulse\\_duration} / (\\rho C_p)

    is applied at exactly the right time. Off-grid events (event times not
    aligned to the output-``dt`` grid) are captured *exactly* -- no snapping,
    no linear interpolation, no lost energy -- because the sub-step size is
    driven by the event stream itself. Multiple events within a single output
    step, including events that would fall at fractional positions, all
    contribute independently.

    Pulse-end events (``focus_idx == -1``) are ignored: in the impulse limit
    the pulse energy has already been deposited instantaneously at the pulse-
    start time and no separate turn-off event is needed.

    The internal sub-step size is bounded by ``dstep = dt / oversample``, so
    when events are sparse the diffusion is still integrated with a step
    small enough to be numerically stable. Total work per focus scales as
    ``O((N + n_events) * volume)`` which is dramatically cheaper than the
    per-focus superposition scheme when the sequence contains many pulses.
    """
    shape = heat_sources.shape[1:]
    if T_init is None:
        T = np.zeros(shape, dtype=np.float32)
    else:
        T = T_init.astype(np.float32).copy()

    T_out = np.zeros((N, *shape), dtype=np.float32)

    # Per-focus energy-deposit fields, expressed as an instantaneous
    # temperature rise (K). deposit[i](x) = Q_i(x) * pulse_duration / (rho * C_p).
    deposits = (
        np.asarray(heat_sources, dtype=np.float32)
        * float(pulse_duration)
        * inv_rho_cp.astype(np.float32)[None, ...]
    )

    # Sort pulse-start events by time; drop events that would fall outside
    # the simulation window. The sentinel event at t=inf lets the inner
    # while-loop terminate without a special case.
    on_events: List[Tuple[float, int]] = sorted(
        (
            (float(t), int(i))
            for t, i in events
            if i >= 0 and t >= 0 and t < N * dt
        ),
        key=lambda e: e[0],
    )
    on_events.append((math.inf, -1))
    ii = 0
    for j in range(N):
        # Store T at t = j*dt BEFORE any events at this sample time.
        T_out[j] = T
        # Compute step boundaries directly from j (rather than accumulating
        # t += dt) to avoid floating-point drift. With 15 iterations of
        # t += 0.05, t drifts to 0.7500000000000001, which then wrongly
        # captures events at exactly t=0.75 in the j=14 iteration instead
        # of the j=15 iteration.
        t = j * dt
        t_target = (j + 1) * dt
        t1 = t
        # Process every event in [t, t_target). Events falling exactly at
        # t_target are handled in the next iteration, preserving the
        # sample-before-events convention.
        while on_events[ii][0] < t_target:
            e_time, focus_idx = on_events[ii]
            if e_time > t1:
                # Diffuse (with Q=0) up to the event time, taking sub-steps
                # bounded by dstep.
                T = _advance(T, e_time - t1, dstep, dx, dy, dz, k_field, inv_rho_cp, 0.0)
                t1 = e_time
            # Apply the instantaneous energy deposit for this focus.
            T = T + deposits[focus_idx]
            ii += 1
        # Diffuse the remainder of the output step to reach t = (j+1)*dt.
        if t_target > t1:
            T = _advance(T, t_target - t1, dstep, dx, dy, dz, k_field, inv_rho_cp, 0.0)
    return T_out


def run_thermal_simulation(
    heat_sources: xa.DataArray,
    params: xa.Dataset,
    events: Sequence[Tuple[float, int]],
    dt: float = 0.1,
    t_end: float | None = None,
    oversample: int = 1,
    T0: float = 0.0,
    T_init: np.ndarray | None = None,
    mode: Literal['direct', 'superpose', 'impulse'] = 'impulse',
    pulse_duration: float | None = None,
) -> xa.Dataset:
    """Run a k-Wave-style thermal diffusion simulation.

    Solves the heterogeneous heat equation
    ``rho * C_p * dT/dt = div(k * grad T) + Q(t)`` with zero-flux boundary
    conditions using an explicit finite-difference solver. Three evaluation
    modes are provided:

    * ``mode='impulse'`` (default): treats each pulse as an instantaneous
      energy deposit ``E_i = Q_i * pulse_duration`` (J/m^3) applied at the
      pulse-start time. The solver marches a single temperature field
      forward, inserting a fractional sub-step at every pulse event so that
      the deposit is applied at *exactly* the correct time. Events do NOT
      need to be aligned to the output-``dt`` grid: sub-``dt`` timing is
      captured exactly, and any number of events per output step -- including
      events that would otherwise fall between samples -- deposit their
      energy correctly.
    * ``mode='superpose'``: precomputes the per-focus temperature response
      to a single pulse (with the source held on for ``pulse_duration``,
      then off) and superposes shifted copies over the sequence. Use this
      when ``pulse_duration`` is comparable to or longer than ``dt`` and the
      pulse-duration cannot be treated as instantaneous.
    * ``mode='direct'``: mirrors the LOFUgen1 MATLAB ``run_thermal_sim`` and
      walks the sequence of pulse-on/pulse-off events, taking fractional
      sub-steps to align with event times.

    All three modes use the sample-before-events convention: ``T[j*dt]``
    reports the temperature-rise at time ``j*dt`` *before* any pulse events
    that occur at that instant have taken effect.

    Args:
        heat_sources: The per-focus volumetric heat deposition rate (W/m^3).
            Expected to be an xarray DataArray with a leading
            ``focal_point_index`` dimension and spatial dims ``x, y, z``. The
            spatial coordinates (with ``units`` attributes) are used to derive
            grid spacing.
        params: Tissue parameter maps. Must contain ``density`` (kg/m^3),
            ``specific_heat`` (J/kg/K), and ``thermal_conductivity`` (W/m/K)
            variables on the same grid as ``heat_sources``.
        events: A list of ``(time_s, focus_index)`` events. Use
            :func:`generate_pulse_events` to build this from an
            :class:`openlifu.bf.Sequence`. Off-events (``focus_index == -1``)
            are ignored in ``'impulse'`` mode.
        dt: Output time step (s) at which the temperature field is stored.
            Default 0.1.
        t_end: Simulation duration (s). If None, uses the time of the last
            event plus one ``dt``.
        oversample: Sets the maximum internal computation step:
            ``dstep = dt / oversample``. Increase to enforce numerical
            stability at fine spatial resolution (see the CFL warning) or to
            take smaller diffusion steps between output samples. In
            ``'impulse'`` mode the effective internal step is
            ``min(dstep, event-to-event gap)``.
        T0: Background temperature offset (degC). Only affects the reported
            ``Tmax``; the stored ``temperature_rise`` is always
            ``T - ambient``.
        T_init: Optional initial temperature-rise field ``(nx, ny, nz)``.
        mode: ``'impulse'`` (default), ``'superpose'``, or ``'direct'``.
        pulse_duration: Duration of a single pulse (s). Required for
            ``'impulse'`` and ``'superpose'`` modes.

    Returns:
        An xarray Dataset with variables:

        * ``temperature_rise`` (``t, x, y, z``): temperature rise (degC).
        * ``Tmax`` (scalar): ``T0 + max(temperature_rise)`` (degC).
        * ``sim_time`` (scalar): wall-clock simulation time (s).

        The ``T0`` value used is stored as a Dataset attribute.
    """
    if not isinstance(heat_sources, xa.DataArray):
        raise TypeError("heat_sources must be an xarray DataArray with a leading 'focal_point_index' dim.")
    if 'focal_point_index' not in heat_sources.dims:
        raise ValueError("heat_sources must have a 'focal_point_index' dimension.")
    heat_sources_t = heat_sources.transpose('focal_point_index', 'x', 'y', 'z')
    Q_arr = np.asarray(heat_sources_t.data, dtype=np.float32)

    for name in ('density', 'specific_heat', 'thermal_conductivity'):
        if name not in params:
            raise ValueError(f"params is missing required variable '{name}'.")

    density = np.asarray(params['density'].data, dtype=np.float32)
    specific_heat = np.asarray(params['specific_heat'].data, dtype=np.float32)
    thermal_conductivity = np.asarray(params['thermal_conductivity'].data, dtype=np.float32)
    rho_cp = density * specific_heat
    if np.any(rho_cp <= 0):
        raise ValueError("params.density * params.specific_heat must be strictly positive everywhere.")
    inv_rho_cp = 1.0 / rho_cp

    # Diagnostic: catch the common "no temperature rise" failure mode where
    # the attenuation map is zero (all default openlifu Materials except
    # STANDOFF have attenuation=0). Q = alpha * p^2 / (rho * c) is zero, so
    # nothing will heat regardless of pulse handling.
    q_peak = float(np.max(np.abs(Q_arr))) if Q_arr.size > 0 else 0.0
    if q_peak == 0:
        att = params.get('attenuation')
        att_is_zero = att is not None and float(np.max(np.abs(np.asarray(att.data)))) == 0
        logger.warning(
            "All heat sources are zero%s; no temperature rise will be simulated. "
            "Check that params['attenuation'] is nonzero in the region of interest.",
            " (params['attenuation'] is zero everywhere)" if att_is_zero else "",
        )
    elif pulse_duration is not None:
        # Report the impulse-limit peak dT per pulse so the user can sanity
        # check whether their setup is expected to produce a visible rise.
        dT_peak_per_pulse = q_peak * float(pulse_duration) * float(np.max(inv_rho_cp))
        logger.info(
            "Peak Q = %.3g W/m^3; impulse-limit peak dT per pulse = %.3g K "
            "(pulse_duration = %g s).",
            q_peak, dT_peak_per_pulse, pulse_duration,
        )

    dx, dy, dz = _get_dx(heat_sources_t.coords)
    if oversample < 1:
        raise ValueError("oversample must be >= 1.")
    dstep = dt / oversample
    cfl = _cfl_dt(dx, dy, dz, thermal_conductivity, rho_cp)
    if dstep > cfl:
        logger.warning(
            "Requested internal step %g s exceeds explicit-diffusion CFL bound "
            "%g s (dx=%g, dy=%g, dz=%g m). The simulation may be unstable; "
            "consider increasing 'oversample' to at least %d.",
            dstep, cfl, dx, dy, dz, max(1, math.ceil(dt / cfl)),
        )

    events_list = list(events)
    if t_end is None:
        if events_list:
            t_end = max(e[0] for e in events_list) + dt
        else:
            t_end = dt
    if t_end <= 0:
        raise ValueError("t_end must be positive.")
    N = round(t_end / dt) + 1

    if mode in ('superpose', 'impulse') and pulse_duration is None:
        raise ValueError(f"pulse_duration must be provided when mode='{mode}'.")

    t0_wall = time.perf_counter()
    if mode == 'direct':
        T_data = _run_direct(
            Q_arr, events_list, N, dt, dstep,
            dx, dy, dz, thermal_conductivity, inv_rho_cp, T_init,
        )
    elif mode == 'superpose':
        T_data = _run_superpose(
            Q_arr, events_list, N, dt, dstep,
            dx, dy, dz, thermal_conductivity, inv_rho_cp, T_init,
            float(pulse_duration),
        )
    elif mode == 'impulse':
        T_data = _run_impulse(
            Q_arr, events_list, N, dt, dstep,
            dx, dy, dz, thermal_conductivity, inv_rho_cp, T_init,
            float(pulse_duration),
        )
    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Expected 'direct', 'superpose', or 'impulse'."
        )
    sim_time = time.perf_counter() - t0_wall

    coords_xyz = heat_sources_t.coords
    t_coord = xa.DataArray(
        np.arange(N) * dt,
        dims=['t'],
        attrs={'units': 's', 'long_name': 'Time'},
    )
    T_da = xa.DataArray(
        T_data,
        dims=['t', 'x', 'y', 'z'],
        coords={
            't': t_coord,
            'x': coords_xyz['x'],
            'y': coords_xyz['y'],
            'z': coords_xyz['z'],
        },
        attrs={'units': 'degC', 'long_name': 'Temperature rise'},
    )
    Tmax = float(T_data.max()) + float(T0)
    ds = xa.Dataset(
        {
            'temperature_rise': T_da,
            'Tmax': xa.DataArray(Tmax, attrs={'units': 'degC', 'long_name': 'Peak temperature'}),
            'sim_time': xa.DataArray(sim_time, attrs={'units': 's', 'long_name': 'Wall-clock simulation time'}),
        },
        attrs={'T0': float(T0), 'mode': mode},
    )
    logger.info("Thermal simulation complete in %.1f s (mode=%s, N=%d)", sim_time, mode, N)
    return ds
